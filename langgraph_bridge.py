"""
LangGraph Pipeline — Full Speech/Feeding classification + RAG workflow.

This module implements the complete LangGraph pipeline:
  Node 1 (Classification) → Conditional Router → Speech RAG or Feeding RAG

Usage:
    from langgraph_bridge import create_pipeline
    pipeline = create_pipeline()
    result = pipeline.invoke({"query": "patient has trouble swallowing"})
    print(result["final_output"])
"""

from __future__ import annotations

import asyncio
import json
import uuid
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from google.genai import types

from agent.speech_feeding_agent import create_agent, create_runner
from agent.config import AGENT_NAME
from rag.rag_agent import speech_rag_node, feeding_rag_node


# ── Pipeline State Schema ──────────────────────────────────────────────

class PipelineState(TypedDict):
    """State schema for the full LangGraph pipeline."""
    query: str                                                  # User's original query
    messages: Annotated[list[BaseMessage], operator.add]         # Message history
    classification: dict | None                                  # Node 1 output dict
    category: str                                                # "Speech" or "Feeding"
    ailment: str                                                 # Identified condition
    ailment_description: str                                     # Condition description
    treatment_questions: list[str]                               # 3 questions from Node 1
    rag_context: str                                             # Retrieved documents
    rag_response: str                                            # RAG-enriched response
    final_output: dict                                           # Final dictionary output


# ── Node 1: Classification Agent ───────────────────────────────────────

def classification_node(state: PipelineState) -> dict:
    """Node 1: Run the ADK classification agent and capture structured output.

    Takes the user query, runs it through the Google ADK agent which
    classifies it into Speech/Feeding, identifies the ailment, and
    generates 3 treatment questions. Captures the structured dictionary.

    Args:
        state: The pipeline state.

    Returns:
        Updated state with classification results.
    """
    query = state["query"]

    # Run the ADK agent synchronously
    classification_dict = asyncio.run(_run_classification_agent(query))

    if classification_dict:
        return {
            "classification": classification_dict,
            "category": classification_dict.get("category", "Speech"),
            "ailment": classification_dict.get("ailment", "Unknown"),
            "ailment_description": classification_dict.get("ailment_description", ""),
            "treatment_questions": classification_dict.get("treatment_questions", []),
            "messages": [
                AIMessage(content=f"Classification: {json.dumps(classification_dict, indent=2)}")
            ],
        }
    else:
        # Fallback — keyword-based classification if ADK agent fails
        return _fallback_classification(query)


async def _run_classification_agent(query: str) -> dict | None:
    """Run the ADK classification agent and extract the structured dict.

    Args:
        query: The user's query text.

    Returns:
        The structured dictionary from generate_treatment_questions tool,
        or None if extraction fails.
    """
    runner, session_service = create_runner()

    user_id = "pipeline_user"
    session_id = f"pipeline_{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name=AGENT_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)],
    )

    structured_output = None

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_response:
                    fn_name = part.function_response.name
                    fn_result = part.function_response.response
                    if fn_name == "generate_treatment_questions" and fn_result:
                        result_data = fn_result.get("result", fn_result)
                        if isinstance(result_data, dict) and "category" in result_data:
                            structured_output = result_data

    return structured_output


def _fallback_classification(query: str) -> dict:
    """Keyword-based fallback if the ADK agent fails.

    Args:
        query: The user's query text.

    Returns:
        A minimal state update with keyword-based classification.
    """
    query_lower = query.lower()

    feeding_keywords = [
        "swallowing", "dysphagia", "feeding", "aspiration", "choking",
        "food refusal", "texture", "bolus", "pharyngeal", "esophageal",
        "eating", "drinking", "gag", "failure to thrive",
    ]
    speech_keywords = [
        "speech", "language", "articulation", "stuttering", "fluency",
        "voice", "aphasia", "apraxia", "dysarthria", "pronunciation",
        "communication", "phonological", "resonance", "mutism", "hoarse",
    ]

    feeding_score = sum(1 for kw in feeding_keywords if kw in query_lower)
    speech_score = sum(1 for kw in speech_keywords if kw in query_lower)
    category = "Feeding" if feeding_score > speech_score else "Speech"

    return {
        "classification": {"category": category, "source": "fallback"},
        "category": category,
        "ailment": "To be identified",
        "ailment_description": "",
        "treatment_questions": [
            "When did the symptoms first appear and how have they progressed?",
            "How does this condition impact daily activities and quality of life?",
            "What treatments or interventions have been tried previously?",
        ],
        "messages": [AIMessage(content=f"Fallback classification: {category}")],
    }


# ── Conditional Router ─────────────────────────────────────────────────

def route_by_category(state: PipelineState) -> Literal["speech_rag", "feeding_rag"]:
    """Route to the appropriate RAG agent based on the classification category.

    Args:
        state: The pipeline state containing the category field.

    Returns:
        The name of the next node: "speech_rag" or "feeding_rag".
    """
    category = state.get("category", "Speech")
    if category == "Feeding":
        return "feeding_rag"
    return "speech_rag"


# ── Pipeline Factory ───────────────────────────────────────────────────

def create_pipeline():
    """Create the full LangGraph pipeline with conditional routing.

    Pipeline flow:
        START → classification_node → route_by_category →
            speech_rag_node (if Speech) / feeding_rag_node (if Feeding) → END

    Returns:
        A compiled LangGraph StateGraph ready for invocation.

    Example:
        >>> pipeline = create_pipeline()
        >>> result = pipeline.invoke({
        ...     "query": "patient has trouble swallowing after stroke",
        ...     "messages": [],
        ... })
        >>> print(result["final_output"])
    """
    builder = StateGraph(PipelineState)

    # Add nodes
    builder.add_node("classification", classification_node)
    builder.add_node("speech_rag", speech_rag_node)
    builder.add_node("feeding_rag", feeding_rag_node)

    # Define edges
    builder.add_edge(START, "classification")

    # Conditional routing based on category
    builder.add_conditional_edges(
        "classification",
        route_by_category,
        {
            "speech_rag": "speech_rag",
            "feeding_rag": "feeding_rag",
        },
    )

    # Both RAG nodes lead to END
    builder.add_edge("speech_rag", END)
    builder.add_edge("feeding_rag", END)

    # Compile
    compiled = builder.compile()
    return compiled


# ── Quick Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🏥 Testing LangGraph Pipeline with Conditional Routing...\n")

    pipeline = create_pipeline()

    test_queries = [
        "My child stutters when nervous and has trouble with R sounds",
        "Patient is coughing when swallowing liquids after their stroke",
    ]

    for query in test_queries:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = pipeline.invoke({
            "query": query,
            "messages": [HumanMessage(content=query)],
        })

        print(f"\n📋 Final Output:")
        print(json.dumps(result.get("final_output", {}), indent=2))
        print(f"\n{'='*60}\n")
