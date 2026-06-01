"""
RAG Agent Nodes — Speech and Feeding RAG nodes for the LangGraph pipeline.

Each node retrieves relevant context from its category-specific FAISS vector
store and uses Gemini to generate evidence-based treatment recommendations.
"""

import json
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import load_speech_retriever, load_feeding_retriever


# ── LLM Config ─────────────────────────────────────────────────────────
LLM_MODEL = "gemini-2.5-flash"

# ── RAG Prompts ────────────────────────────────────────────────────────

SPEECH_RAG_PROMPT = ChatPromptTemplate.from_template("""You are a Speech-Language Pathology specialist with access to clinical reference materials.

Based on the retrieved clinical context below, provide evidence-based treatment recommendations 
for the identified condition.

## Patient Assessment Summary
- **Category**: {category}
- **Identified Ailment**: {ailment}
- **Treatment Questions Asked**:
{treatment_questions_formatted}

## Retrieved Clinical Context
{retrieved_context}

## Instructions
Using the clinical reference materials above, provide:
1. **Evidence-Based Treatment Approach**: What does the clinical literature recommend for this condition?
2. **Key Assessment Considerations**: What specific assessments should be conducted based on the references?
3. **Treatment Recommendations**: Specific, actionable treatment strategies supported by the retrieved context.

Be specific, cite relevant information from the retrieved context, and provide clinically actionable guidance.
Format your response in a clear, structured manner.
""")

FEEDING_RAG_PROMPT = ChatPromptTemplate.from_template("""You are a Dysphagia and Feeding Disorders specialist with access to clinical reference materials.

Based on the retrieved clinical context below, provide evidence-based treatment recommendations 
for the identified feeding/swallowing condition.

## Patient Assessment Summary
- **Category**: {category}
- **Identified Ailment**: {ailment}
- **Treatment Questions Asked**:
{treatment_questions_formatted}

## Retrieved Clinical Context
{retrieved_context}

## Instructions
Using the clinical reference materials above, provide:
1. **Evidence-Based Treatment Approach**: What does the clinical literature recommend for this swallowing/feeding condition?
2. **Key Assessment Considerations**: What specific swallowing assessments (e.g., FEES, MBS) or feeding evaluations should be conducted?
3. **Treatment Recommendations**: Specific, actionable strategies (e.g., diet modifications, therapeutic exercises, compensatory strategies) supported by the retrieved context.

Be specific, cite relevant information from the retrieved context, and provide clinically actionable guidance.
Format your response in a clear, structured manner.
""")


def _format_treatment_questions(questions: list[str]) -> str:
    """Format treatment questions as a numbered list."""
    return "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))


def _format_retrieved_docs(docs: list) -> str:
    """Format retrieved documents into a context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"--- Reference {i} (Source: {source}, Page: {page}) ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted)


def speech_rag_node(state: dict) -> dict:
    """LangGraph node: Speech RAG Agent.

    Retrieves relevant context from the Speech FAISS vector store
    and generates evidence-based treatment recommendations using Gemini.

    Args:
        state: The pipeline state dictionary containing classification results.

    Returns:
        Updated state with RAG context and response.
    """
    ailment = state.get("ailment", "Unknown")
    category = state.get("category", "Speech")
    treatment_questions = state.get("treatment_questions", [])

    # Build retrieval query from the ailment and questions
    retrieval_query = (
        f"{ailment} treatment assessment. "
        + " ".join(treatment_questions)
    )

    # Retrieve from Speech vector store
    try:
        retriever = load_speech_retriever(k=5)
        retrieved_docs = retriever.invoke(retrieval_query)
        retrieved_context = _format_retrieved_docs(retrieved_docs)
    except FileNotFoundError as e:
        retrieved_context = f"(Vector store not available: {e})"
        retrieved_docs = []

    # Generate RAG response with Gemini
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    chain = SPEECH_RAG_PROMPT | llm

    response = chain.invoke({
        "category": category,
        "ailment": ailment,
        "treatment_questions_formatted": _format_treatment_questions(treatment_questions),
        "retrieved_context": retrieved_context,
    })

    rag_response_text = response.content if hasattr(response, "content") else str(response)

    # Build the enriched output dictionary
    final_output = {
        "category": category,
        "ailment": ailment,
        "treatment_questions": treatment_questions,
        "rag_response": rag_response_text,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "?"),
            }
            for doc in retrieved_docs
        ],
    }

    return {
        "rag_context": retrieved_context,
        "rag_response": rag_response_text,
        "final_output": final_output,
        "messages": [AIMessage(content=rag_response_text)],
    }


def feeding_rag_node(state: dict) -> dict:
    """LangGraph node: Feeding RAG Agent.

    Retrieves relevant context from the Feeding FAISS vector store
    and generates evidence-based treatment recommendations using Gemini.

    Args:
        state: The pipeline state dictionary containing classification results.

    Returns:
        Updated state with RAG context and response.
    """
    ailment = state.get("ailment", "Unknown")
    category = state.get("category", "Feeding")
    treatment_questions = state.get("treatment_questions", [])

    # Build retrieval query
    retrieval_query = (
        f"{ailment} treatment assessment swallowing feeding. "
        + " ".join(treatment_questions)
    )

    # Retrieve from Feeding vector store
    try:
        retriever = load_feeding_retriever(k=5)
        retrieved_docs = retriever.invoke(retrieval_query)
        retrieved_context = _format_retrieved_docs(retrieved_docs)
    except FileNotFoundError as e:
        retrieved_context = f"(Vector store not available: {e})"
        retrieved_docs = []

    # Generate RAG response with Gemini
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    chain = FEEDING_RAG_PROMPT | llm

    response = chain.invoke({
        "category": category,
        "ailment": ailment,
        "treatment_questions_formatted": _format_treatment_questions(treatment_questions),
        "retrieved_context": retrieved_context,
    })

    rag_response_text = response.content if hasattr(response, "content") else str(response)

    # Build the enriched output dictionary
    final_output = {
        "category": category,
        "ailment": ailment,
        "treatment_questions": treatment_questions,
        "rag_response": rag_response_text,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "?"),
            }
            for doc in retrieved_docs
        ],
    }

    return {
        "rag_context": retrieved_context,
        "rag_response": rag_response_text,
        "final_output": final_output,
        "messages": [AIMessage(content=rag_response_text)],
    }
