"""
Synthesis Agent Node — Combines RAG context and PubMed Research context.

This LangGraph node merges the outputs from the RAG agent (conditional)
and the PubMed Search agent (always-on parallel) to produce a comprehensive
final answer using Gemini LLM.
"""

import os
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


# ── LLM Config ─────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")

# ── Synthesis Prompt ───────────────────────────────────────────────────

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""You are a senior Speech-Language Pathology specialist providing a comprehensive clinical assessment.

Your task is to **answer the user's original query** using all the gathered context below.

You have access to TWO sources of information that were gathered to help you answer:
1. **Clinical Reference Materials** (from RAG retrieval over curated clinical documents)
2. **PubMed Research** (peer-reviewed biomedical literature from PubMed/MEDLINE)

## Previous Conversation Context
{session_history}

## User's Original Query
{query}

## Background Context (for your reference only — do NOT answer these questions directly)
- **Category**: {category}
- **Identified Ailment**: {ailment}
- **Ailment Description**: {ailment_description}
- The following treatment questions were used internally to gather context. They are NOT part of the user's query:
{treatment_questions_formatted}

## Source 1: Clinical Reference Materials (RAG)
{rag_context}

## Source 2: PubMed Research (Peer-Reviewed Literature)
{web_search_summary}

## Instructions
Using ALL the context above, provide a direct, comprehensive answer to the **user's original query**: "{query}"

Your response should:
1. **Directly address what the user asked** — focus entirely on their query, not on the internal treatment questions.
2. **Clinical Overview**: Explain the identified condition in clear, accessible terms relevant to the user's concern.
3. **Evidence-Based Guidance**: Provide treatment approaches, assessment recommendations, and actionable strategies drawn from both clinical references and PubMed research.
4. **Practical Next Steps**: What should the user do next? Include referrals, home strategies, or professional guidance as appropriate.

If there is relevant previous conversation context, use it to provide continuity and avoid repeating information already discussed. Reference previous queries when appropriate to show awareness of the conversation flow.

Be empathetic, clinically accurate, and practical. Cite relevant information from both sources where helpful. When citing PubMed articles, include the PMID for reference.
Do NOT list or answer the 3 treatment questions — they were only used to gather information for you.
Format your response in a clear, structured manner.
""")


def _invoke_with_retry(chain, params: dict, agent_name: str = "Agent", max_retries: int = 3, base_delay: float = 15.0):
    """Invoke an LLM chain with retry logic for 429 rate-limit errors.

    Args:
        chain: The LangChain chain to invoke.
        params: Parameters to pass to the chain.
        agent_name: Name of the calling agent for log messages.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries (multiplied by attempt number).

    Returns:
        The chain's response.

    Raises:
        The last exception if all retries are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(params)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries:
                    wait_time = base_delay * attempt
                    print(f"   ⏳ [{agent_name}] Rate limited (429). Retrying in {wait_time:.0f}s... (attempt {attempt}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ [{agent_name}] Rate limit exceeded after {max_retries} attempts. Giving up.")
                    raise
            else:
                raise


def _format_treatment_questions(questions: list[str]) -> str:
    """Format treatment questions as a numbered list."""
    return "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))


def _format_session_history(history: list[dict]) -> str:
    """Format the session history (last N Q&A pairs) into a prompt-ready string.

    Args:
        history: List of dicts with 'query' and 'answer' keys.

    Returns:
        Formatted string of previous conversation, or a note if empty.
    """
    if not history:
        return "(No previous conversation in this session)"

    parts = []
    for i, entry in enumerate(history, 1):
        q = entry.get("query", "")
        a = entry.get("answer", "")
        # Truncate long answers to keep prompt manageable
        if len(a) > 500:
            a = a[:500] + "..."
        parts.append(f"Q{i}: {q}\nA{i}: {a}")
    return "\n\n".join(parts)


def synthesis_node(state: dict) -> dict:
    """LangGraph node: Synthesis Agent.

    Merges RAG context and PubMed Research context, then uses Gemini to
    produce a comprehensive final answer.

    Args:
        state: The pipeline state containing rag_context, rag_response,
               web_search_context, web_search_summary, and classification data.

    Returns:
        Updated state with the synthesized final_output.
    """
    query = state.get("query", "")
    category = state.get("category", "Unknown")
    ailment = state.get("ailment", "Unknown")
    ailment_description = state.get("ailment_description", "")
    treatment_questions = state.get("treatment_questions", [])
    session_history = state.get("session_history", [])

    print(f"\n{'='*60}")
    print(f"🧪 [Synthesis Agent] Combining RAG + PubMed Research context...")
    print(f"   Query: \"{query}\"")
    print(f"   Ailment: {ailment} | Category: {category}")
    print(f"   Session history: {len(session_history)} previous entries")

    # Get RAG context
    rag_context = state.get("rag_context", "(No RAG context available)")
    rag_response = state.get("rag_response", "")
    print(f"   RAG context: {len(rag_context)} chars")

    # Get Web Search context
    web_search_context = state.get("web_search_context", {})
    web_search_summary = state.get(
        "web_search_summary", "(No web search context available)"
    )
    print(f"   PubMed context: {len(web_search_summary)} chars ({len(web_search_context)} questions)")

    # Generate synthesized response with Gemini (with retry for rate limits)
    print(f"   🤖 Generating synthesized answer with Gemini...")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    chain = SYNTHESIS_PROMPT | llm

    response = _invoke_with_retry(chain, {
        "query": query,
        "category": category,
        "ailment": ailment,
        "ailment_description": ailment_description,
        "treatment_questions_formatted": _format_treatment_questions(
            treatment_questions
        ),
        "rag_context": rag_context,
        "web_search_summary": web_search_summary,
        "session_history": _format_session_history(session_history),
    }, agent_name="Synthesis Agent")

    raw_content = response.content if hasattr(response, "content") else str(response)

    # Gemini may return content as a list of dicts like [{'type': 'text', 'text': '...'}]
    # Extract the actual text string from it
    if isinstance(raw_content, list):
        text_parts = []
        for block in raw_content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        synthesis_text = "\n".join(text_parts)
    else:
        synthesis_text = str(raw_content)

    print(f"   ✅ [Synthesis Agent] Final answer generated ({len(synthesis_text)} chars)")
    print(f"{'='*60}\n")

    # Build the final enriched output dictionary
    final_output = {
        "category": category,
        "ailment": ailment,
        "ailment_description": ailment_description,
        "treatment_questions": treatment_questions,
        "rag_context": rag_context,
        "rag_response": rag_response,
        "web_search_context": web_search_context,
        "web_search_summary": web_search_summary,
        "synthesis_response": synthesis_text,
    }

    return {
        "synthesis_response": synthesis_text,
        "final_output": final_output,
        "messages": [AIMessage(content=synthesis_text)],
    }
