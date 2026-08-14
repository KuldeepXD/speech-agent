"""
Streamlit UI for the Speech/Feeding Medical AI Agent.

Provides a chat-based interface that triggers the full LangGraph pipeline,
displays real-time execution status of each agent node, and maintains
per-user session history (last 5 Q&A pairs passed to synthesis agent only).

Run with: streamlit run streamlit_app.py
"""

import uuid
import json
import time
import sys
import io
import os
import contextlib
from datetime import datetime

# Ensure Python uses UTF-8 for stdout (must be set before pipeline prints)
os.environ["PYTHONIOENCODING"] = "utf-8"

import streamlit as st
from langchain_core.messages import HumanMessage

from langgraph_bridge import create_pipeline
from rag.rag_agent import speech_rag_node, feeding_rag_node
from rag.web_search_agent import web_search_node
from rag.synthesis_agent import synthesis_node


# ── Page Config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Speech & Feeding AI Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Import Google Font ────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* ── Main container ────────────────────────────────── */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ── Hero Header ───────────────────────────────────── */
    .hero-header {
        background: linear-gradient(135deg, #6C63FF 0%, #3F3D9E 50%, #1A1D29 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(108, 99, 255, 0.3);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }
    .hero-header h1 {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: rgba(255,255,255,0.75);
        font-size: 0.95rem;
        margin: 0;
        font-weight: 300;
    }

    /* ── Chat Messages ─────────────────────────────────── */
    .user-msg {
        background: linear-gradient(135deg, #6C63FF, #4B45B2);
        color: #fff;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.75rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.2);
    }
    .agent-msg {
        background: #1E2130;
        color: #E8E8F0;
        padding: 1.25rem 1.5rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.75rem 0;
        border: 1px solid rgba(108, 99, 255, 0.15);
        font-size: 0.93rem;
        line-height: 1.6;
    }

    /* ── Pipeline Stage Cards ──────────────────────────── */
    .pipeline-stage {
        background: #15171F;
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .pipeline-stage:hover {
        border-color: rgba(108, 99, 255, 0.5);
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
    }
    .stage-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        color: #C4C0FF;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .stage-content {
        color: #A0A3B8;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* ── Sidebar ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12141C 0%, #0E1017 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.15);
    }
    .sidebar-card {
        background: rgba(108, 99, 255, 0.08);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .sidebar-card h4 {
        color: #C4C0FF;
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .sidebar-card p, .sidebar-card span {
        color: #8B8FA8;
        font-size: 0.82rem;
    }

    /* ── History Items ──────────────────────────────────── */
    .history-item {
        background: rgba(30, 33, 48, 0.7);
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        font-size: 0.8rem;
    }
    .history-item .query-text {
        color: #D0D0E0;
        font-weight: 500;
    }
    .history-item .meta-text {
        color: #6B6E85;
        font-size: 0.72rem;
        margin-top: 0.2rem;
    }

    /* ── Status Badges ─────────────────────────────────── */
    .status-running {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(255, 193, 7, 0.15);
        color: #FFC107;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    .status-done {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    .status-error {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(244, 67, 54, 0.15);
        color: #F44336;
        border: 1px solid rgba(244, 67, 54, 0.3);
    }

    /* ── Metric Card ───────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 0.75rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        flex: 1;
        background: rgba(108, 99, 255, 0.06);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #C4C0FF;
    }
    .metric-card .label {
        font-size: 0.7rem;
        color: #6B6E85;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Input area styling ────────────────────────────── */
    .stChatInput > div {
        border-color: rgba(108, 99, 255, 0.3) !important;
    }
    .stChatInput > div:focus-within {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 1px #6C63FF !important;
    }

    /* ── Expander styling ──────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    /* ── Hide Streamlit branding ───────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ───────────────────────────────────────

def init_session():
    """Initialize session state for a new user session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []  # Full history of all queries
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Chat display messages
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False


def get_session_history(max_entries: int = 5) -> list[dict]:
    """Get the last N conversation entries for synthesis agent context.

    Args:
        max_entries: Maximum number of Q&A pairs to return.

    Returns:
        List of dicts with 'query' and 'answer' keys.
    """
    history = st.session_state.conversation_history
    recent = history[-max_entries:] if len(history) > max_entries else history
    return [
        {"query": entry["query"], "answer": entry.get("final_answer", "")}
        for entry in recent
    ]


# ── Pipeline Initialization ────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load and cache the LangGraph pipeline."""
    return create_pipeline()


def ensure_pipeline():
    """Ensure the pipeline is loaded."""
    if st.session_state.pipeline is None:
        st.session_state.pipeline = load_pipeline()
        st.session_state.pipeline_ready = True


# ── Pipeline Execution with UI ─────────────────────────────────────────

def run_pipeline_with_ui(query: str) -> dict:
    """Run the full pipeline and display each stage's output in the UI.

    Executes the LangGraph pipeline and shows real-time status for each
    agent node (Classification → RAG → PubMed → Synthesis).

    Args:
        query: The user's query text.

    Returns:
        The full pipeline result dictionary.
    """
    ensure_pipeline()
    pipeline = st.session_state.pipeline

    # Get session history (last 5 entries) for synthesis agent only
    session_history = get_session_history(max_entries=5)

    # Run the pipeline
    with st.status("** Running AI Pipeline...**", expanded=True) as status:

        st.markdown("##### Processing your query through the multi-agent pipeline")

        # ── Stage 1: Classification ────────────────────────────────
        st.markdown("---")
        st.markdown("####  Stage 1: Classification Agent")
        classification_placeholder = st.empty()
        classification_placeholder.markdown(
            '<span class="status-running">⏳ Classifying...</span>',
            unsafe_allow_html=True,
        )

        start_time = time.time()

        # Invoke the full pipeline
        # Redirect stdout/stderr to capture emoji print statements
        # that would crash on Windows' default charmap encoding
        try:
            log_buffer = io.StringIO()
            with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                result = pipeline.invoke({
                    "query": query,
                    "messages": [HumanMessage(content=query)],
                    "session_history": session_history,
                })
        except Exception as e:
            status.update(label="Pipeline Error", state="error", expanded=True)
            st.error(f"Pipeline execution failed: {e}")
            return {}

        elapsed = time.time() - start_time

        # ── Display Classification Results ─────────────────────────
        classification = result.get("classification", {})
        category = result.get("category", "Unknown")
        ailment = result.get("ailment", "Unknown")
        ailment_desc = result.get("ailment_description", "")
        treatment_questions = result.get("treatment_questions", [])

        with classification_placeholder.container():
            st.markdown(
                '<span class="status-done"> Complete</span>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Category", category)
            with col2:
                st.metric("Ailment", ailment)

            if ailment_desc:
                st.caption(f"{ailment_desc}")

            if treatment_questions:
                with st.expander("Generated Treatment Questions", expanded=False):
                    for i, q in enumerate(treatment_questions, 1):
                        st.markdown(f"**{i}.** {q}")

        # ── Stage 2: RAG Agent ─────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### Stage 2: {category} RAG Agent")
        rag_context = result.get("rag_context", "")
        rag_response = result.get("rag_response", "")

        st.markdown(
            '<span class="status-done"> Complete</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Retrieved context: {len(rag_context):,} characters")
        with st.expander("RAG Response", expanded=False):
            st.markdown(rag_response if rag_response else "_No RAG response generated_")

        # ── Stage 3: PubMed Search ─────────────────────────────────
        st.markdown("---")
        st.markdown("####  Stage 3: PubMed Search Agent")
        web_search_context = result.get("web_search_context", {})
        web_search_summary = result.get("web_search_summary", "")

        st.markdown(
            '<span class="status-done"> Complete</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Searched {len(web_search_context)} questions • "
            f"{len(web_search_summary):,} chars of context"
        )
        with st.expander("PubMed Search Results", expanded=False):
            if web_search_context:
                for key, entry in web_search_context.items():
                    q = entry.get("original_question", key)
                    search_q = entry.get("query", "N/A")
                    results_text = entry.get("results", "")
                    st.markdown(f"**Question:** {q}")
                    st.caption(f"Search query: `{search_q}`")
                    # Show truncated results
                    display_text = results_text[:800] + "..." if len(results_text) > 800 else results_text
                    st.text(display_text)
                    st.markdown("---")
            else:
                st.info("No PubMed search results were generated.")

        # ── Stage 4: Synthesis Agent ───────────────────────────────
        st.markdown("---")
        st.markdown("####  Stage 4: Synthesis Agent")
        st.markdown(
            '<span class="status-done"> Complete</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Combined RAG + PubMed context with "
            f"{len(session_history)} previous conversation(s)"
        )

        # ── Final Timing ──────────────────────────────────────────
        status.update(
            label=f"**Pipeline Complete** — {elapsed:.1f}s",
            state="complete",
            expanded=False,
        )

    return result


# ── Sidebar ────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with session info and history."""
    with st.sidebar:
        # Logo / branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
            <span style="font-size: 2.5rem;">🏥</span>
            <h3 style="margin: 0.3rem 0 0 0; color: #C4C0FF; font-weight: 700; letter-spacing: -0.5px;">
                SLP AI Agent
            </h3>
            <p style="color: #6B6E85; font-size: 0.75rem; margin: 0;">
                Speech-Language Pathology Assistant
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Session Info
        st.markdown("""
        <div class="sidebar-card">
            <h4>Session</h4>
            <p><code style="color: #C4C0FF; font-size: 0.78rem;">{session_id}</code></p>
        </div>
        """.format(session_id=st.session_state.session_id), unsafe_allow_html=True)

        # Metrics
        history_count = len(st.session_state.conversation_history)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="value">{history_count}</div>
                <div class="label">Queries</div>
            </div>
            <div class="metric-card">
                <div class="value">{min(history_count, 5)}</div>
                <div class="label">In Context</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # History
        st.markdown("""
        <div class="sidebar-card">
            <h4>Conversation History</h4>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.conversation_history:
            for i, entry in enumerate(reversed(st.session_state.conversation_history), 1):
                category = entry.get("category", "?")
                ailment = entry.get("ailment", "?")
                query = entry.get("query", "")
                ts = entry.get("timestamp", "")
                # Truncate long queries
                display_query = query[:60] + "..." if len(query) > 60 else query

                emoji = "Sppech" if category == "Speech" else "Feeding"

                st.markdown(f"""
                <div class="history-item">
                    <div class="query-text">{emoji} {display_query}</div>
                    <div class="meta-text">{category} • {ailment} • {ts}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color: #4A4D60; font-size: 0.82rem; text-align: center; '
                'padding: 1rem 0;">No queries yet. Start a conversation!</p>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Clear History button
        if st.button("  Clear History", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.messages = []
            st.rerun()

        # Info
        st.markdown("""
        <div style="padding: 0.8rem; margin-top: 0.5rem;">
            <p style="color: #4A4D60; font-size: 0.72rem; line-height: 1.5;">
                  <strong style="color: #6B6E85;">How it works:</strong><br>
                Your query goes through a multi-agent pipeline:<br>
                <span style="color: #6C63FF;">Classification</span> →
                <span style="color: #6C63FF;">RAG Retrieval</span> →
                <span style="color: #6C63FF;">PubMed Search</span> →
                <span style="color: #6C63FF;">Synthesis</span>
            </p>
            <p style="color: #4A4D60; font-size: 0.72rem; margin-top: 0.5rem;">
                  Last 5 conversations are passed to the Synthesis Agent for context continuity.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── Chat Display ───────────────────────────────────────────────────────

def render_chat_history():
    """Render the chat message history."""
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(
                f'<div class="user-msg">{content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="agent-msg">{content}</div>',
                unsafe_allow_html=True,
            )


# ── Main App ───────────────────────────────────────────────────────────

def main():
    """Main Streamlit application entry point."""
    init_session()
    render_sidebar()

    # ── Hero Header ────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1> Speech & Feeding AI Agent</h1>
        <p>
            AI-powered clinical assessment for Speech-Language Pathology.
            Describe a patient's symptoms and get evidence-based guidance
            combining clinical references and PubMed research.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Chat History ───────────────────────────────────────────────
    render_chat_history()

    # ── Chat Input ─────────────────────────────────────────────────
    if query := st.chat_input("Describe a patient's speech or feeding concern..."):
        # Add user message to display
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user message immediately
        st.markdown(
            f'<div class="user-msg">{query}</div>',
            unsafe_allow_html=True,
        )

        # Run the pipeline with UI
        result = run_pipeline_with_ui(query)

        if result:
            # Extract final answer
            final_output = result.get("final_output", {})
            synthesis_response = final_output.get(
                "synthesis_response",
                result.get("synthesis_response", "No response generated.")
            )

            # Safety: handle if synthesis_response is still a list of content blocks
            if isinstance(synthesis_response, list):
                text_parts = []
                for block in synthesis_response:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                synthesis_response = "\n".join(text_parts)
            synthesis_response = str(synthesis_response)

            # Display the final synthesized answer with proper markdown rendering
            st.markdown("### Clinical Assessment")
            st.markdown(synthesis_response)

            # Add to chat messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": synthesis_response,
            })

            # Save to conversation history
            st.session_state.conversation_history.append({
                "query": query,
                "category": result.get("category", "Unknown"),
                "ailment": result.get("ailment", "Unknown"),
                "classification": result.get("classification", {}),
                "rag_context_length": len(result.get("rag_context", "")),
                "web_search_questions": len(result.get("web_search_context", {})),
                "final_answer": synthesis_response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
        else:
            error_msg = "Pipeline execution failed. Please try again."
            st.markdown(
                f'<div class="agent-msg">{error_msg}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })


if __name__ == "__main__":
    main()
