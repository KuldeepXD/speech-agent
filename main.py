"""
Speech/Feeding Medical AI Agent — CLI Entry Point.

Provides both single-query and interactive conversation modes.
Run with: python main.py
"""

import asyncio
import uuid
import json
import sys
import os
from pprint import pformat

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agent.speech_feeding_agent import create_agent, create_runner
from agent.config import AGENT_NAME
from agent.models import AgentOutput


# ── Colours for terminal output ─────────────────────────────────────────
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner():
    """Print the application banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗
║         🏥  Speech/Feeding Medical AI Agent  🏥              ║
║                                                              ║
║   Classifies conditions · Identifies ailments                ║
║   Generates treatment questions · Remembers context          ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def print_help():
    """Print help information."""
    help_text = f"""
{Colors.YELLOW}{Colors.BOLD}Commands:{Colors.RESET}
  {Colors.GREEN}Type any patient description{Colors.RESET} — The agent will classify, identify, and generate questions
  {Colors.GREEN}history{Colors.RESET}                       — View conversation history from memory
  {Colors.GREEN}help{Colors.RESET}                          — Show this help message
  {Colors.GREEN}quit / exit / q{Colors.RESET}               — Exit the application

{Colors.YELLOW}{Colors.BOLD}Example queries:{Colors.RESET}
  {Colors.DIM}• "My child has trouble pronouncing the R sound and stutters when nervous"
  • "Patient is coughing when swallowing liquids after their stroke"
  • "3-year-old refuses to eat solid foods and gags on any texture"
  • "Adult patient has hoarse voice and vocal fatigue after prolonged speaking"{Colors.RESET}
"""
    print(help_text)


async def run_agent_query(
    runner: Runner,
    session_service: InMemorySessionService,
    user_id: str,
    session_id: str,
    query: str,
) -> tuple[dict | None, str]:
    """Run a single query through the agent and return the structured dictionary.

    Intercepts the `generate_treatment_questions` tool result to capture
    the Pydantic-validated dictionary output directly.

    Args:
        runner: The ADK Runner instance.
        session_service: The session service for state management.
        user_id: The user identifier.
        session_id: The session identifier.
        query: The user's query text.

    Returns:
        A tuple of (structured_dict, text_response). The structured_dict
        is the Pydantic-validated AgentOutput dictionary if available,
        otherwise None. text_response is the agent's prose reply.
    """
    # Create the user message content
    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)],
    )

    # Run the agent and collect responses
    final_response_parts = []
    structured_output = None

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        # Intercept function call results to capture the structured dict
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_response:
                    fn_name = part.function_response.name
                    fn_result = part.function_response.response
                    if fn_name == "generate_treatment_questions" and fn_result:
                        # Extract the validated dictionary from the tool result
                        result_data = fn_result.get("result", fn_result)
                        if isinstance(result_data, dict) and "category" in result_data:
                            structured_output = result_data

        # Collect the final agent text response
        if event.is_final_response():
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_response_parts.append(part.text)

    text_response = "\n".join(final_response_parts) if final_response_parts else "(No response)"
    return structured_output, text_response


async def interactive_mode():
    """Run the agent in interactive conversation mode."""
    print_banner()
    print_help()

    # Create agent and runner
    print(f"{Colors.DIM}Initializing agent...{Colors.RESET}")
    runner, session_service = create_runner()

    # Create a session for this conversation
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    # Create the session
    session = await session_service.create_session(
        app_name=AGENT_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    print(f"{Colors.GREEN}✓ Agent ready!{Colors.RESET}")
    print(f"{Colors.DIM}  Session: {session_id} | User: {user_id}{Colors.RESET}")
    print(f"{Colors.DIM}  (Conversational memory is active across turns){Colors.RESET}\n")

    # Conversation loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.BOLD}{Colors.BLUE}You ▶ {Colors.RESET}").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ("quit", "exit", "q"):
                print(f"\n{Colors.CYAN}Goodbye! 👋{Colors.RESET}\n")
                break

            if user_input.lower() == "help":
                print_help()
                continue

            if user_input.lower() == "history":
                # Retrieve memory from session state
                session = await session_service.get_session(
                    app_name=AGENT_NAME,
                    user_id=user_id,
                    session_id=session_id,
                )
                if session and session.state:
                    history = session.state.get("user:conversation_history", "[]")
                    if isinstance(history, str):
                        try:
                            history = json.loads(history)
                        except (json.JSONDecodeError, TypeError):
                            history = []
                    if history:
                        print(f"\n{Colors.YELLOW}{Colors.BOLD}📋 Conversation History:{Colors.RESET}")
                        for i, entry in enumerate(history, 1):
                            print(
                                f"  {Colors.GREEN}{i}.{Colors.RESET} "
                                f"[{entry.get('category', '?')}] "
                                f"{Colors.BOLD}{entry.get('ailment', '?')}{Colors.RESET}"
                            )
                        print()
                    else:
                        print(f"\n{Colors.DIM}No history yet. Ask about a condition!{Colors.RESET}\n")
                else:
                    print(f"\n{Colors.DIM}No history yet. Ask about a condition!{Colors.RESET}\n")
                continue

            # Run the query
            print(f"\n{Colors.DIM}🔍 Analyzing...{Colors.RESET}\n")

            structured_dict, text_response = await run_agent_query(
                runner=runner,
                session_service=session_service,
                user_id=user_id,
                session_id=session_id,
                query=user_input,
            )

            # Display the structured dictionary output
            if structured_dict:
                print(f"{Colors.GREEN}{Colors.BOLD}Agent ▶ Structured Output (dict):{Colors.RESET}")
                print(f"{Colors.CYAN}{json.dumps(structured_dict, indent=2)}{Colors.RESET}\n")
            else:
                print(f"{Colors.GREEN}{Colors.BOLD}Agent ▶{Colors.RESET} {text_response}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Goodbye! 👋{Colors.RESET}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.RESET}\n")
            continue


async def single_query_mode(query: str):
    """Run a single query and print the result.

    Args:
        query: The user's query text.
    """
    runner, session_service = create_runner()

    user_id = "cli_user"
    session_id = f"single_{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name=AGENT_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    structured_dict, text_response = await run_agent_query(
        runner=runner,
        session_service=session_service,
        user_id=user_id,
        session_id=session_id,
        query=query,
    )

    # Print the structured dictionary output
    if structured_dict:
        print(json.dumps(structured_dict, indent=2))
    else:
        print(text_response)


def pipeline_mode(query: str):
    """Run the full LangGraph pipeline (classification + parallel RAG & PubMed Search + synthesis).

    This runs the complete pipeline:
        Classification → Parallel [RAG (conditional) + PubMed Search (always)] → Synthesis

    Args:
        query: The user's query text.
    """
    from langchain_core.messages import HumanMessage
    from langgraph_bridge import create_pipeline

    print(f"\n[Running full LangGraph pipeline (RAG + PubMed Search + Synthesis)...]")
    print(f"   Query: {query}\n")

    pipeline = create_pipeline()

    result = pipeline.invoke({
        "query": query,
        "messages": [HumanMessage(content=query)],
    })

    # Print the classification result
    classification = result.get("classification")
    if classification:
        print(f"[Node 1] Classification Result:")
        print(f"{json.dumps(classification, indent=2)}\n")

    # Print routing info
    category = result.get("category", "Unknown")
    print(f"[Router] Routed to: {category} RAG Agent\n")

    # Print web search results
    web_search_context = result.get("web_search_context", {})
    if web_search_context:
        print(f"[PubMed Search] Results for {len(web_search_context)} questions:")
        for key, entry in web_search_context.items():
            print(f"  • {entry.get('original_question', key)}")
            print(f"    Query: {entry.get('query', 'N/A')}")
            results_text = entry.get('results', '')
            # Truncate long results for display
            if len(results_text) > 200:
                print(f"    Results: {results_text[:200]}...")
            else:
                print(f"    Results: {results_text}")
        print()

    # Print the final synthesized output
    final_output = result.get("final_output")
    if final_output:
        synthesis = final_output.get("synthesis_response", "")
        if synthesis:
            print(f"[Synthesis] Combined RAG + PubMed Search Answer:")
            print(f"{synthesis}\n")
        else:
            print(f"[Final Output] (dict):")
            print(json.dumps(final_output, indent=2))
    else:
        # Fallback to RAG response text
        rag_response = result.get("rag_response", "(No RAG response)")
        print(f"[RAG Response]:\n{rag_response}")


def main():
    """Entry point — routes to interactive, single-query, or pipeline mode."""
    args = sys.argv[1:]

    if "--pipeline" in args:
        # Pipeline mode: python main.py --pipeline "patient has trouble swallowing"
        args.remove("--pipeline")
        if args:
            query = " ".join(args)
            pipeline_mode(query)
        else:
            print("Usage: python main.py --pipeline \"your query here\"")
            sys.exit(1)
    elif args:
        # Single query mode: python main.py "patient has trouble swallowing"
        query = " ".join(args)
        asyncio.run(single_query_mode(query))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
