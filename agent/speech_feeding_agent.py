"""
Speech/Feeding Medical AI Agent — Core Agent Definition.

Creates and configures the Google ADK Agent with tools and session management.
"""

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agent.config import MODEL_NAME, AGENT_NAME, AGENT_DESCRIPTION
from agent.prompts import AGENT_INSTRUCTION
from agent.tools import (
    classify_query,
    identify_ailment,
    generate_treatment_questions,
    save_to_memory,
)


def create_agent() -> Agent:
    """Create and return the Speech/Feeding Medical AI Agent.

    Returns:
        A configured Google ADK Agent instance with all tools attached.
    """
    agent = Agent(
        model=MODEL_NAME,
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        instruction=AGENT_INSTRUCTION,
        tools=[
            classify_query,
            identify_ailment,
            generate_treatment_questions,
            save_to_memory,
        ],
    )
    return agent


def create_runner(agent: Agent = None) -> tuple[Runner, InMemorySessionService]:
    """Create a Runner and SessionService for executing the agent.

    Args:
        agent: An optional pre-created Agent instance. If None, a new
            agent will be created via create_agent().

    Returns:
        A tuple of (Runner, InMemorySessionService) for running the agent.
    """
    if agent is None:
        agent = create_agent()

    session_service = InMemorySessionService()

    runner = Runner(
        agent=agent,
        app_name=AGENT_NAME,
        session_service=session_service,
    )

    return runner, session_service
