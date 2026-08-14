"""
Tool functions for the Speech/Feeding Medical AI Agent.

These tools are invoked by the agent during its reasoning process.
Each tool receives a `tool_context` from Google ADK which provides
access to session state for conversational memory.

All tool outputs are validated through Pydantic models to ensure
structured dictionary output.
"""

import json
from typing import Optional

from google.adk.tools import ToolContext

from agent.config import (
    CATEGORY_SPEECH,
    CATEGORY_FEEDING,
    VALID_CATEGORIES,
    SPEECH_CONDITIONS,
    FEEDING_CONDITIONS,
    STATE_CONVERSATION_HISTORY,
    STATE_PREVIOUS_AILMENTS,
    STATE_LAST_CATEGORY,
    STATE_LAST_AILMENT,
    STATE_INTERACTION_COUNT,
)
from agent.models import (
    ClassificationResult,
    AilmentIdentification,
    ConversationContext,
    AgentOutput,
)


def classify_query(
    query: str,
    reasoning: str,
    tool_context: ToolContext,
) -> dict:
    """Classify a user query into either 'Speech' or 'Feeding' category.

    Analyze the user's description of a patient's condition and determine
    whether it falls under Speech disorders or Feeding disorders.

    Args:
        query: The user's original query or patient description.
        reasoning: Your clinical reasoning for why this query belongs to
            the chosen category. Explain which symptoms or keywords led
            to your classification.
        tool_context: Provided automatically by the ADK framework.

    Returns:
        A dictionary with the classification result including category
        and the reasoning behind the classification.
    """
    # Determine category based on the agent's reasoning
    # The agent itself does the classification via the reasoning parameter;
    # this tool structures and persists the result.
    category = CATEGORY_SPEECH  # default
    reasoning_lower = reasoning.lower()

    # Check for feeding-related keywords in the agent's reasoning
    feeding_keywords = [
        "feeding", "swallowing", "dysphagia", "aspiration", "bolus",
        "pharyngeal", "esophageal", "oral phase", "food refusal",
        "texture aversion", "failure to thrive", "gerd", "reflux",
        "choking", "coughing when eating", "coughing when drinking",
        "oral motor", "mastication", "deglutition",
    ]
    speech_keywords = [
        "speech", "language", "articulation", "phonological", "stuttering",
        "fluency", "voice", "aphasia", "apraxia", "dysarthria",
        "resonance", "communication", "pronunciation", "selective mutism",
        "cleft palate", "laryngeal", "dysphonia", "vocal fold",
    ]

    feeding_score = sum(1 for kw in feeding_keywords if kw in reasoning_lower)
    speech_score = sum(1 for kw in speech_keywords if kw in reasoning_lower)

    if feeding_score > speech_score:
        category = CATEGORY_FEEDING
    else:
        category = CATEGORY_SPEECH

    # Store in session state for memory
    tool_context.state[STATE_LAST_CATEGORY] = category

    # Validate through Pydantic
    validated = ClassificationResult(category=category, reasoning=reasoning)

    result = validated.model_dump()
    result["available_conditions"] = (
        SPEECH_CONDITIONS if category == CATEGORY_SPEECH else FEEDING_CONDITIONS
    )

    return result


def identify_ailment(
    ailment_name: str,
    ailment_description: str,
    category: str,
    symptoms_noted: str,
    tool_context: ToolContext,
) -> dict:
    """Identify the specific medical ailment from the user's description.

    Based on the classification and the user's description, identify the
    precise clinical condition the patient is experiencing.

    Args:
        ailment_name: The identified clinical name of the condition
            (e.g., 'Dysarthria', 'Oropharyngeal Dysphagia').
        ailment_description: A brief clinical description of the condition,
            including key characteristics.
        category: The category from classification ('Speech' or 'Feeding').
        symptoms_noted: Key symptoms from the user's description that led
            to this identification.
        tool_context: Provided automatically by the ADK framework.

    Returns:
        A dictionary with the identified ailment details.
    """
    # Validate category
    if category not in VALID_CATEGORIES:
        category = tool_context.state.get(STATE_LAST_CATEGORY, CATEGORY_SPEECH)

    # Store in session state
    tool_context.state[STATE_LAST_AILMENT] = ailment_name

    # Validate through Pydantic
    validated = AilmentIdentification(
        ailment=ailment_name,
        description=ailment_description,
        category=category,
        symptoms_noted=symptoms_noted,
    )

    return validated.model_dump()


def generate_treatment_questions(
    ailment: str,
    category: str,
    question_1_onset_severity: str,
    question_2_functional_impact: str,
    question_3_previous_treatments: str,
    tool_context: ToolContext,
) -> dict:
    """Generate 3 clinically relevant treatment questions for the identified ailment.

    Create targeted assessment questions that a speech-language pathologist
    would ask to guide treatment planning. This produces the final structured
    dictionary that is the agent's primary output.

    Args:
        ailment: The identified medical condition name.
        category: The classification category ('Speech' or 'Feeding').
        question_1_onset_severity: A question about when the condition
            started and how severe it is.
        question_2_functional_impact: A question about how the condition
            affects the patient's daily life and activities.
        question_3_previous_treatments: A question about what treatments
            or interventions have been tried previously.
        tool_context: Provided automatically by the ADK framework.

    Returns:
        The final structured dictionary containing the complete assessment
        result with category, ailment, description, and treatment questions.
    """
    # Retrieve context from memory
    previous_ailments = tool_context.state.get(STATE_PREVIOUS_AILMENTS, "[]")
    if isinstance(previous_ailments, str):
        try:
            previous_ailments = json.loads(previous_ailments)
        except (json.JSONDecodeError, TypeError):
            previous_ailments = []

    interaction_count = tool_context.state.get(STATE_INTERACTION_COUNT, 0)
    if isinstance(interaction_count, str):
        try:
            interaction_count = int(interaction_count)
        except (ValueError, TypeError):
            interaction_count = 0

    ailment_description = tool_context.state.get(STATE_LAST_AILMENT, ailment)

    # Validate through Pydantic — this is the primary structured output
    validated = AgentOutput(
        category=category,
        ailment=ailment,
        ailment_description=ailment_description,
        treatment_questions=[
            question_1_onset_severity,
            question_2_functional_impact,
            question_3_previous_treatments,
        ],
        conversation_context=ConversationContext(
            previous_ailments=previous_ailments,
            interaction_number=interaction_count + 1,
        ),
    )

    return validated.to_dict()


def save_to_memory(
    ailment: str,
    category: str,
    questions_summary: str,
    tool_context: ToolContext,
) -> str:
    """Save the current interaction to conversational memory.

    Persists the ailment, category, and a summary of the questions generated
    so they can be recalled in future conversation turns.

    Args:
        ailment: The identified medical condition name.
        category: The classification category ('Speech' or 'Feeding').
        questions_summary: A brief summary of the 3 treatment questions
            that were generated.
        tool_context: Provided automatically by the ADK framework.

    Returns:
        A confirmation message that the interaction was saved to memory.
    """
    # Update interaction count
    interaction_count = tool_context.state.get(STATE_INTERACTION_COUNT, 0)
    if isinstance(interaction_count, str):
        try:
            interaction_count = int(interaction_count)
        except (ValueError, TypeError):
            interaction_count = 0
    tool_context.state[STATE_INTERACTION_COUNT] = interaction_count + 1

    # Update previous ailments list
    previous_ailments = tool_context.state.get(STATE_PREVIOUS_AILMENTS, "[]")
    if isinstance(previous_ailments, str):
        try:
            previous_ailments = json.loads(previous_ailments)
        except (json.JSONDecodeError, TypeError):
            previous_ailments = []

    # Add current ailment to history
    current_entry = {
        "ailment": ailment,
        "category": category,
        "questions_summary": questions_summary,
        "interaction_number": interaction_count + 1,
    }
    previous_ailments.append(current_entry)
    tool_context.state[STATE_PREVIOUS_AILMENTS] = json.dumps(previous_ailments)

    # Update conversation history
    history = tool_context.state.get(STATE_CONVERSATION_HISTORY, "[]")
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except (json.JSONDecodeError, TypeError):
            history = []

    history.append(current_entry)
    tool_context.state[STATE_CONVERSATION_HISTORY] = json.dumps(history)

    return (
        f"Saved to memory: {category} → {ailment} "
        f"(interaction #{interaction_count + 1}). "
        f"Total ailments discussed: {len(previous_ailments)}."
    )
