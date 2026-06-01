"""
Pydantic models for structured output of the Speech/Feeding Medical AI Agent.

These models enforce a strict schema for the agent's return values,
ensuring all outputs are validated dictionaries.
"""

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Result of classifying a user query into Speech or Feeding."""

    category: str = Field(
        ...,
        description="The classification category: 'Speech' or 'Feeding'",
        pattern=r"^(Speech|Feeding)$",
    )
    reasoning: str = Field(
        ...,
        description="Clinical reasoning for the classification decision",
    )


class AilmentIdentification(BaseModel):
    """Result of identifying a specific medical ailment."""

    ailment: str = Field(
        ...,
        description="The identified clinical condition name (e.g., 'Dysarthria')",
    )
    description: str = Field(
        ...,
        description="Brief clinical description of the identified condition",
    )
    category: str = Field(
        ...,
        description="The classification category: 'Speech' or 'Feeding'",
    )
    symptoms_noted: str = Field(
        ...,
        description="Key symptoms from the user's description",
    )


class ConversationContext(BaseModel):
    """Conversational memory context from the session."""

    previous_ailments: list[dict] = Field(
        default_factory=list,
        description="List of previously discussed ailments in this session",
    )
    interaction_number: int = Field(
        default=1,
        description="The sequential interaction number in this session",
    )


class AgentOutput(BaseModel):
    """The final structured output dictionary returned by the agent.

    This is the primary return type — a validated dictionary containing
    the classification, identified ailment, and treatment questions.
    """

    category: str = Field(
        ...,
        description="The classification category: 'Speech' or 'Feeding'",
    )
    ailment: str = Field(
        ...,
        description="The identified medical condition name",
    )
    ailment_description: str = Field(
        ...,
        description="Brief clinical description of the condition",
    )
    treatment_questions: list[str] = Field(
        ...,
        description="Exactly 3 clinically relevant treatment questions",
        min_length=3,
        max_length=3,
    )
    conversation_context: ConversationContext = Field(
        default_factory=ConversationContext,
        description="Memory context from the conversation session",
    )

    def to_dict(self) -> dict:
        """Return the validated output as a plain Python dictionary."""
        return self.model_dump()
