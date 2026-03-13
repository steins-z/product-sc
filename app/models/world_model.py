"""Pydantic models for the extracted world model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Actor(BaseModel):
    """An actor (person, organization, entity) in the world model."""

    name: str = Field(..., description="Name of the actor")
    role: str = Field(..., description="Role or function of the actor")
    description: str = Field(..., description="Detailed description")
    source_ref: list[str] = Field(
        ..., description="List of chunk_ids that support this extraction"
    )


class Relationship(BaseModel):
    """A relationship between two actors."""

    from_actor: str = Field(..., alias="from", description="Source actor name")
    to_actor: str = Field(..., alias="to", description="Target actor name")
    type: str = Field(..., description="Relationship type (e.g., competes_with, supplies)")
    description: str = Field(..., description="Description of the relationship")
    source_ref: list[str] = Field(
        ..., description="List of chunk_ids that support this extraction"
    )

    model_config = {"populate_by_name": True}


class TimelineEvent(BaseModel):
    """A timeline event extracted from the document."""

    date: str = Field(..., description="Date or time period (ISO or descriptive)")
    event: str = Field(..., description="Brief event title")
    description: str = Field(..., description="Detailed event description")
    source_ref: list[str] = Field(
        ..., description="List of chunk_ids that support this extraction"
    )


class Variable(BaseModel):
    """A key variable that could affect the prediction."""

    name: str = Field(..., description="Variable name")
    current_value: str = Field(..., description="Current or latest known value")
    description: str = Field(..., description="What this variable represents and why it matters")
    source_ref: list[str] = Field(
        ..., description="List of chunk_ids that support this extraction"
    )


class WorldModel(BaseModel):
    """Complete world model extracted from a document for a prediction question."""

    actors: list[Actor] = Field(default_factory=list, description="Key actors")
    relationships: list[Relationship] = Field(
        default_factory=list, description="Relationships between actors"
    )
    timeline: list[TimelineEvent] = Field(
        default_factory=list, description="Chronological events"
    )
    variables: list[Variable] = Field(
        default_factory=list, description="Key variables affecting the prediction"
    )
    question: str = Field(..., description="The prediction question")


class ExtractionRequest(BaseModel):
    """Request body for the extraction endpoint."""

    question: str = Field(..., description="The prediction question to guide extraction")


class ExtractionResponse(BaseModel):
    """Full extraction response including metadata."""

    document_id: str
    question: str
    world_model: WorldModel
    chunks_processed: int
