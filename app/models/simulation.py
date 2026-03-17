"""Pydantic models for the P1 simulation (prediction) engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.world_model import WorldModel


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------


class SimulationConfig(BaseModel):
    """User-supplied configuration for a simulation run."""

    world_model_id: str = Field(
        ..., description="ID referencing a previously extracted world model"
    )
    question: str = Field(..., description="The prediction question to simulate")
    rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of simulation rounds",
    )
    actors_override: list[str] | None = Field(
        default=None,
        description="Subset of actor names to include; None = all",
    )
    interventions: list[Intervention] | None = Field(
        default=None,
        description="Pre-scheduled events injected at specific rounds",
    )

    model_config = {"populate_by_name": True}


class Intervention(BaseModel):
    """An external event the user wants injected at a specific round."""

    round: int = Field(..., ge=1, description="Round number to inject this event")
    event: str = Field(..., description="Short title of the intervention")
    description: str = Field(..., description="Detailed description of what happens")


# Fix forward reference — Intervention is used before its definition in
# SimulationConfig, so we re-declare the field after Intervention exists.
SimulationConfig.model_rebuild()


# ---------------------------------------------------------------------------
# World / actor state
# ---------------------------------------------------------------------------


class ActorState(BaseModel):
    """Per-actor state tracked across rounds."""

    name: str
    role: str
    description: str
    memory: list[str] = Field(
        default_factory=list,
        description="Running memory of what this actor has experienced",
    )
    status: str = Field(
        default="active",
        description="Current status: active, eliminated, passive …",
    )
    resources: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form resource tracking (e.g. market_share, budget)",
    )


class WorldState(BaseModel):
    """Full snapshot of the simulation world at a point in time."""

    round_number: int = 0
    actors: dict[str, ActorState] = Field(default_factory=dict)
    global_context: str = Field(
        default="",
        description="Narrative context visible to all actors",
    )
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Key variables and their current values",
    )
    events_log: list[str] = Field(
        default_factory=list,
        description="Chronological log of everything that happened",
    )


# ---------------------------------------------------------------------------
# Per-round artefacts
# ---------------------------------------------------------------------------


class NarratorOutput(BaseModel):
    """Output from the Narrator step: scene-setting for a round."""

    scene: str = Field(..., description="Narrative description of the current situation")
    new_developments: list[str] = Field(
        default_factory=list,
        description="New events or developments introduced this round",
    )
    pressure_points: list[str] = Field(
        default_factory=list,
        description="Key tensions or decision points actors must face",
    )


class ActorAction(BaseModel):
    """A single actor's decided action for the round."""

    actor_name: str
    reasoning: str = Field(..., description="Internal reasoning (chain-of-thought)")
    action: str = Field(..., description="The action taken")
    target: str | None = Field(
        default=None, description="Target actor/entity if applicable"
    )
    expected_outcome: str = Field(
        default="", description="What the actor hopes will happen"
    )


class AdjudicationResult(BaseModel):
    """Adjudicator's resolution of all actions in a round."""

    outcomes: list[ActionOutcome] = Field(
        ..., description="Resolved outcome per action"
    )
    world_changes: list[str] = Field(
        default_factory=list,
        description="Summary of changes to the world state",
    )
    variable_updates: dict[str, str] = Field(
        default_factory=dict,
        description="Updated variable values after this round",
    )
    narrative_summary: str = Field(
        ..., description="Human-readable summary of what happened"
    )
    should_terminate: bool = Field(
        default=False,
        description="True if the simulation should end early",
    )
    termination_reason: str | None = Field(
        default=None, description="Why the simulation is ending early"
    )


class ActionOutcome(BaseModel):
    """Resolution of a single actor's action."""

    actor_name: str
    action: str
    success: bool
    outcome: str = Field(..., description="What actually happened")
    side_effects: list[str] = Field(
        default_factory=list, description="Unintended consequences"
    )


# Rebuild models with forward references
AdjudicationResult.model_rebuild()


class RoundResult(BaseModel):
    """Complete record of one simulation round."""

    round_number: int
    narrator: NarratorOutput
    actions: list[ActorAction]
    adjudication: AdjudicationResult
    world_state_snapshot: WorldState


# ---------------------------------------------------------------------------
# Final outputs
# ---------------------------------------------------------------------------


class PredictionReport(BaseModel):
    """Final prediction report generated after the simulation."""

    question: str
    prediction: str = Field(..., description="Clear prediction answer")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score 0-1"
    )
    reasoning: str = Field(..., description="Detailed reasoning chain")
    key_factors: list[str] = Field(
        ..., description="Top factors that drove the prediction"
    )
    risks: list[str] = Field(
        default_factory=list, description="Key risks / uncertainties"
    )
    alternative_scenarios: list[AlternativeScenario] = Field(
        default_factory=list,
        description="Alternative outcomes considered",
    )


class AlternativeScenario(BaseModel):
    """An alternative outcome the simulation explored."""

    scenario: str
    probability: float = Field(ge=0.0, le=1.0)
    description: str


PredictionReport.model_rebuild()


class SimulationStatus(str, Enum):
    """Status of a simulation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SimulationResult(BaseModel):
    """Top-level result of a simulation run."""

    simulation_id: str
    status: SimulationStatus
    config: SimulationConfig
    rounds: list[RoundResult] = Field(default_factory=list)
    report: PredictionReport | None = None
    error: str | None = None
    current_round: int = 0
    total_rounds: int = 0
