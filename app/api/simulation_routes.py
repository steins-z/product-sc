"""API routes for the P1 simulation engine."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.models.simulation import (
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
)
from app.services.simulation import (
    AskResponse,
    ask_simulation,
    get_simulation,
    list_simulations,
    run_simulation,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulations", tags=["simulation"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateSimulationRequest(BaseModel):
    """Request body for POST /simulations."""

    world_model_id: str = Field(
        ..., description="Document ID of the world model to simulate"
    )
    question: str = Field(..., description="Prediction question")
    rounds: int = Field(default=3, ge=1, le=10, description="Number of rounds")
    actors_override: list[str] | None = Field(
        default=None, description="Subset of actor names to simulate"
    )
    interventions: list[InterventionInput] | None = Field(
        default=None, description="Pre-scheduled events to inject"
    )


class InterventionInput(BaseModel):
    round: int = Field(..., ge=1)
    event: str
    description: str


# Rebuild for forward ref
CreateSimulationRequest.model_rebuild()


class SimulationSummary(BaseModel):
    """Lightweight simulation info for listing."""

    simulation_id: str
    status: SimulationStatus
    question: str
    current_round: int
    total_rounds: int


class AskRequest(BaseModel):
    """Request body for POST /simulations/{id}/ask."""

    question: str = Field(..., description="Follow-up question about the simulation")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=202, response_model=SimulationSummary)
async def create_simulation(
    request: CreateSimulationRequest,
    background_tasks: BackgroundTasks,
) -> SimulationSummary:
    """
    Start a new simulation run (async — returns 202 immediately).

    The simulation runs in the background. Poll GET /simulations/{id}
    for status updates.
    """
    from app.models.simulation import Intervention

    config = SimulationConfig(
        world_model_id=request.world_model_id,
        question=request.question,
        rounds=request.rounds,
        actors_override=request.actors_override,
        interventions=[
            Intervention(round=iv.round, event=iv.event, description=iv.description)
            for iv in request.interventions
        ]
        if request.interventions
        else None,
    )

    # Pre-create a placeholder so we can return the ID immediately
    from app import db as _db

    import uuid

    sim_id = f"sim_{uuid.uuid4().hex[:12]}"

    placeholder = SimulationResult(
        simulation_id=sim_id,
        status=SimulationStatus.PENDING,
        config=config,
        total_rounds=config.rounds,
    )
    await _db.save_simulation(placeholder)

    async def _run() -> None:
        """Background task wrapper."""
        try:
            result = await run_simulation(config)
            # Update the placeholder with the real result
            # (run_simulation creates its own ID, so we patch)
            result.simulation_id = sim_id
            await _db.save_simulation(result)
        except Exception as e:
            logger.exception("Background simulation %s failed", sim_id)
            placeholder.status = SimulationStatus.FAILED
            placeholder.error = str(e)
            await _db.save_simulation(placeholder)

    background_tasks.add_task(_run)

    logger.info("Simulation %s queued (rounds=%d)", sim_id, config.rounds)

    return SimulationSummary(
        simulation_id=sim_id,
        status=SimulationStatus.PENDING,
        question=config.question,
        current_round=0,
        total_rounds=config.rounds,
    )


@router.get("", response_model=list[SimulationSummary])
async def list_all_simulations() -> list[SimulationSummary]:
    """List all simulation runs."""
    return [
        SimulationSummary(
            simulation_id=s.simulation_id,
            status=s.status,
            question=s.config.question,
            current_round=s.current_round,
            total_rounds=s.total_rounds,
        )
        for s in await list_simulations()
    ]


@router.get("/{simulation_id}", response_model=SimulationResult)
async def get_simulation_detail(simulation_id: str) -> SimulationResult:
    """Get full simulation result (including all rounds and report)."""
    sim = await get_simulation(simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation '{simulation_id}' not found")
    return sim


@router.get("/{simulation_id}/report")
async def get_simulation_report(simulation_id: str) -> dict:
    """Get just the prediction report for a completed simulation."""
    sim = await get_simulation(simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation '{simulation_id}' not found")
    if sim.status != SimulationStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Simulation is {sim.status.value}. Report available only when completed.",
        )
    if not sim.report:
        raise HTTPException(status_code=404, detail="No report generated")

    return sim.report.model_dump()


@router.post("/{simulation_id}/ask", response_model=AskResponse)
async def ask_about_simulation(
    simulation_id: str,
    request: AskRequest,
) -> AskResponse:
    """
    Ask a follow-up question about a completed simulation.

    The AI will answer based on the full simulation context.
    """
    try:
        return await ask_simulation(simulation_id, request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
