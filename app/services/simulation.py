"""
P1 Simulation Engine — multi-agent prediction through simulated interaction.

Architecture:
  1. Narrator  → sets the scene each round (temp 0.3)
  2. Actors    → each actor decides an action in parallel (temp 0.5)
  3. Adjudicator → resolves all actions, updates world state (temp 0.1)
  4. Report    → after all rounds, generates a prediction report (temp 0.2)

All LLM calls use JSON mode + structured prompts.
The engine reuses P0's Qwen/OpenAI adapter via app.config.Settings.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import Settings
from app.models.world_model import WorldModel
from app.models.simulation import (
    ActorAction,
    ActorState,
    ActionOutcome,
    AdjudicationResult,
    AlternativeScenario,
    Intervention,
    NarratorOutput,
    PredictionReport,
    RoundResult,
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
    WorldState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory stores (swap for DB later, same pattern as P0)
# ---------------------------------------------------------------------------

_simulations: dict[str, SimulationResult] = {}
_world_models: dict[str, WorldModel] = {}  # populated by P0 extract results


def register_world_model(doc_id: str, wm: WorldModel) -> None:
    """Register an extracted world model so the simulation engine can reference it."""
    _world_models[doc_id] = wm


def get_simulation(sim_id: str) -> SimulationResult | None:
    return _simulations.get(sim_id)


def list_simulations() -> list[SimulationResult]:
    return list(_simulations.values())


# ---------------------------------------------------------------------------
# LLM helper (mirrors P0 extractor pattern)
# ---------------------------------------------------------------------------

# Mock responses for testing without LLM
_MOCK_NARRATOR = {
    "scene": "市场竞争加剧，各方开始调整战略。",
    "new_developments": ["新政策出台", "消费者偏好变化"],
    "pressure_points": ["价格战是否继续", "渠道争夺白热化"],
}

_MOCK_ACTION = {
    "reasoning": "基于当前市场环境，需要采取防御策略。",
    "action": "加强渠道布局，巩固现有市场份额",
    "target": None,
    "expected_outcome": "维持市场地位",
}

_MOCK_ADJUDICATION = {
    "outcomes": [],
    "world_changes": ["市场格局小幅调整"],
    "variable_updates": {},
    "narrative_summary": "各方行动相对温和，格局未发生剧烈变化。",
    "should_terminate": False,
    "termination_reason": None,
}

_MOCK_REPORT = {
    "prediction": "基于模拟推演，市场份额将维持当前格局，小幅波动在 ±2% 以内。",
    "confidence": 0.65,
    "reasoning": "三轮推演显示各方均采取防御策略，无突破性变化。",
    "key_factors": ["政策稳定性", "消费者惯性", "渠道成本"],
    "risks": ["突发政策变化", "新进入者冲击"],
    "alternative_scenarios": [
        {
            "scenario": "价格战升级",
            "probability": 0.2,
            "description": "如果龙头企业发动新一轮价格战，市场份额可能重新洗牌",
        }
    ],
}


async def _llm_call(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    settings: Settings,
) -> dict[str, Any]:
    """
    Make a structured JSON LLM call.

    Returns parsed dict from the LLM response.
    Uses the same provider/model config as P0.
    """
    if settings.use_mock_llm:
        logger.info("Mock LLM call (temp=%.1f)", temperature)
        return {}  # caller handles mock

    api_key = settings.get_api_key()
    if not api_key:
        raise ValueError("No API key configured. Set LLM_API_KEY in .env")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=settings.get_base_url(),
    )

    model = settings.get_model()
    logger.info("LLM call: model=%s temp=%.1f", model, temperature)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("LLM returned empty response")

    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# World-state initialisation from P0 WorldModel
# ---------------------------------------------------------------------------


def init_world_state(world_model: WorldModel) -> WorldState:
    """Convert a P0 WorldModel into the initial WorldState for simulation."""
    actors: dict[str, ActorState] = {}
    for actor in world_model.actors:
        actors[actor.name] = ActorState(
            name=actor.name,
            role=actor.role,
            description=actor.description,
            memory=[f"Initial context: {actor.description}"],
        )

    variables: dict[str, str] = {}
    for var in world_model.variables:
        variables[var.name] = var.current_value

    # Build initial context from timeline
    timeline_summary = "; ".join(
        f"[{ev.date}] {ev.event}" for ev in world_model.timeline
    )
    global_ctx = (
        f"Prediction question: {world_model.question}\n"
        f"Timeline so far: {timeline_summary}\n"
        f"Key relationships: "
        + "; ".join(
            f"{r.from_actor}→{r.to_actor} ({r.type})"
            for r in world_model.relationships
        )
    )

    return WorldState(
        round_number=0,
        actors=actors,
        global_context=global_ctx,
        variables=variables,
        events_log=[f"Simulation initialised with {len(actors)} actors"],
    )


# ---------------------------------------------------------------------------
# Narrator step
# ---------------------------------------------------------------------------

NARRATOR_SYSTEM = """\
You are the **Narrator** in a multi-agent simulation engine called MiroFish.

Your job: given the current world state, set the scene for the next round.
Describe what's happening, introduce new developments, and highlight pressure
points that actors must respond to.

Be vivid but concise. Ground your narration in the world state data.

Return JSON:
{
  "scene": "string — narrative description of the current situation",
  "new_developments": ["string — new events or developments this round"],
  "pressure_points": ["string — key tensions or decision points"]
}
"""


async def narrator_step(
    world_state: WorldState,
    interventions: list[Intervention] | None,
    round_number: int,
    question: str,
    settings: Settings,
) -> NarratorOutput:
    """Generate the scene for a round via the Narrator LLM."""
    if settings.use_mock_llm:
        mock = copy.deepcopy(_MOCK_NARRATOR)
        # Inject intervention if present
        if interventions:
            for iv in interventions:
                if iv.round == round_number:
                    mock["new_developments"].append(f"[INTERVENTION] {iv.event}: {iv.description}")
        return NarratorOutput.model_validate(mock)

    # Build user prompt
    actor_summaries = "\n".join(
        f"- {a.name} ({a.role}): {a.status}"
        for a in world_state.actors.values()
    )
    var_summary = "\n".join(
        f"- {k}: {v}" for k, v in world_state.variables.items()
    )
    recent_events = "\n".join(world_state.events_log[-5:]) if world_state.events_log else "None"

    intervention_text = ""
    if interventions:
        for iv in interventions:
            if iv.round == round_number:
                intervention_text += (
                    f"\n⚠️ EXTERNAL INTERVENTION: {iv.event}\n{iv.description}\n"
                )

    user_prompt = f"""\
## Round {round_number}
## Question: {question}

### Current Actors
{actor_summaries}

### Variables
{var_summary}

### Recent Events
{recent_events}

### Global Context
{world_state.global_context}
{intervention_text}
Set the scene for Round {round_number}. What is happening? What pressures exist?
"""

    data = await _llm_call(NARRATOR_SYSTEM, user_prompt, temperature=0.3, settings=settings)
    return NarratorOutput.model_validate(data)


# ---------------------------------------------------------------------------
# Actor step (one per actor, run in parallel)
# ---------------------------------------------------------------------------

ACTOR_SYSTEM = """\
You are role-playing as **{actor_name}** in a multi-agent simulation.

Your identity:
- Name: {actor_name}
- Role: {actor_role}
- Description: {actor_description}
- Status: {actor_status}

Based on the scene and your memory, decide ONE action for this round.
Think step by step about your goals, the current situation, and your options.

Return JSON:
{{
  "reasoning": "string — your internal chain-of-thought reasoning",
  "action": "string — the specific action you take",
  "target": "string or null — target actor/entity if applicable",
  "expected_outcome": "string — what you hope will happen"
}}
"""


async def _single_actor_step(
    actor: ActorState,
    narrator_output: NarratorOutput,
    world_state: WorldState,
    question: str,
    settings: Settings,
) -> ActorAction:
    """Generate one actor's action."""
    if settings.use_mock_llm:
        mock = copy.deepcopy(_MOCK_ACTION)
        mock["reasoning"] = f"As {actor.name}, I need to respond to the situation."
        mock["action"] = f"{actor.name} takes a defensive stance"
        return ActorAction(actor_name=actor.name, **mock)

    system = ACTOR_SYSTEM.format(
        actor_name=actor.name,
        actor_role=actor.role,
        actor_description=actor.description,
        actor_status=actor.status,
    )

    memory_text = "\n".join(f"- {m}" for m in actor.memory[-10:])
    resources_text = json.dumps(actor.resources, ensure_ascii=False) if actor.resources else "None"

    user_prompt = f"""\
## Current Scene
{narrator_output.scene}

## New Developments
{chr(10).join('- ' + d for d in narrator_output.new_developments)}

## Pressure Points
{chr(10).join('- ' + p for p in narrator_output.pressure_points)}

## Your Memory
{memory_text}

## Your Resources
{resources_text}

## Variables
{chr(10).join(f'- {k}: {v}' for k, v in world_state.variables.items())}

## Prediction Question
{question}

What action do you take this round?
"""

    data = await _llm_call(system, user_prompt, temperature=0.5, settings=settings)
    return ActorAction(actor_name=actor.name, **data)


async def actor_step(
    world_state: WorldState,
    narrator_output: NarratorOutput,
    question: str,
    settings: Settings,
) -> list[ActorAction]:
    """Run all actors in parallel, each deciding their action."""
    active_actors = [
        a for a in world_state.actors.values() if a.status == "active"
    ]

    tasks = [
        _single_actor_step(actor, narrator_output, world_state, question, settings)
        for actor in active_actors
    ]

    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Adjudicator step
# ---------------------------------------------------------------------------

ADJUDICATOR_SYSTEM = """\
You are the **Adjudicator** in a multi-agent simulation engine called MiroFish.

Your job: resolve all actor actions for this round. Determine what actually
happens when these actions play out simultaneously. Be realistic — not every
action succeeds. Consider interactions, conflicts, and unintended consequences.

You also decide whether the simulation should terminate early (e.g. decisive
outcome reached, stalemate, or question clearly answered).

Return JSON:
{{
  "outcomes": [
    {{
      "actor_name": "string",
      "action": "string — what they tried to do",
      "success": true/false,
      "outcome": "string — what actually happened",
      "side_effects": ["string — unintended consequences"]
    }}
  ],
  "world_changes": ["string — summary of changes to the world"],
  "variable_updates": {{"variable_name": "new_value"}},
  "narrative_summary": "string — human-readable summary of the round",
  "should_terminate": false,
  "termination_reason": null
}}
"""


async def adjudicator_step(
    world_state: WorldState,
    narrator_output: NarratorOutput,
    actions: list[ActorAction],
    question: str,
    round_number: int,
    total_rounds: int,
    settings: Settings,
) -> AdjudicationResult:
    """Resolve all actions and determine world-state changes."""
    if settings.use_mock_llm:
        mock = copy.deepcopy(_MOCK_ADJUDICATION)
        # Generate mock outcomes for each action
        mock["outcomes"] = [
            {
                "actor_name": a.actor_name,
                "action": a.action,
                "success": True,
                "outcome": f"{a.actor_name}'s action proceeds as planned.",
                "side_effects": [],
            }
            for a in actions
        ]
        mock["narrative_summary"] = (
            f"Round {round_number}: All actors acted cautiously. "
            "The situation evolves slowly."
        )
        # Terminate on last round
        if round_number >= total_rounds:
            mock["should_terminate"] = True
            mock["termination_reason"] = "Final round reached"
        return AdjudicationResult.model_validate(mock)

    actions_text = "\n".join(
        f"### {a.actor_name}\n"
        f"- Action: {a.action}\n"
        f"- Reasoning: {a.reasoning}\n"
        f"- Target: {a.target or 'N/A'}\n"
        f"- Expected: {a.expected_outcome}"
        for a in actions
    )

    var_text = "\n".join(f"- {k}: {v}" for k, v in world_state.variables.items())

    user_prompt = f"""\
## Round {round_number} of {total_rounds}
## Question: {question}

### Scene
{narrator_output.scene}

### All Actor Actions
{actions_text}

### Current Variables
{var_text}

### Recent Events
{chr(10).join(world_state.events_log[-5:]) if world_state.events_log else 'None'}

Resolve all actions. What actually happens? Update variables as needed.
If the prediction question can be clearly answered or a decisive outcome is reached,
set should_terminate to true.
"""

    data = await _llm_call(
        ADJUDICATOR_SYSTEM, user_prompt, temperature=0.1, settings=settings
    )
    return AdjudicationResult.model_validate(data)


# ---------------------------------------------------------------------------
# State update helpers
# ---------------------------------------------------------------------------


def apply_adjudication(
    world_state: WorldState,
    adjudication: AdjudicationResult,
    round_number: int,
) -> WorldState:
    """Apply adjudication results to the world state (mutates in place)."""
    # Update variables
    for key, value in adjudication.variable_updates.items():
        world_state.variables[key] = value

    # Update events log
    world_state.events_log.append(
        f"[Round {round_number}] {adjudication.narrative_summary}"
    )

    # Update world changes
    for change in adjudication.world_changes:
        world_state.events_log.append(f"  → {change}")

    # Update actor states based on outcomes
    for outcome in adjudication.outcomes:
        actor = world_state.actors.get(outcome.actor_name)
        if actor:
            actor.memory.append(
                f"Round {round_number}: I tried '{outcome.action}' → "
                f"{'Success' if outcome.success else 'Failed'}: {outcome.outcome}"
            )
            if outcome.side_effects:
                actor.memory.append(
                    f"  Side effects: {'; '.join(outcome.side_effects)}"
                )

    world_state.round_number = round_number
    return world_state


def update_actor_memory(
    world_state: WorldState,
    narrator_output: NarratorOutput,
    round_number: int,
) -> None:
    """Add narrator context to all active actors' memory."""
    for actor in world_state.actors.values():
        if actor.status == "active":
            actor.memory.append(
                f"Round {round_number} scene: {narrator_output.scene[:200]}"
            )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

REPORT_SYSTEM = """\
You are the **Report Generator** for MiroFish, a prediction engine.

Given the full simulation history — all rounds, actor actions, outcomes, and
world state evolution — generate a structured prediction report.

Be analytical and evidence-based. Ground your prediction in what happened
during the simulation. Assign a calibrated confidence score.

Return JSON:
{{
  "prediction": "string — clear, specific prediction answer",
  "confidence": 0.0-1.0,
  "reasoning": "string — detailed reasoning chain",
  "key_factors": ["string — top factors driving the prediction"],
  "risks": ["string — key uncertainties"],
  "alternative_scenarios": [
    {{
      "scenario": "string — alternative outcome name",
      "probability": 0.0-1.0,
      "description": "string — what would happen"
    }}
  ]
}}
"""


async def generate_report(
    config: SimulationConfig,
    rounds: list[RoundResult],
    final_state: WorldState,
    settings: Settings,
) -> PredictionReport:
    """Generate the final prediction report after simulation completes."""
    if settings.use_mock_llm:
        mock = copy.deepcopy(_MOCK_REPORT)
        mock["question"] = config.question
        return PredictionReport.model_validate(mock)

    # Build round summaries
    round_summaries = []
    for r in rounds:
        actions_brief = "; ".join(
            f"{a.actor_name}: {a.action}" for a in r.actions
        )
        round_summaries.append(
            f"**Round {r.round_number}**\n"
            f"Scene: {r.narrator.scene[:300]}\n"
            f"Actions: {actions_brief}\n"
            f"Result: {r.adjudication.narrative_summary}"
        )

    final_vars = "\n".join(
        f"- {k}: {v}" for k, v in final_state.variables.items()
    )
    actor_final = "\n".join(
        f"- {a.name} ({a.status}): last memory = {a.memory[-1] if a.memory else 'none'}"
        for a in final_state.actors.values()
    )

    user_prompt = f"""\
## Prediction Question
{config.question}

## Simulation Summary ({len(rounds)} rounds)
{chr(10).join(round_summaries)}

## Final Variables
{final_vars}

## Final Actor States
{actor_final}

## Events Log
{chr(10).join(final_state.events_log[-20:])}

Generate the prediction report.
"""

    data = await _llm_call(REPORT_SYSTEM, user_prompt, temperature=0.2, settings=settings)
    data["question"] = config.question
    return PredictionReport.model_validate(data)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


async def run_simulation(
    config: SimulationConfig,
    settings: Settings | None = None,
) -> SimulationResult:
    """
    Run a full multi-agent simulation.

    Steps per round:
      1. Narrator sets the scene
      2. Each actor decides an action (in parallel)
      3. Adjudicator resolves actions and updates world state
      4. Snapshot the world state

    After all rounds, generate the prediction report.
    """
    if settings is None:
        settings = Settings()

    sim_id = f"sim_{uuid.uuid4().hex[:12]}"

    # Look up the world model
    world_model = _world_models.get(config.world_model_id)
    if not world_model:
        result = SimulationResult(
            simulation_id=sim_id,
            status=SimulationStatus.FAILED,
            config=config,
            error=f"World model '{config.world_model_id}' not found",
            total_rounds=config.rounds,
        )
        _simulations[sim_id] = result
        return result

    # Initialise
    result = SimulationResult(
        simulation_id=sim_id,
        status=SimulationStatus.RUNNING,
        config=config,
        total_rounds=config.rounds,
    )
    _simulations[sim_id] = result

    try:
        world_state = init_world_state(world_model)

        # Filter actors if override specified
        if config.actors_override:
            world_state.actors = {
                name: actor
                for name, actor in world_state.actors.items()
                if name in config.actors_override
            }

        rounds: list[RoundResult] = []

        for round_num in range(1, config.rounds + 1):
            logger.info("Simulation %s: starting round %d/%d", sim_id, round_num, config.rounds)
            result.current_round = round_num

            # 1. Narrator
            narrator_output = await narrator_step(
                world_state=world_state,
                interventions=config.interventions,
                round_number=round_num,
                question=config.question,
                settings=settings,
            )

            # Update actor memories with scene context
            update_actor_memory(world_state, narrator_output, round_num)

            # 2. Actors (parallel)
            actions = await actor_step(
                world_state=world_state,
                narrator_output=narrator_output,
                question=config.question,
                settings=settings,
            )

            # 3. Adjudicator
            adjudication = await adjudicator_step(
                world_state=world_state,
                narrator_output=narrator_output,
                actions=actions,
                question=config.question,
                round_number=round_num,
                total_rounds=config.rounds,
                settings=settings,
            )

            # 4. Apply & snapshot
            # Deep-copy BEFORE applying changes for the snapshot
            snapshot = copy.deepcopy(world_state)
            apply_adjudication(world_state, adjudication, round_num)
            snapshot = copy.deepcopy(world_state)  # snapshot AFTER changes

            round_result = RoundResult(
                round_number=round_num,
                narrator=narrator_output,
                actions=actions,
                adjudication=adjudication,
                world_state_snapshot=snapshot,
            )
            rounds.append(round_result)
            result.rounds = rounds

            # Check early termination
            if adjudication.should_terminate:
                logger.info(
                    "Simulation %s: early termination at round %d — %s",
                    sim_id,
                    round_num,
                    adjudication.termination_reason,
                )
                break

        # 5. Generate report
        report = await generate_report(config, rounds, world_state, settings)

        result.status = SimulationStatus.COMPLETED
        result.report = report
        logger.info("Simulation %s: completed with %d rounds", sim_id, len(rounds))

    except Exception as e:
        logger.exception("Simulation %s failed", sim_id)
        result.status = SimulationStatus.FAILED
        result.error = str(e)

    _simulations[sim_id] = result
    return result


# ---------------------------------------------------------------------------
# Ask (follow-up question on a completed simulation)
# ---------------------------------------------------------------------------

ASK_SYSTEM = """\
You are the MiroFish prediction analyst. A simulation has been completed
and the user has a follow-up question. Answer based on the simulation data.

Return JSON:
{{
  "answer": "string — your answer to the follow-up question",
  "evidence": ["string — supporting evidence from the simulation"],
  "confidence": 0.0-1.0
}}
"""


class AskResponse(BaseModel):
    """Response to a follow-up question about a simulation."""

    answer: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


async def ask_simulation(
    sim_id: str,
    question: str,
    settings: Settings | None = None,
) -> AskResponse:
    """Ask a follow-up question about a completed simulation."""
    if settings is None:
        settings = Settings()

    sim = _simulations.get(sim_id)
    if not sim:
        raise ValueError(f"Simulation '{sim_id}' not found")
    if sim.status != SimulationStatus.COMPLETED:
        raise ValueError(f"Simulation is {sim.status.value}, not completed")

    if settings.use_mock_llm:
        return AskResponse(
            answer="Based on the simulation, the answer is consistent with the main prediction.",
            evidence=["Mock evidence from simulation rounds"],
            confidence=0.6,
        )

    # Summarise simulation for context
    round_briefs = []
    for r in sim.rounds:
        round_briefs.append(
            f"Round {r.round_number}: {r.adjudication.narrative_summary}"
        )

    report_text = ""
    if sim.report:
        report_text = (
            f"Prediction: {sim.report.prediction}\n"
            f"Confidence: {sim.report.confidence}\n"
            f"Reasoning: {sim.report.reasoning}"
        )

    user_prompt = f"""\
## Original Question
{sim.config.question}

## Simulation Summary
{chr(10).join(round_briefs)}

## Prediction Report
{report_text}

## Follow-up Question
{question}
"""

    data = await _llm_call(ASK_SYSTEM, user_prompt, temperature=0.2, settings=settings)
    return AskResponse.model_validate(data)
