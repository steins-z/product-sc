"""Tests for the P1 simulation engine."""

from __future__ import annotations

import asyncio
import copy
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import Settings
from app.models.world_model import (
    Actor,
    Relationship,
    TimelineEvent,
    Variable,
    WorldModel,
)
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
from app.services.simulation import (
    AskResponse,
    apply_adjudication,
    ask_simulation,
    generate_report,
    init_world_state,
    run_simulation,
    update_actor_memory,
)
from app import db
from app.models.world_model import ExtractionResponse


async def _save_test_world_model(doc_id: str, wm: WorldModel) -> None:
    """Helper: save a world model to DB so simulation can find it."""
    extraction = ExtractionResponse(
        document_id=doc_id,
        question=wm.question,
        world_model=wm,
        chunks_processed=1,
    )
    await db.save_world_model(doc_id, extraction)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_world_model() -> WorldModel:
    """A minimal world model for testing."""
    return WorldModel(
        actors=[
            Actor(
                name="农夫山泉",
                role="market_leader",
                description="中国最大的包装饮用水企业",
                source_ref=["chunk_0"],
            ),
            Actor(
                name="华润怡宝",
                role="challenger",
                description="纯净水领域领导品牌",
                source_ref=["chunk_0"],
            ),
        ],
        relationships=[
            Relationship(
                from_actor="农夫山泉",
                to_actor="华润怡宝",
                type="competition",
                description="天然水vs纯净水之争",
                source_ref=["chunk_0"],
                **{"from": "农夫山泉", "to": "华润怡宝"},  # alias for serialization
            ),
        ],
        timeline=[
            TimelineEvent(
                date="2024-02",
                event="绿瓶纯净水攻势",
                description="农夫山泉推出1元绿瓶纯净水",
                source_ref=["chunk_0"],
            ),
        ],
        variables=[
            Variable(
                name="market_share",
                current_value="农夫山泉 26%, 华润怡宝 21%",
                description="包装水市场份额",
                source_ref=["chunk_0"],
            ),
        ],
        question="未来12个月农夫山泉市场份额会如何变化？",
    )


@pytest.fixture
def mock_settings() -> Settings:
    """Settings with mock LLM enabled."""
    return Settings(use_mock_llm=True)


@pytest_asyncio.fixture(autouse=True)
async def clean_stores(tmp_path):
    """Init a fresh test DB before each test."""
    db_path = str(tmp_path / "test_sim.db")
    await db.init_db(db_path)
    yield


# ---------------------------------------------------------------------------
# TestInitWorldState
# ---------------------------------------------------------------------------


class TestInitWorldState:
    """Tests for converting P0 WorldModel → simulation WorldState."""

    def test_basic_init(self, sample_world_model: WorldModel):
        state = init_world_state(sample_world_model)

        assert state.round_number == 0
        assert len(state.actors) == 2
        assert "农夫山泉" in state.actors
        assert "华润怡宝" in state.actors

    def test_actor_properties(self, sample_world_model: WorldModel):
        state = init_world_state(sample_world_model)
        nfs = state.actors["农夫山泉"]

        assert nfs.name == "农夫山泉"
        assert nfs.role == "market_leader"
        assert nfs.status == "active"
        assert len(nfs.memory) == 1
        assert "Initial context" in nfs.memory[0]

    def test_variables_carried_over(self, sample_world_model: WorldModel):
        state = init_world_state(sample_world_model)

        assert "market_share" in state.variables
        assert "26%" in state.variables["market_share"]

    def test_global_context_includes_timeline(self, sample_world_model: WorldModel):
        state = init_world_state(sample_world_model)

        assert "绿瓶纯净水攻势" in state.global_context

    def test_global_context_includes_relationships(self, sample_world_model: WorldModel):
        state = init_world_state(sample_world_model)

        assert "农夫山泉" in state.global_context
        assert "华润怡宝" in state.global_context
        assert "competition" in state.global_context

    def test_empty_world_model(self):
        wm = WorldModel(
            actors=[],
            relationships=[],
            timeline=[],
            variables=[],
            question="空模型测试",
        )
        state = init_world_state(wm)
        assert len(state.actors) == 0
        assert state.global_context != ""  # still has question


# ---------------------------------------------------------------------------
# TestApplyAdjudication
# ---------------------------------------------------------------------------


class TestApplyAdjudication:
    """Tests for applying adjudication results to world state."""

    def _make_state(self) -> WorldState:
        return WorldState(
            round_number=0,
            actors={
                "Alice": ActorState(
                    name="Alice", role="leader", description="Test actor"
                ),
                "Bob": ActorState(
                    name="Bob", role="follower", description="Test actor 2"
                ),
            },
            variables={"score": "0"},
            events_log=["init"],
        )

    def test_variable_update(self):
        state = self._make_state()
        adj = AdjudicationResult(
            outcomes=[],
            world_changes=["Score updated"],
            variable_updates={"score": "10"},
            narrative_summary="Score changed",
        )

        apply_adjudication(state, adj, round_number=1)
        assert state.variables["score"] == "10"

    def test_events_logged(self):
        state = self._make_state()
        adj = AdjudicationResult(
            outcomes=[],
            world_changes=["Something happened"],
            variable_updates={},
            narrative_summary="A round happened",
        )

        apply_adjudication(state, adj, round_number=1)
        assert any("Round 1" in e for e in state.events_log)
        assert any("Something happened" in e for e in state.events_log)

    def test_actor_memory_updated(self):
        state = self._make_state()
        adj = AdjudicationResult(
            outcomes=[
                ActionOutcome(
                    actor_name="Alice",
                    action="attack",
                    success=True,
                    outcome="Alice succeeded",
                ),
            ],
            world_changes=[],
            variable_updates={},
            narrative_summary="Alice attacked",
        )

        apply_adjudication(state, adj, round_number=1)
        assert any("Success" in m for m in state.actors["Alice"].memory)

    def test_round_number_updated(self):
        state = self._make_state()
        adj = AdjudicationResult(
            outcomes=[],
            world_changes=[],
            variable_updates={},
            narrative_summary="",
        )

        apply_adjudication(state, adj, round_number=3)
        assert state.round_number == 3


# ---------------------------------------------------------------------------
# TestUpdateActorMemory
# ---------------------------------------------------------------------------


class TestUpdateActorMemory:
    """Tests for updating actor memory with narrator output."""

    def test_memory_added(self):
        state = WorldState(
            actors={
                "Alice": ActorState(
                    name="Alice", role="test", description="test"
                ),
            },
        )
        narrator = NarratorOutput(
            scene="The market heats up.",
            new_developments=["New policy"],
            pressure_points=["Price war"],
        )

        update_actor_memory(state, narrator, round_number=1)
        assert any("market heats up" in m for m in state.actors["Alice"].memory)

    def test_inactive_actors_skipped(self):
        state = WorldState(
            actors={
                "Alice": ActorState(
                    name="Alice", role="test", description="test", status="eliminated"
                ),
            },
        )
        narrator = NarratorOutput(
            scene="Something happens",
            new_developments=[],
            pressure_points=[],
        )

        update_actor_memory(state, narrator, round_number=1)
        assert len(state.actors["Alice"].memory) == 0


# ---------------------------------------------------------------------------
# TestRunSimulation (mock LLM — full integration)
# ---------------------------------------------------------------------------


class TestRunSimulation:
    """End-to-end simulation tests with mock LLM."""

    @pytest.mark.asyncio
    async def test_full_3_round_simulation(
        self, sample_world_model: WorldModel, mock_settings: Settings
    ):
        await _save_test_world_model("doc_001", sample_world_model)

        config = SimulationConfig(
            world_model_id="doc_001",
            question="市场份额变化？",
            rounds=3,
        )

        result = await run_simulation(config, settings=mock_settings)

        assert result.status == SimulationStatus.COMPLETED
        assert len(result.rounds) == 3
        assert result.report is not None
        assert result.report.confidence > 0

    @pytest.mark.asyncio
    async def test_missing_world_model(self, mock_settings: Settings):
        config = SimulationConfig(
            world_model_id="nonexistent",
            question="test?",
            rounds=1,
        )

        result = await run_simulation(config, settings=mock_settings)
        assert result.status == SimulationStatus.FAILED
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_actors_override(
        self, sample_world_model: WorldModel, mock_settings: Settings
    ):
        await _save_test_world_model("doc_002", sample_world_model)

        config = SimulationConfig(
            world_model_id="doc_002",
            question="test?",
            rounds=1,
            actors_override=["农夫山泉"],
        )

        result = await run_simulation(config, settings=mock_settings)
        assert result.status == SimulationStatus.COMPLETED
        # Only one actor should have acted
        assert len(result.rounds[0].actions) == 1
        assert result.rounds[0].actions[0].actor_name == "农夫山泉"

    @pytest.mark.asyncio
    async def test_intervention_injected(
        self, sample_world_model: WorldModel, mock_settings: Settings
    ):
        await _save_test_world_model("doc_003", sample_world_model)

        config = SimulationConfig(
            world_model_id="doc_003",
            question="test?",
            rounds=2,
            interventions=[
                Intervention(
                    round=1,
                    event="政策变化",
                    description="新水质标准出台",
                )
            ],
        )

        result = await run_simulation(config, settings=mock_settings)
        assert result.status == SimulationStatus.COMPLETED
        # Intervention should appear in round 1 narrator output
        r1_narrator = result.rounds[0].narrator
        assert any("INTERVENTION" in d for d in r1_narrator.new_developments)

    @pytest.mark.asyncio
    async def test_world_state_snapshots_independent(
        self, sample_world_model: WorldModel, mock_settings: Settings
    ):
        """Verify that each round's snapshot is an independent copy."""
        await _save_test_world_model("doc_004", sample_world_model)

        config = SimulationConfig(
            world_model_id="doc_004",
            question="test?",
            rounds=2,
        )

        result = await run_simulation(config, settings=mock_settings)
        assert len(result.rounds) == 2

        # Snapshots should have different round numbers
        assert result.rounds[0].world_state_snapshot.round_number == 1
        assert result.rounds[1].world_state_snapshot.round_number == 2

        # Mutating one snapshot should not affect another
        result.rounds[0].world_state_snapshot.variables["test"] = "mutated"
        assert "test" not in result.rounds[1].world_state_snapshot.variables


# ---------------------------------------------------------------------------
# TestAskSimulation
# ---------------------------------------------------------------------------


class TestAskSimulation:
    """Tests for follow-up questions on completed simulations."""

    @pytest.mark.asyncio
    async def test_ask_completed(
        self, sample_world_model: WorldModel, mock_settings: Settings
    ):
        await _save_test_world_model("doc_ask", sample_world_model)

        config = SimulationConfig(
            world_model_id="doc_ask",
            question="test?",
            rounds=1,
        )
        result = await run_simulation(config, settings=mock_settings)

        answer = await ask_simulation(
            result.simulation_id,
            "具体哪些因素最关键？",
            settings=mock_settings,
        )
        assert answer.answer != ""
        assert 0 <= answer.confidence <= 1

    @pytest.mark.asyncio
    async def test_ask_nonexistent(self, mock_settings: Settings):
        with pytest.raises(ValueError, match="not found"):
            await ask_simulation("fake_id", "test?", settings=mock_settings)


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for Pydantic model validation."""

    def test_simulation_config_defaults(self):
        config = SimulationConfig(
            world_model_id="test",
            question="test?",
        )
        assert config.rounds == 3
        assert config.actors_override is None
        assert config.interventions is None

    def test_simulation_config_validation(self):
        with pytest.raises(Exception):
            SimulationConfig(
                world_model_id="test",
                question="test?",
                rounds=0,  # must be >= 1
            )

    def test_intervention_model(self):
        iv = Intervention(round=2, event="test", description="test desc")
        assert iv.round == 2

    def test_narrator_output(self):
        n = NarratorOutput(
            scene="test scene",
            new_developments=["dev1"],
            pressure_points=["pp1"],
        )
        assert n.scene == "test scene"

    def test_actor_action(self):
        a = ActorAction(
            actor_name="Test",
            reasoning="because",
            action="do something",
        )
        assert a.actor_name == "Test"
        assert a.target is None
