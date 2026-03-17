"""Unit tests for the SQLite persistence layer (app/db.py)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from app import db
from app.models.document import Chunk, ChunksResponse, DocumentUploadResponse
from app.models.task import TaskResponse, TaskStatus
from app.models.world_model import (
    Actor,
    ExtractionResponse,
    Relationship,
    TimelineEvent,
    Variable,
    WorldModel,
)
from app.models.simulation import (
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
)


@pytest_asyncio.fixture(autouse=True)
async def _init_test_db(tmp_path):
    """Initialise a fresh test DB for each test."""
    db_path = str(tmp_path / "test.db")
    await db.init_db(db_path)
    yield


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_document():
    doc = DocumentUploadResponse(
        document_id="doc_001",
        filename="test.md",
        content_type="md",
        text="Hello world",
        char_count=11,
    )
    await db.save_document(doc)

    loaded = await db.get_document("doc_001")
    assert loaded is not None
    assert loaded.document_id == "doc_001"
    assert loaded.text == "Hello world"


@pytest.mark.asyncio
async def test_get_document_not_found():
    result = await db.get_document("nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_chunks():
    chunks_resp = ChunksResponse(
        document_id="doc_001",
        total_chunks=2,
        chunks=[
            Chunk(chunk_id="doc_001_chunk_0", document_id="doc_001", index=0, text="chunk 0", token_count=2),
            Chunk(chunk_id="doc_001_chunk_1", document_id="doc_001", index=1, text="chunk 1", token_count=2),
        ],
    )
    await db.save_chunks(chunks_resp)

    loaded = await db.get_chunks("doc_001")
    assert loaded is not None
    assert loaded.total_chunks == 2
    assert loaded.chunks[0].chunk_id == "doc_001_chunk_0"
    assert loaded.chunks[1].text == "chunk 1"


@pytest.mark.asyncio
async def test_get_chunks_not_found():
    result = await db.get_chunks("nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_task():
    task = TaskResponse(task_id="task_001", status=TaskStatus.PROCESSING)
    await db.save_task(task)

    loaded = await db.get_task("task_001")
    assert loaded is not None
    assert loaded.status == TaskStatus.PROCESSING
    assert loaded.result is None


@pytest.mark.asyncio
async def test_task_update_status():
    task = TaskResponse(task_id="task_002", status=TaskStatus.PROCESSING)
    await db.save_task(task)

    task.status = TaskStatus.COMPLETED
    await db.save_task(task)

    loaded = await db.get_task("task_002")
    assert loaded is not None
    assert loaded.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# World Models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_world_model():
    wm = WorldModel(
        actors=[Actor(name="A", role="r", description="d", source_ref=["c1"])],
        relationships=[],
        timeline=[],
        variables=[],
        question="test question?",
    )
    extraction = ExtractionResponse(
        document_id="doc_wm",
        question="test question?",
        world_model=wm,
        chunks_processed=1,
    )
    await db.save_world_model("doc_wm", extraction)

    # Full extraction
    loaded = await db.get_world_model_extraction("doc_wm")
    assert loaded is not None
    assert loaded.question == "test question?"
    assert len(loaded.world_model.actors) == 1
    assert loaded.world_model.actors[0].name == "A"

    # Just WorldModel
    wm_only = await db.get_world_model("doc_wm")
    assert wm_only is not None
    assert wm_only.question == "test question?"


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_simulation():
    config = SimulationConfig(
        world_model_id="doc_001",
        question="Will X happen?",
        rounds=3,
    )
    sim = SimulationResult(
        simulation_id="sim_001",
        status=SimulationStatus.PENDING,
        config=config,
        total_rounds=3,
    )
    await db.save_simulation(sim)

    loaded = await db.get_simulation("sim_001")
    assert loaded is not None
    assert loaded.status == SimulationStatus.PENDING
    assert loaded.config.question == "Will X happen?"


@pytest.mark.asyncio
async def test_list_simulations():
    config = SimulationConfig(
        world_model_id="doc_001",
        question="Q?",
        rounds=1,
    )
    for i in range(3):
        sim = SimulationResult(
            simulation_id=f"sim_{i:03d}",
            status=SimulationStatus.PENDING,
            config=config,
            total_rounds=1,
        )
        await db.save_simulation(sim)

    all_sims = await db.list_simulations()
    assert len(all_sims) == 3


@pytest.mark.asyncio
async def test_get_simulation_not_found():
    result = await db.get_simulation("nonexistent")
    assert result is None
