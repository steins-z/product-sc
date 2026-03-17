"""SQLite persistence layer for MiroFish (P2).

All DB operations are async via aiosqlite.
Complex nested Pydantic models are stored as JSON text columns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

from app.models.document import Chunk, ChunksResponse, DocumentUploadResponse
from app.models.simulation import SimulationConfig, SimulationResult, SimulationStatus
from app.models.task import TaskResponse, TaskStatus
from app.models.world_model import ExtractionResponse, WorldModel

logger = logging.getLogger(__name__)

# Module-level connection reference (set during init)
_db_path: str = ""

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS documents (
    document_id  TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    content_type TEXT NOT NULL,
    text         TEXT NOT NULL,
    char_count   INTEGER NOT NULL,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

CREATE TABLE IF NOT EXISTS tasks (
    task_id    TEXT PRIMARY KEY,
    status     TEXT NOT NULL DEFAULT 'processing',
    error      TEXT,
    result_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS world_models (
    document_id      TEXT PRIMARY KEY,
    question         TEXT NOT NULL,
    world_model_json TEXT NOT NULL,
    chunks_processed INTEGER NOT NULL,
    created_at       TEXT DEFAULT (datetime('now')),
    updated_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS simulations (
    simulation_id TEXT PRIMARY KEY,
    status        TEXT NOT NULL DEFAULT 'pending',
    config_json   TEXT NOT NULL,
    rounds_json   TEXT,
    report_json   TEXT,
    error         TEXT,
    current_round INTEGER DEFAULT 0,
    total_rounds  INTEGER NOT NULL,
    created_at    TEXT DEFAULT (datetime('now')),
    updated_at    TEXT DEFAULT (datetime('now'))
);
"""


def _get_db_path() -> str:
    """Return the configured DB path (set during init)."""
    return _db_path


async def init_db(db_path: str) -> None:
    """Initialise the database: create tables if they don't exist."""
    global _db_path
    _db_path = db_path

    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(_SCHEMA)
        await conn.commit()

    logger.info("Database initialised at %s", db_path)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


async def save_document(doc: DocumentUploadResponse) -> None:
    async with aiosqlite.connect(_db_path) as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO documents (document_id, filename, content_type, text, char_count) VALUES (?, ?, ?, ?, ?)",
            (doc.document_id, doc.filename, doc.content_type, doc.text, doc.char_count),
        )
        await conn.commit()


async def get_document(document_id: str) -> DocumentUploadResponse | None:
    async with aiosqlite.connect(_db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return DocumentUploadResponse(
                document_id=row["document_id"],
                filename=row["filename"],
                content_type=row["content_type"],
                text=row["text"],
                char_count=row["char_count"],
            )


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


async def save_chunks(chunks_resp: ChunksResponse) -> None:
    async with aiosqlite.connect(_db_path) as conn:
        # Delete old chunks for this document first
        await conn.execute(
            "DELETE FROM chunks WHERE document_id = ?", (chunks_resp.document_id,)
        )
        for chunk in chunks_resp.chunks:
            await conn.execute(
                "INSERT INTO chunks (chunk_id, document_id, idx, text, token_count) VALUES (?, ?, ?, ?, ?)",
                (chunk.chunk_id, chunks_resp.document_id, chunk.index, chunk.text, chunk.token_count),
            )
        await conn.commit()


async def get_chunks(document_id: str) -> ChunksResponse | None:
    async with aiosqlite.connect(_db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY idx", (document_id,)
        ) as cur:
            rows = await cur.fetchall()
            if not rows:
                return None
            chunks = [
                Chunk(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    index=r["idx"],
                    text=r["text"],
                    token_count=r["token_count"],
                )
                for r in rows
            ]
            return ChunksResponse(
                document_id=document_id,
                total_chunks=len(chunks),
                chunks=chunks,
            )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


async def save_task(task: TaskResponse) -> None:
    result_json = task.result.model_dump_json() if task.result else None
    async with aiosqlite.connect(_db_path) as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO tasks (task_id, status, error, result_json) VALUES (?, ?, ?, ?)",
            (task.task_id, task.status.value, task.error, result_json),
        )
        await conn.commit()


async def get_task(task_id: str) -> TaskResponse | None:
    async with aiosqlite.connect(_db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            result = None
            if row["result_json"]:
                result = ExtractionResponse.model_validate_json(row["result_json"])
            return TaskResponse(
                task_id=row["task_id"],
                status=TaskStatus(row["status"]),
                error=row["error"],
                result=result,
            )


# ---------------------------------------------------------------------------
# World Models
# ---------------------------------------------------------------------------


async def save_world_model(document_id: str, extraction: ExtractionResponse) -> None:
    """Save an ExtractionResponse. Both P0 routes and P1 simulation read from this."""
    async with aiosqlite.connect(_db_path) as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO world_models (document_id, question, world_model_json, chunks_processed, updated_at) VALUES (?, ?, ?, ?, datetime('now'))",
            (
                document_id,
                extraction.question,
                extraction.world_model.model_dump_json(),
                extraction.chunks_processed,
            ),
        )
        await conn.commit()


async def get_world_model_extraction(document_id: str) -> ExtractionResponse | None:
    """Get the full ExtractionResponse (used by P0 routes)."""
    async with aiosqlite.connect(_db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM world_models WHERE document_id = ?", (document_id,)
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            wm = WorldModel.model_validate_json(row["world_model_json"])
            return ExtractionResponse(
                document_id=row["document_id"],
                question=row["question"],
                world_model=wm,
                chunks_processed=row["chunks_processed"],
            )


async def get_world_model(document_id: str) -> WorldModel | None:
    """Get just the WorldModel (used by P1 simulation engine)."""
    async with aiosqlite.connect(_db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT world_model_json FROM world_models WHERE document_id = ?",
            (document_id,),
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return WorldModel.model_validate_json(row["world_model_json"])


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------


async def save_simulation(sim: SimulationResult) -> None:
    rounds_json = json.dumps(
        [r.model_dump(mode="json") for r in sim.rounds]
    ) if sim.rounds else None
    report_json = sim.report.model_dump_json() if sim.report else None
    config_json = sim.config.model_dump_json()

    async with aiosqlite.connect(_db_path) as conn:
        await conn.execute(
            """INSERT OR REPLACE INTO simulations
               (simulation_id, status, config_json, rounds_json, report_json,
                error, current_round, total_rounds, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                sim.simulation_id,
                sim.status.value,
                config_json,
                rounds_json,
                report_json,
                sim.error,
                sim.current_round,
                sim.total_rounds,
            ),
        )
        await conn.commit()


async def get_simulation(simulation_id: str) -> SimulationResult | None:
    async with aiosqlite.connect(_db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM simulations WHERE simulation_id = ?", (simulation_id,)
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return _row_to_simulation(row)


async def list_simulations() -> list[SimulationResult]:
    async with aiosqlite.connect(_db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM simulations ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
            return [_row_to_simulation(r) for r in rows]


def _row_to_simulation(row: aiosqlite.Row) -> SimulationResult:
    from app.models.simulation import (
        PredictionReport,
        RoundResult,
    )

    config = SimulationConfig.model_validate_json(row["config_json"])
    rounds = []
    if row["rounds_json"]:
        rounds_data = json.loads(row["rounds_json"])
        rounds = [RoundResult.model_validate(r) for r in rounds_data]
    report = None
    if row["report_json"]:
        report = PredictionReport.model_validate_json(row["report_json"])

    return SimulationResult(
        simulation_id=row["simulation_id"],
        status=SimulationStatus(row["status"]),
        config=config,
        rounds=rounds,
        report=report,
        error=row["error"],
        current_round=row["current_round"],
        total_rounds=row["total_rounds"],
    )
