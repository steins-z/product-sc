"""API routes for the MiroFish P0 pipeline."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile, File

from app.models.document import ChunksResponse, DocumentUploadResponse
from app.models.task import TaskResponse, TaskStatus
from app.models.world_model import (
    Actor,
    ExtractionRequest,
    ExtractionResponse,
    PatchOperation,
    Relationship,
    TimelineEvent,
    Variable,
    WorldModel,
    WorldModelPatch,
    WorldModelUpdate,
)
from app.services.parser import parse_document
from app.services.chunker import chunk_text
from app.services.extractor import extract_world_model
from app import db

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
#  Task 1: Document Upload & Parsing                                           #
# --------------------------------------------------------------------------- #


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    Upload a document (PDF, TXT, or MD) and parse it to plain text.

    Returns the extracted text along with a unique document_id.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    allowed_extensions = {".pdf", ".txt", ".md", ".markdown"}
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = parse_document(file.filename, file_bytes)
    except Exception as e:
        logger.exception("Failed to parse document: %s", file.filename)
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {e}")

    # Store document
    await db.save_document(result)

    # Pre-compute and store chunks
    chunks_resp = chunk_text(result.text, result.document_id)
    await db.save_chunks(chunks_resp)

    logger.info(
        "Uploaded document %s → %d chars, %d chunks",
        result.document_id,
        result.char_count,
        chunks_resp.total_chunks,
    )

    return result


# --------------------------------------------------------------------------- #
#  Task 2: Get Chunks                                                          #
# --------------------------------------------------------------------------- #


@router.get("/documents/{document_id}/chunks", response_model=ChunksResponse)
async def get_chunks(document_id: str) -> ChunksResponse:
    """
    Get all chunks for a previously uploaded document.
    """
    chunks_resp = await db.get_chunks(document_id)
    if not chunks_resp:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return chunks_resp


# --------------------------------------------------------------------------- #
#  Task 3: Extract World Model                                                 #
# --------------------------------------------------------------------------- #


@router.post("/documents/{document_id}/extract", response_model=ExtractionResponse)
async def extract(document_id: str, request: ExtractionRequest) -> ExtractionResponse:
    """
    Extract a structured world model from a document's chunks,
    guided by a prediction question.

    Uses GPT-4o structured output (or mock in dev mode).
    """
    chunks_resp = await db.get_chunks(document_id)
    if not chunks_resp:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    if not chunks_resp.chunks:
        raise HTTPException(status_code=422, detail="Document has no chunks")

    try:
        world_model = await extract_world_model(
            chunks=chunks_resp.chunks,
            question=request.question,
        )
        result = ExtractionResponse(
            document_id=document_id,
            question=request.question,
            world_model=world_model,
            chunks_processed=len(chunks_resp.chunks),
        )
    except Exception as e:
        logger.exception("Extraction failed for document %s", document_id)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    # Store world model (serves both P0 and P1)
    await db.save_world_model(document_id, result)

    return result


# --------------------------------------------------------------------------- #
#  Task 4: One-shot Extract API + Async Task Tracking                          #
# --------------------------------------------------------------------------- #


async def _run_extraction(task_id: str, document_id: str, question: str) -> None:
    """Background task: extract world model, then update task status."""
    task = await db.get_task(task_id)
    if not task:
        return
    try:
        chunks_resp = await db.get_chunks(document_id)
        if not chunks_resp:
            raise ValueError(f"Document '{document_id}' not found")

        world_model = await extract_world_model(
            chunks=chunks_resp.chunks,
            question=question,
        )
        result = ExtractionResponse(
            document_id=document_id,
            question=question,
            world_model=world_model,
            chunks_processed=len(chunks_resp.chunks),
        )
        await db.save_world_model(document_id, result)

        task.status = TaskStatus.COMPLETED
        task.result = result
        await db.save_task(task)
    except Exception as e:
        logger.exception("Background extraction failed for task %s", task_id)
        task.status = TaskStatus.FAILED
        task.error = str(e)
        await db.save_task(task)


@router.post("/extract", response_model=TaskResponse, status_code=202)
async def extract_one_shot(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    question: str = Form(...),
) -> TaskResponse:
    """
    One-shot extraction: upload a document + question, get back a task_id.

    The extraction runs asynchronously. Poll GET /api/tasks/{task_id} for results.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    allowed_extensions = {".pdf", ".txt", ".md", ".markdown"}
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        doc_result = parse_document(file.filename, file_bytes)
    except Exception as e:
        logger.exception("Failed to parse document: %s", file.filename)
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {e}")

    await db.save_document(doc_result)
    chunks_resp = chunk_text(doc_result.text, doc_result.document_id)
    await db.save_chunks(chunks_resp)

    # Create async task
    task_id = uuid.uuid4().hex[:12]
    task = TaskResponse(task_id=task_id, status=TaskStatus.PROCESSING)
    await db.save_task(task)

    background_tasks.add_task(_run_extraction, task_id, doc_result.document_id, question)

    logger.info("Created extraction task %s for document %s", task_id, doc_result.document_id)
    return task


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """Query the status and result of an async extraction task."""
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


# --------------------------------------------------------------------------- #
#  Task 5: World Model Confirm / Edit API                                      #
# --------------------------------------------------------------------------- #

# Field name → Pydantic model class mapping
_FIELD_MODEL_MAP: dict[str, type] = {
    "actors": Actor,
    "relationships": Relationship,
    "timeline": TimelineEvent,
    "variables": Variable,
}


@router.get("/world-model/{document_id}", response_model=ExtractionResponse)
async def get_world_model(document_id: str) -> ExtractionResponse:
    """Get the current world model for a document."""
    result = await db.get_world_model_extraction(document_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"World model for '{document_id}' not found")
    return result


@router.put("/world-model/{document_id}", response_model=ExtractionResponse)
async def replace_world_model(
    document_id: str, update: WorldModelUpdate
) -> ExtractionResponse:
    """
    Full replacement of a world model's content.

    Replaces actors, relationships, timeline, and variables entirely.
    """
    existing = await db.get_world_model_extraction(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"World model for '{document_id}' not found")

    existing.world_model = WorldModel(
        actors=update.actors,
        relationships=update.relationships,
        timeline=update.timeline,
        variables=update.variables,
        question=existing.world_model.question,
    )
    await db.save_world_model(document_id, existing)
    return existing


@router.patch("/world-model/{document_id}", response_model=ExtractionResponse)
async def patch_world_model(
    document_id: str, patch: WorldModelPatch
) -> ExtractionResponse:
    """
    Incremental update of a world model.

    Supports operations:
    - add: Append an item to a field (actors/relationships/timeline/variables)
    - remove: Remove an item by index from a field
    - replace: Replace an item at index in a field
    """
    existing = await db.get_world_model_extraction(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"World model for '{document_id}' not found")

    wm = existing.world_model

    for op in patch.operations:
        if op.path not in _FIELD_MODEL_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid path '{op.path}'. Must be one of: {list(_FIELD_MODEL_MAP.keys())}",
            )

        items: list = getattr(wm, op.path)
        model_cls = _FIELD_MODEL_MAP[op.path]

        if op.op == "add":
            if op.value is None:
                raise HTTPException(status_code=400, detail="'add' operation requires 'value'")
            items.append(model_cls(**op.value))

        elif op.op == "remove":
            if op.index is None:
                raise HTTPException(status_code=400, detail="'remove' operation requires 'index'")
            if op.index < 0 or op.index >= len(items):
                raise HTTPException(
                    status_code=400,
                    detail=f"Index {op.index} out of range for '{op.path}' (length {len(items)})",
                )
            items.pop(op.index)

        elif op.op == "replace":
            if op.index is None or op.value is None:
                raise HTTPException(
                    status_code=400,
                    detail="'replace' operation requires both 'index' and 'value'",
                )
            if op.index < 0 or op.index >= len(items):
                raise HTTPException(
                    status_code=400,
                    detail=f"Index {op.index} out of range for '{op.path}' (length {len(items)})",
                )
            items[op.index] = model_cls(**op.value)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown operation '{op.op}'. Must be: add, remove, replace",
            )

    await db.save_world_model(document_id, existing)
    return existing
