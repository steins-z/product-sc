"""API routes for the MiroFish P0 pipeline."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.models.document import ChunksResponse, DocumentUploadResponse
from app.models.world_model import ExtractionRequest, ExtractionResponse
from app.services.parser import parse_document
from app.services.chunker import chunk_text
from app.services.extractor import extract_world_model

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory store (sufficient for P0 — swap for DB later)
_documents: dict[str, DocumentUploadResponse] = {}
_chunks: dict[str, ChunksResponse] = {}


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

    # Store for later retrieval
    _documents[result.document_id] = result

    # Also pre-compute chunks
    chunks_resp = chunk_text(result.text, result.document_id)
    _chunks[result.document_id] = chunks_resp

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
    if document_id not in _chunks:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return _chunks[document_id]


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
    if document_id not in _chunks:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    chunks_resp = _chunks[document_id]
    if not chunks_resp.chunks:
        raise HTTPException(status_code=422, detail="Document has no chunks")

    try:
        result = extract_world_model(
            document_id=document_id,
            question=request.question,
            chunks=chunks_resp.chunks,
        )
    except Exception as e:
        logger.exception("Extraction failed for document %s", document_id)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    return result
