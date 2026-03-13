"""Pydantic models for documents and chunks."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned after uploading and parsing a document."""

    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="Detected content type (pdf/txt/md)")
    text: str = Field(..., description="Extracted plain text")
    char_count: int = Field(..., description="Number of characters in extracted text")


class Chunk(BaseModel):
    """A single text chunk with a unique identifier."""

    chunk_id: str = Field(
        ...,
        description="Unique chunk ID in format {document_id}_chunk_{index}",
    )
    document_id: str = Field(..., description="Parent document ID")
    index: int = Field(..., description="0-based chunk index")
    text: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., description="Approximate token count")


class ChunksResponse(BaseModel):
    """Response containing all chunks for a document."""

    document_id: str
    total_chunks: int
    chunks: list[Chunk]
