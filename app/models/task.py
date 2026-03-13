"""Pydantic models for async task tracking."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.world_model import ExtractionResponse


class TaskStatus(str, Enum):
    """Status of an async extraction task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResponse(BaseModel):
    """Response for GET /api/tasks/{id}."""

    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    error: str | None = Field(None, description="Error message if failed")
    result: ExtractionResponse | None = Field(None, description="Extraction result if completed")
