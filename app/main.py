"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="MiroFish",
    description="Multi-agent simulation prediction engine — P0: Document → World Model pipeline",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
