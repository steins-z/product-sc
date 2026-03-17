"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import router
from app.api.simulation_routes import router as sim_router
from app.config import settings
from app.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise SQLite database
    db_dir = Path(__file__).resolve().parent.parent / "data"
    db_path = str(db_dir / settings.db_filename)
    await init_db(db_path)
    yield


app = FastAPI(
    title="MiroFish",
    description="Multi-agent simulation prediction engine — P0: Document → World Model pipeline",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")
app.include_router(sim_router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
