"""Task 3: LLM Extraction — extracts a structured world model from document chunks."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.config import settings
from app.models.document import Chunk
from app.models.world_model import (
    Actor,
    ExtractionResponse,
    Relationship,
    TimelineEvent,
    Variable,
    WorldModel,
)
from app.prompts.extraction import build_extraction_prompt

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  JSON Schema for OpenAI structured output (response_format)                  #
# --------------------------------------------------------------------------- #

WORLD_MODEL_JSON_SCHEMA: dict[str, Any] = {
    "name": "world_model",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "actors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "description": {"type": "string"},
                        "source_ref": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "role", "description", "source_ref"],
                    "additionalProperties": False,
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                        "source_ref": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["from", "to", "type", "description", "source_ref"],
                    "additionalProperties": False,
                },
            },
            "timeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "event": {"type": "string"},
                        "description": {"type": "string"},
                        "source_ref": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["date", "event", "description", "source_ref"],
                    "additionalProperties": False,
                },
            },
            "variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "current_value": {"type": "string"},
                        "description": {"type": "string"},
                        "source_ref": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "current_value", "description", "source_ref"],
                    "additionalProperties": False,
                },
            },
            "question": {"type": "string"},
        },
        "required": ["actors", "relationships", "timeline", "variables", "question"],
        "additionalProperties": False,
    },
}


# --------------------------------------------------------------------------- #
#  Mock LLM — returns realistic fake data for testing                          #
# --------------------------------------------------------------------------- #


def _mock_extract(question: str, chunks: list[Chunk]) -> dict[str, Any]:
    """Generate a mock world model from chunks (no API call).

    Produces realistic-looking output using chunk_ids from the input.
    """
    chunk_ids = [c.chunk_id for c in chunks]

    # Distribute chunk_ids across mock entities
    def _refs(indices: list[int]) -> list[str]:
        return [chunk_ids[i] for i in indices if i < len(chunk_ids)]

    return {
        "actors": [
            {
                "name": "农夫山泉 (Nongfu Spring)",
                "role": "Leading bottled water manufacturer in China",
                "description": (
                    "Founded by Zhong Shanshan, Nongfu Spring is China's largest "
                    "bottled water company by market share, known for its natural "
                    "water sources and diversified beverage portfolio."
                ),
                "source_ref": _refs([0, 1]),
            },
            {
                "name": "钟睒睒 (Zhong Shanshan)",
                "role": "Founder and Chairman of Nongfu Spring",
                "description": (
                    "Billionaire entrepreneur, previously richest person in Asia. "
                    "Known for secretive management style and long-term brand building."
                ),
                "source_ref": _refs([0]),
            },
            {
                "name": "怡宝 (C'estbon / China Resources Beverage)",
                "role": "Major competitor — purified water market leader",
                "description": (
                    "Subsidiary of China Resources, the largest purified water brand "
                    "in China. Recently IPO'd in Hong Kong."
                ),
                "source_ref": _refs([1, 2] if len(chunk_ids) > 2 else [1]),
            },
            {
                "name": "百岁山 (Ganten)",
                "role": "Premium natural mineral water competitor",
                "description": (
                    "Positioned as a premium alternative, Ganten competes in the "
                    "natural mineral water segment with growing market share."
                ),
                "source_ref": _refs([2] if len(chunk_ids) > 2 else [1]),
            },
            {
                "name": "元气森林 (Genki Forest)",
                "role": "Disruptive beverage brand — sugar-free segment",
                "description": (
                    "Fast-growing challenger brand targeting health-conscious younger "
                    "consumers with sugar-free sparkling water and teas."
                ),
                "source_ref": _refs([2, 3] if len(chunk_ids) > 3 else [1]),
            },
        ],
        "relationships": [
            {
                "from": "农夫山泉 (Nongfu Spring)",
                "to": "怡宝 (C'estbon / China Resources Beverage)",
                "type": "competes_with",
                "description": (
                    "Direct competition in China's bottled water market. Nongfu leads "
                    "in natural water, C'estbon leads in purified water."
                ),
                "source_ref": _refs([1, 2] if len(chunk_ids) > 2 else [0, 1]),
            },
            {
                "from": "农夫山泉 (Nongfu Spring)",
                "to": "元气森林 (Genki Forest)",
                "type": "competes_with",
                "description": (
                    "Competition in the flavored/functional beverage segment, especially "
                    "sugar-free tea and sparkling water."
                ),
                "source_ref": _refs([2, 3] if len(chunk_ids) > 3 else [1]),
            },
            {
                "from": "钟睒睒 (Zhong Shanshan)",
                "to": "农夫山泉 (Nongfu Spring)",
                "type": "controls",
                "description": (
                    "Zhong holds ~84% of Nongfu Spring's shares and maintains "
                    "tight control over company strategy."
                ),
                "source_ref": _refs([0]),
            },
        ],
        "timeline": [
            {
                "date": "1996",
                "event": "Nongfu Spring founded",
                "description": "Company established in Hangzhou, Zhejiang province.",
                "source_ref": _refs([0]),
            },
            {
                "date": "2020-09-08",
                "event": "Nongfu Spring IPO on HKEX",
                "description": (
                    "Listed on Hong Kong Stock Exchange, making Zhong Shanshan "
                    "briefly the richest person in Asia."
                ),
                "source_ref": _refs([0, 1] if len(chunk_ids) > 1 else [0]),
            },
            {
                "date": "2024",
                "event": "Purified water controversy",
                "description": (
                    "Online backlash over natural vs purified water debate, "
                    "affecting brand sentiment temporarily."
                ),
                "source_ref": _refs([2, 3] if len(chunk_ids) > 3 else [1]),
            },
        ],
        "variables": [
            {
                "name": "Nongfu Spring market share (bottled water)",
                "current_value": "~26.4%",
                "description": (
                    "Nongfu Spring's share of China's packaged drinking water market, "
                    "maintaining the #1 position for over a decade."
                ),
                "source_ref": _refs([1]),
            },
            {
                "name": "China bottled water market size",
                "current_value": "~230 billion RMB (2024)",
                "description": (
                    "Total addressable market for packaged drinking water in China, "
                    "growing at ~8% CAGR."
                ),
                "source_ref": _refs([1, 2] if len(chunk_ids) > 2 else [1]),
            },
            {
                "name": "Consumer health consciousness trend",
                "current_value": "Rising",
                "description": (
                    "Growing consumer preference for natural/mineral water over purified, "
                    "and sugar-free beverages. Benefits Nongfu's positioning."
                ),
                "source_ref": _refs([2, 3] if len(chunk_ids) > 3 else [1]),
            },
            {
                "name": "Raw material + logistics costs",
                "current_value": "Moderately increasing",
                "description": (
                    "PET plastic and transportation costs affect margins across the "
                    "industry. Nongfu's remote water sources increase logistics cost."
                ),
                "source_ref": _refs([3] if len(chunk_ids) > 3 else [1]),
            },
        ],
        "question": question,
    }


# --------------------------------------------------------------------------- #
#  Real LLM extraction                                                         #
# --------------------------------------------------------------------------- #


def _llm_extract(question: str, chunks: list[Chunk]) -> dict[str, Any]:
    """Call GPT-4o with structured output to extract a world model."""
    client = OpenAI(api_key=settings.openai_api_key)

    chunk_dicts = [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks]
    system_prompt, user_prompt = build_extraction_prompt(question, chunk_dicts)

    logger.info(
        "Calling %s with %d chunks (%d chars in prompt)",
        settings.openai_model,
        len(chunks),
        len(user_prompt),
    )

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": WORLD_MODEL_JSON_SCHEMA,
        },
        temperature=0.1,  # Low temp for deterministic extraction
    )

    raw = response.choices[0].message.content
    assert raw is not None, "LLM returned empty content"
    return json.loads(raw)


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #


def extract_world_model(
    document_id: str,
    question: str,
    chunks: list[Chunk],
    use_mock: bool | None = None,
) -> ExtractionResponse:
    """
    Extract a world model from document chunks given a prediction question.

    Args:
        document_id: The document these chunks belong to.
        question: The prediction question guiding extraction.
        chunks: List of Chunk objects to process.
        use_mock: Override settings.use_mock_llm if provided.

    Returns:
        ExtractionResponse with the full world model.
    """
    mock = use_mock if use_mock is not None else settings.use_mock_llm

    if mock:
        logger.info("Using MOCK LLM for extraction (document=%s)", document_id)
        raw = _mock_extract(question, chunks)
    else:
        raw = _llm_extract(question, chunks)

    # Parse into Pydantic models
    world_model = WorldModel(
        actors=[Actor(**a) for a in raw["actors"]],
        relationships=[Relationship(**r) for r in raw["relationships"]],
        timeline=[TimelineEvent(**t) for t in raw["timeline"]],
        variables=[Variable(**v) for v in raw["variables"]],
        question=raw["question"],
    )

    return ExtractionResponse(
        document_id=document_id,
        question=question,
        world_model=world_model,
        chunks_processed=len(chunks),
    )
