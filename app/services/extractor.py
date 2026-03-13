"""
LLM-based world-model extraction service.

Supports two providers:
- OpenAI (GPT-4o) — uses strict structured output (response_format with JSON schema)
- Qwen (DashScope) — uses JSON mode + schema in prompt (OpenAI-compatible SDK)

Both go through the `openai` Python SDK.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.models.document import Chunk
from app.models.world_model import WorldModel
from app.prompts.extraction import (
    SYSTEM_PROMPT as EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON Schema for OpenAI structured output
# ---------------------------------------------------------------------------

WORLD_MODEL_JSON_SCHEMA = {
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
                    "source_ref": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "role", "description", "source_ref"],
                "additionalProperties": False
            }
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
                    "source_ref": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["from", "to", "type", "description", "source_ref"],
                "additionalProperties": False
            }
        },
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "event": {"type": "string"},
                    "description": {"type": "string"},
                    "source_ref": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["date", "event", "description", "source_ref"],
                "additionalProperties": False
            }
        },
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "current_value": {"type": "string"},
                    "description": {"type": "string"},
                    "source_ref": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "current_value", "description", "source_ref"],
                "additionalProperties": False
            }
        },
        "question": {"type": "string"}
    },
    "required": ["actors", "relationships", "timeline", "variables", "question"],
    "additionalProperties": False
}

# ---------------------------------------------------------------------------
# Mock response for testing without LLM
# ---------------------------------------------------------------------------

_MOCK_RESPONSE: dict[str, Any] = {
    "actors": [
        {
            "name": "农夫山泉",
            "role": "market_leader",
            "description": "中国最大的包装饮用水企业，同时拥有东方树叶等茶饮品牌",
            "source_ref": ["mock_chunk_0"],
        }
    ],
    "relationships": [
        {
            "from": "农夫山泉",
            "to": "华润怡宝",
            "type": "competition",
            "description": "天然水vs纯净水之争，直接竞争对手",
            "source_ref": ["mock_chunk_0"],
        }
    ],
    "timeline": [
        {
            "date": "2024-02",
            "event": "绿瓶纯净水攻势",
            "description": "农夫山泉推出1元绿瓶纯净水，发动价格战",
            "source_ref": ["mock_chunk_0"],
        }
    ],
    "variables": [
        {
            "name": "消费升级vs降级趋势",
            "current_value": "分化中",
            "description": "宏观经济放缓影响高端水，但健康意识提升利好无糖茶",
            "source_ref": ["mock_chunk_0"],
        }
    ],
    "question": "未来12个月内农夫山泉市场份额会如何变化？",
}


def _build_schema_instruction() -> str:
    """Build a human-readable schema instruction to embed in the prompt for JSON mode."""
    return """
你必须严格按照以下 JSON Schema 输出，不要添加任何额外字段：

```json
{
  "actors": [
    {
      "name": "string (角色/实体名称，使用规范名称)",
      "role": "string (角色定位)",
      "description": "string (详细描述)",
      "source_ref": ["string (chunk_id列表)"]
    }
  ],
  "relationships": [
    {
      "from": "string (关系起点角色名)",
      "to": "string (关系终点角色名)",
      "type": "string (关系类型)",
      "description": "string (关系描述)",
      "source_ref": ["string (chunk_id列表)"]
    }
  ],
  "timeline": [
    {
      "date": "string (ISO日期或描述性日期如 Q3 2025)",
      "event": "string (事件名称)",
      "description": "string (事件详情)",
      "source_ref": ["string (chunk_id列表)"]
    }
  ],
  "variables": [
    {
      "name": "string (变量名称)",
      "current_value": "string (当前值)",
      "description": "string (变量描述)",
      "source_ref": ["string (chunk_id列表)"]
    }
  ],
  "question": "string (原始预测问题，原样复制)"
}
```

只输出合法 JSON，不要输出任何解释文字或 markdown 代码块标记。
"""


async def extract_world_model(
    chunks: list[Chunk],
    question: str,
    settings: Settings | None = None,
) -> WorldModel:
    """
    Extract a structured world model from document chunks.

    Args:
        chunks: list of Chunk objects
        question: the prediction question
        settings: app settings (auto-loaded if None)

    Returns:
        Validated WorldModel instance
    """
    if settings is None:
        settings = Settings()

    # ---- Mock mode ----
    if settings.use_mock_llm:
        logger.info("Using mock LLM response (USE_MOCK_LLM=true)")
        mock = _MOCK_RESPONSE.copy()
        mock["question"] = question
        # Replace mock chunk_ids with real ones if available
        real_ids = [c.chunk_id for c in chunks]
        if real_ids:
            for category in ("actors", "relationships", "timeline", "variables"):
                for item in mock[category]:
                    item["source_ref"] = [real_ids[0]]
        return WorldModel.model_validate(mock)

    # ---- Real LLM call ----
    api_key = settings.get_api_key()
    if not api_key:
        raise ValueError(
            "No API key configured. Set LLM_API_KEY (or OPENAI_API_KEY) in .env"
        )

    base_url = settings.get_base_url()
    model = settings.get_model()

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Convert Chunk objects to dict format for prompt building
    chunks_dict = [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks]
    # build_extraction_prompt returns (system_prompt, user_prompt)
    _, user_prompt = build_extraction_prompt(question, chunks_dict)

    logger.info(
        "Calling LLM: provider=%s model=%s chunks=%d",
        settings.llm_provider,
        model,
        len(chunks),
    )

    if settings.llm_provider == "openai":
        # OpenAI: use strict structured output
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "world_model",
                    "strict": True,
                    "schema": WORLD_MODEL_JSON_SCHEMA,
                },
            },
            temperature=0.1,
        )
    else:
        # Qwen / other OpenAI-compatible: use JSON mode + schema in prompt
        system_with_schema = (
            EXTRACTION_SYSTEM_PROMPT + "\n\n" + _build_schema_instruction()
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_with_schema},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

    raw = response.choices[0].message.content
    logger.info("LLM response received: %d chars", len(raw) if raw else 0)

    if not raw:
        raise ValueError("LLM returned empty response")

    # Strip markdown code fences if present (some models wrap JSON in ```json...```)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove first line (```json) and last line (```)
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    data = json.loads(cleaned)
    return WorldModel.model_validate(data)
