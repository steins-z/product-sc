"""Extraction prompt templates for the world model pipeline."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a world-model extraction engine for a prediction system called MiroFish.

Your task: given document chunks and a prediction question, extract a structured
"world model" that captures the key elements needed to reason about the question.

## What to Extract

### Actors
People, organizations, government bodies, companies — any entity that can *act*
and influence the outcome of the prediction question. Include:
- Their name (canonical form)
- Their role in the context
- A concise but informative description of their relevance

### Relationships
Connections between actors: competition, cooperation, regulation, supply chains,
political alliances, ownership, etc. Be specific about the *type* of relationship.

### Timeline
Key events with dates (exact or approximate). Include past events that set context
and any announced future events. Use ISO dates when possible (YYYY-MM-DD),
otherwise use descriptive dates ("Q3 2025", "early 2024").

### Variables
Quantitative or qualitative factors that could shift the prediction outcome:
market share, revenue, policy changes, consumer sentiment, regulatory decisions,
pricing, capacity, etc. Record their *current* known value.

## Rules
1. Every extracted item MUST include `source_ref` — a list of chunk_ids from which
   the information was derived. Never fabricate references.
2. Be thorough: extract ALL relevant actors, relationships, events, and variables.
   Err on the side of inclusion.
3. Prefer specificity over vagueness. "Revenue grew 15% YoY" > "Revenue increased".
4. If the same entity appears in multiple chunks, merge into one actor entry
   and list all relevant chunk_ids in source_ref.
5. The `question` field in output must be the exact prediction question provided.
6. For relationships, use the actor *names* (not descriptions) in `from` and `to`.
"""

USER_PROMPT_TEMPLATE = """\
## Prediction Question
{question}

## Document Chunks
{chunks_text}

---

Extract the world model from the above chunks, focusing on elements relevant to
the prediction question. Return ONLY the JSON object matching the required schema.
"""


def format_chunks_for_prompt(chunks: list[dict[str, str]]) -> str:
    """Format a list of chunks into the prompt text block.

    Each chunk dict should have 'chunk_id' and 'text' keys.
    """
    parts: list[str] = []
    for chunk in chunks:
        parts.append(f"### [{chunk['chunk_id']}]\n{chunk['text']}")
    return "\n\n".join(parts)


def build_extraction_prompt(
    question: str,
    chunks: list[dict[str, str]],
) -> tuple[str, str]:
    """Build the system and user prompts for extraction.

    Returns (system_prompt, user_prompt).
    """
    chunks_text = format_chunks_for_prompt(chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        chunks_text=chunks_text,
    )
    return SYSTEM_PROMPT, user_prompt
