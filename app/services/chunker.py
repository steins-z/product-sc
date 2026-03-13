"""Task 2: Text Chunker — splits text into overlapping chunks with unique IDs."""

from __future__ import annotations

import tiktoken

from app.config import settings
from app.models.document import Chunk, ChunksResponse


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Count tokens in a text string."""
    return len(encoding.encode(text))


def chunk_text(
    text: str,
    document_id: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> ChunksResponse:
    """
    Split text into chunks of approximately `max_tokens` tokens each,
    with `overlap_tokens` overlap between consecutive chunks.

    Chunks are split on paragraph/sentence boundaries when possible
    to avoid cutting mid-sentence.

    Each chunk gets a unique chunk_id: {document_id}_chunk_{index}
    """
    max_tokens = max_tokens or settings.chunk_max_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    encoding = tiktoken.encoding_for_model("gpt-4o")

    # Split into paragraphs first (preserve natural boundaries)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[Chunk] = []
    current_paragraphs: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para, encoding)

        # If a single paragraph exceeds max_tokens, split it by sentences
        if para_tokens > max_tokens:
            # Flush current buffer first
            if current_paragraphs:
                chunk_text_str = "\n\n".join(current_paragraphs)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        document_id=document_id,
                        index=len(chunks),
                        text=chunk_text_str,
                        token_count=_count_tokens(chunk_text_str, encoding),
                    )
                )
                # Keep overlap: take trailing paragraphs
                current_paragraphs, current_tokens = _compute_overlap(
                    current_paragraphs, overlap_tokens, encoding
                )

            # Split long paragraph into sentence-level chunks
            sentences = _split_sentences(para)
            for sentence in sentences:
                s_tokens = _count_tokens(sentence, encoding)
                if current_tokens + s_tokens > max_tokens and current_paragraphs:
                    chunk_text_str = " ".join(current_paragraphs)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document_id}_chunk_{len(chunks)}",
                            document_id=document_id,
                            index=len(chunks),
                            text=chunk_text_str,
                            token_count=_count_tokens(chunk_text_str, encoding),
                        )
                    )
                    current_paragraphs, current_tokens = _compute_overlap(
                        current_paragraphs, overlap_tokens, encoding
                    )
                current_paragraphs.append(sentence)
                current_tokens += s_tokens
            continue

        # Normal case: paragraph fits in budget
        if current_tokens + para_tokens > max_tokens and current_paragraphs:
            chunk_text_str = "\n\n".join(current_paragraphs)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id,
                    index=len(chunks),
                    text=chunk_text_str,
                    token_count=_count_tokens(chunk_text_str, encoding),
                )
            )
            current_paragraphs, current_tokens = _compute_overlap(
                current_paragraphs, overlap_tokens, encoding
            )

        current_paragraphs.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_paragraphs:
        chunk_text_str = "\n\n".join(current_paragraphs)
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                document_id=document_id,
                index=len(chunks),
                text=chunk_text_str,
                token_count=_count_tokens(chunk_text_str, encoding),
            )
        )

    return ChunksResponse(
        document_id=document_id,
        total_chunks=len(chunks),
        chunks=chunks,
    )


def _split_sentences(text: str) -> list[str]:
    """Naïve sentence splitter — splits on period/question/exclamation + space."""
    import re

    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _compute_overlap(
    paragraphs: list[str],
    overlap_tokens: int,
    encoding: tiktoken.Encoding,
) -> tuple[list[str], int]:
    """
    Given the current list of paragraphs, return the trailing subset
    whose total tokens ≤ overlap_tokens (for chunk overlap).
    """
    if overlap_tokens <= 0:
        return [], 0

    result: list[str] = []
    total = 0
    for para in reversed(paragraphs):
        t = _count_tokens(para, encoding)
        if total + t > overlap_tokens:
            break
        result.insert(0, para)
        total += t
    return result, total
