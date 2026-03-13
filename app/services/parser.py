"""Task 1: Document Parser — converts PDF/TXT/MD files to plain text."""

from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path
from typing import BinaryIO

import pdfplumber

from app.models.document import DocumentUploadResponse


def _generate_document_id(filename: str, content_bytes: bytes) -> str:
    """Generate a deterministic document ID from filename + content hash."""
    content_hash = hashlib.sha256(content_bytes).hexdigest()[:12]
    safe_name = Path(filename).stem.replace(" ", "_")[:20]
    return f"{safe_name}_{content_hash}"


def _detect_content_type(filename: str) -> str:
    """Detect content type from file extension."""
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".pdf": "pdf",
        ".txt": "txt",
        ".md": "md",
        ".markdown": "md",
    }
    return mapping.get(suffix, "txt")


def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF using pdfplumber."""
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def parse_text(file_bytes: bytes) -> str:
    """Read plain text / markdown files."""
    # Try UTF-8 first, fall back to latin-1
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def parse_document(filename: str, file_bytes: bytes) -> DocumentUploadResponse:
    """
    Parse an uploaded file into plain text.

    Supports: PDF, TXT, MD.
    Returns a DocumentUploadResponse with a unique document_id.
    """
    content_type = _detect_content_type(filename)

    if content_type == "pdf":
        text = parse_pdf(file_bytes)
    else:
        text = parse_text(file_bytes)

    document_id = _generate_document_id(filename, file_bytes)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        content_type=content_type,
        text=text,
        char_count=len(text),
    )
