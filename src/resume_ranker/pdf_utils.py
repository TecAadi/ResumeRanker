"""Utilities for extracting text from PDF resumes."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

from pdfminer.high_level import extract_text


PdfInput = Union[str, Path, bytes, bytearray]



def extract_text_from_pdf(pdf_source: PdfInput) -> str:
    """Extract text from a PDF file path or in-memory bytes.

    Args:
        pdf_source: PDF source as a filesystem path or bytes.

    Returns:
        Extracted UTF-8 text with normalized whitespace.
    """
    if isinstance(pdf_source, (str, Path)):
        raw_text = extract_text(str(pdf_source))
    else:
        raw_text = extract_text(BytesIO(bytes(pdf_source)))

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)
