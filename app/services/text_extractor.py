from pathlib import Path
from typing import Optional

import pdfplumber
from docx import Document as DocxDocument


def extract_text(file_path: Path) -> str:
    """
    Универсальный извлекатель текста:
    - .docx → python-docx
    - .pdf  → pdfplumber (текстовые PDF)
    - иначе вернёт пустую строку.

    OCR для сканов добавим позже.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".docx":
        return _extract_docx(file_path)
    elif suffix == ".pdf":
        return _extract_pdf(file_path)
    else:
        return ""


def _extract_docx(file_path: Path) -> str:
    try:
        doc = DocxDocument(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs]
        paragraphs = [p for p in paragraphs if p]
        return "\n".join(paragraphs)
    except Exception as e:
        return f"[Ошибка извлечения текста из DOCX: {e}]"


def _extract_pdf(file_path: Path) -> str:
    try:
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                text_parts.append(t.strip())
        return "\n".join(text_parts)
    except Exception as e:
        return f"[Ошибка извлечения текста из PDF: {e}]"
