from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Union

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError  # добавь
import docx2txt  # если ещё не установлено: pip install docx2txt

BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
CACHE_DIR = BASE_DIR / "data" / "knowledge_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_paths(pdf_path: Path) -> tuple[Path, Path]:
    safe_name = pdf_path.stem
    txt_path = CACHE_DIR / f"{safe_name}.txt"
    meta_path = CACHE_DIR / f"{safe_name}.meta.json"
    return txt_path, meta_path


def _is_cache_valid(pdf_path: Path, meta_path: Path) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pdf_mtime = pdf_path.stat().st_mtime
        return abs(meta.get("mtime", 0) - pdf_mtime) < 1e-3
    except Exception:
        return False


def _extract_pdf_with_cache(pdf_path: Path) -> str:
    txt_path, meta_path = _get_cache_paths(pdf_path)

    # кеш только для PDF из knowledge
    if _is_cache_valid(pdf_path, meta_path) and txt_path.exists():
        return txt_path.read_text(encoding="utf-8")

    t0 = time.time()
    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as e:
        print(f"[ERROR] Битый PDF {pdf_path.name}: {e}")
        return ""
    except Exception as e:
        print(f"[ERROR] Ошибка чтения PDF {pdf_path.name}: {e}")
        return ""

    parts = []
    for page_idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            print(f"[WARN] Ошибка при чтении страницы {page_idx} в {pdf_path.name}: {e}")
            page_text = ""
        parts.append(page_text)

    text = "\n\n".join(parts)

    dt = time.time() - t0
    print(
        f"[INFO] Прочитан PDF {pdf_path.name}: {len(reader.pages)} стр., "
        f"{len(text)} символов, {dt:.1f} с на парсинг"
    )

    # кешируем ТОЛЬКО знания, а не пользовательские файлы
    try:
        txt_path.write_text(text, encoding="utf-8")
        meta = {
            "mtime": pdf_path.stat().st_mtime,
            "pages": len(reader.pages),
            "chars": len(text),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Не удалось сохранить кеш для {pdf_path.name}: {e}")

    return text


def _extract_pdf_no_cache(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as e:
        print(f"[ERROR] Битый PDF {pdf_path.name}: {e}")
        return ""
    except Exception as e:
        print(f"[ERROR] Ошибка чтения PDF {pdf_path.name}: {e}")
        return ""

    parts = []
    for page_idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            print(f"[WARN] Ошибка при чтении страницы {page_idx} в {pdf_path.name}: {e}")
            page_text = ""
        parts.append(page_text)

    return "\n\n".join(parts)


def _extract_docx(docx_path: Path) -> str:
    try:
        text = docx2txt.process(str(docx_path))
        return text or ""
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать DOCX {docx_path.name}: {e}")
        return ""


def extract_text(path: Union[str, Path]) -> str:
    """
    Универсальный экстрактор:
    - PDF из data/knowledge -> с кешем
    - любые другие PDF -> без кеша
    - DOCX -> без кеша
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # если это наш "knowledge" PDF - кешируем
        try:
            # Python 3.9+: is_relative_to
            if path.is_relative_to(KNOWLEDGE_DIR):
                return _extract_pdf_with_cache(path)
        except AttributeError:
            # если 3.8–3.9, можно заменить своей проверкой
            if str(path).startswith(str(KNOWLEDGE_DIR)):
                return _extract_pdf_with_cache(path)
        # все остальные pdf -> без кеша
        return _extract_pdf_no_cache(path)

    if suffix == ".docx":
        return _extract_docx(path)

    print(f"[WARN] Неподдерживаемый формат файла: {path.name}")
    return ""
