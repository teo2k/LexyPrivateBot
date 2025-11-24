from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Union

from PyPDF2 import PdfReader

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


def extract_text(pdf_path: Union[str, Path]) -> str:
    """
    Извлекает текст из PDF с простым кешированием.
    - Если есть валидный .txt в CACHE_DIR и PDF не менялся, читаем из кеша.
    - Иначе парсим PDF, сохраняем .txt + .meta.json и возвращаем текст.
    """
    pdf_path = Path(pdf_path)

    txt_path, meta_path = _get_cache_paths(pdf_path)

    # 1) Пробуем кеш
    if _is_cache_valid(pdf_path, meta_path) and txt_path.exists():
        text = txt_path.read_text(encoding="utf-8")
        return text

    # 2) Нет кеша / PDF обновился - парсим заново
    t0 = time.time()
    reader = PdfReader(str(pdf_path))

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

    # 3) Сохраняем кеш
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
