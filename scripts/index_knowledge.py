import asyncio
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from pinecone.exceptions import PineconeApiException

from app.services.text_extractor import extract_text
from app.integrations.openai_client import get_embedding
from app.integrations.pinecone_client import get_pinecone_index

DATA_DIR = Path("data/knowledge")

BATCH_SIZE = 50          # сколько векторов шлём за раз в Pinecone
MAX_RETRIES = 5          # сколько раз пробуем повторить запрос при ошибке
BASE_DELAY = 2.0         # базовая задержка между ретраями (секунды)


@dataclass
class ChunkItem:
    file_path: Path
    chunk_index: int
    text: str
    metadata_base: dict


def split_into_chunks(text: str, max_chars: int = 2000) -> List[str]:
    """
    Делим текст на куски примерно по max_chars символов.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


def _build_metadata(file_path: Path) -> dict:
    """
    Эвристика под твои названия файлов.
    """
    name = file_path.stem
    lower = name.lower()

    if "ksrf" in lower or "decision" in lower or "конституцион" in lower:
        type_ = "КС РФ"
    elif "обзор судебной практики верховного суда" in lower:
        type_ = "Обзор ВС РФ"
    elif "постановление пленума верховного суда" in lower:
        type_ = "ПП ВС РФ"
    elif "mesto_gosudarstvennoy" in lower or "госпошлин" in lower:
        type_ = "Доктрина"
    else:
        type_ = "Доктрина"

    # Ищем № в названии
    m = re.search(r"№\s*([0-9,\s/()-]+)", name)
    if m:
        number = m.group(1).strip()
    else:
        m = re.search(r"\d+", name)
        number = m.group(0) if m else name

    return {
        "type": type_,
        "number": number,
        "short_title": name.replace("_", " "),
        "url": "",
    }


def make_vector_id(file_path: Path, chunk_idx: int) -> str:
    """
    ASCII-only ID из хэша имени файла + индекса чанка.
    Pinecone требует ASCII id.
    """
    stem = file_path.stem
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{chunk_idx}"


def upsert_with_retry(index, vectors_batch: List[dict]) -> None:
    """
    Отправляет батч векторов в Pinecone с ретраями.
    """
    if not vectors_batch:
        return

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            index.upsert(vectors=vectors_batch)
            return
        except PineconeApiException as e:
            attempt += 1
            delay = BASE_DELAY * attempt
            print(
                f"\n[WARN] Ошибка Pinecone при upsert (попытка {attempt}/{MAX_RETRIES}): {e}. "
                f"Повтор через {delay:.1f} сек."
            )
            time.sleep(delay)
        except Exception as e:
            attempt += 1
            delay = BASE_DELAY * attempt
            print(
                f"\n[WARN] Неожиданная ошибка при upsert (попытка {attempt}/{MAX_RETRIES}): {e}. "
                f"Повтор через {delay:.1f} сек."
            )
            time.sleep(delay)

    print(
        f"\n[ERROR] Не удалось загрузить батч из {len(vectors_batch)} векторов "
        f"после {MAX_RETRIES} попыток. Эти векторы будут пропущены. "
        f"Вы можете перезапустить скрипт позже — upsert идемпотентен."
    )


async def get_embedding_with_retry(text: str) -> Optional[List[float]]:
    """
    Обёртка над get_embedding с ретраями на случай проблем с интернетом/лимитами.
    """
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            return await get_embedding(text)
        except Exception as e:
            attempt += 1
            delay = BASE_DELAY * attempt
            print(
                f"\n[WARN] Ошибка при получении эмбеддинга (попытка {attempt}/{MAX_RETRIES}): {e}. "
                f"Повтор через {delay:.1f} сек."
            )
            await asyncio.sleep(delay)

    print(
        f"\n[ERROR] Не удалось получить эмбеддинг для чанка "
        f"после {MAX_RETRIES} попыток. Чанк будет пропущен."
    )
    return None


async def main() -> None:
    index = get_pinecone_index()

    # 1) Собираем все PDF-файлы
    pdf_files = [
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    ]

    if not pdf_files:
        print("Нет PDF-файлов в data/knowledge — индексировать нечего.")
        return

    print(f"Найдено {len(pdf_files)} PDF-файлов для индексации.\n")

    # 2) Первый проход: читаем файлы, режем на чанки, собираем все ChunkItem в память
    all_chunks: List[ChunkItem] = []

    for file_path in tqdm(pdf_files, desc="Чтение файлов", unit="файл"):
        text = extract_text(file_path).strip()
        if not text:
            print(f"  ⚠ {file_path} — нет текста (возможно, скан). Пропускаю.")
            continue

        chunks = split_into_chunks(text, max_chars=1800)
        if not chunks:
            print(f"  ⚠ {file_path} — не удалось разбить на чанки. Пропускаю.")
            continue

        meta_base = _build_metadata(file_path)
        meta_base["summary"] = text[:700]

        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                ChunkItem(
                    file_path=file_path,
                    chunk_index=idx,
                    text=chunk,
                    metadata_base=meta_base,
                )
            )

    if not all_chunks:
        print("Нет чанков для индексации.")
        return

    print(f"\nВсего чанков для индексации: {len(all_chunks)}\n")

    # 3) Второй проход: считаем эмбеддинги и шлём в Pinecone с прогресс-баром
    total_chunks = 0
    vectors_batch: List[dict] = []

    for item in tqdm(all_chunks, desc="Эмбеддинги + upsert", unit="chunk"):
        emb = await get_embedding_with_retry(item.text)
        if emb is None:
            # чанк пропускаем (ошибка при получении эмбеддинга)
            continue

        vector_id = make_vector_id(item.file_path, item.chunk_index)

        vectors_batch.append(
            {
                "id": vector_id,
                "values": emb,
                "metadata": {
                    **item.metadata_base,
                    "chunk_index": item.chunk_index,
                },
            }
        )
        total_chunks += 1

        if len(vectors_batch) >= BATCH_SIZE:
            upsert_with_retry(index, vectors_batch)
            vectors_batch = []

    # Остаток батча
    if vectors_batch:
        upsert_with_retry(index, vectors_batch)

    print(f"\n✅ Индексация завершена. Всего загружено чанков: {total_chunks}")


if __name__ == "__main__":
    asyncio.run(main())
