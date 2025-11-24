import asyncio
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm as tqdm_sync

from app.services.text_extractor import extract_text
from app.integrations.openai_client import get_embedding
from app.integrations.pinecone_client import get_pinecone_index

DATA_DIR = Path("data/knowledge")

CHUNK_SIZE = 1800
BATCH_SIZE = 50
MAX_RETRIES = 5
BASE_DELAY = 2.0
CONCURRENCY = 10          # одновременно 10 запросов к OpenAI
EMBED_TIMEOUT = 60        # таймаут на один запрос эмбеддинга (секунд)


@dataclass
class ChunkItem:
    file_path: Path
    chunk_index: int
    text: str
    metadata_base: dict


def split_into_chunks(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    return [
        text[i:i + max_chars].strip()
        for i in range(0, len(text), max_chars)
    ]


def build_metadata(file_path: Path) -> dict:
    name = file_path.stem
    lower = name.lower()

    if "ksrf" in lower or "конституцион" in lower:
        type_ = "КС РФ"
    elif "обзор судебной практики" in lower:
        type_ = "Обзор ВС РФ"
    elif "постановление пленума" in lower:
        type_ = "ПП ВС РФ"
    else:
        type_ = "Доктрина"

    m = re.search(r"№\s*([0-9,\s/-]+)", name)
    number = m.group(1) if m else name

    return {
        "type": type_,
        "number": number,
        "short_title": name.replace("_", " "),
        "url": "",
    }


def make_vector_id(file_path: Path, chunk_idx: int) -> str:
    digest = hashlib.md5(file_path.stem.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{chunk_idx}"


async def get_embedding_with_retry(text: str) -> Optional[List[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            # таймаут на запрос к OpenAI
            return await asyncio.wait_for(
                get_embedding(text),
                timeout=EMBED_TIMEOUT
            )
        except asyncio.TimeoutError:
            delay = BASE_DELAY * (attempt + 1)
            print(f"[WARN] get_embedding timeout, retry через {delay}s")
            await asyncio.sleep(delay)
        except Exception as e:
            delay = BASE_DELAY * (attempt + 1)
            print(f"[WARN] get_embedding ошибка: {e}, retry через {delay}s")
            await asyncio.sleep(delay)

    print("[ERROR] эмбеддинг не получен, пропуск чанка.")
    return None


def upsert_with_retry(index, batch):
    for attempt in range(MAX_RETRIES):
        try:
            index.upsert(vectors=batch)
            return
        except Exception as e:
            delay = BASE_DELAY * (attempt + 1)
            print(f"[WARN] Pinecone upsert ошибка: {e}, retry через {delay}s")
            time.sleep(delay)

    print(f"[ERROR] batch ({len(batch)}) пропущен.")


async def process_chunk_item(item: ChunkItem):
    emb = await get_embedding_with_retry(item.text)
    if emb is None:
        return None

    return {
        "id": make_vector_id(item.file_path, item.chunk_index),
        "values": emb,
        "metadata": {
            **item.metadata_base,
            "chunk_index": item.chunk_index,
        }
    }


async def main():
    # -------------------------------
    # 1) ЧТЕНИЕ PDF + СПЛИТ
    # -------------------------------
    pdf_files = [p for p in DATA_DIR.rglob("*.pdf")]

    print(f"Найдено файлов: {len(pdf_files)}")

    all_chunks: List[ChunkItem] = []

    for file_path in tqdm_sync(pdf_files, desc="Чтение PDF", unit="file"):
        text = extract_text(file_path).strip()
        if not text:
            continue

        chunks = split_into_chunks(text)
        meta = build_metadata(file_path)
        meta["summary"] = text[:700]

        for i, ch in enumerate(chunks):
            all_chunks.append(
                ChunkItem(
                    file_path=file_path,
                    chunk_index=i,
                    text=ch,
                    metadata_base=meta
                )
            )

    print(f"Всего чанков: {len(all_chunks)}")

    # -------------------------------
    # 2) ПАРАЛЛЕЛЬНО СЧИТАЕМ ЭМБЕДДИНГИ (БАТЧАМИ)
    # -------------------------------
    print("Получаю эмбеддинги...")

    results = []
    pbar = tqdm_sync(
        total=len(all_chunks),
        desc="Эмбеддинги",
        unit="chunk"
    )

    # обрабатываем по CONCURRENCY чанков за раз
    for i in range(0, len(all_chunks), CONCURRENCY):
        batch_items = all_chunks[i:i + CONCURRENCY]

        # запускаем задачи на текущий батч
        tasks = [process_chunk_item(item) for item in batch_items]
        batch_results = await asyncio.gather(*tasks)

        for res in batch_results:
            if res is not None:
                results.append(res)

        # обновляем прогресс на количество обработанных чанков
        pbar.update(len(batch_items))

    pbar.close()

    print(f"Эмбеддингов получено: {len(results)}")

    # -------------------------------
    # 3) ПОДКЛЮЧАЕМСЯ К PINECONE
    # -------------------------------
    print("Подключаюсь к Pinecone...")
    index = get_pinecone_index()
    print("Готово.")

    # -------------------------------
    # 4) ЗАГРУЗКА В PINECONE (БАТЧИ)
    # -------------------------------
    print("Загружаю в Pinecone...")
    for i in tqdm_sync(range(0, len(results), BATCH_SIZE), unit="batch", desc="Upsert"):
        batch = results[i:i + BATCH_SIZE]
        upsert_with_retry(index, batch)

    print("✅ Индексация завершена")


if __name__ == "__main__":
    asyncio.run(main())
