import asyncio
import hashlib
import re
import time
from pathlib import Path
from typing import List

from tqdm import tqdm
from pinecone.exceptions import PineconeApiException

from app.services.text_extractor import extract_text
from app.integrations.openai_client import get_embedding
from app.integrations.pinecone_client import get_pinecone_index

DATA_DIR = Path("data/knowledge")

BATCH_SIZE = 50          # сколько векторов шлём за раз в Pinecone
MAX_RETRIES = 5          # сколько раз пробуем повторить upsert при ошибке
BASE_DELAY = 2.0         # базовая задержка между ретраями (секунды)


def split_into_chunks(text: str, max_chars: int = 2000) -> List[str]:
    """
    Делим текст на куски примерно по max_chars символов.
    Простейший чанкер: для RAG уже достаточно.
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
    Определяем type/number/short_title по имени файла.
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
    Делаем ASCII-only ID из хэша имени файла + индекса чанка.
    Pinecone требует ASCII id.
    """
    stem = file_path.stem
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{chunk_idx}"


def upsert_with_retry(index, vectors_batch: List[dict]) -> None:
    """
    Отправляет батч векторов в Pinecone с ретраями.
    Если после MAX_RETRIES всё равно ошибка - просто логируем и идём дальше.
    Повторный запуск скрипта безопасен, т.к. upsert идемпотентен.
    """
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            if not vectors_batch:
                return
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

    # Если дошли сюда - батч так и не залился
    print(
        f"\n[ERROR] Не удалось загрузить батч из {len(vectors_batch)} векторов "
        f"после {MAX_RETRIES} попыток. Эти векторы будут пропущены. "
        f"Вы можете перезапустить скрипт позже - upsert в Pinecone идемпотентен."
    )


async def main() -> None:
    index = get_pinecone_index()

    # Собираем список PDF-файлов
    pdf_files = [
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    ]

    if not pdf_files:
        print("Нет PDF-файлов в data/knowledge — индексировать нечего.")
        return

    total_chunks = 0

    print(f"Найдено {len(pdf_files)} PDF-файлов для индексации.\n")

    # Проходим по файлам с прогресс-баром
    for file_path in tqdm(pdf_files, desc="Файлы", unit="файл"):
        print(f"\nОбрабатываю PDF: {file_path}")

        text = extract_text(file_path).strip()
        if not text:
            print("  ⚠ Нет текста — возможно, PDF-скан. Пропускаю.")
            continue

        chunks = split_into_chunks(text, max_chars=1800)
        if not chunks:
            print("  ⚠ Не удалось разбить текст на чанки. Пропускаю.")
            continue

        print(f"  ➜ Разбито на {len(chunks)} чанков")

        meta_base = _build_metadata(file_path)
        meta_base["summary"] = text[:700]

        vectors_batch: List[dict] = []
        # Прогресс по чанкам внутри файла
        for idx, chunk in enumerate(
            tqdm(chunks, desc="  Чанки", unit="chunk", leave=False)
        ):
            emb = await get_embedding(chunk)

            vector_id = make_vector_id(file_path, idx)

            vectors_batch.append(
                {
                    "id": vector_id,
                    "values": emb,
                    "metadata": {
                        **meta_base,
                        "chunk_index": idx,
                    },
                }
            )
            total_chunks += 1

            # Если набрали батч - отправляем с ретраями
            if len(vectors_batch) >= BATCH_SIZE:
                upsert_with_retry(index, vectors_batch)
                vectors_batch = []

        # Остаток чанков по файлу
        if vectors_batch:
            upsert_with_retry(index, vectors_batch)
            vectors_batch = []

    print(f"\n✅ Индексация завершена. Всего обработано чанков: {total_chunks}")


if __name__ == "__main__":
    asyncio.run(main())
