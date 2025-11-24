import asyncio
import hashlib
import re
from pathlib import Path

from app.services.text_extractor import extract_text
from app.integrations.openai_client import get_embedding
from app.integrations.pinecone_client import get_pinecone_index

DATA_DIR = Path("data/knowledge")


def split_into_chunks(text: str, max_chars: int = 2000):
    """
    Делим текст на куски примерно по max_chars символов.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
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

    # № в названии
    m = re.search(r"№\s*([0-9,\s/()-]+)", name)
    if m:
        number = m.group(1).strip()
    else:
        m = re.search(r"\d+", name)
        number = m.group(0) if m else name

    return {
        "type": type_,
        "number": number,
        "short_title": name.replace("_", " "),  # здесь можно оставить русский
        "url": "",
    }


def make_vector_id(file_path: Path, chunk_idx: int) -> str:
    """
    Делаем ASCII-only ID из хэша имени файла + индекса чанка.
    Pinecone требует, чтобы id был ASCII.
    """
    stem = file_path.stem
    # md5 от имени файла, чтобы ID был стабильным
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{chunk_idx}"


async def main() -> None:
    index = get_pinecone_index()

    vectors = []
    total_chunks = 0

    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix != ".pdf":
            continue

        print(f"Обрабатываю PDF: {file_path}")

        text = extract_text(file_path).strip()
        if not text:
            print("  ⚠ Нет текста — возможно, PDF-скан. Пропускаю.")
            continue

        chunks = split_into_chunks(text, max_chars=1800)
        print(f"  ➜ Разбито на {len(chunks)} чанков")

        meta_base = _build_metadata(file_path)
        meta_base["summary"] = text[:700]

        for idx, chunk in enumerate(chunks):
            emb = await get_embedding(chunk)

            vector_id = make_vector_id(file_path, idx)

            vectors.append(
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

    if not vectors:
        print("Нет файлов для индексации.")
        return

    print(f"Загружаю {total_chunks} чанков в Pinecone...")
    index.upsert(vectors=vectors)
    print(f"✅ Успешно загружено {total_chunks} чанков в Pinecone")


if __name__ == "__main__":
    asyncio.run(main())
