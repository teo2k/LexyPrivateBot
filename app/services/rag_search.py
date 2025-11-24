from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.integrations.openai_client import get_embedding
from app.integrations.pinecone_client import get_pinecone_index


@dataclass
class NormItem:
    """
    Одна норма/позиция из базы:
    - ПП ВС РФ / КС РФ / доктрина.
    """
    type: str          # "ПП ВС РФ", "КС РФ", "Доктрина"
    number: str        # № постановления / определения / статьи
    short_title: str   # краткое текстовое описание
    url: str | None    # ссылка (если есть)
    summary: str       # 2–3 предложения по сути нормы


async def find_relevant_norms(
    fragment_text: str,
    k: int = 5,
) -> List[NormItem]:
    """
    Строит эмбеддинг фрагмента и ищет релевантные нормы в Pinecone.
    Ожидается, что в metadata хранятся:
      - type
      - number
      - short_title
      - url
      - summary
    """
    embedding = await get_embedding(fragment_text)
    index = get_pinecone_index()

    res = index.query(
        vector=embedding,
        top_k=k,
        include_metadata=True,
    )

    # в новом SDK res может быть dict или объект – обрабатываем оба варианта
    matches = res["matches"] if isinstance(res, dict) else res.matches

    norms: List[NormItem] = []

    for m in matches or []:
        md = m["metadata"] if isinstance(m, dict) else getattr(m, "metadata", {}) or {}

        norms.append(
            NormItem(
                type=str(md.get("type", "")),
                number=str(md.get("number", "")),
                short_title=str(md.get("short_title", "")),
                url=md.get("url"),
                summary=str(md.get("summary", "")),
            )
        )

    return norms
