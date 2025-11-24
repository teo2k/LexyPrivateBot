from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence


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
    summary: str       # 2-3 предложения по сути нормы


class BaseVectorStore(Protocol):
    """
    Интерфейс для любой векторной БД (Pinecone, Qdrant, Chroma и т.п.).
    """

    async def search(
        self,
        query_embedding: Sequence[float],
        k: int = 5,
    ) -> List[NormItem]:
        ...


class DummyVectorStore(BaseVectorStore):
    """
    Временная заглушка вместо настоящей векторной базы.
    Возвращает фиксированный набор норм.
    """

    async def search(
        self,
        query_embedding: Sequence[float],
        k: int = 5,
    ) -> List[NormItem]:
        # Здесь потом вместо этого будет реальный запрос в Pinecone / др. векторку
        items = [
            NormItem(
                type="ПП ВС РФ",
                number="№ 45",
                short_title="О применении главы 25.3 НК РФ (госпошлина)",
                url=None,
                summary=(
                    "Пленум разъясняет порядок исчисления и уплаты государственной "
                    "пошлины, применение льгот и возврат излишне уплаченных сумм."
                ),
            ),
            NormItem(
                type="КС РФ",
                number="Определение № 123-О",
                short_title="О конституционности норм о госпошлине",
                url=None,
                summary=(
                    "Конституционный Суд оценивает соразмерность размера "
                    "госпошлины и гарантии доступа к правосудию."
                ),
            ),
        ]
        return items[:k]
