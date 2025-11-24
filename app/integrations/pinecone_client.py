from __future__ import annotations

from typing import Any

from pinecone import Pinecone, ServerlessSpec

from config import settings

_pc: Pinecone | None = None
_index: Any | None = None


def get_pinecone_client() -> Pinecone:
    global _pc
    if _pc is None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_pinecone_index():
    """
    Возвращает объект индекса.
    Если индекс не существует – создаёт serverless index.
    """
    global _index
    if _index is not None:
        return _index

    pc = get_pinecone_client()
    index_name = settings.pinecone_index_name

    existing = pc.list_indexes()
    existing_names = [i["name"] for i in existing.get("indexes", [])]

    # dimension 1536 – у text-embedding-3-small
    if index_name not in existing_names:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    _index = pc.Index(index_name)
    return _index
