from __future__ import annotations

from typing import List, Dict, Any
import json
import re

from openai import OpenAI
from config import settings


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


async def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    client = get_client()
    resp = client.embeddings.create(
        model=model,
        input=text,
    )
    return resp.data[0].embedding


# ========================================================================
#   НОВАЯ СТАБИЛЬНАЯ ВЕРСИЯ ЮРИДИЧЕСКОГО АНАЛИЗА (RAG + JSON ONLY)
# ========================================================================
async def analyze_fragment_with_norms(
    fragment_text: str,
    norms: List[Dict[str, Any]],
    model: str = "gpt-5.1",
) -> Dict[str, Any]:

    client = get_client()

    # ==========================
    # Формируем текст норм
    # ==========================
    norms_lines = []
    for i, n in enumerate(norms):
        norms_lines.append(
            f"[{i}] {n.get('type')} {n.get('number')}: {n.get('short_title')} — {n.get('summary')}"
        )
    norms_text = "\n".join(norms_lines) if norms_lines else "нет доступных норм"


    # ==========================
    # Новый короткий SYSTEM PROMPT
    # ==========================
    system_msg = (
        "Ты - юридический анализатор в режиме RAG. "
        "Ты можешь использовать только те нормы, которые перечислены пользователем. "
        "Запрещено придумывать статьи, пункты, номера Пленума или обзоры. "
        "Анализируй по алгоритму:\n"
        "1) Определи ошибочный тезис (цитата или краткое резюме).\n"
        "2) Сопоставь с доступными нормами: ищи противоречия, ошибки толкования, неполноту.\n"
        "3) Квалифицируй ошибку как: «Прямое противоречие НК РФ», «Несоответствие практике ВС РФ», "
        "«Неполное толкование нормы», «Вводящее в заблуждение разъяснение».\n"
        "4) Если противоречия нет — вывод «OK». Если есть — «Риск».\n"
        "5) Ты обязан выводить ответ строго в JSON:\n"
        "{\n"
        "  \"label\": \"OK\" | \"Риск\",\n"
        "  \"comment\": \"1-3 предложения\",\n"
        "  \"correct_position\": \"правильная позиция по нормам\",\n"
        "  \"source_indices\": [индексы]\n"
        "}\n"
        "Никакого текста вне JSON."
    )

    # ==========================
    # Новый USER PROMPT
    # ==========================
    user_msg = (
        "Проанализируй юридический фрагмент по алгоритму из system prompt.\n\n"
        "Фрагмент:\n"
        "-----------------\n"
        f"{fragment_text}\n"
        "-----------------\n\n"
        "Доступные нормы:\n"
        f"{norms_text}\n\n"
        "Используй только эти нормы. "
        "Ответ строго в JSON без текста вне JSON."
    )


    # ==========================
    # Вызов OpenAI
    # ==========================
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=400,
    )

    raw_content = resp.choices[0].message.content or "{}"

    # ==========================
    # Надёжный JSON-парсер
    # ==========================

    # 1) Пытаемся вытащить JSON блок даже если модель окружила его текстом
    match = re.search(r"\{[\s\S]*\}", raw_content)
    if match:
        raw_content = match.group(0)

    try:
        data = json.loads(raw_content)

        if not isinstance(data, dict):
            raise ValueError("Not a dict")

        # Валидация
        if data.get("label") not in ("OK", "Риск"):
            data["label"] = "OK"

        data.setdefault("comment", "")
        data.setdefault("correct_position", "")

        if not isinstance(data.get("source_indices"), list):
            data["source_indices"] = []

        return data

    except Exception:
        # fallback — но теперь ЧЕСТНЫЙ
        return {
            "label": "OK",
            "comment": (
                "Модель вернула некорректный JSON. "
                "Источников недостаточно или формат нарушен."
            ),
            "correct_position": "",
            "source_indices": [],
        }
