from __future__ import annotations

from typing import List, Dict, Any

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
    return resp.data[0].embedding  # type: ignore[no-any-return]


async def analyze_fragment_with_norms(
    fragment_text: str,
    norms: List[Dict[str, Any]],
    model: str = "gpt-5.1",
) -> Dict[str, Any]:
    """
    norms: список словарей с полями:
      - type
      - number
      - short_title
      - summary

    LLM МОЖЕТ использовать только эти нормы и
    обязан возвращать индексы выбранных источников.
    """
    client = get_client()

    # нумерованный список норм
    norms_lines = []
    for i, n in enumerate(norms):
        norms_lines.append(
            f"[{i}] {n.get('type')} {n.get('number')}: "
            f"{n.get('short_title')} — {n.get('summary')}"
        )
    norms_text = "\n".join(norms_lines) if norms_lines else "нет норм"

    system_msg = (
        "Ты юридический ассистент, который анализирует фрагменты документа "
        "на тему государственной пошлины (госпошлина) в России.\n\n"
        "У тебя есть СТРОГО ограниченный набор источников (норм/доктрины), "
        "перечисленный ниже в виде списка с индексами [0], [1], [2] и т.д.\n"
        "Ты НЕ ИМЕЕШЬ права ссылаться на какие-либо другие источники, кроме этих. "
        "Нельзя придумывать новые номера постановлений, определений, статей и т.п.\n\n"
        "Если среди доступных источников нет подходящих для аргументации риска, "
        "просто ставь label = \"OK\" и объясняй в комментарии, что недостаточно информации.\n\n"
        "Отвечай строго в формате JSON:\n"
        "{\n"
        '  \"label\": \"OK\" или \"Риск\",\n'
        '  \"comment\": \"1–3 коротких предложения, что не так / где риск (или почему всё ок)\",\n'
        '  \"correct_position\": \"краткая суть корректной позиции по нормам\",\n'
        '  \"source_indices\": [0, 2, ...]  // индексы ИЗ ПРЕДОСТАВЛЕННОГО списка норм\n'
        "}\n"
        "Никаких других полей добавлять нельзя."
    )

    user_msg = (
        "Фрагмент документа:\n"
        "-----------------\n"
        f"{fragment_text}\n"
        "-----------------\n\n"
        "Релевантные нормы/позиции (допустимые источники):\n"
        f"{norms_text}\n\n"
        "Проанализируй фрагмент с учётом только этих норм и верни JSON-ответ."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )

    content = resp.choices[0].message.content or "{}"

    import json

    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("LLM returned non-dict JSON")

        # минимальная валидация
        data.setdefault("label", "OK")
        data.setdefault("comment", "")
        data.setdefault("correct_position", "")
        if not isinstance(data.get("source_indices"), list):
            data["source_indices"] = []
        return data
    except Exception:
        return {
            "label": "OK",
            "comment": "Не удалось корректно разобрать ответ модели. "
                       "Считаем фрагмент условно безопасным.",
            "correct_position": "",
            "source_indices": [],
        }
