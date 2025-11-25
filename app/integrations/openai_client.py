from __future__ import annotations

from typing import List, Dict, Any
import json

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
    Анализирует фрагмент документа с учётом переданных норм.

    ОЖИДАЕМЫЙ JSON-ОТВЕТ МОДЕЛИ (СТРОГО):

    {
      "has_contradiction": "найдено" или "не найдено",
      "contradiction_summary": "кратко в чём суть противоречия (если найдено, иначе пустая строка)",
      "legal_basis": "конкретные нормы закона и/или ссылки на пленум/обзор, строго из списка источников",
      "recommendation": "что учесть, как переформулировать норму/положение документа",
      "source_indices": [0, 2, ...]  // индексы использованных источников из переданного списка norms
    }

    norms: список словарей с полями:
      - type
      - number
      - short_title
      - summary

    Модель МОЖЕТ использовать только эти нормы и
    обязана возвращать индексы выбранных источников.
    """
    client = get_client()

    # Нумерованный список норм
    norms_lines = []
    for i, n in enumerate(norms):
        norms_lines.append(
            f"[{i}] {n.get('type')} {n.get('number')}: "
            f"{n.get('short_title')} — {n.get('summary')}"
        )
    norms_text = "\n".join(norms_lines) if norms_lines else "нет доступных норм"

    system_msg = (
        "Ты юридический ассистент, который анализирует фрагменты документа "
        "на предмет противоречий действующему законодательству РФ и правовым позициям высших судов "
        "по теме государственной пошлины (госпошлина).\n\n"
        "У тебя есть СТРОГО ограниченный набор источников (норм/доктрины), "
        "перечисленный ниже в виде списка с индексами [0], [1], [2] и т.д.\n"
        "Ты НЕ ИМЕЕШЬ права ссылаться на какие-либо другие источники, кроме этих. "
        "Нельзя придумывать новые номера постановлений, определений, статей и т.п.\n\n"
        "Если среди доступных источников нет подходящих для выявления противоречия, "
        "нужно указать, что противоречие не найдено и пояснить, что источников недостаточно.\n\n"
        "Твоя задача — сформировать аналитический вывод в структуре:\n"
        "Фрагмент\n"
        "Наличие противоречия с действующим законодательством и правовыми позициями высших судов: не найдено/найдено\n"
        "Если найдено противоречие:\n"
        "  Суть противоречия:\n"
        "  Правовое обоснование: (конкретные нормы закона или ссылки на пленум/обзор)\n"
        "  Рекомендация: (что учесть, как переформулировать)\n\n"
        "НО ОТВЕЧАТЬ ТЫ ДОЛЖЕН СТРОГО В ФОРМАТЕ JSON СЛЕДУЮЩЕГО ВИДА (БЕЗ ДОПОЛНИТЕЛЬНОГО ТЕКСТА):\n"
        "{\n"
        '  \"has_contradiction\": \"найдено\" или \"не найдено\",\n'
        '  \"contradiction_summary\": \"кратко опиши суть противоречия, если оно найдено; иначе пустая строка\",\n'
        '  \"legal_basis\": \"конкретные нормы (статьи закона, пункты постановлений/обзоров) строго из переданного списка источников\",\n'
        '  \"recommendation\": \"краткая рекомендация: что учесть, как переформулировать текст фрагмента\",\n'
        '  \"source_indices\": [0, 2, ...]  // индексы ИЗ ПРЕДОСТАВЛЕННОГО списка норм\n'
        "}\n"
        "Никаких других полей добавлять нельзя. Никакого текста вне JSON добавлять нельзя."
    )

    user_msg = (
        "Фрагмент документа:\n"
        "-----------------\n"
        f"{fragment_text}\n"
        "-----------------\n\n"
        "Доступные нормы/правовые позиции (единственные допустимые источники):\n"
        f"{norms_text}\n\n"
        "Проанализируй фрагмент с учётом ТОЛЬКО этих источников и верни JSON-ответ "
        "в описанном формате. Если подходящих норм нет или их явно недостаточно, "
        "укажи \"не найдено\" и поясни это в рекомендованных полях."
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

    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("LLM returned non-dict JSON")

        # Минимальная валидация и дефолты
        if data.get("has_contradiction") not in ("найдено", "не найдено"):
            # если модель что-то выдумала, считаем, что противоречие не найдено
            data["has_contradiction"] = "не найдено"

        data.setdefault("contradiction_summary", "")
        data.setdefault("legal_basis", "")
        data.setdefault("recommendation", "")

        if not isinstance(data.get("source_indices"), list):
            data["source_indices"] = []

        return data

    except Exception:
        # Fallback, если модель сломала формат
        return {
            "has_contradiction": "не найдено",
            "contradiction_summary": "",
            "legal_basis": "",
            "recommendation": "Не удалось корректно разобрать ответ модели. "
                              "Фрагмент считается условно не содержащим явного противоречия.",
            "source_indices": [],
        }
