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
    Анализирует фрагмент документа по алгоритму табличной схемы.

    ОЖИДАЕМЫЙ JSON-ОТВЕТ:

    {
      "has_contradiction": "найдено" | "не найдено",
      "contradiction_summary": "кратко в чём суть противоречия (если найдено)",
      "legal_basis": "конкретные нормы закона / ссылки на Пленум / Обзор (из переданных источников)",
      "recommendation": "что учесть, как переформулировать",
      "source_indices": [0, 2, ...]  // индексы источников из списка norms
    }
    """

    client = get_client()

    # Нумерованный список норм с индексами
    norms_lines = []
    for i, n in enumerate(norms):
        norms_lines.append(
            f"[{i}] {n.get('type')} {n.get('number')}: "
            f"{n.get('short_title')} — {n.get('summary')}"
        )
    norms_text = "\n".join(norms_lines) if norms_lines else "нет доступных норм"

    system_msg = (
        "Ты юридический аналитик. Твоя задача — анализировать фрагменты проекта постановления "
        "и выявлять противоречия действующему законодательству РФ и правовым позициям высших судов, "
        "используя табличную схему.\n\n"
        "Алгоритм, которому ты строго следуешь:\n\n"
        "Шаг 1: Подготовка инструментов\n"
        "Создаётся таблица с 4 колонками:\n"
        "  1) № п/п (не нужно выводить в JSON),\n"
        "  2) Проект Постановления (Ошибочный тезис),\n"
        "  3) Действующая норма / Позиция ВС РФ,\n"
        "  4) Вывод.\n\n"
        "Источники для проверки:\n"
        "- Налоговый кодекс РФ,\n"
        "- процессуальные кодексы (ГПК, АПК, КАС),\n"
        "- актуальные постановления Пленума ВС РФ,\n"
        "- обзоры судебной практики ВС РФ.\n"
        "Ты можешь использовать ТОЛЬКО те источники, которые даны в списке норм с индексами. "
        "Нельзя придумывать новые номера статей, Пленумов, Обзоров.\n\n"
        "Шаг 2: Последовательный анализ текста фрагмента\n"
        "- мысленно разбей фрагмент на смысловые тезисы,\n"
        "- для каждого тезиса проверь:\n"
        "  * соответствие нормам кодексов,\n"
        "  * соответствие практике ВС РФ,\n"
        "  * полноту отражения нормы,\n"
        "  * отсутствие внутренних противоречий.\n\n"
        "Шаг 3: Заполнение таблицы (для каждого выявленного противоречия)\n"
        "Колонка \"Проект Постановления\":\n"
        "- процитируй проблемный фрагмент ИЛИ кратко сформулируй ошибочный тезис.\n\n"
        "Колонка \"Действующая норма / Позиция ВС РФ\":\n"
        "- укажи конкретную статью кодекса с номером пункта, ИЛИ\n"
        "- номер и пункт постановления Пленума ВС РФ, ИЛИ\n"
        "- номер Обзора практики ВС РФ.\n\n"
        "Колонка \"Вывод\":\n"
        "- краткая юридическая квалификация, например:\n"
        "  \"Прямое противоречие НК РФ\",\n"
        "  \"Несоответствие практике ВС РФ\",\n"
        "  \"Неполное толкование нормы\",\n"
        "  \"Вводящее в заблуждение разъяснение\".\n\n"
        "Шаг 4: Контроль качества\n"
        "- проверь корректность цитирования,\n"
        "- актуальность указанных норм,\n"
        "- логичность вывода,\n"
        "- отсутствие повторов.\n\n"
        "Шаг 5: Ранжирование по значимости (внутренне, для себя):\n"
        "- прямые противоречия кодексам,\n"
        "- противоречия практике ВС РФ,\n"
        "- неполнота регулирования,\n"
        "- технические неточности.\n\n"
        "ОТВЕЧАТЬ ТЫ ДОЛЖЕН СТРОГО В ФОРМАТЕ JSON:\n"
        "{\n"
        '  \"has_contradiction\": \"найдено\" или \"не найдено\",\n'
        '  \"contradiction_summary\": \"кратко опиши суть противоречия, если оно есть; иначе пустая строка\",\n'
        '  \"legal_basis\": \"конкретные нормы (статьи, пункты Пленума/Обзора) строго из переданного списка источников\",\n'
        '  \"recommendation\": \"что учесть и как переформулировать проблемный тезис\",\n'
        '  \"source_indices\": [0, 2, ...]\n'
        "}\n"
        "Если подходящих норм среди переданного списка нет, поставь "
        "\"has_contradiction\": \"не найдено\", опиши в recommendation, что "
        "источников недостаточно для выявления противоречия, и оставь legal_basis пустым или с пометкой \"не найдено\".\n"
        "Никакого текста вне JSON добавлять нельзя."
    )

    user_msg = (
        "Фрагмент документа для анализа:\n"
        "-----------------\n"
        f"{fragment_text}\n"
        "-----------------\n\n"
        "Доступные нормы (единственные допустимые источники):\n"
        f"{norms_text}\n\n"
        "Проанализируй фрагмент по описанному алгоритму и верни СТРОГО JSON."
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

        # дефолты и минимальная валидация
        if data.get("has_contradiction") not in ("найдено", "не найдено"):
            data["has_contradiction"] = "не найдено"

        data.setdefault("contradiction_summary", "")
        data.setdefault("legal_basis", "")
        data.setdefault("recommendation", "")

        if not isinstance(data.get("source_indices"), list):
            data["source_indices"] = []

        return data
    except Exception:
        # fallback, если модель вернула мусор
        return {
            "has_contradiction": "не найдено",
            "contradiction_summary": "",
            "legal_basis": "",
            "recommendation": (
                "Не удалось корректно разобрать ответ модели. "
                "Фрагмент считается условно не содержащим явного противоречия."
            ),
            "source_indices": [],
        }
