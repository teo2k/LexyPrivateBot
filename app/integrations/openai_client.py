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
    Новый формат анализа: модель должна заполнить табличную схему анализа
    юридического фрагмента.

    Модель может использовать только переданные нормы и обязана вернуть
    JSON следующего вида:

    {
      "project_text": "... ошибочный тезис или цитата ...",
      "legal_reference": "... конкретная норма кодекса/Пленума/Обзора ...",
      "conclusion": "... юридическая квалификация ...",
      "source_indices": [0, 2, ...]
    }
    """

    client = get_client()

    # Формируем список норм с индексами
    norms_lines = []
    for i, n in enumerate(norms):
        norms_lines.append(
            f"[{i}] {n.get('type')} {n.get('number')}: "
            f"{n.get('short_title')} — {n.get('summary')}"
        )
    norms_text = "\n".join(norms_lines) if norms_lines else "нет доступных норм"

    # ---------------------------
    # НОВЫЙ ПРОМПТ
    # ---------------------------

    system_msg = (
        "Ты юридический аналитик. Твоя задача — анализировать фрагменты документа "
        "и заполнять табличную схему, используемую для юридической экспертизы проектов актов.\n\n"

        "Полный алгоритм, которому ты должен строго следовать:\n\n"
        "Шаг 1 — Подготовка инструмента анализа:\n"
        "Создаётся таблица с 4 колонками:\n"
        "1) № п/п (не нужно выводить в JSON)\n"
        "2) Проект Постановления (Ошибочный тезис)\n"
        "3) Действующая норма / Позиция ВС РФ\n"
        "4) Вывод\n\n"

        "Источники для проверки — только переданные нормы:\n"
        "- Налоговый кодекс РФ\n"
        "- ГПК / АПК / КАС\n"
        "- Постановления Пленума ВС РФ\n"
        "- Обзоры судебной практики ВС РФ\n"
        "Ты НЕ имеешь права ссылаться на источники вне списка.\n\n"

        "Шаг 2 — Анализ фрагмента текста:\n"
        "Разбивай смысловой фрагмент на элементы (если нужно) и проверяй:\n"
        "- соответствие нормам кодексов,\n"
        "- соответствие практике ВС РФ,\n"
        "- полноту отражения нормы,\n"
        "- отсутствие внутренних противоречий.\n\n"

        "Шаг 3 — Заполнение таблицы:\n"
        "Колонка «Проект Постановления»:\n"
        "- процитируй ошибочный тезис ИЛИ кратко сформулируй проблему.\n\n"
        "Колонка «Действующая норма / Позиция ВС РФ»:\n"
        "- укажи конкретную статью кодекса, или номер пункта Пленума, или номер Обзора.\n\n"
        "Колонка «Вывод»:\n"
        "- Краткая квалификация:\n"
        "  «Прямое противоречие норме НК РФ»\n"
        "  «Несоответствие практике ВС РФ»\n"
        "  «Неполное толкование нормы»\n"
        "  «Вводящее в заблуждение разъяснение»\n\n"

        "Шаг 4 — Контроль качества:\n"
        "- корректность цитирования,\n"
        "- актуальность нормы,\n"
        "- логичность вывода,\n"
        "- отсутствие повторов.\n\n"

        "Шаг 5 — Ранжирование по значимости (используй для корректного вывода):\n"
        "1) прямые противоречия нормам кодексов,\n"
        "2) противоречия практике ВС РФ,\n"
        "3) неполнота регулирования,\n"
        "4) технические неточности.\n\n"

        "Теперь главное: отвечай СТРОГО В ФОРМАТЕ JSON, без пояснений.\n"
        "Формат JSON, который ты обязан выдать:\n\n"
        "{\n"
        '  "project_text": "ошибочный тезис или цитата",\n'
        '  "legal_reference": "статья/пункт кодекса или Пленума/Обзора, строго из списка источников",\n'
        '  "conclusion": "краткая юридическая квалификация",\n'
        '  "source_indices": [0, 2, ...]\n'
        "}\n"
        "Где source_indices — индексы норм из предоставленного списка.\n"
        "Если подходящих норм нет — верни корректный JSON, но legal_reference = \"не найдено\", "
        "conclusion = \"источников недостаточно для выявления противоречия\"."
    )

    user_msg = (
        "Фрагмент документа для анализа:\n"
        "-----------------\n"
        f"{fragment_text}\n"
        "-----------------\n\n"
        "Доступные нормы (единственные допустимые источники):\n"
        f"{norms_text}\n\n"
        "Заполни таблицу по алгоритму выше и верни JSON-ответ."
    )

    # ---------------------------
    # ВЫЗОВ LLM
    # ---------------------------
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )

    content = resp.choices[0].message.content or "{}"

    # ---------------------------
    # Валидация JSON
    # ---------------------------
    try:
        data = json.loads(content)

        if not isinstance(data, dict):
            raise ValueError("JSON is not an object")

        data.setdefault("project_text", "")
        data.setdefault("legal_reference", "не найдено")
        data.setdefault("conclusion", "")
        if not isinstance(data.get("source_indices"), list):
            data["source_indices"] = []

        return data

    except Exception:
        # fallback-ответ
        return {
            "project_text": fragment_text[:100] + "...",
            "legal_reference": "не найдено",
            "conclusion": "Ошибка парсинга ответа модели. "
                          "Источников недостаточно для выявления противоречия.",
            "source_indices": [],
        }
