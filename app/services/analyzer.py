from __future__ import annotations

from pathlib import Path
from typing import List

from app.models.analysis import (
    DocumentAnalysis,
    FragmentAnalysis,
    RiskLabel,
    SourceRef,
)
from app.services.text_extractor import extract_text
from app.services.splitter import split_into_fragments
from app.services.topic_filter import filter_fragments_by_topic
from app.services.rag_search import find_relevant_norms, NormItem
from app.integrations.openai_client import analyze_fragment_with_norms


async def run_full_analysis(file_path: Path, topic: str) -> DocumentAnalysis:
    """
    Полный пайплайн анализа документа:

      1) Извлечение текста из файла.
      2) Разбиение текста на фрагменты.
      3) Фильтрация фрагментов по теме (сейчас: 'госпошлина').
      4) Для каждого фрагмента по теме:
         - поиск релевантных норм в Pinecone (RAG),
         - анализ через LLM (OK / Риск + комментарий + корректная позиция),
         - маппинг выбранных источников по индексам.

    Возвращает DocumentAnalysis, который потом форматируется для Telegram.
    """
    # 1. Извлекаем текст
    raw_text = extract_text(file_path)

    # 2. Режем на фрагменты
    fragments_text: List[str] = split_into_fragments(raw_text)

    # 3. Фильтруем по теме
    duty_fragments: List[str] = filter_fragments_by_topic(fragments_text, topic=topic)

    fragments_models: List[FragmentAnalysis] = []

    # Случай 1: вообще не смогли вытащить текст
    if not fragments_text:
        fragments_models.append(
            FragmentAnalysis(
                fragment_text="Не удалось извлечь текст из документа.",
                label=RiskLabel.ok,
                comment="Проверьте формат файла или попробуйте другой документ.",
                correct_position="Для анализа нужен текстовый docx/pdf (не скан без OCR).",
                sources=[
                    SourceRef(
                        type="Доктрина",
                        number="N/A",
                        short_title="Технический комментарий бота",
                        url=None,
                    )
                ],
            )
        )
        return DocumentAnalysis(topic=topic, fragments=fragments_models)

    # Случай 2: текст есть, но по теме ничего нет
    if not duty_fragments:
        fragments_models.append(
            FragmentAnalysis(
                fragment_text=(
                    "В документе не найдено фрагментов, связанных с темой "
                    f"«{topic}»."
                ),
                label=RiskLabel.ok,
                comment="Бот не нашёл упоминаний госпошлины и связанных с ней конструкций.",
                correct_position=(
                    "Чтобы провести анализ, добавьте в документ блоки про размер, "
                    "уплату, льготы или распределение государственной пошлины."
                ),
                sources=[
                    SourceRef(
                        type="Доктрина",
                        number="N/A",
                        short_title="Внутренняя логика бота (фильтр по теме)",
                        url=None,
                    )
                ],
            )
        )
        return DocumentAnalysis(topic=topic, fragments=fragments_models)

    # Случай 3: есть фрагменты по теме — анализируем каждый
    for idx, frag_text in enumerate(duty_fragments[:5], start=1):
        # 4.1. Ищем релевантные нормы в Pinecone
        norms: List[NormItem] = await find_relevant_norms(frag_text, k=5)

        # Приводим к простому dict-формату для LLM
        norms_for_llm = [
            {
                "type": n.type,
                "number": n.number,
                "short_title": n.short_title,
                "summary": getattr(n, "summary", ""),
            }
            for n in norms
        ]

        # 4.2. LLM-анализ: OK / Риск + комментарий + корректная позиция + индексы источников
        llm_result = await analyze_fragment_with_norms(
            fragment_text=frag_text,
            norms=norms_for_llm,
        )

        # 4.3. Маппим label
        label_str = (llm_result.get("label") or "OK").strip()
        if label_str.upper() == "RISK" or label_str == "Риск":
            label = RiskLabel.risk
        elif label_str.upper() == "OK" or label_str == "ОК":
            label = RiskLabel.ok
        else:
            label = RiskLabel.ok

        comment = llm_result.get("comment") or ""
        correct_position = llm_result.get("correct_position") or ""

        # 4.4. Источники: берём только те, индексы которых вернула модель
        idx_list = llm_result.get("source_indices") or []
        sources: List[SourceRef] = []

        for i in idx_list:
            try:
                i_int = int(i)
                if 0 <= i_int < len(norms):
                    n: NormItem = norms[i_int]
                    sources.append(
                        SourceRef(
                            type=n.type,
                            number=n.number,
                            short_title=n.short_title,
                            url=None,  # url можно добавить в метадату Pinecone, если понадобится
                        )
                    )
            except Exception:
                continue

        # Если модель не выбрала ничего – подставим все найденные нормы, чтобы источники были обязательно
        if not sources:
            sources = [
                SourceRef(
                    type=n.type,
                    number=n.number,
                    short_title=n.short_title,
                    url=None,
                )
                for n in norms
            ] or [
                SourceRef(
                    type="Доктрина",
                    number="N/A",
                    short_title="Источники не указаны моделью",
                    url=None,
                )
            ]
        # Убираем дубликаты источников
        unique = {}
        for s in sources:
            key = (s.type, s.number, s.short_title, s.url)
            if key not in unique:
                unique[key] = s
        sources = list(unique.values())

        # 4.5. Собираем результат по фрагменту
        fragments_models.append(
            FragmentAnalysis(
                fragment_text=frag_text,  # чистый текст фрагмента
                label=label,
                comment=comment,
                correct_position=correct_position,
                sources=sources,
            )
        )

    return DocumentAnalysis(topic=topic, fragments=fragments_models)
