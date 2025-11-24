from app.models.analysis import DocumentAnalysis, RiskLabel


def format_document_analysis(analysis: DocumentAnalysis) -> str:
    lines: list[str] = [f"<b>Тема:</b> {analysis.topic}", ""]

    if not analysis.fragments:
        lines.append("Не найдено фрагментов для анализа.")
        return "\n".join(lines)

    for idx, frag in enumerate(analysis.fragments, start=1):
        lines.append(f"<b>Фрагмент {idx}</b>")
        lines.append(f"{frag.fragment_text}")
        lines.append(f"Статус: <b>{frag.label.value}</b>")

        if frag.label == RiskLabel.risk:
            lines.append(f"Комментарий: {frag.comment}")
            lines.append(f"Корректная позиция: {frag.correct_position}")

        if frag.sources:
            lines.append("Источники:")
            for src in frag.sources:
                base = f"{src.type} {src.number} - {src.short_title}"
                if src.url:
                    base += f" ({src.url})"
                lines.append(f"- {base}")

        lines.append("")  # пустая строка между фрагментами

    return "\n".join(lines)
