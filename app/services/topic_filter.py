from __future__ import annotations

from typing import Iterable, List


def filter_fragments_by_topic(
    fragments: Iterable[str],
    topic: str,
) -> List[str]:
    """
    Очень простой фильтр по теме.
    На первом этапе поддерживаем только 'госпошлина':
    ищем по ключевым словам и шаблонам.
    """
    topic = topic.lower().strip()

    if topic == "госпошлина":
        return _filter_state_duty_fragments(fragments)

    # на будущее - другие темы
    return list(fragments)


def _filter_state_duty_fragments(fragments: Iterable[str]) -> List[str]:
    """
    Фильтр фрагментов по теме госпошлины.
    """
    keywords = [
        "госпошлина",
        "государственная пошлина",
        "гос. пошлина",
        "госпошлины",
        "государственной пошлины",
        "оплата пошлины",
        "уплата пошлины",
        "размер пошлины",
        "льгота по пошлине",
        "освобождение от уплаты пошлины",
        "статья 333",
        "ст. 333",
        "налоговый кодекс",
        "нк рф",
        "подпункт",
        "подп. ",
    ]

    result: list[str] = []

    for frag in fragments:
        text = frag.lower()
        if any(kw in text for kw in keywords):
            result.append(frag)

    return result
