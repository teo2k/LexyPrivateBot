from __future__ import annotations

from typing import List


# Примерные ограничения, потом можно подкрутить
MIN_FRAGMENT_LEN = 100      # минимальная длина фрагмента в символах
MAX_FRAGMENT_LEN = 800      # максимальная длина фрагмента в символах


def split_into_fragments(text: str) -> List[str]:
    """
    Делит текст на смысловые фрагменты:
    - сначала по пустым строкам (абзацы),
    - слишком длинные абзацы дополнительно режет на куски.
    """
    if not text:
        return []

    # нормализуем перевод строк
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    raw_blocks = _split_by_empty_lines(text)
    blocks: list[str] = []

    # чистим и режем длинные блоки
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        if len(block) <= MAX_FRAGMENT_LEN:
            blocks.append(block)
        else:
            blocks.extend(_split_long_block(block, max_len=MAX_FRAGMENT_LEN))

    # можно ещё склеивать слишком короткие блоки
    merged = _merge_short_blocks(blocks, min_len=MIN_FRAGMENT_LEN)

    return merged


def _split_by_empty_lines(text: str) -> List[str]:
    """
    Разбивает текст по пустым строкам (двойной перевод строки).
    """
    parts = text.split("\n\n")
    return parts


def _split_long_block(block: str, max_len: int) -> List[str]:
    """
    Делит длинный блок на куски не длиннее max_len, стараясь резать по пробелам.
    """
    result: list[str] = []

    current = block.strip()
    while len(current) > max_len:
        # ищем позицию пробела ближе к max_len
        split_pos = current.rfind(" ", 0, max_len)
        if split_pos == -1:
            # нет пробелов - рубим жестко
            split_pos = max_len

        part = current[:split_pos].strip()
        if part:
            result.append(part)
        current = current[split_pos:].strip()

    if current:
        result.append(current)

    return result


def _merge_short_blocks(blocks: List[str], min_len: int) -> List[str]:
    """
    Склеивает слишком короткие куски с соседними, чтобы не было совсем мелкой дроби.
    """
    if not blocks:
        return []

    merged: list[str] = []
    buffer = blocks[0]

    for block in blocks[1:]:
        if len(buffer) < min_len:
            # добавляем следующий блок к текущему
            buffer = buffer.rstrip() + "\n\n" + block.lstrip()
        else:
            merged.append(buffer)
            buffer = block

    if buffer:
        merged.append(buffer)

    return merged
