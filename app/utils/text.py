from typing import List


TELEGRAM_MAX_MESSAGE_LEN = 4096
DEFAULT_CHUNK_SIZE = 4000


def split_text_for_telegram(
    text: str,
    max_len: int = DEFAULT_CHUNK_SIZE,
) -> List[str]:
    """
    Делит текст на части не длиннее max_len, стараясь резать по пустым строкам или пробелам.
    """
    if not text:
        return [""]

    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_len:
        # сначала пробуем резать по двойному \n\n
        split_pos = remaining.rfind("\n\n", 0, max_len)
        if split_pos == -1:
            # если нет - пробуем по одиночному \n
            split_pos = remaining.rfind("\n", 0, max_len)
        if split_pos == -1:
            # если и тут нет - по пробелу
            split_pos = remaining.rfind(" ", 0, max_len)
        if split_pos == -1:
            # вообще без разделителей - рубим жестко
            split_pos = max_len

        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks
