from pathlib import Path

from aiogram import Bot
from aiogram.types import Document


UPLOAD_DIR = Path("data/uploads")


async def save_document(
    bot: Bot,
    document: Document,
    user_id: int,
) -> Path:
    """
    Скачивает документ от пользователя в локальную папку и возвращает путь.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    filename = document.file_name or f"{document.file_unique_id}.bin"
    safe_filename = filename.replace("/", "_").replace("\\", "_")

    file_path = UPLOAD_DIR / f"{user_id}_{document.file_unique_id}_{safe_filename}"

    await bot.download(file=document, destination=file_path)

    return file_path
