from aiogram import Router, F
from aiogram.types import Message
from aiogram import Bot
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.services.analyzer import run_full_analysis
from app.services.formatter import format_document_analysis
from app.utils.text import split_text_for_telegram

router = Router(name="upload")


@router.message(F.document)
async def handle_document_upload(message: Message) -> None:
    document = message.document
    if not document:
        return

    await message.answer("Файл получил, начинаю проверку...")

    bot: Bot = message.bot

    # 1. Создаем временный файл (без сохранения на диск на постоянной основе)
    suffix = Path(document.file_name).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        await message.answer("Поддерживаю только PDF и DOCX.")
        return

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        await bot.download(document, destination=tmp)

    try:
        # 2. Запускаем анализ
        analysis = await run_full_analysis(file_path=tmp_path, topic="госпошлина")

        # 3. Формируем текст ответа
        formatted = format_document_analysis(analysis)

        # 4. Режем на части и отправляем
        for part in split_text_for_telegram(formatted):
            await message.answer(part)

    finally:
        # 5. Удаляем временный файл в любом случае
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Не удалось удалить временный файл {tmp_path}: {e}")
