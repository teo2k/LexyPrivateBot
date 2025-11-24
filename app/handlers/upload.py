from aiogram import Router, F
from aiogram.types import Message

from app.services.file_loader import save_document
from app.services.analyzer import run_full_analysis
from app.services.formatter import format_document_analysis
from app.utils.text import split_text_for_telegram  # 👈 НОВОЕ

router = Router(name="upload")


@router.message(F.document)
async def handle_document_upload(message: Message) -> None:
    if not message.document:
        return

    await message.answer("Файл получил, начинаю проверку...")

    bot = message.bot
    user_id = message.from_user.id if message.from_user else 0

    # 1. скачиваем файл
    file_path = await save_document(
        bot=bot,
        document=message.document,
        user_id=user_id,
    )

    # 2. запускаем анализ
    analysis = await run_full_analysis(file_path=file_path, topic="госпошлина")

    # 3. форматируем результат
    text = format_document_analysis(analysis)

    # 4. режем на части и отправляем по очереди
    for part in split_text_for_telegram(text):
        await message.answer(part)
