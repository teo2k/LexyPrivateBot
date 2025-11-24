from aiogram import Router
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

router = Router(name="start")


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    text = (
        "Привет. Я бот для проверки юридических документов.\n\n"
        "Сейчас я умею работать с темой: госпошлина.\n"
        "Отправь мне файл (docx/pdf), и я попробую найти рискованные формулировки."
    )
    await message.answer(text)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    text = (
        "Я анализирую загруженные документы и отмечаю фрагменты про госпошлину как:\n"
        " - OK\n"
        " - Риск (с кратким комментарием и ссылками на источники).\n\n"
        "Просто отправь файл документа сообщением."
    )
    await message.answer(text)
