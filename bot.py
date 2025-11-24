import asyncio

from app.bot_factory import create_bot, create_dispatcher


async def main() -> None:
    bot = create_bot()
    dp = create_dispatcher()

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
