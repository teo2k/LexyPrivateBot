# LexyPrivateBot

Телеграм-бот для автоматического анализа правовых документов, связанных с госпошлиной.

## Возможности
- Извлекает текст из PDF/DOCX
- Разделяет на смысловые фрагменты
- Выделяет фрагменты по теме "госпошлина"
- Делает RAG-поиск по базе норм (ПП ВС РФ, КС РФ, доктрина)
- Анализирует риски через OpenAI LLM
- Отображает "OK" / "Риск" с источниками

## Технологии
- Aiogram 3
- OpenAI API
- Pinecone Vector DB
- Pydantic v2 + pydantic-settings
- PyPDF2
- python-docx

## Запуск
1. Создайте `.env`:
OPENAI_API_KEY=...
PINECONE_API_KEY=...
BOT_TOKEN=...
PINECONE_INDEX_NAME=lexy-legal-norms
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

2. Установите зависимости:
pip install -r requirements.txt

3. Запустите бота:
python bot.py

4. Индексация базы знаний:
python -m scripts.index_knowledge
