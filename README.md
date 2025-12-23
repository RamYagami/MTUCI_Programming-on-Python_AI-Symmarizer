# Суммаризатор текстов, созданный в рамках обучения MTUCI MC1S1 "Программирование на Python"

## Запуск приложения

1. Загрузить зависимости.

`pip install -r requirements.txt`

`python download_models.py`

`python download_nltk.py`

2. Поднять контейнеризированное приложение (соберется, если нет образа)

`docker compose up --remove-orphans`

## Доступные эндпоинты

Получить список доступных методов и их описание можно сделав вызов `http://<address>:8000/docs`

