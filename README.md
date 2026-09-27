# RAG-агент по SIP RFC

RAG + память поверх корпуса из шести SIP-документов: RFC 3261, 3311, 3515, 3262, 3428, 2976.

## Быстрый старт

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # положить OPENROUTER_API_KEY (нужен только для генерации ответов)
```

### 1. Нарезка корпуса

```bash
.venv/bin/python -m chunker run
# → data/processed/naive.jsonl (803), data/processed/structural.jsonl (482)
```

### 2. Milvus

```bash
docker compose up -d          # Milvus Standalone (один контейнер: embedEtcd + локальный сторадж)
.venv/bin/python -m vectorstore.ingest   # коллекции rag_naive / rag_structural
```

Требуется доступ к docker (группа `docker`). Если docker недоступен — допустим
Milvus Lite: `.venv/bin/python -m vectorstore.ingest --uri ./milvus_lite.db`
(так работают и тесты; для сдачи с Lite нужна пометка в отчёте).

Эмбеддер: `intfloat/multilingual-e5-large` (1024d, cosine; e5-base был заменён после
замера — recall@5 structural вырос 0.775 → 0.850). Кэш эмбеддингов —
`data/cache/embeddings.sqlite3` (ключ включает имя модели): повторный запуск не считает
те же тексты второй раз.

### 3. Поиск

```bash
# смоук-запрос
.venv/bin/python -m vectorstore.ingest --search "Что делает метод REFER?" \
    --collection rag_structural
# демонстрация фильтра
.venv/bin/python -m vectorstore.ingest --search "Как клиент ведёт себя при INVITE?" \
    --collection rag_structural --filter 'section_path like "9.%"'
```

### 4. Вопросы и замер recall@k

`data/questions.jsonl` — 40 ответимых вопросов с цитатой `evidence` (верный кусок =
чанк, содержащий цитату) и 10 неотвечаемых «по теме, но не из корпуса».

```bash
.venv/bin/python -m retrieval.evaluate --plot reports/recall_curve.png
.venv/bin/python -m retrieval.hybrid --method structural   # вектор / слова / гибрид RRF
```

Результаты (40 вопросов, e5-large):

| коллекция | @1 | @3 | @5 | @10 | MRR |
|---|---|---|---|---|---|
| rag_naive | 0.425 | 0.725 | 0.850 | 0.900 | 0.597 |
| rag_structural | 0.575 | 0.800 | 0.850 | 0.900 | 0.707 |

Гибрид (BM25 + RRF) на этом корпусе **не** обгоняет чистый вектор
(structural @5: vector 0.850 / words 0.525 / hybrid 0.775): запросы русские,
корпус английский, лексических пересечений мало и слияние подмешивает шум —
гибрид оставлен вне основного конвейера, таблица сохранена для отчёта.
Выбор k для генерации: **k=5** — recall выходит на плато (0.850), а k=10
добавляет лишь +0.05 ценой вдвое большего шума в контексте.

### 5. Генерация и бенчмарк (тратит деньги OpenRouter)

```bash
.venv/bin/python -m benchmark.run            # 50 вопросов x 5 конфигураций + judge
.venv/bin/python -m benchmark.report         # таблицы + reports/money_chart.png
```

Модели: cheap = `openai/gpt-4o-mini`, mid = `google/gemini-2.5-flash`,
strong = `openai/gpt-4.1`; judge — cheap. Прогон обошёлся в ~$0.14 при
бюджет-стопе $1.50 (клавиша бюджета — `--budget`). Прогон чекпоинтируется
после каждого ответа (`reports/benchmark_results.jsonl`), перезапуск
продолжает, а не платит заново.

Результаты (40 ответимых + 10 неотвечаемых):

| Конфигурация | Доля верных | Цена вопроса | Цена верного ответа |
|---|---|---|---|
| сильная модель, без инструментов | 65.00% | $0.0004 | $0.0008 |
| дешёвая модель, без инструментов | 55.00% | $0.0000 | $0.0001 |
| дешёвая модель, RAG всегда | **95.00%** | $0.0003 | **$0.0004** |
| дешёвая модель, агент с knowledge_base | 90.00% | $0.0005 | $0.0008 |
| средняя модель, RAG всегда | 92.50% | $0.0008 | $0.0011 |

Тезис курса подтверждён: дешёвая модель с RAG обгоняет сильную без инструментов
и стоит вдвое дешевле за верный ответ. Отказы: RAG-конфигурации честно отклоняют
10/10 неотвечаемых (0 ложных отказов); без инструментов модель выдумывает 9-10 из 10.
Разбор провалов — `reports/failures.md`, график — `reports/money_chart.png`.

### 6. Тесты

```bash
.venv/bin/python -m pytest tests/ -q
```

Тесты гоняются на Milvus Lite с фейковым эмбеддером: без docker, без скачивания
модели, без ключа, без LLM.

### 7. Telegram-бот

```bash
.venv/bin/python -m bot
```

- Обязателен только `TELEGRAM_BOT_TOKEN` (от @BotFather) — без него бот
  завершается с понятной ошибкой.
- `AGENT_BACKEND=fake` — бесплатный живой прогон: бот отвечает заглушкой
  с цитатой, расходы $0.
- `AGENT_BACKEND=openrouter` — настоящий RAG-агент; дополнительно требуются:
  `OPENROUTER_API_KEY` в `.env`, запущенный Milvus (`docker compose up -d`)
  и загруженная коллекция (`python -m vectorstore.ingest`). Расходы
  ограничивают `SPEND_CAP_USD` (журнал бота) и `OPENROUTER_BUDGET_USD`
  (жёсткий стоп клиента на процесс).
- Лимиты и адрес Milvus (`MILVUS_URI`, `AGENT_MODEL`, бюджет) задаются
  в `.env` — см. комментарии в `.env.example`.
- Команды: `/start`, `/help` — справка; `/reset` — очищает историю
  диалога только этого чата.

## Структура

```
chunker/       нарезка PDF: naive (окно) + structural (по секциям RFC)
vectorstore/   эмбеддинги с кэшем + инжест в Milvus
retrieval/     скоринг (recall@k, MRR, RRF), BM25, замер и гибрид
agent/         агентное ядро: tool-use цикл + адаптер AgentPort для бота
bot/           Telegram-бот: middleware-конвейер, порт агента, рендеринг
journal/       локальное состояние бота (сессии, журнал расходов), в .gitignore
data/          корпус PDF; data/processed — нарезки; data/cache — кэш эмбеддингов
problem/       spec, лекции, чужие скрипты (не наш пайплайн)
```
