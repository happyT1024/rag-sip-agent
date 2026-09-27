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

### 5. Тесты

```bash
.venv/bin/python -m pytest tests/ -q
```

Тесты гоняются на Milvus Lite с фейковым эмбеддером: без docker, без скачивания
модели, без ключа, без LLM.

## Структура

```
chunker/       нарезка PDF: naive (окно) + structural (по секциям RFC)
vectorstore/   эмбеддинги с кэшем + инжест в Milvus
retrieval/     скоринг (recall@k, MRR, RRF), BM25, замер и гибрид
data/          корпус PDF; data/processed — нарезки; data/cache — кэш эмбеддингов
problem/       spec, лекции, чужие скрипты (не наш пайплайн)
```
