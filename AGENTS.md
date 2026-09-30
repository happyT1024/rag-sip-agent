# AGENTS.md

Homework 2 of an LLM-agents course: RAG + memory over a corpus of SIP telephony RFCs. The assignment spec (`problem/homework02.pdf`, in Russian) is the source of truth for deliverables, grading, and the deadline — read it before adding features.

## Hard constraints

- `.env` holds a real `OPENROUTER_API_KEY`. Never commit it, echo it, or put it in fixtures — grading fails if a key appears anywhere in commit history. The repo has no commits yet and **no `.gitignore`**: create it (`.env`, Milvus data, chat journal) before the first `git add`, and never stage blindly.
- LLM calls spend real money. The spec's budget for the full 50-question × 5-config run is 30–50 cents. If spend spikes, first check that embeddings are cached on disk (re-runs must never re-embed the same texts) and that whole PDF pages aren't going into the context.
- Tests for the `knowledge_base` tool and for memory must run **without an LLM call and without a key** (course-wide skill requirement).

## Corpus and data

- `data/*.pdf` are the corpus: SIP RFCs 3261, 3311, 3515, 3262, 3428, 2976. This is the "own documents / PDF" corpus option (page-level citations in answers are a +1 bonus), not the seminar's Wikipedia corpus.
- `problem/fetch_corpus.py` is copied from the seminar repo for the *other* corpus option: it reads `../seminar01/data/fresh_2026.jsonl` — a sibling checkout outside this repo — and writes to `problem/data/`, not root `data/`. It is not part of this project's pipeline; don't run or extend it.
- Question JSONL fields (per spec / starter format): `question`, `answer`, `evidence` (quote used to score recall@k), `page`, plus 10 unanswerable questions on-topic with no answer.

## Expected build-out

- Per spec, none of this exists yet: `README.md`, `requirements.txt`, Milvus `docker-compose.yml` (same as seminar; Milvus Lite fallback allowed but must be flagged in the report), notebook, on-disk embedding cache, recall@k curve (k = 1, 3, 5, 10) for both chunkings, hybrid BM25+vector with RRF table.
- Order matters: measure recall@k **before** any generation. If the correct chunk isn't in the top-5, no model can answer — fix chunking first instead of paying for generation.
- README/report is written in Russian (course language).
