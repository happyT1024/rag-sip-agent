"""Embeddings with an on-disk cache: the same text is never embedded twice.

Model: ``intfloat/multilingual-e5-large`` — corpus is English RFCs, questions
come in Russian, so a cross-lingual model is required (lecture 5, mistake #7).
E5 needs the ``query:`` / ``passage:`` prefixes, they are applied here.
The cache key includes the model name, so switching models never reuses
stale vectors.

Cache: one sqlite file, key = sha256(model + prefixed text), value = float32
vector. First run fills it, every re-run is free.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_CACHE = Path("data/cache/embeddings.sqlite3")

# Known dims, so ``dim`` works before the (heavy) model is loaded.
DIMS = {DEFAULT_MODEL: 1024}


def cache_key(model: str, prefixed_text: str) -> str:
    return hashlib.sha256(f"{model}\x00{prefixed_text}".encode("utf-8")).hexdigest()


def _connect(cache_path: Path) -> sqlite3.Connection:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "model TEXT NOT NULL, key TEXT NOT NULL, vec BLOB NOT NULL, "
        "PRIMARY KEY (model, key))"
    )
    return conn


def embed_texts(
    model: str,
    prefixed: list[str],
    keys: list[str],
    encode: Callable[[list[str]], np.ndarray],
    cache_path: Path | str,
) -> np.ndarray:
    """Embed ``prefixed`` texts, filling gaps from ``encode`` and caching.

    ``encode`` receives only the texts missing from the cache and must return
    normalized float vectors. Returns an (n, dim) float32 array.
    """
    if not prefixed:
        return np.zeros((0, 0), dtype=np.float32)

    conn = _connect(Path(cache_path))
    try:
        vectors: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for i, key in enumerate(keys):
            row = conn.execute(
                "SELECT vec FROM embeddings WHERE model = ? AND key = ?",
                (model, key),
            ).fetchone()
            if row is not None:
                vectors[i] = np.frombuffer(row[0], dtype=np.float32)
            else:
                missing.append(i)

        if missing:
            encoded = encode([prefixed[i] for i in missing])
            encoded = np.asarray(encoded, dtype=np.float32)
            for j, i in enumerate(missing):
                vec = encoded[j]
                vectors[i] = vec
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (model, key, vec) VALUES (?, ?, ?)",
                    (model, keys[i], vec.tobytes()),
                )
            conn.commit()

        out = np.stack([vectors[i] for i in range(len(prefixed))])
        return out.astype(np.float32, copy=False)
    finally:
        conn.close()


class Embedder:
    """E5 embedder with lazy model loading and a sqlite-backed cache."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_path: Path | str = DEFAULT_CACHE,
    ) -> None:
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self._model = None

    @property
    def dim(self) -> int:
        if self.model_name in DIMS:
            return DIMS[self.model_name]
        self._ensure_model()
        return self._model.get_sentence_embedding_dimension()

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        return self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 128,
        )

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        keys = [cache_key(self.model_name, p) for p in prefixed]
        return embed_texts(self.model_name, prefixed, keys, self._encode, self.cache_path)

    def embed_query(self, text: str) -> np.ndarray:
        prefixed = f"query: {text}"
        key = cache_key(self.model_name, prefixed)
        return embed_texts(self.model_name, [prefixed], [key], self._encode, self.cache_path)[0]
