"""Ingest processed chunk sets into Milvus: collections ``rag_naive`` / ``rag_structural``.

Works against the standalone server from ``docker-compose.yml``
(``--uri http://localhost:19530``, default) or a Milvus Lite file
(``--uri ./local.db``) — which is also what the tests use.

CLI::

    .venv/bin/python -m vectorstore.ingest                 # ingest both sets
    .venv/bin/python -m vectorstore.ingest --search "..."  # smoke query
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Protocol

from pymilvus import DataType, MilvusClient

from .embed import Embedder

COLLECTIONS = {"naive": "rag_naive", "structural": "rag_structural"}
TEXT_MAX = 8192
BATCH = 256

OUTPUT_FIELDS = [
    "chunk_id",
    "rfc",
    "title",
    "section_id",
    "section_path",
    "page_start_printed",
]


class Client(Protocol):
    """The slice of MilvusClient the ingest pipeline needs (real or fake)."""

    def create_schema(self, *args, **kwargs): ...

    def prepare_index_params(self, *args, **kwargs): ...

    def has_collection(self, *args, **kwargs): ...

    def drop_collection(self, *args, **kwargs): ...

    def create_collection(self, *args, **kwargs): ...

    def insert(self, *args, **kwargs): ...

    def flush(self, *args, **kwargs): ...

    def load_collection(self, *args, **kwargs): ...

    def get_collection_stats(self, *args, **kwargs): ...

    def search(self, *args, **kwargs): ...


class EmbedsPassages(Protocol):
    model_name: str
    dim: int

    def embed_passages(self, texts: list[str]): ...

    def embed_query(self, text: str): ...


def build_schema(client: Client, dim: int):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("method", DataType.VARCHAR, max_length=32)
    schema.add_field("rfc", DataType.INT64)
    schema.add_field("title", DataType.VARCHAR, max_length=512, nullable=True)
    schema.add_field("section_id", DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field("section_path", DataType.VARCHAR, max_length=1024, nullable=True)
    schema.add_field("page_start_printed", DataType.INT64, nullable=True)
    schema.add_field("page_end_printed", DataType.INT64, nullable=True)
    schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    return schema


def ensure_collection(client: Client, name: str, dim: int, recreate: bool = True) -> bool:
    """Create the collection (FLAT + COSINE = exact search, honest recall@k).

    Returns True when the collection was (re)created, False if it already
    existed and was left as is.
    """
    if client.has_collection(name):
        if not recreate:
            return False
        client.drop_collection(name)
    schema = build_schema(client, dim)
    index = client.prepare_index_params()
    index.add_index(
        field_name="vector",
        index_name="vec_flat",
        index_type="FLAT",
        metric_type="COSINE",
    )
    client.create_collection(collection_name=name, schema=schema, index_params=index)
    return True


def row_from_record(rec: dict, vector) -> dict:
    page_start = rec.get("page_start") or {}
    page_end = rec.get("page_end") or {}
    return {
        "chunk_id": rec["chunk_id"],
        "method": rec["method"],
        "rfc": int(rec["rfc"]),
        "title": rec.get("title"),
        "section_id": rec.get("section_id"),
        "section_path": rec.get("section_path"),
        "page_start_printed": page_start.get("printed"),
        "page_end_printed": page_end.get("printed"),
        "text": rec["text"],
        "vector": [float(x) for x in vector],
    }


def ingest_records(
    client: Client,
    embedder: EmbedsPassages,
    records: list[dict],
    collection: str,
    batch: int = BATCH,
) -> int:
    """Embed and insert records in batches. Returns the number of rows."""
    total = 0
    for start in range(0, len(records), batch):
        chunk = records[start : start + batch]
        vectors = embedder.embed_passages([r["text"] for r in chunk])
        rows = [row_from_record(r, v) for r, v in zip(chunk, vectors)]
        client.insert(collection_name=collection, data=rows)
        total += len(rows)
    try:
        client.flush(collection)
    except NotImplementedError:
        pass  # Milvus Lite persists inserts automatically
    return total


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ingest_file(client: Client, embedder: EmbedsPassages, path: Path, collection: str) -> int:
    return ingest_records(client, embedder, load_jsonl(path), collection)


def search(
    client: Client,
    embedder: EmbedsPassages,
    collection: str,
    query: str,
    limit: int = 5,
    filter_expr: str | None = None,
) -> list[dict]:
    """Top-k search with provenance fields; ``filter_expr`` for the demo."""
    client.load_collection(collection)
    qvec = embedder.embed_query(query)
    kwargs = {
        "collection_name": collection,
        "data": [list(map(float, qvec))],
        "limit": limit,
        "output_fields": OUTPUT_FIELDS + ["text"],
        "anns_field": "vector",
    }
    if filter_expr:
        kwargs["filter"] = filter_expr
    results = client.search(**kwargs)
    hits = results[0] if results else []
    return [
        {
            "chunk_id": h["entity"]["chunk_id"],
            "rfc": h["entity"]["rfc"],
            "title": h["entity"]["title"],
            "section_path": h["entity"]["section_path"],
            "page": h["entity"]["page_start_printed"],
            "distance": h["distance"],
            "text": h["entity"]["text"],
        }
        for h in hits
    ]


def run(
    data_dir: Path | str = "data/processed",
    uri: str = "http://localhost:19530",
    recreate: bool = True,
    embedder: EmbedsPassages | None = None,
    client_factory: Callable[[str], Client] = MilvusClient,
) -> dict:
    data_dir = Path(data_dir)
    client = client_factory(uri)
    embedder = embedder or Embedder()
    counts: dict[str, int] = {}
    for method, collection in COLLECTIONS.items():
        path = data_dir / f"{method}.jsonl"
        created = ensure_collection(client, collection, embedder.dim, recreate=recreate)
        n = ingest_file(client, embedder, path, collection)
        stored = client.get_collection_stats(collection)["row_count"]
        print(f"{collection}: {n} rows ingested from {path} (stored={stored}, created={created})")
        counts[collection] = n
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vectorstore.ingest")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument(
        "--uri",
        default="http://localhost:19530",
        help="Milvus endpoint; a *.db path means Milvus Lite",
    )
    parser.add_argument("--no-recreate", action="store_true")
    parser.add_argument("--search", default=None, help="smoke query after ingest")
    parser.add_argument("--collection", default="rag_structural", help="collection for --search")
    parser.add_argument("--filter", default=None, help="filter expression for --search")
    args = parser.parse_args(argv)

    if args.search is None:
        run(args.data_dir, args.uri, recreate=not args.no_recreate)
        return

    client = MilvusClient(args.uri)
    embedder = Embedder()
    for hit in search(client, embedder, args.collection, args.search, limit=5, filter_expr=args.filter):
        loc = f"§{hit['section_path']}" if hit["section_path"] else f"p.{hit['page']}"
        print(f"{hit['distance']:.4f}  {hit['chunk_id']}  [{loc}]  {hit['text'][:80]!r}")


if __name__ == "__main__":
    main()
