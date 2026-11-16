"""The ingestion job: S3 -> strict parse -> is_latest -> chunks -> embeddings -> OpenSearch.

Every external system (S3, the embedding model, OpenSearch) is passed in, so the same function
runs locally, on ECS, and in offline tests with fakes.
"""

import logging
import math
from collections import Counter
from collections.abc import Iterable
from typing import Any, Protocol

from aero_rag.ingestion.chunker import CHILD_TYPES, Chunk, build_chunks, estimate_tokens
from aero_rag.ingestion.index import bulk_index, ensure_index, put_search_pipeline
from aero_rag.ingestion.parser import ManualParseError, parse_manual
from aero_rag.models import ManualDoc

log = logging.getLogger(__name__)

# Approximate list price of text-embedding-3-small, used only for the pre-run estimate.
EMBEDDING_USD_PER_MILLION_TOKENS = 0.02


class IngestionError(RuntimeError):
    pass


class Source(Protocol):
    def iter_documents(self) -> Iterable[tuple[str, str]]: ...


class Embedder(Protocol):
    batch_size: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def load_documents(source: Source) -> list[tuple[str, ManualDoc]]:
    """Parse everything first. One bad file stops the job before anything is written."""
    loaded: list[tuple[str, ManualDoc]] = []
    seen: dict[str, str] = {}
    for key, text in source.iter_documents():
        try:
            doc = parse_manual(text)
        except ManualParseError as exc:
            raise IngestionError(f"{key}: {exc}") from exc
        if doc.doc_id in seen:
            raise IngestionError(f"{key} and {seen[doc.doc_id]} are the same task revision")
        seen[doc.doc_id] = key
        loaded.append((key, doc))
    if not loaded:
        raise IngestionError("no .md documents found")
    return loaded


def mark_latest(docs: Iterable[ManualDoc]) -> dict[str, bool]:
    """doc_id -> is_latest. The latest revision is the highest revision number among all
    documents for the same (aircraft_type, task_number)."""
    docs = list(docs)
    highest: dict[tuple[str, str], int] = {}
    for doc in docs:
        key = (doc.aircraft_type, doc.task_number)
        highest[key] = max(highest.get(key, 0), doc.revision)
    return {d.doc_id: d.revision == highest[d.aircraft_type, d.task_number] for d in docs}


def build_all_chunks(source: Source) -> list[Chunk]:
    loaded = load_documents(source)
    latest = mark_latest(doc for _, doc in loaded)
    chunks: list[Chunk] = []
    for key, doc in loaded:
        chunks += build_chunks(doc, latest[doc.doc_id], key)
    return chunks


def run_ingestion(
    source: Source,
    embedder: Embedder,
    client: Any,
    *,
    index: str,
    pipeline: str,
    embedding_dim: int,
    dry_run: bool = False,
    recreate_index: bool = False,
) -> dict:
    chunks = build_all_chunks(source)
    children = [c for c in chunks if c.chunk_type in CHILD_TYPES]
    parents = [c for c in chunks if c.chunk_type == "parent"]
    tokens = sum(estimate_tokens(c.text) for c in children)
    stats = {
        "documents": len(parents),
        "latest": sum(c.is_latest for c in parents),
        "superseded": sum(not c.is_latest for c in parents),
        "chunks": dict(Counter(c.chunk_type for c in chunks)),
        "embedding_texts": len(children),
        "embedding_calls": math.ceil(len(children) / embedder.batch_size),
        "estimated_tokens": tokens,
        "estimated_cost_usd": round(tokens / 1_000_000 * EMBEDDING_USD_PER_MILLION_TOKENS, 5),
        "dry_run": dry_run,
    }
    if dry_run:
        log.info("dry run: nothing embedded, nothing written", extra=stats)
        return stats

    # Only children are embedded: search runs on children, parents are fetched by id.
    embeddings = embedder.embed([c.text for c in children])
    if len(embeddings) != len(children) or any(len(v) != embedding_dim for v in embeddings):
        raise IngestionError("embedding count or dimension does not match the index")
    vectors = {c.id: v for c, v in zip(children, embeddings, strict=True)}

    ensure_index(client, index, embedding_dim, recreate=recreate_index)
    put_search_pipeline(client, pipeline)
    stats["indexed"] = bulk_index(client, index, chunks, vectors)
    log.info("ingestion complete", extra=stats)
    return stats
