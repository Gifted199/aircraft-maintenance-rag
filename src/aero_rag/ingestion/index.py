"""The OpenSearch index, the hybrid search pipeline, and bulk indexing."""

import logging
from typing import Any

from aero_rag.ingestion.chunker import Chunk

log = logging.getLogger(__name__)

# Order matters: it must match the order of the sub-queries in the hybrid query (keyword, vector).
HYBRID_WEIGHTS = (0.5, 0.5)


class BulkIndexError(RuntimeError):
    pass


def index_body(embedding_dim: int) -> dict:
    return {
        "settings": {
            "index": {
                "knn": True,  # enables the k-NN (vector) data structures for this index
                "number_of_shards": 1,  # tiny data set; one shard keeps BM25 statistics exact
                "number_of_replicas": 0,  # single-node domain: a replica could never be placed
            }
        },
        "mappings": {
            "dynamic": "strict",  # an unexpected field is an error, not a silent new column
            "properties": {
                "text": {"type": "text"},  # analysed for BM25 keyword search
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",  # graph index: fast approximate nearest neighbours
                        "space_type": "cosinesimil",
                        "engine": "lucene",  # supports filtering during the vector search
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
                # keyword = exact match only: used for filters, never analysed
                "aircraft_type": {"type": "keyword"},
                "ata_chapter": {"type": "keyword"},
                "task_number": {"type": "keyword"},
                "chunk_type": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "title": {"type": "keyword"},
                "source_key": {"type": "keyword"},
                "revision": {"type": "integer"},
                "revision_date": {"type": "date", "format": "strict_date"},
                "is_latest": {"type": "boolean"},
            },
        },
    }


def pipeline_body() -> dict:
    """BM25 scores are unbounded and cosine scores are 0..1, so they cannot simply be added.
    Min-max rescales each list to 0..1; the weighted mean then combines them."""
    return {
        "description": "Hybrid BM25 + k-NN: min-max normalisation, weighted arithmetic mean",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": list(HYBRID_WEIGHTS)},
                    },
                }
            }
        ],
    }


def ensure_index(client: Any, index: str, embedding_dim: int, recreate: bool = False) -> bool:
    """Create the index if it is missing. Returns True if it was created."""
    exists = client.indices.exists(index=index)
    if exists and recreate:
        client.indices.delete(index=index)
        exists = False
    if not exists:
        client.indices.create(index=index, body=index_body(embedding_dim))
        log.info("index created", extra={"index": index})
        return True
    return False


def put_search_pipeline(client: Any, name: str) -> None:
    """PUT is idempotent: it creates the pipeline or replaces it with the same definition."""
    client.search_pipeline.put(id=name, body=pipeline_body())


def bulk_index(
    client: Any, index: str, chunks: list[Chunk], vectors: dict[str, list[float]]
) -> int:
    """Index every chunk in one bulk request. `vectors` maps child chunk id -> embedding."""
    body: list[dict] = []
    for chunk in chunks:
        source = chunk.to_source()
        if chunk.id in vectors:
            source["embedding"] = vectors[chunk.id]
        # "index" with an explicit _id means create-or-replace, which is what makes re-runs safe.
        body.append({"index": {"_index": index, "_id": chunk.id}})
        body.append(source)
    response = client.bulk(body=body)
    if response.get("errors"):
        failed = [
            item["index"].get("error", {}).get("type", "unknown")
            for item in response.get("items", [])
            if item.get("index", {}).get("error")
        ]
        raise BulkIndexError(f"{len(failed)} documents failed: {sorted(set(failed))}")
    client.indices.refresh(index=index)  # make the documents searchable immediately
    return len(chunks)
