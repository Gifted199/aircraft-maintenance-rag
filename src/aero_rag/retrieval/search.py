"""Search: keyword (BM25), vector (k-NN) or hybrid, on child chunks, collapsed to parent tasks.

Filters (latest revision, aircraft type) are placed inside each sub-query. Filtering afterwards
would let superseded or wrong-aircraft chunks take up places in the top results and then be thrown
away, leaving fewer than k good ones; filtering inside means they are never candidates at all.
"""

import math
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from aero_rag.ingestion.chunker import CHILD_TYPES
from aero_rag.retrieval.aircraft import resolve_aircraft

# Each parent has two children, so we fetch several children per wanted parent before collapsing.
CHILDREN_PER_RESULT = 4


class SearchMode(StrEnum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


class SearchHit(BaseModel):
    parent_id: str
    score: float  # mode-specific: BM25, k-NN score, or normalised hybrid score (0..1)
    cosine: float | None  # raw cosine similarity of the best child; None in keyword mode
    matched_chunk_type: str  # which child matched best: "reference" or "procedure"
    aircraft_type: str
    ata_chapter: str
    task_number: str
    title: str
    revision: int
    revision_date: date
    is_latest: bool
    source_key: str  # S3 key of the original document, for citations
    text: str  # the full parent task: this is what goes to the LLM


class SearchResult(BaseModel):
    query: str
    mode: SearchMode
    aircraft_type: str | None  # the filter that was applied, if any
    aircraft_source: str  # "given", "detected" or "none"
    aircraft_unspecified: bool  # True when every aircraft was searched
    include_superseded: bool
    hits: list[SearchHit]


def build_filters(aircraft_type: str | None, include_superseded: bool) -> list[dict]:
    filters: list[dict] = [{"terms": {"chunk_type": list(CHILD_TYPES)}}]
    if not include_superseded:
        filters.append({"term": {"is_latest": True}})
    if aircraft_type:
        filters.append({"term": {"aircraft_type": aircraft_type}})
    return filters


def build_query(
    query: str, vector: list[float] | None, mode: SearchMode, filters: list[dict], size: int
) -> dict:
    keyword = {"bool": {"must": [{"match": {"text": {"query": query}}}], "filter": filters}}
    if mode is SearchMode.KEYWORD:
        return {"size": size, "_source": {"excludes": ["embedding"]}, "query": keyword}
    # The filter sits inside the knn clause so it is applied while the graph is searched.
    knn_filter = {"bool": {"filter": filters}}
    knn = {"knn": {"embedding": {"vector": vector, "k": size, "filter": knn_filter}}}
    if mode is SearchMode.VECTOR:
        return {"size": size, "query": knn}
    # Sub-query order (keyword, vector) must match HYBRID_WEIGHTS in the search pipeline.
    return {"size": size, "query": {"hybrid": {"queries": [keyword, knn]}}}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class Searcher:
    def __init__(self, client: Any, embedder: Any, index: str, pipeline: str):
        self._client = client
        self._embedder = embedder
        self._index = index
        self._pipeline = pipeline

    def search(
        self,
        query: str,
        aircraft_type: str | None = None,
        mode: SearchMode = SearchMode.HYBRID,
        k: int = 5,
        include_superseded: bool = False,  # evaluation only: lets old revisions compete
    ) -> SearchResult:
        mode = SearchMode(mode)
        aircraft, source = resolve_aircraft(query, aircraft_type)

        vector = None if mode is SearchMode.KEYWORD else self._embedder.embed([query])[0]
        filters = build_filters(aircraft, include_superseded)
        body = build_query(query, vector, mode, filters, size=k * CHILDREN_PER_RESULT)
        params = {"search_pipeline": self._pipeline} if mode is SearchMode.HYBRID else {}
        response = self._client.search(index=self._index, body=body, params=params)

        # Collapse children to parents in Python (the hybrid query in OpenSearch 2.19 does not
        # support field collapsing): hits arrive best first, so keep the first child per parent.
        best: dict[str, dict] = {}
        for hit in response["hits"]["hits"]:
            best.setdefault(hit["_source"]["parent_id"], hit)
        top = list(best.items())[:k]

        parents = {}
        if top:
            found = self._client.mget(index=self._index, body={"ids": [pid for pid, _ in top]})
            parents = {d["_id"]: d["_source"] for d in found["docs"] if d.get("found")}

        hits = []
        for parent_id, child in top:
            source_doc = child["_source"]
            embedding = source_doc.get("embedding")
            hits.append(
                SearchHit(
                    parent_id=parent_id,
                    score=child["_score"],
                    cosine=cosine_similarity(vector, embedding) if vector and embedding else None,
                    matched_chunk_type=source_doc["chunk_type"],
                    text=parents[parent_id]["text"],
                    **{
                        field: source_doc[field]
                        for field in (
                            "aircraft_type",
                            "ata_chapter",
                            "task_number",
                            "title",
                            "revision",
                            "revision_date",
                            "is_latest",
                            "source_key",
                        )
                    },
                )
            )
        return SearchResult(
            query=query,
            mode=mode,
            aircraft_type=aircraft,
            aircraft_source=source,
            aircraft_unspecified=aircraft is None,
            include_superseded=include_superseded,
            hits=hits,
        )
