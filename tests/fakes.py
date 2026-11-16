"""In-memory stand-ins for S3, the embedding model and OpenSearch, so tests run offline and free.

FakeOpenSearch really applies the filters and ranks documents (crudely), so tests can check
behaviour ("a superseded revision is never returned") and not just the shape of the request.
"""

import hashlib
import io
import math
import re
from pathlib import Path

MANUALS_DIR = Path(__file__).resolve().parents[1] / "data" / "manuals"
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class FakeS3:
    """Serves files as objects under a prefix, two keys per page to exercise pagination."""

    def __init__(self, objects: dict[str, str]):
        self.objects = dict(objects)

    @classmethod
    def from_manuals(cls, prefix: str = "manuals/") -> "FakeS3":
        objects = {
            f"{prefix}{p.name}": p.read_text(encoding="utf-8") for p in MANUALS_DIR.glob("*.md")
        }
        objects[f"{prefix}README.txt"] = "not a manual"
        return cls(objects)

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str):  # noqa: N803 - boto3's argument names
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        for start in range(0, len(keys), 2):
            yield {"Contents": [{"Key": k} for k in keys[start : start + 2]]}
        if not keys:
            yield {}

    def get_object(self, Bucket: str, Key: str):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key].encode("utf-8"))}


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts that share words get similar vectors."""

    def __init__(self, dim: int = 64, batch_size: int = 64):
        self.dim = dim
        self.batch_size = batch_size
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded += texts
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokens(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


def _matches(source: dict, filters: list[dict]) -> bool:
    for clause in filters:
        if "term" in clause:
            ((field, value),) = clause["term"].items()
            if source.get(field) != value:
                return False
        elif "terms" in clause:
            ((field, allowed),) = clause["terms"].items()
            if source.get(field) not in allowed:
                return False
        else:
            raise NotImplementedError(clause)
    return True


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    return {k: 1.0 if high == low else (v - low) / (high - low) for k, v in scores.items()}


class _Indices:
    def __init__(self, parent: "FakeOpenSearch"):
        self._parent = parent

    def exists(self, index: str) -> bool:
        return index in self._parent.indexes

    def create(self, index: str, body: dict) -> None:
        self._parent.indexes[index] = {}
        self._parent.index_bodies[index] = body

    def delete(self, index: str) -> None:
        del self._parent.indexes[index]

    def refresh(self, index: str) -> None:
        self._parent.refreshed += 1


class _SearchPipeline:
    def __init__(self, parent: "FakeOpenSearch"):
        self._parent = parent

    def put(self, id: str, body: dict) -> None:  # noqa: A002 - opensearch-py's argument name
        self._parent.pipelines[id] = body


class FakeOpenSearch:
    def __init__(self):
        self.indexes: dict[str, dict[str, dict]] = {}
        self.index_bodies: dict[str, dict] = {}
        self.pipelines: dict[str, dict] = {}
        self.searches: list[dict] = []
        self.refreshed = 0
        self.indices = _Indices(self)
        self.search_pipeline = _SearchPipeline(self)

    def bulk(self, body: list[dict]) -> dict:
        for action, source in zip(body[0::2], body[1::2], strict=True):
            meta = action["index"]
            self.indexes[meta["_index"]][meta["_id"]] = source
        return {"errors": False, "items": []}

    def mget(self, index: str, body: dict) -> dict:
        docs = self.indexes[index]
        return {
            "docs": [
                {"_id": i, "found": i in docs, **({"_source": docs[i]} if i in docs else {})}
                for i in body["ids"]
            ]
        }

    def search(self, index: str, body: dict, params: dict | None = None) -> dict:
        params = params or {}
        self.searches.append({"body": body, "params": params})
        docs = self.indexes[index]
        query = body["query"]
        if "hybrid" in query:
            pipeline = self.pipelines[params["search_pipeline"]]  # KeyError = pipeline missing
            processor = pipeline["phase_results_processors"][0]["normalization-processor"]
            weights = processor["combination"]["parameters"]["weights"]
            lists = [_min_max(self._score(q, docs)) for q in query["hybrid"]["queries"]]
            scores = {
                doc_id: sum(w * s.get(doc_id, 0.0) for w, s in zip(weights, lists, strict=True))
                / sum(weights)
                for doc_id in set().union(*lists)
            }
        else:
            assert "search_pipeline" not in params
            scores = self._score(query, docs)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: body["size"]]
        excludes = body.get("_source", {}).get("excludes", [])
        return {
            "hits": {
                "hits": [
                    {
                        "_id": doc_id,
                        "_score": score,
                        "_source": {k: v for k, v in docs[doc_id].items() if k not in excludes},
                    }
                    for doc_id, score in ranked
                ]
            }
        }

    @staticmethod
    def _score(query: dict, docs: dict[str, dict]) -> dict[str, float]:
        if "bool" in query:
            wanted = set(tokens(query["bool"]["must"][0]["match"]["text"]["query"]))
            scores = {}
            for doc_id, source in docs.items():
                if _matches(source, query["bool"]["filter"]):
                    doc_tokens = tokens(source["text"])
                    score = sum(doc_tokens.count(t) for t in wanted) / math.sqrt(len(doc_tokens))
                    if score > 0:
                        scores[doc_id] = score
            return scores
        knn = query["knn"]["embedding"]
        scores = {}
        for doc_id, source in docs.items():
            if "embedding" in source and _matches(source, knn["filter"]["bool"]["filter"]):
                dot = sum(a * b for a, b in zip(knn["vector"], source["embedding"], strict=True))
                scores[doc_id] = (1 + dot) / 2
        top = sorted(scores.items(), key=lambda item: -item[1])[: knn["k"]]
        return dict(top)
