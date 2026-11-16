"""Assemble a full evaluation from components, compute every metric, return one results dict.

Kept separate from the command-line script so the whole flow runs offline in the tests.
"""

from datetime import UTC, datetime
from typing import Any

from aero_rag.evaluation.dataset import EvalItem
from aero_rag.evaluation.metrics import (
    SpecIndex,
    category_rows,
    retrieval_rows,
    score_item,
    summarise,
    superseded_in_retrieval,
    trap_comparison,
)
from aero_rag.evaluation.runner import (
    RecordingGenerator,
    SupersededSearcher,
    actual_cost,
    run_end_to_end,
    run_retrieval,
    trap_items,
)
from aero_rag.ingestion.index import HYBRID_WEIGHTS
from aero_rag.models import Spec
from aero_rag.retrieval.search import SearchMode
from aero_rag.service import RagService

FILTER_ON = "on (default)"
FILTER_OFF = "off (include_superseded)"


def evaluate(
    items: list[EvalItem],
    spec: Spec,
    searcher: Any,
    generator: Any,
    embedder: Any,
    *,
    cosine_threshold: float,
    max_question_chars: int,
    meta: dict | None = None,
    end_to_end: bool = True,
) -> dict:
    index = SpecIndex(spec)

    # 1. Retrieval only (embedding calls only, cached across runs).
    modes = {m.value: run_retrieval(searcher, items, m) for m in SearchMode}
    hybrid_off = run_retrieval(searcher, items, SearchMode.HYBRID, include_superseded=True)
    mode_rows = retrieval_rows(items, modes)
    results: dict = {
        "meta": {
            "run_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "questions": len(items),
            "cosine_threshold": cosine_threshold,
            "hybrid_weights": list(HYBRID_WEIGHTS),
            **(meta or {}),
        },
        "retrieval_modes": mode_rows,
        "recall5_hybrid": next(r for r in mode_rows if r["group"] == "all")["hybrid recall@5"],
        "filter_retrieval": {
            FILTER_ON: superseded_in_retrieval(items, modes[SearchMode.HYBRID.value]),
            FILTER_OFF: superseded_in_retrieval(items, hybrid_off),
        },
        "retrieval_raw": {"modes": modes, "hybrid_filter_off": hybrid_off},
    }
    if not end_to_end:
        results["texts_embedded"] = getattr(embedder, "texts_embedded", None)
        return results

    # 2. End to end with the real service: filter on (all questions), filter off (trap questions).
    def service(search):
        recorder = RecordingGenerator(generator)
        svc = RagService(
            search, recorder, cosine_threshold=cosine_threshold,
            max_question_chars=max_question_chars,
        )  # fmt: skip
        return svc, recorder

    svc_on, rec_on = service(searcher)
    records_on = run_end_to_end(svc_on, rec_on, items)
    svc_off, rec_off = service(SupersededSearcher(searcher))
    records_off = run_end_to_end(svc_off, rec_off, trap_items(items))

    by_id = {i.id: i for i in items}
    rows_on = [score_item(i, records_on[i.id], index) for i in items]
    rows_off = [score_item(by_id[k], v, index) for k, v in records_off.items()]
    summary = summarise(items, rows_on)
    all_records = [*records_on.values(), *records_off.values()]
    results |= {
        "summary": summary,
        "summary_retries_total": sum(r.get("retries", 0) for r in all_records),
        "categories": category_rows(items, rows_on),
        "rows": rows_on,
        "filter_off_rows": rows_off,
        "filter_e2e": trap_comparison(items, {FILTER_ON: rows_on, FILTER_OFF: rows_off}),
        "cost": actual_cost(all_records),
        "texts_embedded": getattr(embedder, "texts_embedded", None),
        "responses": {
            "filter_on": {k: v.get("response") for k, v in records_on.items()},
            "filter_off": {k: v.get("response") for k, v in records_off.items()},
        },
    }
    return results
