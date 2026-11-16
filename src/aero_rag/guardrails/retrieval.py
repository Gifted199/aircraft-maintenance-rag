"""Retrieval guardrail: decide, before any LLM call, whether to answer, abstain or ask.

The rule, in order:

1. No results                                            -> abstain
2. The question contains a part number or task number:
     found verbatim in a retrieved task                  -> strong evidence, continue
     not found in any retrieved task                     -> abstain (unknown or superseded id)
3. Otherwise, cosine similarity of the top result
     below the threshold                                 -> abstain
4. Aircraft not specified, and the best results are the
   same task on different aircraft (and no part number
   settles it)                                           -> ask which aircraft
5. Otherwise                                             -> answer from the top tasks

Why two signals. Cosine similarity measures meaning, and a bare part number has almost none: the
correct task for "A200-32-4112" scores only 0.46, below many wrong answers to ordinary questions.
An exact identifier match is the opposite: meaningless for prose, decisive for codes.

Why not the hybrid score. Min-max normalisation gives the best result about 1.0 whatever the
question; "Who won the 2022 World Cup?" scores 0.81.

Threshold (measured on the live index, 26 probe questions):
    off-topic questions                  0.07 - 0.30
    in-domain but not in the manuals     0.41 - 0.52
    answerable questions                 0.42 - 0.80
0.35 is the midpoint of the gap between off-topic (max 0.30) and in-domain (min 0.41). It is set
low on purpose: its job is to stop off-topic questions for free. "In-domain but not in the manuals"
cannot be separated by score, so those go to the model, which is instructed to abstain.
"""

import re
from typing import Literal

from pydantic import BaseModel

from aero_rag.retrieval.search import SearchHit, SearchResult

PART_NUMBER = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]\d{2,3}-\d{2}-\d{4}(?![A-Za-z0-9])")
TASK_NUMBER = re.compile(r"(?<![\w-])\d{2}-\d{2}-\d{2}(?![\w-])")


class RetrievalDecision(BaseModel):
    action: Literal["answer", "abstain", "clarify"]
    reason: str
    context: list[SearchHit] = []  # the tasks to give the model, best first
    exact_identifier_match: bool = False
    options: list[str] = []  # aircraft types to choose from when action == "clarify"


def extract_part_numbers(text: str) -> set[str]:
    return {match.upper() for match in PART_NUMBER.findall(text)}


def extract_task_numbers(text: str) -> set[str]:
    return set(TASK_NUMBER.findall(text))


def identifier_sources(question: str, context: list[SearchHit]) -> dict[str, list[int]]:
    """Each part or task number in the question -> the source ids (1-based) that contain it."""
    found: dict[str, list[int]] = {}
    for ident in sorted(extract_part_numbers(question)):
        ids = [i for i, h in enumerate(context, start=1) if ident in h.text.upper()]
        if ids:
            found[ident] = ids
    for ident in sorted(extract_task_numbers(question)):
        ids = [i for i, h in enumerate(context, start=1) if h.task_number == ident]
        if ids:
            found[ident] = ids
    return found


def decide(
    question: str, result: SearchResult, cosine_threshold: float, context_k: int
) -> RetrievalDecision:
    hits = result.hits
    if not hits:
        return RetrievalDecision(action="abstain", reason="no_results")

    parts, tasks = extract_part_numbers(question), extract_task_numbers(question)
    part_hits = [h for h in hits if any(p in h.text.upper() for p in parts)]
    exact_hits = [h for h in hits if h in part_hits or h.task_number in tasks]

    if (parts or tasks) and not exact_hits:
        return RetrievalDecision(action="abstain", reason="identifier_not_found")
    if not exact_hits and (hits[0].cosine or 0.0) < cosine_threshold:
        return RetrievalDecision(action="abstain", reason="low_similarity")

    candidates = (exact_hits or hits)[:context_k]
    decision = {"context": candidates, "exact_identifier_match": bool(exact_hits)}

    # A part number belongs to one aircraft, so it settles the question by itself.
    if result.aircraft_unspecified and not part_hits:
        same_task = [h for h in candidates if h.task_number == candidates[0].task_number]
        aircraft = sorted({h.aircraft_type for h in same_task})
        if len(aircraft) > 1:
            return RetrievalDecision(
                action="clarify", reason="same_task_on_several_aircraft", options=aircraft,
                **decision,
            )  # fmt: skip
    return RetrievalDecision(action="answer", reason="ok", **decision)
