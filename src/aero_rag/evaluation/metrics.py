"""Metric definitions. Pure functions over recorded results, so they are tested offline.

Retrieval
  Recall@5      share of the relevant parent tasks found in the top 5 (one relevant task for most
                questions; all aircraft versions for clarification questions).
  Top-1         the first result is a relevant task.

End to end
  Exact match   an expected torque value or part number appears in the answer, written the same
                way (units normalised: "135 N·m" counts as "135 Nm"). Abstaining counts as a miss.
  Step complete share of the task's steps returned with the same number, the same text and the
                same WARNINGs and CAUTIONs.
  Outdated      an answer that cites a superseded revision, or states a value that exists only in a
                superseded revision of the task. Target: 0.
"""

import re
from statistics import mean

from aero_rag.evaluation.dataset import CATEGORY_ORDER, EvalItem
from aero_rag.guardrails.output import extract_torques
from aero_rag.guardrails.retrieval import extract_part_numbers
from aero_rag.models import Revision, Spec

ABSTAIN_STATUSES = {"abstained", "rejected"}


class SpecIndex:
    def __init__(self, spec: Spec):
        self._tasks = {(t.aircraft_type, t.task_number): t for t in spec.tasks}

    def latest(self, aircraft: str, task_number: str) -> Revision:
        return self._tasks[aircraft, task_number].latest

    def latest_revision(self, aircraft: str, task_number: str) -> int | None:
        task = self._tasks.get((aircraft, task_number))
        return task.latest.revision if task else None

    def latest_values(self, aircraft: str, task_number: str) -> set[str]:
        latest = self.latest(aircraft, task_number)
        return {t.text for t in latest.torques} | {p.part_number for p in latest.parts}


# ------------------------------------------------------------------ retrieval


def recall_at_k(ranked: list[str], relevant: list[str], k: int = 5) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def top1(ranked: list[str], relevant: list[str]) -> bool:
    return bool(ranked) and ranked[0] in relevant


def _groups(items: list[EvalItem]) -> list[tuple[str, list[EvalItem]]]:
    scored = [i for i in items if i.relevant_parent_ids]
    groups = [(c, [i for i in scored if i.category == c]) for c in CATEGORY_ORDER]
    return [(name, group) for name, group in groups if group] + [("all", scored)]


def retrieval_rows(items: list[EvalItem], runs: dict[str, dict], k: int = 5) -> list[dict]:
    """runs: label -> {item_id: {"hits": [{"parent_id": ...}, ...]}}"""
    rows = []
    for name, group in _groups(items):
        row: dict = {"group": name, "n": len(group)}
        for label, run in runs.items():
            ranked = {i.id: [h["parent_id"] for h in run[i.id]["hits"]] for i in group}
            row[f"{label} recall@{k}"] = mean(
                recall_at_k(ranked[i.id], i.relevant_parent_ids, k) for i in group
            )
            row[f"{label} top1"] = mean(top1(ranked[i.id], i.relevant_parent_ids) for i in group)
        rows.append(row)
    return rows


def superseded_in_retrieval(items: list[EvalItem], run: dict, context_k: int = 3) -> dict:
    """For trap questions: how often an old revision reaches the model's context, or ranks first."""
    traps = [i for i in items if i.is_trap]
    in_context = sum(
        any(not h["is_latest"] for h in run[i.id]["hits"][:context_k]) for i in traps
    )
    first = sum(bool(run[i.id]["hits"]) and not run[i.id]["hits"][0]["is_latest"] for i in traps)
    return {"n": len(traps), "superseded_in_context": in_context, "superseded_ranked_first": first}


# ------------------------------------------------------------------ end to end


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def response_text(response: dict) -> str:
    return " ".join([response["answer"], *[s["text"] for s in response["steps"]]])


def found_values(text: str) -> set[str]:
    return extract_torques(text) | extract_part_numbers(text)


def step_completeness(returned: list[dict], expected) -> tuple[int, int, int, int]:
    """(steps matched, steps expected, alerts matched, alerts expected)."""
    by_number = {s["number"]: s for s in returned}
    steps_ok = alerts_ok = alerts_total = 0
    for step in expected:
        got = by_number.get(step.number)
        wanted_alerts = [*step.warnings, *step.cautions]
        alerts_total += len(wanted_alerts)
        if got is None:
            continue
        got_alerts = {_norm(a) for a in [*got["warnings"], *got["cautions"]]}
        alerts_ok += sum(_norm(a) in got_alerts for a in wanted_alerts)
        same_alerts = sorted(map(_norm, got["warnings"])) == sorted(
            map(_norm, step.warnings)
        ) and sorted(map(_norm, got["cautions"])) == sorted(map(_norm, step.cautions))
        steps_ok += _norm(got["text"]) == _norm(step.text) and same_alerts
    return steps_ok, len(expected), alerts_ok, alerts_total


def is_outdated(text: str, citations: list[tuple[str, str, int]], item, index: SpecIndex) -> bool:
    stale_citation = any(index.latest_revision(a, t) != r for a, t, r in citations)
    stale_value = bool(set(item.superseded_values) & found_values(text))
    return stale_citation or stale_value


def score_item(item: EvalItem, record: dict, index: SpecIndex) -> dict:
    """One row of the per-question table."""
    response = record.get("response")
    row: dict = {
        "id": item.id,
        "category": item.category,
        "expected": item.expected_status,
        "status": response["status"] if response else "error",
        "llm_calls": record.get("llm_calls", 0),
        "retries": record.get("retries", 0),
    }
    row["status_correct"] = row["status"] == item.expected_status
    if response is None:
        return row | {"error": record.get("error_type"), "fully_correct": False}

    text = response_text(response)
    values = found_values(text)
    cited = [(c["aircraft_type"], c["task_number"], c["revision"]) for c in response["citations"]]
    row |= {
        "cosine": response["scores"]["cosine"],
        "cited": [f"{a} {t} r{r}" for a, t, r in cited],
        "torques_expected": len(item.expected_torques),
        "torques_found": sum(t in values for t in item.expected_torques),
        "parts_expected": len(item.expected_parts),
        "parts_found": sum(p in values for p in item.expected_parts),
        "warnings_attached": len(response.get("warnings", [])),
    }
    answered = row["status"] == "answered"
    if answered:
        row["outdated"] = is_outdated(text, cited, item, index)
    if item.target_aircraft:
        target = (item.target_aircraft, item.target_task_number, item.target_revision)
        row["target_cited"] = answered and target in cited
        if answered:
            allowed = index.latest_values(item.target_aircraft, item.target_task_number)
            row["unexpected_values"] = sorted(values - allowed)
        if item.expected_steps:
            expected = index.latest(item.target_aircraft, item.target_task_number).steps
            steps_ok, steps_total, alerts_ok, alerts_total = step_completeness(
                response["steps"], expected
            )
            row |= {"steps_matched": steps_ok, "steps_expected": steps_total,
                    "alerts_matched": alerts_ok, "alerts_expected": alerts_total}  # fmt: skip
    if item.expected_status == "clarification_needed":
        wanted = sorted({pid.split("_")[0] for pid in item.relevant_parent_ids})
        row["clarification_correct"] = (
            row["status"] == "clarification_needed"
            and sorted(response["clarification_options"]) == wanted
        )
    first = record.get("first_attempt")
    if first is not None:
        first_cited = [
            (c["aircraft_type"], c["task_number"], c["revision"]) for c in first["cited"]
        ]
        row["first_attempt_outdated"] = is_outdated(first["text"], first_cited, item, index)
    row["fully_correct"] = _fully_correct(item, row)
    return row


def _fully_correct(item: EvalItem, row: dict) -> bool:
    if not row["status_correct"]:
        return False
    if item.expected_status == "clarification_needed":
        return row.get("clarification_correct", False)
    if item.expected_status != "answered":
        return True
    return (
        row.get("target_cited", False)
        and not row.get("outdated", False)
        and row["torques_found"] == row["torques_expected"]
        and row["parts_found"] == row["parts_expected"]
        and row.get("steps_matched", 0) == row.get("steps_expected", 0)
    )


def summarise(items: list[EvalItem], rows: list[dict]) -> dict:
    by_id = {i.id: i for i in items}
    answerable = [r for r in rows if by_id[r["id"]].expected_status == "answered"]
    answered = [r for r in rows if r["status"] == "answered"]
    unanswerable = [r for r in rows if by_id[r["id"]].category == "unanswerable"]
    clarify = [r for r in rows if by_id[r["id"]].expected_status == "clarification_needed"]
    procedures = [r for r in rows if r.get("steps_expected")]

    def total(key, group):
        return sum(r.get(key, 0) for r in group)

    return {
        "questions": len(rows),
        "torques_found": total("torques_found", answerable),
        "torques_expected": total("torques_expected", answerable),
        "parts_found": total("parts_found", answerable),
        "parts_expected": total("parts_expected", answerable),
        "procedures": len(procedures),
        "step_completeness": mean(r["steps_matched"] / r["steps_expected"] for r in procedures)
        if procedures else None,
        "steps_matched": total("steps_matched", procedures),
        "steps_expected": total("steps_expected", procedures),
        "alerts_matched": total("alerts_matched", procedures),
        "alerts_expected": total("alerts_expected", procedures),
        "answered": len(answered),
        "outdated": sum(r.get("outdated", False) for r in answered),
        "unanswerable": len(unanswerable),
        "correct_abstentions": sum(r["status"] in ABSTAIN_STATUSES for r in unanswerable),
        "clarification_items": len(clarify),
        "correct_clarifications": sum(r.get("clarification_correct", False) for r in clarify),
        "answerable": len(answerable),
        "answerable_answered": sum(r["status"] == "answered" for r in answerable),
        "false_abstentions": sum(r["status"] in ABSTAIN_STATUSES for r in answerable),
        "false_clarifications": sum(r["status"] == "clarification_needed" for r in answerable),
        "target_cited": sum(r.get("target_cited", False) for r in answerable),
        "answers_with_unexpected_values": sum(bool(r.get("unexpected_values")) for r in answerable),
        "fully_correct": sum(r["fully_correct"] for r in rows),
        "llm_calls": total("llm_calls", rows),
        "retries": total("retries", rows),
    }  # fmt: skip


def category_rows(items: list[EvalItem], rows: list[dict]) -> list[dict]:
    by_id = {i.id: i for i in items}
    out = []
    for category in CATEGORY_ORDER:
        group = [r for r in rows if by_id[r["id"]].category == category]
        if group:
            out.append({
                "category": category,
                "n": len(group),
                "expected": ", ".join(sorted({r["expected"] for r in group})),
                "outcome_correct": sum(r["status_correct"] for r in group),
                "fully_correct": sum(r["fully_correct"] for r in group),
            })  # fmt: skip
    return out


def trap_comparison(items: list[EvalItem], rows_by_setting: dict[str, list[dict]]) -> list[dict]:
    """End-to-end outdated-revision rates on trap questions, per filter setting."""
    traps = {i.id for i in items if i.is_trap}
    out = []
    for setting, rows in rows_by_setting.items():
        group = [r for r in rows if r["id"] in traps]
        answered = [r for r in group if r["status"] == "answered"]
        out.append({
            "setting": setting,
            "n": len(group),
            "first_attempt_outdated": sum(r.get("first_attempt_outdated", False) for r in group),
            "first_attempts": sum("first_attempt_outdated" in r for r in group),
            "answered": len(answered),
            "final_outdated": sum(r.get("outdated", False) for r in answered),
            "fully_correct": sum(r["fully_correct"] for r in group),
            "retries": sum(r.get("retries", 0) for r in group),
        })  # fmt: skip
    return out
