"""Phase 5: the labelled set is correct, the metrics compute what they claim, and the whole
evaluation runs offline with fakes."""

import json
from collections import Counter
from pathlib import Path

import pytest

from aero_rag.evaluation.dataset import build_items
from aero_rag.evaluation.evaluate import FILTER_OFF, FILTER_ON, evaluate
from aero_rag.evaluation.metrics import (
    SpecIndex,
    recall_at_k,
    score_item,
    step_completeness,
    top1,
)
from aero_rag.evaluation.report import write_reports
from aero_rag.evaluation.runner import CachingEmbedder, estimate
from aero_rag.guardrails.input import check_input
from aero_rag.ingestion.chunker import parse_procedure
from aero_rag.retrieval.aircraft import detect_aircraft
from aero_rag.retrieval.search import Searcher
from aero_rag.schemas import AnswerStep, LLMAnswer
from tests.fakes import FakeEmbedder
from tests.test_ingestion import DIM, INDEX, PIPELINE, ingest
from tests.test_manuals import ROOT, SPEC

ITEMS = build_items(SPEC)
BY_ID = {i.id: i for i in ITEMS}
INDEX_DOC_IDS = {doc.doc_id: latest for doc, latest in SPEC.documents()}
SPEC_INDEX = SpecIndex(SPEC)


# ------------------------------------------------------------------ the labelled set


def test_questions_file_is_exactly_what_the_spec_builds():
    on_disk = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))
    assert on_disk == [i.model_dump() for i in ITEMS]


def test_size_and_mix_meet_the_brief():
    counts = Counter(i.category for i in ITEMS)
    assert len(ITEMS) >= 35
    assert counts["identifier"] >= 3
    assert counts["unanswerable"] >= 6
    kinds = Counter(i.unanswerable_kind for i in ITEMS if i.unanswerable_kind)
    assert kinds["off_topic"] >= 2 and kinds["in_domain"] >= 2
    assert counts["clarification"] >= 3 and counts["procedure"] >= 3
    assert counts["no_aircraft"] >= 3
    assert len({i.id for i in ITEMS}) == len(ITEMS)
    assert len({i.question for i in ITEMS}) == len(ITEMS)


def test_the_brief_example_questions_are_included():
    questions = {i.question for i in ITEMS}
    assert "What is the torque for the main wheel axle nut on the A-200?" in questions
    assert "What is the torque for the main wheel axle nut?" in questions


@pytest.mark.parametrize("item", [i for i in ITEMS if i.expected_status == "answered"],
                         ids=lambda i: i.id)  # fmt: skip
def test_answerable_labels_point_at_a_latest_task_that_contains_the_values(item):
    (parent_id,) = item.relevant_parent_ids
    assert INDEX_DOC_IDS[parent_id] is True  # it exists and it is the latest revision
    latest = SPEC_INDEX.latest(item.target_aircraft, item.target_task_number)
    assert latest.revision == item.target_revision
    values = {t.text for t in latest.torques} | {p.part_number for p in latest.parts}
    assert set(item.expected_torques) | set(item.expected_parts) <= values
    assert item.expected_torques or item.expected_parts or item.expected_steps
    if item.expected_steps:
        assert item.expected_steps == len(latest.steps)
    assert not set(item.superseded_values) & values


def test_aircraft_in_the_question_matches_the_label():
    for item in ITEMS:
        if item.category in ("torque", "part_number", "procedure") and not item.aircraft_type:
            assert detect_aircraft(item.question) == item.target_aircraft, item.id
        if item.category in ("no_aircraft", "clarification") or item.id in ("I01", "I02", "I03"):
            assert detect_aircraft(item.question) is None, item.id


def test_identifier_questions_quote_a_part_or_task_number():
    for item in (i for i in ITEMS if i.category == "identifier"):
        assert any(p in item.question for p in item.expected_parts) or (
            item.target_task_number in item.question
        ), item.id


def test_superseded_values_come_from_the_spec():
    assert BY_ID["T01"].superseded_values == ["120 Nm", "A200-32-4109"]
    assert BY_ID["P03"].superseded_values == ["B340-28-5208"]
    assert BY_ID["T05"].superseded_values == []
    assert BY_ID["T01"].is_trap and not BY_ID["T05"].is_trap


def test_clarification_items_list_every_aircraft_with_the_task():
    assert BY_ID["C01"].relevant_parent_ids == [
        "A-200_32-41-01_rev03", "B-340_32-41-01_rev05", "C-75_32-41-01_rev02",
    ]  # fmt: skip


def test_only_the_injection_item_is_rejected_by_the_input_guardrail():
    rejected = [i.id for i in ITEMS if check_input(i.question, 500)]
    assert rejected == ["U09"] and BY_ID["U09"].expected_status == "rejected"


def test_estimate_counts_calls_before_any_paid_run():
    e = estimate(ITEMS, 500)
    answerable = sum(i.expected_status == "answered" for i in ITEMS)
    assert e["llm_calls_main_run"] == answerable + 4  # plus 4 in-domain unanswerable
    assert e["llm_calls_filter_off_run"] == sum(i.is_trap for i in ITEMS)
    assert e["llm_calls_max"] == 2 * e["llm_calls_expected"]
    assert e["embedding_calls"] == len(ITEMS) - 1  # the injection item never reaches search
    assert 0 < e["llm_cost_expected_usd"] < 1


# ------------------------------------------------------------------ metric definitions


def test_recall_and_top1():
    assert recall_at_k(["a", "b", "c"], ["c"], 5) == 1.0
    assert recall_at_k(["a", "b", "c", "d", "e", "f"], ["f"], 5) == 0.0
    assert recall_at_k(["a", "x", "b"], ["a", "b", "c"], 5) == pytest.approx(2 / 3)
    assert top1(["a"], ["a"]) and not top1(["b", "a"], ["a"]) and not top1([], ["a"])


def _steps(item_id):
    item = BY_ID[item_id]
    return SPEC_INDEX.latest(item.target_aircraft, item.target_task_number).steps


def test_step_completeness_counts_text_and_alerts():
    expected = _steps("S05")
    perfect = [s.model_dump() for s in expected]
    assert step_completeness(perfect, expected)[:2] == (10, 10)
    missing_warning = [dict(s) for s in perfect]
    missing_warning[6]["warnings"] = []
    ok, total, alerts_ok, alerts_total = step_completeness(missing_warning, expected)
    assert (ok, total) == (9, 10) and alerts_ok == alerts_total - 1
    assert step_completeness(perfect[:5], expected)[:2] == (5, 10)


def response(answer, status="answered", cited=(("A-200", "32-41-01", 3),), steps=(), **extra):
    return {
        "status": status, "answer": answer, "steps": list(steps), "warnings": [],
        "citations": [{"aircraft_type": a, "task_number": t, "revision": r, "title": "",
                       "revision_date": "2026-01-01", "source_key": ""} for a, t, r in cited],
        "scores": {"hybrid": 0.9, "cosine": 0.7, "exact_identifier_match": False},
        "clarification_options": [], "aircraft_unspecified": False, **extra,
    }  # fmt: skip


def test_a_correct_answer_scores_fully_correct():
    row = score_item(BY_ID["T01"], {"response": response("Final torque 135 N·m.")}, SPEC_INDEX)
    assert row["torques_found"] == 1 and row["target_cited"] and not row["outdated"]
    assert row["fully_correct"]


def test_an_old_value_or_old_citation_counts_as_outdated():
    old_value = score_item(BY_ID["T01"], {"response": response("It is 120 Nm.")}, SPEC_INDEX)
    assert old_value["outdated"] and old_value["torques_found"] == 0
    old_cite = response("It is 135 Nm.", cited=[("A-200", "32-41-01", 2)])
    row = score_item(BY_ID["T01"], {"response": old_cite}, SPEC_INDEX)
    assert row["outdated"] and not row["target_cited"] and not row["fully_correct"]


def test_a_value_from_another_aircraft_is_flagged():
    row = score_item(BY_ID["T01"], {"response": response("It is 190 Nm.")}, SPEC_INDEX)
    assert row["unexpected_values"] == ["190 Nm"] and not row["fully_correct"]


def test_abstaining_on_an_answerable_question_misses_its_values():
    row = score_item(BY_ID["T01"], {"response": response("I don't know.", "abstained", cited=())},
                     SPEC_INDEX)  # fmt: skip
    assert row["torques_found"] == 0 and not row["status_correct"] and not row["fully_correct"]


def test_clarification_must_offer_exactly_the_right_aircraft():
    good = response("Which aircraft?", "clarification_needed", cited=(),
                    clarification_options=["A-200", "B-340", "C-75"])  # fmt: skip
    del good["clarification_options"]
    good["clarification_options"] = ["C-75", "A-200", "B-340"]
    assert score_item(BY_ID["C01"], {"response": good}, SPEC_INDEX)["fully_correct"]
    good["clarification_options"] = ["A-200", "B-340"]
    assert not score_item(BY_ID["C01"], {"response": good}, SPEC_INDEX)["fully_correct"]


def test_an_errored_question_is_scored_as_wrong_not_dropped():
    row = score_item(BY_ID["T01"], {"error_type": "TimeoutError", "llm_calls": 0}, SPEC_INDEX)
    assert row["status"] == "error" and not row["fully_correct"]


def test_caching_embedder_embeds_each_text_once():
    inner = FakeEmbedder(DIM)
    cache = CachingEmbedder(inner)
    cache.embed(["a", "b"])
    cache.embed(["b", "a", "c"])
    assert inner.embedded == ["a", "b", "c"] and cache.texts_embedded == 3


# ------------------------------------------------------------------ the whole run, offline


class ReferenceCopyingGenerator:
    """A well-behaved fake model: quotes source 1's reference section, or its full procedure."""

    def __init__(self):
        self.calls = 0

    def generate(self, question, context, problems=None):
        from aero_rag.generation.generator import Generation

        self.calls += 1
        hit = context[0]
        wants_steps = any(w in question.lower() for w in ("step", "procedure", "how do i"))
        steps = []
        if wants_steps:
            steps = [AnswerStep(**s.model_dump()) for s in parse_procedure(hit.text)]
        reference = hit.text.split("Procedure:")[0]
        answer = LLMAnswer(status="answered", answer=reference, steps=steps, source_ids=[1])
        return Generation(answer=answer, prompt_tokens=1000, completion_tokens=50)


@pytest.fixture(scope="module")
def offline_results():
    _, client, _ = ingest()
    embedder = CachingEmbedder(FakeEmbedder(DIM))
    searcher = Searcher(client, embedder, INDEX, PIPELINE)
    results = evaluate(
        ITEMS, SPEC, searcher, ReferenceCopyingGenerator(), embedder,
        cosine_threshold=0.0, max_question_chars=500,
        meta={"index": INDEX, "chat_deployment": "fake"},
    )  # fmt: skip
    return results, client


def test_offline_run_produces_every_table(offline_results, tmp_path: Path):
    results, _ = offline_results
    paths = write_reports(tmp_path, results)
    text = paths[0].read_text(encoding="utf-8")
    for heading in ("Headline metrics", "By question type", "Recall@5 by search mode",
                    "top-1 accuracy", "Latest-revision filter: off vs on", "Cost of this run"):
        assert heading in text  # fmt: skip
    assert len(results["rows"]) == len(ITEMS)
    groups = {r["group"] for r in results["retrieval_modes"]}
    assert {"torque", "identifier", "clarification", "all"} <= groups
    assert json.loads(paths[2].read_text(encoding="utf-8"))["meta"]["questions"] == len(ITEMS)


def test_offline_run_filter_on_never_cites_an_old_revision(offline_results):
    results, _ = offline_results
    on, off = results["filter_e2e"]
    assert on["setting"] == FILTER_ON and on["final_outdated"] == 0
    assert results["filter_retrieval"][FILTER_ON]["superseded_in_context"] == 0
    assert results["summary"]["outdated"] == 0
    # With the filter off, old revisions do reach the model's context in the fake index too.
    assert off["setting"] == FILTER_OFF
    assert results["filter_retrieval"][FILTER_OFF]["superseded_in_context"] > 0


def test_offline_run_never_writes_to_the_index(offline_results):
    results, client = offline_results
    assert len(client.indexes[INDEX]) == 93  # unchanged after the evaluation
    assert results["summary"]["questions"] == len(ITEMS)


def test_offline_run_clarifies_and_abstains_where_expected(offline_results):
    results, _ = offline_results
    rows = {r["id"]: r for r in results["rows"]}
    assert rows["U09"]["status"] == "rejected" and rows["U09"]["llm_calls"] == 0
    assert rows["U05"]["status"] == "abstained"  # superseded part number: identifier not found
    assert rows["C01"]["status"] == "clarification_needed" and rows["C01"]["llm_calls"] == 0
