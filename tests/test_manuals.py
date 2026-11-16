"""Phase 1: the spec is sound, the documents match it, and every document parses back to it."""

import re
from collections import defaultdict
from pathlib import Path

import pytest

from aero_rag.docgen.render import load_spec, render, render_all
from aero_rag.ingestion.parser import ManualParseError, parse_manual

ROOT = Path(__file__).resolve().parents[1]
MANUALS_DIR = ROOT / "data" / "manuals"

SPEC = load_spec(ROOT / "data" / "spec" / "manuals.json")
DOCS = [doc for doc, _ in SPEC.documents()]
STEP_HEADING = re.compile(r"^#### Step (\d+)$", re.MULTILINE)


def by_key(task):
    return task.aircraft_type, task.task_number


def values(revision):
    """The facts an answer could get wrong: torque values and part numbers."""
    torques = {(t.fastener, t.text) for t in revision.torques}
    parts = {p.part_number for p in revision.parts}
    return torques, parts


# ------------------------------------------------------------------ the spec itself


def test_three_aircraft_with_about_eight_tasks_each():
    per_aircraft = defaultdict(list)
    for task in SPEC.tasks:
        per_aircraft[task.aircraft_type].append(task)
    assert set(per_aircraft) == {"A-200", "B-340", "C-75"}
    assert all(len(tasks) == 8 for tasks in per_aircraft.values())
    assert all(len({t.ata_chapter for t in tasks}) >= 5 for tasks in per_aircraft.values())


def test_brief_example_questions_exist_as_tasks():
    tasks = {by_key(t): t for t in SPEC.tasks}
    axle = tasks["A-200", "32-41-01"]
    assert "Main Wheel Axle Nut" in axle.title
    assert any("final torque" in t.fastener for t in axle.latest.torques)
    sensor = tasks["A-200", "21-31-04"]
    assert "Cabin Pressure Sensor" in sensor.title


def test_between_28_and_32_documents():
    assert 28 <= len(DOCS) <= 32
    assert len({d.doc_id for d in DOCS}) == len(DOCS)


def test_at_least_six_tasks_have_a_superseded_revision_with_a_different_value():
    changed = []
    for task in SPEC.tasks:
        for old in task.revisions[:-1]:
            # Every superseded revision must be a real trap: a different torque or part number.
            assert values(old) != values(task.latest), (by_key(task), old.revision)
        if len(task.revisions) > 1:
            changed.append(by_key(task))
    assert len(changed) >= 6


def test_same_task_on_different_aircraft_has_different_values():
    """Why the aircraft-type filter matters: the wrong aircraft gives a wrong answer."""
    by_number = defaultdict(list)
    for task in SPEC.tasks:
        by_number[task.task_number].append(task)
    shared = {n: tasks for n, tasks in by_number.items() if len(tasks) == 3}
    assert len(shared) >= 5
    for number, tasks in shared.items():
        assert len({t.title for t in tasks}) == 1, number
        torque_sets = [frozenset(values(t.latest)[0]) for t in tasks]
        part_sets = [values(t.latest)[1] for t in tasks]
        assert len(set(torque_sets)) == 3, number
        assert not (part_sets[0] & part_sets[1] or part_sets[1] & part_sets[2]), number
        # Stronger: no single torque for the same fastener is shared by two aircraft.
        by_fastener = defaultdict(list)
        for task in tasks:
            for torque in task.latest.torques:
                by_fastener[torque.fastener].append(torque.text)
        for fastener, texts in by_fastener.items():
            assert len(set(texts)) == len(texts), (number, fastener)


def test_every_torque_and_part_is_written_in_the_step_it_is_listed_against():
    for doc in DOCS:
        steps = {s.number: s.text for s in doc.steps}
        for torque in doc.torques:
            assert torque.text in steps[torque.step], doc.doc_id
        for part in doc.parts:
            assert part.part_number in steps[part.step], doc.doc_id


# ------------------------------------------------------------------ the rendered documents


def test_files_on_disk_are_exactly_what_the_spec_renders():
    """Fails if someone edits the spec without regenerating, or edits a document by hand."""
    expected = render_all(SPEC)
    on_disk = {p.name: p.read_text(encoding="utf-8") for p in MANUALS_DIR.glob("*.md")}
    assert sorted(on_disk) == sorted(expected)
    for name, text in expected.items():
        assert on_disk[name] == text, name


def test_rendering_is_deterministic():
    assert render_all(SPEC) == render_all(SPEC)


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_document_parses_back_to_its_spec_entry(doc):
    assert parse_manual(render(doc)) == doc


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_header_block_is_consistent(doc):
    text = render(doc)
    supersedes = f"Revision {doc.supersedes}" if doc.supersedes else "None (initial issue)"
    for line in (
        f"- **Aircraft type:** {doc.aircraft_type}",
        f"- **ATA chapter:** {doc.ata_chapter} - {doc.ata_title}",
        f"- **Task number:** {doc.task_number}",
        f"- **Revision:** {doc.revision}",
        f"- **Revision date:** {doc.revision_date.isoformat()}",
        f"- **Supersedes:** {supersedes}",
    ):
        assert text.count(line) == 1, line


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_no_step_is_separated_from_its_warning(doc):
    """Checked on the raw text, independently of the parser: each WARNING/CAUTION lies inside the
    block of the step it belongs to, before that step's text, and nowhere else."""
    text = render(doc)
    pieces = STEP_HEADING.split(text)  # [before, "1", block1, "2", block2, ...]
    assert "**WARNING:**" not in pieces[0] and "**CAUTION:**" not in pieces[0]
    blocks = {int(n): block for n, block in zip(pieces[1::2], pieces[2::2], strict=True)}
    assert sorted(blocks) == [s.number for s in doc.steps]

    for step in doc.steps:
        block = blocks[step.number]
        assert block.count("**WARNING:**") == len(step.warnings)
        assert block.count("**CAUTION:**") == len(step.cautions)
        for alert in [*step.warnings, *step.cautions]:
            assert block.index(alert) < block.index(step.text), (doc.doc_id, step.number)
    assert any(s.warnings for s in doc.steps), "every task carries at least one WARNING"


def test_alerts_are_not_only_on_the_first_step():
    """A WARNING in the middle of a procedure is the case a naive splitter gets wrong."""
    mid = [d.doc_id for d in DOCS if any(s.warnings and s.number > 1 for s in d.steps)]
    assert len(mid) >= len(DOCS) // 2


# ------------------------------------------------------------------ the parser is strict


def test_parser_rejects_a_warning_outside_a_step():
    text = render(DOCS[0]).replace(
        "### Procedure\n", "### Procedure\n\n> **WARNING:** ORPHANED WARNING.\n", 1
    )
    with pytest.raises(ManualParseError):
        parse_manual(text)


def test_parser_rejects_a_missing_header_field():
    text = "\n".join(
        line for line in render(DOCS[0]).splitlines() if not line.startswith("- **Revision:**")
    )
    with pytest.raises(ManualParseError, match="Revision"):
        parse_manual(text)


def test_parser_rejects_a_torque_table_that_disagrees_with_the_procedure():
    doc = DOCS[0]
    torque = doc.torques[0]
    text = render(doc).replace(f"| {torque.text} |", f"| {torque.value + 1} {torque.unit} |", 1)
    with pytest.raises(ManualParseError):
        parse_manual(text)
