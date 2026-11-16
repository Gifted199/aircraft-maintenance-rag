"""Output guardrail: deterministic checks on the model's answer against the cited tasks.

These are plain code, not another model: a wrong torque value is the worst possible output, so the
check must be exact, repeatable and explainable. Each problem is a sentence that can be sent back
to the model for one retry.
"""

import re

from aero_rag.guardrails.component import component_mismatch
from aero_rag.guardrails.retrieval import PART_NUMBER
from aero_rag.ingestion.chunker import parse_procedure
from aero_rag.models import Step
from aero_rag.retrieval.search import SearchHit
from aero_rag.schemas import AnswerStep, LLMAnswer

# "135 Nm", "135 N·m", "45 in-lb", "45 in lbs", "45 lbf-in"
TORQUE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(N\s?[·.]?\s?m|in[- ]?lbs?|lbf?[·. -]?in)(?![A-Za-z])",
    re.IGNORECASE,
)


def extract_torques(text: str) -> set[str]:
    """Torque values normalised to the manuals' spelling: '135 Nm' or '45 in-lb'."""
    values = set()
    for number, unit in TORQUE.findall(text):
        values.add(f"{number} {'Nm' if unit[0] in 'Nn' else 'in-lb'}")
    return values


def _norm(text: str) -> str:
    text = re.sub(r"^\s*(WARNING|CAUTION)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _compare_steps(steps: list[AnswerStep], expected: list[Step], source_id: int) -> list[str]:
    numbers = [s.number for s in steps]
    wanted = [s.number for s in expected]
    if numbers != wanted:
        return [
            f"steps: the procedure in source {source_id} has {len(wanted)} steps "
            f"(1 to {len(wanted)}) but you returned step numbers {numbers}. "
            "Return every step, in order, with none skipped or merged."
        ]
    problems = []
    for got, want in zip(steps, expected, strict=True):
        if _norm(got.text) != _norm(want.text):
            problems.append(f"step {want.number}: the text was changed; copy it word for word.")
        for label, got_alerts, want_alerts in (
            ("WARNING", got.warnings, want.warnings),
            ("CAUTION", got.cautions, want.cautions),
        ):
            if sorted(map(_norm, got_alerts)) != sorted(map(_norm, want_alerts)):
                problems.append(
                    f"step {want.number}: its {label} entries are missing, changed or extra; "
                    f"the source has {len(want_alerts)} for this step. Copy them word for word."
                )
    return problems


def check_output(answer: LLMAnswer, context: list[SearchHit], question: str = "") -> list[str]:
    """Problems with an 'answered' response; an empty list means it passed.

    `context` is the list of tasks the model was given; source id N is context[N-1]. With the
    question given, each cited task must also be about the component the question asks about.
    """
    problems: list[str] = []
    ids = list(dict.fromkeys(answer.source_ids))
    invalid = [i for i in ids if not 1 <= i <= len(context)]
    if invalid:
        problems.append(f"source_ids {invalid} do not exist; use only ids of the given sources.")
    cited = [(i, context[i - 1]) for i in ids if 1 <= i <= len(context)]
    if not cited:
        problems.append("An answer must cite at least one source in source_ids.")
        return problems

    # Search already filters to the latest revision; this second check is defence in depth.
    for source_id, hit in cited:
        if not hit.is_latest:
            problems.append(
                f"source {source_id} is a superseded revision and must not be used or cited."
            )
        if conflict := component_mismatch(question, hit):
            asked, covered = conflict
            problems.append(
                f'source {source_id} ("{hit.title}") is about the {covered}, but the question asks '
                f"about the {asked}. Answer only about the component asked; if no source covers "
                'it, use status "cannot_answer".'
            )

    said =" ".join([answer.answer, *[s.text for s in answer.steps]])
    cited_text = " ".join(hit.text for _, hit in cited)
    cited_parts = {match.upper() for match in PART_NUMBER.findall(cited_text)}
    cited_torques = extract_torques(cited_text)
    for part in sorted({m.upper() for m in PART_NUMBER.findall(said)} - cited_parts):
        problems.append(f"part number {part} does not appear in the sources you cited.")
    for torque in sorted(extract_torques(said) - cited_torques):
        problems.append(
            f"torque value {torque} does not appear in the sources you cited. "
            "Copy values exactly and do not convert units."
        )

    if answer.steps:
        # The steps must be the complete procedure of one cited task.
        attempts = [_compare_steps(answer.steps, parse_procedure(h.text), i) for i, h in cited]
        problems += min(attempts, key=len)
    return problems
