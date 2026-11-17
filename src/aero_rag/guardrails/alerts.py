"""Attach WARNINGs and CAUTIONs to short answers, in code.

A short answer such as "the final torque is 135 Nm" comes from one step of a procedure, and that
step may carry a WARNING the technician must see. The model is not trusted to repeat it: like
citations, the alert text is copied by code from the indexed task, so it is always word for word
and can never be forgotten or reworded.

A step counts as relied on if either:
  * the answer quotes a torque value or part number that the step states (decided by code), or
  * the model pointed at it in `relied_on_steps` (a pointer only; the text comes from my code).
The first rule is the safety net: it works even if the model points at nothing.
"""

from aero_rag.guardrails.output import extract_torques
from aero_rag.guardrails.retrieval import extract_part_numbers
from aero_rag.ingestion.chunker import parse_procedure
from aero_rag.retrieval.search import SearchHit
from aero_rag.schemas import LLMAnswer, StepAlert


def collect_alerts(answer: LLMAnswer, context: list[SearchHit]) -> list[StepAlert]:
    """Alerts for the steps a short answer relies on. `context[N-1]` is source id N."""
    if answer.steps:
        return []  # a full procedure already carries every alert inside its steps
    parts = extract_part_numbers(answer.answer)
    torques = extract_torques(answer.answer)
    pointed = {(ref.source_id, ref.number) for ref in answer.relied_on_steps}

    alerts: list[StepAlert] = []
    for source_id in dict.fromkeys(answer.source_ids):  # cited sources only, in order
        if not 1 <= source_id <= len(context):
            continue
        hit = context[source_id - 1]
        for step in parse_procedure(hit.text):
            relied_on = (
                (source_id, step.number) in pointed
                or any(part in step.text.upper() for part in parts)
                or bool(torques & extract_torques(step.text))
            )
            if not relied_on:
                continue
            for kind, texts in (("WARNING", step.warnings), ("CAUTION", step.cautions)):
                alerts += [
                    StepAlert(
                        kind=kind,
                        text=text,
                        aircraft_type=hit.aircraft_type,
                        task_number=hit.task_number,
                        step_number=step.number,
                    )
                    for text in texts
                ]
    return alerts
