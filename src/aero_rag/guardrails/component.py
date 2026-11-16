"""Component guardrail: is the cited task about the component the question asks about?

Similar components share almost every word ("nose wheel axle nut" and "main wheel axle nut"), so
search ranks the wrong one first and the model happily quotes its value. The value really is in the
cited task, so the part number and torque checks cannot catch it.

The deterministic part of the check covers the common case: a position word. Words in each group
below name different physical components. If the question uses one word of a group and the task
title uses another word of the same group but not the question's, the task is about a different
component. The check reads the task *title* only, because task text often mentions neighbouring
components (the main wheel task may say "the nose gear is chocked").

It does not cover different components with no shared position word (for example "engine oil
filter" vs "hydraulic return filter"); the prompt rule handles those, and evaluation measures it.
"""

import re

from aero_rag.retrieval.search import SearchHit

QUALIFIER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("nose", "main"),
    ("left", "right"),
    ("inboard", "outboard"),
    ("forward", "aft"),
    ("upper", "lower"),
)
# "nosewheel", "maingear": split the compounds technicians often write as one word.
_COMPOUND = re.compile(r"\b(nose|main)(wheel|gear)s?\b")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", _COMPOUND.sub(r"\1 \2", text.lower())))


def component_mismatch(question: str, hit: SearchHit) -> tuple[str, str] | None:
    """(word in the question, conflicting word in the task title), or None if no conflict."""
    asked, title = _words(question), _words(hit.title)
    for group in QUALIFIER_GROUPS:
        in_question = [w for w in group if w in asked]
        in_title = [w for w in group if w in title]
        if in_question and in_title and not set(in_question) & set(in_title):
            return in_question[0], in_title[0]
    return None


def mismatch_message(question: str, hit: SearchHit) -> str | None:
    """A code-built explanation for an abstention, naming the component the manual does cover.
    Built only from fixed words and index fields, so it is safe to show."""
    conflict = component_mismatch(question, hit)
    if conflict is None:
        return None
    asked, covered = conflict
    return (
        f"The manuals do not cover this for the {asked}; the closest task is for the {covered}: "
        f"{hit.aircraft_type} task {hit.task_number} revision {hit.revision}, "
        f'"{hit.title}".'
    )
