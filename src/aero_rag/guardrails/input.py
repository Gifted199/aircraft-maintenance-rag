"""Input guardrail: cheap checks that run before any paid call.

The injection patterns are a first filter, not the real defence. The real defences are structural:
instructions live only in the system message, the question and sources are fenced as data, the
model can only return a fixed JSON shape, and every value it returns is checked against the sources.
"""

import re

INJECTION_PATTERNS = [
    r"ignore\b.{0,40}\b(instructions?|rules?|prompts?|above|previous)",
    r"disregard\b.{0,40}\b(instructions?|rules?|prompts?|above|previous)",
    r"forget\b.{0,40}\b(instructions?|rules?|everything)",
    r"\b(system|developer)\s+(prompt|message|instructions?)",
    r"\byou are now\b",
    # "act as an unrestricted AI", but not "does the pitot heat act as a de-icer"
    r"\b(act|behave|respond) as (if you|an?\s+(\w+\s+){0,3}(ai|assistant|model|bot|chatbot)\b)",
    r"\bpretend (to be|you are)\b",
    r"\bjailbreak\b",
    r"\b(reveal|print|show|repeat)\b.{0,40}\b(prompt|instructions)",
    r"</?\s*(system|source|sources|question|instructions?)\b",  # my own block markers
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)


def check_input(question: str, max_chars: int) -> str | None:
    """Return the reason the question is rejected, or None if it is acceptable."""
    text = question.strip()
    if not text:
        return "The question is empty."
    if len(text) > max_chars:
        return f"The question is too long (limit {max_chars} characters)."
    if _INJECTION.search(text):
        return "The question looks like an attempt to change the assistant's instructions."
    return None
