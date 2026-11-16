"""Work out which aircraft a question is about.

The aircraft type is a hard filter, so guessing wrong is worse than not guessing: with no clear
answer we search every aircraft and say so (`aircraft_unspecified`).
"""

import re

AIRCRAFT_TYPES = ("A-200", "B-340", "C-75")


def _pattern(aircraft_type: str) -> re.Pattern:
    letter, number = aircraft_type.split("-")
    # Accept "A-200", "A200", "a 200". Reject it inside a longer code: the part number
    # "A200-32-4112" must not count as someone naming the aircraft.
    return re.compile(rf"(?<![A-Za-z0-9-]){letter}[- ]?{number}(?![A-Za-z0-9]|-\d)", re.IGNORECASE)


_PATTERNS = {aircraft: _pattern(aircraft) for aircraft in AIRCRAFT_TYPES}


def detect_aircraft(text: str) -> str | None:
    """The single aircraft type named in the text, or None if there is none or more than one."""
    found = [aircraft for aircraft, pattern in _PATTERNS.items() if pattern.search(text)]
    return found[0] if len(found) == 1 else None


def resolve_aircraft(question: str, aircraft_type: str | None) -> tuple[str | None, str]:
    """The aircraft filter for a search and where it came from: 'given', 'detected' or 'none'.
    The single place this decision is made (the search and the trace both use it)."""
    if aircraft_type:
        return normalise_aircraft(aircraft_type), "given"
    if detected := detect_aircraft(question):
        return detected, "detected"
    return None, "none"


def normalise_aircraft(value: str) -> str:
    """Turn user input such as 'a200' into the canonical 'A-200'. Raises on an unknown type."""
    aircraft = detect_aircraft(value.strip())
    if aircraft is None:
        raise ValueError(f"unknown aircraft type; expected one of {', '.join(AIRCRAFT_TYPES)}")
    return aircraft
