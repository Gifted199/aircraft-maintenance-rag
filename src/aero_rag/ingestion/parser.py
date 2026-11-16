"""Parse one manual Markdown file back into a `ManualDoc`.

Structure-aware: it reads the header block, the tables and each step block (with the WARNINGs and
CAUTIONs that belong to it) rather than treating the file as flat text. It is strict on purpose:
anything unexpected raises `ManualParseError`, because indexing a half-understood maintenance
procedure is worse than failing the ingestion job.
"""

import re

from aero_rag.models import ManualDoc

TITLE = re.compile(r"^## TASK (\S+): (.+)$")
HEADER_FIELD = re.compile(r"^- \*\*(.+?):\*\* (.+)$")
SECTION = re.compile(r"^### (.+)$")
STEP = re.compile(r"^#### Step (\d+)$")
ALERT = re.compile(r"^> \*\*(WARNING|CAUTION):\*\* (.+)$")
ATA = re.compile(r"^(\d{2}) - (.+)$")
SUPERSEDES = re.compile(r"^Revision (\d+)$")
TORQUE = re.compile(r"^(\d+) (\S+)$")

HEADER_FIELDS = (
    "Aircraft type",
    "ATA chapter",
    "Task number",
    "Revision",
    "Revision date",
    "Supersedes",
)
SECTIONS = (
    "Revision summary",
    "Tools and equipment",
    "Parts and materials",
    "Torque values",
    "Procedure",
)


class ManualParseError(ValueError):
    pass


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_rows(lines: list[str], width: int) -> list[list[str]]:
    rows = [_cells(line) for line in lines if line.startswith("|")]
    if len(rows) < 2 or any(len(row) != width for row in rows):
        raise ManualParseError("malformed table")
    return rows[2:]  # drop the header row and the |---| separator


def parse_manual(text: str) -> ManualDoc:
    title_match = None
    header: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if section := SECTION.match(line):
            name = section.group(1)
            if name not in SECTIONS or name in sections:
                raise ManualParseError(f"unexpected section {name!r}")
            current = sections.setdefault(name, [])
        elif current is not None:
            current.append(line)
        elif match := TITLE.match(line):
            title_match = match
        elif match := HEADER_FIELD.match(line):
            header[match.group(1)] = match.group(2)
        elif ALERT.match(line) or STEP.match(line):
            raise ManualParseError("WARNING, CAUTION or step found outside the Procedure section")

    if title_match is None:
        raise ManualParseError("no '## TASK <number>: <title>' heading")
    if missing := [f for f in HEADER_FIELDS if f not in header] + [
        s for s in SECTIONS if s not in sections
    ]:
        raise ManualParseError(f"missing: {', '.join(missing)}")
    if header["Task number"] != title_match.group(1):
        raise ManualParseError("task number in the heading and the header block differ")
    ata = ATA.match(header["ATA chapter"])
    if ata is None:
        raise ManualParseError("ATA chapter must look like '32 - Landing Gear'")
    supersedes = SUPERSEDES.match(header["Supersedes"])

    for name in ("Tools and equipment", "Parts and materials", "Torque values"):
        if any(ALERT.match(line) or STEP.match(line) for line in sections[name]):
            raise ManualParseError("WARNING, CAUTION or step found outside the Procedure section")

    torques = []
    for fastener, torque, step in _table_rows(sections["Torque values"], 3):
        match = TORQUE.match(torque)
        if match is None:
            raise ManualParseError(f"cannot read torque {torque!r}")
        torques.append(
            {"fastener": fastener, "value": match.group(1), "unit": match.group(2), "step": step}
        )

    try:
        return ManualDoc(
            aircraft_type=header["Aircraft type"],
            ata_chapter=ata.group(1),
            ata_title=ata.group(2),
            task_number=header["Task number"],
            title=title_match.group(2),
            revision=header["Revision"],
            revision_date=header["Revision date"],
            supersedes=int(supersedes.group(1)) if supersedes else None,
            change_summary=" ".join(line for line in sections["Revision summary"] if line),
            tools=[line[2:] for line in sections["Tools and equipment"] if line.startswith("- ")],
            parts=[
                {"part_number": pn, "description": desc, "quantity": qty, "step": step}
                for pn, desc, qty, step in _table_rows(sections["Parts and materials"], 4)
            ],
            torques=torques,
            steps=_parse_steps(sections["Procedure"]),
        )
    except ManualParseError:
        raise
    except ValueError as exc:  # pydantic validation, e.g. a torque that is not in its step
        raise ManualParseError(str(exc)) from exc


def _parse_steps(lines: list[str]) -> list[dict]:
    steps: list[dict] = []
    step: dict | None = None
    for line in lines:
        if line == "---":  # end-of-task rule
            break
        if not line:
            continue
        if match := STEP.match(line):
            step = {"number": int(match.group(1)), "warnings": [], "cautions": [], "text": None}
            steps.append(step)
        elif step is None:
            raise ManualParseError("content in the Procedure section before the first step")
        elif match := ALERT.match(line):
            if step["text"] is not None:
                raise ManualParseError(f"step {step['number']}: alert after the step text")
            key = "warnings" if match.group(1) == "WARNING" else "cautions"
            step[key].append(match.group(2))
        elif step["text"] is None:
            step["text"] = line
        else:
            raise ManualParseError(f"step {step['number']} has more than one line of text")
    if not steps or any(s["text"] is None for s in steps):
        raise ManualParseError("a step has no text")
    return steps
