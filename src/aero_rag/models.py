"""Data models for the manual spec and for a single manual document.

`Spec` is the shape of data/spec/manuals.json, the single source of truth. `ManualDoc` is one task
at one revision: the unit that is rendered to a Markdown file, parsed back at ingestion, and (in
phase 2) turned into one chunk. Using the same model on both sides is what lets the tests prove
that every document parses back to exactly its spec entry.
"""

from collections.abc import Iterator
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


def _check_single_line(*values: str) -> None:
    for value in values:
        if not value or value != value.strip() or "\n" in value or "|" in value:
            raise ValueError(f"text must be one trimmed line without '|': {value!r}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Step(_Model):
    number: int
    text: str
    warnings: list[str] = []  # danger to people; rendered immediately before the step text
    cautions: list[str] = []  # danger to equipment


class Part(_Model):
    part_number: str
    description: str
    quantity: int
    step: int  # step in which the part is installed


class Torque(_Model):
    fastener: str
    value: int
    unit: Literal["Nm", "in-lb"]
    step: int  # step in which the torque is applied

    @property
    def text(self) -> str:
        """Exactly how the value is written in the procedure, e.g. '135 Nm'."""
        return f"{self.value} {self.unit}"


class _Content(_Model):
    """The body shared by a spec revision and a rendered document."""

    revision: int
    revision_date: date
    change_summary: str
    tools: list[str]
    parts: list[Part]
    torques: list[Torque]
    steps: list[Step]

    @model_validator(mode="after")
    def _consistent(self):
        if [s.number for s in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError("steps must be numbered 1..n in order")
        by_number = {s.number: s for s in self.steps}
        _check_single_line(self.change_summary, *self.tools)
        for step in self.steps:
            _check_single_line(step.text, *step.warnings, *step.cautions)
        # A torque or part listed against a step must literally appear in that step's text, so
        # the summary tables can never disagree with the procedure.
        for torque in self.torques:
            _check_single_line(torque.fastener)
            if torque.step not in by_number or torque.text not in by_number[torque.step].text:
                raise ValueError(f"torque {torque.text!r} is not in the text of step {torque.step}")
        for part in self.parts:
            _check_single_line(part.part_number, part.description)
            if part.step not in by_number or part.part_number not in by_number[part.step].text:
                raise ValueError(f"part {part.part_number} is not in the text of step {part.step}")
        return self


class Revision(_Content):
    pass


class ManualDoc(_Content):
    """One task at one revision, with its identifying header."""

    aircraft_type: str
    ata_chapter: str
    ata_title: str
    task_number: str
    title: str
    supersedes: int | None  # previous revision number, or None for an initial issue

    @property
    def doc_id(self) -> str:
        return f"{self.aircraft_type}_{self.task_number}_rev{self.revision:02d}"

    @property
    def filename(self) -> str:
        return f"{self.doc_id}.md"


class Task(_Model):
    aircraft_type: str
    ata_chapter: str
    task_number: str
    title: str
    revisions: list[Revision]  # oldest first; the last one is the latest

    @model_validator(mode="after")
    def _ordered(self):
        _check_single_line(self.title)
        if not self.revisions:
            raise ValueError(f"task {self.task_number} has no revisions")
        if not self.task_number.startswith(f"{self.ata_chapter}-"):
            raise ValueError(f"task {self.task_number} is not in ATA chapter {self.ata_chapter}")
        for older, newer in zip(self.revisions, self.revisions[1:], strict=False):
            if newer.revision != older.revision + 1 or newer.revision_date <= older.revision_date:
                raise ValueError(f"task {self.task_number}: revisions out of order")
        return self

    @property
    def latest(self) -> Revision:
        return self.revisions[-1]


class Aircraft(_Model):
    aircraft_type: str
    description: str
    part_number_prefix: str


class Spec(_Model):
    spec_version: int
    notice: str
    aircraft: list[Aircraft]
    ata_chapters: dict[str, str]
    tasks: list[Task]

    @model_validator(mode="after")
    def _references(self):
        prefixes = {a.aircraft_type: a.part_number_prefix for a in self.aircraft}
        seen = set()
        for task in self.tasks:
            key = (task.aircraft_type, task.task_number)
            if key in seen:
                raise ValueError(f"duplicate task {key}")
            seen.add(key)
            if task.aircraft_type not in prefixes or task.ata_chapter not in self.ata_chapters:
                raise ValueError(f"task {key} refers to an unknown aircraft or ATA chapter")
            for revision in task.revisions:
                for part in revision.parts:
                    if not part.part_number.startswith(prefixes[task.aircraft_type] + "-"):
                        raise ValueError(f"{part.part_number} has the wrong prefix for {key}")
        return self

    def documents(self) -> Iterator[tuple[ManualDoc, bool]]:
        """Every task revision as a document, with whether it is the latest revision.

        `is_latest` is deliberately not a field of the document: a printed manual page does not
        know it has been superseded. Ingestion works it out by comparing revisions.
        """
        for task in self.tasks:
            for revision in task.revisions:
                doc = ManualDoc(
                    aircraft_type=task.aircraft_type,
                    ata_chapter=task.ata_chapter,
                    ata_title=self.ata_chapters[task.ata_chapter],
                    task_number=task.task_number,
                    title=task.title,
                    supersedes=revision.revision - 1 if revision.revision > 1 else None,
                    **revision.model_dump(),
                )
                yield doc, revision is task.latest
