"""Hierarchical, structure-aware chunking: one parent and two children per task revision.

  parent     the full task. Not embedded and not searched; it is what the LLM reads.
  reference  header details, tools, parts table, torque table. Searched.
  procedure  every numbered step with its WARNINGs and CAUTIONs. Searched. Never split.

Small, focused children match a question precisely ("what torque" hits the reference child, "how do
I" hits the procedure child); returning the whole parent then guarantees the model sees the complete
task, including every WARNING. Chunks are built from the parsed structure, not by counting
characters, so a step cannot be cut away from its warning.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel

from aero_rag.models import ManualDoc

CHILD_TYPES = ("reference", "procedure")
MAX_EMBEDDING_TOKENS = 8000  # text-embedding-3-small accepts 8,191 tokens per input


class ChunkTooLargeError(ValueError):
    pass


class Chunk(BaseModel):
    id: str
    parent_id: str
    chunk_type: Literal["parent", "reference", "procedure"]
    text: str
    aircraft_type: str
    ata_chapter: str
    task_number: str
    title: str
    revision: int
    revision_date: date
    is_latest: bool
    source_key: str

    def to_source(self) -> dict:
        """The document body stored in OpenSearch (the id is sent separately)."""
        return self.model_dump(mode="json", exclude={"id"})


def estimate_tokens(text: str) -> int:
    """Rough rule of thumb for English text: about 4 characters per token."""
    return len(text) // 4 + 1


def header_line(doc: ManualDoc) -> str:
    """One line that makes a chunk understandable on its own."""
    return (
        f"{doc.aircraft_type} | Task {doc.task_number} | {doc.title} | Revision {doc.revision}"
    )


def reference_body(doc: ManualDoc) -> str:
    supersedes = f"Revision {doc.supersedes}" if doc.supersedes else "None (initial issue)"
    lines = [
        f"ATA chapter: {doc.ata_chapter} - {doc.ata_title}",
        f"Revision date: {doc.revision_date.isoformat()}",
        f"Supersedes: {supersedes}",
        f"Revision summary: {doc.change_summary}",
        "",
        "Tools and equipment:",
        *[f"- {tool}" for tool in doc.tools],
        "",
        "Parts and materials:",
        *[
            f"- P/N {p.part_number}: {p.description}. Quantity {p.quantity}. Used in step {p.step}."
            for p in doc.parts
        ],
        "",
        "Torque values:",
        *[f"- {t.fastener}: {t.text} (step {t.step})" for t in doc.torques],
    ]
    return "\n".join(lines)


def procedure_body(doc: ManualDoc) -> str:
    blocks = []
    for step in doc.steps:
        lines = [f"Step {step.number}"]
        lines += [f"WARNING: {warning}" for warning in step.warnings]
        lines += [f"CAUTION: {caution}" for caution in step.cautions]
        lines.append(step.text)
        blocks.append("\n".join(lines))
    return "Procedure:\n\n" + "\n\n".join(blocks)


def build_chunks(doc: ManualDoc, is_latest: bool, source_key: str) -> list[Chunk]:
    """Parent first, then the reference and procedure children."""
    header = header_line(doc)
    reference, procedure = reference_body(doc), procedure_body(doc)
    texts = {
        "parent": f"{header}\n\n{reference}\n\n{procedure}",
        "reference": f"{header}\n\n{reference}",
        "procedure": f"{header}\n\n{procedure}",
    }
    for chunk_type in CHILD_TYPES:
        if estimate_tokens(texts[chunk_type]) > MAX_EMBEDDING_TOKENS:
            # A procedure is never split, so an oversize one stops the job instead.
            raise ChunkTooLargeError(f"{doc.doc_id} {chunk_type} is too large to embed whole")
    common = {
        "parent_id": doc.doc_id,
        "aircraft_type": doc.aircraft_type,
        "ata_chapter": doc.ata_chapter,
        "task_number": doc.task_number,
        "title": doc.title,
        "revision": doc.revision,
        "revision_date": doc.revision_date,
        "is_latest": is_latest,
        "source_key": source_key,
    }
    # Deterministic ids: re-running ingestion overwrites the same documents.
    return [
        Chunk(
            id=doc.doc_id if chunk_type == "parent" else f"{doc.doc_id}#{chunk_type}",
            chunk_type=chunk_type,
            text=text,
            **common,
        )
        for chunk_type, text in texts.items()
    ]
