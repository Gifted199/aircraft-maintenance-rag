"""Render manual documents from the spec. Pure string templating: deterministic, no LLM, no cost.

The Markdown layout is a contract with `aero_rag.ingestion.parser`; change them together.
"""

from pathlib import Path

from aero_rag.models import ManualDoc, Spec

NOTICE = (
    "> **SYNTHETIC TRAINING DATA - NOT FOR MAINTENANCE USE.** "
    "The {aircraft} is a fictional aircraft and all part numbers and values are invented."
)
NO_SUPERSEDES = "None (initial issue)"


def load_spec(path: Path | str) -> Spec:
    return Spec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def render(doc: ManualDoc) -> str:
    supersedes = f"Revision {doc.supersedes}" if doc.supersedes else NO_SUPERSEDES
    lines = [
        f"# {doc.aircraft_type} Aircraft Maintenance Manual",
        "",
        NOTICE.format(aircraft=doc.aircraft_type),
        "",
        f"## TASK {doc.task_number}: {doc.title}",
        "",
        f"- **Aircraft type:** {doc.aircraft_type}",
        f"- **ATA chapter:** {doc.ata_chapter} - {doc.ata_title}",
        f"- **Task number:** {doc.task_number}",
        f"- **Revision:** {doc.revision}",
        f"- **Revision date:** {doc.revision_date.isoformat()}",
        f"- **Supersedes:** {supersedes}",
        "",
        "### Revision summary",
        "",
        doc.change_summary,
        "",
        "### Tools and equipment",
        "",
        *[f"- {tool}" for tool in doc.tools],
        "",
        "### Parts and materials",
        "",
        "| Part number | Description | Quantity | Used in step |",
        "|---|---|---|---|",
        *[f"| {p.part_number} | {p.description} | {p.quantity} | {p.step} |" for p in doc.parts],
        "",
        "### Torque values",
        "",
        "| Fastener | Torque | Applied in step |",
        "|---|---|---|",
        *[f"| {t.fastener} | {t.text} | {t.step} |" for t in doc.torques],
        "",
        "### Procedure",
        "",
    ]
    for step in doc.steps:
        # WARNINGs and CAUTIONs sit inside the step block, before the action, so the reader sees
        # the hazard first and so a step can never be cut away from its warning.
        lines += [f"#### Step {step.number}", ""]
        for warning in step.warnings:
            lines += [f"> **WARNING:** {warning}", ""]
        for caution in step.cautions:
            lines += [f"> **CAUTION:** {caution}", ""]
        lines += [step.text, ""]
    lines += [
        "---",
        "",
        f"*End of task {doc.task_number}, revision {doc.revision} ({doc.aircraft_type}).*",
        "",
    ]
    return "\n".join(lines)


def render_all(spec: Spec) -> dict[str, str]:
    """filename -> Markdown text, for every task revision in the spec."""
    return {doc.filename: render(doc) for doc, _ in spec.documents()}


def write_manuals(spec: Spec, out_dir: Path | str) -> list[Path]:
    """Write every document and remove stale .md files, so the folder always mirrors the spec."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_all(spec)
    for stale in out_dir.glob("*.md"):
        if stale.name not in rendered:
            stale.unlink()
    paths = []
    for name, text in sorted(rendered.items()):
        path = out_dir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        paths.append(path)
    return paths
