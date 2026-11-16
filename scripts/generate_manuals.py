"""Generate data/manuals/*.md from data/spec/manuals.json and print a summary.

Deterministic and offline: no LLM calls, no AWS calls, no cost.

    uv run python scripts/generate_manuals.py
"""

import sys
from collections import Counter
from pathlib import Path

from aero_rag.docgen.render import load_spec, write_manuals

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "data" / "spec" / "manuals.json"
OUT_DIR = ROOT / "data" / "manuals"


def main() -> int:
    spec = load_spec(SPEC_PATH)
    paths = write_manuals(spec, OUT_DIR)

    tasks, latest, superseded = Counter(), Counter(), Counter()
    for task in spec.tasks:
        tasks[task.aircraft_type] += 1
    for doc, is_latest in spec.documents():
        (latest if is_latest else superseded)[doc.aircraft_type] += 1

    print(f"{'Aircraft':<10}{'Tasks':>7}{'Latest':>8}{'Superseded':>12}{'Documents':>11}")
    for aircraft in [a.aircraft_type for a in spec.aircraft]:
        total = latest[aircraft] + superseded[aircraft]
        print(
            f"{aircraft:<10}{tasks[aircraft]:>7}{latest[aircraft]:>8}"
            f"{superseded[aircraft]:>12}{total:>11}"
        )
    print(
        f"{'Total':<10}{sum(tasks.values()):>7}{sum(latest.values()):>8}"
        f"{sum(superseded.values()):>12}{len(paths):>11}"
    )
    print(f"Wrote {len(paths)} documents to {OUT_DIR.relative_to(ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
