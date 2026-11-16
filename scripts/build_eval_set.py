"""Write data/eval/questions.json from the hand-written questions and the spec (offline, free).

    uv run python scripts/build_eval_set.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

from aero_rag.docgen.render import load_spec
from aero_rag.evaluation.dataset import build_items

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval" / "questions.json"


def render(items) -> str:
    return json.dumps([i.model_dump(exclude_defaults=False) for i in items], indent=2) + "\n"


def main() -> int:
    items = build_items(load_spec(ROOT / "data" / "spec" / "manuals.json"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(items), encoding="utf-8", newline="\n")
    counts = Counter(i.category for i in items)
    print(f"{len(items)} questions: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"trap questions (task has a superseded revision): {sum(i.is_trap for i in items)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
