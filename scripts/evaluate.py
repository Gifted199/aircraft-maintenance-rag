"""Run the evaluation against the live index and model and write data/results/.

    uv run python scripts/evaluate.py --dry-run          # counts and cost estimate; free
    uv run python scripts/evaluate.py --retrieval-only   # embedding calls only
    uv run python scripts/evaluate.py                    # full run (paid: see --dry-run)

Read-only against OpenSearch: it searches, it never writes.
"""

import argparse
import json
import sys
from pathlib import Path

from aero_rag.clients import AzureEmbedder, make_azure_openai, make_opensearch
from aero_rag.config import get_settings
from aero_rag.docgen.render import load_spec
from aero_rag.evaluation.dataset import EvalItem
from aero_rag.evaluation.evaluate import evaluate
from aero_rag.evaluation.report import write_reports
from aero_rag.evaluation.runner import CachingEmbedder, estimate
from aero_rag.generation.generator import AzureGenerator
from aero_rag.logging_config import setup_logging
from aero_rag.retrieval.search import Searcher

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--dry-run", action="store_true", help="print counts and cost only")
    parser.add_argument("--retrieval-only", action="store_true", help="no LLM calls")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "results")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("WARNING", redact=settings.secret_values())
    spec = load_spec(ROOT / "data" / "spec" / "manuals.json")
    raw = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))
    items = [EvalItem(**item) for item in raw]

    if args.dry_run:
        e = estimate(items, settings.max_question_chars)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()},
                         indent=2))  # fmt: skip
        return 0

    azure = make_azure_openai(settings)
    embedder = CachingEmbedder(AzureEmbedder(azure, settings.azure_openai_embed_deployment))
    searcher = Searcher(
        make_opensearch(settings), embedder, settings.opensearch_index,
        settings.opensearch_search_pipeline,
    )  # fmt: skip
    generator = AzureGenerator(
        azure, settings.azure_openai_chat_deployment, settings.llm_max_completion_tokens
    )
    results = evaluate(
        items, spec, searcher, generator, embedder,
        cosine_threshold=settings.abstain_cosine_threshold,
        max_question_chars=settings.max_question_chars,
        meta={"index": settings.opensearch_index,
              "chat_deployment": settings.azure_openai_chat_deployment},
        end_to_end=not args.retrieval_only,
    )  # fmt: skip
    if args.retrieval_only:
        for row in results["retrieval_modes"]:
            print(row)
        return 0
    for path in write_reports(args.out, results):
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    print(json.dumps(results["summary"], indent=2))
    print(json.dumps(results["cost"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
