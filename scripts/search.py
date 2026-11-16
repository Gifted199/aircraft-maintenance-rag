"""Search the index from the command line and print the top results.

    uv run python scripts/search.py "main wheel axle nut torque" --aircraft A-200
    uv run python scripts/search.py "A200-32-4112" --mode keyword
    uv run python scripts/search.py --stats

Hybrid and vector modes make one small paid embedding call per search (a tiny fraction of a cent).
"""

import argparse
import sys

from aero_rag.clients import AzureEmbedder, make_azure_openai, make_opensearch
from aero_rag.config import get_settings
from aero_rag.retrieval.search import Searcher, SearchMode


def print_stats(client, index: str) -> None:
    response = client.search(
        index=index,
        body={
            "size": 0,
            "track_total_hits": True,
            "aggs": {
                "by_type": {
                    "terms": {"field": "chunk_type"},
                    "aggs": {"by_latest": {"terms": {"field": "is_latest"}}},
                }
            },
        },
    )
    print(f"{'chunk_type':<12}{'is_latest=true':>16}{'is_latest=false':>17}{'total':>8}")
    for bucket in sorted(response["aggregations"]["by_type"]["buckets"], key=lambda b: b["key"]):
        latest = {b["key_as_string"]: b["doc_count"] for b in bucket["by_latest"]["buckets"]}
        print(
            f"{bucket['key']:<12}{latest.get('true', 0):>16}{latest.get('false', 0):>17}"
            f"{bucket['doc_count']:>8}"
        )
    print(f"{'all':<12}{'':>16}{'':>17}{response['hits']['total']['value']:>8}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the manuals index.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--aircraft", help="A-200, B-340 or C-75; detected if absent")
    parser.add_argument("--mode", choices=[m.value for m in SearchMode], default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--include-superseded", action="store_true", help="evaluation only")
    parser.add_argument("--stats", action="store_true", help="document counts in the index")
    args = parser.parse_args()

    settings = get_settings()
    client = make_opensearch(settings)
    if args.stats:
        print_stats(client, settings.opensearch_index)
        return 0
    if not args.query:
        parser.error("a query is required unless --stats is given")

    embedder = AzureEmbedder(make_azure_openai(settings), settings.azure_openai_embed_deployment)
    searcher = Searcher(
        client, embedder, settings.opensearch_index, settings.opensearch_search_pipeline
    )
    result = searcher.search(
        args.query, args.aircraft, SearchMode(args.mode), args.k, args.include_superseded
    )

    aircraft = "all"
    if result.aircraft_type:
        aircraft = f"{result.aircraft_type} ({result.aircraft_source})"
    print(f'Query: "{result.query}"')
    print(
        f"Mode: {result.mode.value} | aircraft filter: {aircraft} | "
        f"aircraft_unspecified: {result.aircraft_unspecified} | "
        f"include_superseded: {result.include_superseded}"
    )
    print(
        f"{'#':<3}{'score':>7}{'cosine':>8}  {'aircraft':<9}{'task':<10}{'rev':>3}  "
        f"{'latest':<7}{'matched':<11}title"
    )
    for rank, hit in enumerate(result.hits, start=1):
        cosine = f"{hit.cosine:.3f}" if hit.cosine is not None else "-"
        print(
            f"{rank:<3}{hit.score:>7.3f}{cosine:>8}  {hit.aircraft_type:<9}{hit.task_number:<10}"
            f"{hit.revision:>3}  {str(hit.is_latest):<7}{hit.matched_chunk_type:<11}{hit.title}"
        )
    if not result.hits:
        print("(no results)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
