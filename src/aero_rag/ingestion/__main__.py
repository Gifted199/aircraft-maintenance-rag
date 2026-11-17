"""Ingestion entry point. The same command runs on a laptop and as a one-off ECS Fargate task:

    uv run python -m aero_rag.ingestion --dry-run    # counts and cost estimate; no writes, free
    uv run python -m aero_rag.ingestion              # embed and index

Output is JSON log lines on stdout (CloudWatch Logs on ECS). Exit code 0 on success, 1 on failure.
"""

import argparse
import logging
import sys

from aero_rag.clients import AzureEmbedder, make_azure_openai, make_opensearch
from aero_rag.config import aws_session, get_settings
from aero_rag.ingestion.pipeline import run_ingestion
from aero_rag.ingestion.source import S3Source
from aero_rag.logging_config import setup_logging

log = logging.getLogger("aero_rag.ingestion")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest manuals from S3 into OpenSearch.")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; write nothing")
    parser.add_argument(
        "--recreate-index", action="store_true", help="delete and recreate my index first"
    )
    args = parser.parse_args(argv)

    setup_logging()
    try:
        settings = get_settings()
        setup_logging(settings.log_level, redact=settings.secret_values())
        source = S3Source(
            aws_session(settings).client("s3"), settings.s3_bucket, settings.s3_prefix
        )
        embedder = AzureEmbedder(
            make_azure_openai(settings), settings.azure_openai_embed_deployment
        )
        run_ingestion(
            source,
            embedder,
            make_opensearch(settings),
            index=settings.opensearch_index,
            pipeline=settings.opensearch_search_pipeline,
            embedding_dim=settings.embedding_dim,
            dry_run=args.dry_run,
            recreate_index=args.recreate_index,
        )
    except Exception:
        log.exception("ingestion failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
