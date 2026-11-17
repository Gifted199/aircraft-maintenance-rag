"""Run the API:  uv run python -m aero_rag.api

Starting uvicorn from Python (rather than the `uvicorn` command) lets me install the JSON log
formatter first and tell uvicorn not to install its own, so every line on stdout, including
uvicorn's start-up messages, is JSON that CloudWatch Logs can parse.
"""

import os

import uvicorn

from aero_rag.logging_config import setup_logging


def main() -> None:
    setup_logging()
    uvicorn.run(
        "aero_rag.api.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104 - must listen on all interfaces in a container
        port=int(os.environ.get("PORT", "8000")),
        log_config=None,  # keep my JSON logging
        access_log=False,  # the app writes its own one-line-per-request log
    )


if __name__ == "__main__":
    main()
