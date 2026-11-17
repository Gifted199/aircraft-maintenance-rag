# syntax=docker/dockerfile:1
#
# One image, two jobs:
#   API (default)   docker run ... akinsds1126-aero-rag-api:0.2.0
#   Ingestion       docker run ... akinsds1126-aero-rag-api:0.2.0 python -m aero_rag.ingestion
#
# linux/amd64 is stated explicitly because that is what the Fargate task runs on. Without it, a
# build on an ARM laptop would produce an image that fails on Fargate with "exec format error".

# ---------- stage 1: build the virtual environment with uv ----------
FROM --platform=linux/amd64 python:3.12-slim AS builder

# uv is copied in as a single binary and exists only in this stage, not in the final image.
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first. This layer is rebuilt only when pyproject.toml or uv.lock changes, so a
# code-only change rebuilds in seconds. --locked fails the build if the lock file is out of date,
# so the image gets exactly the versions that were tested.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Then my code, installed into the environment as a normal (non-editable) package.
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# ---------- stage 2: the runtime image ----------
FROM --platform=linux/amd64 python:3.12-slim

# A dedicated unprivileged user: if the app is compromised, it is not root in the container.
RUN useradd --system --uid 10001 --user-group --no-create-home app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

USER app
EXPOSE 8000

# Liveness only (/health has no dependencies). python is used because slim has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

# The JSON-logging entry point; exec form so the process receives SIGTERM directly from ECS.
CMD ["python", "-m", "aero_rag.api"]
