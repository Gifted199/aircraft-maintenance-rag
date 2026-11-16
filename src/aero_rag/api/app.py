"""FastAPI service.

    uv run uvicorn aero_rag.api.app:app --port 8000 --no-access-log

GET /health   liveness: the process is up. No dependencies, so a slow OpenSearch never makes the
              orchestrator kill a healthy container.
GET /ready    readiness: OpenSearch is reachable, so the service can actually answer.
POST /query   ask a question.
"""

import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from aero_rag.schemas import QueryRequest, QueryResponse
from aero_rag.service import RagService
from aero_rag.tracing import NULL_TRACER, Tracer, build_tracer

log = logging.getLogger("aero_rag.api")
APP_VERSION = "0.2.0"

OPENSEARCH_TIMEOUT_SECONDS = 10
AZURE_TIMEOUT_SECONDS = 60


def build_real_service() -> tuple[RagService, Callable[[], bool], Tracer]:
    """Create every client once, at start-up, and reuse them for all requests (connection
    pooling; no per-request credential fetch). Tracing is optional and never stops start-up."""
    from aero_rag.clients import AzureEmbedder, make_azure_openai, make_opensearch
    from aero_rag.config import get_settings
    from aero_rag.generation.generator import AzureGenerator
    from aero_rag.logging_config import setup_logging
    from aero_rag.retrieval.search import Searcher

    setup_logging()
    settings = get_settings()
    setup_logging(settings.log_level, redact=settings.secret_values())
    # Send uvicorn's own messages through the JSON formatter too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # The middleware below already writes one line per request.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    tracer = build_tracer(settings)
    opensearch = make_opensearch(settings, timeout=OPENSEARCH_TIMEOUT_SECONDS)
    azure = make_azure_openai(settings, timeout=AZURE_TIMEOUT_SECONDS)
    searcher = Searcher(
        opensearch,
        AzureEmbedder(azure, settings.azure_openai_embed_deployment),
        settings.opensearch_index,
        settings.opensearch_search_pipeline,
    )
    service = RagService(
        searcher,
        AzureGenerator(
            azure, settings.azure_openai_chat_deployment, settings.llm_max_completion_tokens
        ),
        cosine_threshold=settings.abstain_cosine_threshold,
        max_question_chars=settings.max_question_chars,
        tracer=tracer,
    )
    return service, opensearch.ping, tracer


def create_app(
    service: RagService | None = None,
    ready_check: Callable[[], bool] | None = None,
    tracer: Tracer | None = None,
) -> FastAPI:
    """Pass a service and ready_check to run with fakes; otherwise real clients are built."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            app.state.service, app.state.ready_check, app.state.tracer = build_real_service()
        else:
            app.state.service, app.state.ready_check = service, ready_check or (lambda: True)
            app.state.tracer = tracer or NULL_TRACER
        log.info("startup complete")
        yield
        # Shutdown (uvicorn runs this on SIGTERM, which is how ECS stops a task): send any
        # buffered traces before the process exits.
        app.state.tracer.shutdown()

    app = FastAPI(title="Aircraft maintenance manual assistant", version=APP_VERSION,
                  lifespan=lifespan)  # fmt: skip

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - last line of defence; never leak details
            log.error("unhandled error", extra={"request_id": request_id, "exc_type": type(exc).__name__})  # noqa: E501
            response = JSONResponse(
                {"detail": "Internal error.", "request_id": request_id}, status_code=500
            )
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        try:
            ok = bool(request.app.state.ready_check())
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            return {"status": "ready"}
        return JSONResponse({"status": "not_ready", "opensearch": "unreachable"}, status_code=503)

    # A plain `def` endpoint runs in a worker thread, so the blocking OpenSearch and Azure calls
    # do not freeze the event loop for other requests.
    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, request: Request):
        request_id = request.state.request_id
        try:
            return request.app.state.service.answer(body.question, body.aircraft_type, request_id)
        except Exception as exc:  # noqa: BLE001 - a dependency failed or timed out
            log.error("query failed", extra={"request_id": request_id, "exc_type": type(exc).__name__})  # noqa: E501
            return JSONResponse(
                {"detail": "A dependency is unavailable. Try again.", "request_id": request_id},
                status_code=503,
            )

    return app


app = create_app()
