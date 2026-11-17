"""Optional request tracing with Langfuse Cloud.

One trace per /query, with a child observation per step (input guardrail, aircraft detection,
retrieval, abstain decision, each generation attempt and its output guardrail check) and the final
status. The trace id is derived from the request id, so a request id from a log line or from the
UI finds its trace directly.

Three rules:
  * Tracing never breaks a request. Every call into the SDK is guarded; the first failure logs one
    warning and switches tracing off for the rest of the process. Spans are recorded in memory and
    exported by a background thread, so a slow or unreachable Langfuse never delays an answer.
  * By default traces follow the logging rules: sizes, decisions, scores, ids, tokens and cost;
    never the question, the prompt or the answer. LANGFUSE_CAPTURE_TEXT=true adds the question
    and the final answer text (never the prompt or the retrieved manual text). That is for the
    synthetic demo data only and must stay false for real, proprietary manuals. Every value sent
    is passed through the same secret and hostname redaction as the JSON logs.
  * Missing keys mean no tracing (one warning at start-up), not an error.

The service talks only to the small interface below (Tracer / QueryTrace / Span). The base classes
do nothing, so tests, evaluation runs and a process without keys pay nothing.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from aero_rag.logging_config import redact_value, redaction_list

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ the interface (no-op)


class Span:
    """A handle on one step. The base class ignores everything."""

    def update(self, **fields: Any) -> None:
        pass

    def end(self) -> None:
        pass


class QueryTrace:
    trace_id: str | None = None

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", **fields: Any) -> Iterator[Span]:
        yield Span()

    def finish(
        self, *, status: str, aircraft_type: str | None, output: dict, text: dict | None = None
    ) -> None:
        """`text` is the final answer text; a tracer sends it only if text capture is on."""


class Tracer:
    enabled = False
    capture_text = False

    def start_query(
        self, request_id: str, input: dict, question: str | None = None  # noqa: A002
    ) -> QueryTrace:
        """`question` is the raw question; a tracer sends it only if text capture is on."""
        return QueryTrace()

    def shutdown(self) -> None:
        pass


NULL_TRACER = Tracer()


# ------------------------------------------------------------------ Langfuse implementation


def _release() -> str:
    try:
        return version("aero-rag")
    except PackageNotFoundError:
        return "unknown"


class LangfuseTracer(Tracer):
    enabled = True

    def __init__(
        self, client: Any, redact: list[str] | tuple[str, ...] = (), capture_text: bool = False
    ):
        self._client = client
        self._redact = redaction_list(redact)
        self._broken = False
        self.capture_text = capture_text

    # -- guard: one warning, then off
    def safe(self, action: Callable[[], Any]) -> Any:
        if self._broken:
            return None
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - tracing must never fail a request
            self._broken = True
            log.warning(
                "tracing failed and is now disabled for this process",
                extra={"exc_type": type(exc).__name__},
            )
            return None

    def clean(self, fields: dict) -> dict:
        return {k: redact_value(v, self._redact) for k, v in fields.items() if v is not None}

    def start_query(
        self, request_id: str, input: dict, question: str | None = None  # noqa: A002
    ) -> QueryTrace:
        if self.capture_text and question is not None:
            input = {**input, "question": question}  # redacted by clean() like everything else
        trace = self.safe(lambda: _LangfuseQueryTrace(self, request_id, input))
        return trace or QueryTrace()

    def shutdown(self) -> None:
        """Send everything still buffered. Called when the API stops (ECS sends SIGTERM)."""
        if self._client is not None:
            self.safe(self._client.shutdown)
            log.info("tracing flushed")


class _LangfuseSpan(Span):
    def __init__(self, tracer: LangfuseTracer, observation: Any):
        self._tracer, self._observation = tracer, observation

    def update(self, **fields: Any) -> None:
        clean = self._tracer.clean(fields)
        self._tracer.safe(lambda: self._observation.update(**clean))

    def end(self) -> None:
        self._tracer.safe(self._observation.end)


class _LangfuseQueryTrace(QueryTrace):
    def __init__(self, tracer: LangfuseTracer, request_id: str, input: dict):  # noqa: A002
        client = tracer._client
        self._tracer, self._request_id = tracer, request_id
        self.trace_id = client.create_trace_id(seed=request_id)
        # The root is made "current" so that propagate_attributes() in finish() writes the
        # trace-level tags onto it; the step observations are created explicitly under it.
        self._context = client.start_as_current_observation(
            trace_context={"trace_id": self.trace_id},
            name="query",
            **tracer.clean({"input": input, "metadata": {"request_id": request_id}}),
        )
        self._root = self._context.__enter__()

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", **fields: Any) -> Iterator[Span]:
        clean = self._tracer.clean(fields)
        observation = self._tracer.safe(
            lambda: self._root.start_observation(name=name, as_type=as_type, **clean)
        )
        handle = _LangfuseSpan(self._tracer, observation) if observation is not None else Span()
        try:
            yield handle
        except Exception as exc:
            handle.update(level="ERROR", status_message=type(exc).__name__)
            raise
        finally:
            handle.end()

    def finish(
        self, *, status: str, aircraft_type: str | None, output: dict, text: dict | None = None
    ) -> None:
        from langfuse import propagate_attributes

        tracer = self._tracer
        if tracer.capture_text and text:
            output = {**text, **output}  # answer text first, so it leads in the Langfuse UI
        tags = [f"status:{status}", f"aircraft:{aircraft_type or 'none'}",
                f"request_id:{self._request_id}"]  # fmt: skip
        clean = tracer.clean({"output": output})

        def record() -> None:
            self._root.create_event(name="final_status", **tracer.clean({"output": {
                "status": status, "reason": output.get("reason")}}))  # fmt: skip
            self._root.update(**clean)
            metadata = {"request_id": self._request_id, "status": status}
            with propagate_attributes(tags=tags, trace_name="query", metadata=metadata):
                pass

        tracer.safe(record)
        tracer.safe(lambda: self._context.__exit__(None, None, None))


# ------------------------------------------------------------------ start-up


def build_tracer(settings: Any, client_factory: Callable[..., Any] | None = None) -> Tracer:
    """A LangfuseTracer if keys are configured and Langfuse accepts them; otherwise NULL_TRACER
    and exactly one warning. Never raises."""
    if not settings.langfuse_enabled:
        log.warning("tracing disabled: Langfuse keys are not configured")
        return NULL_TRACER
    client = None
    try:
        if client_factory is None:
            from langfuse import Langfuse
            from opentelemetry.sdk.trace import TracerProvider

            # My own provider: Langfuse does not become the process-wide OpenTelemetry tracer.
            client_factory = lambda **kw: Langfuse(tracer_provider=TracerProvider(), **kw)  # noqa: E731
        client = client_factory(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
            timeout=settings.langfuse_timeout_seconds,
            environment=settings.environment,
            release=_release(),
        )
        ok, problem = bool(client.auth_check()), "keys_rejected"
    except Exception as exc:  # noqa: BLE001 - unreachable, DNS, TLS, bad keys...
        ok, problem = False, type(exc).__name__
    if not ok:
        log.warning("tracing disabled: Langfuse is unreachable or rejected the keys",
                    extra={"problem": problem})  # fmt: skip
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001
                pass
        return NULL_TRACER
    log.info("tracing enabled", extra={"tracing": "langfuse", "environment": settings.environment,
                                       "capture_text": settings.langfuse_capture_text})  # fmt: skip
    return LangfuseTracer(
        client, redact=settings.secret_values(), capture_text=settings.langfuse_capture_text
    )
