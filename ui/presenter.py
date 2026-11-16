"""Everything the UI decides, as plain functions with no Streamlit import, so it can be tested
offline: calling the API, turning a response into what to show, and the API health indicator.

app.py only lays out what this module returns. The UI never talks to Azure OpenAI or OpenSearch and
holds no secrets: its only dependency is the API's URL.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 120.0  # the API may make two model calls (answer + one retry)
HEALTH_TIMEOUT_SECONDS = 3.0
MAX_QUESTION_CHARS = 500  # the API's input guardrail limit
AUTO_DETECT = "Auto-detect"
AIRCRAFT_CHOICES = (AUTO_DETECT, "A-200", "B-340", "C-75")
NOTICE = "Synthetic training data. Not for use on real aircraft."

# Characters Streamlit's Markdown treats specially ($ starts LaTeX, :red[...] colours text).
_MARKDOWN_SPECIAL = "\\`*_{}[]<>()#+-.!|$:~"


@dataclass(frozen=True)
class Example:
    label: str
    question: str
    aircraft: str = AUTO_DETECT


EXAMPLES = (
    Example("Torque value", "What is the torque for the main wheel axle nut on the A-200?"),
    Example("Full procedure", "List the steps to replace the cabin pressure sensor", "B-340"),
    Example("Part number", "What is the part number of the heated pitot probe on the B-340?"),
    Example("Not in the manuals", "What is the torque for the A-200 nose wheel axle nut?"),
)


@dataclass(frozen=True)
class UiConfig:
    api_url: str
    timeout_seconds: float


def ui_config(environ: Mapping[str, str]) -> UiConfig:
    """The UI's only settings, from environment variables (the UI image has no aero_rag package,
    so it cannot use get_settings())."""
    api_url = (environ.get("API_URL") or DEFAULT_API_URL).strip().rstrip("/")
    try:
        timeout = float(environ.get("API_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS)
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SECONDS
    return UiConfig(
        api_url=api_url, timeout_seconds=timeout if timeout > 0 else DEFAULT_TIMEOUT_SECONDS
    )


def escape_markdown(text: str) -> str:
    """Show text exactly as written: answer text must never be reinterpreted as formatting."""
    return "".join(f"\\{c}" if c in _MARKDOWN_SPECIAL else c for c in text)


# ------------------------------------------------------------------ calling the API

Opener = Callable[..., Any]  # urllib.request.urlopen, or a fake in tests


@dataclass(frozen=True)
class ApiError:
    kind: Literal["unreachable", "timeout", "unavailable", "invalid", "server", "bad_response"]
    message: str
    request_id: str | None = None


_ERROR_MESSAGES = {
    "unreachable": "The assistant service is not reachable. It may be starting up or stopped. "
    "Try again in a minute.",
    "timeout": "The assistant took too long to answer. Try again; if it keeps happening, "
    "ask a shorter question.",
    "unavailable": "The assistant is running, but the manual search or the language model is "
    "unavailable right now. Try again shortly.",
    "invalid": "The question could not be sent. Check it is not empty and under "
    f"{MAX_QUESTION_CHARS} characters.",
    "server": "The assistant hit an internal error. Try again; if it repeats, quote the "
    "request ID.",
    "bad_response": "The assistant sent a reply the interface could not read.",
}


def api_error(kind: str, request_id: str | None = None) -> ApiError:
    return ApiError(kind=kind, message=_ERROR_MESSAGES[kind], request_id=request_id)  # type: ignore


def query_payload(question: str, aircraft_choice: str) -> dict:
    aircraft = None if aircraft_choice in ("", AUTO_DETECT) else aircraft_choice
    return {"question": question.strip(), "aircraft_type": aircraft}


def _request_id(body: bytes, headers: Any) -> str | None:
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and parsed.get("request_id"):
            return str(parsed["request_id"])
    except ValueError:
        pass
    return headers.get("X-Request-ID") if headers is not None else None


def _is_timeout(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", None)
    return isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError)


def call_api(
    config: UiConfig, question: str, aircraft_choice: str, opener: Opener = urllib.request.urlopen
) -> dict | ApiError:
    """POST /query. Returns the response body as a dict, or an ApiError with a friendly message.
    Exception details are never shown: they can contain internal host names."""
    request = urllib.request.Request(
        f"{config.api_url}/query",
        data=json.dumps(query_payload(question, aircraft_choice)).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=config.timeout_seconds) as response:
            body, headers = response.read(), response.headers
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read() or b""
        except (OSError, AttributeError):
            error_body = b""
        request_id = _request_id(error_body, exc.headers)
        kind = {422: "invalid", 503: "unavailable"}.get(exc.code, "server")
        return api_error(kind, request_id)
    except (urllib.error.URLError, OSError) as exc:  # refused, DNS failure, reset, timeout
        return api_error("timeout" if _is_timeout(exc) else "unreachable")
    try:
        parsed = json.loads(body)
    except ValueError:
        return api_error("bad_response", _request_id(b"", headers))
    if not isinstance(parsed, dict) or "status" not in parsed or "answer" not in parsed:
        return api_error("bad_response", _request_id(b"", headers))
    return parsed


@dataclass(frozen=True)
class Health:
    level: Literal["ok", "degraded", "down"]
    label: str
    detail: str


def check_health(config: UiConfig, opener: Opener = urllib.request.urlopen) -> Health:
    """/health says the API process is up; /ready says it can reach OpenSearch."""
    try:
        with opener(f"{config.api_url}/health", timeout=HEALTH_TIMEOUT_SECONDS):
            pass
    except (urllib.error.URLError, OSError):
        return Health("down", "API unreachable", "Questions cannot be answered right now.")
    try:
        with opener(f"{config.api_url}/ready", timeout=HEALTH_TIMEOUT_SECONDS):
            pass
    except (urllib.error.URLError, OSError):
        return Health(
            "degraded", "API up, search unavailable", "Answers may fail until search recovers."
        )
    return Health("ok", "API ready", "Connected to the manual search.")


# ------------------------------------------------------------------ what to show

Kind = Literal["answered", "clarification", "abstained", "rejected", "error"]


@dataclass(frozen=True)
class Alert:
    kind: Literal["WARNING", "CAUTION"]
    text: str

    @property
    def markdown(self) -> str:
        return f"**{self.kind}:** {escape_markdown(self.text)}"


@dataclass(frozen=True)
class StepView:
    number: int
    text: str
    alerts: tuple[Alert, ...]  # WARNINGs first, then CAUTIONs: read before doing the step

    @property
    def markdown(self) -> str:
        return f"**{self.number}.** {escape_markdown(self.text)}"


@dataclass(frozen=True)
class AlertGroup:
    """WARNINGs and CAUTIONs of one step that a short answer relies on."""

    heading: str
    alerts: tuple[Alert, ...]


@dataclass(frozen=True)
class CitationView:
    aircraft_type: str
    task_number: str
    title: str
    revision: int
    revision_date: str
    source_key: str

    @property
    def heading(self) -> str:
        return escape_markdown(f"{self.aircraft_type} task {self.task_number}: {self.title}")

    @property
    def detail(self) -> str:
        return escape_markdown(f"Revision {self.revision}, dated {self.revision_date}")


@dataclass(frozen=True)
class View:
    kind: Kind
    title: str
    message: str  # already escaped for Markdown
    steps: tuple[StepView, ...] = ()
    alert_groups: tuple[AlertGroup, ...] = ()
    citations: tuple[CitationView, ...] = ()
    options: tuple[str, ...] = ()  # aircraft to choose from, for a clarification
    request_id: str | None = None
    details: tuple[tuple[str, str], ...] = field(default=())  # retrieval scores, for debugging


_TITLES = {
    "answered": "Answer",
    "clarification": "Which aircraft?",
    "abstained": "No verified answer",
    "rejected": "Question not accepted",
    "error": "Something went wrong",
}


def _alerts(warnings: list, cautions: list) -> tuple[Alert, ...]:
    return tuple(Alert("WARNING", str(t)) for t in warnings) + tuple(
        Alert("CAUTION", str(t)) for t in cautions
    )


def _alert_groups(alerts: list[dict]) -> tuple[AlertGroup, ...]:
    grouped: dict[tuple, list[Alert]] = {}
    for a in alerts:
        key = (a.get("aircraft_type", ""), a.get("task_number", ""), a.get("step_number", 0))
        grouped.setdefault(key, []).append(Alert(a.get("kind", "WARNING"), str(a.get("text", ""))))
    groups = []
    for (aircraft, task, step), items in sorted(grouped.items(), key=lambda kv: kv[0]):
        items.sort(key=lambda alert: alert.kind != "WARNING")  # WARNINGs first
        heading = escape_markdown(f"{aircraft} task {task}, step {step}")
        groups.append(AlertGroup(heading, tuple(items)))
    return tuple(groups)


def _details(scores: dict | None) -> tuple[tuple[str, str], ...]:
    if not scores:
        return ()
    rows = []
    if scores.get("cosine") is not None:
        rows.append(("Similarity of best match (cosine)", f"{scores['cosine']:.2f}"))
    if scores.get("hybrid") is not None:
        rows.append(("Hybrid rank score", f"{scores['hybrid']:.2f}"))
    rows.append(
        ("Exact part/task number match", "yes" if scores.get("exact_identifier_match") else "no")
    )
    return tuple(rows)


def build_view(response: dict | ApiError) -> View:
    """Turn an API response (or an ApiError) into exactly what the page should show."""
    if isinstance(response, ApiError):
        return View("error", _TITLES["error"], escape_markdown(response.message),
                    request_id=response.request_id)  # fmt: skip

    status = response.get("status")
    kind: Kind = {
        "answered": "answered",
        "clarification_needed": "clarification",
        "abstained": "abstained",
        "rejected": "rejected",
    }.get(status, "error")  # type: ignore[assignment]
    message = escape_markdown(str(response.get("answer", "")))
    common = {
        "request_id": response.get("request_id"),
        "details": _details(response.get("scores")),
    }
    if kind == "error":
        return View("error", _TITLES["error"], escape_markdown(_ERROR_MESSAGES["bad_response"]),
                    **common)  # fmt: skip
    if kind != "answered":
        options = (
            tuple(response.get("clarification_options") or ()) if kind == "clarification" else ()
        )
        return View(kind, _TITLES[kind], message, options=options, **common)

    steps = tuple(
        StepView(
            int(s["number"]), str(s["text"]), _alerts(s.get("warnings", []), s.get("cautions", []))
        )
        for s in response.get("steps") or []
    )
    citations = tuple(
        CitationView(
            c["aircraft_type"],
            c["task_number"],
            c["title"],
            int(c["revision"]),
            str(c["revision_date"]),
            c["source_key"],
        )  # fmt: skip
        for c in response.get("citations") or []
    )
    return View(
        "answered", _TITLES["answered"], message, steps=steps,
        alert_groups=_alert_groups(response.get("warnings") or []), citations=citations, **common,
    )  # fmt: skip
