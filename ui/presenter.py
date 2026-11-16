"""Everything the UI decides, as plain functions with no Streamlit import, so it can be tested
offline: calling the API, turning a response into what to show, the API health indicator, and the
chat conversation (what to send for each message, clarifications, the remembered aircraft).

app.py only lays out what this module returns. The UI never talks to Azure OpenAI or OpenSearch and
holds no secrets: its only dependency is the API's URL.
"""

import json
import re
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

Kind = Literal["answered", "clarification", "abstained", "rejected", "error", "notice"]


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

    @property
    def markdown(self) -> str:
        """One compact block for a chat message: task and title, revision and date, S3 key."""
        return f"**{self.heading}**  \n{self.detail} · `{self.source_key.replace('`', '')}`"


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


# ------------------------------------------------------------------ the chat conversation
#
# The conversation remembers two things only: the messages to display, and the current aircraft.
# Every message becomes one complete, standalone question sent to /query with that aircraft:
# previous answers are never sent to the API or the model, so each answer is retrieved and
# verified on its own, exactly like a single API call.

# Same rule as the API's aero_rag.retrieval.aircraft.detect_aircraft (a test keeps them in step):
# "A-200", "A200", "a 200", but not inside a part number such as "A200-32-4112".
_AIRCRAFT_PATTERNS = {
    aircraft: re.compile(
        rf"(?<![A-Za-z0-9-]){aircraft[0]}[- ]?{aircraft.split('-')[1]}(?![A-Za-z0-9]|-\d)",
        re.IGNORECASE,
    )
    for aircraft in AIRCRAFT_CHOICES[1:]
}
# Words that may surround an aircraft name in a reply that only picks an aircraft:
# "A-200", "the B-340 please", "and for the C-75?", "what about the B340".
_FILLER = frozenset(
    "a an and the for on please what about how ok okay yes it its that same then one aircraft "
    "type i mean meant is now instead use with to of again same question answer".split()
)


def detect_aircraft(text: str) -> str | None:
    """The single aircraft type named in the text, or None if none or more than one."""
    found = [a for a, pattern in _AIRCRAFT_PATTERNS.items() if pattern.search(text)]
    return found[0] if len(found) == 1 else None


def is_aircraft_only(text: str) -> bool:
    """True for a reply that just names an aircraft, rather than asking a new question."""
    if detect_aircraft(text) is None:
        return False
    rest = text
    for pattern in _AIRCRAFT_PATTERNS.values():
        rest = pattern.sub(" ", rest)
    return all(word in _FILLER for word in re.findall(r"[a-z]+", rest.lower()))


@dataclass(frozen=True)
class AskRequest:
    question: str  # sent to the API: always a complete, standalone question
    aircraft: str  # AUTO_DETECT or an aircraft type
    shown: str  # the user's own words, for their chat bubble


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    text: str = ""  # user: what they typed or clicked
    view: View | None = None  # assistant: what to show
    question: str = ""  # assistant: the standalone question this turn answered


@dataclass(frozen=True)
class Conversation:
    turns: tuple[Turn, ...] = ()
    aircraft: str = AUTO_DETECT

    @property
    def last_question(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.role == "assistant" and turn.question:
                return turn.question
        return None

    @property
    def pending_clarification(self) -> Turn | None:
        """The last message, if it is the assistant asking which aircraft."""
        if self.turns:
            last = self.turns[-1]
            if last.role == "assistant" and last.view and last.view.kind == "clarification":
                return last
        return None


@dataclass(frozen=True)
class Plan:
    """What to do with a message: ask the API, or only reply locally (no API call)."""

    request: AskRequest | None
    notice: str | None = None
    aircraft: str | None = None  # new conversation aircraft for a local reply


def plan_message(conversation: Conversation, text: str) -> Plan | None:
    text = text.strip()
    if not text:
        return None
    named = detect_aircraft(text)
    if named and is_aircraft_only(text):
        pending = conversation.pending_clarification
        if pending is not None and named in pending.view.options:
            return Plan(AskRequest(pending.question, named, text))  # answers "which aircraft?"
        if conversation.last_question:
            return Plan(AskRequest(conversation.last_question, named, text))  # same, other type
        return Plan(None, notice=f"Aircraft set to {named}. What would you like to know?",
                    aircraft=named)  # fmt: skip
    return Plan(AskRequest(text, named or conversation.aircraft, text))


def plan_clarification(conversation: Conversation, option: str) -> Plan | None:
    """An aircraft button under "which aircraft?": re-ask the original question for it."""
    pending = conversation.pending_clarification
    if pending is None or option not in pending.view.options:
        return None
    return Plan(AskRequest(pending.question, option, option))


def plan_example(example: Example) -> Plan:
    return Plan(AskRequest(example.question, example.aircraft, example.question))


def record_exchange(conversation: Conversation, request: AskRequest, view: View) -> Conversation:
    """Add the user's message and the answer. An aircraft that was used becomes the current one."""
    aircraft = request.aircraft if request.aircraft != AUTO_DETECT else conversation.aircraft
    turns = (
        *conversation.turns,
        Turn("user", text=request.shown),
        Turn("assistant", view=view, question=request.question),
    )
    return Conversation(turns, aircraft)


def record_notice(conversation: Conversation, shown: str, plan: Plan) -> Conversation:
    notice = View("notice", "", escape_markdown(plan.notice or ""))
    turns = (*conversation.turns, Turn("user", text=shown), Turn("assistant", view=notice))
    return Conversation(turns, plan.aircraft or conversation.aircraft)


def with_aircraft(conversation: Conversation, aircraft: str) -> Conversation:
    if aircraft not in AIRCRAFT_CHOICES:
        return conversation
    return Conversation(conversation.turns, aircraft)


CONVERSATION_NOTE = (
    "The assistant remembers the **aircraft** for this conversation, nothing else. Every "
    "question is searched and checked against the manuals on its own: earlier answers are never "
    "sent back to the model, so one answer cannot leak into the next. Ask complete questions; "
    "to repeat the last question for another aircraft, just type the aircraft (for example "
    "*B-340*)."
)
