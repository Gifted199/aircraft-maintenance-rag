"""Offline tests for the UI's logic (ui/presenter.py). No Streamlit, no network: the API is a fake
opener, and responses are built with the API's own response model so the UI and API cannot drift."""

import io
import json
import re
import tomllib
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from aero_rag.retrieval.aircraft import AIRCRAFT_TYPES
from aero_rag.schemas import QueryResponse
from presenter import (
    AIRCRAFT_CHOICES,
    AUTO_DETECT,
    DEFAULT_API_URL,
    EXAMPLES,
    ApiError,
    UiConfig,
    build_view,
    call_api,
    check_health,
    escape_markdown,
    query_payload,
    ui_config,
)
from tests.fakes import make_hit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = UiConfig(api_url="http://api:8000", timeout_seconds=5)


def api_response(**fields) -> dict:
    """A response body exactly as the API serialises it."""
    return QueryResponse(request_id="req-1", **fields).model_dump(mode="json")


# ------------------------------------------------------------------ fake HTTP


class FakeResponse:
    def __init__(self, body: bytes, headers: dict | None = None):
        self._body = body
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Maps a URL path to a body (dict), an HTTP status code (int) or an exception."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.requests: list = []

    def __call__(self, request, timeout):
        url = request if isinstance(request, str) else request.full_url
        self.requests.append((request, timeout))
        outcome = self.routes[url.rsplit("/", 1)[1]]
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, int):
            headers = Message()
            headers["X-Request-ID"] = "hdr-9"
            body = json.dumps({"detail": "x", "request_id": "err-7"}).encode()
            raise urllib.error.HTTPError(url, outcome, "error", headers, io.BytesIO(body))
        return FakeResponse(json.dumps(outcome).encode())


# ------------------------------------------------------------------ configuration and request


def test_config_defaults_and_overrides():
    assert ui_config({}) == UiConfig(DEFAULT_API_URL, 120.0)
    config = ui_config({"API_URL": " http://api:8000/ ", "API_TIMEOUT_SECONDS": "30"})
    assert config == UiConfig("http://api:8000", 30.0)
    assert ui_config({"API_TIMEOUT_SECONDS": "soon"}).timeout_seconds == 120.0


def test_auto_detect_sends_no_aircraft_and_a_choice_is_sent_as_is():
    assert query_payload("  torque?  ", AUTO_DETECT) == {
        "question": "torque?",
        "aircraft_type": None,
    }
    assert query_payload("torque?", "B-340")["aircraft_type"] == "B-340"


def test_aircraft_choices_and_examples_match_the_manuals():
    assert AIRCRAFT_CHOICES == (AUTO_DETECT, *AIRCRAFT_TYPES)
    assert len(EXAMPLES) == 4 and all(e.aircraft in AIRCRAFT_CHOICES for e in EXAMPLES)
    labels = " ".join(e.label.lower() for e in EXAMPLES)
    for kind in ("torque", "procedure", "part number", "not in the manuals"):
        assert kind in labels


def test_call_api_posts_json_and_returns_the_body():
    opener = FakeOpener({"query": api_response(status="answered", answer="135 Nm.")})
    body = call_api(CONFIG, "torque?", "A-200", opener)
    assert body["status"] == "answered"
    request, timeout = opener.requests[0]
    assert request.full_url == "http://api:8000/query" and request.get_method() == "POST"
    assert json.loads(request.data) == {"question": "torque?", "aircraft_type": "A-200"}
    assert timeout == 5


@pytest.mark.parametrize(
    ("outcome", "kind", "request_id"),
    [
        (urllib.error.URLError(ConnectionRefusedError()), "unreachable", None),
        (urllib.error.URLError(OSError("name resolution")), "unreachable", None),
        (urllib.error.URLError(TimeoutError()), "timeout", None),
        (TimeoutError(), "timeout", None),
        (503, "unavailable", "err-7"),
        (500, "server", "err-7"),
        (422, "invalid", "err-7"),
    ],
)
def test_call_api_turns_failures_into_friendly_errors(outcome, kind, request_id):
    error = call_api(CONFIG, "q", AUTO_DETECT, FakeOpener({"query": outcome}))
    assert isinstance(error, ApiError) and error.kind == kind and error.request_id == request_id
    assert "api:8000" not in error.message  # no internal host names or exception text


def test_call_api_rejects_a_reply_that_is_not_a_query_response():
    error = call_api(CONFIG, "q", AUTO_DETECT, FakeOpener({"query": {"unexpected": True}}))
    assert isinstance(error, ApiError) and error.kind == "bad_response"


@pytest.mark.parametrize(
    ("routes", "level"),
    [
        ({"health": {"status": "ok"}, "ready": {"status": "ready"}}, "ok"),
        ({"health": {"status": "ok"}, "ready": 503}, "degraded"),
        ({"health": urllib.error.URLError(ConnectionRefusedError()), "ready": 503}, "down"),
    ],
)
def test_health_indicator(routes, level):
    assert check_health(CONFIG, FakeOpener(routes)).level == level


# ------------------------------------------------------------------ rendering by status


def test_answered_procedure_puts_each_warning_and_caution_on_its_own_step():
    steps = [
        {"number": 1, "text": "Jack the aircraft.", "warnings": [], "cautions": []},
        {"number": 2, "text": "Torque the nut to 135 Nm.", "warnings": ["Use the final value."],
         "cautions": ["Do not exceed 140 Nm.", "Use a calibrated wrench."]},
    ]  # fmt: skip
    hit = make_hit("A-200_32-41-01_rev03")
    fields = ("aircraft_type", "task_number", "title", "revision", "revision_date", "source_key")
    citation = {k: getattr(hit, k) for k in fields}
    view = build_view(api_response(status="answered", answer="See the procedure.", steps=steps,
                                   citations=[citation]))  # fmt: skip

    assert view.kind == "answered" and view.request_id == "req-1"
    assert [s.number for s in view.steps] == [1, 2]
    assert view.steps[0].alerts == ()
    assert [a.kind for a in view.steps[1].alerts] == ["WARNING", "CAUTION", "CAUTION"]
    assert view.steps[1].alerts[0].markdown == "**WARNING:** Use the final value\\."
    assert view.steps[1].markdown.startswith("**2.** Torque the nut to 135 Nm")
    (c,) = view.citations
    assert (c.task_number, c.revision, c.revision_date) == ("32-41-01", 3, "2026-01-19")
    assert c.source_key == "manuals/A-200_32-41-01_rev03.md"
    assert "Main Wheel Axle Nut" in c.heading and c.detail == "Revision 3, dated 2026\\-01\\-19"


def test_short_answer_shows_the_alerts_of_the_steps_it_relies_on():
    warnings = [
        {"kind": "CAUTION", "text": "Check the lock bolt.", "aircraft_type": "A-200",
         "task_number": "32-41-01", "step_number": 7},
        {"kind": "WARNING", "text": "Not the seating torque.", "aircraft_type": "A-200",
         "task_number": "32-41-01", "step_number": 7},
    ]  # fmt: skip
    view = build_view(api_response(status="answered", answer="135 Nm.", warnings=warnings))
    (group,) = view.alert_groups
    assert "step 7" in group.heading and "32\\-41\\-01" in group.heading
    assert [a.kind for a in group.alerts] == ["WARNING", "CAUTION"]  # WARNING first


def test_clarification_shows_the_question_and_the_aircraft_options():
    view = build_view(api_response(status="clarification_needed", answer="Which aircraft type?",
                                   clarification_options=["A-200", "B-340", "C-75"]))  # fmt: skip
    assert view.kind == "clarification" and view.options == ("A-200", "B-340", "C-75")
    assert view.message == "Which aircraft type?" and view.steps == () and view.citations == ()


def test_abstention_shows_the_decline_message_and_nothing_else():
    message = "I don't know. The manuals do not cover this for the nose."
    view = build_view(api_response(status="abstained", answer=message,
                                   scores={"cosine": 0.72, "hybrid": 1.0}))  # fmt: skip
    assert view.kind == "abstained" and view.title == "No verified answer"
    assert view.message == escape_markdown(message)
    assert view.steps == () and view.citations == () and view.options == ()
    assert ("Similarity of best match (cosine)", "0.72") in view.details


def test_rejected_and_errors_have_their_own_presentation():
    assert build_view(api_response(status="rejected", answer="Too long.")).kind == "rejected"
    error = build_view(ApiError("timeout", "Took too long.", "abc"))
    assert (
        error.kind == "error" and error.request_id == "abc" and error.message == "Took too long\\."
    )
    assert build_view({"status": "surprise", "answer": "x"}).kind == "error"


def test_answer_text_is_never_interpreted_as_markdown_or_latex():
    text = "Use *only* $5 [link](http://x) :red[alert] <b>bold</b> # heading"
    escaped = escape_markdown(text)
    unescaped = re.sub(r"\\(.)", r"\1", escaped)
    assert unescaped == text
    assert not re.search(r"(?<!\\)[*$\[\]<>#:]", escaped)


# ------------------------------------------------------------------ packaging


def test_ui_image_pins_the_same_streamlit_as_the_local_group():
    group = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]["ui"]
    lock = (ROOT / "ui" / "requirements.txt").read_text()
    assert "streamlit==1.67.0" in group and "\nstreamlit==1.67.0\n" in lock


def test_ui_has_no_access_to_secrets_or_backends():
    source = (ROOT / "ui" / "app.py").read_text() + (ROOT / "ui" / "presenter.py").read_text()
    imports = " ".join(re.findall(r"^\s*(?:from|import) (\S+)", source, flags=re.MULTILINE))
    for forbidden in ("aero_rag", "openai", "opensearch", "boto3", "requests", "httpx"):
        assert forbidden not in imports
    environ_keys = set(re.findall(r"environ\.get\(\"(\w+)\"", source))
    assert environ_keys == {"API_URL", "API_TIMEOUT_SECONDS"}  # the UI's only settings
    allowed = (ROOT / "ui" / ".dockerignore").read_text().split()
    assert sorted(p for p in allowed if p.startswith("!")) == ["!app.py", "!presenter.py",
                                                               "!requirements.txt"]  # fmt: skip
