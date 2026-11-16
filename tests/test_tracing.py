"""Tracing: what each /query records, that it never carries text or secrets, and that tracing can
never fail or slow a request (keys missing, Langfuse down, SDK errors). All offline: the Langfuse
SDK is used with an in-memory exporter or replaced by fakes."""

import json
import logging
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from aero_rag.api.app import create_app
from aero_rag.config import Settings, merge_secret
from aero_rag.generation.generator import GenerationError
from aero_rag.guardrails.output import problem_kind
from aero_rag.logging_config import TracingOnceFilter, setup_logging
from aero_rag.schemas import LLMAnswer
from aero_rag.service import RagService
from aero_rag.tracing import NULL_TRACER, LangfuseTracer, QueryTrace, Span, Tracer, build_tracer
from tests.fakes import FakeGenerator, StubSearcher, faithful_answer, make_hit

A200_AXLE = "A-200_32-41-01_rev03"
QUESTION = "What is the torque for the main wheel axle nut on the A-200?"
GOOD = "The final torque is 135 Nm (A-200, task 32-41-01, revision 3)."
CANNOT = LLMAnswer(status="cannot_answer", answer="Not in the sources.", steps=[], source_ids=[])


def good(_q, context, _p):
    return faithful_answer(context, text=GOOD)


def bad(_q, context, _p):
    return faithful_answer(context, text="The final torque is 120 Nm.")


# ------------------------------------------------------------------ a recording fake tracer


class RecordingSpan(Span):
    def __init__(self, record: dict):
        self.record = record

    def update(self, **fields):
        self.record.setdefault("updates", {}).update(fields)


class RecordingTrace(QueryTrace):
    def __init__(self, request_id, input):  # noqa: A002
        self.request_id, self.input, self.spans, self.finished = request_id, input, [], None

    @contextmanager
    def span(self, name, *, as_type="span", **fields):
        record = {"name": name, "as_type": as_type, **fields}
        self.spans.append(record)
        yield RecordingSpan(record)

    def finish(self, *, status, aircraft_type, output):
        self.finished = {"status": status, "aircraft_type": aircraft_type, "output": output}

    def names(self):
        return [s["name"] for s in self.spans]

    def out(self, name, index=0):
        return [s for s in self.spans if s["name"] == name][index]["updates"]["output"]


class RecordingTracer(Tracer):
    enabled = True

    def __init__(self):
        self.traces: list[RecordingTrace] = []

    def start_query(self, request_id, input):  # noqa: A002
        self.traces.append(RecordingTrace(request_id, input))
        return self.traces[-1]


def make_service(generator, tracer, hits=None, aircraft_type="A-200"):
    if hits is None:
        hits = [make_hit(A200_AXLE, score=0.97, cosine=0.73)]
    generator.model = "gpt-5.4-mini"
    return RagService(StubSearcher(hits, aircraft_type), generator, cosine_threshold=0.35,
                      max_question_chars=500, tracer=tracer)  # fmt: skip


# ------------------------------------------------------------------ what a trace contains


def test_answered_query_has_one_span_per_step_and_a_final_status():
    tracer = RecordingTracer()
    response = make_service(FakeGenerator(good), tracer).answer(QUESTION, request_id="req-1")
    (trace,) = tracer.traces

    assert response.status == "answered" and trace.request_id == "req-1"
    assert trace.names() == ["input_guardrail", "aircraft_detection", "retrieval",
                             "abstain_decision", "generation", "output_guardrail"]  # fmt: skip
    assert trace.out("input_guardrail") == {"result": "passed"}
    assert trace.out("aircraft_detection")["source"] == "detected"
    retrieval = [s for s in trace.spans if s["name"] == "retrieval"][0]
    assert retrieval["as_type"] == "retriever" and retrieval["input"]["top_k"] == 5
    assert retrieval["input"]["mode"] == "hybrid" and retrieval["input"]["latest_only"] is True
    (hit,) = trace.out("retrieval")["hits"]
    assert hit["chunk_id"] == f"{A200_AXLE}#procedure" and hit["score"] == 0.97
    assert trace.out("abstain_decision")["action"] == "answer"

    generation = [s for s in trace.spans if s["name"] == "generation"][0]
    assert generation["as_type"] == "generation" and generation["model"] == "gpt-5.4-mini"
    updates = generation["updates"]
    assert updates["usage_details"] == {"input": 100, "output": 20, "total": 120}
    assert updates["cost_details"]["total"] == pytest.approx((100 * 0.75 + 20 * 4.5) / 1e6)
    assert "latency_ms" in updates["metadata"]
    assert trace.out("output_guardrail") == {"result": "pass", "problem_count": 0,
                                             "problem_kinds": []}  # fmt: skip

    assert trace.finished["status"] == "answered" and trace.finished["aircraft_type"] == "A-200"
    assert trace.finished["output"]["llm_calls"] == 1


def test_retry_is_two_generations_and_the_guardrail_says_why():
    tracer = RecordingTracer()
    make_service(FakeGenerator(bad, good), tracer).answer(QUESTION)
    (trace,) = tracer.traces
    assert trace.names()[-4:] == ["generation", "output_guardrail", "generation",
                                  "output_guardrail"]  # fmt: skip
    first, second = trace.out("output_guardrail", 0), trace.out("output_guardrail", 1)
    assert first == {"result": "retry", "problem_count": 1, "problem_kinds": ["torque"]}
    assert second["result"] == "pass"
    assert [s["input"]["retry"] for s in trace.spans if s["name"] == "generation"] == [False, True]


def test_unusable_model_output_is_an_error_generation_then_abstain():
    tracer = RecordingTracer()
    service = make_service(FakeGenerator(GenerationError("x"), GenerationError("x")), tracer)
    assert service.answer(QUESTION).status == "abstained"
    (trace,) = tracer.traces
    generation = [s for s in trace.spans if s["name"] == "generation"][0]
    assert generation["updates"]["level"] == "ERROR"
    assert [trace.out("output_guardrail", i)["result"] for i in (0, 1)] == ["retry", "abstain"]
    assert trace.finished["output"]["reason"] == "output_guardrail_failed"


def test_abstain_and_reject_stop_the_trace_early():
    tracer = RecordingTracer()
    weak = [make_hit(A200_AXLE, 0.8, 0.10)]
    make_service(FakeGenerator(good), tracer, hits=weak, aircraft_type=None).answer("Who won?")
    make_service(FakeGenerator(good), tracer).answer("x" * 600)
    abstained, rejected = tracer.traces
    assert "generation" not in abstained.names()
    assert abstained.out("abstain_decision")["reason"] == "low_similarity"
    assert abstained.finished["status"] == "abstained"
    assert rejected.names() == ["input_guardrail"] and rejected.finished["status"] == "rejected"
    assert rejected.finished["aircraft_type"] is None


def test_unknown_aircraft_is_recorded_by_the_detection_span():
    tracer = RecordingTracer()
    response = make_service(FakeGenerator(good), tracer).answer(QUESTION, aircraft_type="Z-9")
    assert response.status == "rejected"
    assert tracer.traces[0].out("aircraft_detection")["source"] == "unknown"


def test_a_trace_never_contains_the_question_or_the_answer_text():
    tracer = RecordingTracer()
    question = "What is the torque for the main wheel axle nut on the A-200? zebra-marker"
    make_service(FakeGenerator(good), tracer).answer(question)
    (trace,) = tracer.traces
    dumped = json.dumps([trace.input, trace.spans, trace.finished], default=str)
    assert "zebra-marker" not in dumped and "main wheel" not in dumped
    assert "135 Nm" not in dumped and "final torque" not in dumped


def test_problem_kinds_never_carry_values():
    assert problem_kind("part number A200-32-4109 does not appear in the sources") == "part_number"
    assert problem_kind("torque value 120 Nm does not appear in the sources") == "torque"
    assert problem_kind("source 2 is a superseded revision and must not be used") == (
        "superseded_revision"
    )
    assert problem_kind('source 1 ("Main") is about the main, but the question asks about the '
                        "nose.") == "component"  # fmt: skip
    assert problem_kind("step 3: the text was changed; copy it word for word.") == "steps"


# ------------------------------------------------------------------ the real SDK, offline


def real_tracer(exporter, redact=()):
    client = Langfuse(
        public_key=f"pk-lf-test-{uuid.uuid4().hex}", secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9", span_exporter=exporter, tracer_provider=TracerProvider(),
        timeout=1,
    )  # fmt: skip
    return LangfuseTracer(client, redact=redact), client


def test_real_sdk_exports_one_trace_with_tags_and_typed_observations():
    exporter = InMemorySpanExporter()
    tracer, client = real_tracer(exporter)
    make_service(FakeGenerator(good), tracer).answer(QUESTION, request_id="req-42")
    tracer.shutdown()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    expected = {"query", "input_guardrail", "aircraft_detection", "retrieval",
                "abstain_decision", "generation", "output_guardrail", "final_status"}  # fmt: skip
    assert set(spans) >= expected
    expected_trace = Langfuse.create_trace_id(seed="req-42")
    assert {format(s.context.trace_id, "032x") for s in spans.values()} == {expected_trace}
    root = spans["query"].attributes
    assert set(root["langfuse.trace.tags"]) == {"status:answered", "aircraft:A-200",
                                                "request_id:req-42"}  # fmt: skip
    assert spans["retrieval"].attributes["langfuse.observation.type"] == "retriever"
    generation = spans["generation"].attributes
    assert generation["langfuse.observation.type"] == "generation"
    assert generation["langfuse.observation.model.name"] == "gpt-5.4-mini"
    assert json.loads(generation["langfuse.observation.usage_details"])["input"] == 100
    everything = json.dumps([dict(s.attributes) for s in spans.values()], default=str)
    assert "main wheel axle nut" not in everything.lower() and GOOD not in everything


def test_secrets_and_hostnames_are_redacted_before_they_reach_langfuse():
    exporter = InMemorySpanExporter()
    host = "search-akinsds.eu-west-2.es.amazonaws.com"
    tracer, _ = real_tracer(exporter, redact=[host])
    trace = tracer.start_query("req-7", {"note": f"host {host}"})
    with trace.span("retrieval") as span:
        span.update(output={"error": f"timeout talking to {host}"})
    trace.finish(status="error", aircraft_type=None, output={})
    tracer.shutdown()
    everything = json.dumps([dict(s.attributes) for s in exporter.get_finished_spans()])
    assert "amazonaws" not in everything and "***" in everything


# ------------------------------------------------------------------ never fail a request


class FailingExporter(SpanExporter):
    """Langfuse is down: every export fails."""

    def export(self, spans):
        raise ConnectionError("langfuse unreachable")

    def shutdown(self):
        pass


def test_langfuse_down_during_requests_does_not_affect_answers():
    tracer, _ = real_tracer(FailingExporter())
    service = make_service(FakeGenerator(good), tracer)
    responses = [service.answer(QUESTION) for _ in range(3)]
    assert [r.status for r in responses] == ["answered"] * 3
    tracer.shutdown()  # flushing to a dead endpoint must not raise


class ExplodingClient:
    """An SDK that raises on every call (bug, version mismatch, ...)."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise RuntimeError("sdk failure")

        return fail


def test_sdk_errors_are_swallowed_with_exactly_one_warning(caplog):
    tracer = LangfuseTracer(ExplodingClient())
    service = make_service(FakeGenerator(good), tracer)
    with caplog.at_level(logging.WARNING, logger="aero_rag.tracing"):
        statuses = [service.answer(QUESTION).status for _ in range(3)]
        tracer.shutdown()
    assert statuses == ["answered"] * 3
    warnings = [r for r in caplog.records if r.name == "aero_rag.tracing"]
    assert len(warnings) == 1 and warnings[0].exc_type == "RuntimeError"
    assert "sdk failure" not in caplog.text  # the exception message is never logged


def settings_with_keys(**extra) -> Settings:
    return Settings(_env_file=None, langfuse_public_key="pk-lf-x",
                    langfuse_secret_key=SecretStr("sk-lf-y"), **extra)  # fmt: skip


@pytest.mark.parametrize(
    ("settings", "factory", "problem"),
    [
        (Settings(_env_file=None), None, None),  # keys missing
        (settings_with_keys(), lambda **kw: SimpleNamespace(
            auth_check=lambda: (_ for _ in ()).throw(ConnectionError()), shutdown=lambda: None),
         "ConnectionError"),  # unreachable
        (settings_with_keys(), lambda **kw: SimpleNamespace(auth_check=lambda: False,
                                                            shutdown=lambda: None),
         "keys_rejected"),
        (settings_with_keys(), lambda **kw: (_ for _ in ()).throw(ValueError()), "ValueError"),
    ],
)  # fmt: skip
def test_tracing_is_off_with_one_warning_when_keys_are_missing_or_langfuse_is_down(
    settings, factory, problem, caplog
):
    with caplog.at_level(logging.INFO, logger="aero_rag.tracing"):
        tracer = build_tracer(settings, client_factory=factory)
    assert tracer is NULL_TRACER
    records = [r for r in caplog.records if r.name == "aero_rag.tracing"]
    assert len(records) == 1 and records[0].levelno == logging.WARNING
    assert getattr(records[0], "problem", None) == problem
    # ...and the service still answers normally with it.
    assert make_service(FakeGenerator(good), tracer).answer(QUESTION).status == "answered"


def test_tracing_is_on_when_langfuse_accepts_the_keys():
    calls = {}

    def factory(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(auth_check=lambda: True, shutdown=lambda: None)

    tracer = build_tracer(settings_with_keys(langfuse_host="https://eu.example.test/"), factory)
    assert isinstance(tracer, LangfuseTracer)
    assert calls["base_url"] == "https://eu.example.test" and calls["secret_key"] == "sk-lf-y"
    assert calls["environment"] == "dev" and calls["timeout"] == 5


# ------------------------------------------------------------------ config, shutdown, logging


def test_langfuse_keys_come_from_the_secret_and_environment_wins():
    secret = {"LANGFUSE_PUBLIC_KEY": "pk-from-secret", "LANGFUSE_SECRET_KEY": "sk-from-secret",
              "LANGFUSE_HOST": "https://cloud.langfuse.com"}  # fmt: skip
    merged = merge_secret(Settings(_env_file=None), secret)
    assert merged.langfuse_enabled and merged.langfuse_secret_key.get_secret_value() == (
        "sk-from-secret"
    )
    assert "sk-from-secret" in merged.secret_values() and "pk-from-secret" in merged.secret_values()
    env_first = merge_secret(Settings(_env_file=None, langfuse_public_key="pk-env"), secret)
    assert env_first.langfuse_public_key == "pk-env"
    assert merge_secret(Settings(_env_file=None), {}).langfuse_enabled is False
    assert Settings(_env_file=None).langfuse_base_url == "https://cloud.langfuse.com"
    assert "sk-from-secret" not in repr(merged)


def test_api_shutdown_flushes_the_tracer():
    flushed = []

    class FlushTracer(Tracer):
        def shutdown(self):
            flushed.append(True)

    service = make_service(FakeGenerator(good), NULL_TRACER)
    with TestClient(create_app(service, tracer=FlushTracer())) as client:
        assert client.get("/health").status_code == 200
        assert flushed == []
    assert flushed == [True]


def test_tracing_library_logs_are_limited_to_one_line(capsys):
    setup_logging("INFO")
    for _ in range(3):
        logging.getLogger("opentelemetry.sdk._shared_internal").error("Exception while exporting")
        logging.getLogger("langfuse").warning("export failed")
    logging.getLogger("aero_rag.api").info("still logging")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and "still logging" in lines[1]
    assert any(isinstance(f, TracingOnceFilter) for f in logging.getLogger().handlers[0].filters)
