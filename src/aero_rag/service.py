"""The question-answering flow, independent of FastAPI so it can be tested and reused directly.

    input guardrail -> aircraft detection -> hybrid search -> abstain decision (abstain / clarify)
        -> generate -> output guardrail -> (one retry) -> answer or abstain

Each step is one span in the request's trace (tracing.py); without Langfuse keys the tracer does
nothing. Traces and logs carry the same kind of data: sizes, decisions, scores, ids, tokens, never
the question, the prompt or the model's text.
"""

import logging
import time
import uuid

from aero_rag.generation.generator import GenerationError, Generator
from aero_rag.guardrails.alerts import collect_alerts
from aero_rag.guardrails.component import mismatch_message
from aero_rag.guardrails.input import check_input
from aero_rag.guardrails.output import check_output, problem_kind
from aero_rag.guardrails.retrieval import decide, identifier_sources
from aero_rag.pricing import llm_cost
from aero_rag.retrieval.aircraft import AIRCRAFT_TYPES, resolve_aircraft
from aero_rag.retrieval.search import Searcher, SearchHit, SearchMode, SearchResult
from aero_rag.schemas import Citation, LLMAnswer, QueryResponse, Scores
from aero_rag.tracing import NULL_TRACER, QueryTrace, Tracer

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # the first try plus one retry
DONT_KNOW = "I don't know. "
INVALID_JSON = "Your reply was not valid JSON matching the required schema."


def _citation(hit: SearchHit) -> Citation:
    return Citation(**hit.model_dump(include=set(Citation.model_fields)))


def _hit_summary(hit: SearchHit) -> dict:
    """What a trace records about a retrieved task: ids and scores, never its text."""
    return {
        "chunk_id": f"{hit.parent_id}#{hit.matched_chunk_type}",
        "parent_id": hit.parent_id,
        "score": round(hit.score, 4),
        "cosine": round(hit.cosine, 4) if hit.cosine is not None else None,
        "is_latest": hit.is_latest,
    }


class RagService:
    def __init__(
        self,
        searcher: Searcher,
        generator: Generator,
        *,
        cosine_threshold: float,
        max_question_chars: int,
        search_k: int = 5,
        context_k: int = 3,
        tracer: Tracer = NULL_TRACER,
    ):
        self._searcher = searcher
        self._generator = generator
        self._cosine_threshold = cosine_threshold
        self._max_question_chars = max_question_chars
        self._search_k = search_k
        self._context_k = context_k
        self._tracer = tracer
        self._model = getattr(generator, "model", None)

    def answer(
        self, question: str, aircraft_type: str | None = None, request_id: str | None = None
    ) -> QueryResponse:
        request_id = request_id or uuid.uuid4().hex
        started = time.perf_counter()
        # Never the question text or the prompt: only sizes, decisions and scores are logged.
        trace: dict = {"request_id": request_id, "question_chars": len(question), "llm_calls": 0,
                       "prompt_tokens": 0, "completion_tokens": 0}  # fmt: skip
        # The texts are handed to the tracer, which sends them only if LANGFUSE_CAPTURE_TEXT is on.
        spans = self._tracer.start_query(
            request_id, {"question_chars": len(question), "aircraft_given": bool(aircraft_type)},
            question=question,
        )  # fmt: skip
        response = None
        try:
            response = self._answer(question, aircraft_type, request_id, trace, spans)
        finally:
            if response is not None:
                cited = [
                    f"{c.aircraft_type} {c.task_number} r{c.revision}" for c in response.citations
                ]
                trace |= {
                    "status": response.status,
                    "aircraft_unspecified": response.aircraft_unspecified,
                    "hybrid": response.scores.hybrid,
                    "cosine": response.scores.cosine,
                    "exact_identifier_match": response.scores.exact_identifier_match,
                    "citations": cited,
                }
            trace["duration_ms"] = round((time.perf_counter() - started) * 1000)
            trace["llm_cost_usd"] = round(
                llm_cost(trace["prompt_tokens"], trace["completion_tokens"])["total"], 6
            )
            spans.finish(
                status=response.status if response is not None else "error",
                aircraft_type=trace.get("aircraft_type"),
                output={k: v for k, v in trace.items() if k != "request_id"},
                text=_answer_text(response) if response is not None else None,
            )
        log.info("query", extra=trace)
        return response

    def _answer(
        self, question: str, aircraft_type: str | None, request_id: str, trace: dict,
        spans: QueryTrace,
    ) -> QueryResponse:  # fmt: skip
        def reply(status: str, answer: str, **fields) -> QueryResponse:
            return QueryResponse(status=status, answer=answer, request_id=request_id, **fields)

        # 1. Input guardrail (free).
        with spans.span("input_guardrail", as_type="guardrail",
                        input={"question_chars": len(question),
                               "max_chars": self._max_question_chars}) as span:  # fmt: skip
            reason = check_input(question, self._max_question_chars)
            span.update(output={"result": "rejected" if reason else "passed"})
        if reason:
            trace["reason"] = "input_rejected"
            return reply("rejected", reason)
        question = question.strip()
        given = aircraft_type.strip() if aircraft_type and aircraft_type.strip() else None

        # 2. Aircraft: given (normalised), detected in the question, or none (search all).
        with spans.span("aircraft_detection", input={"aircraft_given": given is not None}) as span:
            try:
                aircraft, source = resolve_aircraft(question, given)
            except ValueError:
                aircraft, source = None, "unknown"
            span.update(output={"aircraft_type": aircraft, "source": source,
                                "aircraft_unspecified": aircraft is None})  # fmt: skip
        if source == "unknown":
            trace["reason"] = "unknown_aircraft"
            known = ", ".join(AIRCRAFT_TYPES)
            return reply("rejected", f"Unknown aircraft type. Choose one of: {known}.")

        # 3. Retrieval (one embedding call). Latest revisions only, always.
        search_input = {"mode": SearchMode.HYBRID.value, "top_k": self._search_k,
                        "aircraft_filter": aircraft, "latest_only": True}  # fmt: skip
        with spans.span("retrieval", as_type="retriever", input=search_input) as span:
            result = self._searcher.search(
                question, aircraft if source == "given" else None, SearchMode.HYBRID, self._search_k
            )
            span.update(output={"hit_count": len(result.hits),
                                "hits": [_hit_summary(h) for h in result.hits]})  # fmt: skip
        trace |= {"aircraft_type": result.aircraft_type, "aircraft_source": result.aircraft_source}

        # 4. Abstain decision (free): abstain or ask before spending an LLM call.
        with spans.span("abstain_decision", as_type="guardrail",
                        input={"cosine_threshold": self._cosine_threshold,
                               "context_k": self._context_k}) as span:  # fmt: skip
            decision = decide(question, result, self._cosine_threshold, self._context_k)
            top_cosine = result.hits[0].cosine if result.hits else None
            span.update(output={
                "action": decision.action, "reason": decision.reason,
                "exact_identifier_match": decision.exact_identifier_match,
                "top_cosine": round(top_cosine, 4) if top_cosine is not None else None,
                "context_parent_ids": [h.parent_id for h in decision.context],
                "clarification_options": decision.options,
            })  # fmt: skip
        trace["reason"] = decision.reason
        common = self._common(decision.context, result, decision.exact_identifier_match)
        if decision.action == "abstain":
            not_found = DONT_KNOW + "I could not find this in the maintenance manuals."
            return reply("abstained", not_found, **common)
        if decision.action == "clarify":
            top_hit = decision.context[0]
            return reply(
                "clarification_needed",
                f"Which aircraft type do you mean? Task {top_hit.task_number} ({top_hit.title}) "
                f"exists for {', '.join(decision.options)}, with different values for each. "
                "Please ask again with the aircraft type.",
                clarification_options=decision.options,
                **common,
            )

        # 5. Generate, check, retry once with the problems listed, then abstain.
        # An abstention names the component the manual does cover when the question asked about a
        # different one (nose vs main wheel). The text is built by code from the index, never by
        # the model.
        context = decision.context
        mismatch = mismatch_message(question, context[0])
        identifiers = identifier_sources(question, context)
        problems: list[str] = []
        for attempt in range(MAX_ATTEMPTS):
            last = attempt + 1 == MAX_ATTEMPTS
            answer = self._generate(question, context, problems, attempt, trace, spans)

            with spans.span("output_guardrail", as_type="guardrail",
                            input={"attempt": attempt + 1}) as span:  # fmt: skip
                if answer is None:
                    problems, outcome = [INVALID_JSON], "abstain" if last else "retry"
                    kinds = ["invalid_json"]
                elif answer.status == "cannot_answer":
                    # An exact part or task number found in a source is decisive evidence: the
                    # model must answer from it. Retry once with the location spelled out.
                    if identifiers and not mismatch and not last:
                        problems, outcome = [_identifier_problem(identifiers)], "retry"
                        kinds = ["identifier_abstained"]
                        trace["identifier_retry"] = True
                    else:
                        outcome, kinds = "model_abstained", []
                else:
                    problems = check_output(answer, context, question)
                    outcome = "pass" if not problems else ("abstain" if last else "retry")
                    kinds = sorted({problem_kind(p) for p in problems})
                found = len(problems) if outcome in ("retry", "abstain") else 0
                span.update(output={"result": outcome, "problem_count": found,
                                    "problem_kinds": kinds})  # fmt: skip

            if outcome == "model_abstained":
                trace["reason"] = "model_abstained"
                not_found = "The maintenance tasks I found do not contain this information."
                return reply("abstained", DONT_KNOW + (mismatch or not_found), **common)
            if outcome == "pass":
                cited = [context[i - 1] for i in dict.fromkeys(answer.source_ids)]
                return reply(
                    "answered",
                    answer.answer,
                    steps=answer.steps,
                    warnings=collect_alerts(answer, context),
                    citations=[_citation(hit) for hit in cited],
                    **common,
                )
            if answer is not None and answer.status == "answered":
                trace.setdefault("guardrail_problems", []).append(problems)

        trace["reason"] = "output_guardrail_failed"
        unverified = (
            "I could not produce an answer that I can verify against the latest revision of the "
            "manual."
        )
        return reply("abstained", DONT_KNOW + (mismatch or unverified), **common)

    def _generate(
        self, question: str, context: list[SearchHit], problems: list[str], attempt: int,
        trace: dict, spans: QueryTrace,
    ) -> LLMAnswer | None:  # fmt: skip
        """One model call, traced as a generation. None if the reply was unusable."""
        call_input = {"attempt": attempt + 1, "retry": bool(problems),
                      "feedback_problems": len(problems),
                      "context_parent_ids": [h.parent_id for h in context]}  # fmt: skip
        with spans.span("generation", as_type="generation", model=self._model,
                        input=call_input) as span:  # fmt: skip
            started = time.perf_counter()
            try:
                generation = self._generator.generate(question, context, problems or None)
            except GenerationError:
                trace["llm_calls"] += 1
                latency_ms = round((time.perf_counter() - started) * 1000)
                span.update(level="ERROR", status_message="GenerationError",
                            metadata={"latency_ms": latency_ms})  # fmt: skip
                return None
            latency_ms = round((time.perf_counter() - started) * 1000)
            trace["llm_calls"] += 1
            trace["prompt_tokens"] += generation.prompt_tokens
            trace["completion_tokens"] += generation.completion_tokens
            answer = generation.answer
            span.update(
                usage_details={"input": generation.prompt_tokens,
                               "output": generation.completion_tokens,
                               "total": generation.prompt_tokens + generation.completion_tokens},
                cost_details=llm_cost(generation.prompt_tokens, generation.completion_tokens),
                metadata={"latency_ms": latency_ms},
                output={"status": answer.status, "source_ids": answer.source_ids,
                        "steps": len(answer.steps)},
            )  # fmt: skip
            return answer

    @staticmethod
    def _common(context: list[SearchHit], result: SearchResult, exact: bool) -> dict:
        top = (context or result.hits or [None])[0]
        return {
            "scores": Scores(
                hybrid=top.score if top else None,
                cosine=top.cosine if top else None,
                exact_identifier_match=exact,
            ),
            "aircraft_unspecified": result.aircraft_unspecified,
        }


def _answer_text(response: QueryResponse) -> dict:
    """The final, verified answer as the user saw it (for optional text capture in traces)."""
    text: dict = {"answer": response.answer}
    if response.steps:
        text["steps"] = [f"{s.number}. {s.text}" for s in response.steps]
    return text


def _identifier_problem(identifiers: dict[str, list[int]]) -> str:
    where = "; ".join(
        f"{ident} appears word for word in source {', '.join(map(str, ids))}"
        for ident, ids in identifiers.items()
    )
    return (
        f"You replied cannot_answer, but {where}. A question that is, or contains, an identifier "
        "asks what it is: answer from that source with what the item is (its description and "
        "quantity, or the task title), the task number, aircraft type and revision."
    )
