"""The question-answering flow, independent of FastAPI so it can be tested and reused directly.

    input guardrail -> hybrid search -> retrieval guardrail (abstain / clarify)
        -> generate -> output guardrail -> (one retry) -> answer or abstain
"""

import logging
import time
import uuid

from aero_rag.generation.generator import GenerationError, Generator
from aero_rag.guardrails.alerts import collect_alerts
from aero_rag.guardrails.component import mismatch_message
from aero_rag.guardrails.input import check_input
from aero_rag.guardrails.output import check_output
from aero_rag.guardrails.retrieval import decide, identifier_sources
from aero_rag.retrieval.aircraft import AIRCRAFT_TYPES, normalise_aircraft
from aero_rag.retrieval.search import Searcher, SearchHit, SearchMode
from aero_rag.schemas import Citation, QueryResponse, Scores

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # the first try plus one retry
DONT_KNOW = "I don't know. "


def _citation(hit: SearchHit) -> Citation:
    return Citation(**hit.model_dump(include=set(Citation.model_fields)))


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
    ):
        self._searcher = searcher
        self._generator = generator
        self._cosine_threshold = cosine_threshold
        self._max_question_chars = max_question_chars
        self._search_k = search_k
        self._context_k = context_k

    def answer(
        self, question: str, aircraft_type: str | None = None, request_id: str | None = None
    ) -> QueryResponse:
        request_id = request_id or uuid.uuid4().hex
        started = time.perf_counter()
        # Never the question text or the prompt: only sizes, decisions and scores are logged.
        trace: dict = {"request_id": request_id, "question_chars": len(question), "llm_calls": 0,
                       "prompt_tokens": 0, "completion_tokens": 0}  # fmt: skip
        response = self._answer(question, aircraft_type, request_id, trace)
        cited = [f"{c.aircraft_type} {c.task_number} r{c.revision}" for c in response.citations]
        trace |= {
            "status": response.status,
            "aircraft_unspecified": response.aircraft_unspecified,
            "hybrid": response.scores.hybrid,
            "cosine": response.scores.cosine,
            "exact_identifier_match": response.scores.exact_identifier_match,
            "citations": cited,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }
        log.info("query", extra=trace)
        return response

    def _answer(
        self, question: str, aircraft_type: str | None, request_id: str, trace: dict
    ) -> QueryResponse:
        def reply(status: str, answer: str, **fields) -> QueryResponse:
            return QueryResponse(status=status, answer=answer, request_id=request_id, **fields)

        # 1. Input guardrail (free).
        if reason := check_input(question, self._max_question_chars):
            trace["reason"] = "input_rejected"
            return reply("rejected", reason)
        question = question.strip()
        if aircraft_type and aircraft_type.strip():
            try:
                aircraft_type = normalise_aircraft(aircraft_type)
            except ValueError:
                trace["reason"] = "unknown_aircraft"
                known = ", ".join(AIRCRAFT_TYPES)
                return reply("rejected", f"Unknown aircraft type. Choose one of: {known}.")
        else:
            aircraft_type = None

        # 2. Retrieval (one embedding call). Latest revisions only, always.
        result = self._searcher.search(question, aircraft_type, SearchMode.HYBRID, self._search_k)
        trace |= {"aircraft_type": result.aircraft_type, "aircraft_source": result.aircraft_source}

        # 3. Retrieval guardrail (free): abstain or ask before spending an LLM call.
        decision = decide(question, result, self._cosine_threshold, self._context_k)
        trace["reason"] = decision.reason
        top = (decision.context or result.hits or [None])[0]
        common = {
            "scores": Scores(
                hybrid=top.score if top else None,
                cosine=top.cosine if top else None,
                exact_identifier_match=decision.exact_identifier_match,
            ),
            "aircraft_unspecified": result.aircraft_unspecified,
        }
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

        # 4. Generate, check, retry once with the problems listed, then abstain.
        # An abstention names the component the manual does cover when the question asked about a
        # different one (nose vs main wheel). The text is built by code from the index, never by
        # the model.
        mismatch = mismatch_message(question, decision.context[0])
        identifiers = identifier_sources(question, decision.context)
        problems: list[str] = []
        for attempt in range(MAX_ATTEMPTS):
            try:
                generation = self._generator.generate(question, decision.context, problems or None)
            except GenerationError:
                trace["llm_calls"] += 1
                problems = ["Your reply was not valid JSON matching the required schema."]
                continue
            trace["llm_calls"] += 1
            trace["prompt_tokens"] += generation.prompt_tokens
            trace["completion_tokens"] += generation.completion_tokens
            answer = generation.answer
            if answer.status == "cannot_answer":
                # An exact part or task number found in a source is decisive evidence: the model
                # must answer from it. Retry once with the location spelled out.
                if identifiers and not mismatch and attempt + 1 < MAX_ATTEMPTS:
                    trace["identifier_retry"] = True
                    problems = [_identifier_problem(identifiers)]
                    continue
                trace["reason"] = "model_abstained"
                not_found = "The maintenance tasks I found do not contain this information."
                return reply("abstained", DONT_KNOW + (mismatch or not_found), **common)
            problems = check_output(answer, decision.context, question)
            if not problems:
                cited = [decision.context[i - 1] for i in dict.fromkeys(answer.source_ids)]
                return reply(
                    "answered",
                    answer.answer,
                    steps=answer.steps,
                    warnings=collect_alerts(answer, decision.context),
                    citations=[_citation(hit) for hit in cited],
                    **common,
                )
            trace.setdefault("guardrail_problems", []).append(problems)

        trace["reason"] = "output_guardrail_failed"
        unverified = (
            "I could not produce an answer that I can verify against the latest revision of the "
            "manual."
        )
        return reply("abstained", DONT_KNOW + (mismatch or unverified), **common)


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
