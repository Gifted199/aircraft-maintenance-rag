"""Run the evaluation: retrieval in three modes, the latest-revision filter on and off, and the
full question-answering service end to end.

Everything external is passed in, so the same runner works against the live index and model, or
offline against fakes in the tests. Two small wrappers make the comparisons possible without
changing the service:
  * CachingEmbedder   embeds each question once, however many runs use it (saves paid calls);
  * RecordingGenerator records every model call (tokens, retries, the first attempt's citations);
  * SupersededSearcher turns the latest-revision filter off, for the before/after comparison.
"""

from typing import Any

from aero_rag.evaluation.dataset import EvalItem
from aero_rag.generation.generator import Generation, GenerationError
from aero_rag.guardrails.input import check_input
from aero_rag.retrieval.search import SearchMode

# Price assumptions for the estimate (USD per 1M tokens); see CLAUDE.md.
LLM_INPUT_PRICE = 0.75
LLM_OUTPUT_PRICE = 4.50
EMBED_PRICE = 0.02
# Measured in phase 3: about 2,600 prompt tokens per call; short answers about 60 output tokens,
# full procedures about 600. Rounded up for the estimate.
EST_PROMPT_TOKENS = 2800
EST_OUTPUT_SHORT = 100
EST_OUTPUT_PROCEDURE = 800
EST_EMBED_TOKENS = 20


class CachingEmbedder:
    def __init__(self, inner: Any):
        self._inner = inner
        self.batch_size = inner.batch_size
        self._cache: dict[str, list[float]] = {}
        self.texts_embedded = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, self._inner.embed(missing), strict=True))
            self.texts_embedded += len(missing)
        return [self._cache[t] for t in texts]


class RecordingGenerator:
    def __init__(self, inner: Any):
        self._inner = inner
        self.calls: list[dict] = []

    def generate(self, question, context, problems=None) -> Generation:
        entry: dict = {"retry": bool(problems), "prompt_tokens": 0, "completion_tokens": 0}
        self.calls.append(entry)
        try:
            generation = self._inner.generate(question, context, problems)
        except GenerationError:
            entry["error"] = "GenerationError"
            raise
        answer = generation.answer
        entry |= {
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "status": answer.status,
            "text": " ".join([answer.answer, *[s.text for s in answer.steps]]),
            "cited": [
                {
                    "aircraft_type": context[i - 1].aircraft_type,
                    "task_number": context[i - 1].task_number,
                    "revision": context[i - 1].revision,
                    "is_latest": context[i - 1].is_latest,
                }
                for i in dict.fromkeys(answer.source_ids)
                if 1 <= i <= len(context)
            ],
        }
        return generation


class SupersededSearcher:
    """Evaluation only: the same search with the latest-revision filter switched off."""

    def __init__(self, inner: Any):
        self._inner = inner

    def search(self, query, aircraft_type=None, mode=SearchMode.HYBRID, k=5, **_ignored):
        return self._inner.search(query, aircraft_type, mode, k, include_superseded=True)


def run_retrieval(
    searcher: Any, items: list[EvalItem], mode: SearchMode, include_superseded: bool = False
) -> dict[str, dict]:
    out = {}
    for item in items:
        if not item.relevant_parent_ids:
            continue
        result = searcher.search(item.question, item.aircraft_type, mode, 5, include_superseded)
        out[item.id] = {
            "aircraft_unspecified": result.aircraft_unspecified,
            "hits": [
                {
                    "parent_id": h.parent_id,
                    "aircraft_type": h.aircraft_type,
                    "task_number": h.task_number,
                    "revision": h.revision,
                    "is_latest": h.is_latest,
                    "score": h.score,
                    "cosine": h.cosine,
                }
                for h in result.hits
            ],
        }
    return out


def run_end_to_end(service: Any, recorder: RecordingGenerator, items: list[EvalItem]) -> dict:
    out = {}
    for item in items:
        start = len(recorder.calls)
        record: dict = {}
        try:
            response = service.answer(item.question, item.aircraft_type, f"eval-{item.id}")
            record["response"] = response.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
            record["error_type"] = type(exc).__name__
        calls = recorder.calls[start:]
        record |= {
            "llm_calls": len(calls),
            "retries": sum(c["retry"] for c in calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
        }
        # What the model proposed before any guardrail feedback: shows what the checks caught.
        if calls and calls[0].get("status") == "answered":
            record["first_attempt"] = {"text": calls[0]["text"], "cited": calls[0]["cited"]}
        out[item.id] = record
    return out


def llm_items(items: list[EvalItem]) -> list[EvalItem]:
    """Questions expected to reach the model: answerable ones, and in-domain unanswerable ones
    (they pass the similarity threshold, so the model is the one that must abstain)."""
    return [
        i for i in items if i.expected_status == "answered" or i.unanswerable_kind == "in_domain"
    ]


def trap_items(items: list[EvalItem]) -> list[EvalItem]:
    return [i for i in items if i.is_trap]


def estimate(items: list[EvalItem], max_question_chars: int) -> dict:
    def cost(group: list[EvalItem]) -> float:
        out = sum(EST_OUTPUT_PROCEDURE if i.expected_steps else EST_OUTPUT_SHORT for i in group)
        tokens_in = len(group) * EST_PROMPT_TOKENS
        return (tokens_in * LLM_INPUT_PRICE + out * LLM_OUTPUT_PRICE) / 1e6

    main, traps = llm_items(items), trap_items(items)
    questions = {i.question for i in items if check_input(i.question, max_question_chars) is None}
    expected_cost = cost(main) + cost(traps)
    return {
        "questions": len(items),
        "embedding_calls": len(questions),
        "embedding_cost_usd": len(questions) * EST_EMBED_TOKENS * EMBED_PRICE / 1e6,
        "llm_calls_main_run": len(main),
        "llm_calls_filter_off_run": len(traps),
        "llm_calls_expected": len(main) + len(traps),
        "llm_calls_max": 2 * (len(main) + len(traps)),  # every call retried once
        "llm_cost_expected_usd": expected_cost,
        "llm_cost_max_usd": 2 * expected_cost,
    }


def actual_cost(records: list[dict]) -> dict:
    prompt = sum(r.get("prompt_tokens", 0) for r in records)
    completion = sum(r.get("completion_tokens", 0) for r in records)
    return {
        "llm_calls": sum(r.get("llm_calls", 0) for r in records),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "llm_cost_usd": (prompt * LLM_INPUT_PRICE + completion * LLM_OUTPUT_PRICE) / 1e6,
    }
