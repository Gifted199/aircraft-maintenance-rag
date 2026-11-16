"""Turn evaluation results into Markdown tables (data/results/) and a JSON record."""

import json
from pathlib import Path


def pct(part: float, whole: float) -> str:
    return "-" if not whole else f"{100 * part / whole:.0f}%"


def frac(part: int, whole: int) -> str:
    return f"{part}/{whole} ({pct(part, whole)})"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def headline_table(summary: dict, recall5: float) -> str:
    s = summary
    rows = [
        ["Recall@5 (hybrid, answerable and clarification questions)", f"{recall5:.2f}", "high"],
        ["Exact match: torque values", frac(s["torques_found"], s["torques_expected"]), "100%"],
        ["Exact match: part numbers", frac(s["parts_found"], s["parts_expected"]), "100%"],
        ["Step completeness (steps with text and alerts intact)",
         frac(s["steps_matched"], s["steps_expected"]), "100%"],
        ["WARNINGs and CAUTIONs kept with their step",
         frac(s["alerts_matched"], s["alerts_expected"]), "100%"],
        ["Answers citing an outdated revision", frac(s["outdated"], s["answered"]), "0"],
        ["Correct abstentions (unanswerable questions)",
         frac(s["correct_abstentions"], s["unanswerable"]), "100%"],
        ["Correct clarification requests",
         frac(s["correct_clarifications"], s["clarification_items"]), "100%"],
        ["Answerable questions answered", frac(s["answerable_answered"], s["answerable"]), "high"],
        ["Answerable questions citing the correct task", frac(s["target_cited"], s["answerable"]),
         "high"],
        ["Answers with a value not in the correct task",
         frac(s["answers_with_unexpected_values"], s["answerable"]), "0"],
        ["Questions fully correct (all checks above)", frac(s["fully_correct"], s["questions"]),
         "high"],
    ]  # fmt: skip
    return md_table(["Metric", "Result", "Target"], rows)


def category_table(rows: list[dict]) -> str:
    return md_table(
        ["Question type", "n", "Expected outcome", "Outcome correct", "Fully correct"],
        [[r["category"], r["n"], r["expected"], frac(r["outcome_correct"], r["n"]),
          frac(r["fully_correct"], r["n"])] for r in rows],
    )  # fmt: skip


def modes_table(rows: list[dict], metric: str) -> str:
    return md_table(
        ["Question type", "n", "Keyword (BM25)", "Vector (k-NN)", "Hybrid"],
        [[r["group"], r["n"],
          *(f"{r[f'{mode} {metric}']:.2f}" for mode in ("keyword", "vector", "hybrid"))]
         for r in rows],
    )  # fmt: skip


def filter_table(retrieval: dict[str, dict], e2e: list[dict]) -> str:
    rows = []
    for row in e2e:
        r = retrieval[row["setting"]]
        rows.append([
            row["setting"],
            frac(r["superseded_in_context"], r["n"]),
            frac(r["superseded_ranked_first"], r["n"]),
            frac(row["first_attempt_outdated"], row["first_attempts"]),
            frac(row["final_outdated"], row["answered"]),
            frac(row["answered"], row["n"]),
            frac(row["fully_correct"], row["n"]),
        ])  # fmt: skip
    return md_table(
        ["Latest-revision filter", "Old revision in the model's context",
         "Old revision ranked first", "Model's first draft cites or quotes an old revision",
         "Final answers citing an old revision", "Answered", "Fully correct"],
        rows,
    )  # fmt: skip


def items_table(rows: list[dict]) -> str:
    def values(r):
        parts = []
        if r.get("torques_expected"):
            parts.append(f"torque {r['torques_found']}/{r['torques_expected']}")
        if r.get("parts_expected"):
            parts.append(f"P/N {r['parts_found']}/{r['parts_expected']}")
        if r.get("steps_expected"):
            parts.append(f"steps {r['steps_matched']}/{r['steps_expected']}")
        return ", ".join(parts) or "-"

    return md_table(
        ["id", "type", "expected", "got", "cited", "values", "outdated", "LLM calls", "cosine",
         "correct"],
        [[r["id"], r["category"], r["expected"], r["status"], "; ".join(r.get("cited", [])) or "-",
          values(r), "yes" if r.get("outdated") else "-", r["llm_calls"],
          f"{r['cosine']:.2f}" if r.get("cosine") is not None else "-",
          "yes" if r["fully_correct"] else "NO"] for r in rows],
    )  # fmt: skip


def write_reports(out_dir: Path, results: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = results["meta"]
    header = (
        f"Run {m['run_at']} | {m['questions']} questions | index `{m['index']}` | "
        f"chat `{m['chat_deployment']}` | cosine threshold {m['cosine_threshold']} | "
        f"hybrid weights {m['hybrid_weights']}"
    )
    cost = results["cost"]
    metrics_md = "\n\n".join([
        "# Evaluation results",
        header,
        "## Headline metrics (hybrid search, latest-revision filter on)",
        headline_table(results["summary"], results["recall5_hybrid"]),
        "## By question type",
        category_table(results["categories"]),
        "## Retrieval: Recall@5 by search mode",
        modes_table(results["retrieval_modes"], "recall@5"),
        "## Retrieval: top-1 accuracy by search mode",
        modes_table(results["retrieval_modes"], "top1"),
        "## Latest-revision filter: off vs on (questions whose task has a superseded revision)",
        filter_table(results["filter_retrieval"], results["filter_e2e"]),
        "## Cost of this run",
        md_table(["LLM calls", "Retries", "Prompt tokens", "Completion tokens", "LLM cost (USD)",
                  "Questions embedded"],
                 [[cost["llm_calls"], results["summary_retries_total"], cost["prompt_tokens"],
                   cost["completion_tokens"], f"{cost['llm_cost_usd']:.3f}",
                   results["texts_embedded"]]]),
    ]) + "\n"  # fmt: skip
    items_md = "\n\n".join([
        "# Evaluation: every question (latest-revision filter on)",
        header,
        items_table(results["rows"]),
    ]) + "\n"  # fmt: skip

    paths = [out_dir / "metrics.md", out_dir / "questions.md", out_dir / "results.json"]
    paths[0].write_text(metrics_md, encoding="utf-8", newline="\n")
    paths[1].write_text(items_md, encoding="utf-8", newline="\n")
    paths[2].write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    return paths
