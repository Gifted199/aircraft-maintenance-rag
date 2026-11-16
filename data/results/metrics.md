# Evaluation results

Run 2026-11-16 14:22 UTC | 47 questions | index `aero-manuals` | chat `gpt-5.4-mini` | cosine threshold 0.35 | hybrid weights [0.5, 0.5]

## Headline metrics (hybrid search, latest-revision filter on)

| Metric | Result | Target |
|---|---|---|
| Recall@5 (hybrid, answerable and clarification questions) | 1.00 | high |
| Exact match: torque values | 15/15 (100%) | 100% |
| Exact match: part numbers | 13/13 (100%) | 100% |
| Step completeness (steps with text and alerts intact) | 61/61 (100%) | 100% |
| WARNINGs and CAUTIONs kept with their step | 23/23 (100%) | 100% |
| Answers citing an outdated revision | 0/34 (0%) | 0 |
| Correct abstentions (unanswerable questions) | 9/9 (100%) | 100% |
| Correct clarification requests | 4/4 (100%) | 100% |
| Answerable questions answered | 34/34 (100%) | high |
| Answerable questions citing the correct task | 34/34 (100%) | high |
| Answers with a value not in the correct task | 0/34 (0%) | 0 |
| Questions fully correct (all checks above) | 47/47 (100%) | high |

## By question type

| Question type | n | Expected outcome | Outcome correct | Fully correct |
|---|---|---|---|---|
| torque | 11 | answered | 11/11 (100%) | 11/11 (100%) |
| part_number | 8 | answered | 8/8 (100%) | 8/8 (100%) |
| procedure | 6 | answered | 6/6 (100%) | 6/6 (100%) |
| no_aircraft | 4 | answered | 4/4 (100%) | 4/4 (100%) |
| identifier | 5 | answered | 5/5 (100%) | 5/5 (100%) |
| clarification | 4 | clarification_needed | 4/4 (100%) | 4/4 (100%) |
| unanswerable | 9 | abstained, rejected | 9/9 (100%) | 9/9 (100%) |

## Retrieval: Recall@5 by search mode

| Question type | n | Keyword (BM25) | Vector (k-NN) | Hybrid |
|---|---|---|---|---|
| torque | 11 | 1.00 | 1.00 | 1.00 |
| part_number | 8 | 1.00 | 1.00 | 1.00 |
| procedure | 6 | 1.00 | 1.00 | 1.00 |
| no_aircraft | 4 | 1.00 | 1.00 | 1.00 |
| identifier | 5 | 1.00 | 0.80 | 1.00 |
| clarification | 4 | 1.00 | 1.00 | 1.00 |
| all | 38 | 1.00 | 0.97 | 1.00 |

## Retrieval: top-1 accuracy by search mode

| Question type | n | Keyword (BM25) | Vector (k-NN) | Hybrid |
|---|---|---|---|---|
| torque | 11 | 1.00 | 1.00 | 1.00 |
| part_number | 8 | 0.88 | 1.00 | 1.00 |
| procedure | 6 | 1.00 | 1.00 | 1.00 |
| no_aircraft | 4 | 1.00 | 1.00 | 1.00 |
| identifier | 5 | 1.00 | 0.40 | 1.00 |
| clarification | 4 | 1.00 | 1.00 | 1.00 |
| all | 38 | 0.97 | 0.92 | 1.00 |

## Latest-revision filter: off vs on (questions whose task has a superseded revision)

| Latest-revision filter | Old revision in the model's context | Old revision ranked first | Model's first draft cites or quotes an old revision | Final answers citing an old revision | Answered | Fully correct |
|---|---|---|---|---|---|---|
| on (default) | 0/13 (0%) | 0/13 (0%) | 0/13 (0%) | 0/13 (0%) | 13/13 (100%) | 13/13 (100%) |
| off (include_superseded) | 13/13 (100%) | 7/13 (54%) | 0/13 (0%) | 0/13 (0%) | 13/13 (100%) | 13/13 (100%) |

## Cost of this run

| LLM calls | Retries | Prompt tokens | Completion tokens | LLM cost (USD) | Questions embedded |
|---|---|---|---|---|---|
| 51 | 0 | 134877 | 7837 | 0.136 | 46 |
