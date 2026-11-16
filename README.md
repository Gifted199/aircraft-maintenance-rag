# Aircraft maintenance manual assistant

A question-answering assistant over aircraft maintenance manuals: hybrid search (BM25 + k-NN) on
Amazon OpenSearch, answers from Azure OpenAI with citations, guardrails that verify every torque
value and part number, deployed on AWS ECS Fargate.

> The full README (architecture diagram, how to run, configuration, "Future: agents") is written in
> phase 7. This draft contains the Metrics section only.

## Metrics

All numbers come from one command, `uv run python scripts/evaluate.py` (52 LLM calls, about $0.13),
run on 16 November 2026 against the live index. Full tables: [data/results/metrics.md](data/results/metrics.md);
every question and its outcome: [data/results/questions.md](data/results/questions.md).

**The evaluation set.** 47 questions written to sound like a technician, labelled from the same
JSON spec the manuals were generated from, so every expected value is correct by construction
([data/eval/questions.json](data/eval/questions.json)): torque (11), part number (8), full
procedure (6), no aircraft named but only one aircraft has the task (4), a bare part or task
number (5), must ask which aircraft (4), and cannot be answered (9: off-topic, in-domain but not
in the manuals, a superseded part number, a prompt injection).

### Headline results (hybrid search, latest-revision filter on)

| Metric | Result | Target |
|---|---|---|
| Recall@5 (correct task in the top 5) | 1.00 | high |
| Exact match: torque values | 15/15 (100%) | 100% |
| Exact match: part numbers | 12/13 (92%) | 100% |
| Step completeness (text, WARNINGs and CAUTIONs intact) | 61/61 steps (100%) | 100% |
| WARNINGs and CAUTIONs kept with their step | 23/23 (100%) | 100% |
| **Answers citing an outdated revision** | **0/34 (0%)** | **0** |
| Correct abstentions on unanswerable questions | 8/9 (89%) | 100% |
| Correct clarification requests (which aircraft?) | 4/4 (100%) | 100% |
| Answers containing a value from the wrong task | 0/34 (0%) | 0 |
| Questions fully correct | 45/47 (96%) | high |

**The two failures, and what they show.**

- *"B340-28-5209"* (a bare part number). Retrieval found the right task first and the exact-match
  rule let it through, but the model replied that it could not answer: a part number alone is not
  phrased as a question. This is the one missed part number above. The same kind of question,
  "A200-32-4112", was answered correctly.
- *"What is the torque for the A-200 nose wheel axle nut?"* The manuals cover only the main wheel
  axle nut, but the model answered with the main wheel value, 135 Nm. The guardrails cannot catch
  this: the value really is in the cited task. It is a wrong-component error, the kind that needs
  either a stricter prompt rule or a check that the component named in the question matches the
  task title.

Both have a cheap candidate fix in the prompt; neither is applied yet because each re-run costs
about $0.13 and the fix must be measured, not assumed.

### Before/after 1: keyword vs vector vs hybrid search

| Question type | n | Recall@5 keyword | Recall@5 vector | Recall@5 hybrid | Top-1 keyword | Top-1 vector | Top-1 hybrid |
|---|---|---|---|---|---|---|---|
| torque | 11 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| part number | 8 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 |
| procedure | 6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| no aircraft named | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **bare part or task number** | 5 | 1.00 | **0.80** | 1.00 | 1.00 | **0.40** | 1.00 |
| clarification | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **all** | 38 | 1.00 | 0.97 | 1.00 | 0.97 | 0.92 | **1.00** |

Vector search fails exactly where expected: a bare part number such as "A200-32-4112" carries
almost no meaning to embed, so vector search put the right task first for only 2 of 5 identifier
questions, and for "A200-32-4112" the right task was not in its top 5 at all. Keyword search
(BM25) matches the code exactly and never missed one, but it ranked the wrong task first for a
plain-language question ("part number of the main battery on the A-200" put the cabin pressure
sensor first, because both tasks contain "part number" and "A-200" many times). Hybrid search
combines both and was the only mode with every correct task ranked first. Caveat: the corpus is
small (8 tasks per aircraft once the aircraft filter is applied), so Recall@5 is nearly saturated
for every mode; top-1 is the more telling number, and each difference here is one to three
questions.

### Before/after 2: latest-revision filter off vs on

13 "trap" questions ask about tasks whose older revision gives a different value (for example
the A-200 axle nut: 120 Nm in revisions 1 and 2, 135 Nm in revision 3).

| Latest-revision filter | Old revision in the model's context | Old revision ranked first | Model's first draft cites or quotes an old revision | Final answers citing an old revision |
|---|---|---|---|---|
| off | 13/13 (100%) | 7/13 (54%) | 1/12 (8%) | 0/12 (0%) |
| **on (default)** | **0/13 (0%)** | **0/13 (0%)** | **0/12 (0%)** | **0/12 (0%)** |

Without the filter, an old revision reached the model in every trap question and was ranked first
in more than half: old and new revisions are word-for-word identical except for one value, so no
ranking method can tell them apart. The model usually picked the newest because each source is
labelled with its revision number, but in 1 of 12 answers its first draft used a superseded
revision (the A-200 lock bolt question). The output guardrail rejected that draft and the retry
was correct, so the final count was 0 either way. That is the point of having two layers: the
filter guarantees old revisions are never candidates, and the guardrail is the backup that caught
the one case where relying on the model alone would have shipped an outdated part number. (12
rather than 13 in the last two columns because one question was not answered in either setting.)

### Caveats

- The corpus is synthetic and small (31 documents, 24 current tasks); real manuals would make
  retrieval harder and the differences between search modes larger.
- Six unanswerable questions overlap with the probe questions used to choose the abstention
  threshold (0.35 cosine), so the abstention result is partly in-sample. Off-topic questions
  scored 0.07 to 0.30, well clear of the threshold.
- Each result is one run of a model that is not fully deterministic; a repeat run may differ by a
  question or two.
