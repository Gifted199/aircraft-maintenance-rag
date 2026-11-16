# Aircraft maintenance manual assistant

A question-answering assistant over aircraft maintenance manuals: hybrid search (BM25 + k-NN) on
Amazon OpenSearch, answers from Azure OpenAI with citations, guardrails that verify every torque
value and part number, deployed on AWS ECS Fargate.

> The full README (architecture diagram, how to run, configuration, "Future: agents") is written in
> phase 7. This draft contains the Metrics section only.

## Metrics

All numbers come from one command, `uv run python scripts/evaluate.py` (51 LLM calls, about $0.14),
run on 16 November 2026 against the live index. Full tables: [data/results/metrics.md](data/results/metrics.md);
every question and its outcome: [data/results/questions.md](data/results/questions.md). The first
run is kept as the v1 baseline in [data/results/v1-baseline/](data/results/v1-baseline/); what
changed and why is in [docs/EVALUATION.md](docs/EVALUATION.md).

| Run | Change | Fully correct | Part numbers | Correct abstentions | Outdated citations | LLM calls | Cost |
|---|---|---|---|---|---|---|---|
| v1 baseline | first full run | 45/47 (96%) | 12/13 | 8/9 | 0/34 | 52 | $0.131 |
| **v1 → v2** | identifier rule, component rule and check | **47/47 (100%)** | **13/13** | **9/9** | 0/34 | 51 | $0.136 |

**The evaluation set.** 47 questions written to sound like a technician, labelled from the same
JSON spec the manuals were generated from, so every expected value is correct by construction
([data/eval/questions.json](data/eval/questions.json)): torque (11), part number (8), full
procedure (6), no aircraft named but only one aircraft has the task (4), a bare part or task
number (5), must ask which aircraft (4), and cannot be answered (9: off-topic, in-domain but not
in the manuals, a superseded part number, a prompt injection).

### Headline results (hybrid search, latest-revision filter on)

| Metric | v1 baseline | v2 | Target |
|---|---|---|---|
| Recall@5 (correct task in the top 5) | 1.00 | 1.00 | high |
| Exact match: torque values | 15/15 (100%) | 15/15 (100%) | 100% |
| Exact match: part numbers | 12/13 (92%) | 13/13 (100%) | 100% |
| Step completeness (text, WARNINGs and CAUTIONs intact) | 61/61 steps (100%) | 61/61 steps (100%) | 100% |
| WARNINGs and CAUTIONs kept with their step | 23/23 (100%) | 23/23 (100%) | 100% |
| **Answers citing an outdated revision** | **0/34 (0%)** | **0/34 (0%)** | **0** |
| Correct abstentions on unanswerable questions | 8/9 (89%) | 9/9 (100%) | 100% |
| Correct clarification requests (which aircraft?) | 4/4 (100%) | 4/4 (100%) | 100% |
| Answers containing a value from the wrong task | 0/34 (0%) | 0/34 (0%) | 0 |
| Questions fully correct | 45/47 (96%) | 47/47 (100%) | high |

**The two v1 failures, and how v2 fixed them.**

- *"B340-28-5209"* (a bare part number). In v1, retrieval found the right task first, but the
  model replied that it could not answer: a part number alone is not phrased as a question.
  *Fix:* a prompt rule that a question which is an identifier asks what it is, plus a code safety
  net: if the model still abstains while the identifier appears word for word in a source, the
  service retries once and names that source. In v2 the model answered correctly first time.
- *"What is the torque for the A-200 nose wheel axle nut?"* The manuals cover only the main wheel
  axle nut, and in v1 the model answered with the main wheel value, 135 Nm. The value checks could
  not catch it, because 135 Nm really is in the cited task. *Fix:* a prompt rule to answer only
  about the exact component named, plus a deterministic check: position words (nose/main,
  left/right, inboard/outboard, forward/aft, upper/lower) in the question must not contradict the
  cited task's title. The abstention is built by code and names what the manual does cover: "The
  manuals do not cover this for the nose; the closest task is for the main: A-200 task 32-41-01
  revision 3". In v2 the model abstained correctly first time.

No other question changed outcome (the per-question comparison is in
[docs/EVALUATION.md](docs/EVALUATION.md)). Two honest limits: both fixes were written after
seeing these two questions fail, so 47/47 is in-sample for them; and in v2 the prompt rules alone
were enough, so the code safety nets are proven by offline tests, not by this run.

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

| Latest-revision filter | Old revision in the model's context | Old revision ranked first | Model's first draft cites or quotes an old revision (v1 / v2) | Final answers citing an old revision (v1 / v2) |
|---|---|---|---|---|
| off | 13/13 (100%) | 7/13 (54%) | 1/12 / 0/13 | 0/12 / 0/13 |
| **on (default)** | **0/13 (0%)** | **0/13 (0%)** | **0/12 / 0/13** | **0/12 / 0/13** |

Without the filter, an old revision reached the model in every trap question and was ranked first
in more than half: old and new revisions are word-for-word identical except for one value, so no
ranking method can tell them apart. The model usually picked the newest because each source is
labelled with its revision number, but in the v1 run its first draft for one question used a
superseded revision (the A-200 lock bolt). The output guardrail rejected that draft and the retry
was correct, so the final count was 0 either way. In v2 the model got that question right first
time, which shows the slip is rare and varies from run to run (1 of 25 drafts across both runs).
That is the point of having two layers: the filter guarantees old revisions are never candidates,
and the guardrail is the backup for the rare case where relying on the model alone would ship an
outdated part number. (v1 has 12 rather than 13 in the last two columns because the I02 question
was not answered in v1.)

### Caveats

- The corpus is synthetic and small (31 documents, 24 current tasks); real manuals would make
  retrieval harder and the differences between search modes larger.
- Six unanswerable questions overlap with the probe questions used to choose the abstention
  threshold (0.35 cosine), so the abstention result is partly in-sample. Off-topic questions
  scored 0.07 to 0.30, well clear of the threshold.
- Each result is one run of a model that is not fully deterministic; a repeat run may differ by a
  question or two.
- The v2 fixes target two known failures; a fresh set of questions written without looking at
  the results (other bare identifiers, other nose/main or left/right near misses) is the honest
  next test of whether they generalise.
