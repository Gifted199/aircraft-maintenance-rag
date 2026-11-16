# Plan: Aircraft maintenance manual assistant

Presentation: **Monday 16 November 2026, 4:30pm UK**. Plan written Sunday 15 November.

## 1. What we are building

A question-answering assistant over aircraft maintenance manuals. A mechanic picks an aircraft type,
asks a question ("What is the torque for the main wheel axle nut?"), and gets an answer with
citations to the exact task and revision, or an honest "I don't know".

```
 spec.json ──► generator ──► manuals (Markdown) ──► S3 bucket
                   │                                   │
                   └──► eval labels                    ▼
                                              ingestion job
                                   (parse → 1 chunk per task → embed)
                                                       │
                                                       ▼
                                   Amazon OpenSearch (BM25 + k-NN index)
                                                       ▲
 Streamlit UI ──► FastAPI /query ──► guardrails ──► hybrid search ──► Azure OpenAI chat
   (same ECS task)        │                                               │
                          └──────────── answer + citations + scores ◄─────┘
                                   Langfuse traces every step
```

Runtime: one Docker image in ECR, run on ECS Fargate as one task with two containers (API and UI).
Secrets come from AWS Secrets Manager.

## 2. Design choices and why

Each choice is written so it can be defended in one or two sentences.

### Documents
- **Synthetic manuals generated from a JSON spec.** Real manuals are copyrighted and we would have
  to hand-label answers; generating from a spec means every torque value, part number and step in
  the evaluation set is known to be correct because it is the same data the document was built from.
- **Fictional aircraft types, with a "synthetic, not for maintenance use" banner.** Nobody can
  mistake a demo answer for real airworthiness data.
- **Markdown, one file per task revision: 31 documents** (3 aircraft x 8 tasks = 24 latest, plus 7
  superseded). Markdown keeps headings, numbered steps and WARNING blocks explicit, which is what
  structure-aware chunking needs; it stands in for the structured formats (S1000D/XML) real manuals use.
- **Three aircraft: A-200, B-340, C-75.** Six tasks (axle nut, cabin pressure sensor, brake, pitot
  probe, fuel boost pump, battery) exist on all three with the same task number and title but
  different torque values and part numbers, so retrieving the wrong aircraft gives a wrong answer
  and the aircraft-type filter is properly tested. Each aircraft also has two tasks of its own.
- **Superseded revisions are deliberately included: 6 tasks, 7 old documents, each with a changed
  torque value or part number.** This is the trap the "outdated revision" metric measures: the old
  and new documents are otherwise word-for-word identical. One task (A-200 axle nut) has two old revisions.
- **A document does not say whether it is current.** Its header gives revision, date and what it
  supersedes, exactly like a printed page, which cannot know it has been replaced. Ingestion decides
  `is_latest` by comparing revisions, so that logic is real and tested rather than read off a label.
- **WARNING and CAUTION sit inside their step block, before the action.** The reader sees the hazard
  first, and a step block is a self-contained unit that cannot be cut away from its warning.
- **Torques and part numbers are listed in tables and written in the step text, and the spec loader
  rejects any mismatch.** The summary table can never disagree with the procedure.
- **Deterministic string templating, no LLM.** Free, reproducible, and a test fails if the files on
  disk are not exactly what the spec renders.
- **The parser is strict and shares one data model with the generator.** Each document must parse
  back to exactly its spec entry; anything unexpected stops ingestion rather than indexing a
  half-understood procedure.

### Chunking and metadata
- **One parent and two children per task revision (93 index documents for 31 manuals).** The
  *parent* is the full task; it is not embedded or searched, it is what the LLM reads. The
  *reference* child (header details, tools, parts table, torque table) and the *procedure* child
  (every step with its WARNINGs and CAUTIONs) are what search runs on.
- **Why children for search and parents for the answer.** A short, focused child matches a question
  more sharply than a whole task ("what torque" lands on the reference child, "how do I" on the
  procedure child). Handing the model the whole parent then guarantees it sees the complete task,
  so it can never answer from steps 4 to 7 without the WARNING on step 3.
- **The procedure is never split.** Chunks are built from the parsed structure, not by counting
  characters. If a procedure were ever too large to embed whole, ingestion stops with an error
  rather than cutting it.
- **Every chunk starts with a one-line header** (aircraft | task number | title | revision) so it
  is understandable on its own and both keyword and vector search see the aircraft and task.
- **Only children are embedded** (62 texts, one API call). Parents are fetched by id, so embedding
  them would be paying for vectors nobody searches.
- **Metadata on every chunk:** `aircraft_type`, `ata_chapter`, `task_number`, `revision`,
  `revision_date`, `is_latest`. These are stored as exact-match fields so they can be filtered, not searched.
- **`is_latest` is computed at ingestion** by comparing all revisions of the same aircraft + task,
  and older ones are flipped to `false` when a newer one arrives. Deciding this once at write time is
  cheaper and less error-prone than working it out on every query.
- **Deterministic document IDs** (`A-200_32-41-01_rev03`, plus `#reference` / `#procedure`).
  Re-running ingestion overwrites instead of duplicating, so the job is safe to run twice.
- **Ingestion always reads the whole bucket and parses everything before writing anything.**
  `is_latest` depends on every revision present, so a full pass is the simple way to keep it right
  (a new revision flips the old one), and one unparseable file stops the job with nothing half-written.
  Known limit: a document deleted from S3 is not removed from the index; `--recreate-index` rebuilds it.
- **Strict mapping.** The index rejects any field we did not declare, so a typo in a field name
  fails loudly instead of silently creating an unfiltered column.

### Retrieval
- **Hybrid search (BM25 + k-NN).** Keyword search (BM25) is exact on part numbers like `AX2-32-4411`,
  which embeddings blur; vector search (k-NN) handles paraphrased questions. Hybrid gets both.
- **OpenSearch search pipeline with a normalization processor.** BM25 and cosine scores are on
  different scales; the pipeline rescales and combines them inside OpenSearch, so we make one request
  instead of merging two result lists in Python.
- **Pipeline `aero-manuals-hybrid`: min-max normalisation, weighted arithmetic mean, weights
  0.5 / 0.5** (keyword, vector). Equal weights are the neutral starting point; phase 5 measures
  keyword-only, vector-only and hybrid so the weighting is justified by numbers.
- **Filters applied inside both sub-queries, not afterwards:** `is_latest = true` always, and
  `aircraft_type` when known. Filtering afterwards would let old or wrong-aircraft chunks occupy
  top places and then be discarded, leaving fewer than k results; inside, they are never candidates.
  `include_superseded` exists only so evaluation can show what happens without the filter.
- **Aircraft type is optional.** If it is not given, it is detected from the question (A-200, A200,
  "a 200"); a part number such as `A200-32-4112` does not count, and two aircraft in one question
  is treated as none. With no aircraft we search all and set `aircraft_unspecified`, because a
  wrong guess on a hard filter is worse than no guess. Phase 3 uses that flag: when the top results
  are the same task on different aircraft, the assistant must ask which aircraft rather than pick one.
- **Collapse children to parents in Python.** The hybrid query in OpenSearch 2.19 has no field
  collapsing, so we fetch 4 children per wanted result, keep the best child per parent, and fetch
  the parents by id.
- **Lucene k-NN engine.** It applies the filter while searching the vector graph, which is what
  "filter inside the sub-query" needs, and suits a small index.
- **Each hit carries two scores:** the mode's own score and the raw cosine similarity of the best
  child (computed from the stored vector). Measured on the live index, the right task scores about
  0.66 to 0.73 cosine and unrelated tasks 0.38 to 0.59, which is the gap the abstention threshold uses.
- **"I don't know" threshold uses raw cosine similarity of the top hit, not the hybrid score.**
  Min-max normalisation always gives the best hit a score of 1.0, even when it is irrelevant, so the
  combined score cannot tell us that nothing matched. The threshold value is tuned in phase 4.
- **Index settings:** 0 replicas (a single-node domain has nowhere to put a replica), HNSW with
  cosine similarity, 1,536 dimensions to match `text-embedding-3-small`.

### Generation and guardrails
The flow is: input guardrail -> hybrid search -> retrieval guardrail (abstain or clarify) ->
generate -> output guardrail -> one retry -> answer or abstain. Every guardrail is plain code.

- **The model reads the top 3 full parent tasks.** Instructions are only in the system message;
  the question and each task sit in marked `<question>` / `<source>` blocks that the system
  message declares to be data, never instructions.
- **Strict JSON schema output** (`status`, `answer`, `steps`, `source_ids`). The model cannot
  return anything outside those fields, so there is nothing to parse loosely.
- **The model cites by source id; our code builds the citation.** Aircraft, task number, revision,
  date and S3 key come from the index, so a citation cannot be invented or mistyped.
- **Input guardrail (free):** empty check, 500-character limit, simple injection patterns. The
  patterns are a first filter; the real defences are the fenced prompt, the fixed output shape and
  the output checks.
- **Abstain before the LLM when retrieval is weak.** Rule: a part or task number in the question
  must be found verbatim in a retrieved task (found = strong evidence; not found = abstain, which
  also covers superseded part numbers); otherwise the top result's cosine similarity must be at
  least 0.35. Measured on the live index with 26 probe questions: off-topic 0.07 to 0.30, in-domain
  0.41 to 0.80, so 0.35 is the middle of the gap. The hybrid score cannot be used: min-max
  normalisation gives "Who won the 2022 World Cup?" 0.81.
- **The threshold only stops off-topic questions.** A question that is in-domain but not in the
  manuals (tyre pressure, 0.52) scores like a real one, so it goes to the model, which is
  instructed to say it cannot answer. Two layers, each doing what it is able to do.
- **Clarify instead of guessing the aircraft.** If no aircraft is given or detected and the best
  results are the same task on different aircraft, the reply is `clarification_needed` with the
  options, and no LLM call is made. A part number settles the aircraft by itself; a task number does not.
- **Output guardrail:** at least one valid citation; every citation is the latest revision
  (second check after the search filter); every part number and torque value in the answer appears
  in a cited task (so unit conversions and values from the wrong aircraft fail); and if steps are
  returned they are the complete procedure of a cited task, in order, word for word, each with its
  WARNINGs and CAUTIONs. A wrong torque value is the worst possible output, so these checks are
  deterministic code, not another LLM.
- **Component check (added in evaluation v2).** Position words in the question (nose/main,
  left/right, inboard/outboard, forward/aft, upper/lower) must not contradict a cited task's title.
  This catches "nose wheel axle nut" answered from the main wheel task, which the value checks
  cannot see because the value really is in that task. The abstention names the component the
  manual does cover, built by code from the index.
- **Identifier retry (added in evaluation v2).** If the model abstains although a part or task
  number from the question appears word for word in a source, it is retried once and told which
  source contains it. An exact identifier match is the strongest evidence the system has.
- **One retry, then abstain.** The failed checks are listed to the model in the system message; a
  second failure returns "I don't know" and the unverified text is never shown.
- **Warnings on short answers are attached by code, not the model.** "The final torque is 135 Nm"
  comes from a step that carries a WARNING. Code finds the steps of the cited task that state a
  torque value or part number quoted in the answer (plus any step the model points at by number),
  and copies their WARNINGs and CAUTIONs from the index into the response's `warnings` field, the
  same way citations are built. The text is therefore always word for word and cannot be forgotten.

### Container
- **One image, two commands.** The default command starts the API; `python -m aero_rag.ingestion`
  runs the ingestion job. The code that was tested is the code that runs in both.
- **Two-stage build.** The first stage uses `uv` to build the virtual environment from the lock
  file; the final image is `python:3.12-slim` plus that environment only (336 MB), with no build
  tools, tests, data or `uv`.
- **Dependencies are a separate layer before the code,** so a code change rebuilds in seconds and
  `uv sync --locked` fails the build if the lock file is stale.
- **`linux/amd64` is fixed in the Dockerfile** because that is what the Fargate task runs; an image
  built on an ARM laptop would otherwise fail with "exec format error".
- **Runs as an unprivileged user (`app`, uid 10001),** so a compromised app is not root.
- **`.dockerignore` is an allow-list:** everything is excluded except `pyproject.toml`, `uv.lock`
  and `src/`, so a `.env` or key file can never enter the build by accident.
- **`HEALTHCHECK` calls `/health`** (liveness, no dependencies) using Python, as slim has no curl.
- **Secrets enter the container only as environment variables,** exactly as ECS injects them, so the
  app never calls Secrets Manager there. `scripts/run_local_container.ps1` reproduces that locally:
  it reads the secret into memory and passes each value to Docker by name, never on a command line
  or on disk.

### API and UI
- **FastAPI with Pydantic models** for `POST /query`. Pydantic validates input and documents the
  contract automatically (`/docs`). The status is one of `answered`, `abstained`,
  `clarification_needed`, `rejected`; all four are HTTP 200 because the service worked. A failed
  dependency is HTTP 503 with a request id and no details.
- **`/health` and `/ready` are separate.** `/health` only says the process is alive, so a slow
  OpenSearch never makes ECS kill a healthy container; `/ready` checks OpenSearch is reachable.
- **Clients are created once at start-up** and reused (connection pooling), with timeouts of 10 s
  on OpenSearch and 60 s on Azure so a stuck dependency cannot hang a request for ever.
- **One JSON log line per request, with a request id** that is also returned in the body and the
  `X-Request-ID` header. Logs hold sizes, decisions, scores and token counts; never the question,
  the prompt or a secret. `python -m aero_rag.api` starts the server with every line as JSON.
- **Streamlit UI calls the API over HTTP** rather than importing the RAG code, so the API is the single
  entry point and is what we test and trace.

### AWS
- **ECS Fargate.** Runs containers without us managing servers; we pay per second while the task runs.
- **One task, two containers (API + UI), two images.** Containers in one Fargate task share
  `localhost`, so the UI reaches the API at `http://localhost:8000` with no extra networking. The
  UI has its own small image (Streamlit only, no `aero_rag`, no secrets) instead of sharing the API
  image, so a UI change never rebuilds or risks the API, and the UI container never receives the
  secret. Changed in phase 6 from the original "same image, two commands".
- **Locally, docker compose runs the same two images** (`docker-compose.yml`,
  `scripts/compose_up.ps1`). Compose puts them on a private network where the service name is the
  host name, so there the UI uses `http://api:8000`; only `API_URL` differs from ECS.
- **One Secrets Manager secret, `akinsds1126-aero-rag/app`, is the only place secrets exist.** It is
  a JSON object with five keys (Azure endpoint and key; OpenSearch endpoint, username and password).
  Nothing secret is in the repo, the image, `.env` or the task definition text.
- **Same code, two ways to receive the secret.** Locally, `config.py` reads it with boto3 using the
  `aero` SSO profile, so no long-lived credential sits on the laptop. On ECS, the task definition
  references each JSON key and ECS injects it as an environment variable at start-up; the code sees
  the variables are already set and does not call Secrets Manager at all.
- **Two IAM roles.** The *execution role* is used by ECS itself to pull the image, write logs and
  read the secret for injection. The *task role* is what our code runs as: read-only on our one S3
  bucket and nothing else. Keeping them separate means the application cannot read secrets it was not handed.
- **Ingestion runs two ways from the same code.** `python -m aero_rag.ingestion` runs locally (profile
  `aero`) and as a one-off Fargate task: same image, same task role, with the container command
  overridden in `aws ecs run-task`. One image means the code that was tested is the code that runs,
  and a one-off task costs nothing once it exits.
- **JSON logs to stdout.** ECS forwards a container's stdout to CloudWatch Logs (the `awslogs` driver),
  so the app needs no AWS logging code. One JSON object per line lets CloudWatch Logs Insights filter by
  field (level, task number, latency). Known secret values are redacted from every line as a last defence.
- **`.env` is non-secret only** (region, profile, bucket, index, deployment names, API version). It stays
  git-ignored because it is machine-specific, and `.env.example` documents it.
- **OpenSearch public endpoint + fine-grained access control.** Simplest thing that is still
  authenticated (HTTPS + username/password). In production it would sit inside a VPC.
- **Azure OpenAI API version `2025-04-01-preview`.** It is the version in the endpoint the Azure portal
  issued for this resource, and `check_connections.py` confirms chat and embeddings both work with it.

### Engineering
- **`uv` + `src/` layout.** Fast, reproducible installs from a lock file; `src/` stops tests from
  accidentally importing un-installed code.
- **Tests use fakes** for Azure OpenAI, OpenSearch and S3 so they run offline, free and in seconds.
- **Langfuse is optional at runtime.** With no keys set, tracing is a no-op, so tests and local runs do not need it.
- **Tracing never touches the answer path's reliability.** Spans are recorded in memory and sent by
  the SDK's background thread; every SDK call is wrapped so the first failure logs one warning and
  turns tracing off. Start-up checks the keys once (`auth_check`, about 2-3 s if Langfuse is down).
- **Traces carry metadata by default, text only by setting.** Step spans hold chunk ids, scores,
  decisions, model, tokens, cost and latency, never text. `LANGFUSE_CAPTURE_TEXT` (default false in
  code) adds the question and the final answer to the trace's root, redacted like the logs; never
  the prompt or the retrieved manual text. It is true for this demo because the manuals are
  synthetic, and must be false for real, proprietary manuals.
- **The trace id is derived from the request id** (`create_trace_id(seed=request_id)`), so the
  request id shown in the UI or a log line finds the trace directly.
- **Langfuse SDK 5.x (OpenTelemetry based), with its own tracer provider**, so it never becomes the
  process-wide OpenTelemetry tracer and exports only our spans.

## 3. Phases

| # | Phase | Deliverable | Check before moving on |
|---|-------|-------------|------------------------|
| 0 | Setup | Repo skeleton, config (Secrets Manager), JSON logging, `check_connections.py` | Five PASS lines (done) |
| 1 | Documents | `data/spec/manuals.json`, generator, 31 manuals, parser, round-trip tests | Done; user uploads `data/manuals/` to S3 under `manuals/` |
| 2 | Ingestion + hybrid search | Chunker, embeddings, index + search pipeline, search function, `python -m aero_rag.ingestion`, `scripts/search.py` | Done: 93 documents indexed; four spot-check searches correct |
| 3 | Generation + API + guardrails | `/health`, `/ready`, `/query`, abstain, clarify, output checks | Done: six test questions behave as designed locally |
| 4a | Containerise the API | Dockerfile, `.dockerignore`, `scripts/run_local_container.ps1` | Done: image `akinsds1126-aero-rag-api:0.1.0` answers locally as user `app` |
| 4b | Deploy the API to ECS Fargate | Commands for ECR, IAM, ECS service, and the one-off ingestion task | Public URL answers `/query`; ingestion task exits 0 with logs in CloudWatch |
| 5 | Evaluation | 47 labelled questions, metrics, before/after tables, README Metrics draft | Done: `scripts/evaluate.py` reproduces `data/results/` (~$0.14); v2 fixed both v1 failures (47/47) |
| 6 | Streamlit UI and redeploy | `ui/` chat (app, presenter, Dockerfile), `docker-compose.yml`, `scripts/compose_up.ps1`; then the UI as a second container in the same task | Done: rev 2 on ECS, UI on port 8501 answers; port 8000 closed |
| 7a | Langfuse tracing | `tracing.py`, a span per step, flush on shutdown, `docs/DEPLOY.md` + task definition rev 2 | Done locally: a compose query produced an 8-observation trace in Langfuse Cloud. ECS: rev 2 deployed by the user |
| 7b | Final polish | USWDS-aligned UI (`ui/theme.py`), small-talk replies, aircraft chip follows detection, UI 0.2.1 + task definition rev 3; full README; `.env.example` with every setting; secret scan | Done locally (compose, UI 0.2.1). ECS: rev 3 deployed by the user |

**Why this order.** The live ECS demo is assessed, so the API is deployed as soon as it works
(phase 4) and everything after that adds to a system that is already live. **If time runs short,
cut from the end:** first Langfuse, then the UI (the API's built-in `/docs` page can be demonstrated
instead), then the size of the evaluation. The README is short but is never cut.

## 4. Evaluation

Results and their explanation are in the README "Metrics" section and `data/results/`; the run
history (v1 baseline 45/47, v2 47/47 after fixing the two v1 failures) is in `docs/EVALUATION.md`.

- **47 hand-written questions, labels looked up in the spec** (`data/eval/questions.json`, built by
  `scripts/build_eval_set.py`). Torque (11), part number (8), full procedure (6), no aircraft named
  but only one aircraft has the task (4), bare part or task number (5), must ask which aircraft
  (4), cannot be answered (9: off-topic, in-domain, a superseded part number, an injection).
- **Labels are correct by construction.** A question names its task and fastener or part; the
  expected value, step count and the values that exist only in old revisions are read from the
  spec, and a test fails if the JSON drifts from the spec.
- **13 "trap" questions** target tasks whose older revision has a different value; they measure the
  latest-revision filter.
- **Metrics:** Recall@5 and top-1 (retrieval); exact match of torque values and part numbers; step
  completeness including every WARNING and CAUTION; outdated citations (target 0); correct
  abstentions; correct clarifications; plus false abstentions and values from the wrong task.
- **Before/after comparisons:** keyword vs vector vs hybrid by question type; latest-revision filter
  off vs on, both at retrieval level and end to end (including what the model drafted before the
  output guardrail).
- **Changed from the original plan:** the before/after is search mode and the revision filter, as the
  brief's phase 5 asked, not fixed-size vs structure-aware chunking.
- **Caveat to state when presenting:** the corpus is small (8 tasks per aircraft), so Recall@5
  saturates; top-1 is the more telling number. Six unanswerable questions overlap with the probes
  used to pick the abstention threshold, so that result is partly in-sample.

## 5. AWS resource names

All prefixed `akinsds1126-` and tagged `project=aero-rag`, `env=dev`, `owner=akinlolu`.

| Resource | Name |
|----------|------|
| OpenSearch domain (exists) | `akinsds1126-aero-rag` |
| S3 bucket (exists) | `akinsds1126-aero-rag-docs` |
| Secret (exists) | `akinsds1126-aero-rag/app` (Langfuse keys are added to it as extra JSON keys in phase 7) |
| OpenSearch index (inside the domain) | `aero-manuals` |
| ECR repository | `akinsds1126-aero-rag` |
| ECS cluster / service | `akinsds1126-aero-rag` / `akinsds1126-aero-rag-svc` |
| ECS task definitions | `akinsds1126-aero-rag` (API + UI), `akinsds1126-aero-rag-ingest` (one-off ingestion, same image) |
| IAM roles | `akinsds1126-aero-rag-exec-role`, `akinsds1126-aero-rag-task-role` |
| Security group | `akinsds1126-aero-rag-sg` |
| CloudWatch log group | `akinsds1126-aero-rag-logs` |

## 6. Cost

Rough figures, to be confirmed against current price pages before each paid run.

- **Embeddings (phase 2):** about 150 chunks for both chunking variants, well under 200k tokens in
  total. At `text-embedding-3-small` pricing this is under one cent.
- **Evaluation (phase 4):** about 60 questions x 2 configurations = about 120 chat calls and 120
  query embeddings, plus retries. Expected to be well under $1; exact call count and price are stated before the run.
- **AWS while it exists:** the OpenSearch t3.small node and the Fargate task are billed hourly
  (a few dollars per day together); each secret has a small monthly fee. Tear-down commands come with phase 6.

## 7. Timeline and cut order

| When | Work |
|------|------|
| Sun 15 Nov | Phase 0 (done) |
| Mon 16 Nov, until 2:30pm | Phases 1 to 7, in order; each phase leaves something demonstrable |
| Mon 16 Nov, 2:30pm | Everything frozen; rehearse the demo; keep a local-run fallback ready |

Phases 1 to 7 all fall on presentation day, so the cut order is likely to be used. The live ECS
demo is assessed, so the first goal is the API answering on ECS (end of phase 4); a local run stays
ready as the fallback.

If we fall behind, cut from the end of the phase list (see section 3); within a phase, drop the
extra retrieval ablation and any HTTPS/load balancer work first. Never cut: the chunking invariants, the latest-revision filter, the grounding
guardrail, the four metrics.

## 8. Open decisions (not blocking)

1. **How the API and UI are exposed (phases 4 and 6).** Recommended: the task's public IP with a security group
   limited to the presenter's IP address; fewer resources in a shared account and quick to build.
   A load balancer with HTTPS is the production answer and goes in the README as a next step.
2. **Abstention threshold value.** Chosen from the phase 4 score distribution, not guessed.
