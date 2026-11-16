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
- **One task = one chunk.** A maintenance task is the unit a mechanic acts on; splitting it could
  return steps 4 to 7 without the WARNING on step 3, which is a safety failure, not just a quality one.
- **Hierarchical context in each chunk.** Every chunk starts with its breadcrumb (aircraft type >
  ATA chapter > task number and title) so the text is self-describing for both keyword and vector search.
- **Oversize guard.** If a task ever exceeded the embedding model's input limit it is split only on
  step boundaries, with each WARNING/CAUTION kept attached to its step. Our generated tasks are short,
  so this is a safety net with a test, not the normal path.
- **Metadata on every chunk:** `aircraft_type`, `ata_chapter`, `task_number`, `revision`,
  `revision_date`, `is_latest`. These are stored as exact-match fields so they can be filtered, not searched.
- **`is_latest` is computed at ingestion** by comparing all revisions of the same aircraft + task,
  and older ones are flipped to `false` when a newer one arrives. Deciding this once at write time is
  cheaper and less error-prone than working it out on every query.
- **Deterministic chunk IDs** (`aircraft-task-revision`). Re-running ingestion overwrites instead of
  duplicating, so the job is safe to run twice.

### Retrieval
- **Hybrid search (BM25 + k-NN).** Keyword search (BM25) is exact on part numbers like `AX2-32-4411`,
  which embeddings blur; vector search (k-NN) handles paraphrased questions. Hybrid gets both.
- **OpenSearch search pipeline with a normalization processor.** BM25 and cosine scores are on
  different scales; the pipeline rescales and combines them inside OpenSearch, so we make one request
  instead of merging two result lists in Python.
- **Filters applied inside both sub-queries:** `aircraft_type = <selected>` and `is_latest = true`.
  Filtering is a hard guarantee; relying on ranking to push old revisions down is not.
- **"I don't know" threshold uses raw cosine similarity of the top hit, not the hybrid score.**
  Min-max normalisation always gives the best hit a score of 1.0, even when it is irrelevant, so the
  combined score cannot tell us that nothing matched. The threshold value is tuned in phase 4.
- **Index settings:** 0 replicas (a single-node domain has nowhere to put a replica), HNSW with
  cosine similarity, 1,536 dimensions to match `text-embedding-3-small`.

### Generation and guardrails
- **Answer only from retrieved chunks, with citations** (task number + revision). The response model
  returns `answer`, `citations`, `scores` and an `abstained` flag.
- **Input guardrails:** length limit and simple prompt-injection patterns, checked before any paid call.
- **Grounding guardrail:** every part number and torque value in the answer must literally appear in
  a cited chunk. If not, retry once with the failure explained to the model, then abstain. A wrong
  torque value is the worst possible output, so this check is deterministic code, not another LLM.
- **Revision guardrail:** citations are checked against `is_latest` a second time after generation.

### API and UI
- **FastAPI with Pydantic models** for `GET /health` and `POST /query`. Pydantic validates input and
  documents the contract automatically (`/docs`).
- **Streamlit UI calls the API over HTTP** rather than importing the RAG code, so the API is the single
  entry point and is what we test and trace.

### AWS
- **ECS Fargate.** Runs containers without us managing servers; we pay per second while the task runs.
- **One task, two containers (API + UI), same image, different start commands.** Containers in one
  Fargate task share `localhost`, so the UI reaches the API at `http://localhost:8000` with no extra networking.
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

## 3. Phases

| # | Phase | Deliverable | Check before moving on |
|---|-------|-------------|------------------------|
| 0 | Setup | Repo skeleton, config (Secrets Manager), JSON logging, `check_connections.py` | Five PASS lines (done) |
| 1 | Documents | `data/spec/manuals.json`, generator, 31 manuals, parser, round-trip tests | Done; user uploads `data/manuals/` to S3 under `manuals/` |
| 2 | Ingestion + hybrid search | Chunker, embeddings, index + search pipeline, search function, `python -m aero_rag.ingestion` | Chunking invariant tests pass; spot-check queries |
| 3 | Generation + API + guardrails | `/health`, `/query`, abstention, grounding check | Curl examples; guardrail tests |
| 4 | Deploy the API to ECS Fargate | Dockerfile, commands for ECR, IAM, ECS service, and the one-off ingestion task | Public URL answers `/query`; ingestion task exits 0 with logs in CloudWatch |
| 5 | Evaluation | Questions from the spec, metrics script, before/after report | Numbers reproducible from one command |
| 6 | Streamlit UI and redeploy | Streamlit app as a second container in the same task | UI on the public URL answers a question |
| 7 | Langfuse + README | Tracing, README with diagram, "Future: agents" | Trace visible; README runs from clean clone |

**Why this order.** The live ECS demo is assessed, so the API is deployed as soon as it works
(phase 4) and everything after that adds to a system that is already live. **If time runs short,
cut from the end:** first Langfuse, then the UI (the API's built-in `/docs` page can be demonstrated
instead), then the size of the evaluation. The README is short but is never cut.

## 4. Evaluation

Questions are generated from the spec (about 60), so each has a known task, revision and expected values.

| Metric | Definition | Target |
|--------|------------|--------|
| Recall@5 | Correct task (latest revision) is in the top 5 chunks | High; reported |
| Exact match | Part numbers / torque values in the answer equal the spec | High; reported |
| Step completeness | Fraction of the task's numbered steps and WARNINGs present, in order | High; reported |
| Outdated revision | % of answers citing a superseded revision | **0** |

**Improvement with before/after:** naive fixed-size chunking (before) versus structure-aware
one-task-per-chunk (after), same retriever. This is the central design decision in the brief and it
directly moves step completeness. If time allows, a free extra ablation: BM25-only vs k-NN-only vs hybrid.

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
