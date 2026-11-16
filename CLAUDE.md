# Aircraft maintenance manual assistant (aero-rag)

Production-grade RAG app on AWS, presented Monday 16 November 2026, 4:30pm UK.
Speed matters, but correctness and clear reasoning matter more. Full plan: [docs/PLAN.md](docs/PLAN.md).

## Hard rules

1. **Secrets.** Never print, log or commit secret values. They live only in AWS Secrets Manager,
   secret `akinsds1126-aero-rag/app` (JSON keys: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
   `OPENSEARCH_ENDPOINT`, `OPENSEARCH_USERNAME`, `OPENSEARCH_PASSWORD`). `.env` holds non-secret
   settings only and is still git-ignored; only `.env.example` is committed. Error output shows the
   exception *type*, never the message, because messages can echo endpoints and keys. When
   debugging, print shapes and error codes, never values.
2. **AWS is hands-off.** Never run a command that creates, changes or deletes an AWS resource.
   Write the command out for the user to run. Read-only calls (describe/list/get/head) are fine.
3. **Shared company account.** Every AWS resource name starts with `akinsds1126-` and is tagged
   `project=aero-rag`, `env=dev`, `owner=akinlolu`. Never modify or delete a resource we did not create.
4. **Paid runs.** Before any batch of embedding or LLM calls, state the number of calls and the
   estimated cost, and wait for a go-ahead.
5. **Phases.** Work one phase at a time and stop after each for the user to check:
   0 setup, 1 documents, 2 ingestion + hybrid search, 3 generation + API + guardrails,
   4 deploy the API to ECS Fargate, 5 evaluation, 6 Streamlit UI and redeploy, 7 Langfuse + README.
   The live ECS demo is assessed, so deployment comes before evaluation. Cut from the end if time runs short.
6. **Explain every design choice** in one or two plain sentences. The user is new to AWS and must
   defend this work, so say what the thing is and why we chose it over the obvious alternative.

## Fixed facts

- AWS: region `eu-west-2`, CLI profile `aero` (SSO, AdministratorAccess). Log in with `aws sso login --profile aero`.
- OpenSearch: domain `akinsds1126-aero-rag`, OpenSearch 2.19, 1 x t3.small.search, public endpoint,
  fine-grained access control, master user `aero_admin`.
- Models (Azure OpenAI): chat `gpt-5.4-mini` (use `max_completion_tokens`, not `max_tokens`);
  embeddings `text-embedding-3-small`, 1,536 dimensions. API version `2025-04-01-preview`
  (verified by `check_connections.py`). The stored endpoint is the portal's full target URI, so
  always build clients from `settings.azure_openai_base_url`, not the raw endpoint.
- S3 bucket `akinsds1126-aero-rag-docs`; OpenSearch index `aero-manuals`.
- Documents are synthetic, generated from a JSON spec, so evaluation labels are correct by construction.
  `data/spec/manuals.json` is the single source of truth: 3 fictional aircraft (A-200, B-340, C-75),
  24 tasks, 31 task revisions (7 superseded). Never edit `data/manuals/*.md` by hand; edit the spec
  and run `scripts/generate_manuals.py`. The Markdown layout is a contract between
  `docgen/render.py` and `ingestion/parser.py`: change them together.

## Engineering conventions

- Python 3.12, managed with `uv`. Package lives in `src/aero_rag/`. Run things with `uv run ...`.
- All configuration goes through `aero_rag.config.get_settings()`. No `os.environ` lookups
  scattered through the code. Secrets load two ways from the same code: locally, boto3 reads
  Secrets Manager with the `aero` profile; on ECS, the task definition injects each key as an
  environment variable (which wins, so no fetch happens). `AWS_PROFILE` is never set on ECS.
- AWS clients come from `aero_rag.config.aws_session()` (profile locally, task role on ECS).
- Logging: JSON, one object per line, to stdout via `aero_rag.logging_config.setup_logging()`,
  so CloudWatch Logs collects it. Use `logging.getLogger(__name__)` and `extra={...}` for fields;
  no `print` in `src/`. Pass `settings.secret_values()` as `redact`.
- Ingestion runs two ways from the same code (`python -m aero_rag.ingestion`): locally, and as a
  one-off ECS Fargate task using the same image with a different command. Nothing in it may assume
  a local file system or a named profile; documents are always read from S3.
- Tests use fakes for Azure OpenAI, OpenSearch and S3, so `uv run pytest` passes offline with no `.env`.
- External clients are passed into functions/classes (dependency injection) so fakes can replace them.
- Chunking invariants (never break these): one task per chunk; a numbered procedure is never split;
  a WARNING/CAUTION always stays with its step.
- Retrieval always filters `is_latest = true` and the requested `aircraft_type`.
- Guardrail: every part number and torque value in an answer must appear in a cited chunk;
  otherwise retry once, then abstain.

## Commands

```powershell
uv sync                                  # install dependencies
uv run pytest                            # offline tests
uv run ruff check .                      # lint
uv run python scripts/generate_manuals.py    # rebuild data/manuals/ from the spec (offline, free)
aws sso login --profile aero             # needed before anything that touches AWS
uv run python scripts/check_connections.py   # PASS/FAIL: Secrets Manager, S3, OpenSearch, Azure chat, embeddings
```
