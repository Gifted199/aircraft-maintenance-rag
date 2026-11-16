"""Connectivity smoke test: Secrets Manager, S3 bucket, OpenSearch cluster health, Azure chat,
Azure embeddings.

Prints one line per check: PASS, or FAIL with the error *type* only. Exception messages are never
printed because they can echo endpoints, usernames or keys.

All five checks are read-only. The two Azure checks make one tiny paid call each (a fraction of a
cent in total).

    uv run python scripts/check_connections.py
"""

import sys
from collections.abc import Callable

from aero_rag.config import SECRET_KEYS, Settings, aws_session, fetch_app_secret, merge_secret

TIMEOUT_SECONDS = 20


class MissingConfig(Exception):
    """Raised with the NAMES of unset variables (names are not secret; values never appear)."""


class EmbeddingDimMismatch(Exception):
    pass


class ClusterStatusRed(Exception):
    pass


def require(settings: Settings, *fields: str) -> None:
    missing = [f.upper() for f in fields if not getattr(settings, f)]
    if missing:
        raise MissingConfig(", ".join(missing))


def describe_error(exc: BaseException) -> str:
    """Error type plus, where the library exposes one, the HTTP status or AWS error code."""
    if isinstance(exc, MissingConfig):
        return f"MissingConfig: {exc}"
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)  # openai and opensearch-py errors
    if isinstance(status, int):
        return f"{name} (HTTP {status})"
    response = getattr(exc, "response", None)  # botocore ClientError
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code:
            return f"{name} ({code})"
    return name


def _boto_config():
    from botocore.config import Config

    return Config(
        connect_timeout=TIMEOUT_SECONDS, read_timeout=TIMEOUT_SECONDS, retries={"max_attempts": 1}
    )


def load_secret(settings: Settings, client=None) -> Settings:
    """Check 1. Returns settings with the secret values filled in.

    If every value is already in the environment (as on ECS) there is nothing to fetch.
    """
    if not settings.missing_secrets():
        return settings
    require(settings, "app_secret_name")
    if client is None:
        client = aws_session(settings).client("secretsmanager", config=_boto_config())
    values = fetch_app_secret(settings, client)
    absent = [key for key in SECRET_KEYS if not values.get(key)]
    if absent:
        raise MissingConfig("secret is missing keys " + ", ".join(absent))
    return merge_secret(settings, values)


def check_s3(settings: Settings) -> None:
    require(settings, "s3_bucket")
    s3 = aws_session(settings).client("s3", config=_boto_config())
    s3.head_bucket(Bucket=settings.s3_bucket)  # bucket exists and we may access it
    s3.list_objects_v2(Bucket=settings.s3_bucket, Prefix=settings.s3_prefix, MaxKeys=1)


def check_opensearch(settings: Settings) -> None:
    from opensearchpy import OpenSearch

    require(settings, "opensearch_endpoint", "opensearch_username", "opensearch_password")
    client = OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": 443}],
        http_auth=(settings.opensearch_username, settings.opensearch_password.get_secret_value()),
        use_ssl=True,
        verify_certs=True,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    )
    # Yellow is normal on a single-node domain (replicas have nowhere to go);
    # red means data is unavailable.
    if client.cluster.health().get("status") == "red":
        raise ClusterStatusRed


def _azure_client(settings: Settings):
    from openai import AzureOpenAI

    require(settings, "azure_openai_endpoint", "azure_openai_api_key")
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_base_url,
        api_key=settings.azure_openai_api_key.get_secret_value(),
        api_version=settings.azure_openai_api_version,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    )


def check_azure_chat(settings: Settings) -> None:
    client = _azure_client(settings)
    client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[{"role": "user", "content": "Reply with OK."}],
        max_completion_tokens=32,
    )


def check_azure_embeddings(settings: Settings) -> None:
    client = _azure_client(settings)
    result = client.embeddings.create(
        model=settings.azure_openai_embed_deployment, input="connection check"
    )
    # The OpenSearch k-NN field is created with a fixed dimension, so a mismatch must fail early.
    if len(result.data[0].embedding) != settings.embedding_dim:
        raise EmbeddingDimMismatch


CHECKS: list[tuple[str, Callable[[Settings], None]]] = [
    ("S3 bucket", check_s3),
    ("OpenSearch health", check_opensearch),
    ("Azure chat", check_azure_chat),
    ("Azure embeddings", check_azure_embeddings),
]


def run_check(name: str, check: Callable[[Settings], object], settings: Settings):
    """Returns (passed, whatever the check returned)."""
    try:
        result = check(settings)
    except Exception as exc:  # noqa: BLE001 - any failure is reported by type only
        print(f"{name:<18} FAIL  {describe_error(exc)}")
        return False, None
    print(f"{name:<18} PASS")
    return True, result


def main() -> int:
    try:
        settings = Settings()
    except Exception as exc:  # noqa: BLE001 - e.g. a malformed .env value
        print(f"{'Settings':<18} FAIL  {type(exc).__name__}")
        return 1
    passed, with_secret = run_check("Secrets Manager", load_secret, settings)
    results = [passed]
    if with_secret is not None:
        settings = with_secret
    results += [run_check(name, check, settings)[0] for name, check in CHECKS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
