"""Single source of configuration.

Non-secret settings come from environment variables, falling back to a local `.env` file.

The five secret values live in one AWS Secrets Manager secret (JSON). They reach the code two ways:
  * locally: fetched with boto3 using the `aero` CLI profile (nothing secret is stored on disk);
  * on ECS:  injected as environment variables by ECS from the same secret, so no fetch happens.
Either way the rest of the code just reads `get_settings()`.
"""

import json
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# JSON keys inside the Secrets Manager secret. The matching Settings field is the lower-case name.
SECRET_KEYS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "OPENSEARCH_ENDPOINT",
    "OPENSEARCH_USERNAME",
    "OPENSEARCH_PASSWORD",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # AWS. aws_profile is unset on ECS, where the task role supplies credentials.
    aws_region: str = "eu-west-2"
    aws_profile: str | None = None
    app_secret_name: str | None = None
    s3_bucket: str | None = None
    s3_prefix: str = "manuals/"

    # Non-secret application settings
    opensearch_index: str = "aero-manuals"
    opensearch_search_pipeline: str = "aero-manuals-hybrid"
    azure_openai_api_version: str = "2025-04-01-preview"
    azure_openai_chat_deployment: str = "gpt-5.4-mini"
    azure_openai_embed_deployment: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    api_url: str = "http://localhost:8000"
    log_level: str = "INFO"

    # Secret values: from Secrets Manager (local) or injected environment variables (ECS).
    # SecretStr shows as '**********' if a settings object is ever printed.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    opensearch_endpoint: str | None = None
    opensearch_username: str | None = None
    opensearch_password: SecretStr | None = None

    # Langfuse Cloud (phase 7; tracing is skipped when keys are absent)
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def opensearch_host(self) -> str | None:
        """Bare hostname, whether the endpoint was given with or without a scheme."""
        if not self.opensearch_endpoint:
            return None
        return self.opensearch_endpoint.removeprefix("https://").removeprefix("http://").rstrip("/")

    @property
    def azure_openai_base_url(self) -> str | None:
        """Resource root (https://<name>.openai.azure.com).

        The Azure portal shows a full "target URI" including a path and ?api-version=...; the SDK
        wants only the root and adds the path itself, so anything after the host is dropped.
        """
        if not self.azure_openai_endpoint:
            return None
        parts = urlsplit(self.azure_openai_endpoint.strip())
        return f"{parts.scheme}://{parts.netloc}" if parts.netloc else None

    def missing_secrets(self) -> list[str]:
        return [key for key in SECRET_KEYS if not getattr(self, key.lower())]

    def secret_values(self) -> list[str]:
        """Plain values, used only to redact them from log output."""
        values = []
        for key in SECRET_KEYS:
            value = getattr(self, key.lower())
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            if value:
                values.append(value)
        # Libraries log request URLs built from the host, not the stored endpoint string, so the
        # bare hostnames are redacted as well.
        azure_host = urlsplit(self.azure_openai_base_url or "").netloc
        values += [host for host in (azure_host, self.opensearch_host) if host]
        return values


def aws_session(settings: Settings):
    """One place that decides how we authenticate to AWS: profile locally, task role on ECS."""
    import boto3

    return boto3.Session(profile_name=settings.aws_profile or None, region_name=settings.aws_region)


def fetch_app_secret(settings: Settings, client: Any = None) -> dict[str, str]:
    """Read the JSON secret from Secrets Manager. `client` can be a fake in tests."""
    if client is None:
        client = aws_session(settings).client("secretsmanager")
    response = client.get_secret_value(SecretId=settings.app_secret_name)
    return json.loads(response["SecretString"])


def merge_secret(settings: Settings, values: dict[str, str]) -> Settings:
    """Fill in secret fields that are not already set. Environment variables win, so ECS-injected
    values are never overridden."""
    update: dict[str, Any] = {}
    for key in settings.missing_secrets():
        value = values.get(key)
        if value:
            field = key.lower()
            is_masked = field in ("azure_openai_api_key", "opensearch_password")
            update[field] = SecretStr(str(value)) if is_masked else str(value)
    return settings.model_copy(update=update)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.missing_secrets() and settings.app_secret_name:
        settings = merge_secret(settings, fetch_app_secret(settings))
    return settings
