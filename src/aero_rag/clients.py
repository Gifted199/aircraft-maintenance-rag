"""Real clients for Azure OpenAI and OpenSearch, built from settings.

Nothing else in the package constructs a client; code receives them as arguments, which is what
lets the tests pass fakes instead.
"""

import logging

from aero_rag.config import Settings

log = logging.getLogger(__name__)


def make_opensearch(settings: Settings):
    from opensearchpy import OpenSearch

    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": 443}],
        http_auth=(settings.opensearch_username, settings.opensearch_password.get_secret_value()),
        use_ssl=True,
        verify_certs=True,
        timeout=30,
        max_retries=2,
        retry_on_timeout=True,
    )


def make_azure_openai(settings: Settings):
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_base_url,
        api_key=settings.azure_openai_api_key.get_secret_value(),
        api_version=settings.azure_openai_api_version,
        timeout=60,
        max_retries=2,
    )


class AzureEmbedder:
    """Embeds texts in batches: one API call per `batch_size` texts, results in input order."""

    def __init__(self, client, deployment: str, batch_size: int = 64):
        self._client = client
        self._deployment = deployment
        self.batch_size = batch_size
        self.calls = 0
        self.total_tokens = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self._deployment, input=batch)
            self.calls += 1
            self.total_tokens += response.usage.total_tokens
            vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        usage = {"texts": len(texts), "calls": self.calls, "tokens": self.total_tokens}
        log.info("embedded", extra=usage)
        return vectors
