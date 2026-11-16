import json
import logging
import re

from pydantic import SecretStr

from aero_rag.config import Settings, merge_secret
from aero_rag.logging_config import JsonFormatter, setup_logging


def test_opensearch_host_accepts_endpoint_with_or_without_scheme():
    host = "search-x.eu-west-2.es.amazonaws.com"
    for endpoint in (f"https://{host}/", host):
        settings = Settings(_env_file=None, opensearch_endpoint=endpoint)
        assert settings.opensearch_host == host


def test_azure_base_url_drops_the_portal_target_uri_path_and_query():
    root = "https://my-resource.openai.azure.com"
    for endpoint in (root, f"{root}/", f"{root}/openai/responses?api-version=2025-04-01-preview"):
        settings = Settings(_env_file=None, azure_openai_endpoint=endpoint)
        assert settings.azure_openai_base_url == root


def test_secrets_are_masked_when_settings_are_printed():
    settings = Settings(_env_file=None, azure_openai_api_key="sk-SUPER-SECRET")
    assert "SUPER-SECRET" not in repr(settings)


def test_environment_value_wins_over_secrets_manager_value():
    settings = Settings(_env_file=None, azure_openai_api_key="from-env")
    merged = merge_secret(
        settings, {"AZURE_OPENAI_API_KEY": "from-secret", "OPENSEARCH_USERNAME": "fake_user"}
    )
    assert merged.azure_openai_api_key.get_secret_value() == "from-env"
    assert merged.opensearch_username == "fake_user"
    assert isinstance(merged.azure_openai_api_key, SecretStr)


def _record(message: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord("aero_rag.test", logging.INFO, __file__, 1, message, None, None)
    record.__dict__.update(extra)
    return record


def test_log_line_is_json_with_extra_fields():
    entry = json.loads(JsonFormatter().format(_record("ingested", chunks=42)))
    assert entry["level"] == "INFO"
    assert entry["logger"] == "aero_rag.test"
    assert entry["message"] == "ingested"
    assert entry["chunks"] == 42
    assert entry["ts"].endswith("+00:00")


def test_secret_values_are_redacted_from_logs():
    formatter = JsonFormatter(redact=["sk-SUPER-SECRET", 'pa"ss-word'])
    line = formatter.format(_record('auth failed for sk-SUPER-SECRET and pa"ss-word'))
    assert "SUPER-SECRET" not in line and "ss-word" not in line
    assert json.loads(line)["message"] == "auth failed for *** and ***"


def test_endpoint_hostnames_are_redacted_even_when_the_stored_value_is_a_longer_url():
    settings = Settings(
        _env_file=None,
        azure_openai_endpoint="https://my-res.openai.azure.com/openai/responses?api-version=x",
        opensearch_endpoint="https://search-dom.eu-west-2.es.amazonaws.com",
    )
    formatter = JsonFormatter(redact=settings.secret_values())
    line = formatter.format(
        _record(
            "POST https://my-res.openai.azure.com/openai/deployments/e/embeddings and "
            "GET https://search-dom.eu-west-2.es.amazonaws.com:443/_search"
        )
    )
    assert "my-res" not in line and "search-dom" not in line


def test_http_client_loggers_do_not_log_request_urls_at_info():
    setup_logging("INFO")
    try:
        for name in ("httpx", "httpx2", "opensearch", "botocore"):
            assert not logging.getLogger(name).isEnabledFor(logging.INFO), name
    finally:
        logging.getLogger().handlers.clear()


def test_setup_logging_writes_json_to_stdout(capsys):
    setup_logging("INFO")
    try:
        logging.getLogger("aero_rag.test").info("hello", extra={"task_number": "32-11-01"})
        entry = json.loads(capsys.readouterr().out.strip())
        assert entry["message"] == "hello" and entry["task_number"] == "32-11-01"
    finally:
        logging.getLogger().handlers.clear()


def test_env_example_lists_every_setting_and_no_secret_values():
    from pathlib import Path

    from aero_rag.config import OPTIONAL_SECRET_KEYS, SECRET_KEYS, Settings

    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    assigned = re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=(.*)$", text, flags=re.MULTILINE)
    names = {name for name, _ in assigned}
    secret_fields = {k.lower() for k in (*SECRET_KEYS, *OPTIONAL_SECRET_KEYS)}
    for field in Settings.model_fields:
        if field in secret_fields:
            assert field.upper() not in names  # only named in the comment, never assigned
            assert field.upper() in text
        else:
            assert field.upper() in names, f"{field.upper()} missing from .env.example"
