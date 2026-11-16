"""Offline tests: no network, no .env. They pin the 'type only, never the message' rule."""

import json

import pytest

import check_connections as cc
from aero_rag.config import SECRET_KEYS, Settings

FAKE_SECRET = {key: f"value-of-{key}" for key in SECRET_KEYS}


class FakeSecretsManager:
    def __init__(self, payload: dict):
        self.payload = payload
        self.requested: list[str] = []

    def get_secret_value(self, SecretId: str) -> dict:  # noqa: N803 - boto3's argument name
        self.requested.append(SecretId)
        return {"SecretString": json.dumps(self.payload)}


@pytest.fixture
def empty_settings(monkeypatch) -> Settings:
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    return Settings(_env_file=None)


def test_failure_prints_type_but_never_the_message(capsys, empty_settings):
    def leaky(_settings):
        raise RuntimeError("401 for key sk-SUPER-SECRET at https://private.example")

    assert cc.run_check("Leaky", leaky, empty_settings) == (False, None)
    out = capsys.readouterr().out
    assert "FAIL" in out and "RuntimeError" in out
    assert "SUPER-SECRET" not in out and "private.example" not in out


def test_pass_prints_pass(capsys, empty_settings):
    assert cc.run_check("Fine", lambda _s: None, empty_settings) == (True, None)
    assert capsys.readouterr().out.split() == ["Fine", "PASS"]


def test_missing_config_names_the_variables(empty_settings):
    with pytest.raises(cc.MissingConfig) as err:
        cc.require(empty_settings, "azure_openai_endpoint", "azure_openai_api_key")
    assert cc.describe_error(err.value) == (
        "MissingConfig: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY"
    )


def test_status_and_aws_error_codes_are_shown():
    class HttpError(Exception):
        status_code = 401

    class AwsError(Exception):
        response = {"Error": {"Code": "AccessDenied", "Message": "do not show"}}

    assert cc.describe_error(HttpError("hidden")) == "HttpError (HTTP 401)"
    assert cc.describe_error(AwsError("hidden")) == "AwsError (AccessDenied)"


def test_load_secret_fetches_from_secrets_manager(empty_settings):
    settings = empty_settings.model_copy(update={"app_secret_name": "akinsds1126-aero-rag/app"})
    fake = FakeSecretsManager(FAKE_SECRET)

    loaded = cc.load_secret(settings, fake)

    assert fake.requested == ["akinsds1126-aero-rag/app"]
    assert loaded.missing_secrets() == []
    assert loaded.opensearch_password.get_secret_value() == "value-of-OPENSEARCH_PASSWORD"
    assert "value-of-OPENSEARCH_PASSWORD" not in repr(loaded)


def test_load_secret_reports_absent_keys_by_name_only(capsys, empty_settings):
    settings = empty_settings.model_copy(update={"app_secret_name": "akinsds1126-aero-rag/app"})
    partial = {k: v for k, v in FAKE_SECRET.items() if k != "OPENSEARCH_PASSWORD"}

    passed, _ = cc.run_check(
        "Secrets Manager", lambda s: cc.load_secret(s, FakeSecretsManager(partial)), settings
    )

    out = capsys.readouterr().out
    assert passed is False
    assert "OPENSEARCH_PASSWORD" in out and "value-of-" not in out


def test_load_secret_does_not_call_aws_when_values_are_injected(monkeypatch):
    """The ECS path: every secret arrives as an environment variable."""
    for key, value in FAKE_SECRET.items():
        monkeypatch.setenv(key, value)
    fake = FakeSecretsManager({})

    loaded = cc.load_secret(Settings(_env_file=None), fake)

    assert fake.requested == []
    assert loaded.missing_secrets() == []
