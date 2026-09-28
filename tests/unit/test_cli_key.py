"""``ucx key``: keys are saved in the settings file, one per provider, and never in ``.env``.

``ucx key setup`` used to write ``GEMINI_API_KEY`` into a ``.env`` file while the
dashboard's Settings wrote the settings file; which key a request used depended on which
was loaded first. These tests pin the one store and the per-provider slots.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from uclone_x.cli.commands.key import mask_key, sanitize_key
from uclone_x.cli.main import app
from uclone_x.llm.connectors.saved_choice import settings_file

runner = CliRunner()

_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "VLLM_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_key_in_the_environment(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _saved() -> dict[str, object]:
    return json.loads(settings_file().read_text(encoding="utf-8"))


def test_sanitize_key() -> None:
    assert sanitize_key("  AIzaSy12345  ") == "AIzaSy12345"
    assert sanitize_key('"sk-ant-12345"') == "sk-ant-12345"
    assert sanitize_key("'sk-proj-12345'") == "sk-proj-12345"


def test_mask_key() -> None:
    assert mask_key("short") == "****"
    assert mask_key("AIzaSy1234567890abcdef") == "AIzaSy...cdef"


def test_set_saves_the_key_under_its_provider_in_the_settings_file(tmp_path: Path) -> None:
    """The key lands in `llm_api_keys`, the file is private, and no `.env` is written.

    Killed by: src/uclone_x/cli/commands/key.py :: save_api_key(spec.id, clean_key)
    Becomes: save_api_key("openai", clean_key)
    """
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["key", "set", "gemini", "--key", "AQ.gemini-key-0001"])
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.output
    assert _saved()["llm_api_keys"] == {"gemini": "AQ.gemini-key-0001"}
    assert stat.S_IMODE(settings_file().stat().st_mode) == 0o600
    assert not (tmp_path / ".env").exists()
    assert "GEMINI_API_KEY" not in os.environ
    assert "AQ.gemini-key-0001" not in result.output  # masked when echoed


def test_saving_one_providers_key_keeps_every_other_providers_key() -> None:
    assert (
        runner.invoke(app, ["key", "set", "openai", "--key", "sk-openai-key-0001"]).exit_code == 0
    )
    assert (
        runner.invoke(app, ["key", "set", "google", "--key", "AQ.gemini-key-0001"]).exit_code == 0
    )
    assert _saved()["llm_api_keys"] == {
        "openai": "sk-openai-key-0001",
        "gemini": "AQ.gemini-key-0001",
    }


def test_remove_deletes_only_that_providers_key() -> None:
    runner.invoke(app, ["key", "set", "openai", "--key", "sk-openai-key-0001"])
    runner.invoke(app, ["key", "set", "anthropic", "--key", "sk-ant-anthropic-0001"])
    result = runner.invoke(app, ["key", "remove", "openai"])
    assert result.exit_code == 0, result.output
    assert _saved()["llm_api_keys"] == {"anthropic": "sk-ant-anthropic-0001"}


def test_list_shows_saved_keys_masked_and_names_an_overriding_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An environment key outranks the saved one, and `list` says which variable does.

    Killed by: src/uclone_x/cli/commands/key.py :: f"{overriding[1]} ({mask_key(overriding[0])}) overrides the saved key"
    Becomes: f"[dim]{spec.key_env_vars[0]} not set[/dim]"
    """
    runner.invoke(app, ["key", "set", "openai", "--key", "sk-openai-key-0001"])
    monkeypatch.setenv("GOOGLE_API_KEY", "AQ.from-the-environment-9")
    result = runner.invoke(app, ["key", "list"], terminal_width=200)
    out = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "sk-ope...0001" in out
    assert "sk-openai-key-0001" not in out
    assert "GOOGLE_API_KEY" in out and "overrides the saved key" in out
    assert "AQ.from-the-environment-9" not in out


def test_set_warns_when_an_environment_variable_will_be_used_instead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-0001")
    result = runner.invoke(app, ["key", "set", "anthropic", "--key", "sk-ant-saved-0001"])
    assert result.exit_code == 0
    assert "ANTHROPIC_API_KEY is set in this environment" in _flat(result.output)


def test_unknown_provider_is_refused_in_plain_words() -> None:
    result = runner.invoke(app, ["key", "set", "acme", "--key", "whatever-key-1"])
    out = _flat(result.output)
    assert result.exit_code == 2
    assert "There is no provider called 'acme' that takes a key." in out
    assert "openai" in out and "gemini" in out
    assert "Traceback" not in out and "ValueError" not in out
    assert not settings_file().exists()


def test_empty_key_is_refused_and_nothing_is_saved() -> None:
    result = runner.invoke(app, ["key", "set", "openai", "--key", "  ''  "])
    assert result.exit_code == 1
    assert "The key is empty, so nothing was saved." in _flat(result.output)
    assert not settings_file().exists()


def test_setup_saves_the_key_in_the_settings_file(tmp_path: Path) -> None:
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(
            app,
            [
                "key",
                "setup",
                "--provider",
                "gemini",
                "--key",
                "AIzaSy123456789012345678901234567890123",
                "--no-open-browser",
            ],
        )
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.output
    assert _saved()["llm_api_keys"] == {"gemini": "AIzaSy123456789012345678901234567890123"}
    assert not (tmp_path / ".env").exists()
