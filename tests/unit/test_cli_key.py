"""Tests for ucx key CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from uclone_x.cli.commands.key import mask_key, sanitize_key, update_env_file
from uclone_x.cli.main import app

runner = CliRunner()


def test_sanitize_key():
    assert sanitize_key("  AIzaSy12345  ") == "AIzaSy12345"
    assert sanitize_key('"sk-ant-12345"') == "sk-ant-12345"
    assert sanitize_key("'sk-proj-12345'") == "sk-proj-12345"


def test_mask_key():
    assert mask_key("short") == "****"
    assert mask_key("AIzaSy1234567890abcdef") == "AIzaSy...cdef"


def test_update_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text('FOO=bar\nGEMINI_API_KEY="old_key"\n', encoding="utf-8")

    update_env_file(env_file, "GEMINI_API_KEY", "new_key_123")
    content = env_file.read_text(encoding="utf-8")
    assert 'GEMINI_API_KEY="new_key_123"' in content
    assert "old_key" not in content
    assert "FOO=bar" in content

    update_env_file(env_file, "OPENAI_API_KEY", "sk-new")
    content2 = env_file.read_text(encoding="utf-8")
    assert 'OPENAI_API_KEY="sk-new"' in content2


def test_cli_key_list():
    result = runner.invoke(app, ["key", "list"])
    assert result.exit_code == 0
    assert "Google Gemini" in result.output
    assert "Anthropic Claude" in result.output
    assert "OpenAI" in result.output


def test_cli_key_setup_direct(tmp_path: Path):
    env_file = tmp_path / ".env"
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
            "--env-file",
            str(env_file),
        ],
    )
    assert result.exit_code == 0
    assert "성공" in result.output
    assert env_file.exists()
    assert 'GEMINI_API_KEY="AIzaSy123456789012345678901234567890123"' in env_file.read_text(
        encoding="utf-8"
    )
