"""Unit tests for ucx llm CLI commands."""

# pyright: reportPrivateUsage=false

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from uclone_x.cli.commands import llm as llm_module
from uclone_x.cli.commands.llm import _check_ollama_endpoint, _check_vllm_endpoint
from uclone_x.cli.main import app
from uclone_x.errors import LLMProviderError

runner = CliRunner()


def test_llm_status_command_online() -> None:
    """Test llm status when endpoints respond with models."""
    mock_models = ["qwen2.5-coder:7b"]
    with patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=mock_models):
        result = runner.invoke(app, ["llm", "status"])
        assert result.exit_code == 0
        assert "Fast Tier" in result.output
        assert "In-Depth Tier" in result.output
        assert "ONLINE" in result.output


def test_llm_status_command_offline() -> None:
    """Test llm status when endpoints are unreachable."""
    with patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=None):
        result = runner.invoke(app, ["llm", "status"])
        assert result.exit_code == 0
        assert "UNREACHABLE" in result.output or "OFFLINE" in result.output
        assert "caffeinate" in result.output
        assert "pmset" in result.output
        assert "sleep" in result.output


def test_llm_pull_fast_tier() -> None:
    """Test llm pull fast alias mapping."""
    with (
        patch("shutil.which", return_value="/usr/local/bin/ollama"),
        patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=[]),
        patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
    ):
        result = runner.invoke(app, ["llm", "pull", "fast"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(["ollama", "pull", "qwen3:1.7b"], check=True)


def test_llm_pull_indepth_tier() -> None:
    """Test llm pull indepth alias mapping."""
    with (
        patch("shutil.which", return_value="/usr/local/bin/ollama"),
        patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=[]),
        patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
    ):
        result = runner.invoke(app, ["llm", "pull", "indepth"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(["ollama", "pull", "qwen3:8b"], check=True)


def test_llm_pull_missing_ollama_cli() -> None:
    """Test llm pull when ollama CLI binary is not installed.

    Killed by: src/uclone_x/cli/commands/llm.py :: if not shutil.which("ollama"):
    """
    with patch("shutil.which", return_value=None):
        result = runner.invoke(app, ["llm", "pull", "fast"])
        assert result.exit_code == 1
        assert "brew install ollama" in result.output
        assert "https://ollama.com" in result.output
        assert "not installed or not found on PATH" in result.output


def test_llm_pull_daemon_unreachable() -> None:
    """Test llm pull when ollama CLI exists but daemon is offline.

    Killed by: src/uclone_x/cli/commands/llm.py :: if _check_ollama_endpoint(daemon_url) is None:
    Becomes: if _check_ollama_endpoint(daemon_url) is not None:
    """
    with (
        patch("shutil.which", return_value="/usr/local/bin/ollama"),
        patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=None),
    ):
        result = runner.invoke(app, ["llm", "pull", "fast"])
        assert result.exit_code == 1
        assert "Ollama daemon is not responding" in result.output
        assert "ollama serve" in result.output


def test_llm_rm_success() -> None:
    """Test llm rm removes a model when the daemon is reachable."""
    with (
        patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=["llama3.2:1b"]),
        patch(
            "uclone_x.cli.commands.llm.delete_model", new=AsyncMock(return_value=None)
        ) as mock_delete,
    ):
        result = runner.invoke(app, ["llm", "rm", "llama3.2:1b"])
        assert result.exit_code == 0
        assert "Successfully removed" in result.output
        mock_delete.assert_awaited_once()
        assert mock_delete.await_args is not None
        assert mock_delete.await_args.args[0] == "llama3.2:1b"


def test_llm_rm_daemon_unreachable() -> None:
    """Test llm rm when the Ollama daemon is offline.

    Killed by: src/uclone_x/cli/commands/llm.py :: if _check_ollama_endpoint(daemon_url) is None:
    Becomes: if _check_ollama_endpoint(daemon_url) is not None:
    """
    with patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=None):
        result = runner.invoke(app, ["llm", "rm", "llama3.2:1b"])
        assert result.exit_code == 1
        assert "Ollama daemon is not responding" in result.output
        assert "ollama serve" in result.output


def test_llm_rm_provider_error() -> None:
    """Test llm rm surfaces an LLMProviderError from the daemon (e.g. unknown model)."""
    with (
        patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=[]),
        patch(
            "uclone_x.cli.commands.llm.delete_model",
            new=AsyncMock(
                side_effect=LLMProviderError("Ollama provider returned status 404: not found")
            ),
        ),
    ):
        result = runner.invoke(app, ["llm", "rm", "no-such-model"])
        assert result.exit_code == 1
        assert "Failed to remove model" in result.output
        assert "status 404" in result.output


def test_check_ollama_endpoint_success() -> None:
    """Test successful model list retrieval from Ollama endpoint with default 5.0s timeout."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(
        {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3:8b"}]}
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        models = _check_ollama_endpoint("http://localhost:11434")
        assert models == ["qwen2.5-coder:7b", "llama3:8b"]
        assert mock_urlopen.call_args is not None
        assert mock_urlopen.call_args.kwargs["timeout"] == 5.0


def test_check_ollama_endpoint_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OLLAMA_CHECK_TIMEOUT env var overrides default timeout."""
    monkeypatch.setenv("OLLAMA_CHECK_TIMEOUT", "10.5")
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({"models": []}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        models = _check_ollama_endpoint("http://localhost:11434")
        assert models == []
        assert mock_urlopen.call_args is not None
        assert mock_urlopen.call_args.kwargs["timeout"] == 10.5


def test_check_ollama_endpoint_invalid_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test invalid OLLAMA_CHECK_TIMEOUT falls back to 5.0."""
    monkeypatch.setenv("OLLAMA_CHECK_TIMEOUT", "not-a-number")
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({"models": []}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        models = _check_ollama_endpoint("http://localhost:11434")
        assert models == []
        assert mock_urlopen.call_args is not None
        assert mock_urlopen.call_args.kwargs["timeout"] == 5.0


def test_check_ollama_endpoint_explicit_timeout() -> None:
    """Test explicit timeout argument takes precedence."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({"models": []}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        models = _check_ollama_endpoint("http://localhost:11434", timeout=3.0)
        assert models == []
        assert mock_urlopen.call_args is not None
        assert mock_urlopen.call_args.kwargs["timeout"] == 3.0


def test_check_ollama_endpoint_exception() -> None:
    """Test error handling during request returns None."""
    with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
        assert _check_ollama_endpoint("http://localhost:11434") is None


def test_check_ollama_endpoint_non_200() -> None:
    """Test non-200 HTTP status code returns None."""
    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert _check_ollama_endpoint("http://localhost:11434") is None


# ======================================================================================
# vLLM's row in `ucx llm status` (#1304)
# ======================================================================================


# Wide enough that Rich does not fold the variable names and the `vllm serve` advice this
# table's whole purpose is to hand the reader; at the default 80 columns the last column is
# narrow enough to break them mid-token.
#
# The width is fixed on the module's console, not asked for through `COLUMNS`: the console
# is built at import, and whether it then honours a `COLUMNS` passed to `runner.invoke`
# depends on what the process looked like when it was built. Public CI on Python 3.12
# rendered these tables at exactly 80 columns with `COLUMNS=200` in the invocation's
# environment, and both assertions on the folded names failed there.
@pytest.fixture
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.setattr(llm_module, "console", Console(width=200))


@pytest.mark.usefixtures("_wide_console")
def test_llm_status_names_the_variable_when_no_vllm_endpoint_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vLLM row is present and unprobed when nothing says where the server is.

    A provider absent from this table is a provider nobody discovers; a provider probed at
    a guessed port reports a refused connection about a server the operator never claimed
    to run (P6). This third state says which variable to set instead, and the assertion
    that nothing was probed is what distinguishes it from a row that guessed and failed.

    Killed by: src/uclone_x/cli/commands/llm.py :: if not has_configured_vllm_endpoint():
    Becomes: if False:
    """
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    with (
        patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=[]),
        patch("uclone_x.cli.commands.llm._check_vllm_endpoint") as probe,
    ):
        result = runner.invoke(app, ["llm", "status"])

    assert result.exit_code == 0
    assert "vLLM" in result.output
    assert "NOT CONFIGURED" in result.output
    assert "VLLM_BASE_URL" in result.output
    assert "VLLM_MODEL" in result.output
    probe.assert_not_called()


@pytest.mark.usefixtures("_wide_console")
def test_llm_status_reports_a_configured_vllm_endpoint_and_the_model_it_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured endpoint is probed, and the model it reports is printed.

    The model string is the point: it is the exact value `vllm serve --model` was given and
    the value `VLLM_MODEL` has to hold for a turn to work, so a row that said only ONLINE
    would leave the operator to guess it.

    Killed by: src/uclone_x/cli/commands/llm.py :: models = _check_vllm_endpoint(vllm_url)
    Becomes: models = None
    """
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box.invalid:8000")
    with (
        patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=[]),
        patch(
            "uclone_x.cli.commands.llm._check_vllm_endpoint",
            return_value=["qwen2.5-coder-32b-instruct"],
        ) as probe,
    ):
        result = runner.invoke(app, ["llm", "status"])

    assert result.exit_code == 0
    assert "ONLINE" in result.output
    assert "qwen2.5-coder-32b-instruct" in result.output
    # Normalized to the `/v1` surface the endpoint actually mounts, not echoed as typed.
    probe.assert_called_once_with("http://gpu-box.invalid:8000/v1")


@pytest.mark.usefixtures("_wide_console")
def test_llm_status_separates_an_unreachable_vllm_endpoint_from_an_unconfigured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An endpoint that does not answer is UNREACHABLE, with the command that starts one.

    Distinct from NOT CONFIGURED above, and the distinction is the diagnosis: one means
    nobody said where the server is, the other that it is not where they said it was.

    Killed by: src/uclone_x/cli/commands/llm.py :: if models is not None:
    Becomes: if models is None:
    """
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box.invalid:8000")
    with (
        patch("uclone_x.cli.commands.llm._check_ollama_endpoint", return_value=[]),
        patch("uclone_x.cli.commands.llm._check_vllm_endpoint", return_value=None),
    ):
        result = runner.invoke(app, ["llm", "status"])

    assert result.exit_code == 0
    assert "UNREACHABLE" in result.output
    assert "vllm serve" in result.output
    assert "NOT CONFIGURED" not in result.output


def test_the_vllm_probe_reads_the_openai_compatible_models_path() -> None:
    """The probe asks `/v1/models` and returns the ids, with no header unless a key is set.

    vLLM speaks OpenAI's surface; Ollama's `/api/tags` does not exist on it, and a probe
    pointed there reports every healthy vLLM server as unreachable. The absent
    `Authorization` is the other half: `vllm serve --api-key` is optional, and an empty
    bearer is a credential a gateway can reject (#385).

    Killed by: src/uclone_x/cli/commands/llm.py :: req = urllib.request.Request(f"{url.rstrip('/')}/models", headers=headers)
    Becomes: req = urllib.request.Request(f"{url.rstrip('/')}/api/tags", headers=headers)
    """
    captured: dict[str, object] = {}

    class _Resp:
        status = 200

        def read(self) -> bytes:
            return json.dumps({"data": [{"id": "qwen2.5-coder-32b-instruct"}]}).encode("utf-8")

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def fake_urlopen(req: Any, timeout: float | None = None) -> _Resp:
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        return _Resp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        models = _check_vllm_endpoint("http://gpu-box.invalid:8000/v1")

    assert models == ["qwen2.5-coder-32b-instruct"]
    assert captured["url"] == "http://gpu-box.invalid:8000/v1/models"
    headers = cast(dict[str, str], captured["headers"])
    assert not any(name.lower() == "authorization" for name in headers)
