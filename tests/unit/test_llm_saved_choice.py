"""The saved model choice: one file, read by `ucx run`, rooms and the dashboard.

Measured before this change: `ucx install --yes` ended with `setup llm: ready
model=qwen3:8b`, and the next `ucx run --prompt hi` refused with "No LLM provider is
configured". Setup had saved nothing any terminal command read. These tests pin the four
parts of the fix:

1. the connector factory builds the saved choice when no argument or environment
   variable names a provider -- and only then -- and asks it for the saved model;
2. setup saves its model only where nothing is saved, keeping the rest of the file;
3. `ucx run`, `ucx room` and `ucx acp` ask for the saved model and say where it came from;
4. the refusal, when nothing is saved, says so in plain words;
5. a dashboard that was already running neither erases the saved choice nor ignores it;
6. `ucx llm use` replaces the saved choice on request.

The session root is already a per-test temporary directory (`tests/conftest.py`), so the
settings file written here is never the invoking user's.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from uclone_x.cli import main
from uclone_x.cli.commands import room as room_cmd
from uclone_x.cli.commands import run
from uclone_x.core.provenance import Provenance
from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    LLMProviderError,
    LLMProviderNotConfiguredError,
)
from uclone_x.llm.connectors import saved_choice as saved_choice_module
from uclone_x.llm.connectors.factory import create_llm_connector, saved_choice_in_effect
from uclone_x.llm.connectors.ollama import OLLAMA_ENDPOINT_ENV_VARS, OllamaConnector
from uclone_x.llm.connectors.saved_choice import (
    lock_file,
    read_saved_choice,
    remember_choice_if_unset,
    settings_file,
    update_settings_file,
)
from uclone_x.llm.connectors.vllm import VLLM_ENDPOINT_ENV_VARS
from uclone_x.llm.models import FinishReason, LLMRequest, ModelResponse, StreamChunk, TokenUsage

runner = CliRunner()

#: Words that would mean a message leaked an implementation detail to the person reading it.
INTERNALS = ("Traceback", "JSONDecodeError", "ValueError", "OSError", "Exception")


@pytest.fixture(autouse=True)
def _nothing_in_the_environment(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No provider, credential or endpoint variable: the state a fresh install leaves."""
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_MODEL",
        "OLLAMA_INDEPTH_MODEL",
        "OLLAMA_FAST_MODEL",
        "VLLM_MODEL",
        *OLLAMA_ENDPOINT_ENV_VARS,
        *VLLM_ENDPOINT_ENV_VARS,
    ):
        monkeypatch.delenv(var, raising=False)


def _save(**values: Any) -> Path:
    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(values), encoding="utf-8")
    return target


def _flat(text: str) -> str:
    """Rich wraps at the terminal width; compare on single spaces."""
    return " ".join(text.split())


def _shown(expected: str, output: str) -> bool:
    """Whether `expected` was printed, ignoring every space.

    Rich also breaks a long path mid-word, which `_flat` cannot rejoin.
    """
    return "".join(expected.split()) in "".join(output.split())


# --- 1. The factory -------------------------------------------------------------------


def test_the_factory_builds_the_saved_choice_when_nothing_else_names_one() -> None:
    """The defect itself: a saved Ollama model is what an unconfigured terminal now gets.

    Killed by: src/uclone_x/llm/connectors/factory.py :: if saved is not None:
    Becomes: if False:

    Mutated, the factory skips the saved choice and refuses as before the fix.
    """
    _save(llm_provider="ollama", llm_model="qwen3:1.7b", llm_base_url="http://127.0.0.1:11999")

    llm = create_llm_connector(fallback_to_mock=False)

    assert isinstance(llm, OllamaConnector)
    assert str(llm.base_url).startswith("http://127.0.0.1:11999")


def test_a_credential_in_the_environment_outranks_the_saved_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flags and environment variables keep precedence over the file.

    Killed by: src/uclone_x/llm/connectors/factory.py :: for name in _CREDENTIAL_ENV_VARS:
    Becomes: for name in ():

    Mutated, an exported OPENAI_API_KEY loses to a model saved months ago.
    """
    _save(llm_provider="ollama", llm_model="qwen3:1.7b")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-saved-choice")

    assert saved_choice_in_effect() is None


@pytest.mark.parametrize(
    ("env", "provider", "base_url"),
    [
        ({"LLM_PROVIDER": "mock"}, None, None),
        ({"OLLAMA_BASE_URL": "http://127.0.0.1:11434"}, None, None),
        ({}, "mock", None),
        ({}, None, "http://127.0.0.1:11434"),
    ],
    ids=["LLM_PROVIDER", "endpoint-variable", "explicit-provider", "explicit-base-url"],
)
def test_anything_the_caller_names_outranks_the_saved_choice(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    provider: str | None,
    base_url: str | None,
) -> None:
    _save(llm_provider="ollama", llm_model="qwen3:1.7b")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert saved_choice_in_effect(provider, base_url) is None


def test_a_saved_provider_this_version_does_not_know_is_refused_naming_the_file() -> None:
    """Ignoring it would report "nothing configured" to someone who configured something."""
    target = _save(llm_provider="some-future-provider", llm_model="x")

    with pytest.raises(LLMProviderError) as caught:
        create_llm_connector(fallback_to_mock=False)

    message = str(caught.value)
    assert str(target) in message
    assert "some-future-provider" in message


def test_the_refusal_says_no_model_has_been_saved_yet() -> None:
    """The refusal names the new source truthfully, in plain words.

    Killed by: src/uclone_x/llm/connectors/factory.py :: f"{saved_choice_note()} "
    Becomes: ""
    """
    with pytest.raises(LLMProviderNotConfiguredError) as caught:
        create_llm_connector(fallback_to_mock=False)

    message = _flat(str(caught.value))
    assert "No model has been saved yet" in message
    assert "`ucx install` sets up a local model and saves it" in message
    assert not any(word in message for word in ("Traceback", "JSONDecodeError"))


def test_the_refusal_says_the_saved_settings_could_not_be_read() -> None:
    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json", encoding="utf-8")

    with pytest.raises(LLMProviderNotConfiguredError) as caught:
        create_llm_connector(fallback_to_mock=False)

    message = _flat(str(caught.value))
    assert f"The saved settings at {target} could not be read." in message
    assert not any(word in message for word in ("Traceback", "JSONDecodeError", "Expecting"))


# --- 2. Saving the first choice -------------------------------------------------------


def test_setup_saves_its_model_when_none_is_saved() -> None:
    written, existing = remember_choice_if_unset(
        provider="ollama", model="qwen3:8b", base_url="http://localhost:11434"
    )

    assert (written, existing) == (True, None)
    saved = read_saved_choice()
    assert saved is not None
    assert (saved.provider, saved.model, saved.base_url) == (
        "ollama",
        "qwen3:8b",
        "http://localhost:11434",
    )


def test_setup_keeps_a_model_the_person_already_chose() -> None:
    """A later `ucx install` must not replace what someone picked in Settings.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: if before is not None and before.provider != provider.lower():
    Becomes: if False:
    """
    _save(llm_provider="openai", llm_model="gpt-4o-mini")

    written, existing = remember_choice_if_unset(
        provider="ollama", model="qwen3:8b", base_url="http://localhost:11434"
    )

    assert written is False
    assert existing is not None and existing.model == "gpt-4o-mini"
    assert json.loads(settings_file().read_text())["llm_provider"] == "openai"


def test_saving_keeps_every_other_setting_in_the_file() -> None:
    """The dashboard keeps read roots and the ComfyUI address in the same file.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: data.update(updates)
    Becomes: data = dict(updates)
    """
    _save(llm_provider=None, read_roots=["/srv/notes"], comfyui_base_url="http://127.0.0.1:8188")

    remember_choice_if_unset(provider="ollama", model="qwen3:8b", base_url=None)

    data = json.loads(settings_file().read_text())
    assert data["read_roots"] == ["/srv/notes"]
    assert data["comfyui_base_url"] == "http://127.0.0.1:8188"
    assert data["llm_model"] == "qwen3:8b"


def test_an_unreadable_settings_file_is_left_as_it_is() -> None:
    """Replacing a file this code cannot parse would throw away whatever was in it."""
    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{half written", encoding="utf-8")

    with pytest.raises(ValueError):
        remember_choice_if_unset(provider="ollama", model="qwen3:8b", base_url=None)

    assert target.read_text(encoding="utf-8") == "{half written"


# --- 3. The terminal commands ---------------------------------------------------------


def _returning(value: MagicMock) -> Callable[..., MagicMock]:
    """A typed stand-in for a constructor or factory that ignores its arguments."""

    def _stand_in(*args: object, **kwargs: object) -> MagicMock:
        return value

    return _stand_in


def _answering_llm() -> MagicMock:
    llm = MagicMock()
    llm.provider_name = "ollama"
    llm.generate = AsyncMock(
        return_value=ModelResponse(
            finish_reason=FinishReason.STOP,
            content="ok",
            usage=TokenUsage(provider="ollama", input_tokens=1, output_tokens=1, total_tokens=2),
            provenance=Provenance.primary("test"),
        )
    )
    return llm


def test_ucx_run_asks_for_the_saved_model_and_says_where_it_came_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Setup may pick `qwen3:1.7b` for a small machine; asking that daemon for 8b fails.

    Killed by: src/uclone_x/cli/commands/run.py :: chosen = model or saved.model
    Becomes: chosen = model
    """
    target = _save(llm_provider="ollama", llm_model="qwen3:1.7b")
    llm = _answering_llm()
    monkeypatch.setattr(run, "create_llm_connector", _returning(llm))

    result = runner.invoke(main.app, ["run", "saved", "--prompt", "hi", "--cwd", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert llm.generate.call_args.args[0].model == "qwen3:1.7b"
    shown = _flat(result.output)
    assert _shown(f"Using qwen3:1.7b (ollama), the model saved in {target}.", shown)
    assert not any(word in shown for word in INTERNALS)


def test_an_explicit_model_flag_outranks_the_saved_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _save(llm_provider="ollama", llm_model="qwen3:1.7b")
    llm = _answering_llm()
    monkeypatch.setattr(run, "create_llm_connector", _returning(llm))

    result = runner.invoke(
        main.app, ["run", "saved", "-m", "llama3.2", "--prompt", "hi", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert llm.generate.call_args.args[0].model == "llama3.2"


def test_no_notice_when_the_saved_choice_is_not_what_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The notice would be false when `--provider` chose the connector instead."""
    _save(llm_provider="ollama", llm_model="qwen3:1.7b")

    assert run.apply_saved_model("mock", None) == (None, None)


def test_a_room_asks_for_the_saved_model_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ucx room say` builds its connector the way `ucx run` does, so it applies the same rule.

    Killed by: src/uclone_x/cli/commands/room.py :: model, saved_notice = apply_saved_model(provider, model)
    Becomes: saved_notice = None
    """
    target = _save(llm_provider="ollama", llm_model="qwen3:1.7b")
    monkeypatch.setattr(run, "get_default_llm", _returning(MagicMock()))
    captured: dict[str, Any] = {}

    def _resolver(host: object, **kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(room_cmd, "RoomAgentResolver", _resolver)
    monkeypatch.setattr(room_cmd, "build_selector_chain", _returning(MagicMock()))
    monkeypatch.setattr(room_cmd, "RoomOrchestrator", _returning(MagicMock()))

    room_cmd.build_orchestrator(store=MagicMock(), policy=MagicMock())

    assert captured["llm_config"].model_name == "qwen3:1.7b"
    shown = _flat(capsys.readouterr().out)
    assert _shown(f"Using qwen3:1.7b (ollama), the model saved in {target}.", shown)
    assert not any(word in shown for word in INTERNALS)


# --- the saved model reaches callers that name none -----------------------------------


def _response() -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content="ok",
        usage=TokenUsage(provider="ollama", input_tokens=1, output_tokens=1, total_tokens=2),
        provenance=Provenance.primary("test"),
    )


def test_a_caller_that_names_no_model_is_sent_to_the_saved_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACP, A2A and the eval answerer name no model; they got `qwen3:8b`, which setup never pulled.

    Killed by: src/uclone_x/llm/connectors/factory.py :: return _default_to_saved_model(connector, choice.model)
    Becomes: return connector
    """
    _save(llm_provider="ollama", llm_model="qwen3:1.7b")
    sent: list[str | None] = []

    async def record(self: OllamaConnector, request: LLMRequest) -> ModelResponse:
        sent.append(request.model)
        return _response()

    monkeypatch.setattr(OllamaConnector, "generate", record)

    llm = create_llm_connector()
    asyncio.run(llm.generate(LLMRequest()))
    asyncio.run(llm.generate(LLMRequest(model="default")))

    assert sent == ["qwen3:1.7b", "qwen3:1.7b"]
    assert isinstance(llm, OllamaConnector)
    assert llm._default_model == "qwen3:1.7b"


def test_a_model_the_caller_names_still_wins_over_the_saved_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: src/uclone_x/llm/connectors/factory.py :: if _names_no_model(request.model):
    Becomes: if True:
    """
    _save(llm_provider="ollama", llm_model="qwen3:1.7b")
    sent: list[str | None] = []

    async def record(self: OllamaConnector, request: LLMRequest) -> ModelResponse:
        sent.append(request.model)
        return _response()

    monkeypatch.setattr(OllamaConnector, "generate", record)

    asyncio.run(create_llm_connector().generate(LLMRequest(model="llama3.2")))

    assert sent == ["llama3.2"]


def test_a_streamed_turn_is_sent_to_the_saved_model_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: src/uclone_x/llm/connectors/factory.py :: def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
    Becomes: def _unused_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
    """
    _save(llm_provider="ollama", llm_model="qwen3:1.7b")
    sent: list[str | None] = []

    async def record(self: OllamaConnector, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        sent.append(request.model)
        return
        yield  # an async generator that yields nothing

    monkeypatch.setattr(OllamaConnector, "stream", record)

    async def drain() -> None:
        async for _chunk in create_llm_connector().stream(LLMRequest()):
            pass

    asyncio.run(drain())

    assert sent == ["qwen3:1.7b"]


class _MemoryOpened(Exception):
    """Stops `start_acp_server` after its connector is built."""


def test_acp_says_which_saved_model_it_uses_on_stderr_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ACP speaks JSON-RPC on stdout; a notice there would corrupt the editor's stream.

    Killed by: src/uclone_x/cli/commands/acp.py :: err_console.print(saved_notice, markup=False, highlight=False)
    Becomes: console.print(saved_notice, markup=False, highlight=False)
    """
    from uclone_x.cli.commands import acp

    target = _save(llm_provider="ollama", llm_model="qwen3:1.7b")

    def stop(agent_id: str) -> Any:
        raise _MemoryOpened()

    monkeypatch.setattr(acp, "memory_for_agent_id", stop)

    with pytest.raises(_MemoryOpened):
        acp.start_acp_server()

    out, err = capsys.readouterr()
    assert out == ""
    assert _shown(f"Using qwen3:1.7b (ollama), the model saved in {target}.", _flat(err))


def test_ucx_run_names_the_saved_choice_before_that_provider_fails(
    tmp_path: Path,
) -> None:
    """A saved OpenAI choice with no key fails to build; the person should know why OpenAI.

    The notice used to be printed after the connector was built, so this failure came
    with no word of the saved choice that caused it.

    Killed by: src/uclone_x/cli/commands/run.py :: notices.print(f"[dim]{escape(saved_notice)}[/dim]")
    Becomes: pass
    """
    target = _save(llm_provider="openai", llm_model="gpt-4o-mini")

    result = runner.invoke(main.app, ["run", "saved", "--prompt", "hi", "--cwd", str(tmp_path)])

    assert result.exit_code != 0
    assert _shown(f"Using gpt-4o-mini (openai), the model saved in {target}.", result.output)


# --- setup fills in only what is missing ----------------------------------------------


def test_setup_fills_in_a_missing_address_but_keeps_the_saved_model() -> None:
    """Killed by: src/uclone_x/llm/connectors/saved_choice.py :: if model is not None and _clean(current.get("llm_model")) is None:
    Becomes: if model is not None:
    """
    _save(llm_provider="ollama", llm_model="qwen3:8b")

    written, _before = remember_choice_if_unset(
        provider="ollama", model="qwen3:1.7b", base_url="http://localhost:11434"
    )

    saved = read_saved_choice()
    assert written is True
    assert saved is not None
    assert (saved.model, saved.base_url) == ("qwen3:8b", "http://localhost:11434")


def test_setup_keeps_an_address_saved_without_a_provider() -> None:
    """Killed by: src/uclone_x/llm/connectors/saved_choice.py :: if base_url is not None and _clean(current.get("llm_base_url")) is None:
    Becomes: if base_url is not None:
    """
    _save(llm_provider=None, llm_base_url="http://gpu-box:11434")

    remember_choice_if_unset(
        provider="ollama", model="qwen3:1.7b", base_url="http://localhost:11434"
    )

    saved = read_saved_choice()
    assert saved is not None
    assert (saved.provider, saved.model, saved.base_url) == (
        "ollama",
        "qwen3:1.7b",
        "http://gpu-box:11434",
    )


# --- a dashboard that was already running --------------------------------------------


def _dashboard(tmp_path: Path) -> Any:
    from uclone_x.ui.app import AgentSessionManager

    return AgentSessionManager(storage_dir=tmp_path / "store", fallback_to_mock=True)


def test_a_dashboard_started_before_setup_keeps_what_setup_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduced: dashboard open, `ucx install`, then any Settings save put back `null`.

    Killed by: src/uclone_x/ui/app.py :: changes, path=self._settings_file, replace_unreadable_with=everything
    Becomes: everything, path=self._settings_file, replace_unreadable_with=everything
    """
    dashboard = _dashboard(tmp_path)
    target = dashboard._settings_file
    remember_choice_if_unset(
        provider="ollama", model="qwen3:1.7b", base_url="http://localhost:11434", path=target
    )
    # The dashboard has a provider of its own from the environment, so it adopts nothing
    # and what it holds for the LLM keys is still empty.
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    dashboard.update_settings(comfyui_base_url="http://127.0.0.1:8188")

    saved = read_saved_choice(target)
    assert saved is not None
    assert (saved.provider, saved.model) == ("ollama", "qwen3:1.7b")
    assert json.loads(target.read_text())["comfyui_base_url"] == "http://127.0.0.1:8188"


def test_a_dashboard_started_before_setup_uses_what_setup_saved(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/ui/app.py :: self._configured_provider = saved.provider
    Becomes: pass
    """
    dashboard = _dashboard(tmp_path)
    target = dashboard._settings_file
    remember_choice_if_unset(
        provider="ollama", model="qwen3:1.7b", base_url="http://localhost:11434", path=target
    )

    dashboard.update_settings(comfyui_base_url="http://127.0.0.1:8188")

    assert dashboard.configured_provider == "ollama"
    assert dashboard.get_settings()["llm_model"] == "qwen3:1.7b"
    saved = read_saved_choice(target)
    assert saved is not None and (saved.provider, saved.model) == ("ollama", "qwen3:1.7b")


def test_the_settings_file_is_replaced_whole_never_written_in_place(tmp_path: Path) -> None:
    """A reader must never see half a file: the write goes to a temporary file first.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: os.replace(temp, target)
    Becomes: target.write_text(Path(temp).read_text())
    """
    target = tmp_path / "settings.json"
    update_settings_file({"llm_provider": "ollama"}, path=target)

    assert json.loads(target.read_text()) == {"llm_provider": "ollama"}
    # No temporary file is left behind; the lock's sidecar is the only other file.
    assert sorted(p.name for p in tmp_path.iterdir()) == [".settings.json.lock", "settings.json"]


# --- `ucx llm use` ---------------------------------------------------------------------


def test_ucx_llm_use_makes_a_model_the_default() -> None:
    """Killed by: src/uclone_x/cli/commands/llm.py :: save_choice(provider=chosen, model=model, base_url=address)
    Becomes: pass
    """
    _save(
        llm_provider="openai", llm_model="gpt-4o-mini", llm_api_key="sk-test", read_roots=["/srv"]
    )

    result = runner.invoke(main.app, ["llm", "use", "qwen3:1.7b"])

    assert result.exit_code == 0, result.output
    saved = read_saved_choice()
    assert saved is not None
    assert (saved.provider, saved.model, saved.api_key) == ("ollama", "qwen3:1.7b", None)
    data = json.loads(settings_file().read_text())
    assert data["read_roots"] == ["/srv"]
    # Not deleted: kept for OpenAI, the provider it was saved for, and not sent to Ollama.
    assert (data["llm_api_key"], data["llm_api_key_provider"]) == ("sk-test", "openai")
    shown = _flat(result.output)
    assert (
        "qwen3:1.7b (ollama) is now the default model for `ucx run`, rooms and the dashboard."
        in shown
    )
    assert not any(word in shown for word in INTERNALS)


def test_ucx_llm_use_keeps_the_saved_address_for_the_same_provider() -> None:
    """Killed by: src/uclone_x/cli/commands/llm.py :: address = before.base_url
    Becomes: pass
    """
    _save(llm_provider="ollama", llm_model="qwen3:8b", llm_base_url="http://gpu-box:11434")

    result = runner.invoke(main.app, ["llm", "use", "qwen3:1.7b"])

    assert result.exit_code == 0, result.output
    saved = read_saved_choice()
    assert saved is not None and saved.base_url == "http://gpu-box:11434"


def test_switching_away_and_back_finds_the_saved_key_again() -> None:
    """`ucx llm use` to Ollama and back to OpenAI used to delete the OpenAI key on the way.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: data["llm_api_key_provider"] = old_provider
    Becomes: pass

    Mutated, the untagged key is taken for Ollama's on the first switch and so is not
    OpenAI's on the way back.
    """
    # Saved before keys were tagged: it belongs to the provider saved with it.
    _save(llm_provider="openai", llm_model="gpt-4o-mini", llm_api_key="sk-test")

    runner.invoke(main.app, ["llm", "use", "qwen3:1.7b"])
    between = read_saved_choice()
    assert between is not None and between.api_key is None  # not Ollama's
    result = runner.invoke(main.app, ["llm", "use", "gpt-4o-mini", "--provider", "openai"])

    assert result.exit_code == 0, result.output
    saved = read_saved_choice()
    assert saved is not None and (saved.provider, saved.api_key) == ("openai", "sk-test")
    llm = create_llm_connector()
    assert llm.provider_name == "openai"
    assert getattr(llm, "api_key", None) == "sk-test"


def test_one_services_key_is_never_sent_to_another() -> None:
    """An OpenAI key kept in the file while Anthropic is saved is not given to Anthropic.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: if key is None or not same_provider(key_owner(data), provider):
    Becomes: if key is None:
    """
    _save(
        llm_provider="anthropic",
        llm_model="claude-sonnet",
        llm_api_key="sk-test",
        llm_api_key_provider="openai",
    )

    saved = read_saved_choice()
    assert saved is not None and saved.provider == "anthropic"
    assert saved.api_key is None
    # Built from the file, Anthropic finds no key at all rather than OpenAI's.
    with pytest.raises(LLMCredentialsNotConfiguredError):
        create_llm_connector()


def test_a_key_saved_for_gemini_is_used_under_its_other_name_google() -> None:
    """Killed by: src/uclone_x/llm/connectors/saved_choice.py :: return "gemini" if name == "google" else name
    Becomes: return name
    """
    _save(llm_provider="google", llm_api_key="g-test", llm_api_key_provider="gemini")

    saved = read_saved_choice()
    assert saved is not None and saved.api_key == "g-test"


def test_a_dashboard_does_not_send_another_services_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/app.py :: if owner is None or same_provider(owner, provider):
    Becomes: if True:
    """
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "LLM_PROVIDER"):
        # Recorded as absent, so what the dashboard exports is removed afterwards.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    store = tmp_path / "store"
    store.mkdir()
    (store / "settings.json").write_text(
        json.dumps(
            {
                "llm_provider": "anthropic",
                "llm_api_key": "sk-test",
                "llm_api_key_provider": "openai",
            }
        ),
        encoding="utf-8",
    )

    dashboard = _dashboard(tmp_path)

    assert dashboard.configured_api_key is None
    assert os.environ.get("ANTHROPIC_API_KEY") is None
    assert dashboard.get_settings()["llm_api_key_set"] is False


def test_a_dashboard_records_which_provider_a_new_key_is_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/app.py :: state_updates["_configured_api_key_provider"] = str(key_for).strip().lower()
    Becomes: pass
    """
    for name in ("OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    dashboard = _dashboard(tmp_path)

    dashboard.update_settings(llm_provider="openai", llm_api_key="sk-test", llm_model="gpt-4o")
    dashboard.update_settings(llm_provider="ollama", llm_model="qwen3:1.7b")

    data = json.loads(dashboard._settings_file.read_text())
    assert (data["llm_api_key"], data["llm_api_key_provider"]) == ("sk-test", "openai")
    assert dashboard.configured_api_key is None


def test_a_dashboard_hands_an_adopted_choice_to_open_rooms(tmp_path: Path) -> None:
    """A choice saved after the dashboard started reaches rooms the way a Settings save does.

    Killed by: src/uclone_x/ui/app.py :: self._install_llm(adopted)
    Becomes: self._llm = adopted
    """
    dashboard = _dashboard(tmp_path)
    received: list[Any] = []
    dashboard.on_llm_replaced(received.append)
    remember_choice_if_unset(
        provider="ollama",
        model="qwen3:1.7b",
        base_url="http://localhost:11434",
        path=dashboard._settings_file,
    )

    dashboard.get_settings()

    assert len(received) == 1
    assert isinstance(received[0], OllamaConnector)
    assert received[0] is dashboard.default_llm


# --- the settings lock and the environment's model variables ---------------------------


def test_a_second_writer_waits_for_the_lock(tmp_path: Path) -> None:
    """Two processes merging at once would each drop the other's keys; the lock orders them.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: fcntl.flock(fd, fcntl.LOCK_EX)
    Becomes: pass
    """
    import fcntl

    target = tmp_path / "settings.json"
    update_settings_file({"read_roots": ["/srv"]}, path=target)
    done = threading.Event()

    def write() -> None:
        update_settings_file({"llm_provider": "ollama"}, path=target)
        done.set()

    with lock_file(target).open("a", encoding="utf-8") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        writer = threading.Thread(target=write)
        writer.start()
        assert not done.wait(0.3), "the writer did not wait for the lock"
        fcntl.flock(held, fcntl.LOCK_UN)
    assert done.wait(5)
    writer.join()

    assert json.loads(target.read_text()) == {"read_roots": ["/srv"], "llm_provider": "ollama"}
    # Only its owner can open it, as for the settings file beside it.
    assert lock_file(target).stat().st_mode & 0o777 == 0o600


def test_a_lock_file_that_cannot_be_opened_does_not_stop_the_save(tmp_path: Path) -> None:
    """A lock left owned by root after `sudo` must not turn every later save into a failure.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: except OSError:
    Becomes: except FileExistsError:
    """
    if os.geteuid() == 0:
        pytest.skip("root opens any file, so the lock cannot be made unopenable")
    target = tmp_path / "settings.json"
    lock = lock_file(target)
    lock.touch()
    lock.chmod(0)  # what another owner's 0600 file is to this user
    try:
        update_settings_file({"llm_provider": "ollama"}, path=target)
    finally:
        lock.chmod(0o600)

    assert json.loads(target.read_text()) == {"llm_provider": "ollama"}


def test_writes_still_work_where_there_is_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `fcntl` (not POSIX) the write is unserialised, not refused."""
    monkeypatch.setattr(saved_choice_module, "fcntl", None)
    target = tmp_path / "settings.json"

    update_settings_file({"llm_provider": "ollama"}, path=target)

    assert json.loads(target.read_text()) == {"llm_provider": "ollama"}


def test_ollama_model_outranks_the_saved_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A variable wins over the file for the model, as it does for the provider.

    Killed by: src/uclone_x/llm/connectors/factory.py :: if choice.model is None or model_env_override(choice.provider) is not None:
    Becomes: if choice.model is None:
    """
    _save(llm_provider="ollama", llm_model="qwen3:1.7b")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    sent: list[str | None] = []

    async def record(self: OllamaConnector, request: LLMRequest) -> ModelResponse:
        sent.append(request.model)
        return _response()

    monkeypatch.setattr(OllamaConnector, "generate", record)

    llm = create_llm_connector()
    asyncio.run(llm.generate(LLMRequest()))

    # Left to the connector, which reads `OLLAMA_MODEL`, instead of rewritten to the file's.
    assert sent == [None]
    assert isinstance(llm, OllamaConnector)
    assert llm._default_model == "llama3.2"


def test_ucx_run_asks_for_the_model_variable_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: src/uclone_x/cli/commands/run.py :: variable = model_env_override(saved.provider)
    Becomes: variable = None
    """
    target = _save(llm_provider="vllm", llm_model="qwen3-small", llm_base_url="http://gpu:8000")
    monkeypatch.setenv("VLLM_MODEL", "qwen3-large")
    # The saved address is what builds vLLM here, not a variable.
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    model, notice = run.apply_saved_model(None, None)

    assert model == "qwen3-large"
    assert notice is not None
    shown = _flat(notice)
    assert _shown(
        f"Using qwen3-large (vllm): the provider saved in {target}, with the model VLLM_MODEL names.",
        shown,
    )
    assert "the model saved in" not in shown
    assert not any(word in shown for word in INTERNALS)


def test_ucx_llm_use_says_which_variable_outranks_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saved is not in use while a variable outranks it, and the message says which.

    Killed by: src/uclone_x/cli/commands/llm.py :: winner = what_outranks_saved_choice()
    Becomes: winner = None
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    result = runner.invoke(main.app, ["llm", "use", "qwen3:1.7b"])

    assert result.exit_code == 0, result.output
    shown = _flat(result.output)
    assert "is now the default" not in shown
    assert (
        "Saved qwen3:1.7b (ollama) as the default, but it is not used yet: OPENAI_API_KEY is "
        "set in this shell, and it takes priority over the saved default. Unset "
        "OPENAI_API_KEY to use qwen3:1.7b." in shown
    )
    assert "sk-env" not in shown
    assert not any(word in shown for word in INTERNALS)


def test_ucx_llm_use_says_when_a_model_variable_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: src/uclone_x/cli/commands/llm.py :: model_variable = model_env_override(chosen)
    Becomes: model_variable = None
    """
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")

    result = runner.invoke(main.app, ["llm", "use", "qwen3:1.7b"])

    assert result.exit_code == 0, result.output
    shown = _flat(result.output)
    assert "is now the default" not in shown
    assert (
        "Saved qwen3:1.7b (ollama) as the default, but OLLAMA_MODEL is set in this shell and "
        "names another model, which is used instead. Unset OLLAMA_MODEL to use qwen3:1.7b." in shown
    )
    assert not any(word in shown for word in INTERNALS)


def test_ucx_llm_use_refuses_an_unknown_provider_plainly() -> None:
    result = runner.invoke(main.app, ["llm", "use", "x", "--provider", "nosuch"])

    assert result.exit_code == 2
    shown = _flat(result.output)
    assert "nosuch is not a provider this version knows" in shown
    assert not any(word in shown for word in INTERNALS)
    assert read_saved_choice() is None
