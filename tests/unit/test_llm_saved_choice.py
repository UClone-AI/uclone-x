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
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from tests.support.clones import make_clones
from uclone_x.agent.clone_builder import AppScope
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
    save_api_key,
    settings_file,
    update_settings_file,
)
from uclone_x.llm.connectors.vllm import VLLM_ENDPOINT_ENV_VARS
from uclone_x.llm.models import FinishReason, LLMRequest, ModelResponse, TokenUsage

# The agent ids these tests run as. Each is a clone now, since a name no clone carries is
# refused rather than given a home (clone-data-scopes §3.4); persona-less, so each speaks as
# the prompt the test gives it.
_SAVED_CLONES = ("saved", "assistant")


@pytest.fixture(autouse=True)
def _saved_clones() -> None:  # pyright: ignore[reportUnusedFunction]
    make_clones(*_SAVED_CLONES)


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


def _row(kind: str, *, base_url: str | None = None, key: str | None = None) -> dict[str, str]:
    """A connection row whose id is its kind, as setup and `ucx llm use` save one."""
    row = {"id": kind, "kind": kind}
    if base_url is not None:
        row["base_url"] = base_url
    if key is not None:
        row["key"] = key
    return row


def _save_choice(
    kind: str,
    model: str | None = None,
    *,
    base_url: str | None = None,
    key: str | None = None,
    **values: Any,
) -> Path:
    """One connection of ``kind`` and, given ``model``, the default deep model on it."""
    data: dict[str, Any] = {"connections": [_row(kind, base_url=base_url, key=key)], **values}
    if model is not None:
        data["default_models"] = {"deep": f"{kind}/{model}"}
    return _save(**data)


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
    _save_choice("ollama", "qwen3:1.7b", base_url="http://127.0.0.1:11999")

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
    _save_choice("ollama", "qwen3:1.7b")
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
    _save_choice("ollama", "qwen3:1.7b")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert saved_choice_in_effect(provider, base_url) is None


def test_a_saved_connection_this_version_does_not_know_is_refused_naming_the_file() -> None:
    """Ignoring it silently would leave the person no way to find what they configured.

    It used to be dropped on read, so the refusal said no default model was saved -- the
    wrong cause, pointing away from the row. It is read now, as unsupported (P6).

    Killed by: src/uclone_x/llm/connections.py :: written_kind = _text(row.get("kind"))
    Becomes: written_kind = None
    """
    target = _save(
        connections=[{"id": "future", "kind": "some-future-provider"}],
        default_models={"deep": "future/x"},
    )

    with pytest.raises(LLMProviderError) as caught:
        create_llm_connector(fallback_to_mock=False)

    assert not isinstance(caught.value, LLMProviderNotConfiguredError)
    message = _flat(str(caught.value))
    assert f"The model choice saved in {target}" in message
    assert "does not support: some-future-provider" in message
    assert not any(word in message for word in INTERNALS)


def test_a_file_with_only_the_old_model_keys_has_no_model_saved() -> None:
    """The pre-gateway keys are never read: no migration, by ruling."""
    _save(llm_provider="ollama", llm_model="qwen3:1.7b", llm_api_keys={"openai": "sk-test"})

    assert read_saved_choice() is None
    with pytest.raises(LLMProviderNotConfiguredError):
        create_llm_connector(fallback_to_mock=False)


def test_the_refusal_says_no_model_has_been_saved_yet() -> None:
    """The refusal names the new source truthfully, in plain words.

    Killed by: src/uclone_x/llm/connectors/factory.py :: f"{saved_choice_note(saved_choice_file)} "
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

    Setup adds its own connection beside the person's, but the default stays theirs.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: if model is not None and saved_default_models(current).deep is None:
    Becomes: if model is not None:
    """
    _save_choice("openai", "gpt-4o-mini", key="sk-test")

    _written, existing = remember_choice_if_unset(
        provider="ollama", model="qwen3:8b", base_url="http://localhost:11434"
    )

    assert existing is not None and existing.model == "gpt-4o-mini"
    data = json.loads(settings_file().read_text())
    assert data["default_models"] == {"deep": "openai/gpt-4o-mini"}
    assert data["connections"][0] == _row("openai", key="sk-test")


def test_saving_keeps_every_other_setting_in_the_file() -> None:
    """The dashboard keeps read roots and the ComfyUI address in the same file.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: data.update(updates)
    Becomes: data = dict(updates)
    """
    _save(read_roots=["/srv/notes"], comfyui_base_url="http://127.0.0.1:8188")

    remember_choice_if_unset(provider="ollama", model="qwen3:8b", base_url=None)

    data = json.loads(settings_file().read_text())
    assert data["read_roots"] == ["/srv/notes"]
    assert data["comfyui_base_url"] == "http://127.0.0.1:8188"
    assert data["default_models"] == {"deep": "ollama/qwen3:8b"}


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
    target = _save_choice("ollama", "qwen3:1.7b")
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
    _save_choice("ollama", "qwen3:1.7b")
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
    _save_choice("ollama", "qwen3:1.7b")

    assert run.apply_saved_model("mock", None) == (None, None)


def test_a_room_asks_for_the_saved_model_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ucx room say` builds its connector the way `ucx run` does, so it applies the same rule.

    Killed by: src/uclone_x/cli/commands/room.py :: model, saved_notice = apply_saved_model(provider, model)
    Becomes: saved_notice = None
    """
    target = _save_choice("ollama", "qwen3:1.7b")
    monkeypatch.setattr(run, "get_default_llm", _returning(MagicMock()))
    captured: dict[str, Any] = {}

    def _resolver(app: AppScope, **kwargs: Any) -> MagicMock:
        captured["llm_config"] = app.llm_override
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

    The connector is built with the saved model as its own, so a request naming none is
    sent to it -- recorded here as the model the connector itself resolves.

    Killed by: src/uclone_x/llm/connectors/factory.py :: return OllamaConnector(base_url=base_url, model=deep, **kwargs)
    Becomes: return OllamaConnector(base_url=base_url, **kwargs)
    """
    for var in ("OLLAMA_MODEL", "OLLAMA_INDEPTH_MODEL", "OLLAMA_FAST_MODEL"):
        monkeypatch.delenv(var, raising=False)
    _save_choice("ollama", "qwen3:1.7b")
    sent: list[str | None] = []

    async def record(self: OllamaConnector, request: LLMRequest) -> ModelResponse:
        sent.append(self._resolve_model(request.model))
        return _response()

    monkeypatch.setattr(OllamaConnector, "generate", record)

    llm = create_llm_connector()
    asyncio.run(llm.generate(LLMRequest()))
    asyncio.run(llm.generate(LLMRequest(model="default")))

    assert sent == ["qwen3:1.7b", "qwen3:1.7b"]
    assert isinstance(llm, OllamaConnector)


def test_a_model_the_caller_names_still_wins_over_the_saved_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: src/uclone_x/llm/connectors/ollama.py :: return resolve_model(requested, resolve_ollama_model(self._model), "Ollama")
    Becomes: return resolve_model(None, resolve_ollama_model(self._model), "Ollama")
    """
    _save_choice("ollama", "qwen3:1.7b")
    sent: list[str | None] = []

    async def record(self: OllamaConnector, request: LLMRequest) -> ModelResponse:
        sent.append(self._resolve_model(request.model))
        return _response()

    monkeypatch.setattr(OllamaConnector, "generate", record)

    asyncio.run(create_llm_connector().generate(LLMRequest(model="llama3.2")))

    assert sent == ["llama3.2"]


def test_the_saved_model_is_not_given_to_another_providers_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved Ollama model means nothing to OpenAI; the OpenAI connector gets no model.

    Killed by: src/uclone_x/llm/connectors/factory.py :: if conn is None or not same_provider(conn.kind, provider):
    Becomes: if conn is None:
    """
    _save_choice("ollama", "qwen3:1.7b")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-0001")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    llm = create_llm_connector()

    assert llm._default_model is None  # type: ignore[attr-defined]


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

    target = _save_choice("ollama", "qwen3:1.7b")

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
    target = _save_choice("openai", "gpt-4o-mini")

    result = runner.invoke(main.app, ["run", "saved", "--prompt", "hi", "--cwd", str(tmp_path)])

    assert result.exit_code != 0
    assert _shown(f"Using gpt-4o-mini (openai), the model saved in {target}.", result.output)


# --- setup fills in only what is missing ----------------------------------------------


def test_setup_writes_nothing_over_a_saved_connection_and_model() -> None:
    """A re-install finds its connection and a default saved, and leaves both as they are.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: if not any(row.id == kind for row in rows):
    Becomes: if True:
    """
    target = _save_choice("ollama", "qwen3:8b", base_url="http://gpu-box:11434")
    before = target.read_text(encoding="utf-8")

    written, _before = remember_choice_if_unset(
        provider="ollama", model="qwen3:1.7b", base_url="http://localhost:11434"
    )

    assert written is False
    assert target.read_text(encoding="utf-8") == before


def test_setup_keeps_the_address_of_a_connection_saved_without_a_model() -> None:
    """A connection added in Settings keeps its address; setup only names the default."""
    _save(connections=[_row("ollama", base_url="http://gpu-box:11434")])

    written, _before = remember_choice_if_unset(
        provider="ollama", model="qwen3:1.7b", base_url="http://localhost:11434"
    )

    assert written is True
    data = json.loads(settings_file().read_text())
    assert data["connections"] == [_row("ollama", base_url="http://gpu-box:11434")]
    assert data["default_models"] == {"deep": "ollama/qwen3:1.7b"}


# --- a dashboard that was already running --------------------------------------------


def _dashboard(tmp_path: Path) -> Any:
    from uclone_x.ui.app import AgentSessionManager

    return AgentSessionManager(storage_dir=tmp_path / "store", fallback_to_mock=True)


def test_a_dashboard_started_before_setup_keeps_what_setup_saved(tmp_path: Path) -> None:
    """Reproduced: dashboard open, `ucx install`, then any Settings save put back `null`.

    The dashboard now holds no model state at all (the gateway reads the file), so no save
    of its own fields can write a model over the one setup saved; this pins that it stays so.
    """
    dashboard = _dashboard(tmp_path)
    target = dashboard._settings_file
    remember_choice_if_unset(
        provider="ollama", model="qwen3:1.7b", base_url="http://localhost:11434", path=target
    )

    dashboard.update_settings(ui_language="ko")

    saved = read_saved_choice(target)
    assert saved is not None
    assert (saved.provider, saved.model) == ("ollama", "qwen3:1.7b")
    assert json.loads(target.read_text())["ui_language"] == "ko"


def test_a_dashboard_started_before_setup_uses_what_setup_saved(tmp_path: Path) -> None:
    """The gateway reads the file on every call, so nothing has to be adopted."""
    dashboard = _dashboard(tmp_path)
    target = dashboard._settings_file
    remember_choice_if_unset(
        provider="ollama", model="qwen3:1.7b", base_url="http://localhost:11434", path=target
    )

    assert dashboard.get_settings()["default_models"]["deep"] == "ollama/qwen3:1.7b"
    llm, model = dashboard.gateway.default_deep()
    assert isinstance(llm, OllamaConnector)
    assert model == "qwen3:1.7b"


def test_the_settings_file_is_replaced_whole_never_written_in_place(tmp_path: Path) -> None:
    """A reader must never see half a file: the write goes to a temporary file first.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: os.replace(temp, target)
    Becomes: target.write_text(Path(temp).read_text())
    """
    target = tmp_path / "settings.json"
    update_settings_file({"read_roots": ["/srv"]}, path=target)

    assert json.loads(target.read_text()) == {"read_roots": ["/srv"]}
    # No temporary file is left behind; the lock's sidecar is the only other file.
    assert sorted(p.name for p in tmp_path.iterdir()) == [".settings.json.lock", "settings.json"]


# --- `ucx llm use` ---------------------------------------------------------------------


def test_ucx_llm_use_makes_a_model_the_default() -> None:
    """Killed by: src/uclone_x/cli/commands/llm.py :: save_choice(provider=chosen, model=model, base_url=address)
    Becomes: pass
    """
    _save_choice("openai", "gpt-4o-mini", key="sk-test", read_roots=["/srv"])

    result = runner.invoke(main.app, ["llm", "use", "qwen3:1.7b"])

    assert result.exit_code == 0, result.output
    saved = read_saved_choice()
    assert saved is not None
    assert (saved.provider, saved.model, saved.api_key) == ("ollama", "qwen3:1.7b", None)
    data = json.loads(settings_file().read_text())
    assert data["read_roots"] == ["/srv"]
    assert data["default_models"] == {"deep": "ollama/qwen3:1.7b"}
    # Not deleted: kept on the OpenAI connection, and not sent to Ollama.
    assert data["connections"][0] == _row("openai", key="sk-test")
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
    _save_choice("ollama", "qwen3:8b", base_url="http://gpu-box:11434")

    result = runner.invoke(main.app, ["llm", "use", "qwen3:1.7b"])

    assert result.exit_code == 0, result.output
    saved = read_saved_choice()
    assert saved is not None and saved.base_url == "http://gpu-box:11434"


def test_switching_away_and_back_finds_the_saved_key_again() -> None:
    """`ucx llm use` to Ollama and back to OpenAI used to delete the OpenAI key on the way."""
    _save_choice("openai", "gpt-4o-mini", key="sk-test")

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


def _anthropic_default_beside_an_openai_key() -> None:
    _save(
        connections=[_row("openai", key="sk-test"), _row("anthropic")],
        default_models={"deep": "anthropic/claude-sonnet"},
    )


def test_one_services_key_is_never_sent_to_another() -> None:
    """An OpenAI key saved beside an Anthropic default is not given to Anthropic."""
    _anthropic_default_beside_an_openai_key()

    saved = read_saved_choice()
    assert saved is not None and saved.provider == "anthropic"
    assert saved.api_key is None
    # Built from the file, Anthropic finds no key at all rather than OpenAI's.
    with pytest.raises(LLMCredentialsNotConfiguredError):
        create_llm_connector()


def test_a_key_saved_under_google_goes_to_the_gemini_connection() -> None:
    """Killed by: src/uclone_x/llm/connectors/saved_choice.py :: kind = _known_provider(clean)
    Becomes: kind = clean.lower()
    """
    _save_choice("gemini", "gemini-x")

    save_api_key("google", "g-test")

    saved = read_saved_choice()
    assert saved is not None and saved.api_key == "g-test"
    assert json.loads(settings_file().read_text())["connections"] == [_row("gemini", key="g-test")]


async def test_a_dashboard_does_not_send_another_services_key(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "settings.json").write_text(
        json.dumps(
            {
                "connections": [_row("openai", key="sk-test"), _row("anthropic")],
                "default_models": {"deep": "anthropic/claude-sonnet"},
            }
        ),
        encoding="utf-8",
    )

    dashboard = _dashboard(tmp_path)
    llm, _model = dashboard.gateway.default_deep()

    assert llm is not None
    assert getattr(llm, "api_key", None) != "sk-test"
    with pytest.raises(LLMCredentialsNotConfiguredError):
        await llm.generate(LLMRequest())


def test_a_dashboard_hands_a_choice_saved_after_it_started_to_the_next_call(
    tmp_path: Path,
) -> None:
    """A choice saved after the dashboard started reaches the next room turn, unadopted."""
    dashboard = _dashboard(tmp_path)
    assert dashboard.gateway.default_deep() == (None, None)
    remember_choice_if_unset(
        provider="ollama",
        model="qwen3:1.7b",
        base_url="http://localhost:11434",
        path=dashboard._settings_file,
    )

    llm, model = dashboard.gateway.default_deep()

    assert isinstance(llm, OllamaConnector)
    assert model == "qwen3:1.7b"


def test_a_dashboard_with_its_own_storage_ignores_the_session_roots_choice(
    tmp_path: Path,
) -> None:
    """A dashboard whose storage saved no model does not borrow the session root's.

    Killed by: src/uclone_x/llm/gateway.py :: return self._settings_path if self._settings_path is not None else settings_file()
    Becomes: return settings_file()

    Mutated, the gateway reads `<session root>/settings.json` and the clones get the Ollama
    model saved there, which this dashboard's Settings never showed.
    """
    _save_choice("ollama", "qwen3:1.7b", base_url="http://127.0.0.1:11999")
    dashboard = _dashboard(tmp_path)
    assert dashboard.settings_file != settings_file()

    assert dashboard.gateway.default_deep() == (None, None)
    assert dashboard.gateway.connections() == ()


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
        update_settings_file({"ui_language": "ko"}, path=target)
        done.set()

    with lock_file(target).open("a", encoding="utf-8") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        writer = threading.Thread(target=write)
        writer.start()
        assert not done.wait(0.3), "the writer did not wait for the lock"
        fcntl.flock(held, fcntl.LOCK_UN)
    assert done.wait(5)
    writer.join()

    assert json.loads(target.read_text()) == {"read_roots": ["/srv"], "ui_language": "ko"}
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
        update_settings_file({"ui_language": "ko"}, path=target)
    finally:
        lock.chmod(0o600)

    assert json.loads(target.read_text()) == {"ui_language": "ko"}


def test_writes_still_work_where_there_is_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `fcntl` (not POSIX) the write is unserialised, not refused."""
    monkeypatch.setattr(saved_choice_module, "fcntl", None)
    target = tmp_path / "settings.json"

    update_settings_file({"ui_language": "ko"}, path=target)

    assert json.loads(target.read_text()) == {"ui_language": "ko"}


def test_ollama_model_outranks_the_saved_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A variable wins over the file for the model, as it does for the provider.

    Killed by: src/uclone_x/llm/connectors/factory.py :: return found[0]
    Becomes: pass
    """
    _save_choice("ollama", "qwen3:1.7b")
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
    target = _save_choice("vllm", "qwen3-small", base_url="http://gpu:8000")
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
