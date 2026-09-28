# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Unit tests for interactive and non-interactive agent REPL execution and provider resolution (Issues #54, #141)."""

import json
import logging
import os
import pty
import subprocess
import sys
import threading
import tty
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from tests.support import nonblocking_stdout
from uclone_x.agent import clone_builder
from uclone_x.agent.hooks import BaseHook, HookAction, HookContext, HookDecision
from uclone_x.agent.models import AgentLLMConfig, TurnResult
from uclone_x.agent.prompts import HONEST_REPORTING
from uclone_x.agent.session import SessionState, SessionStore, default_session_storage_dir
from uclone_x.cli import main
from uclone_x.cli.commands import run
from uclone_x.core.provenance import Provenance
from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    LLMError,
    LLMProviderError,
    MissingDependencyError,
)
from uclone_x.llm.connectors.factory import create_llm_connector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.room.models import ParticipantKind
from uclone_x.room.service import RoomService, participant_session_id
from uclone_x.room.store import RoomStore
from uclone_x.telemetry import TelemetryTracer
from uclone_x.tools import LocalTool, ToolRegistry

runner = CliRunner()


def test_get_default_llm_and_mock_resolution() -> None:
    """Explicit mock provider resolves to MockLLMConnector."""
    llm_from_helper = run.get_default_llm(provider="mock")
    assert isinstance(llm_from_helper, MockLLMConnector)

    llm_from_factory = create_llm_connector(provider="mock")
    assert isinstance(llm_from_factory, MockLLMConnector)

    ollama_llm = run.get_default_llm(provider="ollama")
    assert isinstance(ollama_llm, OllamaConnector)


def test_unconfigured_cloud_providers_raise_the_credential_error_not_a_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing credential raises `LLMCredentialsNotConfiguredError`, never a mock.

    This asserted `LLMProviderError`, which the factory produced by raising its own
    per-provider pre-check ahead of the connector. Those pre-checks are gone (#397):
    `LLMCredentialsNotConfiguredError` is a **sibling** of `LLMProviderError` under
    `LLMError`, and #385/PR #393 chose the credential class deliberately, so the factory
    handing back the provider class defeated that choice one step before the removed
    `except Exception` did.

    `LLMError` is what a caller catching "the provider could not be set up" should name,
    and `uclone_x.cli.commands.run` now does.
    """
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    for prov in ("openai", "anthropic", "gemini", "google"):
        with pytest.raises(LLMCredentialsNotConfiguredError) as exc_info:
            run.get_default_llm(provider=prov)
        # The message names the variables consulted, which is what the pre-check supplied.
        assert "_API_KEY" in str(exc_info.value), str(exc_info.value)
        assert isinstance(exc_info.value, LLMError)
        assert not isinstance(exc_info.value, LLMProviderError)

        with pytest.raises(LLMCredentialsNotConfiguredError):
            create_llm_connector(provider=prov, fallback_to_mock=False)


def test_invalid_provider_fails_cli_with_clear_error() -> None:
    """Running ucx run with an invalid provider exits with non-zero code and prints an error message."""
    result = runner.invoke(main.app, ["run", "--provider", "invalid_provider"])
    assert result.exit_code != 0
    assert "LLM Error:" in result.output
    assert "Unsupported LLM provider" in result.output


def test_unconfigured_provider_fails_cli_with_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running ucx run with an unconfigured cloud provider exits with non-zero code and clear error.

    Mutation this exists to catch (narrowing the handler misses the credential error):
        -   except LLMError as exc:
        +   except LLMProviderError as exc:
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(main.app, ["run", "--provider", "openai"])
    assert result.exit_code != 0
    assert "LLM Error:" in result.output
    assert "OPENAI_API_KEY" in result.output


@pytest.mark.asyncio
async def test_run_agent_repl_single_shot_mocked_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-shot agent execution records span and outputs agent reasoning."""
    mock_llm = MagicMock()
    # A stand-in for an `LLMProviderProtocol` must satisfy it: `provider_name` is declared
    # `str`, and `SpanRecord` validates strictly, so a bare `MagicMock` attribute is
    # rejected. The REPL now records the provider that actually served rather than the
    # one requested, which is why this mock has to be conformant.
    mock_llm.provider_name = "mock"
    mock_llm.generate = AsyncMock(
        return_value=ModelResponse(
            finish_reason=FinishReason.STOP,
            content="Single shot response",
            usage=TokenUsage(
                provider="mock",
                input_tokens=5,
                output_tokens=5,
                total_tokens=10,
            ),
            provenance=Provenance.primary("test"),
        )
    )

    def fake_create(**kwargs: Any) -> MagicMock:
        return mock_llm

    monkeypatch.setattr(
        run,
        "create_llm_connector",
        fake_create,
    )

    await run.run_agent_repl_async(agent_name="test-agent", prompt="Hello there!")
    assert mock_llm.generate.called


def test_cli_run_command_single_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI invoke with prompt flags dispatches correctly."""
    mock_run = MagicMock()
    monkeypatch.setattr("uclone_x.cli.commands.run.run_agent_repl", mock_run)

    result = runner.invoke(main.app, ["run", "my-agent", "--prompt", "Explain quantum computing"])
    assert result.exit_code == 0
    assert mock_run.called
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("agent_name") == "my-agent"
    assert kwargs.get("prompt") == "Explain quantum computing"


def test_cli_run_command_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI invoke with interactive options dispatches correctly."""
    mock_run = MagicMock()
    monkeypatch.setattr("uclone_x.cli.commands.run.run_agent_repl", mock_run)

    result = runner.invoke(main.app, ["run", "default", "--provider", "mock"])
    assert result.exit_code == 0
    assert mock_run.called
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("provider") == "mock"


def test_cli_run_system_option_has_no_hardcoded_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--system`'s Typer option must default to `None` (#1206).

    A hardcoded string default here (`"You are a helpful UClone-X AI Assistant."`)
    permanently shadowed `run_agent_repl_async`'s `system_prompt or
    compose_system_prompt(...)` fallback, because `system_prompt` was never falsy when
    invoked through the CLI — every `ucx run` without an explicit `--system` silently
    lost `HONEST_REPORTING`, `ARTIFACT_REPORTING`, `IMAGE_GENERATION` and
    `ENVIRONMENT_REPAIR`, regardless of model.

    Killed by: src/uclone_x/cli/main.py :: system_prompt=system_prompt
    Becomes: system_prompt=system_prompt or "You are a helpful UClone-X AI Assistant."
    """
    mock_run = MagicMock()
    monkeypatch.setattr("uclone_x.cli.commands.run.run_agent_repl", mock_run)

    result = runner.invoke(main.app, ["run", "default", "--provider", "mock", "--prompt", "hi"])
    assert result.exit_code == 0
    assert mock_run.call_args.kwargs.get("system_prompt") is None


def test_run_command_without_explicit_system_sends_the_composed_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The `ucx run` CLI path actually reaches `compose_system_prompt()` (#1206).

    `compose_system_prompt()` was already tested in isolation
    (`test_agent_prompts.py`), and that test suite stayed green while the CLI silently
    sent a 40-character generic string instead — nothing asserted that the CLI's own
    wiring reached the composed prompt. This drives the real `run` command end to end
    and reads the persisted session back, rather than inspecting `compose_system_prompt`
    directly, so it fails the way #1206 actually failed.

    Killed by: src/uclone_x/cli/main.py :: system_prompt=system_prompt
    Becomes: system_prompt=system_prompt or "You are a helpful UClone-X AI Assistant."
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())

    result = runner.invoke(
        main.app,
        ["run", "prompt-agent", "--provider", "mock", "--prompt", "hi", "--cwd", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output

    state = store.load(_only_seat(store, "prompt-agent"))
    assert state is not None, "the run persisted no session; the harness is not reaching it"
    system_messages = [m.content or "" for m in state.messages if m.role == MessageRole.SYSTEM]
    assert system_messages, "no system message was persisted"
    assert HONEST_REPORTING in system_messages[0]
    assert system_messages[0] != "You are a helpful UClone-X AI Assistant."


def test_run_command_explicit_system_still_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An operator's own `--system` is sent verbatim, not appended to or replaced (#1206).

    The half of the fix that is easy to lose: removing the hardcoded default must not
    make the composed prompt unconditional. What an operator passes is what the model
    gets.

    Killed by: src/uclone_x/cli/commands/run.py :: system_prompt or compose_system_prompt(
    Becomes: compose_system_prompt(
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())

    result = runner.invoke(
        main.app,
        [
            "run",
            "operator-agent",
            "--provider",
            "mock",
            "--system",
            "Answer only in haiku.",
            "--prompt",
            "hi",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output

    state = store.load(_only_seat(store, "operator-agent"))
    assert state is not None, "the run persisted no session; the harness is not reaching it"
    system_messages = [m.content or "" for m in state.messages if m.role == MessageRole.SYSTEM]
    assert system_messages == ["Answer only in haiku."]


@pytest.mark.asyncio
async def test_run_agent_repl_interactive_slash_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interactive REPL handles slash commands cleanly."""
    inputs = iter(
        ["/help", "/tools", "/ontology", "/status", "/history", "/reset", "Hello bot", "/exit"]
    )

    def fake_ask(*args: Any, **kwargs: Any) -> str:
        return next(inputs)

    monkeypatch.setattr(
        "rich.prompt.Prompt.ask",
        fake_ask,
    )

    await run.run_agent_repl_async(agent_name="interactive-agent", provider="mock")


# ======================================================================================
# The REPL's span drain, driven through the real loop (issue #187)
# ======================================================================================


class _CapturingExporter:
    """Exporter that records each batch and can fail, standing in for a collector."""

    def __init__(self, *, fail: bool = False) -> None:
        self.batches: list[tuple[str, ...]] = []
        self.fail = fail

    async def export_spans(self, spans: Any) -> None:
        self.batches.append(tuple(s.span_id for s in spans))
        if self.fail:
            raise RuntimeError("collector unreachable")

    async def export_metrics(self, metrics: Any) -> None:
        return None

    async def shutdown(self) -> None:
        return None


def _mock_llm() -> MagicMock:
    llm = MagicMock()
    # A stand-in for an `LLMProviderProtocol` must satisfy it: `provider_name` is declared
    # `str`, and `SpanRecord` validates strictly, so a bare `MagicMock` attribute is
    # rejected. The REPL now records the provider that actually served rather than the
    # one requested, which is why this mock has to be conformant.
    llm.provider_name = "mock"
    llm.generate = AsyncMock(
        return_value=ModelResponse(
            finish_reason=FinishReason.STOP,
            content="ok",
            usage=TokenUsage(provider="mock", input_tokens=1, output_tokens=1, total_tokens=2),
            provenance=Provenance.primary("test"),
        )
    )
    return llm


def _drive_repl(
    monkeypatch: pytest.MonkeyPatch, turns: int, *, fail: bool
) -> tuple[TelemetryTracer, _CapturingExporter]:
    """Wire a real REPL run with an observable tracer and exporter, then feed it turns."""
    tracer = TelemetryTracer()
    exporter = _CapturingExporter(fail=fail)

    def _connector(**kwargs: Any) -> MagicMock:
        return _mock_llm()

    def _tracer(*args: Any, **kwargs: Any) -> TelemetryTracer:
        return tracer

    def _exporter(*args: Any, **kwargs: Any) -> _CapturingExporter:
        return exporter

    inputs = iter([*(f"question {i}" for i in range(turns)), "/exit"])

    def _ask(*args: Any, **kwargs: Any) -> str:
        return next(inputs)

    monkeypatch.setattr(run, "create_llm_connector", _connector)
    # No binder: a local connector's binder would embed against the Ollama on this machine.
    monkeypatch.setattr(clone_builder, "connector_tool_binder", _no_binder)
    monkeypatch.setattr(run, "TelemetryTracer", _tracer)
    monkeypatch.setattr(run, "create_telemetry_exporter", _exporter)
    monkeypatch.setattr("rich.prompt.Prompt.ask", _ask)
    return tracer, exporter


@pytest.mark.asyncio
async def test_repl_exports_each_span_once_and_leaves_the_buffer_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driven through the real REPL loop, not a replica of it (#187).

    The first version of this test reimplemented `run.py`'s `finally` block in a helper
    and asserted against the replica. Mutation testing showed what that was worth:
    reverting the call site to the #187 defect — `tracer.clear()` on both the success and
    the failure branch — left the whole suite green. A drain contract that is only
    exercised by a copy of the caller is decoration.
    """
    tracer, exporter = _drive_repl(monkeypatch, turns=3, fail=False)

    await run.run_agent_repl_async(agent_name="repl-drain", provider="mock")

    exported = [span_id for batch in exporter.batches for span_id in batch]
    assert exported, "the REPL exported nothing; the harness is not reaching the drain"
    assert len(exported) == len(set(exported)), "a span was exported more than once"
    # Drained every turn: an undrained REPL grows for the life of the process.
    assert tracer.get_completed_spans() == ()
    assert tracer.dropped_span_count == 0


@pytest.mark.asyncio
async def test_repl_accounts_for_spans_lost_to_a_failing_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collector outage loses the turn's spans observably, not silently (#187).

    This is the actual defect behind #187's CLI half, and it is not the one the issue
    described: `tracer.clear()` sat in the `finally` block *outside* the `try`, so it ran
    whether the export had succeeded or failed. Measured on `5f43d16`, four turns against
    a failing exporter left an empty buffer and no record that anything was lost.
    """
    tracer, exporter = _drive_repl(monkeypatch, turns=3, fail=True)

    await run.run_agent_repl_async(agent_name="repl-drop", provider="mock")

    assert exporter.batches, "the harness never reached the export call"
    assert tracer.get_completed_spans() == ()
    assert tracer.dropped_span_count > 0, "spans vanished with no accounting"
    assert set(tracer.drop_reasons) == {"cli_export_failed"}


def test_run_reaches_the_unconfigured_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """`./ucx run` with nothing configured refuses, rather than building a dead connector.

    #539 added that refusal and could not reach it: `provider` defaulted to the literal
    `"ollama"` at every CLI entry point, so the factory returned at its explicit-provider
    branch long before the unconfigured one. The fix it shipped was true of the library and
    false of the product — the very user P0 names never saw it.
    """
    from uclone_x.errors import LLMProviderNotConfiguredError
    from uclone_x.llm.connectors.ollama import OLLAMA_ENDPOINT_ENV_VARS

    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        *OLLAMA_ENDPOINT_ENV_VARS,
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(LLMProviderNotConfiguredError):
        run.get_default_llm()


def test_run_honours_the_environment_it_previously_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured credential reaches `./ucx run`.

    The hardcoded default did not merely hide the refusal: it made the CLI ignore
    `LLM_PROVIDER` and every credential variable, so a user who exported `OPENAI_API_KEY`
    and ran `./ucx run` still got Ollama.
    """
    from uclone_x.llm.connectors.ollama import OLLAMA_ENDPOINT_ENV_VARS
    from uclone_x.llm.connectors.openai import OpenAIConnector

    for var in ("LLM_PROVIDER", *OLLAMA_ENDPOINT_ENV_VARS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-followup")

    assert isinstance(run.get_default_llm(), OpenAIConnector)

    # An explicit choice still wins over the environment.
    from uclone_x.llm.connectors.ollama import OllamaConnector

    assert isinstance(run.get_default_llm(provider="ollama"), OllamaConnector)


# ======================================================================================
# A failed turn is reported as a failure, not as the agent's reply (issue #953)
# ======================================================================================

_TURN_FAILURE = "provider exploded 953"


def _raising_llm() -> MagicMock:
    """A conformant connector whose every call fails, the way an unreachable provider does."""
    llm = _mock_llm()
    llm.generate = AsyncMock(side_effect=RuntimeError(_TURN_FAILURE))
    return llm


def _isolated_run_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SessionStore:
    """Point the CLI's session store at `tmp_path`, and return a reader over the same root."""
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    return SessionStore(storage_dir=default_session_storage_dir())


def _pin_the_new_room_id(monkeypatch: pytest.MonkeyPatch) -> str:
    """Give a run started without `--session-id` a room id with no digits in it (#1885).

    The run prints the id of the room it opens. A random hex id holds `404` or `400` often
    enough that a test asserting those codes stay off the terminal failed on its own
    notice, one run in a few hundred. A fixed id keeps that assertion whole.
    """
    import uclone_x.room.one_seat as one_seat

    room_id = "room_pinned_for_the_test"
    monkeypatch.setattr(one_seat, "new_one_seat_room_id", lambda: room_id)
    return room_id


def _assistant_texts(store: SessionStore, session_id: str) -> list[str]:
    state = store.load(session_id)
    assert state is not None, "the run persisted no session; the harness is not reaching it"
    return [m.content or "" for m in state.messages if m.role == MessageRole.ASSISTANT]


def _seat(room_id: str, agent: str) -> str:
    """The session `agent` keeps in room `room_id` -- where a `--session-id` run saves."""
    return participant_session_id(room_id, agent)


def _only_seat(store: SessionStore, agent: str) -> str:
    """The one seat session `agent` holds, for a run that started a room of its own."""
    ids = [
        s
        for s in store.list_session_ids()
        if s.startswith("sess_room__") and s.endswith(f"__{agent}")
    ]
    assert len(ids) == 1, ids
    return ids[0]


def _use_connector(monkeypatch: pytest.MonkeyPatch, llm: MagicMock) -> None:
    def _connector(**kwargs: Any) -> MagicMock:
        return llm

    monkeypatch.setattr(run, "create_llm_connector", _connector)
    # No binder: a local connector's binder would embed against the Ollama on this machine.
    monkeypatch.setattr(clone_builder, "connector_tool_binder", _no_binder)


def _feed_prompts(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    inputs = iter(lines)

    def _ask(*args: Any, **kwargs: Any) -> str:
        return next(inputs)

    monkeypatch.setattr("rich.prompt.Prompt.ask", _ask)


def _break_the_turn(monkeypatch: pytest.MonkeyPatch, shape: str) -> None:
    """Make the next turn fail in one of the two shapes the CLI receives.

    `result`: the connector raises, and `BaseAgent.execute_turn` converts that into a
    `TurnResult` carrying `error` -- what an unreachable provider actually produces. Before
    #953 this printed `(No response content)` on stdout and exited 0.
    `raise`: `execute_turn` itself raises, which reaches the CLI's own `except` -- the shape
    the #950 reviewer measured, printed as `[Execution Error]: ...` and exited 0.
    """
    if shape == "result":
        _use_connector(monkeypatch, _raising_llm())
        return

    async def _explode(self: Any, content: str, **_kwargs: Any) -> Any:
        raise RuntimeError(_TURN_FAILURE)

    _use_connector(monkeypatch, _mock_llm())
    monkeypatch.setattr("uclone_x.agent.base.BaseAgent.execute_turn", _explode)


@pytest.mark.parametrize("shape", ["result", "raise"])
def test_a_failed_single_shot_turn_exits_non_zero_with_the_error_on_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shape: str
) -> None:
    """`ucx run --prompt` after a failed turn: exit 1, error on stderr, nothing on stdout.

    A script cannot tell a failed turn from a successful one by anything but the exit
    status, and it reads the reply from stdout. Before #953 both lied: exit 0, and the
    error text (or `(No response content)`) printed where the reply goes.

    Killed by: src/uclone_x/cli/commands/run.py :: raise typer.Exit(code=FAILED_TURN_EXIT_CODE)
    Becomes: return
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _break_the_turn(monkeypatch, shape)

    result = runner.invoke(
        main.app, ["run", "failing-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == run.FAILED_TURN_EXIT_CODE == 1, result.output
    assert "Execution failed" in result.stderr
    assert _TURN_FAILURE in result.stderr
    assert "(No response content)" not in result.stdout
    assert _TURN_FAILURE not in result.stdout
    # The session is still written -- the user's message was said -- but no assistant
    # message carries the failure as if the agent had replied with it.
    assert not any(
        _TURN_FAILURE in text
        for text in _assistant_texts(store, _only_seat(store, "failing-agent"))
    )


_NO_TOOLS_SENTENCE = (
    "The model deepseek-r1:14b can't use tools, which UClone-X clones need. Pick a "
    "model that supports tools, for example qwen3:8b. Choose it with --model."
)


def _ollama_refusing_tools() -> OllamaConnector:
    """A real Ollama connector whose daemon refuses the model's tools, as Ollama does."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(
                400,
                text='{"error":"registry.ollama.ai/library/deepseek-r1:14b does not support tools"}',
            )
        return httpx.Response(200, json={"models": []})

    return OllamaConnector(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_a_model_without_tools_is_reported_in_plain_words_on_the_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`ucx run` on a model without tool support says so, names the remedy, and nothing else.

    It printed the connector's `LLMProviderError` text -- a status code and Ollama's JSON
    body -- after a logged traceback. The model name and the remedy are all a user can act
    on, and a traceback for a choice of model reads as a crash.

    Killed by: src/uclone_x/agent/turn_executor.py :: logger.info("Turn for agent %s refused: %s", agent_id, lacking_tools)
    Becomes: logger.exception("Error executing turn for agent %s", agent_id)
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _ollama_refusing_tools())  # type: ignore[arg-type]

    result = runner.invoke(
        main.app,
        ["run", "failing-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
        + ["--model", "deepseek-r1:14b"],
    )

    assert result.exit_code == run.FAILED_TURN_EXIT_CODE, result.output
    stderr = " ".join(result.stderr.split())  # Rich wraps long lines
    assert _NO_TOOLS_SENTENCE in stderr
    assert "Settings" not in stderr  # the command line has a flag, not a Settings page
    for internal in ("Traceback", "status 400", "{", "LLMProviderError", "LLMStreamInterrupted"):
        assert internal not in result.output, internal
    # The traceback went to the log, and with no handler configured Python prints a
    # WARNING-or-worse record, traceback and all, on the terminal.
    loud = [r for r in caplog.records if r.levelno >= logging.WARNING or r.exc_info]
    assert not loud, [r.getMessage() for r in loud]


def test_a_retired_model_is_reported_with_the_flag_that_changes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`ucx run` on a model the provider no longer serves names it and says `--model` (#1630).

    Killed by: src/uclone_x/cli/commands/run.py :: "model_unavailable": "Choose a model with --model.",
    Becomes: "model_unavailable_unused": "Choose a model with --model.",
    """
    from uclone_x.errors import ModelNotAvailableError

    class Retired(MockLLMConnector):
        async def generate(self, request: LLMRequest) -> ModelResponse:
            raise ModelNotAvailableError(provider="Google", model="gemini-1.5-pro")

    _isolated_run_env(monkeypatch, tmp_path)
    room_id = _pin_the_new_room_id(monkeypatch)
    _use_connector(monkeypatch, Retired(responses=[]))  # type: ignore[arg-type]

    result = runner.invoke(
        main.app,
        ["run", "failing-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
        + ["--model", "gemini-1.5-pro"],
    )

    assert result.exit_code == run.FAILED_TURN_EXIT_CODE, result.output
    stderr = " ".join(result.stderr.split())  # Rich wraps long lines
    assert "The model gemini-1.5-pro is not available from Google." in stderr
    assert "Choose a model with --model." in stderr
    assert "Settings" not in stderr  # the command line has a flag, not a Settings page
    assert room_id in result.output, "the pin did not reach the run; the id is random again"
    for internal in ("Traceback", "404", "{", "LLMProviderError", "ModelNotAvailableError"):
        assert internal not in result.output, internal
    loud = [r for r in caplog.records if r.levelno >= logging.WARNING or r.exc_info]
    assert not loud, [r.getMessage() for r in loud]


def test_a_revoked_gemini_key_reaches_the_terminal_as_one_plain_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Through the real connector: Google's 400 body stays off the terminal, the fix is named.

    Python prints an unconfigured WARNING record to stderr as it is, so the raw body is only
    kept off the terminal while nothing on this path logs it at WARNING or above.
    """
    import httpx

    from uclone_x.llm.connectors.gemini import GeminiConnector

    body = (
        '{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", '
        '"status": "INVALID_ARGUMENT"}}'
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(400, text=body))
    )
    _isolated_run_env(monkeypatch, tmp_path)
    room_id = _pin_the_new_room_id(monkeypatch)
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    connector = GeminiConnector(api_key="bad", model="gemini-2.5-flash", http_client=client)
    _use_connector(monkeypatch, connector)  # type: ignore[arg-type]

    result = runner.invoke(
        main.app, ["run", "gemini-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == run.FAILED_TURN_EXIT_CODE, result.output
    stderr = " ".join(result.stderr.split())
    assert "Google did not accept the API key." in stderr
    assert "Save a new key with `ucx key set gemini`." in stderr
    assert room_id in result.output, "the pin did not reach the run; the id is random again"
    for internal in ("{", "400", "INVALID_ARGUMENT", "Traceback", "ProviderAuthError"):
        assert internal not in result.output, internal
    loud = [r for r in caplog.records if r.levelno >= logging.WARNING or r.exc_info]
    assert not loud, [r.getMessage() for r in loud]


@pytest.mark.parametrize(
    ("stop_reason", "remedy"),
    [
        ("provider_auth", " Save a new key with `ucx key set <provider>`."),
        ("provider_unreachable", " If you set a custom endpoint, check that address too."),
        ("provider_quota", ""),
        ("provider_outage", ""),
    ],
)
def test_each_provider_failure_gets_the_command_line_remedy_that_fits_it(
    stop_reason: str, remedy: str
) -> None:
    """Where retrying later is the whole remedy, the Core's sentence already says so."""
    result = TurnResult(
        turn_index=1,
        content="",
        error="Plain sentence.",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        provenance=Provenance.primary("fake"),
    )

    assert run._turn_failure(result) == f"Plain sentence.{remedy}"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("provider", "env", "remedy"),
    [
        ("Google", {}, "Save a new key with `ucx key set gemini`."),
        ("Anthropic", {}, "Save a new key with `ucx key set anthropic`."),
        ("Google", {"GOOGLE_API_KEY": "AQ.env"}, "Set a new key in GOOGLE_API_KEY."),
        ("Anthropic", {"ANTHROPIC_API_KEY": "sk-ant-env"}, "Set a new key in ANTHROPIC_API_KEY."),
        (None, {}, "Save a new key with `ucx key set <provider>`."),
    ],
)
def test_a_rejected_key_names_where_that_key_came_from(
    provider: str | None,
    env: dict[str, str],
    remedy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A beginner does not know where the key is kept; the head does, so it says.

    An environment variable outranks the saved key, so while one is set it is the thing to
    change; otherwise the key came from the settings file, and `ucx key set` replaces it.

    Killed by: src/uclone_x/cli/commands/run.py :: remedy = provider_key_remedy(result.provider_failure.provider)
    Becomes: remedy = provider_key_remedy(None)

    Killed by: src/uclone_x/cli/commands/run.py :: return f"Set a new key in {overriding[1]}."
    Becomes: return f"Save a new key with `ucx key set {spec.id}`."
    """
    from uclone_x.agent.models import ProviderFailure
    from uclone_x.errors import ProviderAuthError

    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)

    failure = ProviderFailure.of(ProviderAuthError(provider="P", model="m"))
    result = TurnResult(
        turn_index=1,
        content="",
        error="Plain sentence.",
        stop_reason="provider_auth",
        provider_failure=failure.model_copy(update={"provider": provider}),
        provenance=Provenance.primary("fake"),
    )

    assert run._turn_failure(result) == f"Plain sentence. {remedy}"  # pyright: ignore[reportPrivateUsage]


def test_a_connector_with_no_model_is_refused_in_one_plain_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No `--model`, no saved model, no model variable: the turn is refused before any request.

    The Core's sentence says what is missing; this head adds where, which is its flag.

    Killed by: src/uclone_x/agent/turn_executor.py :: return "model_unavailable", failure.message, failure
    Becomes: return stop_reason, failure.message, failure
    """
    import httpx

    from uclone_x.llm.connectors.gemini import GeminiConnector

    reached: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        reached.append(str(request.url))
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, GeminiConnector(api_key="k", http_client=client))  # type: ignore[arg-type]

    result = runner.invoke(
        main.app, ["run", "gemini-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == run.FAILED_TURN_EXIT_CODE, result.output
    stderr = " ".join(result.stderr.split())
    assert "No model is chosen for Google. Choose a model with --model." in stderr
    assert stderr.count("--model") == 1
    assert "Settings" not in stderr  # the command line has a flag, not a Settings page
    assert reached == []
    for internal in ("Traceback", "LLMModelNotConfiguredError", "qwen3", "gemini-1.5"):
        assert internal not in result.output, internal


def test_a_failed_single_shot_turn_is_not_printed_where_the_reply_goes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The failure goes to stderr specifically, not to the stdout console the reply uses.

    Killed by: src/uclone_x/cli/commands/run.py :: err_console.print(f"[bold red]✖ Execution failed:
    Becomes: console.print(f"[bold red]✖ Execution failed:
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _break_the_turn(monkeypatch, "result")

    result = runner.invoke(
        main.app, ["run", "failing-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert _TURN_FAILURE in result.stderr
    assert _TURN_FAILURE not in result.stdout
    assert "Execution failed" not in result.stdout


def test_a_failed_turn_is_not_recorded_in_the_session_as_the_agents_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The persisted session holds the user's message and no assistant message for the failure.

    `--session-id` resumes this history on the next invocation, so an error recorded as
    the agent's reply would be fed back to the model as something it said.

    Killed by: src/uclone_x/cli/commands/run.py :: _report_failed_turn(failure)
    Becomes: _report_failed_turn(failure); from uclone_x.llm.models import ChatMessage; agent.load_history((*agent.history, ChatMessage(role=MessageRole.ASSISTANT, content=failure)))
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _break_the_turn(monkeypatch, "result")

    result = runner.invoke(
        main.app, ["run", "failing-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code != 0
    state = store.load(_only_seat(store, "failing-agent"))
    assert state is not None
    assert [m.content for m in state.messages if m.role == MessageRole.USER] == ["hello"]
    assert not any(
        _TURN_FAILURE in text
        for text in _assistant_texts(store, _only_seat(store, "failing-agent"))
    )


def test_a_successful_single_shot_turn_exits_zero_with_the_reply_on_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The counterpart: a turn that answers exits 0 and prints the reply on stdout only.

    Since #960 stdout is the reply and nothing else; the agent's name is on stderr.
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())

    result = runner.invoke(
        main.app, ["run", "fine-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "ok\n"
    assert "fine-agent" in result.stderr
    assert "Execution failed" not in result.output


def test_a_missing_extra_during_the_turn_propagates_to_the_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`MissingDependencyError` is not relabelled as a turn failure.

    The console-script launcher (`uclone_x.shells.entry`) owns the one-line named-extra
    message; catching it here would print it as `Execution failed` and hide which extra to
    install behind a generic label.

    Killed by: src/uclone_x/cli/commands/run.py :: except MissingDependencyError:  # the launcher reports it, not this turn
    Becomes: except MissingDependencyError if False else ():
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())

    async def _needs_extra(self: Any, content: str, **_kwargs: Any) -> Any:
        raise MissingDependencyError(extra="code", package="tree_sitter")

    monkeypatch.setattr("uclone_x.agent.base.BaseAgent.execute_turn", _needs_extra)

    result = runner.invoke(
        main.app, ["run", "extra-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert isinstance(result.exception, MissingDependencyError), result.output
    assert "Execution failed" not in result.output


def test_a_missing_extra_during_a_repl_turn_also_propagates_to_the_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REPL keeps running after a failed turn, but not after a missing extra.

    A missing extra fails every later turn the same way, so staying alive would only repeat
    the failure under a label that does not name the extra.

    Killed by: src/uclone_x/cli/commands/run.py :: except MissingDependencyError:  # as in single-shot mode: the launcher's
    Becomes: except MissingDependencyError if False else ():
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())

    async def _needs_extra(self: Any, content: str, **_kwargs: Any) -> Any:
        raise MissingDependencyError(extra="code", package="tree_sitter")

    monkeypatch.setattr("uclone_x.agent.base.BaseAgent.execute_turn", _needs_extra)
    _feed_prompts(monkeypatch, "hello", "/exit")

    result = runner.invoke(main.app, ["run", "extra-repl", "--cwd", str(tmp_path)])

    assert isinstance(result.exception, MissingDependencyError), result.output
    assert "Execution failed" not in result.output


def test_an_interrupted_single_shot_run_exits_130_not_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ctrl+C before the reply exits 130, the status Typer gives every other command.

    Killed by: src/uclone_x/cli/commands/run.py :: raise typer.Exit(code=130) from None
    Becomes: return
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())

    async def _interrupted(self: Any, content: str, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr("uclone_x.agent.base.BaseAgent.execute_turn", _interrupted)

    result = runner.invoke(
        main.app, ["run", "irq-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 130, result.output
    assert "Interrupted" in result.stderr


def test_a_failed_repl_turn_keeps_the_repl_alive_and_is_not_shown_as_a_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In the REPL a failed turn is reported on stderr, and the next turn still runs.

    Before #953 the REPL printed `✖ Execution Error` and then `[Fallback Error]: ...`
    under the agent's name, as its reply. A REPL is interactive, so it stays alive; what
    changes is that the failure is labelled as one and never shown or stored as a reply.

    Killed by: src/uclone_x/cli/commands/run.py :: _report_failed_turn(failure_in_turn)
    Becomes: console.print(failure_in_turn)
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    llm = _mock_llm()
    answered = llm.generate.return_value
    llm.generate = AsyncMock(side_effect=[RuntimeError(_TURN_FAILURE), answered])
    _use_connector(monkeypatch, llm)
    _feed_prompts(monkeypatch, "first question", "second question", "/exit")

    result = runner.invoke(main.app, ["run", "repl-agent", "--cwd", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Execution failed" in result.stderr
    assert _TURN_FAILURE in result.stderr
    assert _TURN_FAILURE not in result.stdout
    assert "Fallback Error" not in result.output
    # Still alive: the second turn ran and its reply reached stdout.
    assert llm.generate.await_count == 2
    assert "ok" in result.stdout.split("repl-agent", 1)[-1]
    assert not any(
        _TURN_FAILURE in text for text in _assistant_texts(store, _only_seat(store, "repl-agent"))
    )


# ======================================================================================
# Model, tool and user text is printed as text, never read as Rich markup (issue #960)
# ======================================================================================


def _answering_llm(content: str, *tool_calls: ToolCallRequest) -> MagicMock:
    """A conformant connector that answers every call with `content` (and `tool_calls`)."""
    llm = _mock_llm()
    llm.generate = AsyncMock(
        return_value=ModelResponse(
            finish_reason=FinishReason.STOP,
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(provider="mock", input_tokens=1, output_tokens=1, total_tokens=2),
            provenance=Provenance.primary("test"),
        )
    )
    return llm


def _no_binder(llm: object) -> None:
    """No host binding: the Ollama binder would try to embed against localhost."""
    return None


def _configure_agent(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """Build the CLI's clone with `overrides` on its config, as a user's config would."""
    build = clone_builder.build_clone

    def _build(app: clone_builder.AppScope, **kwargs: Any) -> clone_builder.BuiltClone:
        kwargs["config_update"] = {**(kwargs.get("config_update") or {}), **overrides}
        return build(app, **kwargs)

    monkeypatch.setattr(clone_builder, "build_clone", _build)


def _seed_session(session_id: str, *, agent_id: str = "seeded") -> None:
    """Store a one-exchange session under the CLI's (redirected) session root."""
    SessionStore().save(
        SessionState(
            session_id=session_id,
            agent_id=agent_id,
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="system"),
                ChatMessage(role=MessageRole.USER, content="earlier question"),
                ChatMessage(role=MessageRole.ASSISTANT, content="earlier answer"),
            ),
            turn_counter=1,
        )
    )


# A reply a model writes about code. `[i]` and `[b]` are Rich's italic and bold tags, the
# long line is past Rich's 80-column default for a pipe, and the tab is one Rich expands.
_CODE_REPLY = "print(arr[i]) and x[b] done\n" + "word " * 30 + "end\n\tindented :thumbs_up: line"


def test_a_single_shot_reply_reaches_stdout_exactly_as_the_model_wrote_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """stdout is the model's text, byte for byte, and nothing else.

    Before #960 the reply went through Rich markup: `print(arr[i]) and x[b] done` reached
    stdout as `print(arr) and x done`. Escaping alone would still have wrapped the long
    line at 80 columns and expanded the tab, so stdout is written plainly.

    Killed by: src/uclone_x/cli/commands/run.py :: sys.stdout.write(f"{reply_text}
    Becomes: console.print(f"{reply_text}
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm(_CODE_REPLY))

    result = runner.invoke(
        main.app, ["run", "code-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{_CODE_REPLY}\n"


def test_a_single_shot_reply_with_an_unbalanced_closing_tag_exits_zero_and_is_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`[/bold]` in a reply is text. It raised `MarkupError` after the turn had succeeded.

    The crash came before `persist_session`, so the run exited 1 and the answered turn
    was lost from the session as well.

    Killed by: src/uclone_x/cli/commands/run.py :: sys.stdout.write(f"{reply_text}
    Becomes: console.print(f"{reply_text}
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("closing [/bold] tag"))

    result = runner.invoke(
        main.app, ["run", "tag-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "closing [/bold] tag\n"
    assert _assistant_texts(store, _only_seat(store, "tag-agent")) == ["closing [/bold] tag"]


def test_an_empty_single_shot_reply_leaves_stdout_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No placeholder where the reply goes: a script would read it as the model's answer.

    Killed by: src/uclone_x/cli/commands/run.py :: err_console.print("[dim](No response content)[/dim]")
    Becomes: console.print("[dim](No response content)[/dim]")
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm(""))

    result = runner.invoke(
        main.app, ["run", "quiet-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "(No response content)" in result.stderr


def test_single_shot_notices_name_the_agent_and_session_verbatim_on_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Agent names and session ids are the user's text; with `--prompt` they go to stderr.

    The session id carries `[i]`, which Rich removed before #960. The agent name no
    longer can: it names that agent's home directory, so `uclone_x.core.agent_home`
    refuses anything outside `[a-z0-9_-]` (see the refusal test below).

    Killed by: src/uclone_x/cli/commands/run.py :: f"[dim]Resumed session [bold]{escape(hydrated.session_id)}[/bold] — "
    Becomes: f"[dim]Resumed session [bold]{hydrated.session_id}[/bold] — "
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    _seed_session(_seat("s[i]", "markup-agent"))

    result = runner.invoke(
        main.app,
        [
            "run",
            "markup-agent",
            "--prompt",
            "hello",
            "--session-id",
            "s[i]",
            "--reset",
            "--cwd",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "ok\n"
    assert f"Resumed session {_seat('s[i]', 'markup-agent')} —" in result.stderr
    assert f"Session '{_seat('s[i]', 'markup-agent')}' reset before start" in result.stderr
    assert "markup-agent:" in result.stderr


def test_a_run_under_a_name_no_directory_can_carry_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`ucx run "n[/i]"` stops, naming the name, instead of starting a nameless agent.

    The agent name is the directory holding that agent's id and memory, so a name the
    filesystem cannot carry has no home to run in. Repairing it silently would put two
    differently-named agents in one directory, reading each other's facts (P6).

    Killed by: src/uclone_x/cli/commands/run.py :: console.print(f"[bold red]Invalid agent name:[/bold red] {escape(str(exc))}")
    Becomes: console.print("[bold red]Invalid agent name[/bold red]")
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())

    result = runner.invoke(main.app, ["run", "n[/i]", "--prompt", "hello", "--cwd", str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert "Invalid agent name:" in result.stdout
    assert "n[/i]" in result.stdout


def test_the_compact_flag_names_the_session_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--compact` names the session it compacted as typed.

    Killed by: src/uclone_x/cli/commands/run.py :: Compacted '{escape(compaction.session_id)}' "
    Becomes: Compacted '{compaction.session_id}' "
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    _seed_session(_seat("c[i]", "compactor"))

    result = runner.invoke(
        main.app,
        ["run", "compactor", "--prompt", "hi", "--session-id", "c[i]", "--compact"]
        + ["--cwd", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert f"Compacted '{_seat('c[i]', 'compactor')}' via" in result.stderr
    assert result.stdout == "ok\n"


def test_the_repl_prints_replies_history_and_identifiers_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every REPL surface that shows the user's or the model's text shows it unaltered.

    Before #960 `/history` did not even show its role labels: `[user #1]` is valid tag
    syntax, and Rich swallowed it along with any bracketed text in the message.

    Killed by: src/uclone_x/cli/commands/run.py :: {escape(reply_text)}"
    Becomes: {reply_text}"
    """
    _isolated_run_env(monkeypatch, tmp_path)
    reply = "r[i] :thumbs_up: [/bold] done"
    _use_connector(monkeypatch, _answering_llm(reply))
    _feed_prompts(
        monkeypatch, "q[i] asks [/bold]", "/history", "/status", "/ontology", "/reset", "/exit"
    )

    result = runner.invoke(
        main.app,
        ["run", "markup-agent", "--session-id", "s[i]", "--model", "m[i]", "--cwd", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "Agent ID: markup-agent | Provider: mock | Model: m[i]" in out
    assert f"markup-agent:\n{reply}\n" in out
    assert "[user #1] q[i] asks [/bold]" in out
    assert f"[assistant #2] {reply}" in out
    assert "Agent ID:          markup-agent" in out
    assert f"Session ID:        {_seat('s[i]', 'markup-agent')}" in out
    assert "Agent Domain: markup-agent" in out
    assert f"Session '{_seat('s[i]', 'markup-agent')}' reset" in out
    assert "turns with markup-agent." in out


def test_the_repl_compact_command_names_the_session_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`/compact` names the session it compacted as typed.

    Killed by: src/uclone_x/cli/commands/run.py :: Compacted '{escape(compaction.session_id)}' via "
    Becomes: Compacted '{compaction.session_id}' via "
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    _seed_session(_seat("c[i]", "compactor"))
    _feed_prompts(monkeypatch, "/compact", "/exit")

    result = runner.invoke(
        main.app, ["run", "compactor", "--session-id", "c[i]", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert f"Compacted '{_seat('c[i]', 'compactor')}' via" in result.stdout


def test_the_repl_tools_table_prints_tool_names_and_descriptions_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tool descriptions come from MCP servers and plugins; brackets in them are text.

    Killed by: src/uclone_x/cli/commands/run.py :: t_table.add_row(escape(t.name), escape(t.description))
    Becomes: t_table.add_row(t.name, t.description)
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    registry = ToolRegistry()
    registry.register(LocalTool(name="peek[i]", description="Reads [b]one[/b] file [/i]"))

    def _registry(*args: Any, **kwargs: Any) -> ToolRegistry:
        return registry

    monkeypatch.setattr(run, "create_default_registry", _registry)
    _feed_prompts(monkeypatch, "/tools", "/exit")

    result = runner.invoke(main.app, ["run", "tooler", "--cwd", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "peek[i]" in result.stdout
    assert "Reads [b]one[/b] file [/i]" in result.stdout


def _collide_session_id(asked: str, stored: str) -> None:
    """Leave a record at `asked`'s path that names `stored`, as a case-folding filesystem does."""
    _seed_session(stored)
    storage = default_session_storage_dir()
    (storage / f"{stored}.json").rename(storage / f"{asked}.json.tmp-collide")
    (storage / f"{asked}.json.tmp-collide").rename(storage / f"{asked}.json")


@pytest.mark.parametrize(
    ("case", "extra_args", "exit_code", "shown"),
    [
        ("traversal", ["--session-id", "../x[i]"], 2, "Invalid --session-id: Invalid session ID"),
        (
            "collision",
            ["--session-id", "k[i]"],
            2,
            "Unusable --session-id 'sess_room__k[i]__refused': a session",
        ),
        ("provider", ["--provider", "bad[i]"], 1, "LLM Error: Unsupported LLM provider: bad[i]"),
    ],
)
def test_startup_refusals_print_the_users_text_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    extra_args: list[str],
    exit_code: int,
    shown: str,
) -> None:
    """A refusal that quotes the rejected value must quote it as typed.

    `Invalid --session-id` / `Unusable --session-id` / `LLM Error` print an exception
    message that contains the user's argument, and Rich removed `[i]` from it.

    Killed by: src/uclone_x/cli/commands/run.py :: console.print(f"[bold red]LLM Error:[/bold red] {escape(str(exc))}")
    Becomes: console.print(f"[bold red]LLM Error:[/bold red] {exc}")
    """
    _isolated_run_env(monkeypatch, tmp_path)
    if case != "provider":
        _use_connector(monkeypatch, _mock_llm())
    if case == "collision":
        _collide_session_id(_seat("k[i]", "refused"), _seat("K[i]", "refused"))

    result = runner.invoke(
        main.app, ["run", "refused", "--prompt", "hi", *extra_args, "--cwd", str(tmp_path)]
    )

    assert result.exit_code == exit_code, result.output
    assert shown in result.output
    if case == "traversal":
        assert "'../x[i]'" in result.output
    if case == "collision":
        # Rich wraps at 80 columns: at a space for the prose, mid-word for a long path.
        words = " ".join(result.output.split())
        assert "named 'sess_room__K[i]__refused' already occupies" in words
        assert "identifies session 'sess_room__K[i]__refused'" in words
        # Twice: in the headline and in the exception text under it.
        assert result.output.replace("\n", "").count("sess_room__k[i]__refused.json") == 2


# ======================================================================================
# A turn that answered but was not saved is not reported as success (issue #957)
# ======================================================================================

_SAVE_FAILURE = "disk full [/bold] 957"


def _break_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every session write fails, the way a full or read-only disk does."""

    def _refuse(self: SessionStore, *args: Any, **kwargs: Any) -> SessionState:
        raise OSError(_SAVE_FAILURE)

    monkeypatch.setattr(SessionStore, "save", _refuse)


def test_a_single_shot_turn_that_was_not_saved_exits_3_with_the_reply_on_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reply stands, the save error is on stderr, and the status says the turn is lost.

    Before #957 this exited 0 with only a log warning, and the next `--session-id` call
    resumed a conversation that silently lacked the turn. The follow-up call below is
    that consequence, observed: the status is the only warning a script gets.

    Killed by: src/uclone_x/cli/commands/run.py :: if save_failure is not None:
    Becomes: if False:
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    args = ["run", "saver", "--prompt", "first", "--session-id", "kept", "--cwd", str(tmp_path)]

    with monkeypatch.context() as broken:
        _break_the_store(broken)
        lost = runner.invoke(main.app, args)

    assert lost.exit_code == run.UNSAVED_SESSION_EXIT_CODE == 3, lost.output
    assert lost.stdout == "ok\n"
    assert "Session not saved" in lost.stderr
    assert f"OSError: {_SAVE_FAILURE}" in lost.stderr
    assert store.load(_seat("kept", "saver")) is None

    resumed = runner.invoke(main.app, [*args[:3], "second", *args[4:]])

    assert resumed.exit_code == 0, resumed.output
    state = store.load(_seat("kept", "saver"))
    assert state is not None
    assert [m.content for m in state.messages if m.role == MessageRole.USER] == ["second"]


def test_a_failed_turn_that_was_also_not_saved_exits_1_and_reports_both(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No reply is the larger fact, so the status is the failed turn's; stderr has both.

    Killed by: src/uclone_x/cli/commands/run.py :: _report_unsaved_session(save_failure)
    Becomes: pass
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _break_the_turn(monkeypatch, "result")
    _break_the_store(monkeypatch)

    result = runner.invoke(main.app, ["run", "double", "--prompt", "hello", "--cwd", str(tmp_path)])

    assert result.exit_code == run.FAILED_TURN_EXIT_CODE, result.output
    assert result.stdout == ""
    assert "Execution failed" in result.stderr
    assert "Session not saved" in result.stderr


def test_a_repl_whose_session_was_not_saved_on_exit_exits_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The per-turn save warns and moves on; the save on the way out decides the status.

    The per-turn save stays exit-neutral because the exit save retries it
    (`test_agent_session_multitenancy.py` pins a per-turn failure the exit save recovers,
    exiting 0). When the exit save fails too, the session is lost and the status says so.
    Both messages carry the store's error with a markup tag in it, printed verbatim.

    Killed by: src/uclone_x/cli/commands/run.py :: if session_unsaved:
    Becomes: if False:
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    _break_the_store(monkeypatch)
    _feed_prompts(monkeypatch, "hello", "/exit")

    result = runner.invoke(main.app, ["run", "repl-saver", "--cwd", str(tmp_path)])

    assert result.exit_code == run.UNSAVED_SESSION_EXIT_CODE, result.output
    assert f"Turn completed but the session was not saved: {_SAVE_FAILURE}" in result.stderr
    assert f"Session not saved on exit: OSError: {_SAVE_FAILURE}" in result.stderr
    assert "ok" in result.stdout


# ======================================================================================
# Failures that carry content, and how each failure is labelled (#957, from PR #955 review)
# ======================================================================================

_HOOK_REASON = "PII [/bold] detected in prompt"


class _BlockingTurnHook(BaseHook):
    """A `PRE_TURN` hook that refuses every turn with a reason of its own."""

    async def on_pre_turn(self, context: HookContext) -> HookDecision:
        return HookDecision(action=HookAction.BLOCK, reason=_HOOK_REASON)


def test_a_hook_blocked_single_shot_turn_exits_1_naming_the_hook_with_nothing_on_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A blocked turn returns `error` *and* content (`Turn execution blocked by hook: ...`).

    Treating a result with content as a success printed that content as the reply and
    exited 0 -- a mutation PR #955's review showed no test caught. The stderr label names
    the hook: `Execution failed: PII detected in prompt` did not say who refused. The
    persisted session holds only the system message, because the block happens before the
    prompt is appended; `docs/cli-specification.md` §4 says so.

    Killed by: src/uclone_x/agent/turn_executor.py :: stop_reason = "blocked_by_hook"
    Becomes: stop_reason = "not_started"
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    llm = _mock_llm()
    _use_connector(monkeypatch, llm)
    _configure_agent(monkeypatch, hooks=(_BlockingTurnHook(),))

    result = runner.invoke(
        main.app, ["run", "guarded", "--prompt", "my ssn", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == run.FAILED_TURN_EXIT_CODE, result.output
    assert result.stdout == ""
    assert f"✖ Turn blocked by hook: {_HOOK_REASON}" in result.stderr
    assert "Execution failed" not in result.stderr
    assert llm.generate.await_count == 0
    state = store.load(_only_seat(store, "guarded"))
    assert state is not None
    assert [m.role for m in state.messages] == [MessageRole.SYSTEM]


def test_a_step_budget_refusal_with_partial_content_exits_1_with_nothing_on_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The step budget returns `error` with the partial reply as content; that is no answer.

    Killed by: src/uclone_x/cli/commands/run.py :: failure = _turn_failure(result)
    Becomes: failure = _turn_failure(result) if not result.content else None
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    call = ToolCallRequest(id="c1", name="no_such_tool", arguments={})
    _use_connector(monkeypatch, _answering_llm("partial work so far", call))
    _configure_agent(monkeypatch, max_steps=1)

    result = runner.invoke(
        main.app, ["run", "budgeted", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == run.FAILED_TURN_EXIT_CODE, result.output
    assert result.stdout == ""
    assert "✖ Execution failed: Agent step budget exceeded" in result.stderr
    assert "blocked by hook" not in result.stderr
    # The prompt and the partial work are persisted; no final reply is.
    state = store.load(_only_seat(store, "budgeted"))
    assert state is not None
    assert [m.content for m in state.messages if m.role == MessageRole.USER] == ["hello"]


def test_a_hook_blocked_repl_turn_is_labelled_and_not_shown_as_a_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REPL labels a hook block the same way, and never shows it as the reply.

    Killed by: src/uclone_x/cli/commands/run.py :: failure_in_turn = _turn_failure(result)
    Becomes: failure_in_turn = _turn_failure(result) if not result.content else None
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    _configure_agent(monkeypatch, hooks=(_BlockingTurnHook(),))
    _feed_prompts(monkeypatch, "my ssn", "/exit")

    result = runner.invoke(main.app, ["run", "guarded-repl", "--cwd", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f"✖ Turn blocked by hook: {_HOOK_REASON}" in result.stderr
    assert "blocked by hook" not in result.stdout


def test_a_failure_message_with_markup_in_it_is_printed_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An exception message is not markup: `[/bold]` in it raised inside the error path.

    Nor is `:warning:` an emoji code there: stderr's console substitutes none.

    Killed by: src/uclone_x/cli/commands/run.py :: ✖ Execution failed:[/bold red] {escape(failure)}
    Becomes: ✖ Execution failed:[/bold red] {failure}
    """
    _isolated_run_env(monkeypatch, tmp_path)
    llm = _mock_llm()
    llm.generate = AsyncMock(side_effect=RuntimeError("bad [/bold] tag [red :warning:"))
    _use_connector(monkeypatch, llm)

    result = runner.invoke(main.app, ["run", "marked", "--prompt", "hello", "--cwd", str(tmp_path)])

    assert result.exit_code == run.FAILED_TURN_EXIT_CODE, result.output
    assert "✖ Execution failed: bad [/bold] tag [red :warning:" in result.stderr


# ======================================================================================
# Follow-ups from PR #962's review (issue #970)
# ======================================================================================


def _return_turn(monkeypatch: pytest.MonkeyPatch, turn: TurnResult) -> None:
    """Make `execute_turn` return `turn` as given, without running the engine."""

    async def _returned(self: Any, content: str, **_kwargs: Any) -> TurnResult:
        return turn

    _use_connector(monkeypatch, _mock_llm())
    monkeypatch.setattr("uclone_x.agent.base.BaseAgent.execute_turn", _returned)


def test_a_hook_block_is_recognised_by_its_stop_reason_not_by_the_wording_of_its_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The label follows `TurnResult.stop_reason`; the text of `content` decides nothing.

    Before #970 the CLI recognised a hook block by comparing `content` with the engine's
    `Turn execution blocked by hook: <reason>` sentence. Rewording that sentence in
    `agent/base.py` would relabel every hook block as `Execution failed`, and a failure
    whose content happened to read like one was labelled a hook block. Both directions run.

    Killed by: src/uclone_x/cli/commands/run.py :: return result.stop_reason == "blocked_by_hook"
    Becomes: return result.error is not None and result.content == f"Turn execution blocked by hook: {result.error}"
    """
    _isolated_run_env(monkeypatch, tmp_path)
    args = ["run", "labelled", "--prompt", "hello", "--cwd", str(tmp_path)]

    with monkeypatch.context() as blocked:
        _return_turn(
            blocked,
            TurnResult(
                turn_index=1,
                content="",
                error="policy 970",
                stop_reason="blocked_by_hook",
                provenance=None,
            ),
        )
        by_reason = runner.invoke(main.app, args)

    assert by_reason.exit_code == run.FAILED_TURN_EXIT_CODE, by_reason.output
    assert "✖ Turn blocked by hook: policy 970" in by_reason.stderr
    assert "Execution failed" not in by_reason.stderr

    with monkeypatch.context() as look_alike:
        _return_turn(
            look_alike,
            TurnResult(
                turn_index=1,
                content="Turn execution blocked by hook: policy 970",
                error="policy 970",
                stop_reason="model_stopped",
                provenance=None,
            ),
        )
        by_wording = runner.invoke(main.app, args)

    assert by_wording.exit_code == run.FAILED_TURN_EXIT_CODE, by_wording.output
    assert "✖ Execution failed: policy 970" in by_wording.stderr
    assert "Turn blocked by hook" not in by_wording.stderr


# The bytes a well-meant tidy-up would take away: trailing spaces and a tab at the end, a
# CRLF line ending, and the C0 controls Rich used to strip (CR, BEL, BS) beside ESC.
_RAW_REPLY = "trailing spaces  \r\nred \x1b[31mX\x1b[0m bell\a back\bspace\t  "


def test_a_single_shot_reply_keeps_trailing_whitespace_crlf_and_control_characters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """stdout is the reply byte for byte, including the bytes a cleanup would remove.

    `test_a_single_shot_reply_reaches_stdout_exactly_as_the_model_wrote_it` pins what Rich
    did to a reply. PR #962's review applied both mutations below and no test failed.
    Compared as bytes, because `Result.stdout` itself rewrites CRLF to LF.

    Killed by: src/uclone_x/cli/commands/run.py :: sys.stdout.write(f"{reply_text}
    Becomes: sys.stdout.write(f"{reply_text.rstrip()}
    Killed by: src/uclone_x/cli/commands/run.py :: sys.stdout.write(f"{reply_text}
    Becomes: sys.stdout.write(f"{reply_text.replace(chr(13), '')}
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm(_RAW_REPLY))

    result = runner.invoke(
        main.app, ["run", "raw-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == f"{_RAW_REPLY}\n".encode()


def test_a_whitespace_only_single_shot_reply_is_written_not_replaced_by_the_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only an empty reply is "no response". Spaces and a newline are a reply, and are written.

    Killed by: src/uclone_x/cli/commands/run.py :: if reply_text:
    Becomes: if reply_text.strip():
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("  \n "))

    result = runner.invoke(
        main.app, ["run", "blank-agent", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == b"  \n \n"
    assert "(No response content)" not in result.stderr


def test_a_reply_written_into_a_closed_pipe_is_reported_and_the_turn_is_still_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A reader that went away takes the reply with it, but not the turn.

    Before #970 the reply was written before the session was saved, so `BrokenPipeError`
    escaped as a traceback and the answered turn was never saved. This runs a real process
    because a closed pipe belongs to file descriptor 1: the read end is closed before the
    child starts, so the first flush fails with `EPIPE` however slowly the child imports.
    The status (1) says the reply is lost, `--session-id` still resumes the turn, and
    nothing fails again at interpreter shutdown, when the reply that is still buffered
    would be flushed a second time.

    Killed by: src/uclone_x/cli/commands/run.py :: except (OSError, UnicodeError) as write_err:
    Becomes: except ZeroDivisionError as write_err:
    Killed by: src/uclone_x/cli/commands/run.py :: os.dup2(devnull, descriptor)
    Becomes: pass
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        child = _run_single_shot_in_a_real_process(monkeypatch, tmp_path, "piped", write_end)
    finally:
        os.close(write_end)

    stderr = child.stderr.decode()
    assert child.returncode == run.UNWRITTEN_REPLY_EXIT_CODE, stderr
    assert "✖ Reply not written to stdout: BrokenPipeError" in stderr
    assert "Traceback" not in stderr
    assert "Exception ignored" not in stderr
    replies = _assistant_texts(store, _only_seat(store, "piped"))
    assert len(replies) == 1 and "hello" in replies[0], replies


# An OSC 52 clipboard write, a BEL to terminate it, and an SGR colour change. Every byte of
# it is an instruction to a terminal rather than text for a reader (#1282).
_OSC_52_PAYLOAD = "\x1b]52;c;aGVsbG8=\x07before\x1b[31mred\x1b[0m"
#: The same payload after neutralisation: every control character visible, nothing obeyed.
_OSC_52_NEUTRALISED = "^[]52;c;aGVsbG8=^Gbefore^[[31mred^[[0m"


def _single_shot_reply_on_a_pty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent: str, prompt: str
) -> bytes:
    """Return the bytes `ucx run --prompt` puts on a **real terminal**, via `pty.openpty`.

    A real pty and not a monkeypatched `isatty` on purpose. Faking `isatty` proves a branch
    exists; it does not prove that the bytes a terminal receives changed, and #1282 is
    entirely about the bytes a terminal receives.

    The slave is put in raw mode so the line discipline's own `ONLCR` does not rewrite `\\n`
    as `\\r\\n` and get mistaken for something the CLI wrote. stderr stays on a pipe, so what
    comes back is the reply and nothing else.

    The master is drained on a thread *while* the child runs, rather than afterwards: closing
    the last slave descriptor makes the master fail with `EIO` and takes anything still
    buffered with it, so a read that waits for the child to exit returns nothing at all.
    """
    master, slave = pty.openpty()
    tty.setraw(slave)
    collected: list[bytes] = []

    def _drain() -> None:
        try:
            while chunk := os.read(master, 65536):
                collected.append(chunk)
        except OSError:  # `EIO`: the last slave was closed, so there is no more output
            pass

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    try:
        child = _run_single_shot_in_a_real_process(
            monkeypatch, tmp_path, agent, slave, prompt=prompt
        )
    finally:
        os.close(slave)
    reader.join(timeout=30)
    os.close(master)
    assert child.returncode == 0, child.stderr.decode()
    return b"".join(collected)


def test_a_single_shot_reply_is_neutralised_on_a_real_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A control sequence in a reply is displayed, not obeyed, when stdout is a terminal.

    The whole path, not a function in isolation: the mock provider's reply interpolates the
    last message (`Agent processed message: '<it>'.`), so `--prompt` carries the payload into
    the model's reply, and a real `ucx run` process writes that reply onto a real pty. Before
    #1282 the OSC 52 came back off the pty byte-identical -- a model's output deciding what
    the user's terminal does, which for OSC 52 is writing their clipboard.

    Killed by: src/uclone_x/cli/commands/run.py :: if _stdout_is_a_terminal():
    Becomes: if False:
    """
    _isolated_run_env(monkeypatch, tmp_path)

    written = _single_shot_reply_on_a_pty(monkeypatch, tmp_path, "tty-agent", _OSC_52_PAYLOAD)

    assert _OSC_52_NEUTRALISED.encode() in written, written
    # The point of the exercise: not one byte the terminal would act on is left.
    assert b"\x1b" not in written, written
    assert b"\x07" not in written, written


def test_a_single_shot_reply_through_a_pipe_keeps_the_bytes_the_terminal_would_not_get(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same reply, the same process, a pipe instead of a pty: byte-identical.

    The mirror of `test_a_single_shot_reply_is_neutralised_on_a_real_terminal`, differing
    from it only in what file descriptor 1 is. A terminal is a renderer and a pipe is a byte
    channel, so a caller redirecting `ucx run` into a file still gets exactly what the model
    wrote; neutralising there would be data corruption, not safety. Pinning both directions
    is the point -- a guard that only refuses is one-sided, and the guarantee this protects
    is the one #970 item 2 built.

    Killed by: src/uclone_x/cli/commands/run.py :: if _stdout_is_a_terminal():
    Becomes: if True:
    """
    _isolated_run_env(monkeypatch, tmp_path)
    piped = tmp_path / "reply-through-a-pipe"

    with piped.open("wb") as sink:
        child = _run_single_shot_in_a_real_process(
            monkeypatch, tmp_path, "pipe-agent", sink.fileno(), prompt=_OSC_52_PAYLOAD
        )

    assert child.returncode == 0, child.stderr.decode()
    written = piped.read_bytes()
    assert _OSC_52_PAYLOAD.encode() in written, written
    assert _OSC_52_NEUTRALISED.encode() not in written, written


def test_the_repl_neutralises_a_reply_rather_than_relying_on_rich(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REPL's protection is its own, declared, and would survive losing Rich's highlighter.

    Until #1282 this site was protected by accident: `console.print(escape(...))` let ESC
    through and Rich's markup escaping and highlighter merely broke the sequence up, so the
    same payload arrived as `\\x1b\\x1b[1m]\\x1b[0m\\x1b[1;36m52\\x1b[0m;c;…`. Real protection,
    promised by nothing -- `highlight=False` or a move off `console.print` would have removed
    it with no test failing. That is the shape of defect that outlives the review that would
    have caught it, so this asserts the caret form rather than the absence of the payload:
    only the neutraliser produces it, and Rich's mangling never would.

    Killed by: src/uclone_x/cli/commands/run.py :: reply_text = _neutralise_control_sequences(reply_text)
    Becomes: pass
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm(_OSC_52_PAYLOAD))
    _feed_prompts(monkeypatch, "go", "/exit")

    result = runner.invoke(main.app, ["run", "repl-ctl", "--cwd", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert _OSC_52_NEUTRALISED in result.stdout, result.stdout
    assert "\x1b]52" not in result.stdout, result.stdout


def test_the_repl_neutralises_before_escaping_so_rich_cannot_eat_the_caret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Neutralising introduces a `[`, so it must happen before `escape()`, not after.

    `\\x1b` becomes `^[`, and that `[` was never seen by `escape()`. Run in the other order it
    reaches Rich as the start of a markup tag whenever what follows the ESC parses as a tag
    name -- here `\\x1bbold]X` renders as `^X`, Rich having swallowed the caret marker *and*
    the text after it, which is worse than doing nothing. The OSC 52 payload alone cannot
    catch this: `\\x1b]52;…` comes out the same in both orders, because `]` is not a tag name.

    Killed by: src/uclone_x/cli/commands/run.py :: reply_text = _neutralise_control_sequences(reply_text)
    Becomes: pass
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("\x1bbold]X"))
    _feed_prompts(monkeypatch, "go", "/exit")

    result = runner.invoke(main.app, ["run", "repl-order", "--cwd", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "^[bold]X" in result.stdout, result.stdout


@pytest.mark.parametrize(
    ("raw", "rendered"),
    [
        ("\x1b", "^["),  # ESC, the one that starts every sequence
        ("\x07", "^G"),  # BEL
        ("\r", "^M"),  # CR, which rewrites the line already on screen
        ("\x00", "^@"),
        ("\x7f", "^?"),  # DEL
        ("\x9b", "^[["),  # C1 CSI, shown as the 7-bit sequence it is equivalent to
        ("\n", "\n"),  # kept: a reply's layout, not a weapon
        ("\t", "\t"),  # kept, for the same reason
        ("plain text", "plain text"),
    ],
)
def test_the_neutraliser_renders_controls_as_carets_and_keeps_newline_and_tab(
    raw: str, rendered: str
) -> None:
    """C0 except `\\n`/`\\t`, plus DEL and C1, become visible carets; everything else is untouched.

    Neutralising rather than deleting is the deliberate choice: a reader can still see that
    something was there. Deleting would hide a manipulation attempt about as well as
    executing it does.

    The anchor below narrows the C0 range past ESC (0x1B) instead of naming the kept-character
    string, because a declaration is matched against the target's raw source text and a
    backslash written in this docstring would not survive the trip.

    Killed by: src/uclone_x/cli/commands/run.py :: for c in range(0x00, 0x20) if chr(c) not in
    Becomes: for c in range(0x1C, 0x20) if chr(c) not in
    """
    assert run._neutralise_control_sequences(raw) == rendered  # pyright: ignore[reportPrivateUsage]


class _ClosedStdout:
    """A stdout that raises the way a closed stream does."""

    def isatty(self) -> bool:
        raise ValueError("I/O operation on closed file")


class _StdoutWithoutIsatty:
    """A stdout substitute that does not implement `isatty` at all.

    Not hypothetical: `_InterruptedStdout`, the double two tests below already use, is one.
    """


@pytest.mark.parametrize("stdout", [_ClosedStdout(), _StdoutWithoutIsatty()])
def test_stdout_that_cannot_say_whether_it_is_a_terminal_keeps_the_bytes(
    monkeypatch: pytest.MonkeyPatch, stdout: object
) -> None:
    """A stdout that cannot answer `isatty()` is treated as a byte channel, not as a terminal.

    It raises `ValueError` on a closed stream and `AttributeError` when the object does not
    implement it, and this is asked on the paths that exist because stdout can be broken
    (#996) -- so the question has to be total. Answering `False` is safe in both directions:
    the reply keeps its bytes, and a reply that was going to be written does not turn into an
    unhandled exception on the way out.

    Killed by: src/uclone_x/cli/commands/run.py :: except (OSError, ValueError, AttributeError):  # `io.UnsupportedOperation` subclasses two
    Becomes: except NotImplementedError:
    """
    monkeypatch.setattr(run.sys, "stdout", stdout)

    assert run._stdout_is_a_terminal() is False  # pyright: ignore[reportPrivateUsage]


def _run_single_shot_in_a_real_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent: str,
    stdout: int,
    prompt: str = "hello",
) -> subprocess.CompletedProcess[bytes]:
    """Run `ucx run <agent> --provider mock --prompt <prompt>` as a child with `stdout` as fd 1.

    A stdout that fails at the descriptor, and a shutdown flush that fails after it, exist
    only in a real process: under `CliRunner` stdout has no descriptor at all.
    """
    for var in (
        "LLM_PROVIDER",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    # The child must import this checkout's `src`, not the shared venv's editable install.
    monkeypatch.setenv("PYTHONPATH", str(Path(run.__file__).resolve().parents[3]))
    command = [
        sys.executable,
        "-c",
        "from uclone_x.shells.entry import main; main()",
        *["run", agent, "--provider", "mock", "--prompt", prompt, "--cwd", str(tmp_path)],
    ]
    return subprocess.run(command, stdout=stdout, stderr=subprocess.PIPE, timeout=120, check=False)


def test_a_reply_written_to_a_read_only_stdout_exits_1_without_failing_again_at_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Any write error on stdout, not only a closed pipe, leaves nothing to flush at shutdown.

    PR #994's review ran `ucx run --prompt` with stdout on a descriptor opened for reading
    (`1<file`): the write failed with `EBADF` and was reported, but fd 1 was redirected only
    after `BrokenPipeError`. The reply stayed buffered, the interpreter's shutdown flush
    failed a second time, and the process printed `Exception ignored ... OSError` and exited
    120 instead of the documented 1 (#996).

    Killed by: src/uclone_x/cli/commands/run.py :: if isinstance(write_err, OSError):
    Becomes: if isinstance(write_err, BrokenPipeError):
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    readable = tmp_path / "stdout-opened-for-reading"
    readable.touch()
    descriptor = os.open(readable, os.O_RDONLY)
    try:
        child = _run_single_shot_in_a_real_process(monkeypatch, tmp_path, "readonly", descriptor)
    finally:
        os.close(descriptor)

    stderr = child.stderr.decode()
    assert child.returncode == run.UNWRITTEN_REPLY_EXIT_CODE == 1, stderr
    assert "✖ Reply not written to stdout: OSError" in stderr
    assert "The turn was saved; --session-id resumes it." in stderr
    assert "Exception ignored" not in stderr
    assert "Traceback" not in stderr
    replies = _assistant_texts(store, _only_seat(store, "readonly"))
    assert len(replies) == 1 and "hello" in replies[0], replies


def test_a_reply_the_stdout_encoding_cannot_represent_is_reported_and_the_turn_is_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`UnicodeEncodeError` from the write is the other way a finished turn lost its reply.

    PR #962's review measured it with `PYTHONIOENCODING=ascii`: exit 1, a traceback, and no
    session file. An ASCII `CliRunner` gives stdout the same strict encoding (and stderr
    `backslashreplace`, as a real process has).

    Killed by: src/uclone_x/cli/commands/run.py :: except (OSError, UnicodeError) as write_err:
    Becomes: except OSError as write_err:
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("한글 reply 970"))

    result = CliRunner(charset="ascii").invoke(
        main.app,
        ["run", "hangul", "--prompt", "hello", "--session-id", "kept", "--cwd", str(tmp_path)],
    )

    assert result.exit_code == run.UNWRITTEN_REPLY_EXIT_CODE, result.output
    assert result.stdout_bytes == b""
    assert "Reply not written to stdout: UnicodeEncodeError" in result.stderr
    assert "--session-id resumes it" in result.stderr
    assert _assistant_texts(store, _seat("kept", "hangul")) == ["한글 reply 970"]


def test_a_reply_neither_saved_nor_written_exits_1_and_does_not_claim_the_turn_was_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both losses at once: stderr names both, and neither report says the turn was kept.

    `docs/cli-specification.md` §4: if the save failed too, both are reported and the status
    is still 1, since no reply reached the script. The note under the unwritten reply
    ("The turn was saved; --session-id resumes it.") is true only when the save succeeded;
    beside a failed save it would send a script to resume a turn that does not exist.

    Killed by: src/uclone_x/cli/commands/run.py :: saved=save_failure is None)
    Becomes: saved=True)
    Killed by: src/uclone_x/cli/commands/run.py :: raise typer.Exit(code=UNWRITTEN_REPLY_EXIT_CODE)
    Becomes: pass
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("한글 reply 996"))
    _break_the_store(monkeypatch)

    result = CliRunner(charset="ascii").invoke(
        main.app,
        ["run", "hangul", "--prompt", "hello", "--session-id", "lost", "--cwd", str(tmp_path)],
    )

    assert result.exit_code == run.UNWRITTEN_REPLY_EXIT_CODE == 1, result.output
    assert result.stdout_bytes == b""
    assert f"Session not saved: OSError: {_SAVE_FAILURE}" in result.stderr
    assert "--session-id will not resume it" in result.stderr
    assert "Reply not written to stdout: UnicodeEncodeError" in result.stderr
    assert "The turn was saved" not in result.stderr
    assert "--session-id resumes it" not in result.stderr
    assert store.load("lost") is None


class _InterruptedStdout:
    """A stdout whose write raises `KeyboardInterrupt`.

    What a *second* Ctrl+C can do to a write into a slow pipe. A first one only cancels the
    task at its next `await`: the interrupted `write` is retried (PEP 475) and completes.
    """

    def write(self, text: str) -> int:
        raise KeyboardInterrupt

    def flush(self) -> None:
        return None


def test_an_interrupt_while_the_reply_is_written_leaves_the_turn_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The session is saved before the reply is written, so even a write that cannot be
    caught and continued from does not lose the turn.

    The `except` around the write covers the failures that can be named. An interrupt is
    not one of them, and only the order protects the turn from it. The order cannot be
    written as a one-line `Killed by:` declaration. It was checked with `./swx mutate`
    instead, by moving the save block below the reply block (#970's PR records the run).
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    monkeypatch.setattr(run, "sys", SimpleNamespace(stdout=_InterruptedStdout()))

    result = runner.invoke(
        main.app, ["run", "irq-write", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 130, result.output
    assert _assistant_texts(store, _only_seat(store, "irq-write")) == ["ok"]


def _interrupt_saves_after(monkeypatch: pytest.MonkeyPatch, completed: int) -> None:
    """Let `completed` session writes finish, then interrupt the next, as a second Ctrl+C can.

    Under `asyncio.run` a first Ctrl+C only cancels the task at its next `await`, so a
    synchronous save finishes. A second one raises `KeyboardInterrupt` wherever the process
    is, the save included (measured on Python 3.12 with two `SIGINT`s 0.2 s apart).
    """
    real_save = SessionStore.save
    calls = 0

    def _save(self: SessionStore, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > completed:
            raise KeyboardInterrupt
        return real_save(self, *args, **kwargs)

    monkeypatch.setattr(SessionStore, "save", _save)


def test_an_interrupt_during_the_repls_exit_save_says_the_save_did_not_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exit 130 alone said "interrupted", not that the save on the way out never finished.

    Killed by: src/uclone_x/cli/commands/run.py :: _report_interrupted_save("Session not saved on exit")
    Becomes: pass
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    _interrupt_saves_after(monkeypatch, completed=1)
    _feed_prompts(monkeypatch, "hello", "/exit")

    result = runner.invoke(main.app, ["run", "repl-irq", "--cwd", str(tmp_path)])

    assert result.exit_code == 130, result.output
    assert "✖ Session not saved on exit: interrupted before the save completed" in result.stderr
    assert "Interrupted by user." in result.stderr
    # The per-turn save had finished, and saves are atomic, so that copy stands.
    assert _assistant_texts(store, _only_seat(store, "repl-irq")) == ["ok"]


def test_an_interrupt_during_the_single_shot_save_says_the_save_did_not_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same report for `--prompt`. The save comes first now, so no reply was written.

    Killed by: src/uclone_x/cli/commands/run.py :: _report_interrupted_save("Session not saved")
    Becomes: pass
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    _interrupt_saves_after(monkeypatch, completed=0)

    result = runner.invoke(
        main.app, ["run", "irq-save", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 130, result.output
    assert result.stdout == ""
    assert "✖ Session not saved: interrupted before the save completed" in result.stderr
    assert not [s for s in store.list_session_ids() if s.endswith("__irq-save")]


def test_an_interrupt_during_the_save_of_a_failed_single_shot_turn_still_reports_the_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed turn is reported before its save, so an interrupt there cannot swallow it.

    PR #994 moved the save ahead of the reply and, with it, ahead of the failure report. A
    second Ctrl+C inside that save then printed the interrupted-save line and
    `Interrupted by user.`, and never `✖ Execution failed: <error>` (#996): the one line
    saying why the turn produced nothing was lost, and exit 130 read as a plain interrupt.

    Killed by: src/uclone_x/cli/commands/run.py :: _report_failed_turn(failure)
    Becomes: pass
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _break_the_turn(monkeypatch, "result")
    _interrupt_saves_after(monkeypatch, completed=0)

    result = runner.invoke(
        main.app, ["run", "irq-failed", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 130, result.output
    assert result.stdout == ""
    assert f"✖ Execution failed: {_TURN_FAILURE}" in result.stderr
    assert "✖ Session not saved: interrupted before the save completed" in result.stderr
    assert "Interrupted by user." in result.stderr


def test_an_interrupt_during_the_save_of_a_hook_blocked_single_shot_turn_still_reports_the_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hook-block half of the same promise: that report also comes before the save.

    `docs/cli-specification.md` §4's `130` row promises that `✖ Turn blocked by hook`, like
    `✖ Execution failed`, survives an interrupt inside the save. The test above pins only
    the failed-turn branch. PR #1002's review moved the hook-block report alone back below
    the save and every test passed (#1005). A second Ctrl+C there left `Interrupted by
    user.` and nothing saying that a hook, not the user, had stopped the turn.

    Killed by: src/uclone_x/cli/commands/run.py :: _report_blocked_turn(failure)
    Becomes: agent.persist_session(); _report_blocked_turn(failure)
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    _configure_agent(monkeypatch, hooks=(_BlockingTurnHook(),))
    _interrupt_saves_after(monkeypatch, completed=0)

    result = runner.invoke(
        main.app, ["run", "irq-guarded", "--prompt", "my ssn", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 130, result.output
    assert result.stdout == ""
    assert f"✖ Turn blocked by hook: {_HOOK_REASON}" in result.stderr
    assert "Execution failed" not in result.stderr
    assert "✖ Session not saved: interrupted before the save completed" in result.stderr
    assert "Interrupted by user." in result.stderr


def test_a_reply_that_overfills_a_non_blocking_pipe_leaves_its_start_on_stdout_as_documented(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A write that fails part-way has already delivered a prefix, and §4 says so (#1005).

    §4's stdout column for an unwritten reply said "No reply". A reply longer than an empty
    non-blocking pipe that nobody reads goes in part-way and then fails with
    `BlockingIOError` (EAGAIN). Those bytes are the reader's; nothing in the process can take
    them back. So the specification is what changed: the row now says a prefix can remain,
    and this test holds the row to the bytes a real process leaves. The mock connector
    echoes the prompt, which is sized from this pipe's measured capacity (#1008): a fixed
    96 KiB fits whole in the 256 KiB or 1 MiB pipe of a 16 or 64 KiB-page Linux kernel.

    Killed by: docs/cli-specification.md :: can leave the start of the reply on stdout
    Becomes: leaves nothing on stdout
    Killed by: tests/support/nonblocking_stdout.py :: OVERFILL_MARGIN = 32 * 1024
    Becomes: OVERFILL_MARGIN = -32 * 1024
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    read_end, write_end = nonblocking_stdout.nonblocking_pipe()
    try:
        try:
            capacity = nonblocking_stdout.capacity_of(read_end, write_end)
            prompt = "x" * (capacity + nonblocking_stdout.OVERFILL_MARGIN)
            child = _run_single_shot_in_a_real_process(
                monkeypatch, tmp_path, "overflow", write_end, prompt=prompt
            )
        finally:
            os.close(write_end)
        written = nonblocking_stdout.read_to_eof(read_end, timeout=10)
    finally:
        os.close(read_end)

    stderr = child.stderr.decode()
    assert child.returncode == run.UNWRITTEN_REPLY_EXIT_CODE, stderr
    assert "✖ Reply not written to stdout: BlockingIOError" in stderr
    assert "Exception ignored" not in stderr
    replies = _assistant_texts(store, _only_seat(store, "overflow"))
    assert len(replies) == 1 and prompt in replies[0], [len(r) for r in replies]
    reply = f"{replies[0]}\n".encode()
    assert 0 < len(written) < len(reply), (len(written), len(reply))
    assert reply.startswith(written)
    spec = Path(__file__).resolve().parents[2] / "docs" / "cli-specification.md"
    row = next(
        line for line in spec.read_text().splitlines() if "`UNWRITTEN_REPLY_EXIT_CODE`" in line
    )
    assert "can leave the start of the reply on stdout" in row


def test_reading_the_childs_stdout_to_eof_gives_up_while_a_write_end_is_still_open() -> None:
    """The partial-write test's final read is bounded, so a leaked holder cannot hang it (#1008).

    EOF needs every write end closed. A helper the child left holding fd 1 would keep one
    open, and an unbounded read would then wait for good. The read runs on a thread here so
    that an unbounded one fails this test instead of hanging it: closing the write end
    afterwards releases it with EOF.

    Killed by: tests/support/nonblocking_stdout.py :: max(deadline - time.monotonic(), 0)
    Becomes: None
    """
    read_end, write_end = os.pipe()
    outcome: list[str] = []

    def read() -> None:
        try:
            nonblocking_stdout.read_to_eof(read_end, timeout=0.05)
            outcome.append("EOF")
        except TimeoutError:
            outcome.append("timed out")

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(timeout=5)
    os.close(write_end)
    reader.join(timeout=5)
    if reader.is_alive():
        # Left open: closing a descriptor under a live reader could hand its number to the
        # next test's pipe.
        pytest.fail("the read did not return even after its write end closed")
    os.close(read_end)
    assert outcome == ["timed out"]


def test_measuring_a_pipe_reports_the_room_left_in_it_and_drains_only_its_probe() -> None:
    """The reply is sized from the pipe the child gets, not from an assumed size (#1008).

    The partial-write test cannot tell a measured capacity from an assumed one on a machine
    whose pipe is the assumed size, and every machine the gate runs on has a 64 KiB pipe.
    Room already taken shows the difference on any pipe: after a small write the measurement
    must drop, and once the probe is drained the pipe must hold as many bytes as that write
    (a pipe is first in, first out, so they are the probe's last bytes). On the empty pipe,
    a write of the measured size must go in whole and leave no room. That is exact on a
    page-granular Linux pipe too, since an empty pipe measures in whole pages.

    Killed by: tests/support/nonblocking_stdout.py :: return accepted
    Becomes: return 64 * 1024
    Killed by: tests/support/nonblocking_stdout.py :: return accepted
    Becomes: return accepted // 2
    """
    prefill = b"p" * 1000
    read_fd, write_fd = nonblocking_stdout.nonblocking_pipe()
    try:
        empty = nonblocking_stdout.capacity_of(read_fd, write_fd)
        assert os.write(write_fd, prefill) == len(prefill)
        prefilled = nonblocking_stdout.capacity_of(read_fd, write_fd)
        assert 0 < prefilled < empty, (prefilled, empty)
        os.set_blocking(read_fd, False)
        assert len(os.read(read_fd, 1 << 20)) == len(prefill)
        assert os.write(write_fd, bytes(empty)) == empty
        with pytest.raises(BlockingIOError):
            os.write(write_fd, b"x")
    finally:
        os.close(read_fd)
        os.close(write_fd)


class _AttributeExporter:
    """Exporter that keeps each exported span's name and attributes."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, dict[str, Any]]] = []

    async def export_spans(self, spans: Any) -> None:
        self.spans.extend((s.name, dict(s.attributes)) for s in spans)

    async def export_metrics(self, metrics: Any) -> None:
        return None

    async def shutdown(self) -> None:
        return None


def test_a_run_on_a_persona_s_own_model_says_so_and_traces_that_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The saved-choice notice names the saved model; the run says which one it sends.

    The saved model fills only what a persona leaves empty (#1731), so a persona with its
    own model is served by that one. Without the second line the terminal names a model
    the run never sends, and the turn span records it too (P6).

    Killed by: src/uclone_x/cli/commands/run.py :: effective_model = agent.config.llm_config.model_name or effective_model
    Becomes: effective_model = effective_model
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("done"))
    _configure_agent(monkeypatch, llm_config=AgentLLMConfig(model_name="persona-model"))

    def _saved(provider: str | None, model: str | None) -> tuple[str | None, str | None]:
        return "saved-model", "Using the saved model saved-model."

    monkeypatch.setattr(run, "apply_saved_model", _saved)
    exporter = _AttributeExporter()

    def _exporter(*args: Any, **kwargs: Any) -> _AttributeExporter:
        return exporter

    monkeypatch.setattr(run, "create_telemetry_exporter", _exporter)

    result = runner.invoke(
        main.app, ["run", "own-model", "--prompt", "hello", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "own-model runs on its own model, persona-model, not saved-model." in result.stderr
    turn_models = [attrs.get("model") for _, attrs in exporter.spans if "mode" in attrs]
    assert turn_models == ["persona-model"]


def test_a_run_s_turn_span_carries_the_session_it_ran_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--session-id` names the session the turn is saved to; the span said `sess_<agent>`.

    Killed by: src/uclone_x/cli/commands/run.py :: session_id=turn_session,
    Becomes: session_id=f"sess_{agent_name}",
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("done"))
    exporter = _AttributeExporter()

    def _exporter(*args: Any, **kwargs: Any) -> _AttributeExporter:
        return exporter

    monkeypatch.setattr(run, "create_telemetry_exporter", _exporter)

    result = runner.invoke(
        main.app,
        ["run", "spanned", "--prompt", "hi", "--session-id", "notes", "--cwd", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert _assistant_texts(store, _seat("notes", "spanned")) == ["done"]
    turn_sessions = [attrs.get("session_id") for _, attrs in exporter.spans if "mode" in attrs]
    assert turn_sessions == [_seat("notes", "spanned")]


def test_an_interactive_turn_s_span_carries_the_session_it_ran_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REPL's turn span names the session `--session-id` chose, as `--prompt`'s does (#1775).

    Killed by: src/uclone_x/cli/commands/run.py :: session_id=agent.session_id,
    Becomes: session_id=f"sess_{agent_name}",
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("done"))
    exporter = _AttributeExporter()

    def _exporter(*args: Any, **kwargs: Any) -> _AttributeExporter:
        return exporter

    monkeypatch.setattr(run, "create_telemetry_exporter", _exporter)
    _feed_prompts(monkeypatch, "hi", "/exit")

    result = runner.invoke(
        main.app, ["run", "spanned", "--session-id", "notes", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert _assistant_texts(store, _seat("notes", "spanned")) == ["done"]
    turn_spans = [attrs for _, attrs in exporter.spans if attrs.get("mode") == "interactive"]
    assert [attrs.get("session_id") for attrs in turn_spans] == [_seat("notes", "spanned")]


def test_a_run_is_a_one_seat_room_and_leaves_the_old_session_unresumed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run without `--session-id` starts a room: a person and the clone, stored (§5.9).

    The clone's session is the seat's (`participant_session_id`), as it would be in the
    app, and the session the CLI kept before under `sess_<clone>` is neither resumed nor
    touched (owner ruling 2026-09-27: no migration, no deletion).

    Killed by: src/uclone_x/cli/commands/run.py :: session_id=room.session_id,
    Becomes: session_id=f"sess_{agent_name}",
    """
    from uclone_x.room.models import ParticipantKind
    from uclone_x.room.one_seat import ONE_SEAT_HUMAN_ID
    from uclone_x.room.store import RoomStore

    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("fresh"))
    _seed_session("sess_seated", agent_id="seated")
    before = store.session_path("sess_seated").read_bytes()

    result = runner.invoke(main.app, ["run", "seated", "--prompt", "hi", "--cwd", str(tmp_path)])

    assert result.exit_code == 0, result.output
    rooms = RoomStore().list_room_ids()
    assert len(rooms) == 1, rooms
    room = RoomStore().load(rooms[0])
    assert room is not None
    assert [(p.id, p.kind) for p in room.participants] == [
        (ONE_SEAT_HUMAN_ID, ParticipantKind.HUMAN),
        ("seated", ParticipantKind.AGENT),
    ]
    assert room.head == "run"  # the app will not post into it (#1885)
    assert _only_seat(store, "seated") == _seat(rooms[0], "seated")
    assert _assistant_texts(store, _seat(rooms[0], "seated")) == ["fresh"]
    assert f"--session-id {rooms[0]} resumes it" in result.stderr
    assert store.session_path("sess_seated").read_bytes() == before


def test_session_id_resumes_the_room_it_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--session-id` is the room's id: a second run with it continues the same seat session.

    Killed by: src/uclone_x/room/one_seat.py :: room_id=room_id if room_id is not None else new_one_seat_room_id(),
    Becomes: room_id=new_one_seat_room_id(),
    """
    from uclone_x.room.store import RoomStore

    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    args = ["run", "resumer", "--session-id", "thread", "--cwd", str(tmp_path)]

    first = runner.invoke(main.app, [*args, "--prompt", "one"])
    second = runner.invoke(main.app, [*args, "--prompt", "two"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert RoomStore().list_room_ids() == ("thread",)
    state = store.load(_seat("thread", "resumer"))
    assert state is not None
    assert [m.content for m in state.messages if m.role == MessageRole.USER] == ["one", "two"]
    assert "New conversation" not in first.stderr + second.stderr


def test_a_room_that_seats_another_clone_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Naming another clone's room exits 2 instead of opening a second seat in it.

    Killed by: src/uclone_x/room/one_seat.py :: if seat is None:
    Becomes: if False:
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    ran = runner.invoke(
        main.app,
        ["run", "owner", "--prompt", "hi", "--session-id", "mine", "--cwd", str(tmp_path)],
    )

    result = runner.invoke(
        main.app,
        ["run", "visitor", "--prompt", "hi", "--session-id", "mine", "--cwd", str(tmp_path)],
    )

    assert ran.exit_code == 0, ran.output
    assert result.exit_code == 2, result.output
    assert "Invalid --session-id:" in result.output
    assert "does not seat 'visitor'" in " ".join(result.output.split())
    assert store.load(_seat("mine", "visitor")) is None


def test_the_app_opens_a_run_s_room_with_its_turn_in_the_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The app's GET `/api/rooms/{id}` shows a `ucx run` turn, not a bare roster (#1837).

    Opened the way the app opens it: `create_ui_app` on its default storage, with no
    `UCLONE_ROOM_DIR`, so the test fails if the CLI writes its rooms anywhere the app
    does not read them -- which it did, into `~/.uclone/rooms`, until #1837.

    Killed by: src/uclone_x/cli/commands/run.py :: record_in_room(rooms, room_id=room.room_id, clone_id=agent_name, turn=turn, out=out)
    Becomes: None
    Killed by: src/uclone_x/room/store.py :: return default_session_root() / ROOMS_SUBDIR
    Becomes: return Path.home() / ".uclone" / "rooms"
    """
    from fastapi.testclient import TestClient

    from uclone_x.room.store import ROOM_STORAGE_DIR_ENV_VAR
    from uclone_x.ui.app import create_ui_app

    _isolated_run_env(monkeypatch, tmp_path)
    monkeypatch.delenv(ROOM_STORAGE_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _use_connector(monkeypatch, _answering_llm("hello back"))

    result = runner.invoke(
        main.app, ["run", "shown", "--prompt", "hi", "--session-id", "seen", "--cwd", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output

    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector())
    with TestClient(app) as client:
        listed = client.get("/api/rooms")
        opened = client.get("/api/rooms/seen")

    assert listed.status_code == 200, listed.text
    assert "seen" in listed.text
    assert opened.status_code == 200, opened.text
    rows = [r for r in opened.json()["transcript"] if r["kind"] == "utterance"]
    spoken = [(r["sender_id"], r["content"]) for r in rows if r.get("content")]
    assert spoken[-2:] == [("user", "hi"), ("shown", "hello back")]
    asked, answered = rows[-2], rows[-1]
    assert answered["rendered_through"] == asked["seq"]
    assert opened.json()["last_seen_seq"] == {"shown": str(asked["seq"])}


def test_a_run_that_says_nothing_leaves_no_room(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A REPL closed before its first turn stores no room, so it does not list one (#1846).

    Author's choice: a head room is written with its first recorded turn.

    Killed by: src/uclone_x/room/one_seat.py :: return resolve_one_seat_room(
    Becomes: return open_one_seat_room(
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    _feed_prompts(monkeypatch, "/exit")

    result = runner.invoke(main.app, ["run", "quiet", "--cwd", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert RoomStore().list_room_ids() == ()


def test_a_failed_run_turn_is_in_the_room_and_left_unseen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed turn lands in the transcript with its error, and does not advance the seat.

    As a seat's failed turn does in the app: `last_seen_seq` stays, so the next turn's
    span renders the failure again.

    Killed by: src/uclone_x/room/one_seat.py :: if turn.error is None:
    Becomes: if True:
    """
    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _raising_llm())

    result = runner.invoke(
        main.app,
        ["run", "broken", "--prompt", "hi", "--session-id", "oops", "--cwd", str(tmp_path)],
    )

    assert result.exit_code != 0
    state = RoomStore().load("oops")
    assert state is not None, "a failed turn left no room"
    reply = state.transcript[-1]
    assert reply.sender_id == "broken"
    assert reply.error is not None
    assert state.transcript[-2].content == "hi"
    assert "broken" not in state.last_seen_seq


def test_a_repl_loop_tick_is_saved_to_the_seat_before_the_room_shows_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A REPL `/loop` tick reaches the seat's session before the room records it (#1837).

    Otherwise a process killed after the room took the tick leaves a room showing a turn
    that `--session-id` resumes without. The scheduler is replaced so the tick is fired
    by hand, through the REPL's own callback, while the REPL is still open.

    Killed by: src/uclone_x/cli/commands/run.py :: _persist_tick(agent)
    Becomes: pass
    """
    from datetime import UTC, datetime

    from uclone_x.agent.loop import LoopJob, LoopTickResult

    _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("ticked"))
    held: dict[str, Any] = {}
    order: list[str] = []

    class _HandDrivenScheduler:
        def __init__(self, agent: Any, on_tick_completed: Any, **_kwargs: Any) -> None:
            held["tick"] = on_tick_completed
            real_persist = agent.persist_session

            def _persist(*args: Any, **kwargs: Any) -> Any:
                order.append("persist")
                return real_persist(*args, **kwargs)

            agent.persist_session = _persist

        def __getattr__(self, name: str) -> Any:
            return MagicMock()

    real_record = run.record_in_room

    def _spy(*args: Any, **kwargs: Any) -> bool:
        order.append("record")
        return real_record(*args, **kwargs)

    now = datetime.now(UTC)
    job = LoopJob(job_id="j1", interval_seconds=60.0, prompt="tick prompt")
    tick = LoopTickResult(
        tick_index=1,
        started_at=now,
        finished_at=now,
        duration_seconds=0.0,
        success=False,
        error="The turn failed.",
    )
    asks = iter(["/exit"])

    def _ask(*args: Any, **kwargs: Any) -> str:
        if not order:
            held["tick"](job, tick)
        return next(asks)

    monkeypatch.setattr(run, "LoopScheduler", _HandDrivenScheduler)
    monkeypatch.setattr(run, "record_in_room", _spy)
    monkeypatch.setattr("rich.prompt.Prompt.ask", _ask)
    result = runner.invoke(
        main.app, ["run", "looper", "--session-id", "ticks", "--cwd", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output

    assert order[:2] == ["persist", "record"]
    room = RoomStore().load("ticks")
    assert room is not None
    assert [m.content for m in room.transcript if m.kind == "utterance"][:1] == ["tick prompt"]


def test_a_room_that_seats_other_clones_too_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A room `ucx room` made for two clones is not a head's: `run` exits 2 there.

    Answering in it would write a seat session with no seat framing that the room's
    transcript never sees.

    Killed by: src/uclone_x/room/one_seat.py :: if not is_one_seat(state.participants):
    Becomes: if False:
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    rooms = RoomService(RoomStore())
    rooms.create(title="both", room_id="room_multi")
    rooms.add_participant("room_multi", "user", kind=ParticipantKind.HUMAN)
    rooms.add_participant("room_multi", "writer", kind=ParticipantKind.AGENT)
    rooms.add_participant("room_multi", "champion", kind=ParticipantKind.AGENT)

    result = runner.invoke(
        main.app,
        ["run", "writer", "--prompt", "x", "--session-id", "room_multi", "--cwd", str(tmp_path)],
    )

    assert result.exit_code == 2, result.output
    assert "Invalid --session-id:" in result.output
    assert "seats other clones besides 'writer'" in " ".join(result.output.split())
    assert "Traceback" not in result.output
    assert store.load(_seat("room_multi", "writer")) is None


@pytest.mark.parametrize(
    ("head", "keeper"), [(None, "belongs to the app"), ("loop", "belongs to ucx loop")]
)
def test_a_room_another_writer_keeps_is_refused_plainly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, head: str | None, keeper: str
) -> None:
    """A room has one owner: `run` does not continue the app's room or another head's (#1885).

    Continuing it would make `run` a second writer beside the app or `ucx loop`, each
    reading rows and a seat session the other wrote. The refusal says where to go instead.

    Killed by: src/uclone_x/room/one_seat.py :: keeper = "the app" if state.head is None else head_keeper(state.head)
    Becomes: keeper = head_keeper(state.head or "")
    Killed by: src/uclone_x/room/one_seat.py :: if state.head != head and not _unmarked_server_room(state, room_id, head):
    Becomes: if state.head is not None and state.head != head:
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _mock_llm())
    rooms = RoomService(RoomStore())
    before = rooms.create(
        title="elsewhere",
        room_id="room_elsewhere",
        head=head,
        seats=[("user", ParticipantKind.HUMAN), ("writer", ParticipantKind.AGENT)],
    )

    result = runner.invoke(
        main.app,
        [
            "run",
            "writer",
            "--prompt",
            "x",
            "--session-id",
            "room_elsewhere",
            "--cwd",
            str(tmp_path),
        ],
    )

    said = " ".join(result.output.split())
    assert result.exit_code == 2, result.output
    assert "Invalid --session-id:" in said
    assert keeper in said
    assert "leave the id out to start a new one" in said
    assert "Traceback" not in result.output
    assert store.load(_seat("room_elsewhere", "writer")) is None
    assert rooms.get("room_elsewhere") == before


def test_a_run_says_on_stderr_that_it_kept_a_record_it_could_not_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `--session-id` run over a newer build's record says it kept that record (#1860).

    Before #1860 only a room's row said so; `ucx run` set the record aside in silence.
    The line goes to stderr, so stdout still carries the reply alone.

    Killed by: src/uclone_x/cli/commands/run.py :: set_aside = report_set_aside(session_store, turn_session, err_console)
    Becomes: set_aside = False
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("the reply"))
    seat = _seat("kept", "keeper")
    document = json.loads(SessionState.seed(seat, "keeper").model_dump_json())
    document["a_field_from_a_newer_build"] = True
    path = store.session_path(seat)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")

    result = runner.invoke(
        main.app,
        ["run", "keeper", "--prompt", "hello", "--session-id", "kept", "--cwd", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert run.SESSION_SET_ASIDE_NOTICE in " ".join(result.stderr.split())
    assert run.SESSION_SET_ASIDE_NOTICE not in " ".join(result.stdout.split())
    assert "the reply" in result.stdout
    assert [p.name for p in path.parent.iterdir() if ".unreadable-" in p.name], "nothing set aside"


def test_a_runs_room_row_says_it_kept_a_record_so_the_app_says_so_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The clone's row in the run's room carries the flag the app shows a notice for (#1877).

    Killed by: src/uclone_x/cli/commands/run.py :: _record(replace(head_turn, session_set_aside=set_aside), err_console)
    Becomes: _record(head_turn, err_console)
    Killed by: src/uclone_x/room/one_seat.py :: session_set_aside=turn.session_set_aside,
    Becomes: session_set_aside=False,
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("the reply"))
    _a_newer_builds_seat_record(store, _seat("kept", "keeper"), "keeper")

    for _ in range(2):
        result = runner.invoke(
            main.app,
            ["run", "keeper", "--prompt", "hello", "--session-id", "kept", "--cwd", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output

    room = RoomStore().load("kept")
    assert room is not None
    said = [
        row.session_set_aside
        for row in room.transcript
        if row.sender_id == "keeper" and row.kind == "utterance"
    ]
    assert said == [True, False]  # once, on the turn whose save kept it


def _a_newer_builds_seat_record(store: SessionStore, seat: str, agent: str) -> Path:
    document = json.loads(SessionState.seed(seat, agent).model_dump_json())
    document["a_field_from_a_newer_build"] = True
    path = store.session_path(seat)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_repl_says_it_kept_a_record_right_after_the_turn_that_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The line follows the first turn's save, before the next turn, not only on the way out.

    Killed by: src/uclone_x/cli/commands/run.py :: turn_set_aside = report_set_aside(session_store, agent.session_id, console)
    Becomes: turn_set_aside = False
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("the reply"))
    _a_newer_builds_seat_record(store, _seat("kept", "keeper"), "keeper")
    _feed_prompts(monkeypatch, "hello", "again", "/exit")

    result = runner.invoke(
        main.app, ["run", "keeper", "--session-id", "kept", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    out = " ".join(result.stdout.split())
    notice = out.find(run.SESSION_SET_ASIDE_NOTICE)
    first = out.find("keeper: the reply")
    second = out.find("keeper: the reply", first + 1)
    assert 0 <= first < notice < second, out


def test_the_repl_says_it_kept_a_record_when_it_exits_before_any_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No turn ran, but the save on the way out set the record aside: that is said too.

    Killed by: src/uclone_x/cli/commands/run.py :: report_set_aside(session_store, agent.session_id, console)  # the save on the way out
    Becomes: pass
    """
    store = _isolated_run_env(monkeypatch, tmp_path)
    _use_connector(monkeypatch, _answering_llm("the reply"))
    path = _a_newer_builds_seat_record(store, _seat("kept", "keeper"), "keeper")
    _feed_prompts(monkeypatch, "/exit")

    result = runner.invoke(
        main.app, ["run", "keeper", "--session-id", "kept", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert [p for p in path.parent.iterdir() if ".unreadable-" in p.name], "nothing set aside"
    assert run.SESSION_SET_ASIDE_NOTICE in " ".join(result.stdout.split())
