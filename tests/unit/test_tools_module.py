"""The tools layer as a selectable module (#2188).

What the issue requires, each pinned here:

1. the module a request was built under is recorded, and the log-only rebuild of every
   request is byte-identical to what was sent, for each module;
2. a module change opens a new epoch, with the cause `tools_module_changed`, even when
   the conversation only grew;
3. one place selects the module, clone setting then provider default, and a 1:1 chat and
   a room seat of one clone get the same one;
4. (the eval arms run this code) -- pinned beside the eval suite's own tests, which are
   not published with this file;
5. a module name this build does not know is refused in plain words, with no internals.

And the default: a clone that selects nothing runs ``native``, and its requests and its
record are what they were before modules existed.

The fake embedder answers every text it is given, by a fixed table, for as many calls as
it gets, and the scripted connector fails the test if a turn asks for more replies than
it holds, so no loop ends early because a fake ran dry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, PersonaDefinition
from uclone_x.agent.request_record import RequestRecordError, rebuild_requests, serialize_tools
from uclone_x.agent.session import ContextSnapshot, SessionStore
from uclone_x.agent.tools_module import (
    EPOCH_TOOLS_MODULE_CHANGED,
    UNKNOWN_RECORDED_TOOLS_MODULE_MESSAGE,
    UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE,
    UnknownToolsModuleError,
    bound_layer,
    grow_bound,
    recorded_tools_module,
    select_tools_module,
)
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)
from uclone_x.log.reader import read_session_log
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry
from uclone_x.tools.tool_binder import SEARCH_TOOLS_NAME, ToolBinder

_AXES = ("alpha_image", "beta_mail", "gamma_calc")
_SESSION = "sess_modules"


def _toward(name: str) -> tuple[float, ...]:
    """A query vector whose cosine with `name`'s axis is 0.9, and with the others 0."""
    vector = [0.9 if axis == name else 0.0 for axis in _AXES]
    return (*vector, math.sqrt(1.0 - 0.81))


class _Embedder:
    """Embeds by table: a tool's text by its name, a message by `_QUERIES`."""

    @property
    def model_name(self) -> str:
        return "fake-embed"

    @property
    def dimensions(self) -> int:
        return len(_AXES) + 1

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        out: list[tuple[float, ...]] = []
        for text in texts:
            if text in _QUERIES:
                out.append(_QUERIES[text])
            else:
                name = text.split(":", 1)[0]
                out.append((*(1.0 if axis == name else 0.0 for axis in _AXES), 0.0))
        return tuple(out)


_QUERIES = {
    "draw it": _toward("alpha_image"),
    "now mail it": _toward("beta_mail"),
    "and once more": _toward("alpha_image"),
}


class _Params(BaseModel):
    text: str = Field(default="")


def _tool(tool_name: str) -> BaseTool[_Params]:
    class _Named(BaseTool[_Params]):
        name = tool_name
        description = f"The {tool_name} tool."

        def run(self, params: _Params, context: ToolContext) -> str:
            return f"{tool_name} ran"

    return _Named()


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name in ("file_read", *_AXES):
        registry.register(_tool(name))
    return registry


class _Scripted(MockLLMConnector):
    """Answers from a script, one reply per request, and keeps every request."""

    def __init__(self, script: list[tuple[ToolCallRequest, ...] | str]) -> None:
        super().__init__()
        self.script = list(script)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        assert self.script, "the turn asked for more replies than the script holds"
        step = self.script.pop(0)
        calls = step if isinstance(step, tuple) else ()
        ref = ServiceRef(provider="mock", model="mock-model")
        return ModelResponse(
            content=step if isinstance(step, str) else None,
            tool_calls=calls,
            usage=TokenUsage(provider="mock", model="mock-model", input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
            model_name="mock-model",
            provenance=Provenance(path=ExecutionPath.PRIMARY, requested=ref, served_by=ref),
        )


def _call(name: str, call_id: str) -> tuple[ToolCallRequest, ...]:
    return (ToolCallRequest(id=call_id, name=name, arguments={"text": "x"}),)


def _agent(
    wire: _Scripted, store: SessionStore, module: str | None, *, binder: bool = True
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id="moduled",
            name="M",
            llm_config=AgentLLMConfig(model_name="mock-model"),
            tools_module=module,
        ),
        llm=wire,
        tools=_registry(),
        context=AgentContext(session_id=_SESSION, agent_id="moduled"),
        tool_binder=ToolBinder(_Embedder()) if binder else None,
        store=store,
    )


def _names(request: LLMRequest) -> list[str]:
    return [tool.name for tool in request.tools]


def _dump(request: LLMRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


async def _two_turns(
    store: SessionStore, module: str | None, *, binder: bool = True
) -> list[LLMRequest]:
    """Two user messages, each binding a tool and calling it, then a restart and a third."""
    first = _Scripted([_call("alpha_image", "c1"), "drawn", _call("beta_mail", "c2"), "mailed"])
    agent = _agent(first, store, module, binder=binder)
    await agent.start()
    assert (await agent.execute_turn("draw it")).error is None
    assert (await agent.execute_turn("now mail it")).error is None
    agent.persist_session(_SESSION)

    second = _Scripted([_call("alpha_image", "c3"), "again"])
    restarted = _agent(second, store, module, binder=binder)
    assert restarted.hydrate_session(_SESSION) is not None
    await restarted.start()
    assert (await restarted.execute_turn("and once more")).error is None
    restarted.persist_session(_SESSION)
    return [*first.requests, *second.requests]


def _events(store: SessionStore) -> list[dict[str, Any]]:
    path = store.event_log_path(_SESSION)
    assert path is not None
    return [dict(event) for event in read_session_log(path)]


# --------------------------------------------------------------------------------------
# Selection: one place, clone setting then provider default
# --------------------------------------------------------------------------------------


def test_the_clone_setting_wins_and_nothing_chosen_is_native_on_every_provider() -> None:
    """Killed by: src/uclone_x/agent/tools_module.py :: chosen = (setting or "").strip()
    Becomes: chosen = ""
    """
    for provider in ("ollama", "anthropic", "openai", "gemini", "vllm", "mock", ""):
        assert select_tools_module(None, provider) == "native"
        assert select_tools_module("", provider) == "native"
        assert select_tools_module("pinned", provider) == "pinned"
        assert select_tools_module(" bound ", provider) == "bound"


@pytest.mark.parametrize("name", ["k_text", "K", "all", "Pinned"])
def test_an_unknown_setting_is_refused_in_plain_words_with_no_internals(name: str) -> None:
    """Killed by: src/uclone_x/agent/tools_module.py :: raise UnknownToolsModuleError(UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE, name=chosen)
    Becomes: return DEFAULT_TOOLS_MODULE
    """
    with pytest.raises(UnknownToolsModuleError) as caught:
        select_tools_module(name, "ollama")
    message = str(caught.value)
    assert message == UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE
    assert caught.value.name == name
    # "k_act" is a choice the sentence names, as a person writes it; nothing else has "_".
    for internal in ("tools_module", "Error", "Literal", ".py", name):
        assert internal not in message, internal
    assert "_" not in message.replace("k_act", "")


def test_an_unknown_recorded_module_is_refused_in_plain_words_with_no_internals() -> None:
    """Absent reads as native; anything unknown is refused, never read as another module.

    Killed by: src/uclone_x/agent/tools_module.py :: raise UnknownToolsModuleError(UNKNOWN_RECORDED_TOOLS_MODULE_MESSAGE, name=recorded)
    Becomes: return DEFAULT_TOOLS_MODULE
    """
    assert recorded_tools_module(None) == "native"
    assert recorded_tools_module("bound") == "bound"
    with pytest.raises(UnknownToolsModuleError) as caught:
        recorded_tools_module("k_text")
    message = str(caught.value)
    assert message == UNKNOWN_RECORDED_TOOLS_MODULE_MESSAGE
    for internal in ("k_text", "tools_module", "snapshot", "Error", ".py", "_"):
        assert internal not in message, internal


def test_a_chat_and_a_seat_of_one_clone_run_the_module_its_setting_names(
    tmp_path: Path,
) -> None:
    """The clone file's setting reaches both heads through the one builder.

    Killed by: src/uclone_x/agent/bootstrap.py :: tools_module=persona.tools_module,
    Becomes: tools_module=None,
    Killed by: src/uclone_x/agent/base.py :: config.tools_module, str(getattr(llm, "provider_name", "") or "")
    Becomes: None, str(getattr(llm, "provider_name", "") or "")
    """
    from tests.support.app_clone import app_clone
    from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR, create_clone
    from uclone_x.room.models import ParticipantKind
    from uclone_x.ui.app import AgentSessionManager
    from uclone_x.ui.rooms import RoomStack

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
        create_clone(
            "moduled",
            "handle: moduled\nname: moduled\nrole: Tester\n"
            "system_prompt: You test.\ntools_module: pinned\n",
        )
        mgr = AgentSessionManager(
            storage_dir=tmp_path / "sessions",
            llm=MockLLMConnector(),
            workspace_dir=tmp_path / "workspace",
        )
        chat = app_clone(mgr, "moduled")
        stack = RoomStack(mgr)
        state = stack.service.create(title="Modules")
        stack.service.add_participant(
            state.room_id, participant_id="moduled", kind=ParticipantKind.AGENT
        )
        seated = stack.store.load(state.room_id)
        assert seated is not None
        resolver = stack.orchestrator(seated)._resolver  # pyright: ignore[reportPrivateUsage]
        participant = next(p for p in seated.participants if p.id == "moduled")
        seat = cast("BaseAgent", asyncio.run(resolver.resolve(participant)))

    assert (chat.tools_module, seat.tools_module) == ("pinned", "pinned")


def test_a_persona_with_no_setting_is_written_as_before() -> None:
    """The field is left out of a persona's dump when unset, so a session record that
    carries the persona, and every persona file, keeps its shape (#1844).

    Killed by: src/uclone_x/core/models.py :: data.pop("tools_module", None)
    Becomes: pass
    """
    persona = PersonaDefinition(name="plain", role="R", system_prompt="S")
    assert "tools_module" not in persona.model_dump(mode="json")
    chosen = persona.model_copy(update={"tools_module": "bound"})
    assert chosen.model_dump(mode="json")["tools_module"] == "bound"


# --------------------------------------------------------------------------------------
# What each module declares
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pinned_declares_every_held_tool_though_a_binder_is_given(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/tool_invoker.py :: self._binding = binder if tools_module != "pinned" else None
    Becomes: self._binding = binder
    """
    wire = _Scripted(["drawn"])
    agent = _agent(wire, SessionStore(tmp_path), "pinned")
    await agent.execute_turn("draw it")
    assert _names(wire.requests[0]) == sorted(("file_read", *_AXES))


@pytest.mark.asyncio
async def test_bound_binds_like_native_but_offers_no_search(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/tool_invoker.py :: if self._binding is not None and tools_module == "native":
    Becomes: if self._binding is not None:
    """
    wire = _Scripted(["drawn"])
    agent = _agent(wire, SessionStore(tmp_path), "bound")
    await agent.execute_turn("draw it")
    assert _names(wire.requests[0]) == ["file_read", "alpha_image"]


@pytest.mark.asyncio
async def test_native_is_the_layer_it_was_base_search_then_bound(tmp_path: Path) -> None:
    wire = _Scripted(["drawn"])
    agent = _agent(wire, SessionStore(tmp_path), None)
    await agent.execute_turn("draw it")
    assert _names(wire.requests[0]) == ["file_read", SEARCH_TOOLS_NAME, "alpha_image"]


def test_the_bound_set_grows_sorted_and_skips_what_is_held() -> None:
    bound = ["delta"]
    assert grow_bound(bound, ["gamma", "alpha", "delta", "base"], ["base"]) == ["alpha", "gamma"]
    assert bound == ["delta", "alpha", "gamma"]
    defs = [ToolDefinition(name=n, description=n, parameters={}) for n in ("alpha", "delta")]
    base = [ToolDefinition(name="base", description="b", parameters={})]
    assert [d.name for d in bound_layer(base, defs, bound)] == ["base", "delta", "alpha"]


# --------------------------------------------------------------------------------------
# Recorded, and rebuilt byte for byte
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("module", ["native", "pinned", "bound"])
async def test_each_module_is_recorded_and_every_request_rebuilds_byte_for_byte(
    tmp_path: Path, module: str
) -> None:
    """Every request, across turns and a restart, rebuilt from the record and log alone.

    Killed by: src/uclone_x/agent/tools_module.py :: return None if module == DEFAULT_TOOLS_MODULE else module
    Becomes: return None
    Killed by: src/uclone_x/agent/request_record.py :: tools_module = recorded_tools_module(snapshot.tools_module)
    Becomes: tools_module = DEFAULT_TOOLS_MODULE
    """
    store = SessionStore(tmp_path)
    sent = await _two_turns(store, module)
    state = store.load(_SESSION)
    assert state is not None

    rebuilt = rebuild_requests(store, state, _events(store))

    assert len(rebuilt) == len(sent) == 6
    for number, (request, original) in enumerate(zip(rebuilt, sent, strict=True)):
        assert request.verified, number
        assert request.tools_module == module, number
        assert _dump(request.request) == _dump(original.model_copy(update={"context_window": None}))
    recorded = {snapshot.tools_module for snapshot in state.context_snapshots}
    assert recorded == ({None} if module == "native" else {module})


@pytest.mark.asyncio
async def test_the_default_path_sends_and_records_what_native_did(tmp_path: Path) -> None:
    """A clone that selects nothing sends byte-identical requests to an explicit ``native``
    one, and its snapshots are written -- and hashed into their ids -- with no module key,
    the shape they had before modules existed.

    Killed by: src/uclone_x/core/session_state.py :: data.pop("tools_module", None)
    Becomes: pass
    """
    unset = await _two_turns(SessionStore(tmp_path / "unset"), None)
    native = await _two_turns(SessionStore(tmp_path / "native"), "native")
    assert [_dump(r) for r in unset] == [_dump(r) for r in native]

    state = SessionStore(tmp_path / "unset").load(_SESSION)
    assert state is not None
    for snapshot in state.context_snapshots:
        dumped = snapshot.model_dump(mode="json")
        assert "tools_module" not in dumped
        assert ContextSnapshot.model_validate(dumped).snapshot_id == snapshot.snapshot_id


#: SHA-256 of `_two_turns`'s six tools layers (`serialize_tools`, joined by newlines) with
#: no module chosen, as base c9583da1 -- main before modules -- sent them, measured there
#: with this file's scenario (#2188 review). Drift on the default path fails here.
_BASE_TOOLS_LAYERS = {
    True: "67f5f4064470123e7f49200dca4979fa0b64ea81a2911f4b427f7428c0fc0cba",
    False: "50e621806b1ee2ecb26d92d17cdc297697fdd6c8973612723a9221119736d55c",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("binder", [True, False], ids=["binder", "no binder"])
async def test_the_default_path_sends_the_tools_layers_main_sent_before_modules(
    tmp_path: Path, binder: bool
) -> None:
    """Killed by: src/uclone_x/agent/tool_invoker.py :: if self._search_tool is not None:
    Becomes: if False:
    """
    sent = await _two_turns(SessionStore(tmp_path), None, binder=binder)
    layers = "\n".join(serialize_tools(request.tools) for request in sent)
    assert len(sent) == 6
    assert hashlib.sha256(layers.encode()).hexdigest() == _BASE_TOOLS_LAYERS[binder]


# --------------------------------------------------------------------------------------
# A change opens an epoch
# --------------------------------------------------------------------------------------


async def _restart_under(store: SessionStore, module: str | None, script: list[Any]) -> BaseAgent:
    agent = _agent(_Scripted(script), store, module)
    assert agent.hydrate_session(_SESSION) is not None
    await agent.start()
    return agent


@pytest.mark.asyncio
async def test_a_module_change_opens_an_epoch_even_when_the_history_only_grew(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/prompt_assembler.py :: session.declare_new_epoch(EPOCH_TOOLS_MODULE_CHANGED)
    Becomes: pass
    Killed by: src/uclone_x/core/context_state.py :: _EPOCH_FORCING: Final = frozenset({EPOCH_PERSONA_EDITED, EPOCH_TOOLS_MODULE_CHANGED})
    Becomes: _EPOCH_FORCING: Final = frozenset({EPOCH_PERSONA_EDITED})
    """
    store = SessionStore(tmp_path)
    first = _agent(_Scripted(["drawn"]), store, None)
    await first.start()
    await first.execute_turn("draw it")
    first.persist_session(_SESSION)

    # The same module after a restart: the next request extends the epoch.
    same = await _restart_under(store, None, ["mailed"])
    await same.execute_turn("now mail it")
    same.persist_session(_SESSION)
    state = store.load(_SESSION)
    assert state is not None
    assert len(state.context_epochs) == 1

    # Another module: a new epoch, opened by the change, at the turn's first request.
    changed = await _restart_under(store, "pinned", ["again", "and again"])
    await changed.execute_turn("and once more")
    await changed.execute_turn("and once more")
    changed.persist_session(_SESSION)
    state = store.load(_SESSION)
    assert state is not None
    assert len(state.context_epochs) == 2
    opened = state.context_epochs[-1]
    assert EPOCH_TOOLS_MODULE_CHANGED in opened.opened_by
    assert (opened.turn, opened.step) == (3, 1)
    # Every request still rebuilds, each under the module it was sent with.
    rebuilt = rebuild_requests(store, state, _events(store))
    assert [r.tools_module for r in rebuilt] == ["native", "native", "pinned", "pinned"]
    assert all(r.verified for r in rebuilt)


@pytest.mark.asyncio
async def test_a_turn_over_an_unknown_recorded_module_is_refused_plainly_and_sends_nothing(
    tmp_path: Path,
) -> None:
    """A record from a version with a module this one lacks: the turn refuses, the rebuild
    refuses, and neither reads it as some other module.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: previous = recorded_tools_module(session.context_snapshots[-1].tools_module)
    Becomes: previous = DEFAULT_TOOLS_MODULE
    """
    store = SessionStore(tmp_path)
    first = _agent(_Scripted(["drawn"]), store, "pinned")
    await first.start()
    await first.execute_turn("draw it")
    first.persist_session(_SESSION)
    state = store.load(_SESSION)
    assert state is not None
    renamed = {
        s.snapshot_id: s.model_copy(update={"tools_module": "k_text"})
        for s in state.context_snapshots
    }
    future = state.model_copy(update={"context_snapshots": tuple(renamed.values())})
    store.save(future)
    # The log names each snapshot by its id, which the forged module changed.
    events = [
        {**e, "snapshot": renamed[e["snapshot"]].snapshot_id} if e.get("snapshot") else e
        for e in _events(store)
    ]

    wire = _Scripted([])
    later = _agent(wire, store, None)
    assert later.hydrate_session(_SESSION) is not None
    await later.start()
    result = await later.execute_turn("now mail it")

    assert wire.requests == []
    assert result.error == UNKNOWN_RECORDED_TOOLS_MODULE_MESSAGE
    with pytest.raises(RequestRecordError) as caught:
        rebuild_requests(store, future, events)
    assert caught.value.code == "unreadable"
    assert "k_text" not in str(caught.value)


# --------------------------------------------------------------------------------------
# An unknown setting at each head: its sentence, and nothing of the code (#2188 review)
# --------------------------------------------------------------------------------------

#: What a person must never see when a head refuses the setting.
_INTERNALS = ("Traceback", ".py", "Error", "tools_module", "k_text", "uclone_x", "line ")


def _clone_with_an_unknown_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from uclone_x.core.agent_home import create_clone

    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("UCLONE_ROOM_DIR", str(tmp_path / "rooms"))
    create_clone(
        "moduled",
        "handle: moduled\nname: moduled\nrole: Tester\n"
        "system_prompt: You test.\ntools_module: k_text\n",
    )


def _assert_plain_refusal(result: Any) -> None:
    assert result.exit_code == 1, result.output
    # Handled, not escaped: an uncaught exception would be the result's exception.
    assert isinstance(result.exception, SystemExit), result.exception
    assert UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE in " ".join(result.output.split())
    for internal in _INTERNALS:
        assert internal not in result.output, internal


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "moduled", "--provider", "mock", "--prompt", "hi"],
        [
            "loop",
            "run",
            "tick",
            "--interval",
            "1h",
            "--max-runs",
            "1",
            "--provider",
            "mock",
            "--agent",
            "moduled",
        ],
        ["a2a", "serve", "--agent-id", "moduled"],
        ["acp", "serve", "--persona", "moduled"],
    ],
    ids=["run", "loop", "a2a", "acp"],
)
def test_each_cli_head_refuses_an_unknown_module_in_its_plain_sentence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]
) -> None:
    """The sentence and exit 1; no traceback, path, class name, field or the name itself.

    Killed by: src/uclone_x/cli/commands/run.py :: err_console.print(str(exc), markup=False, highlight=False)
    Becomes: raise exc
    Killed by: src/uclone_x/cli/commands/loop.py :: err_console.print(str(exc), markup=False, highlight=False)
    Becomes: raise exc
    Killed by: src/uclone_x/cli/commands/a2a.py :: check_tools_module(app, clone_id)
    Becomes: pass
    Killed by: src/uclone_x/cli/commands/acp.py :: check_tools_module(app, clone_id, persona_def.name)
    Becomes: pass
    """
    from unittest.mock import AsyncMock, MagicMock

    from typer.testing import CliRunner

    from uclone_x.cli import main

    _clone_with_an_unknown_module(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    serving = MagicMock()
    monkeypatch.setattr("uvicorn.run", serving)
    acp_server = MagicMock()
    acp_server.return_value.run_stdio = AsyncMock()
    monkeypatch.setattr("uclone_x.cli.commands.acp.one_seat_acp_server", acp_server)

    result = CliRunner().invoke(main.app, argv)

    _assert_plain_refusal(result)
    assert not serving.called
    assert not acp_server.called


def test_a_room_seat_with_an_unknown_module_shows_the_plain_sentence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The room words a seat it cannot build by the refusal's own sentence, not as a
    problem in the agent runtime.

    Killed by: src/uclone_x/ui/rooms.py :: if isinstance(exc, RoomError | PlainRefusalError):
    Becomes: if isinstance(exc, RoomError):
    """
    from uclone_x.room.models import ParticipantKind
    from uclone_x.ui.app import AgentSessionManager
    from uclone_x.ui.rooms import RoomStack, reader_facing_reason

    _clone_with_an_unknown_module(monkeypatch, tmp_path)
    mgr = AgentSessionManager(
        storage_dir=tmp_path / "sessions",
        llm=MockLLMConnector(),
        workspace_dir=tmp_path / "workspace",
    )
    stack = RoomStack(mgr)
    state = stack.service.create(title="Modules")
    stack.service.add_participant(
        state.room_id, participant_id="moduled", kind=ParticipantKind.AGENT
    )
    seated = stack.store.load(state.room_id)
    assert seated is not None
    resolver = stack.orchestrator(seated)._resolver  # pyright: ignore[reportPrivateUsage]
    participant = next(p for p in seated.participants if p.id == "moduled")

    with pytest.raises(UnknownToolsModuleError) as caught:
        asyncio.run(resolver.resolve(participant))

    reason = reader_facing_reason(caught.value)
    assert reason == UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE
    for internal in _INTERNALS:
        assert internal not in reason, internal
