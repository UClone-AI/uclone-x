"""Host binding and the advertised-tools dispatch check (design §5.1, F3, F12).

The tools layer is a pinned base set that host binding may only append to, once per user
message, and a call runs only if the request that produced it declared its name.

The fake embedder answers every text it is given, by a fixed table, for as many calls as
it gets: it never runs out and so never ends a turn the real one would continue. The
scripted connector fails the test outright if a turn asks it for more replies than the
script holds, rather than quietly answering and letting a loop stop early.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent import BaseAgent
from uclone_x.agent.hooks.models import HookAction, HookContext, HookDecision
from uclone_x.agent.hooks.protocols import BaseHook
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, PersonaDefinition
from uclone_x.agent.session import SessionStore
from uclone_x.agent.tool_invoker import unadvertised_tool_message
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry
from uclone_x.tools.tool_binder import (
    BIND_MIN_SCORE,
    SEARCH_UNAVAILABLE_MESSAGE,
    ToolBinder,
    tool_binder_for,
)
from uclone_x.tools.tool_ranking import closest_tool_names, unknown_tool_message

# One axis per catalog tool. `file_read` is in the base set, so it is never embedded.
_AXES = ("alpha_image", "beta_mail", "delta_web", "epsilon_note", "gamma_calc")


def _axis(name: str) -> tuple[float, ...]:
    return tuple(1.0 if axis == name else 0.0 for axis in _AXES)


def _toward(scores: dict[str, float]) -> tuple[float, ...]:
    """A unit query vector whose cosine with each named tool's axis is the given score."""
    vector = [scores.get(axis, 0.0) for axis in _AXES]
    rest = 1.0 - sum(v * v for v in vector)
    assert rest >= 0.0
    # The remainder goes on an axis no tool has, so each named cosine is exact.
    return (*vector, math.sqrt(rest))


class _FakeEmbedder:
    """Embeds by table: a tool's text by its name, a message by `queries`."""

    def __init__(self, queries: dict[str, tuple[float, ...]]) -> None:
        self.queries = queries
        self.fail = False
        #: Texts whose embedding fails, so one turn can bind and then fail to search.
        self.fail_texts: set[str] = set()
        self.catalog_calls = 0
        self.query_calls = 0

    @property
    def model_name(self) -> str:
        return "fake-embed"

    @property
    def dimensions(self) -> int:
        return len(_AXES) + 1

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if self.fail or self.fail_texts.intersection(texts):
            raise ConnectionError("http://10.0.0.7:11434/api/embed refused: secret-detail")
        out: list[tuple[float, ...]] = []
        for text in texts:
            if text in self.queries:
                self.query_calls += 1
                out.append(self.queries[text])
            else:
                self.catalog_calls += 1
                name = text.split(":", 1)[0]
                out.append((*_axis(name), 0.0))
        return tuple(out)


class _Params(BaseModel):
    text: str = Field(default="")


def _tool(tool_name: str, ran: list[str]) -> BaseTool[_Params]:
    class _Named(BaseTool[_Params]):
        name = tool_name
        description = f"The {tool_name} tool."

        def run(self, params: _Params, context: ToolContext) -> str:
            ran.append(tool_name)
            return f"{tool_name} ran"

    return _Named()


def _room_tool(ran: list[str]) -> BaseTool[_Params]:
    """A held tool that a turn outside a room is not offered (`needs_room`): the case F12
    still refuses once a binding agent binds on call (#2190)."""

    class _RoomOnly(BaseTool[_Params]):
        name = "zeta_room"
        description = "The zeta_room tool."
        needs_room = True

        def run(self, params: _Params, context: ToolContext) -> str:
            ran.append("zeta_room")
            return "zeta_room ran"

    return _RoomOnly()


def _registry(ran: list[str]) -> ToolRegistry:
    registry = ToolRegistry()
    for name in ("file_read", *_AXES):
        registry.register(_tool(name, ran))
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
            usage=TokenUsage(
                provider="mock",
                model="mock-model",
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            ),
            finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
            model_name="mock-model",
            provenance=Provenance(path=ExecutionPath.PRIMARY, requested=ref, served_by=ref),
        )


def _agent(
    wire: _Scripted,
    binder: ToolBinder | None,
    ran: list[str],
    *,
    allowed: tuple[str, ...] = (),
    hooks: tuple[BaseHook, ...] = (),
    store: SessionStore | None = None,
    registry: ToolRegistry | None = None,
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id="binder",
            name="B",
            llm_config=AgentLLMConfig(model_name="mock-model"),
            allowed_tools=allowed,
            hooks=hooks,
        ),
        llm=wire,
        tools=registry if registry is not None else _registry(ran),
        context=AgentContext(session_id="sess_binder", agent_id="binder"),
        tool_binder=binder,
        store=store,
    )


def _names(request: LLMRequest) -> list[str]:
    return [tool.name for tool in request.tools]


def _call(name: str, call_id: str = "c1") -> tuple[ToolCallRequest, ...]:
    return (ToolCallRequest(id=call_id, name=name, arguments={"text": "x"}),)


# --------------------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_binding_appends_its_new_tools_sorted_by_name_not_by_score() -> None:
    """`delta_web` scores higher, and still follows `alpha_image`: the append is sorted.

    Killed by: src/uclone_x/agent/tools_module.py :: added = sorted(set(hits) - set(bound) - set(held))
    Becomes: added = [h for h in hits if h not in bound and h not in held]
    """
    embedder = _FakeEmbedder({"web and pictures": _toward({"delta_web": 0.8, "alpha_image": 0.5})})
    wire = _Scripted(["ok"])
    agent = _agent(wire, ToolBinder(embedder), [])

    await agent.execute_turn("web and pictures")

    assert _names(wire.requests[0]) == ["file_read", "search_tools", "alpha_image", "delta_web"]


@pytest.mark.asyncio
async def test_the_bound_set_only_grows_across_user_messages() -> None:
    """A later message appends; what an earlier one bound stays, in its place.

    Killed by: src/uclone_x/agent/tools_module.py :: bound.extend(added)
    Becomes: bound[:] = added
    """
    embedder = _FakeEmbedder(
        {
            "draw it": _toward({"alpha_image": 0.9}),
            "now mail it": _toward({"beta_mail": 0.9}),
            "draw again": _toward({"alpha_image": 0.9}),
        }
    )
    wire = _Scripted(["one", "two", "three"])
    agent = _agent(wire, ToolBinder(embedder), [])

    await agent.execute_turn("draw it")
    await agent.execute_turn("now mail it")
    await agent.execute_turn("draw again")

    first, second, third = (_names(r) for r in wire.requests)
    assert first == ["file_read", "search_tools", "alpha_image"]
    # No removal: each request's tools are a prefix of the next one's.
    assert second == [*first, "beta_mail"]
    # A message that binds only what is already bound changes nothing.
    assert third == second


@pytest.mark.asyncio
async def test_a_multi_step_turn_sends_one_tools_list_and_a_byte_identical_prefix() -> None:
    """Binding happens once per user message; every step of the turn sends the same layer.

    The second step's request extends the first's byte for byte: same tools, and the first
    request's messages as its prefix. The message is embedded once for the whole turn.

    Killed by: src/uclone_x/agent/tool_invoker.py :: tools = self.declared_tools(tool_defs)
    Becomes: tools = self.declared_tools(self.advertised_tool_definitions())
    """
    embedder = _FakeEmbedder({"draw it": _toward({"alpha_image": 0.9})})
    ran: list[str] = []
    wire = _Scripted([_call("alpha_image"), _call("file_read", "c2"), "done"])
    agent = _agent(wire, ToolBinder(embedder), ran)

    result = await agent.execute_turn("draw it")

    assert result.is_completed is True
    assert ran == ["alpha_image", "file_read"]
    assert len(wire.requests) == 3
    tools_bytes = {b"".join(t.model_dump_json().encode() for t in r.tools) for r in wire.requests}
    assert len(tools_bytes) == 1
    for earlier, later in zip(wire.requests, wire.requests[1:], strict=False):
        head = later.messages[: len(earlier.messages)]
        assert [m.model_dump_json() for m in head] == [
            m.model_dump_json() for m in earlier.messages
        ]
    assert embedder.query_calls == 1


@pytest.mark.asyncio
async def test_the_floor_keeps_low_scores_out_and_top_k_caps_the_rest() -> None:
    """Below 0.35 binds nothing; above it, at most three tools, the best ones.

    Killed by: src/uclone_x/tools/tool_binder.py :: kept = [(score, name) for score, name in scored if score >= self._min_score]
    Becomes: kept = [(score, name) for score, name in scored if score >= 0.0]
    """
    embedder = _FakeEmbedder(
        {
            "just chatting": _toward({"alpha_image": BIND_MIN_SCORE - 0.01, "beta_mail": 0.2}),
            "do everything": _toward(
                {"alpha_image": 0.5, "beta_mail": 0.45, "delta_web": 0.4, "gamma_calc": 0.36}
            ),
        }
    )
    binder = ToolBinder(embedder)
    catalog = [ToolDefinition(name=n, description=f"The {n} tool.", parameters={}) for n in _AXES]

    assert await binder.bind("just chatting", catalog) == ()
    assert await binder.bind("do everything", catalog) == ("alpha_image", "beta_mail", "delta_web")
    # Each description was embedded once, in one call, for both messages.
    assert binder.catalog_embed_calls == 1
    assert embedder.catalog_calls == len(_AXES)


@pytest.mark.asyncio
async def test_a_declared_tool_list_is_a_range_that_pins_only_the_base_set() -> None:
    """The list is what the clone may use: base is pinned, the rest is bound when needed.

    `beta_mail` is in the range and matches, so it binds and runs; `gamma_calc` is in the
    range and does not match, so the request does not carry it (design §5.1, Revision 3).

    Killed by: src/uclone_x/agent/tool_invoker.py :: return _PINNED_BASE_TOOLS
    Becomes: return frozenset(self._scope.allowed_tools()) or _PINNED_BASE_TOOLS
    """
    embedder = _FakeEmbedder({"mail it": _toward({"beta_mail": 0.9})})
    ran: list[str] = []
    wire = _Scripted([_call("beta_mail"), "sent"])
    agent = _agent(
        wire, ToolBinder(embedder), ran, allowed=("file_read", "beta_mail", "gamma_calc")
    )

    await agent.execute_turn("mail it")

    assert _names(wire.requests[0]) == ["file_read", "search_tools", "beta_mail"]
    assert ran == ["beta_mail"]


@pytest.mark.asyncio
async def test_a_range_never_binds_or_runs_a_tool_outside_it() -> None:
    """`alpha_image` is registered and matches best, but the range leaves it out.

    Killed by: src/uclone_x/agent/tool_invoker.py :: allowed = self._scope.allowed_tools() if self._scope.allowed_tools() else None
    Becomes: allowed = None
    """
    embedder = _FakeEmbedder(
        {"draw it": _toward({"alpha_image": 0.9}), "draw": _toward({"alpha_image": 0.9})}
    )
    ran: list[str] = []
    wire = _Scripted([_search("draw"), _call("alpha_image", "c2"), "cannot"])
    agent = _agent(wire, ToolBinder(embedder), ran, allowed=("file_read", "beta_mail"))

    result = await agent.execute_turn("draw it")

    assert [_names(r) for r in wire.requests] == [["file_read", "search_tools"]] * 3
    assert "No tool matched" in _tool_text(wire.requests[1], "s1")
    assert ran == []
    assert result.tool_executions[-1].status is ToolResultStatus.ERROR


@pytest.mark.asyncio
async def test_a_clone_with_a_range_can_search_it_and_call_what_it_found() -> None:
    """No list names `search_tools`, and the range still lets it run.

    Killed by: src/uclone_x/agent/tool_invoker.py :: return name == SEARCH_TOOLS_NAME and self._search_tool is not None
    Becomes: return False
    """
    embedder = _FakeEmbedder({"go": _toward({}), "send mail": _toward({"beta_mail": 0.9})})
    ran: list[str] = []
    wire = _Scripted([_search("send mail"), _call("beta_mail", "c2"), "done"])
    agent = _agent(wire, ToolBinder(embedder), ran, allowed=("file_read", "beta_mail"))

    result = await agent.execute_turn("go")

    assert _names(wire.requests[1]) == ["file_read", "search_tools", "beta_mail"]
    assert ran == ["beta_mail"]
    assert [r.status for r in result.tool_executions] == [
        ToolResultStatus.SUCCESS,
        ToolResultStatus.SUCCESS,
    ]


@pytest.mark.asyncio
async def test_a_range_with_no_base_tool_pins_only_the_search() -> None:
    """Nothing in the range is base, so the whole range is catalog and none of it is pinned.

    Killed by: src/uclone_x/agent/tool_invoker.py :: catalog_names = {d.name for d in catalog}
    Becomes: catalog_names = set()
    """
    embedder = _FakeEmbedder({"mail it": _toward({"beta_mail": 0.9})})
    wire = _Scripted(["ok"])
    agent = _agent(wire, ToolBinder(embedder), [], allowed=("beta_mail", "gamma_calc"))

    await agent.execute_turn("mail it")

    assert _names(wire.requests[0]) == ["search_tools", "beta_mail"]


def _compact_on_check(agent: BaseAgent, monkeypatch: pytest.MonkeyPatch, which: int) -> list[str]:
    """Make the `which`-th compaction check of the session say yes, and only that one.

    A turn checks at its start only (§5.8, #1443).
    """
    checks: list[str] = []

    def should(sid: str, history: object, request: object = None) -> bool:
        checks.append(sid)
        return len(checks) == which

    monkeypatch.setattr(agent, "_should_compact_session", should)
    return checks


@pytest.mark.asyncio
async def test_a_turn_that_ran_tools_is_checked_for_compaction_only_at_its_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No compaction runs between two steps (§5.8, #1443), so a bound set is never
    re-seeded mid-turn: the only check a turn makes is at its start.

    Before #1443 the step after `alpha_image` ran was checked too, and a compaction there
    had to carry the turn's bound set over (#1422).

    Killed by: src/uclone_x/agent/turn_executor.py :: compaction = await self._auto_compact_if_needed(tool_defs, turn_extra_sections)
    Becomes: compaction = None
    """
    embedder = _FakeEmbedder(
        {"draw it": _toward({"alpha_image": 0.9}), "now mail it": _toward({"beta_mail": 0.9})}
    )
    wire = _Scripted([_call("alpha_image"), "one", "two"])
    agent = _agent(wire, ToolBinder(embedder), [])
    checks = _compact_on_check(agent, monkeypatch, 0)

    await agent.execute_turn("draw it")
    await agent.execute_turn("now mail it")

    assert len(checks) == 2, "expected one check at each turn start and none between steps"
    first, second, last = (_names(r) for r in wire.requests if r.tools)
    assert first == second == ["file_read", "search_tools", "alpha_image"]
    assert last == [*first, "beta_mail"]


@pytest.mark.asyncio
async def test_a_compaction_at_turn_start_binds_again_from_the_base_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """That request starts a new prefix anyway, so the old bound set is not carried over.

    Killed by: src/uclone_x/agent/turn_executor.py :: if compaction is not None and self._tools is not None:
    Becomes: if False:
    """
    embedder = _FakeEmbedder(
        {"draw it": _toward({"alpha_image": 0.9}), "now mail it": _toward({"beta_mail": 0.9})}
    )
    wire = _Scripted(["one", "two"])
    agent = _agent(wire, ToolBinder(embedder), [])
    _compact_on_check(agent, monkeypatch, 2)

    await agent.execute_turn("draw it")
    await agent.execute_turn("now mail it")

    first, second = (_names(r) for r in wire.requests if r.tools)
    assert first == ["file_read", "search_tools", "alpha_image"]
    assert second == ["file_read", "search_tools", "beta_mail"]


@pytest.mark.asyncio
async def test_a_compaction_at_turn_start_lets_a_pinned_session_bind_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One embedder failure does not pin every tool for the rest of the session.

    Killed by: src/uclone_x/agent/compaction_driver.py :: live.tools_pin_all = False
    Becomes: pass
    """
    embedder = _FakeEmbedder({"now mail it": _toward({"beta_mail": 0.9})})
    embedder.fail = True
    wire = _Scripted(["one", "two"])
    agent = _agent(wire, ToolBinder(embedder), [])
    _compact_on_check(agent, monkeypatch, 2)

    await agent.execute_turn("draw it")
    embedder.fail = False
    await agent.execute_turn("now mail it")

    first, second = (_names(r) for r in wire.requests if r.tools)
    assert first == sorted(["file_read", *_AXES])
    assert second == ["file_read", "search_tools", "beta_mail"]


# --------------------------------------------------------------------------------------
# Where binding does not apply: pin every held tool
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failing_embedder_pins_every_tool_for_the_session_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed bind pins all held tools, and a later working bind does not shrink them.

    The failure is logged once, by exception type: the embedder's own text (a URL and a
    server's words) reaches neither the log nor the model.

    Killed by: src/uclone_x/agent/tool_invoker.py :: live.tools_pin_all = True
    Becomes: live.tools_pin_all = False
    """
    embedder = _FakeEmbedder(
        {"draw it": _toward({"alpha_image": 0.9}), "again": _toward({"alpha_image": 0.9})}
    )
    embedder.fail = True
    wire = _Scripted(["one", "two", "three"])
    agent = _agent(wire, ToolBinder(embedder), [])
    everything = [
        "alpha_image",
        "beta_mail",
        "delta_web",
        "epsilon_note",
        "file_read",
        "gamma_calc",
    ]

    with caplog.at_level(logging.WARNING, logger="uclone_x.tools.tool_binder"):
        await agent.execute_turn("draw it")
        await agent.execute_turn("again")
        embedder.fail = False
        await agent.execute_turn("draw it")

    assert [_names(r) for r in wire.requests] == [everything, everything, everything]
    warnings = [r for r in caplog.records if r.name == "uclone_x.tools.tool_binder"]
    assert len(warnings) == 1
    assert "ConnectionError" in warnings[0].getMessage()
    assert "secret-detail" not in caplog.text and "10.0.0.7" not in caplog.text
    assert all("secret-detail" not in (m.content or "") for m in wire.requests[-1].messages)


@pytest.mark.asyncio
async def test_the_failure_warning_is_not_repeated_by_one_binder() -> None:
    """One binder serves every seat of a room, and each seat's session pins after its own
    failed bind, so the binder fails once per seat. It still logs once. The agent-level
    test above cannot see this: its one session never binds a second time.

    Killed by: src/uclone_x/tools/tool_binder.py :: if not self._failure_logged:
    Becomes: if True:
    """
    embedder = _FakeEmbedder({})
    embedder.fail = True
    binder = ToolBinder(embedder)
    catalog = [ToolDefinition(name="alpha_image", description="d", parameters={})]
    records: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    log = logging.getLogger("uclone_x.tools.tool_binder")
    handler = _Keep(level=logging.WARNING)
    log.addHandler(handler)
    try:
        assert await binder.bind("a", catalog) is None
        assert await binder.bind("b", catalog) is None
    finally:
        log.removeHandler(handler)

    assert len(records) == 1


def test_only_a_local_provider_with_an_endpoint_gets_a_binder() -> None:
    """Every other host pins every tool: no binder, and no lexical stand-in for one.

    Killed by: src/uclone_x/tools/tool_binder.py :: if provider.strip().lower() not in LOCAL_BINDING_PROVIDERS or not base_url.strip():
    Becomes: if not base_url.strip():
    """
    urls: list[str] = []

    def make(base_url: str) -> _FakeEmbedder:
        urls.append(base_url)
        return _FakeEmbedder({})

    assert isinstance(tool_binder_for("ollama", "http://127.0.0.1:11434", make), ToolBinder)
    assert isinstance(tool_binder_for(" Ollama ", " http://127.0.0.1:11434 ", make), ToolBinder)
    assert tool_binder_for("ollama", "", make) is None
    for provider in ("anthropic", "openai", "gemini", "vllm", "mock", ""):
        assert tool_binder_for(provider, "http://127.0.0.1:8000", make) is None
    # Only the two binding hosts built an embedder.
    assert urls == ["http://127.0.0.1:11434", "http://127.0.0.1:11434"]


def test_an_embedder_that_cannot_be_built_pins_every_tool() -> None:
    def broken(base_url: str) -> _FakeEmbedder:
        raise ValueError("bad url")

    assert tool_binder_for("ollama", "not a url", broken) is None


@pytest.mark.asyncio
async def test_with_no_binder_every_held_tool_is_pinned_in_canonical_order() -> None:
    wire = _Scripted(["ok"])
    agent = _agent(wire, None, [])

    await agent.execute_turn("draw it")

    assert _names(wire.requests[0]) == sorted(["file_read", *_AXES])


# --------------------------------------------------------------------------------------
# F12: dispatch checks what was advertised
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_call_to_a_tool_the_request_did_not_declare_is_refused_in_plain_words() -> None:
    """`zeta_room` is held and allowed but not offered outside a room, so it was never
    shown, and it is no binding catalog tool either: it does not run.

    Killed by: src/uclone_x/agent/tool_execution.py :: and tc.name not in advertised
    Becomes: and tc.name in ()
    """
    embedder = _FakeEmbedder({"draw it": _toward({"alpha_image": 0.9})})
    ran: list[str] = []
    registry = _registry(ran)
    registry.register(_room_tool(ran))
    wire = _Scripted([_call("zeta_room"), "sorry"])
    agent = _agent(wire, ToolBinder(embedder), ran, registry=registry)

    result = await agent.execute_turn("draw it")

    assert "zeta_room" not in _names(wire.requests[0])
    assert ran == []
    (record,) = result.tool_executions
    assert record.status is ToolResultStatus.ERROR
    assert record.error == unadvertised_tool_message("zeta_room")
    tool_message = next(m for m in wire.requests[1].messages if m.tool_call_id == "c1")
    text = tool_message.content or ""
    assert "not available this turn" in text
    for internal in ("Traceback", "binder", "allowed_tools", "Error", "sess_"):
        assert internal not in text
    assert _names(wire.requests[1]) == _names(wire.requests[0])


class _PreToolRecorder(BaseHook):
    """A hook that keeps the name of every tool it was asked about, and allows it."""

    def __init__(self) -> None:
        super().__init__(name="pre_tool_recorder")
        self.asked: list[str] = []

    async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
        self.asked.append(str(context.payload.get("tool_name")))
        return HookDecision(action=HookAction.ALLOW)


@pytest.mark.asyncio
async def test_an_unadvertised_call_is_refused_before_any_pre_tool_hook_sees_it() -> None:
    """No approval is asked for a call that was never going to run.

    Killed by: src/uclone_x/agent/tool_execution.py :: and tc.name not in advertised
    Becomes: and tc.name in ()
    """
    embedder = _FakeEmbedder({"draw it": _toward({"alpha_image": 0.9})})
    ran: list[str] = []
    registry = _registry(ran)
    registry.register(_room_tool(ran))
    recorder = _PreToolRecorder()
    calls = (
        ToolCallRequest(id="c1", name="zeta_room", arguments={"text": "x"}),
        ToolCallRequest(id="c2", name="alpha_image", arguments={"text": "x"}),
    )
    wire = _Scripted([calls, "done"])
    agent = _agent(wire, ToolBinder(embedder), ran, hooks=(recorder,), registry=registry)

    await agent.execute_turn("draw it")

    assert ran == ["alpha_image"]
    assert recorder.asked == ["alpha_image"]


@pytest.mark.asyncio
async def test_a_refusal_for_another_reason_keeps_its_own_words() -> None:
    """A name outside `allowed_tools` is refused by that check, not reported as unadvertised."""
    ran: list[str] = []
    wire = _Scripted([_call("gamma_calc"), "sorry"])
    agent = _agent(wire, None, ran, allowed=("file_read",))

    result = await agent.execute_turn("calc")

    (record,) = result.tool_executions
    assert ran == []
    assert "allowed_tools" in (record.error or "")


@pytest.mark.asyncio
async def test_a_direct_call_is_not_held_to_any_request() -> None:
    """`execute_tool_call` is made by code, not by a model reading a request: it still runs."""
    ran: list[str] = []
    agent = _agent(_Scripted([]), ToolBinder(_FakeEmbedder({})), ran)

    record = await agent.execute_tool_call("gamma_calc", {"text": "x"})

    assert record.status is ToolResultStatus.SUCCESS
    assert ran == ["gamma_calc"]


# --------------------------------------------------------------------------------------
# search_tools: the catalog route for what binding missed (#1545 item 2)
# --------------------------------------------------------------------------------------


def _search(query: str, call_id: str = "s1") -> tuple[ToolCallRequest, ...]:
    return (ToolCallRequest(id=call_id, name="search_tools", arguments={"query": query}),)


def _tool_text(request: LLMRequest, call_id: str) -> str:
    message = next(m for m in request.messages if m.tool_call_id == call_id)
    return message.content or ""


@pytest.mark.asyncio
async def test_a_search_hit_is_declared_from_the_next_step_and_runs() -> None:
    """Binding missed `beta_mail`; the model searches, and its next step can call it.

    The hit is appended after what the turn already sent, so the tools layer only grows,
    and the result lists names and one-line descriptions, never a schema.

    Killed by: src/uclone_x/agent/tool_invoker.py :: tool_defs[len(tool_defs) :] = self.session_tools_layer(live)[len(tool_defs) :]
    Becomes: pass
    """
    embedder = _FakeEmbedder(
        {
            "draw it and mail it": _toward({"alpha_image": 0.9}),
            "send mail": _toward({"beta_mail": 0.9}),
        }
    )
    ran: list[str] = []
    wire = _Scripted([_search("send mail"), _call("beta_mail", "c2"), "done"])
    agent = _agent(wire, ToolBinder(embedder), ran)

    result = await agent.execute_turn("draw it and mail it")

    first, second, third = (_names(r) for r in wire.requests)
    assert first == ["file_read", "search_tools", "alpha_image"]
    assert second == [*first, "beta_mail"]
    assert third == second
    assert ran == ["beta_mail"]
    text = _tool_text(wire.requests[1], "s1")
    assert "beta_mail: The beta_mail tool." in text
    assert "properties" not in text
    assert [r.status for r in result.tool_executions] == [
        ToolResultStatus.SUCCESS,
        ToolResultStatus.SUCCESS,
    ]


@pytest.mark.asyncio
async def test_a_search_hit_stays_bound_for_the_next_user_message() -> None:
    """What a search found joins the grow-only set, like a binding does."""
    embedder = _FakeEmbedder(
        {
            "draw it": _toward({"alpha_image": 0.9}),
            "send mail": _toward({"beta_mail": 0.9}),
            "thanks": _toward({}),
        }
    )
    wire = _Scripted([_search("send mail"), "found it", "you're welcome"])
    agent = _agent(wire, ToolBinder(embedder), [])

    await agent.execute_turn("draw it")
    await agent.execute_turn("thanks")

    assert _names(wire.requests[2]) == ["file_read", "search_tools", "alpha_image", "beta_mail"]


@pytest.mark.asyncio
async def test_a_call_made_beside_the_search_that_found_it_is_bound_on_that_call() -> None:
    """R3 (#2190): the search's hit is a catalog tool in range, so the call beside it runs
    instead of being refused, and the hit is declared once, from the next request."""
    embedder = _FakeEmbedder({"go": _toward({}), "send mail": _toward({"beta_mail": 0.9})})
    ran: list[str] = []
    both = (*_search("send mail"), *_call("beta_mail", "c2"))
    wire = _Scripted([both, "done"])
    agent = _agent(wire, ToolBinder(embedder), ran)

    result = await agent.execute_turn("go")

    assert ran == ["beta_mail"]
    ran_record = next(r for r in result.tool_executions if r.tool_name == "beta_mail")
    assert ran_record.status is ToolResultStatus.SUCCESS
    assert _names(wire.requests[1]) == ["file_read", "search_tools", "beta_mail"]


@pytest.mark.asyncio
async def test_a_search_that_matches_nothing_says_so_and_binds_nothing() -> None:
    embedder = _FakeEmbedder({"go": _toward({}), "fly a kite": _toward({})})
    wire = _Scripted([_search("fly a kite"), "no tool for that"])
    agent = _agent(wire, ToolBinder(embedder), [])

    await agent.execute_turn("go")

    assert _names(wire.requests[1]) == _names(wire.requests[0]) == ["file_read", "search_tools"]
    assert "No tool matched" in _tool_text(wire.requests[1], "s1")


@pytest.mark.asyncio
async def test_a_failed_search_is_a_plain_tool_error_with_no_internals() -> None:
    """The embedder dies mid-turn: the search fails in plain words and the turn goes on.

    Nothing of the exception reaches the model or the person: no URL, address, class name
    or server text.

    Killed by: src/uclone_x/agent/tool_invoker.py :: raise search_unavailable()
    Becomes: raise RuntimeError(str(hits))
    """
    embedder = _FakeEmbedder({"go": _toward({"alpha_image": 0.9})})
    embedder.fail_texts = {"send mail"}
    wire = _Scripted([_search("send mail"), "sorry"])
    agent = _agent(wire, ToolBinder(embedder), [])

    result = await agent.execute_turn("go")

    assert result.content == "sorry"
    (record,) = result.tool_executions
    assert record.status is ToolResultStatus.ERROR
    assert record.error == SEARCH_UNAVAILABLE_MESSAGE
    text = _tool_text(wire.requests[1], "s1")
    assert text == SEARCH_UNAVAILABLE_MESSAGE
    for internal in (
        "http",
        "10.0.0.7",
        "11434",
        "ConnectionError",
        "Error",
        "secret",
        "/",
        "Traceback",
    ):
        assert internal not in text
    assert _names(wire.requests[1]) == _names(wire.requests[0])


@pytest.mark.asyncio
async def test_search_tools_is_offered_only_where_the_tools_layer_binds() -> None:
    """No search where there is nothing to bind: a range of base tools only, or no binder.

    A range that is all base has an empty catalog, so it pins what it holds, as before
    Revision 3, and never embeds the message.

    Killed by: src/uclone_x/agent/tool_invoker.py :: catalog = self.binding_catalog(defs)
    Becomes: catalog = self.binding_catalog(defs) or defs
    """
    embedder = _FakeEmbedder({"hi": _toward({})})
    wire = _Scripted(["a", "b"])
    all_base = _agent(wire, ToolBinder(embedder), [], allowed=("file_read", "load_skill"))
    unbound = _agent(wire, None, [])

    await all_base.execute_turn("hi")
    await unbound.execute_turn("hi")

    assert _names(wire.requests[0]) == ["file_read"]
    assert embedder.query_calls == 0
    assert "search_tools" not in _names(wire.requests[1])


# --------------------------------------------------------------------------------------
# Restore: a restarted process reseeds the bound set from history (#1545 item 3)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_restored_session_still_declares_the_catalog_tools_its_history_called(
    tmp_path: Path,
) -> None:
    """After a restart, "do it again" can repeat a call to a tool binding had added.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: self._tool_invoker.reseed_bound_tools_from_history(hydrated)
    Becomes: None
    """
    store = SessionStore(storage_dir=tmp_path)
    queries = {"draw it": _toward({"alpha_image": 0.9}), "do it again": _toward({})}
    ran: list[str] = []
    before = _agent(
        _Scripted([_call("alpha_image"), "drawn"]),
        ToolBinder(_FakeEmbedder(queries)),
        ran,
        store=store,
    )
    await before.execute_turn("draw it")
    before.persist_session()

    wire = _Scripted([_call("alpha_image", "c2"), "drawn again"])
    after = _agent(wire, ToolBinder(_FakeEmbedder(queries)), ran, store=store)
    assert after.hydrate_session() is not None
    await after.execute_turn("do it again")

    assert _names(wire.requests[0]) == ["file_read", "search_tools", "alpha_image"]
    assert ran == ["alpha_image", "alpha_image"]


@pytest.mark.asyncio
async def test_loaded_history_reseeds_catalog_calls_in_first_call_order_skipping_base() -> None:
    """First-call order, each name once; a base tool or one no longer held is not bound.

    Killed by: src/uclone_x/agent/tool_invoker.py :: if call.name in catalog and call.name not in bound:
    Becomes: if call.name not in bound:
    """
    wire = _Scripted(["ok"])
    agent = _agent(wire, ToolBinder(_FakeEmbedder({"again": _toward({})})), [])

    def called(*names: str) -> ChatMessage:
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=tuple(
                ToolCallRequest(id=f"h{i}", name=n, arguments={}) for i, n in enumerate(names)
            ),
        )

    agent.load_history(
        [
            ChatMessage(role=MessageRole.USER, content="earlier"),
            called("gamma_calc", "file_read", "retired_tool"),
            called("beta_mail", "gamma_calc"),
            ChatMessage(role=MessageRole.ASSISTANT, content="done"),
        ]
    )
    # The set itself, not only the request: the layer would hide a base or unheld name.
    live = agent._live_session("sess_binder")  # pyright: ignore[reportPrivateUsage]
    assert live.bound_tools == ["gamma_calc", "beta_mail"]
    await agent.execute_turn("again")

    assert _names(wire.requests[0]) == ["file_read", "search_tools", "gamma_calc", "beta_mail"]


@pytest.mark.asyncio
async def test_a_restored_session_binds_only_what_the_persona_s_tool_list_allows() -> None:
    """A persona whose `allowed_tools` is a list: history outside that range is not bound (#1775).

    Killed by: src/uclone_x/agent/tool_invoker.py :: held = [t.name for t in self.held_tools()]
    Becomes: held = [t.name for t in self._registry.list_tools()] if self._registry is not None else []
    """
    wire = _Scripted(["ok"])
    agent = _agent(wire, ToolBinder(_FakeEmbedder({"again": _toward({})})), [])
    agent.define_persona(
        PersonaDefinition(
            name="illustrator",
            role="Illustrator",
            system_prompt="You draw.",
            allowed_tools=("alpha_image", "gamma_calc"),
            enable_write_tools=True,  # the fake tools all declare that they write
        )
    )
    agent.persona = "illustrator"

    agent.load_history(
        [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=tuple(
                    ToolCallRequest(id=f"h{i}", name=n, arguments={})
                    for i, n in enumerate(("beta_mail", "gamma_calc", "alpha_image"))
                ),
            )
        ]
    )
    live = agent._live_session("sess_binder")  # pyright: ignore[reportPrivateUsage]
    assert live.bound_tools == ["gamma_calc", "alpha_image"]
    await agent.execute_turn("again")

    assert "beta_mail" not in _names(wire.requests[0])
    assert {"gamma_calc", "alpha_image"} <= set(_names(wire.requests[0]))


@pytest.mark.asyncio
async def test_a_restored_session_of_a_pinning_agent_binds_nothing() -> None:
    """With no binder every held tool is pinned anyway: the history changes nothing."""
    wire = _Scripted(["ok"])
    agent = _agent(wire, None, [])
    agent.load_history(
        [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(ToolCallRequest(id="h", name="gamma_calc", arguments={}),),
            )
        ]
    )

    await agent.execute_turn("again")

    assert _names(wire.requests[0]) == sorted(["file_read", *_AXES])


def test_is_short_follow_up_identifies_retry_and_continuation_patterns() -> None:
    """Short retry/continue directives are recognized in Korean and English (#2168).

    Killed by: src/uclone_x/agent/tool_invoker.py :: return len(cleaned) <= 30 and bool(_FOLLOW_UP_SHORT_PATTERNS.match(cleaned))
    Becomes: return False
    """
    from uclone_x.agent.tool_invoker import is_short_follow_up

    # Positive matches
    assert is_short_follow_up("다시")
    assert is_short_follow_up("다시 해봐")
    assert is_short_follow_up("다시 해줘!")
    assert is_short_follow_up("다시 그려줘")
    assert is_short_follow_up("계속해줘")
    assert is_short_follow_up("재시도")
    assert is_short_follow_up("한번 더")
    assert is_short_follow_up("redo")
    assert is_short_follow_up("retry")
    assert is_short_follow_up("again!")
    assert is_short_follow_up("try again")

    # Negative non-matches
    assert not is_short_follow_up("해변에서 피오나는 비키니를 입고있었어")
    assert not is_short_follow_up("피오나가 탐정인데 살인사건을 조사하는 스토리는 어떄?")
    assert not is_short_follow_up("what is the weather today in seoul?")


@pytest.mark.asyncio
async def test_tools_for_turn_retains_last_turn_tool_on_short_follow_up() -> None:
    """When a short follow-up arrives, tools from last_turn_tool_calls are retained (#2168).

    Killed by: src/uclone_x/agent/tool_invoker.py :: if is_short_follow_up(message):
    Becomes: if False:
    """
    wire = _Scripted(["drawn", "done"])
    ran: list[str] = []
    agent = _agent(wire, ToolBinder(_FakeEmbedder({"again": _toward({})})), ran)

    # In room turns, checkpoint_turn clears last_turn_tool_calls to []; the previous turn's
    # assistant message in history retains the tool call (#2168).
    call = ToolCallRequest(id="c1", name="alpha_image", arguments={})
    agent.load_history(
        [
            ChatMessage(role=MessageRole.USER, content="draw a cat"),
            ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=(call,)),
        ],
        session_id="sess_binder",
    )
    live = agent._live_session("sess_binder")  # pyright: ignore[reportPrivateUsage]
    live.last_turn_tool_calls = []
    live.bound_tools.clear()

    # "다시 해봐" produces 0 embedding hits, but retains alpha_image from recent messages
    defs = await agent._tool_invoker.tools_for_turn("다시 해봐", live)  # pyright: ignore[reportPrivateUsage]

    names = [d.name for d in defs]
    assert "alpha_image" in names
    assert "alpha_image" in live.bound_tools


@pytest.mark.asyncio
async def test_reseed_bound_tools_post_compaction_retains_recent_catalog_calls() -> None:
    """Post-compaction reseeding keeps tools from recent turns and drops older ones (#2168).

    Killed by: src/uclone_x/agent/tool_invoker.py :: turn_count > keep_recent_turns:
    Becomes: turn_count > 0:
    """
    wire = _Scripted(["ok"])
    ran: list[str] = []
    agent = _agent(wire, ToolBinder(_FakeEmbedder({})), ran)

    def msg_with_call(role: MessageRole, name: str) -> ChatMessage:
        return ChatMessage(
            role=role,
            content=None,
            tool_calls=(ToolCallRequest(id="c", name=name, arguments={}),),
        )

    messages = [
        ChatMessage(role=MessageRole.USER, content="turn 1 (old)"),
        msg_with_call(MessageRole.ASSISTANT, "gamma_calc"),
        ChatMessage(role=MessageRole.USER, content="turn 2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        ChatMessage(role=MessageRole.USER, content="turn 3"),
        msg_with_call(MessageRole.ASSISTANT, "alpha_image"),
        ChatMessage(role=MessageRole.USER, content="turn 4 (recent)"),
        ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
    ]
    agent.load_history(messages, session_id="sess_binder")
    live = agent._live_session("sess_binder")  # pyright: ignore[reportPrivateUsage]
    live.bound_tools.clear()

    # Reseed with keep_recent_turns=2: only alpha_image is within the last 2 user turns
    agent._tool_invoker.reseed_bound_tools_post_compaction(live, keep_recent_turns=2)  # pyright: ignore[reportPrivateUsage]

    assert live.bound_tools == ["alpha_image"]
    assert "gamma_calc" not in live.bound_tools


# --------------------------------------------------------------------------------------
# Binding recall (#2190): clauses, hybrid ranking, dedupe, bind on call, did-you-mean
# --------------------------------------------------------------------------------------


def _catalog(*names: str) -> list[ToolDefinition]:
    return [ToolDefinition(name=n, description=f"The {n} tool.", parameters={}) for n in names]


@pytest.mark.asyncio
async def test_each_clause_of_a_two_part_request_binds_its_own_tools() -> None:
    """R1: the whole message leans to pictures and web and would bind three of those; the
    mail clause binds `beta_mail` anyway, the web clause binds its top two (not three), and
    the whole message adds its first. The three queries are embedded in one call.

    Killed by: src/uclone_x/tools/tool_binder.py :: return tuple(dedupe(interleave([*lists, whole[:1]]), message))
    Becomes: return tuple(dedupe(whole, message)[: self._top_k])
    """
    message = "look up the weather, then send a mail"
    embedder = _FakeEmbedder(
        {
            message: _toward({"alpha_image": 0.8, "delta_web": 0.4, "gamma_calc": 0.36}),
            "look up the weather": _toward(
                {"delta_web": 0.8, "epsilon_note": 0.4, "gamma_calc": 0.36}
            ),
            "send a mail": _toward({"beta_mail": 0.9}),
        }
    )
    binder = ToolBinder(embedder)

    bound = await binder.bind(message, _catalog(*_AXES))

    assert bound == ("delta_web", "beta_mail", "alpha_image", "epsilon_note")
    assert embedder.query_calls == 3


@pytest.mark.asyncio
async def test_a_clause_binds_at_most_its_per_clause_budget() -> None:
    """Killed by: src/uclone_x/tools/tool_binder.py :: dedupe(self._fused(clause, vector, catalog), clause)[: self._per_clause]
    Becomes: dedupe(self._fused(clause, vector, catalog), clause)
    """
    message = "look up the weather, then send a mail"
    embedder = _FakeEmbedder(
        {
            message: _toward({"beta_mail": 0.5}),
            "look up the weather": _toward(
                {"delta_web": 0.7, "epsilon_note": 0.5, "gamma_calc": 0.45}
            ),
            "send a mail": _toward({"beta_mail": 0.9}),
        }
    )

    bound = await ToolBinder(embedder).bind(message, _catalog(*_AXES))

    assert bound == ("delta_web", "beta_mail", "epsilon_note")


@pytest.mark.asyncio
async def test_bm25_reorders_the_tools_the_embedding_already_found() -> None:
    """R1 hybrid: `delta_web` is a little closer by embedding, but only `alpha_image` shares
    a word with the message, so fusion puts it first.

    Killed by: src/uclone_x/tools/tool_binder.py :: return [name for name in rrf([dense, lexical]) if name in eligible]
    Becomes: return dense
    """
    embedder = _FakeEmbedder({"an alpha sketch": _toward({"delta_web": 0.62, "alpha_image": 0.6})})

    bound = await ToolBinder(embedder, top_k=1).bind("an alpha sketch", _catalog(*_AXES))

    assert bound == ("alpha_image",)


@pytest.mark.asyncio
async def test_bm25_never_binds_a_tool_the_embedding_left_below_the_floor() -> None:
    """A shared word does not make a tool eligible: restraint stays the embedding's.

    Killed by: src/uclone_x/tools/tool_binder.py :: return [name for name in rrf([dense, lexical]) if name in eligible]
    Becomes: return rrf([dense, lexical])
    """
    embedder = _FakeEmbedder({"alpha and beta, thanks": _toward({"delta_web": 0.5})})

    bound = await ToolBinder(embedder).bind("alpha and beta, thanks", _catalog(*_AXES))

    assert bound == ("delta_web",)


class _NamedEmbedder:
    """Embeds a tool's text by its name and a message by table, over named axes."""

    def __init__(self, axes: Sequence[str], queries: dict[str, dict[str, float]]) -> None:
        self.axes = tuple(axes)
        self.queries = queries

    @property
    def model_name(self) -> str:
        return "fake-embed"

    @property
    def dimensions(self) -> int:
        return len(self.axes) + 1

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        out: list[tuple[float, ...]] = []
        for text in texts:
            if text in self.queries:
                scores = self.queries[text]
                vector = [scores.get(a, 0.0) for a in self.axes]
                out.append((*vector, math.sqrt(1.0 - sum(v * v for v in vector))))
            else:
                name = text.split(":", 1)[0]
                out.append((*(1.0 if a == name else 0.0 for a in self.axes), 0.0))
        return tuple(out)


_ISSUES = ("mcp__github__create_issue", "mcp__gitlab__create_issue", "mcp__email__send")


@pytest.mark.asyncio
async def test_one_tool_per_action_across_servers_unless_the_message_names_the_server() -> None:
    """R2: GitLab's `create_issue` would take a slot from the mail tool; it is bound only
    when the message names GitLab.

    Killed by: src/uclone_x/tools/tool_binder.py :: return tuple(dedupe(whole, message)[: self._top_k])
    Becomes: return tuple(whole[: self._top_k])
    """
    leaning = {_ISSUES[0]: 0.6, _ISSUES[1]: 0.59, _ISSUES[2]: 0.5}
    plain, named = "file it for acme", "file it on github and gitlab"
    embedder = _NamedEmbedder(_ISSUES, {plain: leaning, named: leaning})
    catalog = _catalog(*_ISSUES)

    assert await ToolBinder(embedder, top_k=2).bind(plain, catalog) == (
        "mcp__github__create_issue",
        "mcp__email__send",
    )
    assert await ToolBinder(embedder, top_k=2).bind(named, catalog) == _ISSUES[:2]


@pytest.mark.asyncio
async def test_a_catalog_tool_binding_missed_is_bound_on_the_call_and_runs() -> None:
    """R3: `gamma_calc` is held, in range and not bound; the call that names it runs it,
    and its schema is appended after everything the turn already sent.

    Killed by: src/uclone_x/agent/tool_execution.py :: and not self._tool_invoker.bind_on_call(tc.name, tool_ctx.session_id)
    Becomes: and True
    """
    embedder = _FakeEmbedder({"draw it": _toward({"alpha_image": 0.9})})
    ran: list[str] = []
    wire = _Scripted([_call("gamma_calc"), "done"])
    agent = _agent(wire, ToolBinder(embedder), ran)

    result = await agent.execute_turn("draw it")

    assert ran == ["gamma_calc"]
    (record,) = result.tool_executions
    assert record.status is ToolResultStatus.SUCCESS
    first, second = (_names(r) for r in wire.requests)
    assert first == ["file_read", "search_tools", "alpha_image"]
    assert second == [*first, "gamma_calc"]
    head = wire.requests[1].messages[: len(wire.requests[0].messages)]
    assert [m.model_dump_json() for m in head] == [
        m.model_dump_json() for m in wire.requests[0].messages
    ]


@pytest.mark.asyncio
async def test_a_tool_bound_on_a_call_is_declared_from_the_next_step() -> None:
    """Killed by: src/uclone_x/agent/tool_invoker.py :: if searched or bound_on_call:
    Becomes: if searched:
    """
    embedder = _FakeEmbedder({"draw it": _toward({"alpha_image": 0.9}), "and?": _toward({})})
    wire = _Scripted([_call("gamma_calc"), _call("gamma_calc", "c2"), "done", "ok"])
    ran: list[str] = []
    agent = _agent(wire, ToolBinder(embedder), ran)

    await agent.execute_turn("draw it")
    await agent.execute_turn("and?")

    expected = ["file_read", "search_tools", "alpha_image", "gamma_calc"]
    assert [_names(r) for r in wire.requests[1:]] == [expected] * 3
    assert ran == ["gamma_calc", "gamma_calc"]


@pytest.mark.asyncio
async def test_a_call_outside_the_range_is_refused_and_binds_nothing() -> None:
    """R3 binds only inside the clone's range: `gamma_calc` is registered but not allowed,
    so the call is refused by the range check, nothing runs and nothing is bound.

    Killed by: src/uclone_x/agent/tool_invoker.py :: if name not in {d.name for d in catalog}:
    Becomes: if False:
    """
    embedder = _FakeEmbedder({"draw it": _toward({"alpha_image": 0.9})})
    ran: list[str] = []
    wire = _Scripted([_call("gamma_calc"), "sorry"])
    agent = _agent(
        wire, ToolBinder(embedder), ran, allowed=("file_read", "alpha_image", "beta_mail")
    )

    result = await agent.execute_turn("draw it")

    assert ran == []
    (record,) = result.tool_executions
    assert record.status is ToolResultStatus.ERROR
    assert _names(wire.requests[1]) == _names(wire.requests[0])
    live = agent._live_session("sess_binder")  # pyright: ignore[reportPrivateUsage]
    assert "gamma_calc" not in live.bound_tools
    invoker = agent._tool_invoker  # pyright: ignore[reportPrivateUsage]
    assert invoker.bind_on_call("gamma_calc", "sess_binder") is False
    assert invoker.bind_on_call("file_read", "sess_binder") is False
    assert live.bound_tools == ["alpha_image"]


@pytest.mark.asyncio
async def test_an_agent_with_no_binder_binds_nothing_on_a_call() -> None:
    """With no binder every held tool is declared already, so there is nothing to bind.

    Killed by: src/uclone_x/agent/tool_invoker.py :: if live.tools_pin_all or self._binding is None:
    Becomes: if live.tools_pin_all:
    """
    agent = _agent(_Scripted([]), None, [])
    live = agent._live_session("sess_binder")  # pyright: ignore[reportPrivateUsage]
    invoker = agent._tool_invoker  # pyright: ignore[reportPrivateUsage]

    assert invoker.bind_on_call("beta_mail", "sess_binder") is False
    assert live.bound_tools == []


@pytest.mark.asyncio
async def test_a_misspelled_tool_name_is_told_the_closest_tools_in_plain_words() -> None:
    """R4: the result names the closest tools the agent may use, best first, and only
    those: a registered tool outside the range is never suggested.

    Killed by: src/uclone_x/agent/tool_invoker.py :: for d in self.advertised_tool_definitions()
    Becomes: for d in ()
    """
    embedder = _FakeEmbedder({"mail the team": _toward({"beta_mail": 0.9})})
    ran: list[str] = []
    wire = _Scripted([_call("beta_mial"), "sorry"])
    agent = _agent(wire, ToolBinder(embedder), ran, allowed=("file_read", "beta_mail", "delta_web"))

    result = await agent.execute_turn("mail the team")

    (record,) = result.tool_executions
    assert ran == []
    assert record.status is ToolResultStatus.ERROR
    assert record.error == (
        "There is no tool named 'beta_mial', so nothing was run. "
        "The closest tools you can use are: beta_mail, delta_web, file_read."
    )
    assert _tool_text(wire.requests[1], "c1") == record.error


@pytest.mark.asyncio
async def test_the_did_you_mean_result_carries_no_internals() -> None:
    """No agent id, session id, permission list, registry or exception text reaches it."""
    embedder = _FakeEmbedder({"go": _toward({})})
    wire = _Scripted([_call("mcp__nowhere__lookup"), "sorry"])
    agent = _agent(wire, ToolBinder(embedder), [], allowed=("file_read", "beta_mail"))

    await agent.execute_turn("go")

    text = _tool_text(wire.requests[1], "c1")
    assert text.startswith("There is no tool named 'mcp__nowhere__lookup'")
    for internal in (
        "binder",
        "sess_",
        "allowed_tools",
        "registry",
        "Traceback",
        "Error",
        "not found",
        "gamma_calc",
        "alpha_image",
    ):
        assert internal not in text


def test_closest_tool_names_reads_the_name_and_the_argument_names() -> None:
    """Killed by: src/uclone_x/tools/tool_ranking.py :: scored.append((ratio + overlap, index, candidate))
    Becomes: scored.append((ratio, index, candidate))
    """
    candidates = [
        ("mcp__github__get_pull_request", "Get one pull request.", ["repo", "number"]),
        ("mcp__email__send", "Send an email.", ["to", "subject", "body"]),
        ("mcp__email__draft", "Draft an email.", ["to", "subject", "body"]),
    ]

    assert closest_tool_names("mcp__github__pull_request", [], candidates, 1) == [
        "mcp__github__get_pull_request"
    ]
    # An invented name: by its name part alone `get_pull_request` is closest; the argument
    # names put the mail tools ahead of it.
    assert closest_tool_names("lookup", ["subject", "body"], candidates)[-1] == (
        "mcp__github__get_pull_request"
    )
    assert closest_tool_names("lookup", ["repo", "number"], candidates, 1) == [
        "mcp__github__get_pull_request"
    ]
    assert unknown_tool_message("x", []) == (
        "There is no tool named 'x', so nothing was run. Use only the tools listed in this request."
    )
