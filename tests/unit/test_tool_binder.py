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
        tools=_registry(ran),
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

    Killed by: src/uclone_x/agent/tool_invoker.py :: live.bound_tools.extend(sorted(set(hits) - set(live.bound_tools)))
    Becomes: live.bound_tools.extend(h for h in hits if h not in live.bound_tools)
    """
    embedder = _FakeEmbedder({"web and pictures": _toward({"delta_web": 0.8, "alpha_image": 0.5})})
    wire = _Scripted(["ok"])
    agent = _agent(wire, ToolBinder(embedder), [])

    await agent.execute_turn("web and pictures")

    assert _names(wire.requests[0]) == ["file_read", "search_tools", "alpha_image", "delta_web"]


@pytest.mark.asyncio
async def test_the_bound_set_only_grows_across_user_messages() -> None:
    """A later message appends; what an earlier one bound stays, in its place.

    Killed by: src/uclone_x/agent/tool_invoker.py :: live.bound_tools.extend(sorted(set(hits) - set(live.bound_tools)))
    Becomes: live.bound_tools[:] = sorted(set(hits))
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

    Killed by: src/uclone_x/agent/tool_invoker.py :: return req.model_copy(update={"messages": tuple(messages), "tools": tuple(tool_defs)})
    Becomes: return req.model_copy(update={"messages": tuple(messages), "tools": tuple(self.advertised_tool_definitions())})
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

    A turn checks at its start, and again after each step that ran tools.
    """
    checks: list[str] = []

    def should(sid: str, history: object, request: object = None) -> bool:
        checks.append(sid)
        return len(checks) == which

    monkeypatch.setattr(agent, "_should_compact_session", should)
    return checks


@pytest.mark.asyncio
async def test_a_compaction_between_steps_keeps_what_the_turn_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rest of the turn still sends `alpha_image`, so the next turn keeps it too.

    Killed by: src/uclone_x/agent/compaction_driver.py :: self._tool_invoker.reseed_bound_tools(live, tools, pinned=pinned)
    Becomes: pass
    """
    embedder = _FakeEmbedder(
        {"draw it": _toward({"alpha_image": 0.9}), "now mail it": _toward({"beta_mail": 0.9})}
    )
    wire = _Scripted([_call("alpha_image"), "one", "two"])
    agent = _agent(wire, ToolBinder(embedder), [])
    checks = _compact_on_check(agent, monkeypatch, 2)

    await agent.execute_turn("draw it")
    await agent.execute_turn("now mail it")

    assert len(checks) == 3, "expected a check at each turn start and one between steps"
    first, _, last = (_names(r) for r in wire.requests if r.tools)
    assert first == ["file_read", "search_tools", "alpha_image"]
    assert last == [*first, "beta_mail"]


@pytest.mark.asyncio
async def test_a_compaction_between_steps_keeps_a_pinned_session_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: src/uclone_x/agent/tool_invoker.py :: live.tools_pin_all = pinned
    Becomes: live.tools_pin_all = False
    """
    embedder = _FakeEmbedder({"now mail it": _toward({"beta_mail": 0.9})})
    embedder.fail = True
    wire = _Scripted([_call("alpha_image"), "one", "two"])
    agent = _agent(wire, ToolBinder(embedder), [])
    _compact_on_check(agent, monkeypatch, 2)

    await agent.execute_turn("draw it")
    embedder.fail = False
    await agent.execute_turn("now mail it")

    everything = sorted(["file_read", *_AXES])
    assert [sorted(_names(r)) for r in wire.requests if r.tools] == [everything] * 3
    # Still pinned, not re-bound. The re-seeded set alone would hide this: it holds the
    # whole catalog.
    assert embedder.query_calls == 0


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
    """`gamma_calc` is held and allowed but not bound, so it was never shown: it does not run.

    Killed by: src/uclone_x/agent/tool_execution.py :: and tc.name not in advertised
    Becomes: and tc.name in ()
    """
    embedder = _FakeEmbedder({"draw it": _toward({"alpha_image": 0.9})})
    ran: list[str] = []
    wire = _Scripted([_call("gamma_calc"), "sorry"])
    agent = _agent(wire, ToolBinder(embedder), ran)

    result = await agent.execute_turn("draw it")

    assert "gamma_calc" not in _names(wire.requests[0])
    assert ran == []
    (record,) = result.tool_executions
    assert record.status is ToolResultStatus.ERROR
    assert record.error == unadvertised_tool_message("gamma_calc")
    tool_message = next(m for m in wire.requests[1].messages if m.tool_call_id == "c1")
    text = tool_message.content or ""
    assert "not available this turn" in text
    for internal in ("Traceback", "binder", "allowed_tools", "Error", "sess_"):
        assert internal not in text


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
    recorder = _PreToolRecorder()
    calls = (
        ToolCallRequest(id="c1", name="gamma_calc", arguments={"text": "x"}),
        ToolCallRequest(id="c2", name="alpha_image", arguments={"text": "x"}),
    )
    wire = _Scripted([calls, "done"])
    agent = _agent(wire, ToolBinder(embedder), ran, hooks=(recorder,))

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
async def test_a_call_made_beside_the_search_that_found_it_is_still_refused() -> None:
    """F12 holds: a name the step's own request did not declare does not run."""
    embedder = _FakeEmbedder({"go": _toward({}), "send mail": _toward({"beta_mail": 0.9})})
    ran: list[str] = []
    both = (*_search("send mail"), *_call("beta_mail", "c2"))
    wire = _Scripted([both, "done"])
    agent = _agent(wire, ToolBinder(embedder), ran)

    result = await agent.execute_turn("go")

    assert ran == []
    refused = next(r for r in result.tool_executions if r.tool_name == "beta_mail")
    assert refused.error == unadvertised_tool_message("beta_mail")


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
