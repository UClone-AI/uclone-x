# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Unit tests for Principle 6 (Fail-Fast & Zero Silent Fallbacks) Checks 4 & 5.

Falsifiable Checks:
4. For every result with path != 'primary', a matching PROVIDER_FAILOVER event exists
   on the bus with a lower (priority, sequence) than the result event.
5. For every result with path == 'failover', each attempts[*].span_id resolves to an
   emitted 'failover.event' span in telemetry.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import cast

import httpx
import pytest
from provenance_ast import (
    ConstructionKind,
    PathVerdict,
    constructed_path,
    construction_kind,
    provenance_binding_names,
)

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
    require_provenance,
)
from uclone_x.engine.event_bus import (
    AgentEvent,
    EventBus,
    EventSource,
    EventType,
)
from uclone_x.errors import LLMProviderError, MissingProvenanceError
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.telemetry import SpanStatus, TelemetryTracer
from uclone_x.ui.app import (
    AgentSessionManager,
    create_ui_app,
)


@pytest.mark.asyncio
async def test_p6_check4_provider_failover_notice_ordered_before_result(tmp_path: Path) -> None:
    """Check 4: PROVIDER_FAILOVER event exists on bus with lower (priority, sequence) than result."""
    from unittest.mock import AsyncMock, MagicMock

    from uclone_x.llm.protocols import LLMProviderProtocol

    bus = EventBus()
    tracer = TelemetryTracer()
    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(side_effect=LLMProviderError("Primary provider unavailable"))
    mgr = AgentSessionManager(bus=bus, tracer=tracer, llm=failing_llm)

    app = create_ui_app(static_dir=tmp_path, bus=bus, tracer=tracer, session_manager=mgr)
    sub = bus.subscribe("agent.chat.*")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        resp = await client.post("/api/turn", json={"message": "ping", "agent_id": "test-agent"})
    assert resp.status_code == 200

    # Collect all events on agent.chat.*
    events: list[AgentEvent] = []
    for _ in range(3):
        events.append(await asyncio.wait_for(sub.get(), timeout=2.0))

    evt_input = next(e for e in events if e.type is EventType.USER_INPUT)
    evt_failover = next(e for e in events if e.type is EventType.PROVIDER_FAILOVER)
    evt_reply = next(e for e in events if e.type is EventType.AGENT_REPLY)

    # 1. Event types
    assert evt_input.type is EventType.USER_INPUT
    assert evt_failover.type is EventType.PROVIDER_FAILOVER
    assert evt_reply.type is EventType.AGENT_REPLY

    # 2. Strict total order: notice before result
    assert evt_failover < evt_reply
    assert evt_failover.sequence < evt_reply.sequence
    assert evt_failover.priority <= evt_reply.priority
    assert evt_input.sequence < evt_failover.sequence

    # 3. Payload and attribution truth
    assert evt_failover.payload["requested_provider"] == "test-agent"
    assert evt_failover.payload["served_provider"] == "agent.core"
    assert evt_failover.payload["error_class"] == "LLMProviderError"


@pytest.mark.asyncio
async def test_p6_check5_failover_event_span_correlation_with_attempt_record(
    tmp_path: Path,
) -> None:
    """Check 5: For every failover result, attempts[*].span_id resolves to an emitted failover.event span."""
    from unittest.mock import AsyncMock, MagicMock

    from uclone_x.llm.protocols import LLMProviderProtocol

    bus = EventBus()
    tracer = TelemetryTracer()
    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(
        side_effect=ConnectionRefusedError("Offline connection refused")
    )
    mgr = AgentSessionManager(bus=bus, tracer=tracer, llm=failing_llm)

    app = create_ui_app(static_dir=tmp_path, bus=bus, tracer=tracer, session_manager=mgr)
    sub = bus.subscribe("agent.chat.*")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        resp = await client.post(
            "/api/turn", json={"message": "verify span", "agent_id": "span-agent"}
        )
    assert resp.status_code == 200

    events: list[AgentEvent] = []
    for _ in range(3):
        events.append(await asyncio.wait_for(sub.get(), timeout=2.0))

    evt_failover = next(e for e in events if e.type is EventType.PROVIDER_FAILOVER)
    evt_reply = next(e for e in events if e.type is EventType.AGENT_REPLY)

    # Query completed telemetry spans
    spans = tracer.get_completed_spans()
    failover_spans = [s for s in spans if s.name == "failover.event"]
    assert len(failover_spans) == 1
    failover_span = failover_spans[0]

    assert failover_span.status is SpanStatus.ERROR
    assert failover_span.attributes.get("requested_provider") == "span-agent"
    assert failover_span.attributes.get("served_provider") == "agent.core"
    assert failover_span.attributes.get("error_class") == "ConnectionRefusedError"

    # Span ID threaded into in-band provenance
    assert evt_failover.provenance is not None
    assert evt_reply.provenance is not None
    assert evt_reply.provenance.path is ExecutionPath.FAILOVER
    assert evt_reply.provenance.degraded is True
    assert len(evt_reply.provenance.attempts) == 1

    attempt = evt_reply.provenance.attempts[0]
    assert attempt.span_id is not None
    assert attempt.span_id == failover_span.span_id
    assert attempt.error_class == "ConnectionRefusedError"
    assert attempt.provider == "span-agent"


@pytest.mark.asyncio
async def test_p6_integration_with_unreachable_ollama_connector(tmp_path: Path) -> None:
    """Integration test: Unreachable Ollama connector triggers P6 Checks 4 & 5."""
    bus = EventBus()
    tracer = TelemetryTracer()

    def _fail_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_fail_connect),
        base_url="http://127.0.0.1:11434",
    )
    unreachable_ollama = OllamaConnector(http_client=mock_client)
    mgr = AgentSessionManager(bus=bus, llm=unreachable_ollama, tracer=tracer)
    app = create_ui_app(static_dir=tmp_path, bus=bus, tracer=tracer, session_manager=mgr)
    sub = bus.subscribe("agent.chat.*")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        resp = await client.post(
            "/api/turn", json={"message": "ollama test", "agent_id": "ollama-agent"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "warning"

    # Verify bus notices (USER_INPUT -> PROVIDER_FAILOVER -> AGENT_REPLY)
    events: list[AgentEvent] = []
    for _ in range(3):
        events.append(await asyncio.wait_for(sub.get(), timeout=2.0))

    evt_input = next(e for e in events if e.type is EventType.USER_INPUT)
    evt_failover = next(e for e in events if e.type is EventType.PROVIDER_FAILOVER)
    evt_reply = next(e for e in events if e.type is EventType.AGENT_REPLY)

    assert evt_input.type is EventType.USER_INPUT
    assert evt_failover.type is EventType.PROVIDER_FAILOVER
    assert evt_reply.type is EventType.AGENT_REPLY
    assert evt_failover < evt_reply
    assert evt_failover.sequence < evt_reply.sequence

    # Verify span correlation
    spans = tracer.get_completed_spans()
    failover_spans = [s for s in spans if s.name == "failover.event"]
    assert len(failover_spans) >= 1
    span_ids = {s.span_id for s in failover_spans}

    assert evt_reply.provenance is not None
    assert evt_reply.provenance.attempts[0].span_id in span_ids


def test_p6_provenance_falsifiable_checks_models() -> None:
    """Model-level validation of P6 invariants."""
    # Check 3: Primary has empty attempts
    primary = Provenance.primary("ollama", "qwen2.5-coder")
    assert primary.path is ExecutionPath.PRIMARY
    assert primary.attempts == ()
    assert primary.degraded is False

    # Attempt on primary raises ValueError
    with pytest.raises(ValueError, match="attempts must be empty when path is 'primary'"):
        Provenance(
            path=ExecutionPath.PRIMARY,
            requested=ServiceRef(provider="p1"),
            served_by=ServiceRef(provider="p1"),
            attempts=(AttemptRecord(provider="p1", error_class="Err"),),
        )

    # Failover without attempts raises ValueError
    with pytest.raises(ValueError, match="attempts must be non-empty when path is 'failover'"):
        Provenance(
            path=ExecutionPath.FAILOVER,
            requested=ServiceRef(provider="p1"),
            served_by=ServiceRef(provider="p2"),
            attempts=(),
        )

    # Check 6: Consumer handed None raises MissingProvenanceError
    with pytest.raises(MissingProvenanceError):
        require_provenance(None, "TestEnvelope")


def _sample_agent_config(agent_id: str = "agent-failover-p6") -> AgentConfig:
    return AgentConfig(
        agent_id=agent_id,
        name="P6 Failover Test Agent",
        system_prompt="You are a failover test agent.",
        llm_config=AgentLLMConfig(model_name="mock-failing-llm", temperature=0.7),
    )


@pytest.mark.asyncio
async def test_base_agent_default_tracer_programmatic_execute_turn_satisfies_p6_check5() -> None:
    """Check 5: BaseAgent initialized with default settings (tracer=None) emits failover span on turn error."""
    from unittest.mock import AsyncMock, MagicMock

    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(
        side_effect=LLMProviderError("Primary provider connection timeout")
    )
    config = _sample_agent_config("agent-default-tracer-direct")

    agent = BaseAgent(config=config, llm=failing_llm)
    assert agent.tracer is not None

    res = await agent.execute_turn("Direct execution test prompt")

    assert res.is_completed is False
    assert res.error == "Primary provider connection timeout"
    assert res.provenance is not None
    assert res.provenance.path is ExecutionPath.FAILOVER
    assert res.provenance.degraded is True
    assert len(res.provenance.attempts) == 1

    attempt = res.provenance.attempts[0]
    assert attempt.span_id is not None
    assert attempt.provider == "agent-default-tracer-direct"
    assert attempt.model == "mock-failing-llm"
    assert attempt.error_class == "LLMProviderError"

    completed_spans = cast(TelemetryTracer, agent.tracer).get_completed_spans()
    failover_spans = [s for s in completed_spans if s.name == "failover.event"]
    assert len(failover_spans) == 1
    span = failover_spans[0]
    assert span.span_id == attempt.span_id
    assert span.status is SpanStatus.ERROR
    assert span.attributes.get("requested_provider") == "agent-default-tracer-direct"
    assert span.attributes.get("served_provider") == "agent.core"
    assert span.attributes.get("error_class") == "LLMProviderError"


@pytest.mark.asyncio
async def test_base_agent_default_tracer_direct_bus_satisfies_p6_checks_4_and_5() -> None:
    """Checks 4 & 5: Direct EventBus user input to BaseAgent with default tracer satisfies notice ordering and span resolution."""
    from unittest.mock import AsyncMock, MagicMock

    bus = EventBus()
    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(side_effect=ConnectionResetError("Socket reset by peer"))
    config = _sample_agent_config("agent-bus-default-tracer")

    agent = BaseAgent(config=config, bus=bus, llm=failing_llm)
    assert agent.tracer is not None

    await agent.start()
    try:
        events_sub = bus.subscribe("*")

        user_pub = bus.register_publisher(
            sender_id="user-direct",
            source=EventSource.USER,
        )
        input_event = AgentEvent(
            type=EventType.USER_INPUT,
            sender_id="user-direct",
            recipient_id=agent.agent_id,
            topic=f"session.{agent.context.session_id}",
            payload={"message": "bus trigger failover"},
        )
        await user_pub.publish(input_event)

        events: list[AgentEvent] = []
        for _ in range(3):
            events.append(await asyncio.wait_for(events_sub.get(), timeout=2.0))

        evt_input = next(e for e in events if e.type is EventType.USER_INPUT)
        evt_failover = next(e for e in events if e.type is EventType.PROVIDER_FAILOVER)
        evt_reply = next(e for e in events if e.type is EventType.AGENT_REPLY)

        # Check 4: Strict total ordering (notice strictly before reply)
        assert evt_failover < evt_reply
        assert evt_failover.sequence < evt_reply.sequence
        assert evt_failover.priority <= evt_reply.priority
        assert evt_input.sequence < evt_failover.sequence

        # Check 5: Span correlation
        assert evt_failover.provenance is not None
        assert evt_reply.provenance is not None
        assert evt_reply.provenance.path is ExecutionPath.FAILOVER
        assert evt_reply.provenance.degraded is True
        assert len(evt_reply.provenance.attempts) == 1

        attempt = evt_reply.provenance.attempts[0]
        assert attempt.span_id is not None
        assert attempt.error_class == "ConnectionResetError"

        completed_spans = cast(TelemetryTracer, agent.tracer).get_completed_spans()
        failover_spans = [s for s in completed_spans if s.name == "failover.event"]
        assert len(failover_spans) == 1
        assert failover_spans[0].span_id == attempt.span_id
        assert failover_spans[0].status is SpanStatus.ERROR
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_base_agent_process_event_directly_satisfies_p6_checks_4_and_5() -> None:
    """Checks 4 & 5: Direct process_event call on BaseAgent publishes PROVIDER_FAILOVER before AGENT_REPLY."""
    from unittest.mock import AsyncMock, MagicMock

    bus = EventBus()
    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(side_effect=TimeoutError("Model generation timed out"))
    config = _sample_agent_config("agent-direct-process-event")

    agent = BaseAgent(config=config, bus=bus, llm=failing_llm)
    all_events = bus.subscribe("*")

    input_event = AgentEvent(
        type=EventType.USER_INPUT,
        sender_id="caller-direct",
        recipient_id=agent.agent_id,
        topic=f"session.{agent.context.session_id}",
        payload={"message": "direct process event invocation"},
    )

    handled = await agent.process_event(input_event)
    assert handled is True

    evt_failover = await asyncio.wait_for(all_events.get(), timeout=2.0)
    evt_reply = await asyncio.wait_for(all_events.get(), timeout=2.0)

    assert evt_failover.type is EventType.PROVIDER_FAILOVER
    assert evt_reply.type is EventType.AGENT_REPLY

    # Check 4: Strict total ordering
    assert evt_failover < evt_reply
    assert evt_failover.sequence < evt_reply.sequence
    assert evt_failover.priority <= evt_reply.priority

    # Check 5: Span correlation
    assert evt_reply.provenance is not None
    assert evt_reply.provenance.path is ExecutionPath.FAILOVER
    assert len(evt_reply.provenance.attempts) == 1
    attempt = evt_reply.provenance.attempts[0]
    assert attempt.span_id is not None
    assert attempt.error_class == "TimeoutError"

    completed_spans = cast(TelemetryTracer, agent.tracer).get_completed_spans()
    failover_spans = [s for s in completed_spans if s.name == "failover.event"]
    assert len(failover_spans) == 1
    assert failover_spans[0].span_id == attempt.span_id


# ======================================================================================
# Checks 4 and 5 as tree-wide invariants, not per-call-site behaviour (#148 audit)
# ======================================================================================


# Why `model_validate` / `model_copy` are excluded from the producer obligation, stated
# here rather than left as an omission — an unrecorded exclusion is indistinguishable
# from an oversight, which is the lesson of #96.
#
# Checks 4 and 5 are the *producer's* obligation: emit the span, publish the notice,
# ordered before the result. A value arriving over the A2A wire
# (`Provenance.model_validate`, in `a2a/wire.py`) or restamped by the bus
# (`model_copy(update=...)`) was produced elsewhere — by a remote peer, or already by us.
# Requiring the forwarder to emit a *second* failover.event span and a *second*
# PROVIDER_FAILOVER notice would fabricate a local failover that never happened, which is
# the #157 defect wearing a compliance badge. Forwarders are still enumerated, so a new
# one has to be classified here rather than passing unseen.
_FORWARDING_RATIONALE = "forwarded, not produced: checks 4 and 5 bind the producer"

#: Modules that re-materialise a `Provenance` they did not produce.
#:
#: Note what is *not* here: `model_copy`. One whose `update=` is an **inline dict literal
#: containing a `path` key** is asserting a path the value did not have -- producing
#: attribution wearing a copy's clothing -- and is classified PRODUCED so it lands in the
#: registry above. That is the literal test `_rewrites_path` performs, and it is narrower
#: than "rewrites `path`": `upd = {...}` then `model_copy(update=upd)`, and
#: `model_copy(update=dict(path=...))`, are **not** covered.
#:
#: The code is deliberately not widened to reach them. A `model_copy` receiver's type
#: cannot be resolved by name, so the only way to catch the indirect forms is to treat
#: every `model_copy` as provenance-related -- which was the first draft here, and it
#: swept in six unrelated pydantic models from `ontology/`, `skills/` and `cli/`. A guard
#: that cries wolf is one the next builder deletes.
#: Empty today, and measured rather than assumed. The first draft of this set listed
#: `a2a/wire.py` because the wire module is where a peer's provenance enters -- but it
#: names `Provenance` only in prose: ingress goes through `AgentEvent` validation, which
#: re-validates the nested model without any `Provenance.model_validate` call of its own.
#: A registry entry with no site behind it is the same claim-without-code defect these
#: guards exist to catch, so the entry is gone and the staleness assertion below is what
#: would have caught it.
_PROVENANCE_FORWARDERS: frozenset[str] = frozenset()


def _non_primary_provenance_sites() -> dict[str, list[int]]:
    """Every module producing a provenance not provably `PRIMARY`, with line numbers.

    Two changes from the version this replaces, both from the #188 review:

    * **Fails closed.** It asked `"PRIMARY" not in ast.unparse(value)`, so any identifier
      containing that substring — `NON_PRIMARY_PATH` is the obvious one — silently
      excluded the site, disabling the guard exactly where a guard matters. The shared
      resolver returns `UNKNOWN` when the path is not provable, and `UNKNOWN` counts as a
      producer to be classified.
    * **Sees the forms that evaded it.** `Provenance(**payload)`,
      `Provenance.model_validate({...})` and `model_copy(update={...})` were all
      constructible with `path=failover` and all three were invisible. Producing forms
      now count; forwarding forms are excluded deliberately (above) and separately
      enumerated.
    """
    src_root = Path(__file__).resolve().parents[2] / "src"
    sites: dict[str, list[int]] = {}
    for path in sorted((src_root / "uclone_x").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = provenance_binding_names(tree)
        rel = str(path.relative_to(src_root))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            kind = construction_kind(node, names)
            if kind is not ConstructionKind.PRODUCED:
                continue
            if constructed_path(node, names) is not PathVerdict.PRIMARY:
                sites.setdefault(rel, []).append(node.lineno)
    return sites


def _provenance_forwarding_sites() -> dict[str, list[int]]:
    """Every module that re-materialises a `Provenance` (forwarding, not producing)."""
    src_root = Path(__file__).resolve().parents[2] / "src"
    sites: dict[str, list[int]] = {}
    for path in sorted((src_root / "uclone_x").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = provenance_binding_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                construction_kind(node, names) is ConstructionKind.FORWARDED
            ):
                sites.setdefault(str(path.relative_to(src_root)), []).append(node.lineno)
    return sites


def _module_mentions_in_code(module_source: str, needles: tuple[str, ...]) -> bool:
    """True when a needle appears in *code*, ignoring comments and docstrings (#188).

    The predecessor asked `"failover.event" in source`, which cannot tell code from
    prose: a stale comment naming the span kept a module labelled compliant after the
    span itself was renamed. The reviewer found this the honest way — its mutation landed
    on the comment two lines above the real call and it reported the mistake rather than
    the result.
    """
    tree = ast.parse(module_source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            found.update(n for n in needles if n in node.value)
        elif isinstance(node, ast.Attribute):
            found.update(n for n in needles if n == node.attr)
        elif isinstance(node, ast.Name):
            found.update(n for n in needles if n == node.id)
    return bool(found)


# Every module allowed to produce a non-primary `Provenance`, and its status against
# checks 4 and 5. A new producer fails the test below until it is classified.
#
# `gap:#<issue>` is a legitimate value and must stay one. Check 4 requires the notice to
# be published *before* the result, and the source-level cross-check below cannot see
# ordering — so a producer that emits the span and publishes the notice *after* the
# result is genuinely non-compliant while carrying both symbols. Forbidding `gap` for
# such a module would leave the registry holding a knowingly false `compliant` (#188
# item 4). A gap must cite an issue; that is the whole cost of recording one.
_NON_PRIMARY_PRODUCERS: dict[str, str] = {
    # The only one, since #175/#179 migrated span emission, the notice and the provenance
    # out of `ui/app.py` and into the agent. Emits `failover.event`, threads its span_id
    # into the AttemptRecord, and publishes a PROVIDER_FAILOVER before the reply.
    # Unconditional since #184 gave every agent a default tracer; the behavioural proof
    # is the three `test_base_agent_*` cases above, and this registry is what makes a
    # *new* producer in a new module fail.
    "uclone_x/agent/base.py": "compliant",
}


def test_every_provenance_forwarder_is_registered() -> None:
    """Forwarders are exempt from checks 4 and 5, but not from being noticed (#188).

    `_FORWARDING_RATIONALE` above says why re-materialising a peer's provenance does not
    incur the producer's obligations. **There are zero such modules today** — measured,
    and the reason `_PROVENANCE_FORWARDERS` is empty. The reasoning is not a blanket
    licence, so the first one has to be looked at rather than inheriting the exemption
    silently.

    An earlier draft of this docstring said "the two modules that do it today", flatly
    contradicting the comment three lines above it, in the test written to catch claims
    with no code behind them, about the very set whose fictitious entry motivated it.
    """
    found = set(_provenance_forwarding_sites())
    unregistered = found - _PROVENANCE_FORWARDERS
    assert not unregistered, (
        f"new `Provenance` forwarding site(s): {sorted(unregistered)}\n"
        f"These are exempt from P6 checks 4 and 5 only because {_FORWARDING_RATIONALE}. "
        "Confirm that applies here — if the module is asserting attribution rather than "
        "relaying it, it belongs in _NON_PRIMARY_PRODUCERS instead."
    )
    stale = _PROVENANCE_FORWARDERS - found
    assert not stale, (
        f"registered forwarder(s) with no forwarding site: {sorted(stale)}\n"
        "An exemption for something that does not exist is a claim with no code behind "
        "it. Remove the entry."
    )


def test_every_non_primary_provenance_producer_is_registered() -> None:
    """A new non-primary producer must be classified against checks 4 and 5 (#148 audit).

    This is the generalisation the behavioural tests cannot provide. #148's original
    implementation was correct where it looked and absent where it did not: the producer
    it missed came from #150, which had already landed (`4a82a7b` precedes `740a2fc`), so
    #148 shipped over a producer that was uncovered in the tree at the time. The same
    shape a week earlier: #136 removed the fabrication from `agent/base.py` and `f590f65`
    (#135) reintroduced it in the A2A gateway — #157 is the issue that *reported* that,
    not the change that did it.

    Both of those numbers were wrong in an earlier draft, in the same direction: naming
    the report instead of the change. An enumeration fails the moment the set changes,
    which is the only thing that has reliably worked here.
    """
    found = set(_non_primary_provenance_sites())
    registered = set(_NON_PRIMARY_PRODUCERS)

    unregistered = found - registered
    assert not unregistered, (
        f"unclassified non-primary `Provenance` producer(s): {sorted(unregistered)}\n"
        "P6 checks 4 and 5 apply to EVERY result with path != 'primary': the producer "
        "must emit a `failover.event` span, thread its span_id into attempts[*], and "
        "publish a PROVIDER_FAILOVER before the result. Add it to _NON_PRIMARY_PRODUCERS "
        "as 'compliant', or file an issue and record it as 'gap:#<issue>'."
    )
    departed = registered - found
    assert not departed, (
        f"registered producer(s) no longer construct non-primary provenance: "
        f"{sorted(departed)}\nRemove them from _NON_PRIMARY_PRODUCERS."
    )


def test_a_producer_labelled_compliant_carries_the_machinery_checks_4_and_5_require() -> None:
    """A label is not evidence: cross-check it against the module's source (#148 audit).

    Added because mutation testing found the registry could be silently relabelled — a
    "gap" entry flipped to "compliant" passed every other test in this file, which would
    have made the table assert compliance that nothing verifies. That is precisely the
    class of defect this repository keeps producing.

    Necessary, not sufficient: a module can name both symbols and still order them
    wrongly. Ordering is what the behavioural tests above are for. This is enough to stop
    a relabel-without-a-fix, which is the realistic move.
    """
    src_root = Path(__file__).resolve().parents[2] / "src"
    for module, status in sorted(_NON_PRIMARY_PRODUCERS.items()):
        source = (src_root / module).read_text(encoding="utf-8")
        emits_span = _module_mentions_in_code(
            source, ("failover.event", "FAILOVER_EVENT_SPAN_NAME")
        )
        publishes_notice = _module_mentions_in_code(source, ("PROVIDER_FAILOVER", "RETRY"))

        if status == "compliant":
            assert emits_span, f"{module} is 'compliant' but emits no `failover.event` span"
            assert publishes_notice, f"{module} is 'compliant' but publishes no notice"
        else:
            assert status.startswith("gap:#"), (
                f"{module} has status {status!r}; a non-compliant producer must be recorded "
                "as 'gap:#<issue>' so the gap is owned rather than merely noted."
            )
            # A trade, stated as one. The predecessor asserted `not (emits_span and
            # publishes_notice)` here, which bound the reverse direction: a working
            # producer could not be silently downgraded to a gap. That binding is **given
            # up** -- a downgrade-without-cause is now unbound and this guard is weaker in
            # that one direction than it was before #188.
            #
            # It is given up because check 4 is an *ordering* requirement and nothing at
            # source level reads ordering, so a module publishing the notice after the
            # result carries both symbols while being genuinely non-compliant. The old
            # assertion made `gap` unrecordable for exactly that module, which would
            # eventually have forced a knowingly false `compliant` into the registry. A
            # forced-false label is worse than an unbound downgrade. Requiring
            # `gap:#<issue>` is what survives: a cited issue is reviewable evidence,
            # where the symbol table is not (#188 item 4).
            issue = status.removeprefix("gap:#")
            assert issue.isdigit(), (
                f"{module} is recorded {status!r}; a gap must cite a filed issue number, "
                "because the source-level signals below cannot distinguish an ordering "
                "gap from an unfixed one and the citation is what makes it reviewable."
            )


def test_the_cross_check_reads_code_and_not_prose() -> None:
    """`_module_mentions_in_code` ignores comments and docstrings (#188 item 3).

    The predecessor asked `"failover.event" in source`, so a **stale comment** naming the
    span kept a module labelled `compliant` after the span itself was renamed — the
    registry would have asserted compliance on the strength of a sentence. The reviewer
    found this the honest way: its mutation landed on the comment two lines above the
    real call, and it reported the mistake rather than the result.

    Asserted directly because the behavioural tests catch a renamed span for their own
    reasons; without this, the cross-check's prose-blindness would be unpinned and could
    regress silently.
    """
    only_a_comment = '''"""Module docstring mentioning failover.event and PROVIDER_FAILOVER."""

# A stale comment about the failover.event span and EventType.PROVIDER_FAILOVER.
def f() -> None:
    """Docstring naming failover.event too."""
    return None
'''
    assert not _module_mentions_in_code(only_a_comment, ("failover.event",))
    assert not _module_mentions_in_code(only_a_comment, ("PROVIDER_FAILOVER",))

    real_code = """def f(tracer, bus) -> None:
    tracer.start_span("failover.event")
    bus.publish(AgentEvent(type=EventType.PROVIDER_FAILOVER))
"""
    assert _module_mentions_in_code(real_code, ("failover.event",))
    assert _module_mentions_in_code(real_code, ("PROVIDER_FAILOVER",))


def test_failover_event_span_name_constant_matches_agent_emission() -> None:
    """The constant FAILOVER_EVENT_SPAN_NAME equals 'failover.event' and is used across modules (#202)."""
    from uclone_x.telemetry.tracer import FAILOVER_EVENT_SPAN_NAME

    assert FAILOVER_EVENT_SPAN_NAME == "failover.event"

    code_with_constant = """def f(tracer) -> None:
    tracer.start_span(FAILOVER_EVENT_SPAN_NAME)
"""
    assert _module_mentions_in_code(code_with_constant, ("FAILOVER_EVENT_SPAN_NAME",))
