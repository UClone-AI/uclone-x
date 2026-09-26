import asyncio
from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.hooks.permission import HumanApprovalHook, PermissionMode
from uclone_x.agent.models import AgentConfig
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.llm.models import ToolCallRequest
from uclone_x.tools.models import ToolContext


@pytest.mark.asyncio
async def test_tool_execution_suspended_and_approved():
    bus = EventBus()
    await bus.start()
    config = AgentConfig(agent_id="test_agent", name="test_agent", approval_timeout_seconds=5.0)
    hook = HumanApprovalHook(permission_mode=PermissionMode.DEFAULT)

    agent = BaseAgent(config=config, bus=bus, hooks=[hook])

    tc = ToolCallRequest(id="tc_1", name="write_file", arguments={"path": "/tmp/a.txt"})
    ctx = ToolContext(agent_id="test_agent", session_id="s1", workspace_root=Path("/tmp"))

    # Subscribe BEFORE creating task
    sub = bus.subscribe({"session.sess_test_agent"})

    # We will simulate the approval by waiting for the event and responding
    async def approver():
        req_event = None
        while True:
            event = await sub.get()
            if event.type == EventType.TOOL_APPROVAL_REQUEST:
                req_event = event
                break

        # Respond
        resp = AgentEvent(
            type=EventType.TOOL_APPROVAL_RESPONSE,
            topic="session.sess_test_agent",
            sender_id="ui",
            payload={"request_id": req_event.payload["request_id"], "action": "allow"},
        )
        print(f"Approver sending response for request_id: {req_event.payload['request_id']}")
        await bus.publish(resp)

    await agent.start()

    # Run tool and approver concurrently
    task = asyncio.create_task(approver())

    # Will suspend and then resume when approver sends response
    msg, rec = await agent._execute_single_tool(tc, ctx)  # pyright: ignore[reportPrivateUsage]

    assert (
        rec.status == "error"
    )  # because write_file is not registered in tools here, but it bypassed the ASK!
    assert "not found" in (rec.error or "") or "not found" in (msg.content or "")

    await task
    await agent.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_tool_execution_timeout_fails_closed():
    bus = EventBus()
    await bus.start()
    config = AgentConfig(
        agent_id="test_agent", name="test_agent", approval_timeout_seconds=0.1
    )  # short timeout
    hook = HumanApprovalHook(permission_mode=PermissionMode.DEFAULT)

    agent = BaseAgent(config=config, bus=bus, hooks=[hook])

    tc = ToolCallRequest(id="tc_1", name="write_file", arguments={"path": "/tmp/a.txt"})
    ctx = ToolContext(agent_id="test_agent", session_id="s1", workspace_root=Path("/tmp"))

    await agent.start()
    # Just run it, it should time out
    _, rec = await agent._execute_single_tool(tc, ctx)  # pyright: ignore[reportPrivateUsage]

    assert rec.status == "error"
    assert "Approval request timed out" in (rec.error or "")

    await agent.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_approval_path_carries_nested_tool_arguments():
    """The ASK path survives a tool call whose arguments contain a nested object.

    Two sites on this path took `dict(tc.arguments)` over a recursively frozen mapping.
    The reviewer of #673 cleared both as benign on the reasoning that the destination is
    an `Immutable*Mapping` field, so pydantic would "re-freeze" them. It does not:
    `AfterValidator(freeze_mapping)` runs *after* `JsonValue` validation, and a nested
    `MappingProxyType` is not a valid `JsonValue` — so the field **rejects** the shallow
    copy with `ValidationError` rather than repairing it.

    Both raises are inside `_execute_single_tool`, so the symptom is a tool call that
    dies before the approval request is ever published — not a visible encoder error.
    The assertions are on the outcome the fail-closed design promises: a refusal that
    names the timeout, carrying the arguments it refused.

    Killed by: src/uclone_x/agent/base.py :: else cast(dict[str, Any], unwrap_immutable(tc.arguments))
    Becomes: else cast(dict[str, Any], tc.arguments)
    Killed by: src/uclone_x/agent/base.py :: arguments=cast(dict[str, Any], unwrap_immutable(tc.arguments)),
    """
    nested_arguments: dict[str, Any] = {
        "path": "/tmp/a.txt",
        "options": {"encoding": "utf-8"},
        "ranges": [{"start": 1, "end": 20}],
    }

    bus = EventBus()
    await bus.start()
    config = AgentConfig(agent_id="test_agent", name="test_agent", approval_timeout_seconds=0.1)
    agent = BaseAgent(
        config=config, bus=bus, hooks=[HumanApprovalHook(permission_mode=PermissionMode.DEFAULT)]
    )

    tc = ToolCallRequest(id="tc_nested", name="write_file", arguments=nested_arguments)
    ctx = ToolContext(agent_id="test_agent", session_id="s1", workspace_root=Path("/tmp"))

    sub = bus.subscribe({"session.sess_test_agent"})
    await agent.start()
    try:
        _, rec = await agent._execute_single_tool(tc, ctx)  # pyright: ignore[reportPrivateUsage]

        # The request reached the bus: the ASK payload was publishable.
        seen: list[AgentEvent] = []
        while not sub.empty():
            seen.append(await sub.get())
        approval_requests = [e for e in seen if e.type == EventType.TOOL_APPROVAL_REQUEST]
        assert approval_requests, f"no approval request was published; saw {[e.type for e in seen]}"
        # Read back through `unwrap_immutable`: the field re-froze the plain mapping on
        # the way in, which is the whole point — it could only do so because what it was
        # handed was plain.
        assert unwrap_immutable(approval_requests[0].payload["arguments"]) == nested_arguments

        # And the fail-closed record was constructible with those same arguments.
        assert rec.status == "error"
        assert "Approval request timed out" in (rec.error or "")
        assert unwrap_immutable(rec.arguments) == nested_arguments
    finally:
        sub.close()
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_approval_response_carries_nested_modified_arguments():
    """A MODIFY approval response with a nested `modified_arguments` must not kill the turn.

    The approval-response reader binds `payload = evt.payload` — an `AgentEvent.payload`,
    frozen recursively — and then hands `payload.get("modified_arguments")` to
    `ApprovalDecision.modified_arguments`, which is `dict[str, Any]` under `strict=True`
    and rejects a `MappingProxyType` with `dict_type`. That construction sits inside a
    `try` whose only handler is `except TimeoutError`, so the `ValidationError` escapes
    `_execute_single_tool` and takes the whole turn down rather than degrading.

    Both #673 sweeps missed this site: the region was characterised as "read-only
    `.get()`", and the fitness sweep matches `dict(<expr>.<field>)`, not an aliased
    `.get()` whose result flows into a plain-`dict` field. No in-repo publisher sets the
    key today; `a2a_server.py` was equally latent until someone wired it.

    The assertion is on the *effect*, not on the absence of a raise: the modified
    arguments must actually reach `effective_tc`. `_execute_single_tool` overrides the
    call only `if isinstance(mod_payload["arguments"], dict)` — which a `mappingproxy`
    fails — so a half-fix that stopped the raise but left the proxy in place would
    silently discard the modification, and this test would still fail.

    Killed by: src/uclone_x/agent/base.py :: modified_arguments=cast(dict[str, Any], unwrap_immutable(modified_arguments))
    """
    modified: dict[str, Any] = {
        "path": "/tmp/safe.txt",
        "options": {"encoding": "utf-8", "limits": {"max_bytes": 10}},
        "ranges": [{"start": 1, "end": 20}],
    }

    bus = EventBus()
    await bus.start()
    config = AgentConfig(agent_id="test_agent", name="test_agent", approval_timeout_seconds=5.0)
    agent = BaseAgent(
        config=config, bus=bus, hooks=[HumanApprovalHook(permission_mode=PermissionMode.DEFAULT)]
    )

    tc = ToolCallRequest(id="tc_mod", name="write_file", arguments={"path": "/etc/passwd"})
    ctx = ToolContext(agent_id="test_agent", session_id="s1", workspace_root=Path("/tmp"))

    sub = bus.subscribe({"session.sess_test_agent"})

    async def approver() -> None:
        while True:
            event = await sub.get()
            if event.type == EventType.TOOL_APPROVAL_REQUEST:
                break
        await bus.publish(
            AgentEvent(
                type=EventType.TOOL_APPROVAL_RESPONSE,
                topic="session.sess_test_agent",
                sender_id="ui",
                payload={
                    "request_id": event.payload["request_id"],
                    "action": "modify",
                    "modified_arguments": modified,
                    "decided_by": "ui_operator",
                },
            )
        )

    await agent.start()
    task = asyncio.create_task(approver())
    try:
        _, rec = await agent._execute_single_tool(tc, ctx)  # pyright: ignore[reportPrivateUsage]

        # `write_file` is not registered here, so the call fails at lookup — but it fails
        # *having been modified*, which is what proves the decision survived validation
        # as a plain dict.
        assert rec.status == "error"
        assert "not found" in (rec.error or "")
        assert unwrap_immutable(rec.arguments) == modified
    finally:
        await task
        sub.close()
        await agent.stop()
        await bus.stop()
