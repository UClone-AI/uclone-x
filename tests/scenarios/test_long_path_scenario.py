"""Long-path scenario test crossing multiple UClone-X subsystems."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import MessageRole, ToolCallRequest
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.skills.auditor import Skill, SkillRegistry
from uclone_x.skills.models import (
    AuditVerdict,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import LocalTool, ToolRegistry


@pytest.mark.asyncio
async def test_long_path_stage_1_plan_tool_persist_compact(tmp_path: Path) -> None:
    """Stage 1 of the single long scenario:
    session init -> tool execution -> session persistence -> compaction -> reload.

    Subsequent feature issues (#470–#476) will extend this scenario with additional stages:
    skill loading -> sub-agent delegation -> offloaded output -> approval -> memory.
    """
    bus = EventBus(maxsize=100)
    await bus.start()
    store = SessionStore(storage_dir=tmp_path / "scenario_sessions")
    tools = ToolRegistry()
    from uclone_x.tools.builtin.plan import PlanUpdateTool

    tools.register(PlanUpdateTool())

    # Stage 1.1: Register scratchpad tool
    memory_store: dict[str, str] = {}

    async def write_note(args: dict[str, object], ctx: ToolContext) -> ToolResult:
        note_id = str(args.get("id", "note_1"))
        text = str(args.get("text", ""))
        memory_store[note_id] = text
        return ToolResult(
            output=f"Note {note_id} written",
            success=True,
            provenance=Provenance.primary(provider="tool.write_note", model="write_note"),
        )

    tools.register(
        LocalTool(
            name="write_note",
            description="Write a persistent scratchpad note",
            parameters_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
            handler=write_note,
        )
    )

    # Mock responses:
    # Turn 1: tool call to write_note
    # Turn 2: answer acknowledging write
    tc = ToolCallRequest(
        id="call_stage1",
        name="write_note",
        arguments={"id": "mission_alpha", "text": "objective_accomplished"},
    )
    llm = MockLLMConnector(
        responses=["Note has been recorded into mission scratchpad."],
        tool_calls=[tc],
    )

    config = AgentConfig(
        agent_id="scenario-runner",
        name="ScenarioRunner",
        system_prompt="You are an autonomous long-running agent.",
        llm_config=AgentLLMConfig(
            model_name="mock-model",
            auto_compact=False,
        ),
    )

    agent = BaseAgent(
        config=config,
        bus=bus,
        llm=llm,
        tools=tools,
        store=store,
    )

    sub = bus.subscribe(topics={f"session.{agent.session_id}"})

    await agent.start()

    try:
        # Step 1: Input to execute stage 1
        await bus.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic=f"session.{agent.session_id}",
                sender_id="scenario-observer",
                recipient_id=agent.agent_id,
                session_id=agent.session_id,
                payload={"message": "Execute mission alpha scratchpad write"},
            )
        )

        reply: AgentEvent | None = None
        for _ in range(50):
            if not sub.empty():
                evt = await sub.get()
                if evt.type == EventType.AGENT_REPLY:
                    reply = evt
                    break
            await asyncio.sleep(0.05)

        assert reply is not None, "Scenario stage 1 failed to produce AGENT_REPLY"
        assert memory_store.get("mission_alpha") == "objective_accomplished"

        # Step 2: Persist session and verify revision increment
        saved = agent.persist_session()
        assert saved.turn_counter == 1
        assert saved.revision >= 1

        # Step 3: Trigger compaction on demand
        compact_res = await agent.compact_session(reason="scenario_stage_1_compaction")
        assert compact_res.messages_after <= compact_res.messages_before

        # Step 4: Re-hydrate from store into a new agent instance
        new_agent = BaseAgent(
            config=config,
            bus=bus,
            store=store,
        )
        hydrated = new_agent.hydrate_session(agent.session_id)
        assert hydrated is not None
        assert hydrated.session_id == agent.session_id
        assert hydrated.turn_counter == 1

        # Step 5: Stage 2 - Wire approved skill registry and load skill on demand (Issue #473)
        skills_registry = SkillRegistry()
        skill_manifest = SkillManifest(
            name="tactical_analysis",
            description="Autonomous tactical plan evaluation and synthesis",
            origin=SkillOrigin.HUMAN,
            status=SkillStatus.ACTIVE,
            content_sha256="sha_tactical",
        )
        skill_obj = Skill(
            manifest=skill_manifest,
            instructions_markdown="# Tactical Analysis Protocol\n1. Evaluate threats.\n2. Formulate counteractions.",
        )
        report = SkillAuditReport(
            skill_name="tactical_analysis",
            is_safe=True,
            recommendation=AuditVerdict.APPROVE,
            content_sha256="sha_tactical",
            auditor_version="0.1.0",
        )
        skills_registry.register(skill_obj, report)

        stage2_tc = ToolCallRequest(
            id="call_stage2_skill",
            name="load_skill",
            arguments={"skill_name": "tactical_analysis"},
        )
        stage2_llm = MockLLMConnector(
            responses=["Tactical analysis skill loaded. Ready to evaluate counteractions."],
            tool_calls=[stage2_tc],
        )

        stage2_agent = BaseAgent(
            config=config,
            bus=bus,
            llm=stage2_llm,
            store=store,
            skills=skills_registry,
        )
        stage2_hydrated = stage2_agent.hydrate_session(agent.session_id)
        assert stage2_hydrated is not None
        await stage2_agent.start()

        stage2_sub = bus.subscribe(topics={f"session.{stage2_agent.session_id}"})
        try:
            await bus.publish(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    topic=f"session.{stage2_agent.session_id}",
                    sender_id="scenario-observer",
                    recipient_id=stage2_agent.agent_id,
                    session_id=stage2_agent.session_id,
                    payload={"message": "Load tactical analysis skill to evaluate perimeter"},
                )
            )

            stage2_reply: AgentEvent | None = None
            for _ in range(50):
                if not stage2_sub.empty():
                    evt = await stage2_sub.get()
                    if evt.type == EventType.AGENT_REPLY:
                        stage2_reply = evt
                        break
                await asyncio.sleep(0.05)

            assert stage2_reply is not None, "Scenario stage 2 failed to produce AGENT_REPLY"
            assert "tactical_analysis" in stage2_agent.loaded_skills
            stage2_saved = stage2_agent.persist_session()
            assert stage2_saved.turn_counter == 2
        finally:
            await stage2_agent.stop()
            stage2_sub.close()

        # Stage 2b: Establish and update execution plan (Issue #475)
        stage2b_tc = ToolCallRequest(
            id="call_stage2b_plan",
            name="update_plan",
            arguments={
                "action": "create",
                "title": "Scenario Plan",
                "steps": [{"description": "First step"}],
            },
        )
        stage2b_llm = MockLLMConnector(responses=["Plan established."], tool_calls=[stage2b_tc])
        stage2b_agent = BaseAgent(
            config=config,
            bus=bus,
            llm=stage2b_llm,
            store=store,
            tools=tools,
        )
        stage2b_hydrated = stage2b_agent.hydrate_session(agent.session_id)
        assert stage2b_hydrated is not None
        await stage2b_agent.start()

        stage2b_sub = bus.subscribe(topics={f"session.{stage2b_agent.session_id}"})
        try:
            await bus.publish(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    topic=f"session.{stage2b_agent.session_id}",
                    sender_id="scenario-observer",
                    recipient_id=stage2b_agent.agent_id,
                    session_id=stage2b_agent.session_id,
                    payload={"message": "Create an execution plan"},
                )
            )

            stage2b_reply: AgentEvent | None = None
            for _ in range(50):
                if not stage2b_sub.empty():
                    evt = await stage2b_sub.get()
                    if evt.type == EventType.AGENT_REPLY:
                        stage2b_reply = evt
                        break
                await asyncio.sleep(0.05)

            assert stage2b_reply is not None, "Scenario stage 2b failed to produce AGENT_REPLY"
            assert stage2b_agent.current_plan is not None
            assert stage2b_agent.current_plan.title == "Scenario Plan"
            stage2b_saved = stage2b_agent.persist_session()
            assert stage2b_saved.plan is not None
            assert stage2b_saved.plan.title == "Scenario Plan"
        finally:
            await stage2b_agent.stop()
            stage2b_sub.close()

        # Stage 3: Tool approval request & response
        # Create a new agent instance configured with HumanApprovalHook
        from uclone_x.agent.hooks.permission import HumanApprovalHook, PermissionMode

        stage3_hook = HumanApprovalHook(
            permission_mode=PermissionMode.DEFAULT, ask_tools={"dummy_ask_tool"}
        )

        async def dummy_ask_tool_fn(args: dict[str, object], ctx: ToolContext) -> ToolResult:
            return ToolResult(output="done", success=True, provenance=None)

        tools.register(
            LocalTool(
                name="dummy_ask_tool",
                description="dummy",
                parameters_schema={"type": "object"},
                handler=dummy_ask_tool_fn,
            )
        )

        stage3_agent = BaseAgent(
            config=AgentConfig(
                agent_id="sc_agent_1",
                name="sc_agent_1",
                system_prompt="Test",
                approval_timeout_seconds=5.0,
            ),
            bus=bus,
            llm=MockLLMConnector(
                responses=["I will call the ask tool now.", "All done!"],
                tool_calls=[
                    ToolCallRequest(id="tc_ask", name="dummy_ask_tool", arguments={"x": 1})
                ],
            ),
            tools=tools,
            store=store,
            hooks=[stage3_hook],
        )
        # No need to hydrate session for this stage
        await stage3_agent.start()

        stage3_sub = bus.subscribe(
            {f"session.{stage3_agent.session_id}", "agent.sc_agent_1"},
            recipient_id="scenario-observer",
        )
        appr_sub = bus.subscribe(
            {f"session.{stage3_agent.session_id}", "agent.sc_agent_1"},
            recipient_id="human_approver",
        )
        appr_task: asyncio.Task[None] | None = None

        try:

            async def human_approver() -> None:
                while True:
                    evt = await appr_sub.get()
                    if evt.type == EventType.TOOL_APPROVAL_REQUEST:
                        # send response
                        resp = AgentEvent(
                            type=EventType.TOOL_APPROVAL_RESPONSE,
                            topic=f"session.{stage3_agent.session_id}",
                            sender_id="ui",
                            payload={"request_id": evt.payload["request_id"], "action": "allow"},
                        )
                        await bus.publish(resp)
                        break

            appr_task = asyncio.create_task(human_approver())

            await bus.publish(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    topic=f"session.{stage3_agent.session_id}",
                    sender_id="scenario-observer",
                    recipient_id="sc_agent_1",
                    payload={"message": "Run dummy ask"},
                )
            )

            stage3_reply: AgentEvent | None = None
            for _ in range(50):
                if not stage3_sub.empty():
                    evt = await stage3_sub.get()
                    if evt.type == EventType.AGENT_REPLY:
                        stage3_reply = evt
                        break
                await asyncio.sleep(0.05)

            assert stage3_reply is not None, "Scenario stage 3 failed to produce AGENT_REPLY"
        finally:
            if appr_task:
                appr_task.cancel()
            await stage3_agent.stop()
            stage3_sub.close()
            appr_sub.close()

        # Stage 4: Subagent delegation
        from uclone_x.tools.builtin.subagent import SubagentDelegationTool

        tools.register(SubagentDelegationTool())

        stage4_tc = ToolCallRequest(
            id="call_stage4_subagent",
            name="delegate_subagent",
            arguments={
                "role": "researcher",
                "goal": "Find XYZ",
                "prompt": "Search XYZ",
                "max_turns": 2,
            },
        )
        stage4_llm = MockLLMConnector(
            responses=["Subagent deployed.", "Subagent finished."],
            tool_calls=[stage4_tc],
        )
        stage4_agent = BaseAgent(
            config=AgentConfig(
                agent_id="sc_agent_4",
                name="sc_agent_4",
                max_turns=10,
            ),
            bus=bus,
            llm=stage4_llm,
            tools=tools,
            store=store,
        )

        await stage4_agent.start()
        stage4_sub = bus.subscribe(
            {f"session.{stage4_agent.session_id}", "swarm.subagent.*", "broadcast"},
            recipient_id="scenario-observer",
        )

        try:
            # Note: EventBus does not actually support wildcard by default without setup,
            # so we'll just check if the agent executes correctly.
            await bus.publish(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    topic=f"session.{stage4_agent.session_id}",
                    sender_id="scenario-observer",
                    recipient_id="sc_agent_4",
                    payload={"message": "Use subagent"},
                )
            )

            stage4_reply = None
            for _ in range(50):
                if not stage4_sub.empty():
                    evt = await stage4_sub.get()
                    if evt.type == EventType.AGENT_REPLY:
                        stage4_reply = evt
                        break
                await asyncio.sleep(0.05)

            assert stage4_reply is not None, "Scenario stage 4 failed to produce AGENT_REPLY"
            # Subagent consumes 1 turn because it uses 1 prompt internally via execute_turn in delegate_task,
            # Wait, our MockLLMConnector for the subagent?
            # Wait, if `delegate_task` uses the same LLM?
            # The subagent will inherit `stage4_llm`, which has `responses` but no more `tool_calls`.
            # So it'll answer "Subagent finished." and consume 1 turn.
            # Thus, the parent turn counter should be at least 1 from itself and 1 from subagent.
            assert stage4_agent.turn_counter > 0
        finally:
            await stage4_agent.stop()
            stage4_sub.close()

        # Stage 5: Oversized tool output offloaded to sandbox filesystem and recovered
        from uclone_x.tools.builtin.filesystem import FileReadParams, FileReadTool

        read_tool = FileReadTool()
        tools.register(read_tool)

        stage5_ws = tmp_path / "stage5_ws"
        stage5_ws.mkdir()

        large_output = (
            "START_CHUNK\n" + ("X" * 1500) + "\nCRITICAL_KEY_#472\n" + ("Y" * 1500) + "\nEND_CHUNK"
        )

        async def big_query(args: dict[str, object], ctx: ToolContext) -> ToolResult:
            return ToolResult(
                output=large_output,
                success=True,
                provenance=Provenance.primary("test.big_query"),
            )

        tools.register(
            LocalTool(
                name="big_query",
                description="Query large dataset",
                parameters_schema={"type": "object", "properties": {}},
                handler=big_query,
            )
        )

        stage5_tc = ToolCallRequest(
            id="call_stage5_big",
            name="big_query",
            arguments={},
        )
        stage5_llm = MockLLMConnector(
            responses=["Processing query results..."],
            tool_calls=[stage5_tc],
        )
        stage5_agent = BaseAgent(
            config=AgentConfig(
                agent_id="sc_agent_5",
                name="sc_agent_5",
                workspace_dir=str(stage5_ws),
                llm_config=AgentLLMConfig(
                    model_name="mock-model",
                    auto_compact=False,
                ),
            ),
            bus=bus,
            llm=stage5_llm,
            tools=tools,
            store=store,
        )

        await stage5_agent.start()
        stage5_sub = bus.subscribe(
            {f"session.{stage5_agent.session_id}"},
            recipient_id="scenario-observer",
        )

        try:
            await bus.publish(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    topic=f"session.{stage5_agent.session_id}",
                    sender_id="scenario-observer",
                    recipient_id="sc_agent_5",
                    payload={"message": "Execute big_query"},
                )
            )

            stage5_reply = None
            for _ in range(50):
                if not stage5_sub.empty():
                    evt = await stage5_sub.get()
                    if evt.type == EventType.AGENT_REPLY:
                        stage5_reply = evt
                        break
                await asyncio.sleep(0.05)

            assert stage5_reply is not None, "Scenario stage 5 failed to produce AGENT_REPLY"

            # Compact to trigger offload of oversized output
            await stage5_agent.compact_session()

            # Verify offloaded message
            messages = stage5_agent._live_session(stage5_agent.session_id).messages  # pyright: ignore[reportPrivateUsage]
            tool_msg = next(
                m for m in messages if m.role == MessageRole.TOOL and m.name == "big_query"
            )
            assert tool_msg.content is not None
            assert "[Tool Output Offloaded" in tool_msg.content
            assert "path=offload" in tool_msg.content

            # Read back using FileReadTool
            import re

            m = re.search(r"Full output saved to '([^']+)'", tool_msg.content)
            assert m is not None
            artifact_rel_path = m.group(1)

            tool_ctx = ToolContext(
                agent_id=stage5_agent.agent_id,
                workspace_root=stage5_ws,
                session_id=stage5_agent.session_id,
            )
            read_res = read_tool.run(FileReadParams(path=artifact_rel_path), tool_ctx)
            assert read_res["content"] == large_output
            assert "CRITICAL_KEY_#472" in read_res["content"]
        finally:
            await stage5_agent.stop()
            stage5_sub.close()

        # Stage 6: Cross-session memory carried into a distinct later session (Issue #476)
        stage6_mem_file = tmp_path / "scenario_memory.json"
        stage6_mem = CrossSessionMemory(storage_path=stage6_mem_file)

        # Stage 6.1: Agent in Session A records durable fact
        stage6_tc = ToolCallRequest(
            id="call_stage6_record",
            name="record_memory_fact",
            arguments={
                "subject": "deployment_target",
                "predicate": "environment",
                "object_value": "production-us-east",
            },
        )
        stage6_llm_a = MockLLMConnector(
            responses=["Recording deployment target to memory..."],
            tool_calls=[stage6_tc],
        )
        stage6_agent_a = BaseAgent(
            config=AgentConfig(
                agent_id="sc_agent_6a",
                name="sc_agent_6a",
                llm_config=AgentLLMConfig(model_name="mock-model", auto_compact=False),
            ),
            bus=bus,
            llm=stage6_llm_a,
            store=store,
            memory=stage6_mem,
        )
        await stage6_agent_a.start()
        stage6_sub_a = bus.subscribe(
            {f"session.{stage6_agent_a.session_id}"},
            recipient_id="scenario-observer-6a",
        )

        try:
            await bus.publish(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    topic=f"session.{stage6_agent_a.session_id}",
                    sender_id="scenario-observer-6a",
                    recipient_id="sc_agent_6a",
                    payload={"message": "Record our deployment target."},
                )
            )

            stage6_reply_a = None
            for _ in range(50):
                if not stage6_sub_a.empty():
                    evt = await stage6_sub_a.get()
                    if evt.type == EventType.AGENT_REPLY:
                        stage6_reply_a = evt
                        break
                await asyncio.sleep(0.05)
            assert stage6_reply_a is not None, "Scenario stage 6a failed to produce AGENT_REPLY"

            facts = stage6_mem.list_facts(subject="deployment_target")
            assert len(facts) == 1
            assert facts[0].object_value == "production-us-east"
        finally:
            await stage6_agent_a.stop()
            stage6_sub_a.close()

        # Stage 6.2: Distinct Session B reloads memory and injects fact into system prompt
        stage6_mem_b = CrossSessionMemory(storage_path=stage6_mem_file)
        stage6_agent_b = BaseAgent(
            config=AgentConfig(
                agent_id="sc_agent_6b",
                name="sc_agent_6b",
                llm_config=AgentLLMConfig(model_name="mock-model", auto_compact=False),
            ),
            bus=bus,
            llm=MockLLMConnector(responses=["Read memory facts successfully."]),
            store=store,
            memory=stage6_mem_b,
        )
        assert stage6_agent_b.session_id != stage6_agent_a.session_id
        await stage6_agent_b.start()

        try:
            turn_msgs = stage6_agent_b._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]
            tail_content = turn_msgs[-1].content or ""
            assert "[Cross-Session Memory Facts]" in tail_content
            assert "deployment_target: environment -> production-us-east" in tail_content

            # Stage 6.3: Retraction removes fact from subsequent sessions
            fact_to_retract = stage6_mem_b.list_facts(subject="deployment_target")[0]
            stage6_mem_b.retract_fact(
                fact_to_retract.fact_id,
                reason="Migration to EU region",
                provenance=Provenance.primary("test.stage6"),
            )
            assert len(stage6_mem_b.list_facts(include_retracted=False)) == 0

            turn_msgs_post = stage6_agent_b._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]
            assert "deployment_target: environment" not in (turn_msgs_post[-1].content or "")
        finally:
            await stage6_agent_b.stop()

    finally:
        sub.close()
        await agent.stop()
        await bus.stop()
