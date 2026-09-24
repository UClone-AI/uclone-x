"""Unit tests verifying that BaseAgent.spawn_subagent forwards the SessionStore (#224)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.models import FinishReason, ModelResponse, TokenUsage
from uclone_x.llm.protocols import LLMProviderProtocol


@pytest.mark.asyncio
async def test_spawn_subagent_forwards_store_and_allows_persistence(tmp_path: Path) -> None:
    """A spawned sub-agent receives the parent's SessionStore and can persist its session (#224)."""
    store = SessionStore(storage_dir=tmp_path)
    cfg = AgentConfig(
        agent_id="parent_agent",
        name="Parent Agent",
        workspace_dir=str(tmp_path),
        llm_config=AgentLLMConfig(model_name="mock"),
    )
    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="test", model="mock"),
        served_by=ServiceRef(provider="test", model="mock"),
    )
    llm = MagicMock(spec=LLMProviderProtocol)
    llm.generate = AsyncMock(
        return_value=ModelResponse(
            finish_reason=FinishReason.STOP,
            content="Subagent completed task",
            usage=TokenUsage(provider="test", input_tokens=5, output_tokens=5, total_tokens=10),
            provenance=prov,
        )
    )

    parent = BaseAgent(config=cfg, llm=llm, store=store)
    sub = await parent.spawn_subagent(
        role="coder",
        goal="write unit test",
    )

    # 1. Subagent inherited parent's store
    assert sub.store is store

    # 2. Subagent has an isolated session id
    sub_sid = sub.session_id
    assert sub_sid != parent.session_id
    assert sub_sid.startswith("sess_parent_agent_sub_")

    # 3. Subagent can execute turn and persist its session record
    turn_res = await sub.execute_turn("Do your task")
    assert turn_res.is_completed

    persisted = sub.persist_session()
    assert persisted.session_id == sub_sid

    # 4. Verify the persisted record is loadable from the shared store
    reloaded = store.load(sub_sid)
    assert reloaded is not None
    assert reloaded.agent_id == sub.agent_id
    assert len(reloaded.messages) > 0


@pytest.mark.asyncio
async def test_spawn_subagent_without_store_retains_none() -> None:
    """When parent has store=None, subagent also has store=None."""
    cfg = AgentConfig(
        agent_id="parent_nostore",
        name="No Store Parent",
        llm_config=AgentLLMConfig(model_name="mock"),
    )
    parent = BaseAgent(config=cfg, store=None)
    sub = await parent.spawn_subagent(role="worker", goal="do stuff")
    assert sub.store is None
