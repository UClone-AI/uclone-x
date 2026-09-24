"""Unit tests verifying SessionState and _LiveSession updated_at preservation and update semantics (#223)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.models import ChatMessage, FinishReason, MessageRole, ModelResponse, TokenUsage
from uclone_x.llm.protocols import LLMProviderProtocol


def test_session_state_snapshot_preserves_original_updated_at(tmp_path: Path) -> None:
    """Verify that taking snapshots or hydrating an untouched session preserves updated_at (#223)."""
    past_timestamp = "2026-01-01T00:00:00+00:00"
    store = SessionStore(storage_dir=tmp_path)

    initial = SessionState(
        session_id="sess_past",
        agent_id="agent_past",
        messages=(ChatMessage(role=MessageRole.USER, content="old message"),),
        turn_counter=1,
        created_at=past_timestamp,
        updated_at=past_timestamp,
    )
    # Write directly to disk JSON to emulate a week-old session file
    target_path = store.session_path("sess_past")
    target_path.write_text(initial.model_dump_json(indent=2), encoding="utf-8")

    agent = BaseAgent(
        config=AgentConfig(
            agent_id="agent_past",
            name="Past Agent",
            workspace_dir=str(tmp_path),
            llm_config=AgentLLMConfig(model_name="mock"),
        ),
        store=store,
    )

    # 1. Hydrating session from store preserves the original updated_at
    hydrated = agent.hydrate_session("sess_past")
    assert hydrated is not None
    assert hydrated.updated_at == past_timestamp
    assert hydrated.created_at == past_timestamp

    # 2. Snapshotting via get_session does not regenerate updated_at
    snapshot1 = agent.get_session("sess_past")
    assert snapshot1.updated_at == past_timestamp
    assert snapshot1.created_at == past_timestamp

    # 3. Resetting updates updated_at while preserving created_at
    reset_state = agent.reset_session("sess_past")
    assert reset_state.updated_at != past_timestamp
    assert reset_state.created_at == past_timestamp


@pytest.mark.asyncio
async def test_session_updated_at_updates_on_turn_and_preserves_on_load(tmp_path: Path) -> None:
    """Verify that execute_turn updates updated_at and store load preserves it."""
    store = SessionStore(storage_dir=tmp_path)
    past_timestamp = "2026-01-01T00:00:00+00:00"

    initial_state = SessionState(
        session_id="sess_turn",
        agent_id="agent_turn",
        created_at=past_timestamp,
        updated_at=past_timestamp,
    )
    store.save(initial_state)

    # Reload from store without modification
    loaded = store.load("sess_turn")
    assert loaded is not None
    # save() stamps updated_at at write time, and subsequent loads keep that exact timestamp
    saved_time = loaded.updated_at

    await asyncio.sleep(0.01)
    loaded_again = store.load("sess_turn")
    assert loaded_again is not None
    assert loaded_again.updated_at == saved_time

    # Drive a turn through BaseAgent and verify updated_at changes
    cfg = AgentConfig(
        agent_id="agent_turn",
        name="Turn Agent",
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
            content="Turn completed",
            usage=TokenUsage(provider="test", input_tokens=5, output_tokens=5, total_tokens=10),
            provenance=prov,
        )
    )

    agent = BaseAgent(config=cfg, llm=llm, store=store)
    res = await agent.execute_turn("Test input")
    assert res.is_completed

    # Live session updated_at should now be newer than saved_time
    snap = agent.get_session()
    assert snap.updated_at != past_timestamp
