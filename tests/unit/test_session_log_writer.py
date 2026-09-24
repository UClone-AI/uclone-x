from pathlib import Path

import pytest

from uclone_x.agent.session import SessionState, SessionStore, StaleSessionWriteError
from uclone_x.core.log_writer import RedactingLogWriter
from uclone_x.log.file_allocator import FileLogOffsetAllocator


def test_concurrent_writer_rejection(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/session.py :: if on_disk is not None and on_disk.revision != state.revision:
    Becomes: if on_disk is not None and on_disk.revision < state.revision:
    """
    log_writer = RedactingLogWriter(tmp_path / "log.jsonl")
    allocator = FileLogOffsetAllocator(tmp_path / "allocator")
    store = SessionStore(
        storage_dir=tmp_path / "sessions", log_writer=log_writer, log_allocator=allocator
    )

    state1 = SessionState(session_id="s1", agent_id="a1")
    store.save(state1, pending_events=[{"msg": "1"}])

    state2 = state1.model_copy()
    with pytest.raises(StaleSessionWriteError):
        store.save(state2, pending_events=[{"msg": "2"}])


def test_sync_between_snapshot_and_log(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/session.py :: next_offset"""
    log_writer = RedactingLogWriter(tmp_path / "log.jsonl")
    allocator = FileLogOffsetAllocator(tmp_path / "allocator")
    store = SessionStore(
        storage_dir=tmp_path / "sessions", log_writer=log_writer, log_allocator=allocator
    )

    state = SessionState(session_id="s1", agent_id="a1")
    saved = store.save(state, pending_events=[{"msg": "hello"}])

    assert allocator.next_offset("s1") == 2
    assert saved.revision == 1


def test_resilience_against_partial_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: src/uclone_x/agent/session.py :: tmp_path.unlink(missing_ok=True)"""
    log_writer = RedactingLogWriter(tmp_path / "log.jsonl")
    allocator = FileLogOffsetAllocator(tmp_path / "allocator")
    store = SessionStore(
        storage_dir=tmp_path / "sessions", log_writer=log_writer, log_allocator=allocator
    )

    state = SessionState(session_id="s1", agent_id="a1")
    saved = store.save(state)

    import os
    from typing import Any

    def mock_replace(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Crash")

    monkeypatch.setattr(os, "replace", mock_replace)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    with pytest.raises(RuntimeError):
        store.save(saved, pending_events=[{"msg": "crash"}])

    assert allocator.next_offset("s1") == 1
