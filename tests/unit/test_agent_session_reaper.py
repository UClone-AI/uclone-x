"""Unit tests for crash-orphaned session temp file reaper and fsync durability (#219, #257, #269)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from uclone_x.agent.session import (
    MAX_PID,
    SessionState,
    SessionStore,
    is_pid_alive,
    reap_orphaned_temp_files,
)
from uclone_x.llm.models import ChatMessage, MessageRole
from uclone_x.ui.app import AgentSessionManager


def test_is_pid_alive_detection() -> None:
    """is_pid_alive identifies current process as alive and unused high PIDs as dead."""
    assert is_pid_alive(os.getpid()) is True
    assert is_pid_alive(999999) is False
    assert is_pid_alive(-1) is False
    assert is_pid_alive(0) is False


def test_reap_orphaned_temp_files_with_dead_pid(tmp_path: Path) -> None:
    """Orphaned temp files from dead PIDs are unlinked while live ones survive."""
    valid_record = tmp_path / "sess_a.json"
    valid_record.write_text('{"session_id": "sess_a"}', encoding="utf-8")

    dead_tmp_1 = tmp_path / "sess_a.json.tmp.999999.5733fd3e"
    dead_tmp_1.write_text("partial 1", encoding="utf-8")

    dead_tmp_2 = tmp_path / "sess_b.json.tmp.999998.abcdef12"
    dead_tmp_2.write_text("partial 2", encoding="utf-8")

    live_tmp = tmp_path / f"sess_a.json.tmp.{os.getpid()}.12345678"
    live_tmp.write_text("in-flight write", encoding="utf-8")

    # Foreign non-temp file
    foreign_file = tmp_path / "other.txt"
    foreign_file.write_text("unrelated", encoding="utf-8")

    reaped = reap_orphaned_temp_files(tmp_path, max_age_seconds=3600.0)
    assert reaped == 2

    assert valid_record.is_file()
    assert foreign_file.is_file()
    assert not dead_tmp_1.exists()
    assert not dead_tmp_2.exists()
    assert live_tmp.is_file()


def test_reap_orphaned_temp_files_with_age_threshold(tmp_path: Path) -> None:
    """Old temp files are reaped when their modification age exceeds max_age_seconds."""
    live_pid = os.getpid()
    old_tmp = tmp_path / f"sess_c.json.tmp.{live_pid}.aaaaaaaa"
    old_tmp.write_text("old crashed write from current pid long ago", encoding="utf-8")

    old_time = time.time() - 7200.0  # 2 hours ago
    os.utime(old_tmp, (old_time, old_time))

    recent_tmp = tmp_path / f"sess_c.json.tmp.{live_pid}.bbbbbbbb"
    recent_tmp.write_text("recent in-flight write", encoding="utf-8")

    reaped = reap_orphaned_temp_files(tmp_path, max_age_seconds=300.0)
    assert reaped == 1
    assert not old_tmp.exists()
    assert recent_tmp.is_file()


def test_session_store_init_automatically_reaps_orphans(tmp_path: Path) -> None:
    """SessionStore.__init__ automatically clears dead crash-orphaned temp files."""
    dead_tmp = tmp_path / "sess_d.json.tmp.999999.deadbeef"
    dead_tmp.write_text("orphaned data", encoding="utf-8")

    store = SessionStore(storage_dir=tmp_path)
    assert not dead_tmp.exists()
    assert store.list_session_ids() == ()


def test_list_session_ids_ignores_temp_files(tmp_path: Path) -> None:
    """list_session_ids strictly matches *.json and ignores all *.tmp.* artifacts."""
    store = SessionStore(storage_dir=tmp_path)
    state = SessionState(
        session_id="valid_session",
        agent_id="agent-1",
        messages=(ChatMessage(role=MessageRole.USER, content="hello"),),
    )
    store.save(state)

    # Plant active and dead temp files
    (tmp_path / "valid_session.json.tmp.999999.11111111").write_text("t", encoding="utf-8")
    (tmp_path / f"valid_session.json.tmp.{os.getpid()}.22222222").write_text("t", encoding="utf-8")

    ids = store.list_session_ids()
    assert ids == ("valid_session",)


def test_reaper_with_nonexistent_directory(tmp_path: Path) -> None:
    """reap_orphaned_temp_files cleanly returns 0 for nonexistent directories."""
    assert reap_orphaned_temp_files(tmp_path / "nonexistent") == 0


def test_session_store_save_fsync_durability(tmp_path: Path) -> None:
    """SessionStore.save atomically flushes, fsyncs and commits record to disk."""
    store = SessionStore(storage_dir=tmp_path)
    state = SessionState(
        session_id="durability_test",
        agent_id="agent-1",
        messages=(ChatMessage(role=MessageRole.USER, content="fsync test"),),
    )
    saved = store.save(state)
    assert saved.revision == 1

    record_path = store.session_path("durability_test")
    assert record_path.is_file()

    # Verify reload
    reloaded = store.load("durability_test")
    assert reloaded is not None
    assert reloaded.session_id == "durability_test"
    assert reloaded.messages[0].content == "fsync test"


def test_committed_session_record_with_tmp_pid_in_id_survives_reaper(tmp_path: Path) -> None:
    """Committed session record whose id contains .tmp.<pid>. is never reaped (#269)."""
    record = tmp_path / "chat.tmp.999999.deadbeef.json"
    record.write_text('{"session_id": "chat.tmp.999999.deadbeef"}', encoding="utf-8")

    reaped = reap_orphaned_temp_files(tmp_path, max_age_seconds=0.0)
    assert reaped == 0
    assert record.is_file()
    assert record.read_text(encoding="utf-8") == '{"session_id": "chat.tmp.999999.deadbeef"}'


def test_genuine_temp_file_with_tmp_pid_in_id_is_reaped(tmp_path: Path) -> None:
    """Genuine orphaned temp file for a session ID with .tmp.<pid>. is reaped (#269)."""
    record = tmp_path / "chat.tmp.999999.deadbeef.json"
    record.write_text('{"session_id": "chat.tmp.999999.deadbeef"}', encoding="utf-8")

    orphan_tmp_1 = tmp_path / "chat.tmp.999999.deadbeef.tmp.999999.12345678"
    orphan_tmp_1.write_text("partial orphan 1", encoding="utf-8")

    orphan_tmp_2 = tmp_path / "chat.tmp.999999.deadbeef.json.tmp.999999.87654321"
    orphan_tmp_2.write_text("partial orphan 2", encoding="utf-8")

    live_tmp = tmp_path / f"chat.tmp.999999.deadbeef.json.tmp.{os.getpid()}.abcdef12"
    live_tmp.write_text("in-flight write", encoding="utf-8")

    reaped = reap_orphaned_temp_files(tmp_path, max_age_seconds=3600.0)
    assert reaped == 2
    assert record.is_file()
    assert not orphan_tmp_1.exists()
    assert not orphan_tmp_2.exists()
    assert live_tmp.is_file()


def test_session_store_init_preserves_committed_tmp_pid_records(tmp_path: Path) -> None:
    """SessionStore.__init__ does not unlink committed records containing .tmp.<pid>. (#269)."""
    store = SessionStore(storage_dir=tmp_path)
    state = SessionState(
        session_id="chat.tmp.999999.deadbeef",
        agent_id="agent-1",
        messages=(ChatMessage(role=MessageRole.USER, content="tmp id test"),),
    )
    store.save(state)

    # Plant a dead temp file beside it
    dead_tmp = tmp_path / "chat.tmp.999999.deadbeef.json.tmp.999999.11223344"
    dead_tmp.write_text("crashed temp file", encoding="utf-8")

    # Re-instantiate SessionStore to trigger reaper on __init__
    reloaded_store = SessionStore(storage_dir=tmp_path)
    assert not dead_tmp.exists()

    loaded = reloaded_store.load("chat.tmp.999999.deadbeef")
    assert loaded is not None
    assert loaded.session_id == "chat.tmp.999999.deadbeef"
    assert loaded.messages[0].content == "tmp id test"
    assert reloaded_store.list_session_ids() == ("chat.tmp.999999.deadbeef",)


def test_agent_session_manager_init_preserves_committed_tmp_pid_records(tmp_path: Path) -> None:
    """AgentSessionManager.__init__ does not unlink committed records containing .tmp.<pid>. (#269)."""
    core_dir = tmp_path / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    core_record = core_dir / "chat.tmp.999999.deadbeef.json"
    core_record.write_text('{"session_id": "chat.tmp.999999.deadbeef"}', encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir(parents=True, exist_ok=True)
    ui_transcript = ui_dir / "chat.tmp.999999.deadbeef.json"
    ui_transcript.write_text(
        '{"session_id": "chat.tmp.999999.deadbeef", "turns": []}', encoding="utf-8"
    )

    # Plant dead temp files in both directories
    core_dead_tmp = core_dir / "chat.tmp.999999.deadbeef.json.tmp.999999.dead0001"
    core_dead_tmp.write_text("core orphan", encoding="utf-8")
    ui_dead_tmp = ui_dir / "chat.tmp.999999.deadbeef.json.tmp.999999.dead0002"
    ui_dead_tmp.write_text("ui orphan", encoding="utf-8")

    # Construct UI manager (triggers both session store and transcript dir reaper)
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)

    assert core_record.is_file()
    assert ui_transcript.is_file()
    assert not core_dead_tmp.exists()
    assert mgr.get_session_path("chat.tmp.999999.deadbeef").is_file()


def test_is_pid_alive_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range PIDs return False safely without calling os.kill (#592).

    Killed by: src/uclone_x/agent/session.py :: if pid <= 0 or pid > _MAX_PID:
    """
    calls: list[int] = []

    def fake_kill(pid: int, sig: int) -> None:
        calls.append(pid)

    monkeypatch.setattr(os, "kill", fake_kill)

    assert is_pid_alive(MAX_PID + 1) is False
    assert is_pid_alive(99999999999999999999) is False
    assert is_pid_alive(2**31) is False
    assert calls == []


def test_reap_orphaned_temp_files_with_junk_pid_does_not_abort_real_orphan(
    tmp_path: Path,
) -> None:
    """A junk PID exceeding pid_t does not abort reaping real orphans (#592).

    Killed by: src/uclone_x/agent/session.py :: uninterpretable_pid = pid > _MAX_PID
    """
    junk_tmp = tmp_path / "sess_junk.json.tmp.99999999999999999999.deadbeef"
    junk_tmp.write_text("junk content", encoding="utf-8")

    real_orphan = tmp_path / "sess_real.json.tmp.999999.cafebabe"
    real_orphan.write_text("real orphan", encoding="utf-8")

    reaped = reap_orphaned_temp_files(tmp_path, max_age_seconds=300.0)
    assert reaped == 1
    assert not real_orphan.exists()
    assert junk_tmp.is_file()


def test_reap_orphaned_temp_files_with_junk_pid_collected_when_aged(
    tmp_path: Path,
) -> None:
    """An out-of-range PID file is not leaked forever; it is collected when aged (#592).

    Killed by: src/uclone_x/agent/session.py :: if not is_orphan and max_age_seconds > 0:
    """
    junk_tmp = tmp_path / "sess_junk.json.tmp.99999999999999999999.deadbeef"
    junk_tmp.write_text("old junk content", encoding="utf-8")

    old_time = time.time() - 600.0
    os.utime(junk_tmp, (old_time, old_time))

    reaped = reap_orphaned_temp_files(tmp_path, max_age_seconds=300.0)
    assert reaped == 1
    assert not junk_tmp.exists()
