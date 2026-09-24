"""Tests for session saturation detection and compaction UI integration (#827).

Enforces P0 (Non-Expert Operability) and P6 (Explicit State & Zero Guesswork):
- /api/sessions exposes `is_saturated` flag when session turns >= 20.
- Saturation triggers compaction or fresh session workflows without silent performance degradation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from starlette.testclient import TestClient

from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.llm.models import ChatMessage, MessageRole
from uclone_x.ui.app import create_ui_app


def test_sessions_listing_reports_saturation_flag(tmp_path: Path) -> None:
    """`/api/sessions` reports saturation from the session's *active* context (#872).

    This test previously read saturation off `SessionState.turn_counter`, which is the
    lifetime figure Core deliberately retains across compaction (P5). Encoding it that
    way made the listing agree with a signal that can never clear: a session compacted
    down to two turns still announced itself as saturated. The rule is now the turns
    actually held in the window, and the lifetime counter is reported beside it rather
    than in place of it.

    Killed by: src/uclone_x/ui/app.py :: active = count_active_turns(state.messages)
    Becomes: active = state.turn_counter
    """
    storage_dir = tmp_path / "sessions"
    storage_dir.mkdir(parents=True, exist_ok=True)
    store = SessionStore(storage_dir=storage_dir / "core")

    def _exchange(count: int) -> list[ChatMessage]:
        turns: list[ChatMessage] = []
        for i in range(count):
            turns.append(ChatMessage(role=MessageRole.USER, content=f"q{i}"))
            turns.append(ChatMessage(role=MessageRole.ASSISTANT, content=f"a{i}"))
        return turns

    # 5 turns held, 5 taken: not saturated.
    s1 = SessionState.seed(session_id="sess_normal", agent_id="champion")
    store.save(s1.with_messages(_exchange(5), turn_counter=5))

    # 25 turns held, 25 taken: saturated.
    s2 = SessionState.seed(session_id="sess_saturated", agent_id="champion")
    store.save(s2.with_messages(_exchange(25), turn_counter=25))

    # 25 turns taken but only 2 still held -- the state a compaction leaves behind. The
    # old rule called this saturated, which is the defect users met as a banner that
    # would not clear.
    s3 = SessionState.seed(session_id="sess_compacted", agent_id="champion")
    store.save(s3.with_messages(_exchange(2), turn_counter=25))

    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir)
    client = TestClient(app)

    res = client.get("/api/sessions")
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    sessions = cast(list[dict[str, Any]], data["sessions"])

    normal_sess = next(s for s in sessions if s["session_id"] == "sess_normal")
    assert normal_sess["is_saturated"] is False
    assert normal_sess["turn_counter"] == 5
    assert normal_sess["active_turns"] == 5

    sat_sess = next(s for s in sessions if s["session_id"] == "sess_saturated")
    assert sat_sess["is_saturated"] is True
    assert sat_sess["turn_counter"] == 25
    assert sat_sess["active_turns"] == 25

    compacted_sess = next(s for s in sessions if s["session_id"] == "sess_compacted")
    assert compacted_sess["turn_counter"] == 25
    assert compacted_sess["active_turns"] == 2
    assert compacted_sess["is_saturated"] is False


def test_reconstruct_history_reaches_the_shared_presentable_rule(tmp_path: Path) -> None:
    """Rehydration keeps a ledger and drops a plain system anchor (#872, P6).

    `reconstruct_history` makes this decision on persisted transcript *dicts*, where
    `_is_presentable` cannot reach, so it carried an independent copy of the same rule.
    Two copies that agree today is exactly the defect the single predicate was extracted
    to prevent, one layer down; both sides now reach it through `_is_presentable_role`.

    Killed by: src/uclone_x/ui/app.py :: if not _is_presentable_role(role_str, bool(item_dict.get("compaction_ledger", False))):
    Becomes: if False:
    """
    from uclone_x.ui.app import AgentSessionManager

    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw: list[object] = [
        {"role": "system", "content": "You are Champion.", "sender": "system"},
        {"role": "system", "content": "[Compacted] earlier turns", "compaction_ledger": True},
        {"role": "user", "content": "hello"},
    ]
    history, transcript = mgr.reconstruct_history(raw, session_id="sess_872_rehydrate")

    # The anchor is not conversation and does not come back as one; the ledger does.
    assert [(m.role.value, m.compaction_ledger) for m in history] == [
        ("system", True),
        ("user", False),
    ]
    assert history[0].content == "[Compacted] earlier turns"
    # The presentation view is untouched -- it keeps every persisted entry verbatim.
    assert len(transcript) == 3


def test_save_session_record_changes_the_view_only_after_the_record_lands(
    tmp_path: Path,
) -> None:
    """The in-memory view must not run ahead of the record on disk (#872, P6).

    `save_session_record` is the single write path for both stores, and it assigned the
    in-memory transcript *before* the disk write. An `OSError` from the write therefore
    left the view rewritten over a stale record, and the compaction path re-raises —
    answering 5xx over a view it had already mutated, which is the one outcome a caller
    reading that status would rule out.

    Both halves matter, and the in-memory store is asserted directly rather than through
    `get_session_history`: that accessor falls back to the record on disk when the
    session is not cached, so it reports the right answer even when the memory write
    never happened at all — which would leave this test passing over a store that is
    simply never written.

    Killed by: src/uclone_x/ui/app.py :: self._session_messages[session_id] = list(messages)
    Becomes:
    """
    from unittest.mock import patch

    from uclone_x.ui.app import AgentSessionManager

    mgr = AgentSessionManager(storage_dir=tmp_path)
    in_memory = mgr._session_messages  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    session_id = "sess_872_atomicity"
    first: list[dict[str, Any]] = [{"role": "user", "content": "kept"}]

    mgr.save_session_record(session_id=session_id, agent_id="champion", messages=first, turns=1)
    # One call writes both stores.
    assert in_memory[session_id] == first
    assert mgr.get_session_history("champion", session_id) == first

    doomed: list[dict[str, Any]] = [{"role": "user", "content": "never persisted"}]
    with patch("builtins.open", side_effect=OSError("disk full")):
        try:
            mgr.save_session_record(
                session_id=session_id, agent_id="champion", messages=doomed, turns=2
            )
        except OSError:
            pass
        else:  # pragma: no cover - the write is patched to fail
            raise AssertionError("expected the failing write to raise")

    # The record never changed, so neither may the view the user is served.
    assert in_memory[session_id] == first
    assert mgr.get_session_history("champion", session_id) == first
    record = mgr.load_session_record(session_id)
    assert record is not None
    assert record["messages"] == first
