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
