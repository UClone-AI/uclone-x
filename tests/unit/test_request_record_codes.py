"""Each place a rebuild stops raises the code for its own kind of gap (#1911).

A head words `RequestRecordError.code` in its reader's language (#1907), so a site that
raises the wrong code shows the wrong sentence. `before_capture` and `chain_broken` are
pinned in `test_turn_trace.py`; every other raise site in `agent/request_record.py` is
pinned here, on a record built by hand to have exactly that one gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent.request_record import (
    RequestRecordError,
    rebuild_epoch_conversations,
    rebuild_requests,
)
from uclone_x.agent.session import ContextSnapshot, SessionState, SessionStore, content_digest
from uclone_x.core.context_state import ContextEntry, ContextEpoch, ContextForm
from uclone_x.core.session_log import SessionLogProvenance, logged_message, new_entry
from uclone_x.llm.models import ChatMessage, MessageRole

_SID = "sess_codes"
_TOOLS = "[]"
_IDENTITY = "You are a test clone."
_SLOW = "slow context"
_TURN = ""


def _snapshot(tools: str = _TOOLS) -> ContextSnapshot:
    return ContextSnapshot(
        turn_index=1,
        tools_digest=content_digest(tools),
        identity_digest=content_digest(_IDENTITY),
        slow_context_digest=content_digest(_SLOW),
        system_message=True,
        turn_context_digest=content_digest(_TURN),
        model=None,
        temperature=0.0,
        max_tokens=None,
        auto_compact=False,
        compaction_threshold_tokens=1000,
    )


def _request(snapshot: str | None, **extra: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "REQUEST_CONTEXT",
        "request": 1,
        "step": 1,
        "snapshot": snapshot,
        "kept_message_count": 0,
        "appended_messages": [{"role": "user", "content": "hello"}],
    }
    event.update(extra)
    return event


def _store_bodies(store: SessionStore, *texts: str) -> None:
    for text in texts:
        store.save_context_body(_SID, content_digest(text), text)


def _rebuild_code(store: SessionStore, state: SessionState, event: dict[str, Any]) -> str:
    with pytest.raises(RequestRecordError) as raised:
        rebuild_requests(store, state, [event])
    return raised.value.code


# --------------------------------------------------------------------------------------
# rebuild_requests
# --------------------------------------------------------------------------------------


def test_a_request_whose_layer_body_is_gone_is_coded_body_missing(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: code="body_missing",  # a layer body a request's snapshot names
    Becomes: code="unreadable",  # a layer body a request's snapshot names
    """
    store = SessionStore(tmp_path)
    snapshot = _snapshot()
    state = SessionState(session_id=_SID, agent_id="a", context_snapshots=(snapshot,))

    assert _rebuild_code(store, state, _request(snapshot.snapshot_id)) == "body_missing"


def test_a_request_whose_snapshot_is_gone_is_coded_snapshot_missing(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: code="snapshot_missing",
    Becomes: code="body_missing",
    """
    store = SessionStore(tmp_path)
    state = SessionState(session_id=_SID, agent_id="a")

    assert _rebuild_code(store, state, _request(_snapshot().snapshot_id)) == "snapshot_missing"


def test_a_request_with_no_readable_delta_is_coded_unreadable(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: code="unreadable",  # a request's conversation delta
    Becomes: code="chain_broken",  # a request's conversation delta
    """
    store = SessionStore(tmp_path)
    snapshot = _snapshot()
    state = SessionState(session_id=_SID, agent_id="a", context_snapshots=(snapshot,))
    event = _request(snapshot.snapshot_id)
    del event["kept_message_count"]

    assert _rebuild_code(store, state, event) == "unreadable"


def test_a_request_whose_bodies_do_not_parse_is_coded_unreadable(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: code="unreadable",  # a request's bodies or messages do not parse
    Becomes: code="body_missing",  # a request's bodies or messages do not parse
    """
    store = SessionStore(tmp_path)
    tools = "not a tool list"
    _store_bodies(store, tools, _IDENTITY, _SLOW, _TURN)
    snapshot = _snapshot(tools)
    state = SessionState(session_id=_SID, agent_id="a", context_snapshots=(snapshot,))

    assert _rebuild_code(store, state, _request(snapshot.snapshot_id)) == "unreadable"
    # The same record with a tool list that parses rebuilds: what raised is the parse.
    _store_bodies(store, _TOOLS)
    fine = SessionState(session_id=_SID, agent_id="a", context_snapshots=(_snapshot(),))
    assert len(rebuild_requests(store, fine, [_request(_snapshot().snapshot_id)])) == 1


# --------------------------------------------------------------------------------------
# rebuild_epoch_conversations (epoch_renderer)
# --------------------------------------------------------------------------------------

_HELLO = ChatMessage(role=MessageRole.USER, content="hello")


def _epoch_state(
    store: SessionStore,
    *,
    body: str | None,
    entry: ContextEntry,
) -> SessionState:
    """A session of one logged user message, whose one epoch shows `entry`.

    `body` is what is stored under the log entry's digest: `None` stores nothing.
    """
    logged = logged_message(_HELLO)
    if body is not None:
        store.save_context_body(_SID, logged.digest, body)
    log_entry = new_entry(0, logged, turn=1, provenance=SessionLogProvenance.RECORDED)
    epoch = ContextEpoch(number=0, turn=1, step=1, opened_by=("start",), entries=(entry,))
    return SessionState(
        session_id=_SID,
        agent_id="a",
        messages=(_HELLO,),
        session_log=(log_entry,),
        context_epochs=(epoch,),
    )


def _epoch_code(store: SessionStore, state: SessionState) -> str:
    with pytest.raises(RequestRecordError) as raised:
        rebuild_epoch_conversations(store, state)
    return raised.value.code


def test_an_epoch_naming_an_entry_the_log_lacks_is_coded_log_entry_missing(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: code="log_entry_missing",
    Becomes: code="body_missing",
    """
    store = SessionStore(tmp_path)
    state = _epoch_state(
        store,
        body=logged_message(_HELLO).body,
        entry=ContextEntry(entry="e5", form=ContextForm.FULL),
    )

    assert _epoch_code(store, state) == "log_entry_missing"


def test_an_epoch_whose_entry_body_is_gone_is_coded_body_missing(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: code="body_missing",  # a log entry's body
    Becomes: code="log_entry_missing",  # a log entry's body
    """
    store = SessionStore(tmp_path)
    state = _epoch_state(store, body=None, entry=ContextEntry(entry="e0", form=ContextForm.FULL))

    assert _epoch_code(store, state) == "body_missing"


def test_an_epoch_whose_entry_body_does_not_parse_is_coded_unreadable(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: code="unreadable",  # a log entry's body does not parse
    Becomes: code="body_missing",  # a log entry's body does not parse
    """
    store = SessionStore(tmp_path)
    state = _epoch_state(
        store, body="not a message", entry=ContextEntry(entry="e0", form=ContextForm.FULL)
    )

    assert _epoch_code(store, state) == "unreadable"


def test_an_epoch_that_does_not_match_its_entries_is_coded_epoch_mismatch(
    tmp_path: Path,
) -> None:
    """A person's message listed as a stub: its form is not the one its entry records.

    Killed by: src/uclone_x/agent/request_record.py :: code="epoch_mismatch",
    Becomes: code="unreadable",
    """
    store = SessionStore(tmp_path)
    body = logged_message(_HELLO).body
    stub = _epoch_state(store, body=body, entry=ContextEntry(entry="e0", form=ContextForm.STUB))

    assert _epoch_code(store, stub) == "epoch_mismatch"
    # Listed as it was sent, the same entry renders: what raised is the form.
    full = _epoch_state(store, body=body, entry=ContextEntry(entry="e0", form=ContextForm.FULL))
    assert rebuild_epoch_conversations(store, full) == [[_HELLO]]
