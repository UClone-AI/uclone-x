"""A record this build cannot read is moved aside before anything is written over it (#1844).

The case that matters is not a damaged file: it is a conversation a *newer* build wrote, with a
field this build refuses under `extra="forbid"`. Before #1844 the session store read such a
record as absent, seeded a fresh session, and the first save wrote over it -- one rollback,
one conversation gone. What these pin:

* the headline case: a record with a field this build does not know survives `save`
  byte for byte at `<name>.unreadable-<UTC time>`, and a second save does not touch it;
* `load` alone changes nothing on disk; `delete` moves the record aside instead of unlinking;
* a rename that fails stops the write, and the record stays where it was;
* the tool-artifact reaper spares a session that has only a set-aside copy;
* the shared helper never replaces an earlier copy, even one set aside in the same instant;
* memory and settings use the same helper, so neither loses a second unreadable copy or an
  unreadable settings file;
* a room row says, once and in plain words, that the seat's earlier conversation was kept.

Added by #1860:

* retention: only the newest `KEEP_SET_ASIDE` copies of a record are kept, so two builds
  taking turns saving one conversation do not fill the disk;
* a delete keeps what a kept copy refers to -- event log, context bodies, tool results;
* a Settings save that sets the file aside logs the paths only, and Settings says so;
* clearing a conversation keeps an unreadable UI transcript aside instead of unlinking it;
* the web rooms wire the seats' session store into the room, so a 1:1 chat says so too.

Added by #1877:

* retention orders copies by when they were set aside (their change time), not by the stamp
  in the name, so a clock set back no longer makes each set-aside delete the one before;
* an empty name -- claimed, or left by a crash -- is not a kept copy, and one a day old goes;
* a copy of a record that is gone is deleted thirty days after it was set aside (author's
  choice), and what it referred to with the last of them;
* a key saved over an unreadable settings file keeps the file aside and is saved, instead
  of failing with the file's path as the reason;
* `ucx loop` and ACP say, as the other heads do, that a record was kept.

Only builds from this change on behave this way; an older build still writes over what it
cannot read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from uclone_x.agent.clone_builder import ontology_map
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.core.set_aside import KEEP_SET_ASIDE, set_aside_copies, set_aside_unreadable
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import SessionRecordUnreadableError
from uclone_x.llm.connectors import saved_choice
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.room.models import ParticipantKind, RoomMessage
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import MentionSelector
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from uclone_x.ui.app import AgentSessionManager

SID = "sess_from_a_newer_build"


def _newer_builds_record(store: SessionStore, session_id: str = SID) -> bytes:
    """A record as a newer build writes it: valid today, plus one field this build refuses."""
    document = json.loads(SessionState.seed(session_id, "agent").model_dump_json())
    document["a_field_from_a_newer_build"] = {"kept": True}
    path = store.session_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document, indent=2).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _plain(text: str) -> None:
    """Copy for a person: no file location, no class name, no validator output."""
    assert "/" not in text and "\\" not in text, text
    assert not re.search(r"[A-Za-z]*(Error|Exception)\b", text), text
    for internal in ("json", "SessionState", "extra", "field", "validation"):
        assert internal not in text, (internal, text)


class TestTheSessionStoreNeverWritesOverAnUnreadableRecord:
    def test_a_newer_builds_record_survives_save_byte_for_byte(self, tmp_path: Path) -> None:
        """Load (absent), save a fresh session over the id: the old bytes are kept aside.

        A second save does not touch the copy, and the set-aside is reported once.

        Killed by: src/uclone_x/agent/session.py :: self._set_aside_unreadable(path, unreadable)
        Becomes: pass
        Killed by: src/uclone_x/agent/session.py :: self._set_aside.add(state.session_id)
        Becomes: pass
        Killed by: src/uclone_x/agent/session.py :: self._set_aside.discard(session_id)
        Becomes: pass
        """
        store = SessionStore(tmp_path / "sessions")
        original = _newer_builds_record(store)
        path = store.session_path(SID)

        assert store.load(SID) is None
        saved = store.save(SessionState.seed(SID, "agent"))

        (aside,) = set_aside_copies(path)
        assert aside.parent == path.parent
        assert aside.name.startswith(f"{path.name}.unreadable-")
        assert aside.read_bytes() == original
        assert store.load(SID) == saved
        assert store.take_set_aside(SID) is True
        assert store.take_set_aside(SID) is False

        store.save(saved)
        assert set_aside_copies(path) == (aside,)
        assert aside.read_bytes() == original
        assert store.take_set_aside(SID) is False
        # Not listed as a session of its own, and not loadable by any id.
        assert store.list_session_ids() == (SID,)

    def test_load_alone_changes_nothing_on_disk(self, tmp_path: Path) -> None:
        """A read -- a GET, a diagnostic -- leaves the record for a newer build to find."""
        store = SessionStore(tmp_path / "sessions")
        original = _newer_builds_record(store)

        assert store.load(SID) is None
        assert store.load(SID) is None

        assert store.session_path(SID).read_bytes() == original
        assert set_aside_copies(store.session_path(SID)) == ()
        assert store.take_set_aside(SID) is False

    def test_delete_moves_an_unreadable_record_aside_instead_of_unlinking(
        self, tmp_path: Path
    ) -> None:
        """Clearing a conversation this build showed as empty frees the id and keeps the bytes.

        Killed by: src/uclone_x/agent/session.py :: self._set_aside_unreadable(path, cause=unreadable)
        Becomes: path.unlink()
        """
        store = SessionStore(tmp_path / "sessions")
        original = _newer_builds_record(store)
        path = store.session_path(SID)

        assert store.delete(SID) is True

        assert not path.exists()
        (aside,) = set_aside_copies(path)
        assert aside.read_bytes() == original

    def test_a_rename_that_fails_stops_the_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the record cannot be moved, nothing is written over it, and the refusal is plain.

        Killed by: src/uclone_x/agent/session.py :: raise refusal from exc
        Becomes: return
        """
        store = SessionStore(tmp_path / "sessions")
        original = _newer_builds_record(store)
        path = store.session_path(SID)

        def refuse(target: Path, **_: object) -> Path:
            raise PermissionError(13, "Permission denied", str(target))

        monkeypatch.setattr("uclone_x.agent.session.set_aside_unreadable", refuse)

        with pytest.raises(SessionRecordUnreadableError) as caught:
            store.save(SessionState.seed(SID, "agent"))

        assert path.read_bytes() == original
        assert caught.value.path == path
        assert "a_field_from_a_newer_build" in caught.value.cause
        _plain(str(caught.value))
        assert store.take_set_aside(SID) is False

    def test_the_artifact_reaper_spares_a_session_kept_only_aside(self, tmp_path: Path) -> None:
        """Its history names stored tool results by handle; a build that can read it needs them.

        Killed by: src/uclone_x/agent/session.py :: return path.is_file() or bool(kept_copies(path))
        Becomes: return path.is_file()
        """
        sessions = tmp_path / "sessions"
        artifacts = tmp_path / "artifacts"
        store = SessionStore(sessions, artifacts_dir=artifacts)
        _newer_builds_record(store)
        set_aside_unreadable(store.session_path(SID))
        kept = artifacts / SID / "result.txt"
        kept.parent.mkdir(parents=True)
        kept.write_text("a tool result", encoding="utf-8")
        hours_ago = time.time() - 6 * 3600
        os.utime(kept.parent, (hours_ago, hours_ago))
        orphan = artifacts / "sess_nobody" / "result.txt"
        orphan.parent.mkdir(parents=True)
        orphan.write_text("nobody's", encoding="utf-8")
        os.utime(orphan.parent, (hours_ago, hours_ago))

        SessionStore(sessions, artifacts_dir=artifacts)  # the reaper runs on construction

        assert kept.read_text(encoding="utf-8") == "a tool result"
        assert not orphan.exists()


class TestTheSharedHelper:
    def test_a_copy_set_aside_in_the_same_instant_does_not_replace_the_first(
        self, tmp_path: Path
    ) -> None:
        """`os.rename` replaces an existing destination; the helper claims a free name first.

        Killed by: src/uclone_x/core/set_aside.py :: aside = path.with_name(base if n == 0 else f"{base}-{n}")
        Becomes: aside = path.with_name(base)
        """
        path = tmp_path / "record.json"
        instant = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)

        path.write_bytes(b"first")
        first = set_aside_unreadable(path, now=instant)
        path.write_bytes(b"second")
        second = set_aside_unreadable(path, now=instant)

        assert first != second
        assert first.read_bytes() == b"first"
        assert second.read_bytes() == b"second"
        assert set_aside_copies(path) == (first, second)
        assert not path.exists()
        # The marker follows the whole name, so `*.json` never lists a copy as a record.
        assert list(tmp_path.glob("*.json")) == []

    def test_memory_keeps_two_unreadable_documents_set_aside_in_one_second(
        self, tmp_path: Path
    ) -> None:
        """The old whole-second name replaced the first quarantined copy with the second."""
        path = tmp_path / "memory.json"
        memory = CrossSessionMemory(storage_path=path)

        path.write_text("{ first unreadable", encoding="utf-8")
        memory.load()
        path.write_text("{ second unreadable", encoding="utf-8")
        memory.load()

        kept = sorted(p.read_text(encoding="utf-8") for p in set_aside_copies(path))
        assert kept == ["{ first unreadable", "{ second unreadable"]


class TestSettings:
    def test_an_unreadable_settings_file_is_kept_before_a_whole_state_save(
        self, tmp_path: Path
    ) -> None:
        """The dashboard's Settings save replaces an unreadable file; the old one is kept aside.

        Killed by: src/uclone_x/llm/connectors/saved_choice.py :: aside = set_aside_unreadable(target)
        Becomes: aside = target
        """
        target = tmp_path / "settings.json"
        target.write_text('{"llm_provider": "ollama", "llm_api_keys": {', encoding="utf-8")
        original = target.read_bytes()

        saved_choice.update_settings_file(
            {"llm_model": "qwen3:8b"},
            path=target,
            replace_unreadable_with={"llm_provider": "ollama"},
        )

        assert json.loads(target.read_text(encoding="utf-8")) == {
            "llm_provider": "ollama",
            "llm_model": "qwen3:8b",
        }
        (aside,) = set_aside_copies(target)
        assert aside.read_bytes() == original

    def test_a_settings_file_that_cannot_be_moved_is_not_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "settings.json"
        target.write_text("not json", encoding="utf-8")

        def refuse(path: Path, **_: object) -> Path:
            raise PermissionError(13, "Permission denied", str(path))

        monkeypatch.setattr(saved_choice, "set_aside_unreadable", refuse)

        with pytest.raises(OSError):
            saved_choice.update_settings_file(
                {"llm_model": "qwen3:8b"}, path=target, replace_unreadable_with={}
            )
        assert target.read_text(encoding="utf-8") == "not json"


class TestTheRoomSaysSo:
    @pytest.mark.asyncio
    async def test_the_seats_next_row_says_its_earlier_conversation_was_kept(
        self, tmp_path: Path
    ) -> None:
        """Once, on the row of the turn whose save set the record aside.

        Killed by: src/uclone_x/room/orchestrator.py :: session_set_aside=self._session_set_aside(speaker),
        Becomes: session_set_aside=False,
        Killed by: src/uclone_x/room/orchestrator.py :: return self._sessions.take_set_aside(speaker.session_id)
        Becomes: return False
        """
        rooms = RoomStore(tmp_path / "rooms")
        service = RoomService(rooms)
        service.create("Kept", room_id="room_kept")
        service.add_participant("room_kept", "user", kind=ParticipantKind.HUMAN)
        service.add_participant("room_kept", "scout")
        seat = next(p for p in service.get("room_kept").participants if p.id == "scout")
        sessions = SessionStore(tmp_path / "sessions")
        original = _newer_builds_record(sessions, seat.session_id)
        resolver = RoomAgentResolver(
            HostDependencies(
                bus=EventBus(),
                llm=MockLLMConnector(default_response="noted"),
                tools=ToolRegistry(),
                tracer=TelemetryTracer(),
                store=sessions,
            ),
            ontology_for=ontology_map(),
        )
        orchestrator = RoomOrchestrator(
            store=rooms,
            selectors=(MentionSelector(),),
            resolver=resolver,
            sessions=sessions,
        )

        state = await orchestrator.post("room_kept", "user", "@scout hello")
        row = state.transcript[-1]
        assert row.sender_id == "scout" and row.error is None and row.persist_error is None
        assert row.session_set_aside is True
        (aside,) = set_aside_copies(sessions.session_path(seat.session_id))
        assert aside.read_bytes() == original

        state = await orchestrator.post("room_kept", "user", "@scout again")
        assert state.transcript[-1].session_set_aside is False
        assert aside.read_bytes() == original

    def test_the_flag_is_left_out_of_a_saved_row_while_false(self) -> None:
        """An older build refuses unknown keys, so a row that says nothing carries no new key.

        Killed by: src/uclone_x/room/models.py :: exclude_if=lambda set_aside: not set_aside,
        Becomes: exclude_if=lambda set_aside: False,
        """
        quiet = RoomMessage(seq=1, sender_id="scout", content="hi")
        said = RoomMessage(seq=1, sender_id="scout", content="hi", session_set_aside=True)

        assert "session_set_aside" not in quiet.model_dump(mode="json")
        assert said.model_dump(mode="json")["session_set_aside"] is True


def _newer_builds_record_saying(store: SessionStore, marker: int) -> bytes:
    """`_newer_builds_record`, told apart from the others by `marker`."""
    document = json.loads(SessionState.seed(SID, "agent").model_dump_json())
    document["a_field_from_a_newer_build"] = {"round": marker}
    path = store.session_path(SID)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document).encode("utf-8")
    path.write_bytes(raw)
    return raw


class TestRetention:
    def test_two_builds_alternately_saving_keep_only_the_newest_copies(
        self, tmp_path: Path
    ) -> None:
        """The newer build writes a field the older refuses; the older sets it aside and saves.

        Round after round, each set-aside used to leave one more whole copy. Now the newest
        `KEEP_SET_ASIDE` are kept -- the three most recent rounds, byte for byte.

        Killed by: src/uclone_x/core/set_aside.py :: KEEP_SET_ASIDE = 3
        Becomes: KEEP_SET_ASIDE = 99
        Killed by: src/uclone_x/core/set_aside.py :: _delete_older_copies(path, keep=aside)
        Becomes: path.parent
        """
        store = SessionStore(tmp_path / "sessions")
        path = store.session_path(SID)
        written: list[bytes] = []

        for marker in range(8):
            written.append(_newer_builds_record_saying(store, marker))  # the newer build
            store.save(SessionState.seed(SID, "agent"))  # the older build, over it

        copies = set_aside_copies(path)
        assert KEEP_SET_ASIDE == 3
        assert [c.read_bytes() for c in copies] == written[-3:]
        assert store.load(SID) is not None

    def test_the_newest_are_told_by_their_counter_not_by_the_names_spelling(
        self, tmp_path: Path
    ) -> None:
        """Twelve copies set aside in one instant: `-10` and `-11` are newer than `-9`.

        Killed by: src/uclone_x/core/set_aside.py :: aside = path.with_name(base if n == 0 else f"{base}-{n}")
        Becomes: aside = path.with_name(base)
        """
        path = tmp_path / "record.json"
        instant = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
        for n in range(12):
            path.write_bytes(f"copy {n}".encode())
            set_aside_unreadable(path, now=instant)

        assert [c.read_bytes() for c in set_aside_copies(path)] == [
            b"copy 9",
            b"copy 10",
            b"copy 11",
        ]

    def test_a_name_this_module_did_not_write_is_never_deleted(self, tmp_path: Path) -> None:
        """Only `<name>.unreadable-<stamp>[-<n>]` is ours to delete.

        Killed by: src/uclone_x/core/set_aside.py :: if entry == keep or key is None or not entry.is_file():
        Becomes: if entry == keep or not entry.is_file():
        """
        path = tmp_path / "record.json"
        foreign = tmp_path / "record.json.unreadable-kept-by-hand"
        foreign.write_bytes(b"the person's")
        for n in range(5):
            path.write_bytes(f"copy {n}".encode())
            set_aside_unreadable(path)

        assert foreign.read_bytes() == b"the person's"
        ours = [c for c in set_aside_copies(path) if c != foreign]
        assert [c.read_bytes() for c in ours] == [b"copy 2", b"copy 3", b"copy 4"]


class TestADeleteKeepsWhatAKeptCopyRefersTo:
    def test_event_log_context_bodies_and_tool_results_stay_with_the_copy(
        self, tmp_path: Path
    ) -> None:
        """Restored by hand, the copy still finds its tool results and its context bodies.

        Killed by: src/uclone_x/agent/session.py :: if self._has_kept_copy(path):
        Becomes: if False:
        """
        artifacts = tmp_path / "artifacts"
        store = SessionStore(tmp_path / "sessions", artifacts_dir=artifacts)
        _newer_builds_record(store)
        body = "a layer body the kept copy's snapshot names"
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        store.save_context_body(SID, digest, body)
        log = store.event_log_path(SID)
        assert log is not None
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('{"event": "kept"}\n', encoding="utf-8")
        result = artifacts / SID / "result.txt"
        result.parent.mkdir(parents=True)
        result.write_text("a tool result", encoding="utf-8")

        assert store.delete(SID) is True

        assert not store.session_path(SID).exists()
        assert len(set_aside_copies(store.session_path(SID))) == 1
        assert store.load_context_body(SID, digest) == body
        assert log.read_text(encoding="utf-8") == '{"event": "kept"}\n'
        assert result.read_text(encoding="utf-8") == "a tool result"

    def test_with_no_copy_kept_a_delete_still_removes_them(self, tmp_path: Path) -> None:
        """A readable record deleted with nothing kept: its history goes with it, as before."""
        artifacts = tmp_path / "artifacts"
        store = SessionStore(tmp_path / "sessions", artifacts_dir=artifacts)
        store.save(SessionState.seed(SID, "agent"))
        result = artifacts / SID / "result.txt"
        result.parent.mkdir(parents=True)
        result.write_text("a tool result", encoding="utf-8")

        assert store.delete(SID) is True

        assert not result.exists()
        assert set_aside_copies(store.session_path(SID)) == ()


def _manager(tmp_path: Path) -> AgentSessionManager:
    from uclone_x.ui.app import AgentSessionManager

    return AgentSessionManager(
        storage_dir=tmp_path / "sessions",
        llm=MockLLMConnector(default_response="noted"),
        workspace_dir=tmp_path / "workspace",
    )


class TestSettingsSaysSo:
    def test_a_settings_save_that_kept_the_file_aside_is_logged_and_reported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Logged with both paths and nothing the file holds; Settings reports it from then on.

        Killed by: src/uclone_x/ui/app.py :: self._settings_set_aside = True  # a Settings save kept the file aside
        Becomes: self._settings_set_aside = False
        Killed by: src/uclone_x/llm/connectors/saved_choice.py :: logger.warning(_SET_ASIDE_LOG, target, aside)
        Becomes: pass
        """
        manager = _manager(tmp_path)
        target = manager.settings_file
        secret = "sk-a-key-1860-that-must-not-be-logged"
        target.write_text(
            '{"llm_provider": "openai", "llm_api_keys": {"openai": "' + secret + '"},',
            encoding="utf-8",
        )
        assert manager.get_settings()["settings_set_aside"] is False

        with caplog.at_level(logging.INFO):
            manager.update_settings(ui_language="en")

        (aside,) = set_aside_copies(target)
        assert secret in aside.read_text(encoding="utf-8")
        assert manager.get_settings()["settings_set_aside"] is True
        said = [r.getMessage() for r in caplog.records if str(aside) in r.getMessage()]
        assert said, caplog.text
        assert secret not in caplog.text

    def test_a_readable_settings_file_reports_nothing(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        manager.update_settings(ui_language="en")
        manager.update_settings(ui_language="ko")

        assert manager.get_settings()["settings_set_aside"] is False
        assert set_aside_copies(manager.settings_file) == ()


class TestTheUiTranscript:
    def test_clearing_keeps_an_unreadable_transcript_aside(self, tmp_path: Path) -> None:
        """It predates #1844 -- an unlink, the one copy gone -- and it is now kept like the rest.

        Killed by: src/uclone_x/ui/app.py :: if transcript is None and path.is_file():
        Becomes: if False:
        """
        manager = _manager(tmp_path)
        path = manager.get_session_path(SID)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"session_id": "' + SID.encode() + b'", "messages": [')

        manager.clear_session_history("agent", SID)

        assert not path.exists()
        (aside,) = set_aside_copies(path)
        assert aside.read_bytes().endswith(b'"messages": [')

    def test_clearing_a_readable_transcript_still_removes_it(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        path = manager.get_session_path(SID)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"session_id": SID, "messages": []}), encoding="utf-8")

        manager.clear_session_history("agent", SID)

        assert not path.exists()
        assert set_aside_copies(path) == ()


class TestTheWebRoomSaysSo:
    @pytest.mark.asyncio
    async def test_a_one_seat_room_on_the_dashboard_says_the_record_was_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 1:1 web chat is a one-seat room built by `RoomStack`; its row carries the notice.

        Killed by: src/uclone_x/ui/rooms.py :: sessions=self._session_mgr.core_store,
        Becomes: sessions=None,
        """
        from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
        from uclone_x.ui.rooms import RoomStack

        monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
        manager = _manager(tmp_path)
        stack = RoomStack(manager)
        stack.service.create(title="One to one", room_id="room_one")
        stack.service.add_participant("room_one", "user", kind=ParticipantKind.HUMAN)
        state = stack.service.add_participant("room_one", "scout", kind=ParticipantKind.AGENT)
        seat = next(p for p in state.participants if p.id == "scout")
        original = _newer_builds_record(manager.core_store, seat.session_id)

        posted = await stack.orchestrator(state).post("room_one", "user", "@scout hello")

        row = posted.transcript[-1]
        assert row.sender_id == "scout" and row.error is None, row
        assert row.session_set_aside is True
        (aside,) = set_aside_copies(manager.core_store.session_path(seat.session_id))
        assert aside.read_bytes() == original


class TestTheReviewOf1867:
    """Data-loss cases the review of #1867 found in the set-aside itself."""

    def test_a_clock_set_back_does_not_delete_the_copy_just_made(self, tmp_path: Path) -> None:
        """Behind three later-stamped copies the new one sorts oldest; it is still kept.

        Before the fix retention deleted it at once, and the caller logged a path to nothing.

        Killed by: src/uclone_x/core/set_aside.py :: if entry == keep or key is None or not entry.is_file():
        Becomes: if key is None or not entry.is_file():
        """
        path = tmp_path / "record.json"
        for hour in (13, 14, 15):
            path.write_bytes(f"later {hour}".encode())
            set_aside_unreadable(path, now=datetime(2026, 9, 28, hour, tzinfo=UTC))
        path.write_bytes(b"the newest, after the clock went back")

        aside = set_aside_unreadable(path, now=datetime(2026, 9, 28, 12, tzinfo=UTC))

        assert aside.read_bytes() == b"the newest, after the clock went back"
        kept = set_aside_copies(path)
        assert len(kept) == KEEP_SET_ASIDE
        assert aside in kept
        assert b"later 13" not in {copy.read_bytes() for copy in kept}

    def test_a_name_claimed_since_the_listing_is_never_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two processes list, both pick one name; the second must not rename over the first.

        Process A set the original aside and wrote a record of its own; process B, which
        listed before A renamed, still thinks the name is free. Before the fix B's rename
        replaced A's copy: the original record, gone.

        Killed by: src/uclone_x/core/set_aside.py :: os.close(os.open(aside, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        Becomes: os.close(os.open(aside, os.O_CREAT | os.O_WRONLY, 0o600))
        """
        from uclone_x.core import set_aside

        path = tmp_path / "record.json"
        instant = datetime(2026, 9, 28, 12, tzinfo=UTC)
        path.write_bytes(b"the original")
        first = set_aside_unreadable(path, now=instant)  # process A
        path.write_bytes(b"what process A wrote next")

        # Process B's listing was taken before A's rename: it saw no copy at this instant.
        def _listed_before_a(path: Path, marker: str, stamp: str) -> int:
            return -1

        monkeypatch.setattr(set_aside, "_highest_counter", _listed_before_a)

        second = set_aside_unreadable(path, now=instant)  # process B

        assert second != first
        assert first.read_bytes() == b"the original"
        assert second.read_bytes() == b"what process A wrote next"

    def test_a_failed_rename_leaves_no_claimed_name_behind(self, tmp_path: Path) -> None:
        """The record vanished before the rename: the error propagates, the placeholder goes.

        Killed by: src/uclone_x/core/set_aside.py :: aside.unlink(missing_ok=True)  # the empty placeholder, not a copy
        Becomes: pass
        """
        path = tmp_path / "record.json"

        with pytest.raises(FileNotFoundError):
            set_aside_unreadable(path, now=datetime(2026, 9, 28, 12, tzinfo=UTC))

        assert list(tmp_path.iterdir()) == []

    def test_retention_leaves_a_name_another_process_has_claimed(self, tmp_path: Path) -> None:
        """An empty copy may be a placeholder waiting for its rename; deleting it frees the name.

        Killed by: src/uclone_x/core/set_aside.py :: if entry.stat().st_size == 0:  # not counted: nothing is kept in it
        Becomes: if False:
        """
        path = tmp_path / "record.json"
        claimed = path.with_name(f"{path.name}.unreadable-20260928T000000000000Z")
        claimed.touch()  # the oldest name, claimed by another process mid set-aside
        for hour in (13, 14, 15, 16):
            path.write_bytes(f"copy {hour}".encode())
            set_aside_unreadable(path, now=datetime(2026, 9, 28, hour, tzinfo=UTC))

        assert claimed.exists()

    def test_a_delete_keeps_the_history_when_the_folder_cannot_be_listed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unable to tell whether a copy is kept, a delete keeps what one would need.

        Killed by: src/uclone_x/agent/session.py :: return True  # cannot tell, so keep what a copy may need
        Becomes: return False
        """
        from uclone_x.agent import session as session_module

        artifacts = tmp_path / "artifacts"
        store = SessionStore(tmp_path / "sessions", artifacts_dir=artifacts)
        store.save(SessionState.seed(SID, "agent"))
        result = artifacts / SID / "result.txt"
        result.parent.mkdir(parents=True)
        result.write_text("a tool result", encoding="utf-8")

        def _cannot_list(path: Path) -> tuple[Path, ...]:
            raise PermissionError("the folder cannot be listed")

        monkeypatch.setattr(session_module, "kept_copies", _cannot_list)

        assert store.delete(SID) is True

        assert not store.session_path(SID).exists()
        assert result.read_text(encoding="utf-8") == "a tool result"


def _set_aside_long_ago(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make the named copies (all copies, with no names) read as set aside in 1970.

    A change time cannot be set back from a test; `_changed_at` is the one place it is read.
    """
    from uclone_x.core import set_aside as set_aside_module

    real = set_aside_module._changed_at  # pyright: ignore[reportPrivateUsage]

    def _changed_at(entry: Path) -> float:
        if not names or entry.name in names:
            return 0.0
        return real(entry)

    monkeypatch.setattr(set_aside_module, "_changed_at", _changed_at)


def _deleted_with_its_history_kept(tmp_path: Path) -> tuple[SessionStore, Path, Path, Path]:
    """A newer build's record deleted here: its copy, event log and tool result are kept."""
    sessions, artifacts = tmp_path / "sessions", tmp_path / "artifacts"
    store = SessionStore(sessions, artifacts_dir=artifacts)
    _newer_builds_record(store)
    log = store.event_log_path(SID)
    assert log is not None
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('{"event": "kept"}\n', encoding="utf-8")
    result = artifacts / SID / "result.txt"
    result.parent.mkdir(parents=True)
    result.write_text("a tool result", encoding="utf-8")
    assert store.delete(SID) is True
    (copy,) = set_aside_copies(store.session_path(SID))
    return store, copy, log, result


class TestTheReviewOf1877:
    """What the review of #1860 found: empty names counted, name order, copies kept forever."""

    def test_the_newest_are_told_by_when_they_were_set_aside_not_by_the_stamp(
        self, tmp_path: Path
    ) -> None:
        """A clock set back names the newest copies oldest; retention keeps them anyway.

        By name, the two copies made after the clock went back sorted before three made
        earlier, so each set-aside deleted the one before it.

        Killed by: src/uclone_x/core/set_aside.py :: dated.sort(key=lambda item: (item[0], item[1]))
        Becomes: dated.sort(key=lambda item: item[1])
        """
        path = tmp_path / "record.json"
        for hour in (13, 14, 15):
            path.write_bytes(f"before {hour}".encode())
            set_aside_unreadable(path, now=datetime(2026, 9, 28, hour, tzinfo=UTC))
            time.sleep(0.02)  # change times a coarse clock still tells apart
        for hour in (1, 2):
            path.write_bytes(f"after the clock went back {hour}".encode())
            set_aside_unreadable(path, now=datetime(2026, 9, 28, hour, tzinfo=UTC))
            time.sleep(0.02)

        assert {c.read_bytes() for c in set_aside_copies(path)} == {
            b"before 15",
            b"after the clock went back 1",
            b"after the clock went back 2",
        }

    def test_copies_set_aside_in_one_instant_are_told_by_their_counter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Change times that tie fall back on the name: `-10` and `-11` are newer than `-9`.

        Killed by: src/uclone_x/core/set_aside.py :: return matched.group(1), int(matched.group(2) or 0)
        Becomes: return matched.group(1), 0
        Killed by: src/uclone_x/core/set_aside.py :: return max(counters, default=-1)
        Becomes: return -1
        """
        from uclone_x.core import set_aside as set_aside_module

        def _one_instant(entry: Path) -> float:
            return 0.0

        monkeypatch.setattr(set_aside_module, "_changed_at", _one_instant)
        path = tmp_path / "record.json"
        instant = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
        for n in range(12):
            path.write_bytes(f"copy {n}".encode())
            set_aside_unreadable(path, now=instant)

        assert sorted(c.read_bytes() for c in set_aside_copies(path)) == sorted(
            [b"copy 9", b"copy 10", b"copy 11"]
        )

    def test_an_empty_name_is_not_a_kept_copy(self, tmp_path: Path) -> None:
        """A claimed name, or one a crash left, holds nothing; a delete still clears history.

        Before, an empty placeholder made `delete` keep the event log and tool results for
        a copy that did not exist, and made the reaper spare them.

        Killed by: src/uclone_x/core/set_aside.py :: if entry.stat().st_size == 0:  # a claimed name, or one a crash left
        Becomes: if False:
        """
        artifacts = tmp_path / "artifacts"
        store = SessionStore(tmp_path / "sessions", artifacts_dir=artifacts)
        store.save(SessionState.seed(SID, "agent"))
        path = store.session_path(SID)
        path.with_name(f"{path.name}.unreadable-20260928T000000000000Z").touch()
        result = artifacts / SID / "result.txt"
        result.parent.mkdir(parents=True)
        result.write_text("a tool result", encoding="utf-8")

        assert store.delete(SID) is True

        assert not result.exists()

    def test_an_empty_name_left_for_a_day_is_deleted_and_a_fresh_one_is_left(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A store opening removes a crashed set-aside's placeholder, not one being renamed onto.

        Killed by: src/uclone_x/core/set_aside.py :: _delete_stale_placeholder(entry, clock)
        Becomes: pass
        Killed by: src/uclone_x/core/set_aside.py :: if now - _changed_at(entry) < PLACEHOLDER_MAX_AGE:
        Becomes: if False:
        """
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        stale = sessions / f"{SID}.json.unreadable-20260101T000000000000Z"
        fresh = sessions / f"{SID}.json.unreadable-20260928T000000000000Z"
        stale.touch()
        fresh.touch()
        _set_aside_long_ago(monkeypatch, stale.name)

        SessionStore(sessions)

        assert not stale.exists()
        assert fresh.exists()

    def test_setting_a_record_aside_also_removes_a_day_old_empty_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retention, not only a store opening, removes a crashed set-aside's placeholder.

        Killed by: src/uclone_x/core/set_aside.py :: _delete_stale_placeholder(entry, now)
        Becomes: pass
        """
        record = tmp_path / "record.json"
        record.write_text("{not json", encoding="utf-8")
        stale = tmp_path / "record.json.unreadable-20260101T000000000000Z"
        stale.touch()
        _set_aside_long_ago(monkeypatch, stale.name)

        aside = set_aside_unreadable(record)

        assert not stale.exists()
        assert set_aside_copies(record) == (aside,)

    def test_a_deleted_records_copy_expires_and_takes_its_history_with_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Thirty days after it was set aside, a copy whose record is gone is deleted (#1877).

        Author's choice. A deleted record is never set aside again, so retention never ran
        for it and its copy, event log and tool results were kept forever.

        Killed by: src/uclone_x/agent/session.py :: self._expire_kept_copies()
        Becomes: pass
        Killed by: src/uclone_x/agent/session.py :: self.clear_event_log(session_id)  # the last copy is gone
        Becomes: pass
        Killed by: src/uclone_x/agent/session.py :: cleanup_session_artifacts(self._artifacts_dir, session_id)
        Becomes: pass
        """
        _, copy, log, result = _deleted_with_its_history_kept(tmp_path)
        _set_aside_long_ago(monkeypatch)

        SessionStore(tmp_path / "sessions", artifacts_dir=tmp_path / "artifacts")

        assert not copy.exists()
        assert not log.exists()
        assert not result.exists()

    def test_a_copy_within_thirty_days_or_of_a_record_saved_again_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only an old copy of a record that is gone expires; retention bounds the others.

        Killed by: src/uclone_x/core/set_aside.py :: if clock - _changed_at(entry) < max_age or record_exists(record):
        Becomes: if record_exists(record):
        Killed by: src/uclone_x/core/set_aside.py :: EXPIRED_COPY_MAX_AGE = 30 * 24 * 60 * 60.0
        Becomes: EXPIRED_COPY_MAX_AGE = 0.0
        """
        store, copy, log, result = _deleted_with_its_history_kept(tmp_path)
        SessionStore(tmp_path / "sessions", artifacts_dir=tmp_path / "artifacts")
        assert copy.exists() and log.exists() and result.exists()  # set aside just now

        store.save(SessionState.seed(SID, "agent"))  # the record, back under its id
        _set_aside_long_ago(monkeypatch)
        SessionStore(tmp_path / "sessions", artifacts_dir=tmp_path / "artifacts")

        assert copy.exists()

    def test_the_history_stays_while_a_younger_copy_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two copies of a deleted record, one expired: what the other refers to stays.

        Killed by: src/uclone_x/agent/session.py :: if record.is_file() or kept_copies(record):
        Becomes: if record.is_file():
        """
        store, old, log, result = _deleted_with_its_history_kept(tmp_path)
        _newer_builds_record(store)
        assert store.delete(SID) is True
        (young,) = [c for c in set_aside_copies(store.session_path(SID)) if c != old]
        _set_aside_long_ago(monkeypatch, old.name)

        SessionStore(tmp_path / "sessions", artifacts_dir=tmp_path / "artifacts")

        assert not old.exists()
        assert young.exists() and log.exists() and result.exists()

    def test_an_expired_ui_transcript_copy_is_deleted_when_the_dashboard_starts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clearing keeps an unreadable UI transcript aside; thirty days on, it goes too.

        Killed by: src/uclone_x/ui/app.py :: expire_set_aside(self._transcript_dir, suffix=".json")
        Becomes: pass
        """
        manager = _manager(tmp_path)
        path = manager.get_session_path(SID)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"session_id": "' + SID.encode() + b'", "messages": [')
        manager.clear_session_history("agent", SID)
        (aside,) = set_aside_copies(path)
        _set_aside_long_ago(monkeypatch)

        _manager(tmp_path)

        assert not aside.exists()


class TestAKeySavedOverAnUnreadableSettingsFile:
    def test_the_key_is_saved_and_the_file_kept_aside(self, tmp_path: Path) -> None:
        """Settings used to fail with the file's path as its reason (#1860, #1877).

        Now the unreadable file is kept aside as a Settings save without a key keeps it,
        the key is saved into the new file, and Settings says the earlier file was kept.

        Killed by: src/uclone_x/ui/app.py :: self._save_key(*new_key, held=held)
        Becomes: save_api_key(new_key[0], new_key[1], path=self._settings_file)
        Killed by: src/uclone_x/ui/app.py :: self._settings_set_aside = True  # and Settings says so
        Becomes: pass
        """
        manager = _manager(tmp_path)
        target = manager.settings_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"llm_provider": "openai",', encoding="utf-8")

        manager.update_settings(llm_api_key="sk-new-1877", llm_api_key_provider="openai")

        saved = json.loads(target.read_text(encoding="utf-8"))
        assert "sk-new-1877" in json.dumps(saved)
        (aside,) = set_aside_copies(target)
        assert aside.read_text(encoding="utf-8") == '{"llm_provider": "openai",'
        assert manager.get_settings()["settings_set_aside"] is True

    def test_a_file_unreadable_again_fails_without_its_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another writer made it unreadable again: the failure names no file.

        Killed by: src/uclone_x/ui/app.py :: raise OSError("the settings file could not be read, so the key was not saved") from exc
        Becomes: raise
        """
        from uclone_x.ui import app as app_module

        manager = _manager(tmp_path)
        target = manager.settings_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"llm_provider": "openai",', encoding="utf-8")

        def _unreadable(provider: str, key: str, *, path: Path) -> None:
            raise ValueError(f"The settings file at {path} cannot be read.")

        monkeypatch.setattr(app_module, "save_api_key", _unreadable)

        with pytest.raises(OSError) as caught:
            manager.update_settings(llm_api_key="sk-new-1877", llm_api_key_provider="openai")

        assert not isinstance(caught.value, ValueError)
        _plain(str(caught.value))

    def test_a_model_check_that_fails_leaves_the_earlier_settings_in_the_new_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key stays saved; the model the check refused is not written with it.

        Before, the file that replaced the unreadable one held the model being tried, while
        Settings took it back.

        Killed by: src/uclone_x/ui/app.py :: replace_unreadable_with=held
        Becomes: replace_unreadable_with=self._persisted_settings()
        """
        manager = _manager(tmp_path)
        target = manager.settings_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"llm_provider": "openai",', encoding="utf-8")
        before = manager.get_settings()["llm_model"]

        def _refused(*args: object, **kwargs: object) -> None:
            raise RuntimeError("that model did not answer")

        monkeypatch.setattr(manager, "_build_configured_llm", _refused)

        with pytest.raises(RuntimeError):
            manager.update_settings(
                llm_model="a-model-that-fails-1877",
                llm_api_key="sk-new-1877",
                llm_api_key_provider="openai",
            )

        text = target.read_text(encoding="utf-8")
        assert "sk-new-1877" in text
        assert "a-model-that-fails-1877" not in text
        assert manager.get_settings()["llm_model"] == before

    def test_a_key_refused_over_a_readable_file_is_refused_as_before(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only an unreadable file is kept aside; a refusal about the key itself stands.

        Killed by: src/uclone_x/ui/app.py :: raise refusal  # the file was readable: the refusal is about the key, as before
        Becomes: pass
        """
        from uclone_x.ui import app as app_module

        manager = _manager(tmp_path)
        target = manager.settings_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"llm_provider": "openai"}', encoding="utf-8")

        def _refused(provider: str, key: str, *, path: Path) -> None:
            raise ValueError("That key cannot be saved.")

        monkeypatch.setattr(app_module, "save_api_key", _refused)

        with pytest.raises(ValueError, match="That key cannot be saved."):
            manager.update_settings(llm_api_key="sk-new-1877", llm_api_key_provider="openai")

        assert set_aside_copies(target) == ()
        assert json.loads(target.read_text(encoding="utf-8")) == {"llm_provider": "openai"}
        assert manager.get_settings()["settings_set_aside"] is False
