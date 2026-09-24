"""Core session state, its persistence boundary, and multi-session isolation (#183, P5, P8).

Named by issue #183's acceptance criteria. Covers `uclone_x.agent.session` — the single
Core-owned session store that P8 requires all session state to live behind.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import textwrap
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
from typer.testing import CliRunner

from uclone_x.agent import BaseAgent
from uclone_x.agent.base import VALID_TRANSITIONS
from uclone_x.agent.bootstrap import agent_config_for_persona, bootstrap_session
from uclone_x.agent.models import AgentConfig, AgentLLMConfig, AgentState, PersonaDefinition
from uclone_x.agent.persona_registry import get_default_persona_registry
from uclone_x.agent.protocols import BaseAgentProtocol
from uclone_x.agent.session import (
    CORE_RECORD_SUBDIR,
    SESSION_STORAGE_DIR_ENV_VAR,
    UI_TRANSCRIPT_SUBDIR,
    CompactionResult,
    SessionState,
    SessionStore,
    resolve_session_path,
    validate_session_id,
    verify_record_identity,
)
from uclone_x.cli import main as cli_main
from uclone_x.cli.commands import run as run_module
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.engine.protocols import EventSubscriptionProtocol
from uclone_x.errors import (
    InvalidStateTransitionError,
    PathTraversalError,
    SessionIdCollisionError,
    SessionMutationDuringTurnError,
    SessionStoreNotConfiguredError,
    SessionSwitchWhileRunningError,
    StaleSessionWriteError,
)
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.models import (
    ChatMessage,
    CompactionOutcome,
    FinishReason,
    LedgerSource,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenUsage,
)
from uclone_x.llm.protocols import ContextCompactorProtocol, LLMProviderProtocol
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.ui.app import AgentSessionManager, create_ui_app

SYSTEM_PROMPT = "You are a careful assistant."


def _clone_persona() -> PersonaDefinition:
    """The shipped `clone` persona, which the bootstrap tests seed a session from."""
    persona = get_default_persona_registry().get_persona("clone")
    assert persona is not None
    return persona


def _dialogue() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content="first question"),
        ChatMessage(role=MessageRole.ASSISTANT, content="first answer"),
    )


# --------------------------------------------------------------------------------------
# SessionState: one reset semantics, defined once
# --------------------------------------------------------------------------------------


def test_seed_emits_a_system_message_only_for_a_non_empty_prompt() -> None:
    """`seed` reproduces `BaseAgent.__init__`, which appends only a truthy prompt.

    The CLI REPL's `/reset` seeded `content=config.system_prompt or ""` and so inserted
    an *empty* `SYSTEM` message — a state `__init__` cannot produce. Pinning both arms
    here is what stops that spelling coming back.
    """
    with_prompt = SessionState.seed("s1", "agent-a", system_prompt=SYSTEM_PROMPT)
    assert len(with_prompt.messages) == 1
    assert with_prompt.messages[0].role == MessageRole.SYSTEM
    assert with_prompt.messages[0].content == SYSTEM_PROMPT

    without_prompt = SessionState.seed("s2", "agent-a", system_prompt="")
    assert without_prompt.messages == ()


def test_reset_restores_the_system_prompt_and_zeroes_the_turn_counter() -> None:
    """Reset must re-seed the system prompt: nothing else recomposes it.

    `BaseAgent._prepare_turn_messages` recomposes the P7 asserted invariants every turn,
    but `config.system_prompt` is written into history once in `__init__`. So a reset
    that only cleared messages would silently strip the agent's instructions.
    """
    state = SessionState(
        session_id="s1",
        agent_id="agent-a",
        messages=_dialogue(),
        turn_counter=7,
    )

    reset = state.reset(system_prompt=SYSTEM_PROMPT)

    assert reset.turn_counter == 0
    assert len(reset.messages) == 1
    assert reset.messages[0].content == SYSTEM_PROMPT
    assert reset.messages[0].role == MessageRole.SYSTEM
    # Same session, so its creation time survives the reset.
    assert reset.created_at == state.created_at
    assert reset.session_id == "s1"
    assert reset.agent_id == "agent-a"


def test_reset_without_a_prompt_leaves_no_empty_system_message() -> None:
    state = SessionState(session_id="s1", agent_id="agent-a", messages=_dialogue(), turn_counter=3)
    reset = state.reset(system_prompt="")
    assert reset.messages == ()
    assert reset.turn_counter == 0


def test_session_state_is_frozen_and_forbids_unknown_fields() -> None:
    """P8 strict typing: a turn produces a new state rather than mutating a shared one."""
    state = SessionState.seed("s1", "agent-a", SYSTEM_PROMPT)
    with pytest.raises(ValueError):
        state.turn_counter = 5  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError):
        SessionState(
            session_id="s1",
            agent_id="agent-a",
            unexpected="x",  # pyright: ignore[reportCallIssue]
        )


def test_with_messages_replaces_the_sequence_and_advances_the_counter() -> None:
    state = SessionState.seed("s1", "agent-a", SYSTEM_PROMPT)
    advanced = state.with_messages(_dialogue(), turn_counter=2)
    assert advanced.messages == _dialogue()
    assert advanced.turn_counter == 2
    # The original is untouched.
    assert state.turn_counter == 0
    assert len(state.messages) == 1


def test_with_messages_leaves_the_counter_alone_when_not_given() -> None:
    state = SessionState.seed("s1", "agent-a", SYSTEM_PROMPT).with_messages(
        _dialogue(), turn_counter=4
    )
    same_counter = state.with_messages(_dialogue()[:2])
    assert same_counter.turn_counter == 4


# --------------------------------------------------------------------------------------
# SessionStore: the single persistence boundary
# --------------------------------------------------------------------------------------


def test_save_and_load_round_trip_preserves_messages_and_counter(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    state = SessionState(
        session_id="sess_a",
        agent_id="agent-a",
        messages=_dialogue(),
        turn_counter=3,
    )

    store.save(state)
    loaded = store.load("sess_a")

    assert loaded is not None
    assert loaded.turn_counter == 3
    assert loaded.messages == _dialogue()
    assert loaded.agent_id == "agent-a"


def test_save_preserves_the_compaction_ledger_flag(tmp_path: Path) -> None:
    """The flag `#201` added must survive persistence, or a reloaded session's compactor
    would fail to recognise its own prior ledgers and regrow the resident block (#196)."""
    store = SessionStore(storage_dir=tmp_path)
    ledger = ChatMessage(
        role=MessageRole.SYSTEM,
        content="[Context Auto-Compacted Summary: Heuristic Session Progress Ledger]",
        compaction_ledger=True,
    )
    store.save(
        SessionState(
            session_id="sess_a",
            agent_id="agent-a",
            messages=(ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT), ledger),
        )
    )

    loaded = store.load("sess_a")

    assert loaded is not None
    assert loaded.messages[0].compaction_ledger is False
    assert loaded.messages[1].compaction_ledger is True


def test_load_returns_none_for_an_absent_session(tmp_path: Path) -> None:
    assert SessionStore(storage_dir=tmp_path).load("never_written") is None


def test_save_stamps_updated_at_and_returns_the_stamped_state(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    state = SessionState(session_id="sess_a", agent_id="agent-a", updated_at="1970-01-01T00:00:00")
    stamped = store.save(state)
    assert stamped.updated_at != "1970-01-01T00:00:00"
    reloaded = store.load("sess_a")
    assert reloaded is not None
    assert reloaded.updated_at == stamped.updated_at


def test_save_is_atomic_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    store.save(SessionState.seed("sess_a", "agent-a", SYSTEM_PROMPT))
    assert [p.name for p in tmp_path.iterdir()] == ["sess_a.json"]


def test_save_works_without_a_running_event_loop(tmp_path: Path) -> None:
    """Headless synchronous callers must be able to persist a session.

    `AgentSessionManager.save_session_record` builds its temporary filename from
    `asyncio.get_running_loop().time()`, which raises `RuntimeError` when no loop is
    running — so the UI's writer is unusable from the CLI REPL and the A2A path, the very
    callers P8 says must reach session state without the UI. This test is synchronous on
    purpose: it fails if that spelling is reintroduced here.
    """
    store = SessionStore(storage_dir=tmp_path)
    saved = store.save(SessionState.seed("sess_sync", "agent-a", SYSTEM_PROMPT))
    assert saved.session_id == "sess_sync"
    assert (tmp_path / "sess_sync.json").is_file()


def test_delete_reports_whether_a_record_was_removed(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    store.save(SessionState.seed("sess_a", "agent-a", SYSTEM_PROMPT))
    assert store.delete("sess_a") is True
    assert store.delete("sess_a") is False
    assert store.load("sess_a") is None


def test_list_session_ids_is_sorted_and_only_counts_records(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    for sid in ("sess_c", "sess_a", "sess_b"):
        store.save(SessionState.seed(sid, "agent-a", SYSTEM_PROMPT))
    (tmp_path / "notes.txt").write_text("not a session", encoding="utf-8")
    assert store.list_session_ids() == ("sess_a", "sess_b", "sess_c")


def test_sessions_are_isolated_from_one_another_on_disk(tmp_path: Path) -> None:
    """Requirement 1: each session keeps its own message sequence and turn counter."""
    store = SessionStore(storage_dir=tmp_path)
    store.save(
        SessionState(
            session_id="sess_a",
            agent_id="agent-a",
            messages=_dialogue(),
            turn_counter=3,
        )
    )
    store.save(
        SessionState(
            session_id="sess_b",
            agent_id="agent-a",
            messages=(ChatMessage(role=MessageRole.USER, content="unrelated"),),
            turn_counter=1,
        )
    )

    a = store.load("sess_a")
    b = store.load("sess_b")

    assert a is not None and b is not None
    assert a.turn_counter == 3
    assert b.turn_counter == 1
    assert a.messages != b.messages


def test_resetting_one_session_does_not_touch_another(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    store.save(
        SessionState(session_id="sess_a", agent_id="a", messages=_dialogue(), turn_counter=3)
    )
    store.save(
        SessionState(session_id="sess_b", agent_id="a", messages=_dialogue(), turn_counter=9)
    )

    loaded_a = store.load("sess_a")
    assert loaded_a is not None
    store.save(loaded_a.reset(system_prompt=SYSTEM_PROMPT))

    after_a = store.load("sess_a")
    after_b = store.load("sess_b")
    assert after_a is not None and after_b is not None
    assert after_a.turn_counter == 0
    assert after_b.turn_counter == 9
    assert after_b.messages == _dialogue()


# --------------------------------------------------------------------------------------
# P3 containment: the guard raises, and never fails open
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "session_id",
    [
        "../escape",
        "..",
        "nested/child",
        "back\\slash",
        "nul\x00byte",
        "",
    ],
)
def test_session_path_refuses_ids_that_could_escape_the_storage_dir(
    tmp_path: Path, session_id: str
) -> None:
    """A containment guard must raise. Returning quietly is the fail-open shape."""
    store = SessionStore(storage_dir=tmp_path)
    with pytest.raises(PathTraversalError):
        store.session_path(session_id)


@pytest.mark.parametrize("session_id", ["../escape", "nested/child", ""])
def test_load_save_and_delete_all_enforce_containment(tmp_path: Path, session_id: str) -> None:
    """Every entry point runs the guard, not just the one a caller happens to use."""
    store = SessionStore(storage_dir=tmp_path)
    with pytest.raises(PathTraversalError):
        store.load(session_id)
    with pytest.raises(PathTraversalError):
        store.delete(session_id)
    with pytest.raises(PathTraversalError):
        store.save(SessionState(session_id=session_id, agent_id="agent-a"))


def test_session_path_accepts_an_ordinary_id_and_stays_inside(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    path = store.session_path("sess_agent-orchestrator.1")
    assert path.is_relative_to(store.storage_dir)
    assert path.name == "sess_agent-orchestrator.1.json"


# --------------------------------------------------------------------------------------
# The name rule, usable by a caller that holds no storage directory
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("session_id", ["../escape", "..", "a/b", "c\\d", "e\x00f", ""])
def test_validate_session_id_refuses_an_illegal_name(session_id: str) -> None:
    """Separated from path resolution so `BaseAgent` can apply the same rule.

    An agent addresses sessions by ID long before anything is persisted. Without a name
    rule it would accept `""` or `"../../../etc/evil"` as a live session and only
    discover the problem at the first write, by which point the conversation exists and
    cannot be saved.
    """
    with pytest.raises(PathTraversalError):
        validate_session_id(session_id)


@pytest.mark.parametrize("session_id", ["sess_a", "sess_agent-orchestrator.1", "a.b~c:d@e+f"])
def test_validate_session_id_returns_a_legal_name_unchanged(session_id: str) -> None:
    assert validate_session_id(session_id) == session_id


@pytest.mark.parametrize("session_id", ["../escape", "..", "a/b", "c\\d", "e\x00f", ""])
def test_the_store_and_the_standalone_rule_agree_on_every_illegal_name(
    tmp_path: Path, session_id: str
) -> None:
    """One rule, two callers — asserted by agreement, not by reading the source.

    An earlier version of this test grepped `session_path`'s source for a call to
    `validate_session_id`, which asserts an implementation detail and would pass against
    a second, divergent copy of the rule pasted inside it. Comparing the observable
    outcome of both entry points is what actually catches drift.
    """
    store = SessionStore(storage_dir=tmp_path)
    with pytest.raises(PathTraversalError):
        validate_session_id(session_id)
    with pytest.raises(PathTraversalError):
        store.session_path(session_id)


# --------------------------------------------------------------------------------------
# The containment half of the guard, which the substring check masks (#210 review, M1)
#
# Every hostile *string* is stopped by `_FORBIDDEN_ID_SUBSTRINGS` before containment is
# consulted, so rewriting the containment `raise` to a `logger.warning` — the fail-open
# shape rejected in `3a8b7d6` — left 34/34 tests passing. Reaching it needs an ID that is
# lexically innocent and still resolves outside the root, which is what a planted
# symlink does.
# --------------------------------------------------------------------------------------


def test_a_lexically_clean_id_whose_record_symlinks_outside_is_refused(tmp_path: Path) -> None:
    """A planted symlink is the real attack the containment check exists for.

    `"escapee"` contains no `..`, no separator and no NUL, so it passes every string
    check. If an attacker can place `escapee.json` in the session directory as a symlink
    to somewhere else, `resolve()` follows it and the write lands outside the boundary.
    Only the `is_relative_to` check stops that, and this is the test that fails if it
    stops raising.
    """
    storage = tmp_path / "sessions"
    storage.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (storage / "escapee.json").symlink_to(outside / "escapee.json")
    store = SessionStore(storage_dir=storage)

    with pytest.raises(PathTraversalError):
        store.session_path("escapee")


def test_containment_is_enforced_on_every_entry_point_not_just_path_resolution(
    tmp_path: Path,
) -> None:
    """`load`, `save` and `delete` must each refuse, so no operation slips past."""
    storage = tmp_path / "sessions"
    storage.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (storage / "escapee.json").symlink_to(outside / "escapee.json")
    store = SessionStore(storage_dir=storage)

    with pytest.raises(PathTraversalError):
        store.load("escapee")
    with pytest.raises(PathTraversalError):
        store.delete("escapee")
    with pytest.raises(PathTraversalError):
        store.save(SessionState(session_id="escapee", agent_id="agent-a"))
    # Nothing was written outside the boundary.
    assert list(outside.iterdir()) == []


def test_a_symlinked_storage_root_is_itself_not_an_escape(tmp_path: Path) -> None:
    """The root is resolved in `__init__`, so a symlinked storage dir is legitimate and
    must keep working — the guard rejects escapes, not indirection."""
    real = tmp_path / "real-sessions"
    real.mkdir()
    link = tmp_path / "linked-sessions"
    link.symlink_to(real, target_is_directory=True)
    store = SessionStore(storage_dir=link)

    store.save(SessionState.seed("sess_a", "agent-a", SYSTEM_PROMPT))

    assert (real / "sess_a.json").is_file()
    loaded = store.load("sess_a")
    assert loaded is not None and loaded.session_id == "sess_a"


# --------------------------------------------------------------------------------------
# Atomicity, which cleanup does not imply (#210 review, M4)
#
# `test_save_is_atomic_and_leaves_no_temporary_files` only checks that no temp file is
# left behind, which a plain `path.write_text(payload)` satisfies trivially — deleting
# the temp-and-replace entirely left 34/34 passing. What actually has to be pinned is
# that the destination is reached *through* `os.replace`, so a failure cannot damage the
# record already there.
# --------------------------------------------------------------------------------------


def test_a_failed_write_leaves_the_previous_record_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The previous record must survive a write that dies at the commit point.

    `os.replace` is made to fail, standing in for a crash between writing the new bytes
    and swapping them in. Temp-and-replace leaves the old record untouched. A direct
    write to `path` has already truncated the only copy by this point — and would not
    call `os.replace` at all, so it would report success and this `pytest.raises` would
    fail. That is what makes this a pin on atomicity rather than on cleanup.

    The second write is built from a `load` rather than constructed fresh, because #219's
    revision precondition would otherwise refuse it before `os.replace` was ever reached
    — and this test would then pass for the wrong reason, catching a
    `StaleSessionWriteError` under `pytest.raises(OSError)` while never exercising the
    commit point at all. The `OSError` is asserted by type below for that reason.
    """
    store = SessionStore(storage_dir=tmp_path)
    store.save(
        SessionState(session_id="sess_a", agent_id="agent-a", messages=_dialogue(), turn_counter=3)
    )
    before = (tmp_path / "sess_a.json").read_bytes()
    current = store.load("sess_a")
    assert current is not None

    def _boom(src: object, dst: object) -> None:
        raise OSError("simulated crash at the commit point")

    monkeypatch.setattr("uclone_x.agent.session.os.replace", _boom)

    with pytest.raises(OSError) as commit_failure:
        store.save(current.with_messages((), turn_counter=99))
    # Not a `StaleSessionWriteError` wearing an `OSError`'s clothes: the write must have
    # travelled all the way to the commit point for this to pin atomicity.
    assert not isinstance(commit_failure.value, StaleSessionWriteError)

    assert (tmp_path / "sess_a.json").read_bytes() == before
    monkeypatch.undo()
    survived = store.load("sess_a")
    assert survived is not None
    assert survived.turn_counter == 3
    assert survived.messages == _dialogue()
    # And the failed attempt cleaned up after itself.
    assert [q.name for q in tmp_path.iterdir()] == ["sess_a.json"]


def test_the_destination_is_only_ever_reached_through_a_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No bytes are written directly to the record path.

    A direct write is the mutation this pins: here the destination is opened by nobody
    but `os.replace`, so if `save` ever writes to `path` itself this fails.
    """
    store = SessionStore(storage_dir=tmp_path)
    target = tmp_path / "sess_a.json"
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def _record(src: object, dst: object) -> None:
        replaced.append((str(src), str(dst)))
        real_replace(src, dst)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr("uclone_x.agent.session.os.replace", _record)

    store.save(SessionState.seed("sess_a", "agent-a", SYSTEM_PROMPT))

    assert len(replaced) == 1
    src, dst = replaced[0]
    assert dst == str(target)
    assert src != str(target)
    assert ".tmp." in src


def test_concurrent_writers_to_one_session_never_splice_a_record(tmp_path: Path) -> None:
    """Interleaved writes yield one whole record, never a mixture of two.

    The temp name carries the pid and a random suffix precisely so two writers cannot
    share a scratch file. Threads rather than processes keeps this fast; what it
    exercises is the same temp-and-replace path.

    Each writer now does a real read-modify-write, and a `StaleSessionWriteError` is a
    **correct** outcome rather than a failure: since #219 a writer that lost the race is
    refused instead of silently overwriting. What this test still pins is the other
    property — that whichever writes *do* land leave one whole record and never a mixture
    of two — so conflicts are counted and every other exception is still a failure.
    """
    store = SessionStore(storage_dir=tmp_path)
    turn_counts = list(range(1, 41))
    errors: list[BaseException] = []
    conflicts: list[StaleSessionWriteError] = []
    committed: list[int] = []

    def _write(turn: int) -> None:
        try:
            base = store.load("sess_hot")
            state = (
                SessionState(
                    session_id="sess_hot",
                    agent_id="agent-a",
                    messages=_dialogue(),
                    turn_counter=turn,
                )
                if base is None
                else base.with_messages(_dialogue(), turn_counter=turn)
            )
            store.save(state)
            committed.append(turn)
        except StaleSessionWriteError as conflict:
            conflicts.append(conflict)
        except BaseException as exc:  # noqa: BLE001 - recorded, then asserted on
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write, turn_counts))

    assert errors == [], f"an unexpected exception escaped: {errors}"
    # The instrument has to have observed the contention it claims to test. With eight
    # threads racing forty read-modify-writes, some must commit and some must be refused;
    # if either list were empty this test would be asserting nothing about concurrency.
    assert committed, "no writer committed, so nothing about concurrency was exercised"
    assert conflicts, (
        "eight threads over forty writes produced no conflict at all, so the reads and "
        "writes did not actually interleave and this test is not measuring contention"
    )
    final = store.load("sess_hot")
    assert final is not None, "a concurrent write left an unreadable record"
    assert final.turn_counter in turn_counts
    assert final.messages == _dialogue()
    # Bounded, not equal, and the gap is the honest one. Every commit advances the
    # revision by one, so the final revision cannot exceed the number of commits; it can
    # fall short, because the precondition is a check and not a lock (see
    # `test_the_precondition_is_a_check_not_a_lock`) and two threads can both pass it
    # before either replaces. Asserting equality here would pin a serialisability the
    # store does not claim, and would be flaky exactly when it mattered.
    assert 1 <= final.revision <= len(committed)
    # No scratch files survived, so every writer committed or cleaned up.
    assert [q.name for q in tmp_path.iterdir()] == ["sess_hot.json"]


# --------------------------------------------------------------------------------------
# What `save` accepts, `load` returns (#210 review — the silent-wipe arm)
# --------------------------------------------------------------------------------------


def test_with_messages_rejects_a_non_integer_turn_counter() -> None:
    """`with_messages(msgs, turn_counter="9")` used to be accepted.

    `model_copy(update=...)` performs no validation, so the string reached the model,
    `model_dump(mode="json")` wrote `"9"` to disk, and `load` then returned `None` — the
    write reported success and the record read back as "no session here". Constructing
    through the validating constructor closes it.
    """
    state = SessionState.seed("sess_a", "agent-a", SYSTEM_PROMPT)
    with pytest.raises(ValidationError):
        state.with_messages(_dialogue(), turn_counter="9")  # pyright: ignore[reportArgumentType]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_save_refuses_a_state_that_load_could_not_read_back(tmp_path: Path) -> None:
    """The invariant stated as a test: anything `save` accepts, `load` returns.

    `model_copy` is reachable by any caller and validates nothing, so a bad value can
    still be smuggled into a `SessionState`. `save` validates the serialized payload
    before it touches the disk, so the outcome is a refusal rather than a record that
    reads back as absent.

    The `UserWarning` filter is deliberate: serializing the smuggled value makes Pydantic
    emit `PydanticSerializationUnexpectedValue`, which is the library agreeing with the
    premise of the test. Silencing it here keeps the suite's warning count meaningful
    without hiding it anywhere else.
    """
    store = SessionStore(storage_dir=tmp_path)
    smuggled = SessionState.seed("sess_a", "agent-a", SYSTEM_PROMPT).model_copy(
        update={"turn_counter": "9"}
    )
    assert smuggled.turn_counter == "9"  # model_copy really did not validate

    with pytest.raises(ValueError, match="does not validate as SessionState"):
        store.save(smuggled)

    # And nothing was written, so there is no record to misread later.
    assert store.load("sess_a") is None
    assert list(tmp_path.iterdir()) == []


def test_a_good_state_still_round_trips_after_the_validation_gate(tmp_path: Path) -> None:
    """The gate must not reject legitimate records — including ledger-flagged ones."""
    store = SessionStore(storage_dir=tmp_path)
    state = SessionState(
        session_id="sess_a",
        agent_id="agent-a",
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.SYSTEM, content="[ledger]", compaction_ledger=True),
            ChatMessage(role=MessageRole.USER, content="q"),
        ),
        turn_counter=2,
    )
    saved = store.save(state)
    loaded = store.load("sess_a")
    assert loaded is not None
    assert loaded.messages == state.messages
    assert loaded.turn_counter == 2
    assert loaded.updated_at == saved.updated_at


# --------------------------------------------------------------------------------------
# An unusable path answers "absent" rather than raising past its documented contract
# --------------------------------------------------------------------------------------


def test_an_over_long_id_reads_as_absent_rather_than_raising(tmp_path: Path) -> None:
    """`is_file()` raises `OSError(ENAMETOOLONG)` for an ID past the filesystem limit.

    It used to sit outside the `try`, so `load` raised where its docstring promised
    `None`. No record can exist at a path the filesystem cannot name, so absent is the
    honest answer — and it is the answer the contract already advertised.
    """
    store = SessionStore(storage_dir=tmp_path)
    over_long = "s" * 5000

    assert store.load(over_long) is None
    assert store.delete(over_long) is False


def test_delete_does_not_swallow_a_failure_to_remove_a_record_that_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Absent" is not the same as "could not delete".

    A record that exists and cannot be removed must reach the caller; only the
    cannot-exist case answers `False`.
    """
    store = SessionStore(storage_dir=tmp_path)
    store.save(SessionState.seed("sess_a", "agent-a", SYSTEM_PROMPT))

    def _boom(self: Path, missing_ok: bool = False) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", _boom)

    with pytest.raises(OSError):
        store.delete("sess_a")


# --------------------------------------------------------------------------------------
# Corrupt records read as absent, never as a fabricated empty session (P6)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "{ not json",
        '"a bare string"',
        "[1, 2, 3]",
        '{"session_id": "sess_a"}',
        '{"session_id": "sess_a", "agent_id": "a", "messages": [{"role": "nonsense"}]}',
    ],
)
def test_unreadable_records_load_as_none(tmp_path: Path, raw: str) -> None:
    """`None` is the honest "no session here"; an empty `SessionState` would be a
    fabricated result presented as a real one, which P6 forbids."""
    store = SessionStore(storage_dir=tmp_path)
    (tmp_path / "sess_a.json").write_text(raw, encoding="utf-8")
    assert store.load("sess_a") is None


def test_on_disk_record_is_a_readable_json_object(tmp_path: Path) -> None:
    """The record's key set, pinned so a schema change has to be deliberate.

    This deliberately does **not** claim parity with what `AgentSessionManager` writes.
    An earlier docstring did, and it was false: the UI writes `turns` where this writes
    `turn_counter`, and its `messages` are presentation records rather than
    `ChatMessage`s. The two are different artifacts, which is why the Core record lives
    under a `core/` subdirectory instead of sharing the UI's filename.
    """
    store = SessionStore(storage_dir=tmp_path)
    store.save(
        SessionState(
            session_id="sess_a",
            agent_id="agent-a",
            messages=_dialogue(),
            turn_counter=2,
        )
    )
    raw = json.loads((tmp_path / "sess_a.json").read_text(encoding="utf-8"))
    assert set(raw) == {
        "session_id",
        "agent_id",
        "messages",
        "turn_counter",
        "created_at",
        "updated_at",
        # Added by #219's optimistic concurrency. Deliberate, and deliberately visible
        # here: this is the assertion a schema change has to come through, and the model
        # is `extra="forbid"`, so a record written by a newer version is unreadable by an
        # older one. `test_a_legacy_record_with_no_revision_key_hydrates_at_zero` covers
        # the direction that actually occurs on an upgrade.
        "revision",
        "plan",
        # Added by #1152, and deliberate for the same reason `revision` is. What composed
        # a session's anchored `SYSTEM` turn had lived only on the agent's in-memory
        # working copy, so a restored session could not say whether its anchor was the
        # agent's to re-resolve and every one of them was attributed to the caller. It is
        # written here as `null` because this record was constructed directly rather than
        # snapshotted from a live session: absent provenance is a real state with its own
        # handling, not a hole. See
        # `test_a_record_with_no_recorded_provenance_is_left_alone_and_reported` in
        # `tests/unit/test_agent_base.py` for the upgrade direction.
        "anchor_provenance",
        # Added by #1421: what each turn's requests carried besides the conversation, so a
        # request can be rebuilt from the record. A record written before it has no such
        # key and loads with none; `tests/unit/test_context_snapshot.py` covers that.
        "context_snapshots",
    }
    assert raw["anchor_provenance"] is None
    assert raw["messages"][0]["role"] == "system"


def test_store_creates_its_storage_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "sessions"
    store = SessionStore(storage_dir=target)
    assert store.storage_dir.is_dir()
    assert store.list_session_ids() == ()


def test_the_store_default_root_is_read_from_the_module_constant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default root comes from `DEFAULT_SESSION_STORAGE_DIR`, not a second literal.

    Replaces a test that asserted a hand-written *replica* of the constant
    (`parts[-2:] == (".uclone", "sessions")`) and read nothing the product computes, so
    it would have passed against a `__init__` that hardcoded a different default — the
    weak form of #136. Patching the constant and reading the instance back tests the
    wiring, which is the falsifiable part.

    Its old name also claimed the root is "the one the UI already uses". That is wrong
    twice over: the two records have different schemas, and the Core record deliberately
    lives under a `core/` subdirectory for exactly that reason. Nothing here should
    assert a relationship to `ui.app`.
    """
    fake_root = tmp_path / "elsewhere" / "sessions"
    monkeypatch.delenv(SESSION_STORAGE_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr("uclone_x.agent.session.DEFAULT_SESSION_STORAGE_DIR", fake_root)

    store = SessionStore()

    # `<family root>/core`, not the root: the root is where the UI transcript used to
    # sit, and sharing it destroyed conversations in both directions (#215 review).
    assert store.storage_dir == (fake_root / CORE_RECORD_SUBDIR).resolve()
    assert store.storage_dir.is_dir()


def test_the_env_override_moves_both_layers_together(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`UCLONE_SESSION_DIR` names the family root, so the CLI and the UI stay agreed.

    The UI used to compute its own default inline and ignore the variable entirely, so
    redirecting it moved the Core store and left the UI writing to the real home.
    """
    root = tmp_path / "redirected"
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(root))

    core = SessionStore()
    manager = AgentSessionManager(fallback_to_mock=True)

    assert core.storage_dir == (root / CORE_RECORD_SUBDIR).resolve()
    assert manager.storage_dir == root.resolve()
    # And the UI's Core store is the *same* directory the CLI writes to.
    assert manager.core_store.storage_dir == core.storage_dir


def test_unit_suite_does_not_write_to_home_session_directory() -> None:
    """Every test by default operates within an isolated temporary storage directory (#228).

    Guards against accidental writes into `Path.home() / .uclone / sessions` by asserting
    that default SessionStore and AgentSessionManager instantiations resolve inside the
    autouse temporary directory provided by _isolate_session_storage fixture.
    """
    home_uclone = (Path.home() / ".uclone" / "sessions").resolve()

    store = SessionStore()
    manager = AgentSessionManager(fallback_to_mock=True)

    assert store.storage_dir != home_uclone
    assert store.storage_dir != (home_uclone / CORE_RECORD_SUBDIR).resolve()
    assert manager.storage_dir != home_uclone
    assert manager.core_store.storage_dir != (home_uclone / CORE_RECORD_SUBDIR).resolve()
    assert os.environ.get(SESSION_STORAGE_DIR_ENV_VAR) is not None


# --------------------------------------------------------------------------------------
# The volunteered invariant, held rather than narrowed (#210 delta)
# --------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_save_returns_exactly_what_load_will_return(tmp_path: Path) -> None:
    """`save` used to return the *unvalidated* object it was handed.

    `model_copy(update={"messages": [ {...dict...} ]})` produced a correct record on disk
    and a correct `load`, while `save` still handed back a `SessionState` whose
    `messages` were raw `dict`s where `ChatMessage` was declared. No data loss and the
    disk was right — but the stated invariant said `save` and `load` agree, and they did
    not. Returning the validated payload is what makes the claim true rather than
    aspirational.

    The `UserWarning` filter is deliberate, as in
    `test_save_refuses_a_state_that_load_could_not_read_back`: serializing the smuggled
    value makes Pydantic emit `PydanticSerializationUnexpectedValue`, which is the
    library agreeing with the premise. Silencing it here keeps the suite's warning count
    meaningful without hiding it anywhere else.
    """
    store = SessionStore(storage_dir=tmp_path)
    smuggled = SessionState.seed("sess_a", "agent-a", SYSTEM_PROMPT).model_copy(
        update={"messages": [{"role": "system", "content": "x"}]}
    )
    # `model_copy` really did not validate: these are dicts, not ChatMessage.
    assert isinstance(smuggled.messages[0], dict)

    returned = store.save(smuggled)
    loaded = store.load("sess_a")

    assert loaded is not None
    assert returned == loaded
    assert all(isinstance(m, ChatMessage) for m in returned.messages)
    assert returned.messages[0].role is MessageRole.SYSTEM


def test_save_returns_a_validated_value_on_the_ordinary_path_too(tmp_path: Path) -> None:
    """The guarantee is unconditional, not a special case for smuggled input."""
    store = SessionStore(storage_dir=tmp_path)
    state = SessionState(
        session_id="sess_a", agent_id="agent-a", messages=_dialogue(), turn_counter=4
    )
    returned = store.save(state)
    loaded = store.load("sess_a")
    assert loaded is not None
    assert returned == loaded
    assert returned.turn_counter == 4


@pytest.mark.parametrize("session_id", [123, 4.5, b"bytes", ["a"], {"a": 1}, object()])
def test_a_non_string_session_id_refuses_with_the_typed_error(session_id: object) -> None:
    """`None` refused with `PathTraversalError`; every other non-`str` raised a raw
    `TypeError` out of `bad in session_id`. Nothing was written either way, so this is
    error-type hygiene — but a guard that refuses one class of bad input with two
    different exception types is inconsistent with itself, and a caller catching
    `PathTraversalError` would not catch the other."""
    with pytest.raises(PathTraversalError):
        validate_session_id(session_id)  # pyright: ignore[reportArgumentType]


def test_a_none_session_id_still_refuses_with_the_typed_error() -> None:
    with pytest.raises(PathTraversalError):
        validate_session_id(None)  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------------------
# The family paragraph's own claims, pinned by introspection (#210/#211 delta)
#
# Two reviewers found the same defect independently from opposite seams: the paragraph
# ended "Both now point here" while three of the four call sites carried no pointer at
# all. It is the same unfalsifiable-prose shape as this module's earlier key-set-parity
# claim — reintroduced *in the paragraph written to fix that class of defect*. So the
# claim is now asserted the way the reviewers measured it, against live `__doc__`.
# --------------------------------------------------------------------------------------


FAMILY_HEADING = "Validation is a property of the constructor, not of the type"


def _mentions_family(doc: str | None) -> bool:
    """Whether `doc` cites the family paragraph, ignoring where a formatter wrapped it.

    Whitespace is collapsed on both sides before comparing. A raw substring check looked
    right and was brittle: `ruff format` wraps a long docstring line, so the heading
    arrives as `"Validation is a property of the\n        constructor, not of the type"`
    and an exact match fails on a cross-reference that is perfectly present. That is a
    false failure waiting on the next reflow, and it caught me on the first run of these
    very tests.
    """
    normalized = " ".join((doc or "").split())
    return " ".join(FAMILY_HEADING.split()) in normalized


def test_the_family_heading_exists_in_the_module_docstring() -> None:
    """The heading is the anchor every cross-reference quotes, so it has to be there."""
    from uclone_x.agent import session as session_module

    assert _mentions_family(session_module.__doc__)


@pytest.mark.parametrize(
    ("name", "func"),
    [
        ("SessionState.with_messages", SessionState.with_messages),
        ("SessionState.reset", SessionState.reset),
        ("SessionStore.save", SessionStore.save),
    ],
)
def test_every_first_door_call_site_points_back_at_the_family(name: str, func: object) -> None:
    """Measured before the fix: all three were `False` while the paragraph said otherwise."""
    assert _mentions_family(getattr(func, "__doc__", None)), (
        f"{name} does not point back at the family paragraph"
    )


def test_the_module_docstring_does_not_name_a_symbol_this_seam_does_not_contain() -> None:
    """The paragraph forward-referenced `_LiveSession`, which exists only in the
    `BaseAgent` seam.

    True of the stack, false of this commit — and this commit can merge alone, which is
    what made it matter rather than being pedantry: `main` would then carry a docstring
    naming code that does not exist, which is precisely the key-set-parity failure this
    module was already returned for once.

    The second door is now described by what it *is* — the mutable working copy
    `BaseAgent` keeps — and explicitly assigned to the other seam, so the paragraph is
    true at both SHAs rather than only at the top of the stack.
    """
    from uclone_x.agent import session as session_module

    doc = session_module.__doc__ or ""
    assert "_LiveSession" not in doc
    # And it still tells the reader where that half lives.
    assert "multi-session seam" in doc
    assert "not to this module" in doc


# --------------------------------------------------------------------------------------
# Atomicity is not isolation — and isolation is now provided (#219 defect 2)
#
# Every write goes through `os.replace` and is atomic, so a crash never leaves a
# half-written record. That says nothing about two writers, and until #219 there was no
# version field and no precondition, so the loser's update was discarded with no error
# anywhere — a missing turn rather than an unreadable file, which is the quieter failure
# and the one the atomicity tests above do not reach.
#
# The three tests below were written under #210 to *pin the defect*, deliberately so that
# a fix would break them rather than pass silently. It did. They now pin the fix, and each
# keeps the distinction it was written to draw:
#
#   * `test_two_writers_no_longer_silently_lose_an_update` — the loser is refused, and
#     the winner's update is intact. Was: the loser silently won and one turn vanished.
#   * `test_the_store_offers_a_revision_to_detect_a_stale_write` — the record carries
#     exactly one concurrency token, `revision`, and `updated_at` still cannot arbitrate,
#     which is why it was not used. Was: no such field existed at all.
#   * `test_a_write_is_still_atomic_alongside_the_conflict_refusal` — the two properties
#     stay separate. Was: the same separation, asserted against last-write-wins.
# --------------------------------------------------------------------------------------


def test_two_writers_no_longer_silently_lose_an_update(tmp_path: Path) -> None:
    """Read-modify-write from two stores: the winner commits, the loser is refused.

    The inversion of the #210 pin. No fault injection and no contrived timing — just the
    interleaving two processes holding one session id produce as a matter of course, which
    is why it had to be closed rather than documented.
    """
    a = SessionStore(storage_dir=tmp_path)
    b = SessionStore(storage_dir=tmp_path)
    a.save(SessionState(session_id="sess_hot", agent_id="agent-a", turn_counter=0))

    # Both read the same base state, and therefore the same revision.
    seen_by_a = a.load("sess_hot")
    seen_by_b = b.load("sess_hot")
    assert seen_by_a is not None and seen_by_b is not None
    assert seen_by_a.turn_counter == seen_by_b.turn_counter == 0
    assert seen_by_a.revision == seen_by_b.revision == 1

    # A writes first and wins.
    a.save(seen_by_a.with_messages(_dialogue(), turn_counter=1))

    # B writes back onto the revision it read and is refused, by type and by content.
    with pytest.raises(StaleSessionWriteError) as refusal:
        b.save(
            seen_by_b.with_messages(
                (ChatMessage(role=MessageRole.USER, content="b's only turn"),), turn_counter=1
            )
        )
    assert refusal.value.session_id == "sess_hot"
    assert refusal.value.expected_revision == 1
    assert refusal.value.actual_revision == 2
    # Recoverable, not merely refused: the error hands back the record to rebase onto, so
    # the loser needs no second read to make progress.
    assert isinstance(refusal.value.current, SessionState)
    assert refusal.value.current.messages == _dialogue()

    final = a.load("sess_hot")
    assert final is not None
    # A's turn is intact and B's stale write never landed.
    assert final.messages == _dialogue()
    assert not any("b's only turn" == (m.content or "") for m in final.messages)
    # And the refusal wrote nothing at all — not a conflict file, not a temp file.
    assert [p.name for p in tmp_path.iterdir()] == ["sess_hot.json"]

    # B rebases onto what the error handed it and now commits.
    rebased = refusal.value.current.with_messages(
        (*final.messages, ChatMessage(role=MessageRole.USER, content="b's only turn")),
        turn_counter=2,
    )
    saved = b.save(rebased)
    assert saved.revision == 3
    merged = a.load("sess_hot")
    assert merged is not None
    contents = [m.content or "" for m in merged.messages]
    # Both turns survive, which is the outcome last-write-wins could not produce.
    assert "b's only turn" in contents
    assert [m.content or "" for m in _dialogue()] == contents[: len(_dialogue())]


def test_the_store_offers_a_revision_to_detect_a_stale_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record carries exactly one concurrency token, and it is not a timestamp.

    The #210 pin asserted the *absence* of every such field, which is what made "add
    optimistic concurrency" a schema decision rather than a one-line change. Inverted
    here: `revision` exists, the other spellings still do not, and `updated_at` still
    cannot arbitrate — which is the reason it was not used, and is worth keeping asserted.

    **Why `updated_at` cannot arbitrate, stated correctly.** Not "it compares now against
    now and can never refuse" — that reasoning was wrong twice over, and this docstring is
    the last place it survived. Measured, a compare-and-swap on `updated_at` **refuses an
    ordinary uncontended `load` → `with_messages` → `save`**: it can never *accept*, not
    never refuse. And since #223 made `updated_at` preservable by parameter, a caller
    *can* now carry one across a snapshot, so the "you could never hold one" half is gone
    too. What actually disqualifies it is below, asserted in three parts: preserving it is
    **opt-in**, so the precondition would be silently unenforced for every caller that did
    not pass it; it is **caller-settable**, so a stale writer can supply the value that
    matches; and it is a wall-clock string rather than a counter, so two writes inside one
    clock tick are indistinguishable.
    """
    fields = set(SessionState.model_fields)
    assert "revision" in fields
    # One token, not several. A second one is a second thing to keep in agreement.
    assert not fields & {"version", "etag", "sequence", "generation"}

    store = SessionStore(storage_dir=tmp_path)
    saved = store.save(SessionState(session_id="sess_a", agent_id="agent-a"))
    reloaded = store.load("sess_a")
    assert reloaded is not None
    assert reloaded.revision == saved.revision == 1

    # `updated_at` is set by the writer on the way to disk, so a caller can never hold a
    # value to compare against: the state it passed in and the state it got back differ.
    fresh = SessionState(session_id="sess_b", agent_id="agent-a")
    stored = store.save(fresh)
    assert stored.updated_at != fresh.updated_at, (
        "`updated_at` survived a save unchanged, so it might arbitrate after all and the "
        "reason `revision` exists needs revisiting"
    )
    # `with_messages` refreshes it by default, so preserving one is opt-in — which is the
    # first reason it cannot arbitrate: a precondition built on it would be silently
    # unenforced for every caller that did not pass the parameter #223 added.
    assert stored.with_messages(_dialogue()).updated_at != stored.updated_at
    # And where it *is* preserved, it is preserved because the **caller** said so, so a
    # stale writer can supply whatever value would match. That is the second reason: a
    # token the caller chooses is not a precondition, it is a request.
    forged = stored.with_messages(_dialogue(), updated_at=stored.updated_at)
    assert forged.updated_at == stored.updated_at

    # And it is a wall-clock ISO string rather than a counter, so two writes in one clock
    # tick cannot be distinguished by timestamp alone (#678). When the clock does not
    # advance between writes, updated_at is identical while revision increments strictly
    # monotonically (1 -> 2), and a stale write based on revision 1 is still refused.
    fixed_ts = "2026-09-12T00:00:00.000000Z"
    monkeypatch.setattr("uclone_x.agent.session._now_iso", lambda: fixed_ts)

    clock_tied_a = store.save(SessionState(session_id="sess_clock_tied", agent_id="agent-a"))
    clock_tied_b = store.save(clock_tied_a.with_messages(_dialogue()))

    assert clock_tied_a.updated_at == clock_tied_b.updated_at == fixed_ts
    assert clock_tied_b.revision == 2

    # A concurrent or stale write holding revision 1 is still detected and refused,
    # proving revision arbitrates concurrency even when timestamps are indistinguishable.
    with pytest.raises(StaleSessionWriteError) as exc_info:
        store.save(clock_tied_a.with_messages(_dialogue()))
    assert exc_info.value.expected_revision == 1
    assert exc_info.value.actual_revision == 2

    # `revision`, by contrast, is carried through untouched and takes no parameter, so a
    # caller going through a snapshot door can only restate what it read. That is the
    # whole of the claim — a *directly constructed* state can carry any revision, which is
    # #248 and is pinned by `test_a_directly_constructed_state_can_force_a_revision`.
    assert forged.revision == stored.revision
    assert "revision" not in SessionState.with_messages.__code__.co_varnames, (
        "`with_messages` gained a `revision` parameter, which would make the store's "
        "precondition caller-settable and therefore advisory rather than enforced"
    )


def test_a_write_is_still_atomic_alongside_the_conflict_refusal(tmp_path: Path) -> None:
    """The distinction the docstring draws, asserted: the surviving record is always a
    whole one, and that was never the property #219 was about.

    Losing an update is not the same defect as corrupting a record. The #210 version of
    this test ran two writers blindly past each other, which is now refused, so each
    writer reads before it writes — but the assertion is unchanged: after every write the
    record parses and reports what the last committed writer put there."""
    a = SessionStore(storage_dir=tmp_path)
    b = SessionStore(storage_dir=tmp_path)
    a.save(SessionState(session_id="sess_hot", agent_id="agent-a", turn_counter=0))
    for i in range(1, 21):
        for store, agent in ((a, "agent-a"), (b, "agent-b")):
            base = store.load("sess_hot")
            assert base is not None, "a concurrent write produced an unreadable record"
            # Serialised by construction here — each writer reads immediately before it
            # writes — so no conflict is expected, and one would be a real failure.
            store.save(
                SessionState(
                    session_id="sess_hot",
                    agent_id=agent,
                    turn_counter=i,
                    created_at=base.created_at,
                    revision=base.revision,
                )
            )
        loaded = a.load("sess_hot")
        assert loaded is not None, "a concurrent write produced an unreadable record"
        assert loaded.turn_counter == i
    assert [p.name for p in tmp_path.iterdir()] == ["sess_hot.json"]
    final = a.load("sess_hot")
    assert final is not None
    # 1 seed + 40 writes, none lost, because none of them raced.
    assert final.revision == 41


# --------------------------------------------------------------------------------------
# The two cases the suite lacked (#219 defect 2, acceptance criteria 2 and 4)
#
# Everything above runs two `SessionStore` objects inside one process. That is equivalent
# to two processes — `SessionStore` holds no class- or module-level mutable state and no
# lock, so two objects bypass no in-process serialisation — but "equivalent" is an
# argument, and the shipped configuration is two genuine OS processes. The first test
# below runs them, so the precondition is shown to work through the filesystem rather
# than through anything Python holds.
#
# The second is the case nothing covered: a writer that keeps its own state in memory
# across the race. It is the shape `rev-senior-14` measured and the reason this defect is
# worse than a lost turn — a committed compaction destroyed on disk while the compacting
# process still reasons from the compacted context leaves *no participant holding the
# truth*, which a lost turn does not.
# --------------------------------------------------------------------------------------


_CHILD_WRITER = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path
    from uclone_x.agent.session import SessionState, SessionStore
    from uclone_x.errors import StaleSessionWriteError

    root, payload, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    store = SessionStore(storage_dir=Path(root))
    base = SessionState.model_validate_json(Path(payload).read_text(encoding="utf-8"))
    try:
        saved = store.save(base.with_messages((*base.messages,), turn_counter=base.turn_counter + 1))
    except StaleSessionWriteError as exc:
        print(json.dumps({
            "pid": os.getpid(), "tag": tag, "outcome": "refused",
            "expected": exc.expected_revision, "actual": exc.actual_revision,
        }))
    else:
        print(json.dumps({
            "pid": os.getpid(), "tag": tag, "outcome": "committed",
            "revision": saved.revision,
        }))
    """
)


def _run_child_writer(root: Path, payload: Path, tag: str) -> dict[str, Any]:
    """Run one writer in a real subprocess and return what it reported.

    `PYTHONPATH` is set to this worktree's `src` explicitly rather than inherited by
    luck: the shared `.venv` carries an editable install pointing at the *primary*
    workspace, so a child that does not get this ends up importing a different checkout's
    `uclone_x` and testing code this branch does not contain.
    """
    src_root = Path(__file__).resolve().parents[2] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src_root), env.get("PYTHONPATH", "")])
    env[SESSION_STORAGE_DIR_ENV_VAR] = str(root)
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_WRITER, str(root), str(payload), tag],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, (
        f"child writer {tag!r} did not run: rc={completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )
    line = completed.stdout.strip().splitlines()[-1]
    reported = json.loads(line)
    assert isinstance(reported, dict)
    return cast(dict[str, Any], reported)


def test_two_genuine_os_processes_on_one_session_cannot_lose_an_update(tmp_path: Path) -> None:
    """Two real processes, not two objects in one process (#219 acceptance criterion 2).

    Deterministic rather than timing-based: both children are handed the *same* base
    state, read at the same revision, and run one after the other. The first commits; the
    second is holding a revision the record has moved past and must be refused. A sleep-
    and-race version of this would test the same precondition less reliably.

    The pids are asserted distinct from each other and from this process, because the
    whole point of the test is that it is not the in-process case wearing a subprocess
    costume — and the previous concurrency runs on this store established no *corruption*
    under load, which is a different property from no *lost update*.
    """
    store = SessionStore(storage_dir=tmp_path)
    store.save(SessionState(session_id="sess_shared", agent_id="agent-a", messages=_dialogue()))
    base = store.load("sess_shared")
    assert base is not None and base.revision == 1

    payload = tmp_path / "base.json"
    payload.write_text(base.model_dump_json(), encoding="utf-8")

    first = _run_child_writer(tmp_path, payload, "first")
    second = _run_child_writer(tmp_path, payload, "second")

    # The instrument observed what it claims to: three distinct OS processes.
    pids = {first["pid"], second["pid"], os.getpid()}
    assert len(pids) == 3, f"expected three distinct pids, got {pids}"

    assert first["outcome"] == "committed", first
    assert first["revision"] == 2
    assert second["outcome"] == "refused", second
    assert second["expected"] == 1
    assert second["actual"] == 2

    final = store.load("sess_shared")
    assert final is not None
    # One write landed, and the record advanced exactly once — the second process's
    # update was refused rather than applied on top of a state it had never seen.
    assert final.revision == 2
    assert final.turn_counter == 1
    assert [q.name for q in tmp_path.iterdir()] == sorted(["sess_shared.json", "base.json"])


@pytest.mark.asyncio
async def test_a_writer_holding_memory_across_the_race_is_refused_not_diverged(
    tmp_path: Path,
) -> None:
    """The no-participant-holds-the-truth variant, which nothing covered (#219 criterion 4).

    The measurement this reproduces, from `rev-senior-14` on the #183 stack:

        after A compacts  ->  A memory: 6   | disk: 6  | B memory (stale): 25
        after B persists  ->  disk: 25      | ledger on disk?: False | A memory: 6
        lost-update detected by anything?:  no exception raised

    A's committed compaction was destroyed on disk while A kept the compacted context in
    memory, so A reasoned from six messages, disk held twenty-five, and the compaction
    ledger was in neither. That is worse than a lost turn: a lost turn is recoverable in
    principle by the writer that still holds it, and this is not recoverable by anyone.

    What must hold now: B's stale write is refused, so disk keeps A's compaction, A's
    memory still agrees with disk, and the ledger survives. B is left holding messages the
    record does not contain — that part is unavoidable, since B genuinely has them — but
    it is **told**, at the moment it becomes true, with the on-disk state attached. The
    divergence is observable instead of being created silently by overwriting A.
    """
    store = SessionStore(storage_dir=tmp_path)
    sid = "sess_agent-compact"

    # A and B are separate agents on one session id, which is the CLI-and-UI shape.
    agent_a = _threshold_agent(60_000, store=store)
    agent_b = _threshold_agent(60_000, store=store)
    agent_a.load_history(_bulky_dialogue(), turn_counter=12)
    persisted = agent_a.persist_session()
    assert len(persisted.messages) == 25, "the fixture no longer reproduces the 25-message case"

    # B opens the same session and reads the pre-compaction state into its own memory.
    hydrated_by_b = agent_b.hydrate_session(sid)
    assert hydrated_by_b is not None
    assert len(hydrated_by_b.messages) == 25
    assert hydrated_by_b.revision == persisted.revision

    # A compacts and commits.
    result = await agent_a.compact_session(reason="manual_on_demand")
    assert result.messages_before == 25
    after_compaction = store.load(sid)
    assert after_compaction is not None
    assert len(after_compaction.messages) == result.messages_after < 25
    assert len(agent_a.get_session(sid).messages) == result.messages_after
    # The ledger the compaction wrote is on disk, which is the artifact that used to
    # vanish entirely.
    ledger_on_disk = [m for m in after_compaction.messages if m.compaction_ledger]
    assert ledger_on_disk, "the compaction ledger is not on disk, so there is nothing to lose"

    # B still holds 25 messages in memory and tries to persist them.
    assert len(agent_b.get_session(sid).messages) == 25
    with pytest.raises(StaleSessionWriteError) as refusal:
        agent_b.persist_session(sid)

    assert refusal.value.session_id == sid
    assert refusal.value.expected_revision == hydrated_by_b.revision
    assert refusal.value.actual_revision == after_compaction.revision
    current = refusal.value.current
    assert isinstance(current, SessionState)
    # The refusal handed B the compacted record, so B can rebase rather than re-read.
    assert len(current.messages) == result.messages_after

    # Disk still holds A's compaction, ledger included.
    final = store.load(sid)
    assert final is not None
    assert len(final.messages) == result.messages_after
    assert [m for m in final.messages if m.compaction_ledger]
    # A's memory and the record still agree — A is a participant holding the truth.
    assert len(agent_a.get_session(sid).messages) == len(final.messages)
    assert agent_a.get_session(sid).revision == final.revision
    # B's memory is untouched by the refusal, so nothing of B's was destroyed either; it
    # is merely knowingly stale, which is the observable half of criterion 4.
    assert len(agent_b.get_session(sid).messages) == 25

    # And B's documented recovery works: adopt the record, then write.
    readopted = agent_b.hydrate_session(sid)
    assert readopted is not None
    assert len(readopted.messages) == result.messages_after
    reconciled = agent_b.persist_session(sid)
    assert reconciled.revision == final.revision + 1


def test_the_precondition_is_a_check_not_a_lock(tmp_path: Path) -> None:
    """The residual window, pinned so `save`'s docstring cannot claim serialisability.

    `save` reads the record, compares the revision, then replaces. Two writers can both
    pass the comparison before either replaces, and the second one's bytes win — so the
    precondition closes the read-modify-write window that spans a turn, which is the one
    this defect arrives through, and not the microscopic one between the check and the
    rename. Mutual exclusion is what would close that, and it was rejected: an `O_EXCL`
    lockfile reproduces #219's own defect 3 (crash-orphaned files) as an availability
    failure, and `fcntl.flock` has no Windows equivalent, so it would ship a platform
    branch unexercisable on the single machine P8 confines verification to.

    Asserted by driving the interleaving deterministically rather than by racing threads,
    which would make the *absence* of a guarantee flaky to demonstrate. `os.replace` is
    wrapped so a second writer commits inside the first writer's window.
    """
    store = SessionStore(storage_dir=tmp_path)
    other = SessionStore(storage_dir=tmp_path)
    store.save(SessionState(session_id="sess_hot", agent_id="agent-a", turn_counter=0))
    base = store.load("sess_hot")
    assert base is not None and base.revision == 1

    real_replace = os.replace
    interleaved: list[str] = []

    def _replace_after_a_concurrent_commit(src: object, dst: object) -> None:
        # Fires once: the first writer has already passed its revision check, and this
        # stands in for another process committing in that window.
        if not interleaved:
            interleaved.append("other")
            other.save(
                base.with_messages(
                    (ChatMessage(role=MessageRole.USER, content="other's turn"),),
                    turn_counter=9,
                )
            )
        real_replace(src, dst)  # pyright: ignore[reportArgumentType]

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr("uclone_x.agent.session.os.replace", _replace_after_a_concurrent_commit)
        store.save(base.with_messages(_dialogue(), turn_counter=1))

    # The instrument fired, so the window really was exercised rather than assumed.
    assert interleaved == ["other"]

    final = store.load("sess_hot")
    assert final is not None
    # The interleaved writer's update is gone: inside this window the precondition does
    # not protect it, which is exactly the limit being pinned. Both writes claimed
    # revision 2, so the counter cannot report the loss either.
    assert final.messages == _dialogue()
    assert final.revision == 2
    assert not any("other's turn" == (m.content or "") for m in final.messages)


def test_a_legacy_record_with_no_revision_key_hydrates_at_zero(tmp_path: Path) -> None:
    """A record written before #219 must still load, and its first write must succeed.

    This is the upgrade direction, and the one that actually happens: `SessionState` is
    `extra="forbid"`, so the *forbidden* direction is an older build reading a newer
    record, while a newer build reading an older record only has to tolerate a missing
    key. `revision` defaults to `0`, so a legacy record hydrates at `0`, the precondition
    compares `0` against `0`, and the first write after the upgrade commits at `1`.

    Had `revision` been declared without a default, every pre-#219 record in a real
    `~/.uclone/sessions` would have become unreadable — and `load` reports an
    unparseable record as *absent*, so the upgrade would have presented itself as every
    session having disappeared. That is #28's failure shape, which is why this is pinned
    rather than reasoned about.
    """
    store = SessionStore(storage_dir=tmp_path)
    legacy = {
        "session_id": "sess_legacy",
        "agent_id": "agent-a",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "turn_counter": 4,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-02T00:00:00+00:00",
    }
    (tmp_path / "sess_legacy.json").write_text(json.dumps(legacy), encoding="utf-8")

    loaded = store.load("sess_legacy")
    assert loaded is not None, "a pre-#219 record read as absent, which is the #28 shape"
    assert loaded.revision == 0
    assert loaded.turn_counter == 4
    assert loaded.created_at == "2026-01-01T00:00:00+00:00"

    # And the first write after the upgrade is accepted rather than refused.
    saved = store.save(loaded.with_messages(_dialogue(), turn_counter=5))
    assert saved.revision == 1
    reloaded = store.load("sess_legacy")
    assert reloaded is not None
    assert reloaded.revision == 1
    assert reloaded.turn_counter == 5


def test_the_revision_round_trips_through_save_and_load(tmp_path: Path) -> None:
    """The new field obeys the invariant `save` already promised: what it returns is what
    `load` yields, byte for byte and type for type.

    Asserted for `revision` specifically because it is the one field `save` *changes* on
    the way to disk, so it is the one that could most easily be right on disk and wrong in
    the return value — which is precisely the half of the invariant that was claimed and
    not held before #210 closed it.
    """
    store = SessionStore(storage_dir=tmp_path)
    state = SessionState(session_id="sess_a", agent_id="agent-a", messages=_dialogue())
    for expected in (1, 2, 3):
        state = store.save(state)
        assert state.revision == expected
        reloaded = store.load("sess_a")
        assert reloaded is not None
        assert reloaded == state
        assert type(reloaded.revision) is int


def test_a_mismatched_revision_is_refused_whether_it_is_higher_lower_or_negative(
    tmp_path: Path,
) -> None:
    """Every revision that is not the one on disk is refused, in both directions.

    **Renamed from `test_a_caller_cannot_forge_a_revision_to_get_its_write_accepted`,
    which claimed more than it checked.** Every value below is a *mismatch* against disk,
    so what this establishes is that the comparison is an equality check rather than a
    `>=` — a stale writer guessing high (`99`) is refused exactly like one guessing low
    (`0`). It never tried the case its old name promised: an *exact* guess, which is
    accepted. That case is #248 and is pinned by
    `test_a_directly_constructed_state_can_force_a_revision` below.

    The distinction matters because `>=` would have been the natural spelling and is
    wrong: it would accept any writer that guessed high, which is every stale writer that
    tries.
    """
    store = SessionStore(storage_dir=tmp_path)
    store.save(SessionState(session_id="sess_a", agent_id="agent-a"))
    store.save(store.save(SessionState(session_id="sess_b", agent_id="agent-b")))
    on_disk = store.load("sess_a")
    assert on_disk is not None and on_disk.revision == 1

    for forged in (0, 2, 99, -1):
        with pytest.raises(StaleSessionWriteError) as refusal:
            store.save(
                SessionState(
                    session_id="sess_a",
                    agent_id="agent-a",
                    messages=_dialogue(),
                    revision=forged,
                )
            )
        assert refusal.value.expected_revision == forged
        assert refusal.value.actual_revision == 1

    # Only the revision actually on disk is accepted, and `save` sets the next one itself
    # rather than taking the caller's word for it.
    accepted = store.save(on_disk.with_messages(_dialogue()))
    assert accepted.revision == 2


def test_no_snapshot_door_can_advance_the_revision(tmp_path: Path) -> None:
    """The invariant the fix actually rests on, asserted as the narrow thing it is.

    `with_messages` and `reset` are the two ways a caller turns a state it read into a
    state it writes. Neither exposes `revision` and neither changes it, so a write built
    from a stale read still carries the stale revision and is recognisably stale. This is
    what "a caller cannot advance the revision" means — and all it means.
    """
    store = SessionStore(storage_dir=tmp_path)
    store.save(SessionState(session_id="sess_a", agent_id="agent-a"))
    base = store.load("sess_a")
    assert base is not None and base.revision == 1

    # Neither door takes a `revision` parameter, so neither can be asked to change it.
    for door in (SessionState.with_messages, SessionState.reset):
        assert "revision" not in door.__code__.co_varnames, (
            f"`{door.__name__}` gained a `revision` parameter, which would make the "
            "store's precondition caller-settable and therefore advisory"
        )
    # And neither changes it in passing.
    assert base.with_messages(_dialogue()).revision == 1
    assert base.with_messages(_dialogue(), turn_counter=99).revision == 1
    assert base.reset(system_prompt=SYSTEM_PROMPT).revision == 1

    # So a second writer that commits leaves the first one's snapshot refused, however it
    # was built — which is the property the whole fix depends on.
    store.save(base.with_messages(_dialogue()))
    for stale in (base.with_messages(_dialogue()), base.reset(system_prompt=SYSTEM_PROMPT)):
        with pytest.raises(StaleSessionWriteError):
            store.save(stale)


def test_a_directly_constructed_state_can_force_a_revision(tmp_path: Path) -> None:
    """The counterexample to the wider claim, pinned so it cannot be restated (#248).

    `save`'s docstring once said a caller "cannot forge a precondition". That is false,
    and this is the measurement that settles it: a `SessionState` built by the constructor
    or by `model_copy(update=...)` carries whatever revision it was given, and `save`
    **accepts** it when the value happens to equal the revision on disk — silently
    destroying the committed update the precondition exists to protect. `save`'s pre-write
    `model_validate_json` backstop cannot catch it, because a forged revision is a valid
    `int`.

    So the store carries an effective `force=True`. **It is no longer undocumented, and it
    is now a decision rather than a pending one**: #248 resolved to accept the capability
    and enforce the bound that makes it survivable, rejecting a declared `force=True`
    parameter, an out-of-band handle, and waiting on the constructor-door family. The four
    options and the cost of each rejected one are recorded in `SessionStore.save`. This
    test is therefore *not* inverted — the hole is deliberately open, and the three tests
    below carry the reasons.

    Two claims that used to sit in this docstring were measured and are false, which is
    why they are gone rather than reworded:

    * "Closing it is a decision about the constructor doors named under 'Validation is a
      property of the constructor, not of the type'." Those doors are about values that
      never passed validation; this one is a valid `int`. See
      `test_the_constructor_door_family_closure_would_not_close_this_one`.
    * "No code under `src/` constructs a `SessionState` outside that module."
      `agent/base.py` does. The true and now *enforced* bound is that nothing under `src/`
      **outside the store module** names a revision value — the store itself must, since
      it is what issues them. See `test_no_module_under_src_names_a_revision_value`.

    The drift this test was written to stop is still its job: if the doors below ever stop
    being accepted, `save`'s docstring and this test have to be restated together.
    """
    store = SessionStore(storage_dir=tmp_path)
    forged_msg = (ChatMessage(role=MessageRole.USER, content="forged"),)

    def stale_read_then_winner_commits(sid: str) -> SessionState:
        """Set one session up so a stale reader holds revision 1 while disk is at 2."""
        store.save(SessionState(session_id=sid, agent_id="agent-a"))
        stale = store.load(sid)
        assert stale is not None and stale.revision == 1
        winner = store.save(stale.with_messages(_dialogue()))
        assert winner.revision == 2 and winner.messages == _dialogue()
        return stale

    # Each door gets its own session, so no door is measured against the damage the
    # previous one did.
    doors: dict[str, Callable[[SessionState], SessionState]] = {
        "constructor": lambda stale: SessionState(
            session_id=stale.session_id,
            agent_id="agent-a",
            messages=forged_msg,
            revision=2,
        ),
        "model_copy on a snapshot": lambda stale: stale.with_messages(forged_msg).model_copy(
            update={"revision": 2}
        ),
        "model_copy on the read itself": lambda stale: stale.model_copy(
            update={"messages": forged_msg, "revision": 2}
        ),
    }
    for label, build in doors.items():
        sid = f"sess_{label.replace(' ', '_')}"
        stale = stale_read_then_winner_commits(sid)

        # The forged write is accepted even though this writer never read revision 2.
        applied = store.save(build(stale))
        assert applied.revision == 3, label

        after = store.load(sid)
        assert after is not None
        assert after.messages == forged_msg, (
            f"{label}: the forged write was not applied, so #248 may be closed — restate "
            "`save`'s docstring and this test together"
        )
        # And the winner's committed turn is gone, with no exception raised anywhere.
        assert not any((m.content or "") == "first answer" for m in after.messages), label

    # The bound that makes it a documented force rather than an open bypass: a *wrong*
    # guess is refused like any other mismatch.
    stale = stale_read_then_winner_commits("sess_wrong_guess")
    with pytest.raises(StaleSessionWriteError):
        store.save(SessionState(session_id="sess_wrong_guess", agent_id="agent-a", revision=99))


# --------------------------------------------------------------------------------------
# The #248 decision, and the three things it rests on (#248)
#
# #240 documented the forge and #248 decided what to do about it: accept it, and enforce
# the bound. Accepting is only defensible if the bound is real, and the card was produced
# by a bound that was *stated* and not enforced — and which was, when measured, already
# false. So each load-bearing claim below is asserted rather than described.
#
# That has now happened twice on this card. The docstring shipped a *second* bound — that
# an exact guess is reachable only by reading the record — which a reviewer measured false
# in two ways (a stale writer's blind `+1`; a caller that never read, guessing a small
# monotonic int). It was struck rather than repaired, since making it true means adopting
# the closure this card rejects. The sweep below is what is left holding the decision.
#
#   * `test_no_module_under_src_names_a_revision_value` — the bound that replaces the
#     false one, and now the only one. Nothing under `src/` outside the store picks a
#     revision; every construction forwards one.
#   * `test_the_revision_precondition_is_a_protocol_not_a_boundary` — why a `force=True`
#     parameter and an out-of-band handle both buy nothing. `save` is not the only route
#     to the record, so no control inside it is a boundary.
#   * `test_the_constructor_door_family_closure_would_not_close_this_one` — why waiting
#     for #31/#42 to close centrally is waiting for a fix that would not fix this.
# --------------------------------------------------------------------------------------


class _RevisionSweep(NamedTuple):
    """What the sweep found, and what each of its three matchers actually observed.

    The `*_witnesses` fields exist so a matcher that has silently stopped matching cannot
    pass by finding nothing (§6.9 case 2). They are the matchers' own output, incremented
    inside the matching branch rather than recomputed alongside it — an instrument check
    that re-derives its subject independently is checking the re-derivation.
    """

    offending: list[tuple[str, int, str]]
    constructors: set[str]
    scanned: int
    string_witnesses: set[str]
    keyword_witnesses: set[str]


def _session_state_revision_arguments() -> _RevisionSweep:
    """Sweep `src/uclone_x` for every place a `revision` value could be *chosen*.

    Returns the offending sites, the modules that construct a `SessionState` at all, the
    number of modules scanned — so a glob that stops matching cannot make the assertion
    pass by finding nothing, as it did in #157 — and one witness set per name matcher, so
    neither matcher can stop matching without a test failing.

    **The rule is keyed on the *name*, not on the callee, and that is the whole design.**
    An earlier version of this helper watched three named callees — `SessionState(...)`,
    `model_copy` and `__setattr__` — and claimed those were the three reachable shapes.
    That claim was false and a reviewer broke it in one line: `SessionState.model_validate(
    {**raw, "revision": 2})` is a fourth, it forges end-to-end through `save`, and it goes
    through none of the three. A callee allow-list is incomplete by construction, because
    every release can add another way to build a model — `model_construct`,
    `model_validate_strings`, a `**kwargs` splat — and the sweep would silently stop
    covering the tree while still reading as if it did. That is #248's own defect shape,
    which is exactly what this sweep exists to make impossible.

    So it watches the two ways Python source can *name* a field instead, whatever is being
    called:

    * a **`revision=` keyword argument** on any call, and
    * the **string `"revision"`** anywhere — a dict key in a `model_copy(update=...)`,
      `model_validate({...})` or `model_construct(**{...})`, an `object.__setattr__`
      argument, a `state.__dict__["revision"]` subscript.

    A site is fine when it *forwards* — `<expr>.revision` — rather than naming a value, and
    **that applies to both clauses**: `revision=state.revision` and `{"revision":
    state.revision}` are the same act written two ways, so the string clause exempts a dict
    key whose paired value is a forward exactly as the keyword clause does. It did not,
    once, and the asymmetry was a real defect rather than a stylistic one — the clause was
    wider than the bound this test exists to enforce, and it fired on
    `ui.app`'s `/api/sessions` handler, which reads `state.revision` into an HTTP response
    body and constructs no `SessionState` at all. `agent.session` is separately exempt from
    the string clause: it is the module that issues revisions, so it is the one place
    allowed to write the key over a value it computed.

    Note what stays caught, because this is the narrowing's whole risk: a **literal**
    (`{"revision": 2}`, the `model_validate` forge) and an **expression**
    (`{"revision": stale.revision + 1}`, the blind `+1`) are not `ast.Attribute` forwards
    and are flagged as before.

    **Each matcher records the modules it fired in**, because a name matcher that matches
    nothing produces exactly the same empty `offending` list as a tree with nothing to
    find. Both escaped a mutation check when this helper reported only `offending`: with
    the string comparison rewritten to a sentinel, and again with the keyword comparison
    rewritten to one, the suite stayed green at its full baseline count. `constructors`
    does not cover either — it is built in a branch that never reads `kw.arg` and never
    reaches the string clause — so it survived both intact. The witness sets are what the
    two `assert` statements in the test consume; see there for why they are membership
    checks and not equality.

    What this still cannot see, stated because the last version of this docstring
    overclaimed and that is the finding it was corrected for: a name laundered through a
    variable (`key = "rev" + "ision"`). That is deliberate evasion rather than the accident
    the bound is about, and no static rule reaches it.
    """
    import ast

    src_root = Path(__file__).resolve().parents[2] / "src" / "uclone_x"
    # The stores are the only legitimate producers of a revision, and there are now two:
    # `RoomStore.save` is to `RoomState` what `SessionStore.save` is to `SessionState` —
    # the single writer that stamps the next value and refuses a stale precondition. The
    # sweep matches the *string* `"revision"` wherever it appears and so cannot tell which
    # record a module is stamping; exempting the room store keeps the rule's meaning ("a
    # revision comes from a store and nowhere else") rather than widening it. A third entry
    # here should be argued, not added.
    store_modules = {"agent.session", "room.store"}
    # The exemption is module-wide on the string clause, which is the clause that catches
    # the `SessionState.model_validate({**raw, "revision": 2})` forge — so inside an exempt
    # module that forge would pass. `agent.session` is where `SessionState` lives and is
    # exempt by necessity; `room.store` is exempt only because it stamps a *different*
    # record, and that is a fact about the module, not a promise. Asserted rather than
    # trusted, so the widening stays proved and not merely argued.
    room_store_src = (src_root / "room" / "store.py").read_text(encoding="utf-8")
    assert "SessionState" not in room_store_src, (
        "`room.store` is exempt from the revision-string rule because it stamps "
        "`RoomState`, not `SessionState`. It now references `SessionState`, so the "
        "exemption no longer follows and the forge this rule catches would pass there."
    )
    offending: list[tuple[str, int, str]] = []
    constructors: set[str] = set()
    string_witnesses: set[str] = set()
    keyword_witnesses: set[str] = set()
    scanned = 0
    for path in sorted(src_root.rglob("*.py")):
        scanned += 1
        # `.parts`, not a `/` replace: a hard-coded separator makes the module names —
        # and so the `store_modules` exemption below — wrong on Windows.
        module = ".".join(path.relative_to(src_root).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # `{"revision": state.revision}` is a **forward wearing a string**, and the rule is
        # "naming a value", not "mentioning the field" — so it must be judged exactly as
        # `revision=state.revision` is. Collected in a pre-pass because `ast.walk` hands out
        # the key `Constant` detached from the `Dict` that gives it its meaning. `ast.Dict`
        # pads `keys` with `None` for a `**spread`, and `zip` keeps the pairing right.
        forwarded_keys: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            # `strict=True`: `keys` and `values` are parallel by construction in a valid
            # `ast.Dict`, so a length mismatch is a broken parse, not a case to absorb.
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "revision"
                    and isinstance(value, ast.Attribute)
                    and value.attr == "revision"
                ):
                    forwarded_keys.add(id(key))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "revision":
                # Recorded before either exemption, not after: a witness gathered downstream
                # of an exemption would be empty in a healthy tree and could witness nothing.
                string_witnesses.add(module)
                if module not in store_modules and id(node) not in forwarded_keys:
                    offending.append((module, node.lineno, 'the string "revision"'))
                continue
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            ) == "SessionState":
                constructors.add(module)
            for kw in node.keywords:
                if kw.arg != "revision":
                    continue
                # Recorded before the forward/choose split, so a site stays a witness
                # whether it forwards or offends: the witness answers "did this matcher
                # match", which must not depend on the verdict it feeds.
                keyword_witnesses.add(module)
                # A forward is `something.revision`. Anything else — a literal, an
                # arithmetic expression, a bare local — is the caller choosing.
                if not (isinstance(kw.value, ast.Attribute) and kw.value.attr == "revision"):
                    offending.append((module, node.lineno, "revision= naming a value"))
    return _RevisionSweep(offending, constructors, scanned, string_witnesses, keyword_witnesses)


def test_no_module_under_src_names_a_revision_value() -> None:
    """The bound the #248 decision rests on, enforced instead of asserted in prose.

    **This replaces a sentence that was false when it was written.** `save`'s docstring
    said "no code under `src/` constructs or `model_copy`s a `SessionState` outside this
    module, so it cannot happen by accident". `agent/base.py` does construct one — it is
    where a live working copy becomes a record, and it has to. The bound was doing real
    work in the decision to leave the forge open, and nothing was holding it up.

    The true statement is narrower and stronger, and it is what this asserts: no module
    under `src/` ever **names** a revision. Every construction outside the store either
    omits the field or forwards a value it was handed, so the only producer of a revision
    in the tree is `SessionStore.save`. A new site that picks one fails here — which is
    what "it cannot happen by accident" has to mean to be worth relying on, and what no
    behavioural test can check, since a forged write that guesses right is by construction
    indistinguishable from a legitimate one.

    **The rule is keyed on the name and not on the callee**, after a reviewer forged a
    revision through `SessionState.model_validate({**raw, "revision": 2})` — a fourth shape
    the first version of this sweep did not watch while its helper claimed three were all
    there were. See `_session_state_revision_arguments` for why a callee allow-list cannot
    be made complete, and for the one residual it still cannot see.

    **This is now the only bound the #248 decision rests on**, after the second one — that
    an exact guess is reachable only by reading the record — was measured false and removed
    from `SessionStore.save`'s docstring rather than repaired. So the sweep is load-bearing
    on its own, which is why its own matchers are asserted to be live below rather than
    assumed: a silent-failure mode here is a silent-failure mode in the whole decision.
    """
    sweep = _session_state_revision_arguments()
    offending, constructors, scanned = sweep.offending, sweep.constructors, sweep.scanned

    assert scanned > 50, f"sweep only reached {scanned} modules; the glob is wrong"
    # Four instrument checks, because the sweep is otherwise indistinguishable from a
    # visitor that matches nothing at all: every matcher here reports a violation by
    # *appending*, so a matcher that has stopped matching returns the same empty list as a
    # clean tree. Each check sits before the `offending` comparison it protects, so a
    # matcher that observed nothing cannot reach the emptiness it would satisfy.
    #
    # An earlier version claimed the string half was "proved by `agent.session` being the
    # only module the exemption has to excuse". It was not proved by that or by anything:
    # `agent.session` is excused by the exemption *branch*, which a dead string matcher
    # never reaches, so the claim held equally when the matcher matched nothing. Both name
    # matchers were rewritten to a sentinel and both mutants escaped a full suite run.
    #
    # This one proves the *call* half is live and resolving names.
    #
    # `agent.bootstrap` is permitted because it omits `revision`, the compliant half of
    # `SessionStore.save`'s "omits or forwards" rule; the substantive sweep found nothing
    # there and only this set-equality canary fired.
    #
    # Preserve `bootstrap_session`'s `if state is None` guard —
    # `test_bootstrap_session_guard_prevents_destructive_overwrite_of_revision_zero_record`
    # pins why, and `test_destructive_revision_zero_join_on_session_store_save` pins the
    # behaviour it guards against. This comment deliberately says nothing further about
    # `save`'s revision-0 semantics. Six earlier drafts tried and all six were false: two
    # asserted universals about "any" or "an existing" record, two attributed to a named
    # test an assertion it does not make, and two made claims about what the suite does not
    # cover — the last of those true when written and falsified by #431 merging while it was
    # under review. A coverage negative does not belong in a comment: it can be made false
    # by a PR that never touches this file.
    #
    # The guard is also a check, not a lock: `load` -> `if state is None` -> `save` is a
    # TOCTOU window.
    assert constructors == {"agent.session", "agent.base", "agent.bootstrap"}, (
        f"the set of modules constructing a SessionState changed: {sorted(constructors)}. "
        "That set is not itself the rule — a new construction site is fine if it forwards "
        "a revision — but it is how this test knows its own visitor still resolves callee "
        "names, so update it deliberately and read the #248 decision in `SessionStore.save` "
        "while you are here."
    )
    # Membership, not equality, and deliberately so: a *new* module naming the string or
    # choosing a `revision=` is already a failure below, with a message that explains the
    # #248 damage. These two only have to fail when a matcher goes dead, so they assert the
    # one witness each matcher must always have and say nothing about the rest of the set.
    assert "agent.session" in sweep.string_witnesses, (
        "the string matcher fired in no module, so it is no longer observing anything and "
        "an empty `offending` list below would prove nothing. `agent.session` names the "
        f"key at least once (it issues revisions); witnesses: {sorted(sweep.string_witnesses)}. "
        "If the store legitimately stopped writing the key as a string, re-point this "
        "witness at whatever site now exercises the clause — do not delete the check."
    )
    assert "agent.base" in sweep.keyword_witnesses, (
        "the `revision=` keyword matcher fired in no module, so it is no longer observing "
        "anything and an empty `offending` list below would prove nothing. `agent.base` "
        "forwards a revision when a live working copy becomes a record; witnesses: "
        f"{sorted(sweep.keyword_witnesses)}. A site counts as a witness whether it forwards "
        "or offends, so this fires only when the matcher itself has gone dead."
    )
    assert not offending, (
        "a module under `src/` names a revision value rather than forwarding one: "
        f"{offending}. `revision` is produced by `SessionStore.save` and by nothing else; "
        "a site that picks one can silently destroy another writer's committed turn when "
        "the guess is exact, which `save` cannot detect (#248). Forward the revision from "
        "the state you read, or read the #248 decision in `SessionStore.save` first."
    )


def test_the_revision_precondition_is_a_protocol_not_a_boundary(tmp_path: Path) -> None:
    """Why #248 rejected both a declared `force=True` and an out-of-band handle.

    Neither option was rejected on cost alone. Both were rejected because `save` is not the
    only route to the record, so no control placed inside it can be a boundary: an
    in-process caller that writes the record path directly does the *same damage as the
    forge* — the winner's committed turn gone, no exception anywhere — without needing to
    guess a revision, without going through `SessionState`, and without any parameter to
    declare.

    **This is also why the struck exactness bound in `SessionStore.save` could not have
    been repaired in code.** The route asserted below needs no read at all, so it survives
    every control that could be placed inside `save`, the rejected closure included — which
    is what makes that sentence an overclaim rather than an honest description of a fixable
    defect. The docstring says so; this is where it is measured.

    So `revision` is a cooperation protocol between honest writers. That is the frame in
    which a `force=True` parameter reads as adding a sanctioned way to defect rather than
    as making an existing one auditable, and in which a handle bought no unforgeability.

    Asserted rather than reasoned about, because the reasoning is the whole decision: if
    this ever stops holding — if `save` becomes the only way to reach a record — then the
    rejected options deserve re-costing and `SessionStore.save`'s docstring is wrong.
    """
    store = SessionStore(storage_dir=tmp_path)
    store.save(SessionState(session_id="sess_a", agent_id="agent-a"))
    winner = store.save(store.load("sess_a").with_messages(_dialogue()))  # pyright: ignore[reportOptionalMemberAccess]
    assert any((m.content or "") == "first answer" for m in winner.messages)

    # This writer forwards **nothing** — not the revision, not the timestamps. It has read
    # no record and knows only the session id, so it carries the default `revision=0`,
    # which is not merely unread but *wrong*: disk is at 2. `session_path` only locates the
    # file — a path helper, not a write door, and resolving the path by hand here would
    # hard-code the filename encoding the store owns (#256). What this bypasses is the
    # write door, which is the only thing the precondition sits in.
    #
    # Forwarding `winner.revision` here, as this test used to, demonstrated something
    # strictly weaker — "`save` is not the only door" — while the docstring above claimed
    # "without needing to guess a revision". A reviewer adjudicating the #248 classification
    # had to measure this route itself because the assertion did not reach it. It does now.
    bypass = SessionState(
        session_id="sess_a",
        agent_id="agent-a",
        messages=(ChatMessage(role=MessageRole.USER, content="bypassed"),),
    )
    assert bypass.revision == 0 and winner.revision != 0, (
        "this writer must carry a revision it never read and that disk does not hold, or "
        "the test is demonstrating a forwarded value again"
    )
    store.session_path("sess_a").write_text(
        json.dumps(bypass.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    after = store.load("sess_a")
    assert after is not None
    assert not any((m.content or "") == "first answer" for m in after.messages), (
        "writing the record directly no longer destroys the committed turn, so `save` may "
        "have become the only route to a record — re-cost the options #248 rejected"
    )
    assert after.revision == 0, (
        "the record now on disk should be the intruder's, carrying the revision it never "
        f"read (0), but it is at {after.revision}. The damage this test pins is that a "
        "caller which read nothing can land a wrong revision, which is why no control "
        "inside `save` — the closure #248 rejected included — can close this route."
    )


def test_the_constructor_door_family_closure_would_not_close_this_one() -> None:
    """Why #248 rejected waiting for the constructor-door family (#31/#42) to close.

    The rejection is of the option's *premise*, not of its scope. The family's closure is
    already shipped elsewhere in the tree — override `model_copy` so `update=` is routed
    through `model_validate` — so this is not a hypothetical fix to reason about. Applying
    exactly that route to a `SessionState` is measured below: it starts refusing a `"2"`,
    which is the family's whole point, and goes on accepting a `2`.

    That is the difference the option missed. Those doors are about a value that never
    passed validation; a forged revision is a valid `int` and passes any amount of it.
    Pinned so that nobody spends the central fix expecting #248 to arrive with it.
    """
    from uclone_x.engine.event_bus import AgentEvent as _Event
    from uclone_x.ontology.models import OntologyConcept as _Concept

    # The premise: this closure is a real, shipped pattern rather than a sketch.
    for model in (_Concept, _Event):
        assert model.model_copy is not BaseModel.model_copy, (
            f"{model.__name__} no longer overrides model_copy, so the #31/#42 closure "
            "this test is comparing against has changed shape — re-measure before "
            "trusting the #248 decision's rejection of option 4"
        )

    state = SessionState(session_id="sess_a", agent_id="agent-a", revision=1)

    def validating_copy(update: dict[str, Any]) -> SessionState:
        """`model_copy(update=...)` exactly as the family's override spells it."""
        return SessionState.model_validate({**state.model_dump(), **update})

    # What the family's closure buys: the unvalidated `str` is refused.
    with pytest.raises(ValidationError):
        validating_copy({"revision": "2"})

    # And what it does not touch: the forge, which was never a validation failure.
    forged = validating_copy({"revision": 2})
    assert forged.revision == 2, (
        "routing the update through model_validate now rejects a valid int revision, so "
        "#248 may close with the #31/#42 family after all — restate the decision in "
        "`SessionStore.save`, which rejects option 4 on precisely this measurement"
    )


def test_an_absent_record_accepts_any_revision(tmp_path: Path) -> None:
    """A deleted or never-written record has no update to lose, so the check does not apply.

    Stated as behaviour rather than left implicit, because the alternative is worse in a
    specific way: refusing would make a session unsavable after a legitimate `delete`,
    with the caller holding a revision that can never match anything again. The same
    applies to a record that exists but does not parse — `load` reports it absent, and an
    unreadable record holds no update worth preserving.
    """
    store = SessionStore(storage_dir=tmp_path)
    saved = store.save(SessionState(session_id="sess_a", agent_id="agent-a"))
    assert saved.revision == 1
    assert store.delete("sess_a") is True

    # The caller is still holding revision 1 and the record is gone. Accepted, and the
    # counter restarts rather than resuming, because `save` derives it from disk.
    resaved = store.save(saved.with_messages(_dialogue()))
    assert resaved.revision == 1

    # A corrupt record is the same case, by the same rule, and only once — `save` reads
    # through `load`, so it cannot disagree with it about what counts as absent.
    (tmp_path / "sess_a.json").write_text("{not json at all", encoding="utf-8")
    assert store.load("sess_a") is None
    over_corrupt = store.save(SessionState(session_id="sess_a", agent_id="agent-a", revision=77))
    assert over_corrupt.revision == 1


def test_destructive_revision_zero_join_on_session_store_save(tmp_path: Path) -> None:
    """A direct-written revision-0 record is silently overwritten by a bootstrap-shaped save.

    This test is the join of two existing halves:
    1) `test_the_revision_precondition_is_a_protocol_not_a_boundary` (which pins a direct
       record-path write landing a revision-0 record on disk, but stops before save);
    2) `test_a_mismatched_revision_is_refused_whether_it_is_higher_lower_or_negative` (which
       pins `SessionStore.save` refusing `revision=0` against a record that `save` itself wrote,
       which is always at revision >= 1).

    Neither half pins what happens when `save` meets a revision-0 record on disk.

    Design intent:
    This behaviour is INTENDED under the CAS protocol design (`save` checks
    `on_disk.revision == state.revision`, so `0 == 0` matches and advances to `1`), which is
    why `bootstrap_session`'s `if state is None` guard is required. `save` enforces optimistic
    concurrency by comparing expected revision against current revision; it does not treat
    revision 0 as a special invalid sentinel that refuses CAS matching.
    """
    store = SessionStore(storage_dir=tmp_path)
    session_id = "sess_join"

    # Step 1: Two saves establish a normal session at revision 2.
    store.save(SessionState(session_id=session_id, agent_id="agent-a"))
    winner = store.save(
        store.load(session_id).with_messages(_dialogue())  # pyright: ignore[reportOptionalMemberAccess]
    )
    assert winner.revision == 2
    assert any((m.content or "") == "first answer" for m in winner.messages)

    # Step 2: Direct-write a record bypassing save, landing revision 0 on disk.
    # This writer forwards nothing and carries the default revision=0.
    bypass = SessionState(
        session_id=session_id,
        agent_id="agent-a",
        messages=(ChatMessage(role=MessageRole.USER, content="bypassed"),),
    )
    assert bypass.revision == 0
    store.session_path(session_id).write_text(
        json.dumps(bypass.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    on_disk_before = store.load(session_id)
    assert on_disk_before is not None
    assert on_disk_before.revision == 0
    assert any((m.content or "") == "bypassed" for m in on_disk_before.messages)
    assert not any((m.content or "") == "first answer" for m in on_disk_before.messages)

    # Step 3: Bootstrap-shaped save carrying default revision=0.
    clone = agent_config_for_persona(_clone_persona())
    bootstrap_shaped = SessionState(
        session_id=session_id,
        agent_id=clone.agent_id,
        messages=(ChatMessage(role=MessageRole.SYSTEM, content=clone.system_prompt),),
    )
    assert bootstrap_shaped.revision == 0

    # Step 4: Save is accepted because 0 == 0 CAS passes. Nothing raises.
    saved = store.save(bootstrap_shaped)
    assert saved.revision == 1

    # Step 5: Verify the prior record was silently overwritten.
    on_disk_after = store.load(session_id)
    assert on_disk_after is not None
    assert on_disk_after.revision == 1
    assert on_disk_after.agent_id == clone.agent_id
    assert any((m.content or "") == clone.system_prompt for m in on_disk_after.messages)
    assert not any((m.content or "") == "bypassed" for m in on_disk_after.messages)
    assert not any((m.content or "") == "first answer" for m in on_disk_after.messages)


def test_bootstrap_session_guard_prevents_destructive_overwrite_of_revision_zero_record(
    tmp_path: Path,
) -> None:
    """The `if state is None` guard in `bootstrap_session` prevents overwriting existing revision 0 records.

    Because `SessionStore.save` accepts `0 == 0` (pinned by
    `test_destructive_revision_zero_join_on_session_store_save`), an unguarded call to
    `store.save` with a freshly constructed `SessionState` (revision=0) would silently
    destroy an existing revision-0 record on disk. The `if state is None` guard in
    `bootstrap_session` prevents this destructive route by checking for existence before
    invoking `store.save`.
    """
    store = SessionStore(storage_dir=tmp_path)
    session_id = "sess_bootstrap_guard"

    # Direct-write a revision-0 record to disk.
    direct_written = SessionState(
        session_id=session_id,
        agent_id="agent-existing",
        messages=(ChatMessage(role=MessageRole.USER, content="existing_committed_turn"),),
    )
    assert direct_written.revision == 0
    store.session_path(session_id).write_text(
        json.dumps(direct_written.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    # Calling bootstrap_session returns the config it was handed, but must NOT overwrite
    # the existing record.
    seed = agent_config_for_persona(_clone_persona())
    config = bootstrap_session(store, session_id, seed)
    assert config == seed

    # Verify the on-disk record is completely preserved.
    on_disk = store.load(session_id)
    assert on_disk is not None
    assert on_disk.revision == 0
    assert on_disk.agent_id == "agent-existing"
    assert any((m.content or "") == "existing_committed_turn" for m in on_disk.messages)
    assert not any((m.content or "") == seed.system_prompt for m in on_disk.messages)


# --------------------------------------------------------------------------------------
# Seam 2: BaseAgent hosts many sessions, and `reset_session` is the Core API
# --------------------------------------------------------------------------------------


def _agent(
    agent_id: str = "agent-multi",
    *,
    system_prompt: str = SYSTEM_PROMPT,
    store: SessionStore | None = None,
    llm: LLMProviderProtocol | None = None,
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id=agent_id,
            name="Multi Session Agent",
            system_prompt=system_prompt,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=llm,
        store=store,
    )


def _mock_llm(content: str = "an answer") -> LLMProviderProtocol:
    llm = MagicMock(spec=LLMProviderProtocol)
    llm.generate = AsyncMock(
        return_value=ModelResponse(
            content=content,
            usage=TokenUsage(provider="mock", model="mock-model", input_tokens=3, output_tokens=4),
            finish_reason=FinishReason.STOP,
            model_name="mock-model",
            provenance=Provenance(
                path=ExecutionPath.PRIMARY,
                requested=ServiceRef(provider="mock", model="mock-model"),
                served_by=ServiceRef(provider="mock", model="mock-model"),
            ),
        )
    )
    return llm


class _RecordingLLM(LLMProviderProtocol):
    """Typed connector that keeps the last `LLMRequest` it was handed.

    A `MagicMock(spec=...)` cannot serve here: `mock.generate.await_args` is untyped, so
    reading the captured request back trips `reportUnknownMemberType` under pyright
    strict, and a test that inspects a wire contract should be type-checked against it.
    """

    def __init__(self) -> None:
        self.last_request: LLMRequest | None = None

    @property
    def provider_name(self) -> str:
        return "recording"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.last_request = request
        return ModelResponse(
            content="an answer",
            usage=TokenUsage(provider="recording", model="mock-model"),
            finish_reason=FinishReason.STOP,
            model_name="mock-model",
            provenance=Provenance(
                path=ExecutionPath.PRIMARY,
                requested=ServiceRef(provider="recording", model="mock-model"),
                served_by=ServiceRef(provider="recording", model="mock-model"),
            ),
        )

    def stream(self, request: LLMRequest):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def test_a_new_agent_hosts_exactly_its_default_session() -> None:
    agent = _agent("agent-a")
    assert agent.session_ids == ("sess_agent-a",)
    assert agent.session_id == "sess_agent-a"
    assert len(agent.history) == 1
    assert agent.history[0].role == MessageRole.SYSTEM


def test_addressing_a_new_session_id_seeds_it_without_disturbing_the_active_one() -> None:
    """Requirement 1: reading an unhosted session id returns seeded state without mutating session_ids (#222)."""
    agent = _agent("agent-a")
    other = agent.get_session("sess_other")

    assert other.session_id == "sess_other"
    assert len(other.messages) == 1
    assert other.messages[0].content == SYSTEM_PROMPT
    assert other.turn_counter == 0
    # The active session is unchanged and reading does NOT host the unhosted session.
    assert agent.session_id == "sess_agent-a"
    assert agent.session_ids == ("sess_agent-a",)
    assert "sess_other" not in agent._sessions  # pyright: ignore[reportPrivateUsage]


def test_get_session_unhosted_is_pure_read_and_does_not_mutate_sessions() -> None:
    """Issue #222: get_session returns seeded SessionState without mutating _sessions or session_ids."""
    agent = _agent("agent-a")
    initial_hosted = agent.session_ids
    assert initial_hosted == ("sess_agent-a",)

    state = agent.get_session("unhosted_sid")
    assert isinstance(state, SessionState)
    assert state.session_id == "unhosted_sid"
    assert state.agent_id == "agent-a"
    assert state.turn_counter == 0
    assert len(state.messages) == 1
    assert state.messages[0].content == SYSTEM_PROMPT

    # session_ids and _sessions remain untouched
    assert agent.session_ids == ("sess_agent-a",)
    assert "unhosted_sid" not in agent._sessions  # pyright: ignore[reportPrivateUsage]


def test_get_session_many_unhosted_calls_prevents_unbounded_growth_and_memory_leak() -> None:
    """Issue #222: Executing 2,000 get_session calls on unhosted IDs does not grow _sessions or session_ids."""
    agent = _agent("agent-a")
    assert agent.session_ids == ("sess_agent-a",)

    for i in range(2000):
        sid = f"sess_{i}"
        s = agent.get_session(sid)
        assert s.session_id == sid

    assert agent.session_ids == ("sess_agent-a",)
    assert len(agent.session_ids) == 1
    assert len(agent._sessions) == 1  # pyright: ignore[reportPrivateUsage]
    assert "sess_agent-a" in agent._sessions  # pyright: ignore[reportPrivateUsage]


def test_mutating_operations_host_new_session_id() -> None:
    """Issue #222: switch_session, reset_session, and load_history explicitly host target session IDs."""
    agent = _agent("agent-a")
    assert agent.session_ids == ("sess_agent-a",)

    # 1. load_history hosts target session
    agent.load_history(_dialogue()[:2], turn_counter=1, session_id="sess_loaded")
    assert "sess_loaded" in agent.session_ids
    assert "sess_loaded" in agent._sessions  # pyright: ignore[reportPrivateUsage]
    assert agent.get_session("sess_loaded").turn_counter == 1

    # 2. reset_session hosts target session
    reset_state = agent.reset_session("sess_reset")
    assert reset_state.session_id == "sess_reset"
    assert "sess_reset" in agent.session_ids
    assert "sess_reset" in agent._sessions  # pyright: ignore[reportPrivateUsage]

    # 3. switch_session hosts target session
    switched_state = agent.switch_session("sess_switched")
    assert switched_state.session_id == "sess_switched"
    assert agent.session_id == "sess_switched"
    assert "sess_switched" in agent.session_ids
    assert "sess_switched" in agent._sessions  # pyright: ignore[reportPrivateUsage]

    assert agent.session_ids == ("sess_agent-a", "sess_loaded", "sess_reset", "sess_switched")


def test_each_session_keeps_its_own_messages_and_turn_counter() -> None:
    agent = _agent("agent-a")
    agent.load_history(_dialogue(), turn_counter=3, session_id="sess_a")
    agent.load_history(_dialogue()[:2], turn_counter=1, session_id="sess_b")

    a = agent.get_session("sess_a")
    b = agent.get_session("sess_b")

    assert a.turn_counter == 3
    assert b.turn_counter == 1
    assert len(a.messages) == 3
    assert len(b.messages) == 2


def test_history_and_snapshot_share_one_storage_location() -> None:
    """Pins the aliasing defect in `3a8b7d6`.

    That attempt kept `self._history` as an alias into `self._sessions[sid]`; rebinding
    the dict entry silently detached them, leaving two names for one piece of state.
    Here `_history` is a property onto the single `_LiveSession`, so a mutation through
    the hot path is visible in the snapshot with no flush step.
    """
    agent = _agent("agent-a")
    agent.load_history(_dialogue(), turn_counter=2)

    snapshot = agent.get_session()

    assert snapshot.messages == _dialogue()
    assert snapshot.turn_counter == 2
    assert agent.history == _dialogue()


@pytest.mark.asyncio
async def test_a_turn_advances_only_the_active_session_counter() -> None:
    """Pins the second defect in `3a8b7d6`: its per-session counters were written by
    `__init__`, `load_history` and `reset_session` but never by a turn, so a session
    that had run turns reported zero once it had been switched away from and back."""
    agent = _agent("agent-a", llm=_mock_llm())

    await agent.execute_turn("hello")
    await agent.execute_turn("again")

    assert agent.get_session("sess_agent-a").turn_counter == 2
    # A second, untouched session is still at zero.
    assert agent.get_session("sess_quiet").turn_counter == 0


@pytest.mark.asyncio
async def test_a_turn_count_survives_switching_away_and_back() -> None:
    agent = _agent("agent-a", llm=_mock_llm())
    await agent.execute_turn("hello")
    assert agent.get_session().turn_counter == 1

    agent.switch_session("sess_other")
    assert agent.get_session().turn_counter == 0
    await agent.execute_turn("in the other session")
    assert agent.get_session().turn_counter == 1

    agent.switch_session("sess_agent-a")
    assert agent.get_session().turn_counter == 1
    assert agent.session_id == "sess_agent-a"


@pytest.mark.asyncio
async def test_a_turn_in_one_session_does_not_reach_another(tmp_path: Path) -> None:
    agent = _agent("agent-a", llm=_mock_llm("first"))
    await agent.execute_turn("only in session a")
    a_len = len(agent.get_session("sess_agent-a").messages)

    agent.switch_session("sess_b")

    assert len(agent.get_session("sess_b").messages) == 1
    assert a_len > 1
    assert all(
        "only in session a" != (m.content or "") for m in agent.get_session("sess_b").messages
    )


# --------------------------------------------------------------------------------------
# reset_session
# --------------------------------------------------------------------------------------


def test_reset_session_preserves_the_system_prompt_and_zeroes_the_counter() -> None:
    """Acceptance criterion 2. The prompt is *not* preserved automatically — nothing
    recomposes `config.system_prompt`, so `reset_session` must re-seed it."""
    agent = _agent("agent-a")
    agent.load_history(_dialogue(), turn_counter=5)

    reset = agent.reset_session()

    assert reset.turn_counter == 0
    assert len(reset.messages) == 1
    assert reset.messages[0].role == MessageRole.SYSTEM
    assert reset.messages[0].content == SYSTEM_PROMPT
    assert agent.history == (ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),)


def test_reset_session_with_no_system_prompt_leaves_no_empty_system_message() -> None:
    """The CLI `/reset` seeded `system_prompt or ""`, producing an empty SYSTEM message
    that `__init__` never produces. Both must agree."""
    agent = _agent("agent-a", system_prompt="")
    assert agent.history == ()
    agent.load_history(_dialogue(), turn_counter=2)

    reset = agent.reset_session()

    assert reset.messages == ()
    assert agent.history == ()


def test_reset_session_can_target_a_non_active_session() -> None:
    agent = _agent("agent-a")
    agent.load_history(_dialogue(), turn_counter=4, session_id="sess_other")
    agent.load_history(_dialogue(), turn_counter=9)

    agent.reset_session("sess_other")

    assert agent.get_session("sess_other").turn_counter == 0
    # The active session is untouched.
    assert agent.get_session().turn_counter == 9
    assert len(agent.get_session().messages) == 3
    assert agent.session_id == "sess_agent-a"


def test_reset_session_keeps_the_creation_time() -> None:
    agent = _agent("agent-a")
    before = agent.get_session().created_at
    agent.load_history(_dialogue(), turn_counter=3)
    assert agent.reset_session().created_at == before


def test_reset_session_leaves_the_p7_invariant_injection_working() -> None:
    """The asserted invariants need no re-seeding: `_prepare_turn_messages` recomposes
    them per turn and they were never stored in history. Reset must not break that."""
    ontology = MagicMock(spec=OntologyEngineProtocol)
    invariant = MagicMock()
    invariant.tier = MagicMock()
    invariant.tier.value = "asserted"
    invariant.name = "no_secrets_in_logs"
    invariant.rule_expression = "secrets == redacted"
    invariant.predicate = None
    invariant.object_value = None
    invariant.description = None
    ontology.get_active_invariants = MagicMock(return_value=[invariant])

    agent = BaseAgent(
        config=AgentConfig(
            agent_id="agent-onto",
            name="Ontology Agent",
            system_prompt=SYSTEM_PROMPT,
        ),
        ontology=ontology,
    )
    agent.load_history(_dialogue(), turn_counter=2)
    agent.reset_session()

    prepared = agent._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]

    assert prepared[0].role == MessageRole.SYSTEM
    assert SYSTEM_PROMPT in (prepared[0].content or "")
    assert "no_secrets_in_logs" in (prepared[0].content or "")
    # And the invariants are still not stored in history.
    assert "no_secrets_in_logs" not in (agent.history[0].content or "")


def test_reset_session_does_not_widen_the_tool_allowlist() -> None:
    """Sessions must not become a route around the deny-by-default allowlist.

    `allowed_tools=()` denies everything and there is no wildcard value. No session API
    accepts tools, and none of them touches `config`.
    """
    agent = _agent("agent-a")
    assert agent.config.allowed_tools == ()

    agent.load_history(_dialogue(), turn_counter=1)
    agent.reset_session()
    agent.switch_session("sess_elsewhere")
    agent.get_session("sess_third")

    assert agent.config.allowed_tools == ()


# --------------------------------------------------------------------------------------
# A running agent converses in a session other than the one it started with (#225)
#
# The capability, what a switch costs, and the two refusals that survive it. `start`
# used to bind the subscription to `session.{session_id}` with no route off it, so a
# started agent raised on every switch; `switch_session` now repoints the live
# subscription instead.
# --------------------------------------------------------------------------------------


class _SignallingLLM(LLMProviderProtocol):
    """Records the messages of each turn and signals when a turn has been served."""

    def __init__(self) -> None:
        self.served = asyncio.Event()
        self.seen_messages: tuple[ChatMessage, ...] = ()

    @property
    def provider_name(self) -> str:
        return "signalling"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.seen_messages = request.messages
        self.served.set()
        return ModelResponse(
            content="an answer",
            usage=TokenUsage(provider="signalling", model="mock-model"),
            finish_reason=FinishReason.STOP,
            model_name="mock-model",
            provenance=Provenance(
                path=ExecutionPath.PRIMARY,
                requested=ServiceRef(provider="signalling", model="mock-model"),
                served_by=ServiceRef(provider="signalling", model="mock-model"),
            ),
        )

    def stream(self, request: LLMRequest):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _bus_agent(
    bus: EventBus,
    agent_id: str = "agent-a",
    *,
    llm: LLMProviderProtocol | None = None,
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id=agent_id,
            name="Multi Session Agent",
            system_prompt=SYSTEM_PROMPT,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        bus=bus,
        llm=llm,
    )


def _subscription_of(agent: BaseAgent) -> EventSubscriptionProtocol:
    sub = agent._subscription  # pyright: ignore[reportPrivateUsage]
    assert sub is not None
    return sub


@pytest.mark.asyncio
async def test_a_running_agent_converses_in_a_session_it_did_not_start_with() -> None:
    """#225's first acceptance criterion, end to end over the real bus.

    The agent starts on `sess_agent-a`, switches to `sess_other` while running, and a
    `USER_INPUT` addressed to `sess_other` reaches it and lands in *that* conversation.
    Both halves matter: delivery alone would be the mis-routing the old refusal existed
    to prevent, so the message must be in `sess_other`'s history and absent from the
    session the agent started with.
    """
    bus = EventBus()
    await bus.start()
    llm = _SignallingLLM()
    agent = _bus_agent(bus, llm=llm)
    await agent.start()
    try:
        assert agent.session_id == "sess_agent-a"
        agent.switch_session("sess_other")
        assert agent.session_id == "sess_other"

        await bus.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic="session.sess_other",
                recipient_id="agent-a",
                session_id="sess_other",
                sender_id="a-user",
                payload={"message": "a question for the second session"},
            )
        )
        await asyncio.wait_for(llm.served.wait(), timeout=2.0)

        landed = [m.content for m in agent.get_session("sess_other").messages]
        assert "a question for the second session" in landed

        started_with = [m.content for m in agent.get_session("sess_agent-a").messages]
        assert "a question for the second session" not in started_with
    finally:
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_a_switch_keeps_the_agent_and_broadcast_events_already_queued() -> None:
    """Repointing beats closing, and this is the difference measured.

    #225's option A — close the old subscription and open a new one — discards
    *everything* buffered on it, not only the departing session's events: the agent
    subscribes to `{agent.X, session.S, broadcast}` on one queue, so an unrelated
    `agent.X` event queued at the moment of the switch dies with the object. Retargeting
    re-queues it.

    The event loop is cancelled first so the queue is observable at the instant of the
    switch. That is a scheduling detail, not an artificial state: the running loop holds
    exactly this queue whenever it is parked inside a turn.
    """
    bus = EventBus()
    await bus.start()
    agent = _bus_agent(bus)
    await agent.start()
    try:
        loop_task = agent._loop_task  # pyright: ignore[reportPrivateUsage]
        assert loop_task is not None
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        sub = _subscription_of(agent)
        addressed = AgentEvent(
            type=EventType.TOOL_RESULT,
            topic="agent.agent-a",
            recipient_id="agent-a",
            payload={"kept": "addressed to the agent, not to a session"},
        )
        announced = AgentEvent(
            type=EventType.TOOL_RESULT,
            topic="broadcast",
            payload={"kept": "addressed to nobody in particular"},
        )
        departing = AgentEvent(
            type=EventType.USER_INPUT,
            topic="session.sess_agent-a",
            recipient_id="agent-a",
            session_id="sess_agent-a",
            payload={"message": "for the session being left"},
        )
        for event in (addressed, announced, departing):
            sub.deliver_nowait(event)
        assert sub.qsize() == 3

        agent.switch_session("sess_other")

        survivors = {sub.get_nowait().event_id for _ in range(sub.qsize())}
        assert addressed.event_id in survivors
        assert announced.event_id in survivors
        assert departing.event_id not in survivors
    finally:
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_events_a_switch_discards_are_counted_and_attributed_to_their_session() -> None:
    """#225's second acceptance criterion: no event is *silently* dropped.

    A switch cannot answer what was queued for the session it left — the turn would run
    against the new conversation — so those events are discarded. P6 makes a discard that
    is visible only in a log or a docstring a failure that has not been discharged, so it
    is counted on the agent, attributed to the session that lost it, and mirrored on the
    subscription's own drop accounting under a reason of its own.
    """
    bus = EventBus()
    await bus.start()
    agent = _bus_agent(bus)
    await agent.start()
    try:
        loop_task = agent._loop_task  # pyright: ignore[reportPrivateUsage]
        assert loop_task is not None
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        sub = _subscription_of(agent)
        assert agent.stranded_event_counts == {}

        for i in range(3):
            sub.deliver_nowait(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    topic="session.sess_agent-a",
                    recipient_id="agent-a",
                    session_id="sess_agent-a",
                    payload={"message": f"question {i}"},
                )
            )

        agent.switch_session("sess_other")

        assert agent.stranded_event_counts == {"sess_agent-a": 3}
        assert sub.drop_reasons["retarget_stranded"] == 3
        assert sub.dropped_event_count == 3

        # A second switch attributes to the session *it* leaves, so the counts identify
        # which conversation lost what rather than accumulating one anonymous total.
        sub.deliver_nowait(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic="session.sess_other",
                recipient_id="agent-a",
                session_id="sess_other",
                payload={"message": "and one more"},
            )
        )
        agent.switch_session("sess_third")
        assert agent.stranded_event_counts == {"sess_agent-a": 3, "sess_other": 1}
    finally:
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_a_switch_the_bus_refuses_leaves_the_agent_exactly_as_it_was() -> None:
    """A refused retarget must not half-switch the agent.

    If the bus's topic allowlist rejects the new session's topic, the agent would end up
    reporting a session whose events it cannot receive — the same mis-routing under a
    different cause. So it fails fast and nothing moves: not the active session id, not
    the subscription's topics or session filter, and not the hosted-session map.
    """
    bus = EventBus(topic_allowlist={"agent.*", "broadcast", "session.sess_agent-a"})
    await bus.start()
    agent = _bus_agent(bus)
    await agent.start()
    try:
        sub = _subscription_of(agent)
        topics_before = sub.topics
        sessions_before = agent.session_ids

        with pytest.raises(SessionSwitchWhileRunningError):
            agent.switch_session("sess_forbidden")

        assert agent.session_id == "sess_agent-a"
        assert agent.context.session_id == "sess_agent-a"
        assert sub.topics == topics_before
        assert sub.session_id == "sess_agent-a"
        assert agent.session_ids == sessions_before
        assert "sess_forbidden" not in agent.session_ids
    finally:
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_a_running_switch_still_refuses_while_a_turn_is_in_flight() -> None:
    """The mid-turn guard is now the *only* guard, so it is pinned on a running agent.

    `SessionSwitchWhileRunningError` used to refuse every switch on a started agent,
    which incidentally made the mid-turn case unreachable there. That cover is gone.
    `_refuse_session_mutation_during_turn` is what remains, and it has to hold on exactly
    the configuration the blanket refusal used to hide.
    """
    bus = EventBus()
    await bus.start()
    llm = _BlockingLLM()
    agent = _bus_agent(bus, llm=llm)
    await agent.start()
    try:
        task = asyncio.create_task(agent.execute_turn("the question"))
        await asyncio.wait_for(llm.entered.wait(), timeout=2.0)

        with pytest.raises(SessionMutationDuringTurnError):
            agent.switch_session("sess_other")
        assert agent.session_id == "sess_agent-a"

        llm.release.set()
        await asyncio.wait_for(task, timeout=2.0)
        # And it succeeds once the turn is done — the refusal is about the turn, not
        # about the agent being started.
        agent.switch_session("sess_other")
        assert agent.session_id == "sess_other"
    finally:
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_start_and_a_running_switch_produce_the_same_topic_shape() -> None:
    """The topics a running agent holds must not drift from the ones `start` gave it.

    Both go through `_subscription_topics`, so a switch cannot quietly widen or narrow
    what the agent listens on — only substitute the session. Compared as sets against the
    literal expected shape so that a change to either path has to change this test.
    """
    bus = EventBus()
    await bus.start()
    agent = _bus_agent(bus)
    await agent.start()
    try:
        sub = _subscription_of(agent)
        assert sub.topics == frozenset({"agent.agent-a", "session.sess_agent-a", "broadcast"})
        assert sub.session_id == "sess_agent-a"

        agent.switch_session("sess_other")

        assert sub.topics == frozenset({"agent.agent-a", "session.sess_other", "broadcast"})
        assert sub.session_id == "sess_other"
        assert sub.recipient_id == "agent-a"
    finally:
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_a_running_switch_does_not_widen_the_tool_allowlist() -> None:
    """The deny-by-default allowlist survives the new session route too (#225).

    `allowed_tools=()` denies everything and there is no wildcard value. The running
    switch is a new session path, and every session path has to stay outside `config`.
    """
    bus = EventBus()
    await bus.start()
    agent = _bus_agent(bus)
    await agent.start()
    try:
        assert agent.config.allowed_tools == ()
        agent.switch_session("sess_other")
        agent.switch_session("sess_third")
        assert agent.config.allowed_tools == ()
    finally:
        await agent.stop()
        await bus.stop()


def test_terminated_is_still_a_terminal_state() -> None:
    """#225 option C, rejected and pinned as rejected.

    Making `TERMINATED -> IDLE` legal would have let stop/switch/start stand in for a
    real fix, at the cost of weakening a terminal state in a P1 state machine. It was not
    taken, and this asserts the machine still says so: `stop()` is one-way, and every
    test that relies on `TERMINATED` being terminal keeps its meaning.
    """
    assert VALID_TRANSITIONS[AgentState.TERMINATED] == frozenset()
    for origin, allowed in VALID_TRANSITIONS.items():
        if origin is not AgentState.TERMINATED:
            assert AgentState.TERMINATED not in allowed or origin in {
                AgentState.IDLE,
                AgentState.INGESTING,
                AgentState.AWAITING_INPUT,
                AgentState.EMITTING_RESPONSE,
                AgentState.ERROR,
            }


@pytest.mark.asyncio
async def test_stop_after_a_running_switch_still_reaches_terminated_once() -> None:
    """Switching while running must not disturb the lifecycle it runs inside."""
    bus = EventBus()
    await bus.start()
    agent = _bus_agent(bus)
    await agent.start()
    agent.switch_session("sess_other")
    await agent.stop()
    assert agent.state == AgentState.TERMINATED
    with pytest.raises(InvalidStateTransitionError):
        await agent.start()
    await bus.stop()


def test_switch_session_is_allowed_before_start() -> None:
    agent = _agent("agent-a")
    state = agent.switch_session("sess_other")
    assert state.session_id == "sess_other"
    assert agent.session_id == "sess_other"
    assert agent.context.session_id == "sess_other"


# --------------------------------------------------------------------------------------
# Persistence through the Core store
# --------------------------------------------------------------------------------------


def test_persist_and_hydrate_round_trip_through_the_store(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    agent = _agent("agent-a", store=store)
    agent.load_history(_dialogue(), turn_counter=3)

    agent.persist_session()

    fresh = _agent("agent-a", store=store)
    hydrated = fresh.hydrate_session()

    assert hydrated is not None
    assert hydrated.turn_counter == 3
    assert fresh.history == _dialogue()


def test_reset_session_persists_when_a_store_is_wired(tmp_path: Path) -> None:
    """A reset that the next hydrate undoes is not a reset."""
    store = SessionStore(storage_dir=tmp_path)
    agent = _agent("agent-a", store=store)
    agent.load_history(_dialogue(), turn_counter=3)
    agent.persist_session()

    agent.reset_session()

    on_disk = store.load("sess_agent-a")
    assert on_disk is not None
    assert on_disk.turn_counter == 0
    assert len(on_disk.messages) == 1


def test_hydrate_session_returns_none_for_an_absent_record(tmp_path: Path) -> None:
    """An absent record is not a reason to discard a live conversation."""
    agent = _agent("agent-a", store=SessionStore(storage_dir=tmp_path))
    agent.load_history(_dialogue(), turn_counter=2)

    assert agent.hydrate_session() is None
    assert agent.history == _dialogue()
    assert agent.get_session().turn_counter == 2


def test_persistence_without_a_store_fails_fast() -> None:
    """Returning quietly would leave the caller believing an in-memory session is
    durable, which is the failure mode P6 forbids reporting as a success."""
    agent = _agent("agent-a")
    assert agent.store is None
    with pytest.raises(SessionStoreNotConfiguredError):
        agent.persist_session()
    with pytest.raises(SessionStoreNotConfiguredError):
        agent.hydrate_session()


def test_reset_session_without_a_store_still_resets_in_memory() -> None:
    agent = _agent("agent-a")
    agent.load_history(_dialogue(), turn_counter=4)
    reset = agent.reset_session()
    assert reset.turn_counter == 0
    assert agent.get_session().turn_counter == 0


def test_base_agent_satisfies_the_reset_session_protocol_member() -> None:
    """`reset_session` is on `BaseAgentProtocol` because it replaces three reset paths,
    one of which reached into `agent._history` behind a private-access pragma."""
    agent: BaseAgentProtocol = _agent("agent-a")
    state = agent.reset_session()
    assert state.turn_counter == 0


# --------------------------------------------------------------------------------------
# #211 review: a session must not be mutated under a turn already in flight
# --------------------------------------------------------------------------------------


class _BlockingLLM(LLMProviderProtocol):
    """A connector that parks inside `generate` until released.

    Lets a test hold `execute_turn` at the point where `_turn_lock` is held and the
    state is `REASONING`, which is exactly the window in which a reset used to be
    accepted.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.seen_messages: tuple[ChatMessage, ...] = ()

    @property
    def provider_name(self) -> str:
        return "blocking"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.seen_messages = request.messages
        self.entered.set()
        await self.release.wait()
        return ModelResponse(
            content="an answer",
            usage=TokenUsage(provider="blocking", model="mock-model"),
            finish_reason=FinishReason.STOP,
            model_name="mock-model",
            provenance=Provenance(
                path=ExecutionPath.PRIMARY,
                requested=ServiceRef(provider="blocking", model="mock-model"),
                served_by=ServiceRef(provider="blocking", model="mock-model"),
            ),
        )

    def stream(self, request: LLMRequest):  # type: ignore[no-untyped-def]
        raise NotImplementedError


@pytest.mark.asyncio
async def test_reset_is_refused_while_a_turn_is_in_flight_on_that_session() -> None:
    """Resetting mid-turn produced an answer with no question.

    `execute_turn` holds `_turn_lock` for the whole turn and appends as it goes, so a
    reset in that window discarded the user message the turn was answering. The turn
    then completed into the reset session and returned `[SYSTEM, ASSISTANT]` reported as
    `is_completed=True`, with the turn index the reset had just zeroed — and nothing
    about it was visible to either caller.
    """
    llm = _BlockingLLM()
    agent = _agent("agent-a", llm=llm)
    task = asyncio.create_task(agent.execute_turn("the question"))
    await asyncio.wait_for(llm.entered.wait(), timeout=2.0)

    with pytest.raises(SessionMutationDuringTurnError):
        agent.reset_session()

    llm.release.set()
    result = await asyncio.wait_for(task, timeout=2.0)

    # The turn was left intact: its question is still there, and so is its index.
    assert result.turn_index == 1
    assert result.is_completed is True
    roles = [m.role for m in agent.history]
    assert MessageRole.USER in roles
    assert any((m.content or "") == "the question" for m in agent.history)


@pytest.mark.asyncio
async def test_switch_is_refused_while_a_turn_is_in_flight_even_with_no_bus() -> None:
    """The bus guard could not cover this: it needs `_running` and a subscription.

    A not-yet-started agent, or one built with `bus=None`, has neither — so it slipped
    past `SessionSwitchWhileRunningError` while still being able to land the assistant
    message in whichever conversation became active.
    """
    llm = _BlockingLLM()
    agent = _agent("agent-a", llm=llm)
    assert agent._bus is None  # pyright: ignore[reportPrivateUsage]
    task = asyncio.create_task(agent.execute_turn("the question"))
    await asyncio.wait_for(llm.entered.wait(), timeout=2.0)

    with pytest.raises(SessionMutationDuringTurnError):
        agent.switch_session("sess_elsewhere")

    llm.release.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert agent.session_id == "sess_agent-a"


@pytest.mark.asyncio
async def test_load_history_and_hydrate_are_refused_mid_turn_too() -> None:
    """Every path that replaces a session's messages has the same hazard."""
    llm = _BlockingLLM()
    agent = _agent("agent-a", llm=llm, store=None)
    task = asyncio.create_task(agent.execute_turn("the question"))
    await asyncio.wait_for(llm.entered.wait(), timeout=2.0)

    with pytest.raises(SessionMutationDuringTurnError):
        agent.load_history(_dialogue(), turn_counter=0)

    llm.release.set()
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_a_non_active_session_may_still_be_reset_mid_turn() -> None:
    """The refusal is scoped to the session at risk. A turn never touches another one,
    so blocking every reset would be a stricter contract than the defect requires."""
    llm = _BlockingLLM()
    agent = _agent("agent-a", llm=llm)
    agent.load_history(_dialogue(), turn_counter=5, session_id="sess_other")
    task = asyncio.create_task(agent.execute_turn("the question"))
    await asyncio.wait_for(llm.entered.wait(), timeout=2.0)

    other = agent.reset_session("sess_other")
    assert other.turn_counter == 0

    llm.release.set()
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result.is_completed is True


# --------------------------------------------------------------------------------------
# #211 review: the write-through the CLI depends on, which nothing pinned
# --------------------------------------------------------------------------------------


def test_assigning_history_writes_through_to_the_owning_session() -> None:
    """Turning the `_history` setter into a silent no-op left the whole suite passing.

    That setter is the exact mechanism cited for "`cli/commands/run.py` still builds
    unchanged" — `run.py` assigns `agent._history` — so an unpinned no-op would have
    made the CLI's `/reset` print its banner and change nothing.
    """
    agent = _agent("agent-a")
    replacement = [ChatMessage(role=MessageRole.USER, content="written through")]

    agent._history = replacement  # pyright: ignore[reportPrivateUsage]

    assert agent.history == tuple(replacement)
    assert agent.get_session().messages == tuple(replacement)
    # And it landed in the session dict, not in a detached copy.
    assert agent._sessions["sess_agent-a"].messages == replacement  # pyright: ignore[reportPrivateUsage]


def test_assigning_the_turn_counter_writes_through_to_the_owning_session() -> None:
    agent = _agent("agent-a")

    agent._turn_counter = 7  # pyright: ignore[reportPrivateUsage]

    assert agent.get_session().turn_counter == 7
    assert agent._turn_counter == 7  # pyright: ignore[reportPrivateUsage]


def test_history_reads_through_after_the_session_entry_is_rebound() -> None:
    """`_history` must re-resolve, not cache. `load_history` rebinds the dict entry."""
    agent = _agent("agent-a")
    first = agent.history
    agent.load_history(_dialogue(), turn_counter=2)
    assert agent.history != first
    assert agent.history == _dialogue()


# --------------------------------------------------------------------------------------
# #211 review: late-failing validation, and the empty-id ambiguity
# --------------------------------------------------------------------------------------


def test_load_history_rejects_a_non_integer_turn_counter_at_the_call_that_supplied_it() -> None:
    """`_LiveSession` is a `slots` dataclass and validates nothing.

    `load_history(..., turn_counter="9")` used to be accepted, after which every later
    `get_session`, `reset_session` or `persist_session` raised `ValidationError` at a
    call site that had done nothing wrong. Validation on `SessionState.with_messages`
    alone cannot see this path unless the path goes through the model — so it now does.
    """
    agent = _agent("agent-a")
    with pytest.raises(ValidationError):
        agent.load_history(_dialogue(), turn_counter="9")  # pyright: ignore[reportArgumentType]
    # And the session is untouched, so nothing fails later either.
    assert agent.get_session().turn_counter == 0
    assert agent.persist_session is not None


@pytest.mark.parametrize("method", ["get_session", "reset_session"])
def test_an_empty_session_id_is_not_treated_as_absent(method: str) -> None:
    """`""` must not mean "the active session".

    The `or` idiom made `persist_session("")` silently write the *active* session under
    the active id, and `reset_session("")` reset a session the caller never named. That
    hands back exactly the empty-id ambiguity the store's guard refuses one layer down —
    and that guard is measurably stricter than `ui/app.py`'s on this very case.
    """
    agent = _agent("agent-a")
    with pytest.raises(PathTraversalError):
        getattr(agent, method)("")


def test_persist_session_with_an_empty_id_refuses_rather_than_writing_the_active_one(
    tmp_path: Path,
) -> None:
    store = SessionStore(storage_dir=tmp_path)
    agent = _agent("agent-a", store=store)
    agent.load_history(_dialogue(), turn_counter=3)

    with pytest.raises(PathTraversalError):
        agent.persist_session("")

    # Nothing was written under the active id either.
    assert store.load("sess_agent-a") is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("session_id", ["", "../../../etc/evil", "a/b", "c\\d"])
def test_switch_session_refuses_an_illegal_session_name(session_id: str) -> None:
    """An agent that accepted these would host a conversation it can never persist:
    the store refuses the name at write time, far from the call that chose it."""
    agent = _agent("agent-a")
    with pytest.raises(PathTraversalError):
        agent.switch_session(session_id)
    assert agent.session_id == "sess_agent-a"


@pytest.mark.parametrize("session_id", ["../../../etc/evil", "a/b", ""])
def test_addressing_an_illegal_session_name_does_not_seed_a_session(session_id: str) -> None:
    """A rejected read must not leave a hosted session behind."""
    agent = _agent("agent-a")
    with pytest.raises(PathTraversalError):
        agent.get_session(session_id)
    assert agent.session_ids == ("sess_agent-a",)


@pytest.mark.asyncio
async def test_each_refusal_names_the_operation_the_caller_actually_performed() -> None:
    """A refusal that names something other than what the caller did is a wrong message.

    `load_history` passed `"hydrate"` as its label, so a caller that called
    `load_history` was told it "cannot hydrate" — the same species as the
    `switch_session` message that named a remedy which did not exist. Pinned here rather
    than left as prose, because an unpinned wording claim is exactly what a later edit
    silently repeals.
    """
    llm = _BlockingLLM()
    agent = _agent("agent-a", llm=llm)
    task = asyncio.create_task(agent.execute_turn("the question"))
    await asyncio.wait_for(llm.entered.wait(), timeout=2.0)

    with pytest.raises(SessionMutationDuringTurnError, match="cannot reset"):
        agent.reset_session()
    with pytest.raises(SessionMutationDuringTurnError, match="cannot load history into"):
        agent.load_history(_dialogue())
    with pytest.raises(SessionMutationDuringTurnError, match="cannot switch away from"):
        agent.switch_session("sess_other")

    llm.release.set()
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_hydrate_session_still_says_hydrate(tmp_path: Path) -> None:
    """The label is per-route, so the one route that really does hydrate keeps the word."""
    llm = _BlockingLLM()
    agent = _agent("agent-a", llm=llm, store=SessionStore(storage_dir=tmp_path))
    task = asyncio.create_task(agent.execute_turn("the question"))
    await asyncio.wait_for(llm.entered.wait(), timeout=2.0)

    with pytest.raises(SessionMutationDuringTurnError, match="cannot hydrate"):
        agent.hydrate_session()

    llm.release.set()
    await asyncio.wait_for(task, timeout=2.0)


def test_the_second_door_points_back_at_the_family_too() -> None:
    """`BaseAgent.load_history` closes the family's second door, so it carries the pointer.

    Asserted in this seam rather than the store seam because that is where the fix lives:
    the module docstring says the door "and its fix both belong to the `BaseAgent`
    multi-session seam", and this is that seam. Splitting the assertion the same way the
    code is split is what keeps both PRs' paragraphs true at their own SHA — the defect
    two reviewers converged on was a sentence in one seam claiming something about call
    sites in another.
    """
    doc = BaseAgent.load_history.__doc__ or ""
    assert _mentions_family(doc), "load_history does not point back at the family paragraph"
    # And it names the concrete door it closes, which the store seam deliberately cannot:
    # `_LiveSession` does not exist there, and naming it was the forward reference that
    # made the store seam's paragraph false at its own SHA.
    assert "_LiveSession" in doc


# --------------------------------------------------------------------------------------
# Seam 3: compaction wiring, the LLMRequest propagation fix, CONTEXT_COMPACTED
# --------------------------------------------------------------------------------------


def _bulky_dialogue(turns: int = 12, filler: int = 400) -> tuple[ChatMessage, ...]:
    msgs: list[ChatMessage] = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
    for i in range(turns):
        msgs.append(ChatMessage(role=MessageRole.USER, content=f"q{i} " + "x" * filler))
        msgs.append(ChatMessage(role=MessageRole.ASSISTANT, content=f"a{i} " + "y" * filler))
    return tuple(msgs)


def _threshold_agent(
    threshold: int,
    *,
    auto_compact: bool = True,
    llm: LLMProviderProtocol | None = None,
    store: SessionStore | None = None,
    budget: TokenBudgetManager | None = None,
    bus: EventBus | None = None,
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id="agent-compact",
            name="Compacting Agent",
            system_prompt=SYSTEM_PROMPT,
            llm_config=AgentLLMConfig(
                model_name="mock-model",
                auto_compact=auto_compact,
                compaction_threshold_tokens=threshold,
            ),
        ),
        llm=llm,
        store=store,
        budget=budget,
        bus=bus,
    )


@pytest.mark.asyncio
async def test_compact_session_returns_measured_token_deltas() -> None:
    agent = _threshold_agent(60_000)
    agent.load_history(_bulky_dialogue(), turn_counter=12)
    before = len(agent.history)

    result = await agent.compact_session()

    assert result.session_id == "sess_agent-compact"
    assert result.reason == "manual_on_demand"
    assert result.ledger_source is LedgerSource.HEURISTIC
    assert result.messages_before == before
    assert result.messages_after < before
    assert result.tokens_after < result.tokens_before
    assert result.saved_tokens == result.tokens_before - result.tokens_after
    assert 0.0 < result.compression_ratio_pct <= 100.0
    assert result.keep_recent_turns == 4
    # The session now holds the compacted sequence.
    assert len(agent.history) == result.messages_after


@pytest.mark.asyncio
async def test_compact_session_carries_provenance_and_never_synthesizes_it() -> None:
    """P6: a compaction is a result crossing a boundary. The heuristic ledger is a local
    execution of the declared algorithm, so `primary` and undegraded."""
    agent = _threshold_agent(60_000)
    agent.load_history(_bulky_dialogue(), turn_counter=12)

    result = await agent.compact_session()

    assert result.provenance is not None
    assert result.provenance.path is ExecutionPath.PRIMARY
    assert result.provenance.served_by.provider == "uclone_x.llm.compactor"
    assert result.provenance.served_by.model == "heuristic-ledger"
    assert result.provenance.degraded is False
    # Never attributed to the agent itself.
    assert result.provenance.served_by.provider != "agent.core"


@pytest.mark.asyncio
async def test_compaction_result_provenance_is_the_compactor_outcome_forwarded() -> None:
    """The Core forwards the outcome's attribution verbatim, including its absence."""
    compactor = MagicMock(spec=ContextCompactorProtocol)
    compactor.keep_recent_turns = 4
    compactor.estimate_tokens = MagicMock(side_effect=[900, 300])
    compactor.compact = AsyncMock(
        return_value=CompactionOutcome(
            messages=(ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),),
            ledger_source=LedgerSource.LLM,
            superseded_ledger_count=2,
            provenance=None,
        )
    )
    agent = BaseAgent(
        config=AgentConfig(agent_id="agent-fwd", name="Forwarder"),
        compactor=compactor,
    )

    result = await agent.compact_session()

    assert result.provenance is None
    assert result.ledger_source is LedgerSource.LLM
    assert result.superseded_ledger_count == 2
    assert result.tokens_before == 900
    assert result.tokens_after == 300


@pytest.mark.asyncio
async def test_compact_session_does_not_reset_the_turn_counter() -> None:
    """Compaction discards context, not the fact that turns happened."""
    agent = _threshold_agent(60_000)
    agent.load_history(_bulky_dialogue(), turn_counter=12)
    await agent.compact_session()
    assert agent.get_session().turn_counter == 12


@pytest.mark.asyncio
async def test_compact_session_can_target_a_non_active_session() -> None:
    agent = _threshold_agent(60_000)
    agent.load_history(_bulky_dialogue(), turn_counter=12, session_id="sess_other")
    agent.load_history(_dialogue(), turn_counter=1)

    result = await agent.compact_session("sess_other")

    assert result.session_id == "sess_other"
    # The active session is untouched.
    assert agent.get_session().messages == _dialogue()


@pytest.mark.asyncio
async def test_compact_session_persists_through_the_store(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    agent = _threshold_agent(60_000, store=store)
    agent.load_history(_bulky_dialogue(), turn_counter=12)

    result = await agent.compact_session()

    on_disk = store.load("sess_agent-compact")
    assert on_disk is not None
    assert len(on_disk.messages) == result.messages_after
    assert on_disk.turn_counter == 12


@pytest.mark.asyncio
async def test_compact_session_records_metrics_on_the_budget_tracker() -> None:
    budget = TokenBudgetManager()
    agent = _threshold_agent(60_000, budget=budget)
    agent.load_history(_bulky_dialogue(), turn_counter=12)

    await agent.compact_session(reason="manual_on_demand")

    summary = budget.get_summary()
    history = summary.get("compaction_history", [])
    assert len(history) == 1
    assert history[0]["reason"] == "manual_on_demand"
    assert history[0]["saved_tokens"] > 0


# --------------------------------------------------------------------------------------
# Automatic compaction inside execute_turn (requirement 3)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_turn_compacts_when_the_configured_threshold_is_reached() -> None:
    """The compactor was fully implemented and never invoked from `execute_turn`, so a
    long conversation would overflow the context window."""
    agent = _threshold_agent(500, llm=_mock_llm())
    agent.load_history(_bulky_dialogue(), turn_counter=12)
    before = len(agent.history)

    await agent.execute_turn("one more question")

    # Compaction ran, so the history is shorter than the pre-turn dialogue.
    assert len(agent.history) < before
    assert any(m.compaction_ledger for m in agent.history)


@pytest.mark.asyncio
async def test_execute_turn_does_not_compact_when_auto_compact_is_false() -> None:
    """An agent configured `auto_compact=False` must be honoured, not silently ignored."""
    agent = _threshold_agent(500, auto_compact=False, llm=_mock_llm())
    agent.load_history(_bulky_dialogue(), turn_counter=12)
    before = len(agent.history)

    await agent.execute_turn("one more question")

    assert not any(m.compaction_ledger for m in agent.history)
    # Grew by the user message and the assistant reply, rather than shrinking.
    assert len(agent.history) == before + 2


@pytest.mark.asyncio
async def test_execute_turn_does_not_compact_below_the_threshold() -> None:
    agent = _threshold_agent(60_000, llm=_mock_llm())
    agent.load_history(_dialogue(), turn_counter=1)

    await agent.execute_turn("a short question")

    assert not any(m.compaction_ledger for m in agent.history)


@pytest.mark.asyncio
async def test_the_turn_request_is_built_from_the_compacted_sequence() -> None:
    """Compaction runs before `_prepare_turn_messages`, so the request the connector
    receives is the compacted one rather than being compacted after dispatch."""
    llm = _RecordingLLM()
    agent = _threshold_agent(500, llm=llm)
    agent.load_history(_bulky_dialogue(), turn_counter=12)

    await agent.execute_turn("one more question")

    sent = llm.last_request
    assert sent is not None
    assert any(m.compaction_ledger for m in sent.messages)
    assert len(sent.messages) < len(_bulky_dialogue())


@pytest.mark.asyncio
async def test_the_turn_request_carries_the_agents_compaction_policy() -> None:
    """The propagation gap: `BaseAgent` omitted both fields from the `LLMRequest` it
    built, so every request carried `LLMRequest`'s defaults and an agent configured
    `auto_compact=False` was silently ignored by the LLM layer.

    `llm-agnostic-interface.md` declares the fields on `LLMRequest`, which makes it the
    documented owner and `AgentLLMConfig` the per-agent declaration — the same
    relationship `temperature` and `max_tokens` already had.
    """
    llm = _RecordingLLM()
    agent = _threshold_agent(12_345, auto_compact=False, llm=llm)

    await agent.execute_turn("hello")

    sent = llm.last_request
    assert sent is not None
    assert sent.auto_compact is False
    assert sent.compaction_threshold_tokens == 12_345
    # The fields that already propagated still do.
    assert sent.model == "mock-model"


@pytest.mark.asyncio
async def test_a_non_positive_threshold_disables_automatic_compaction() -> None:
    agent = _threshold_agent(0, llm=_mock_llm())
    agent.load_history(_bulky_dialogue(), turn_counter=12)
    await agent.execute_turn("one more question")
    assert not any(m.compaction_ledger for m in agent.history)


# --------------------------------------------------------------------------------------
# CONTEXT_COMPACTED on the bus
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_publishes_an_attributed_context_compacted_event() -> None:
    """`AgentEvent` is frozen, strict and `extra="forbid"`, so the notice has to be a
    legal envelope, and P6 forbids an unattributed result."""
    bus = EventBus()
    agent = _threshold_agent(60_000, bus=bus)
    events = bus.subscribe("session.sess_agent-compact")
    agent.load_history(_bulky_dialogue(), turn_counter=12)

    result = await agent.compact_session(reason="manual_on_demand")

    received = await asyncio.wait_for(events.get(), timeout=2.0)
    assert received.type is EventType.CONTEXT_COMPACTED
    assert received.session_id == "sess_agent-compact"
    assert received.provenance is not None
    assert received.provenance.served_by.model == "heuristic-ledger"
    assert received.payload["reason"] == "manual_on_demand"
    assert received.payload["ledger_source"] == "heuristic"
    assert received.payload["tokens_before"] == result.tokens_before
    assert received.payload["tokens_after"] == result.tokens_after
    assert received.payload["saved_tokens"] == result.saved_tokens
    events.close()


@pytest.mark.asyncio
async def test_compaction_without_a_bus_is_supported_not_an_error() -> None:
    """A headless agent with no publisher is a supported configuration."""
    agent = _threshold_agent(60_000)
    agent.load_history(_bulky_dialogue(), turn_counter=12)
    result = await agent.compact_session()
    assert result.messages_after < result.messages_before


@pytest.mark.asyncio
async def test_context_compacted_is_a_first_class_event_type() -> None:
    """Joins `PROVIDER_FAILOVER` and `RETRY` on the enum's own stated standard: a
    decision-plane notice, not a free-form string."""
    assert EventType.CONTEXT_COMPACTED.value == "CONTEXT_COMPACTED"
    assert EventType("CONTEXT_COMPACTED") is EventType.CONTEXT_COMPACTED


# --------------------------------------------------------------------------------------
# Compactor lifecycle and the max_ledgers decision
# --------------------------------------------------------------------------------------


def test_each_session_gets_its_own_compactor_by_default() -> None:
    """Per-session ownership is what makes `superseded_ledger_count` a session figure in
    the default configuration — a shared instance would mix counts across conversations.
    """
    agent = _threshold_agent(60_000)
    first = agent._session_compactor("sess_a")  # pyright: ignore[reportPrivateUsage]
    second = agent._session_compactor("sess_b")  # pyright: ignore[reportPrivateUsage]
    again = agent._session_compactor("sess_a")  # pyright: ignore[reportPrivateUsage]

    assert first is not second
    assert first is again


def test_an_injected_compactor_is_honoured_and_shared() -> None:
    injected = ContextCompactor(keep_recent_turns=2)
    agent = BaseAgent(
        config=AgentConfig(agent_id="agent-inj", name="Injected"),
        compactor=injected,
    )
    assert agent._session_compactor("sess_a") is injected  # pyright: ignore[reportPrivateUsage]
    assert agent._session_compactor("sess_b") is injected  # pyright: ignore[reportPrivateUsage]


def test_max_ledgers_is_not_exposed_through_agent_configuration() -> None:
    """`max_ledgers` is unvalidated above (`max_ledgers=8` reproduces #196 — issue #204),
    so a configuration surface for it would hand callers a documented route back to the
    unbounded resident-block growth #201 fixed. This test fails if one is added before
    that bound lands."""
    assert "max_ledgers" not in AgentLLMConfig.model_fields
    with pytest.raises(ValueError):
        AgentLLMConfig(max_ledgers=8)  # pyright: ignore[reportCallIssue]


def test_the_default_compactor_has_the_bounded_ledger_budget() -> None:
    agent = _threshold_agent(60_000)
    compactor = agent._session_compactor("sess_a")  # pyright: ignore[reportPrivateUsage]
    assert isinstance(compactor, ContextCompactor)
    assert compactor.max_ledgers == 2
    # No summarizer by default: an extra provider call per turn is not imposed silently.
    assert compactor.summarizer is None


@pytest.mark.asyncio
async def test_base_agent_satisfies_the_compact_session_protocol_member() -> None:
    agent: BaseAgentProtocol = _threshold_agent(60_000)
    result = await agent.compact_session()
    assert result.session_id == "sess_agent-compact"


@pytest.mark.asyncio
async def test_explicit_compaction_is_refused_while_a_turn_is_in_flight() -> None:
    """Compaction rewrites the sequence the turn is mid-way through building.

    An explicit `compact_session` in that window discards the user message being
    answered — the same defect as a mid-turn reset. Automatic compaction is exempt by
    construction and goes through the unguarded private path; that exemption is pinned by
    `test_execute_turn_compacts_when_the_configured_threshold_is_reached`, which fails if
    the guard is applied to its own caller.
    """
    llm = _BlockingLLM()
    agent = _threshold_agent(60_000, llm=llm)
    agent.load_history(_bulky_dialogue(), turn_counter=12)
    task = asyncio.create_task(agent.execute_turn("the question"))
    await asyncio.wait_for(llm.entered.wait(), timeout=2.0)

    with pytest.raises(SessionMutationDuringTurnError):
        await agent.compact_session()

    llm.release.set()
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result.is_completed is True
    assert any("the question" == (m.content or "") for m in agent.history)


# --------------------------------------------------------------------------------------
# #214 review F2: the controls the field descriptions *state* must be pinned
#
# The general lesson, not just these four: a field description that asserts a control is
# a claim, and an unpinned claim is repealed by a one-line default. Both of the
# CompactionOutcome asserts below passed the full suite before these tests existed.
# --------------------------------------------------------------------------------------


def test_compaction_outcome_requires_its_ledger_source() -> None:
    """Its own description: "Required. A defaulted source would let an unattributed
    ledger pass as a heuristic one, which is the distinction this type exists to
    carry." Giving it `default=HEURISTIC` passed all 1164 tests."""
    assert CompactionOutcome.model_fields["ledger_source"].is_required()
    with pytest.raises(ValidationError):
        CompactionOutcome(provenance=None)  # pyright: ignore[reportCallIssue]


def test_compaction_outcome_requires_its_provenance_explicitly() -> None:
    """Its own description: "never inherited silently". `None` must be *stated*, not
    defaulted — the distinction P6 rests on, and `default=None` passed all 1164 tests."""
    assert CompactionOutcome.model_fields["provenance"].is_required()
    with pytest.raises(ValidationError):
        CompactionOutcome(ledger_source=LedgerSource.NONE)  # pyright: ignore[reportCallIssue]


def test_compaction_result_requires_its_provenance_explicitly() -> None:
    """The Core-side result carries the same P6 obligation as the LLM-side outcome."""
    assert CompactionResult.model_fields["provenance"].is_required()
    assert CompactionResult.model_fields["ledger_source"].is_required()


@pytest.mark.asyncio
async def test_the_short_dialogue_path_reports_the_ledgers_it_superseded() -> None:
    """`superseded_ledger_count` must be the real figure on the no-new-ledger path too.

    Hardcoding `0` there passed the suite. The whole budget is available to prior
    ledgers on that path, so a cap can still discard some, and their content is gone —
    an undercount is a loss reported as no loss (#196).
    """
    compactor = ContextCompactor(keep_recent_turns=8, max_ledgers=1)
    ledgers = [
        ChatMessage(role=MessageRole.SYSTEM, content=f"[ledger {i}]", compaction_ledger=True)
        for i in range(3)
    ]
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        *ledgers,
        ChatMessage(role=MessageRole.USER, content="only one recent turn"),
    ]

    outcome = await compactor.compact(messages)

    assert outcome.ledger_source is LedgerSource.NONE
    assert outcome.superseded_ledger_count == 2
    assert len([m for m in outcome.messages if m.compaction_ledger]) == 1


@pytest.mark.asyncio
async def test_compaction_result_reports_the_configured_recent_window_not_a_constant() -> None:
    """Hardcoding `keep_recent_turns=4` on the result passed the suite.

    The field's own description says it is "read off the compactor rather than assumed,
    so a non-default configuration is reported accurately" — so a non-default value is
    what has to be asserted.
    """
    agent = BaseAgent(
        config=AgentConfig(agent_id="agent-kw", name="Window", system_prompt=SYSTEM_PROMPT),
        compactor=ContextCompactor(keep_recent_turns=2),
    )
    agent.load_history(_bulky_dialogue(), turn_counter=9)

    result = await agent.compact_session()

    assert result.keep_recent_turns == 2


# --------------------------------------------------------------------------------------
# #214 review F1: a raising compaction is a turn failure, not an escape
# --------------------------------------------------------------------------------------


class _ExplodingStore(SessionStore):
    """A store whose `save` fails, standing in for a full or read-only disk."""

    def __init__(self, storage_dir: Path) -> None:
        super().__init__(storage_dir=storage_dir)
        self.armed = False

    def save(
        self, state: SessionState, pending_events: Sequence[Any] | None = None
    ) -> SessionState:
        if self.armed:
            raise OSError(28, "No space left on device")
        return super().save(state, pending_events=pending_events)


@pytest.mark.asyncio
async def test_a_failing_compaction_is_reported_as_a_failed_turn(tmp_path: Path) -> None:
    """It used to escape `execute_turn` raw from outside the `try` (#136b class).

    Reachable in the default no-summarizer configuration as soon as a `SessionStore` is
    wired: one failed write was enough. The state stayed on `INGESTING` rather than
    `ERROR`, the turn counter was burned, no notice was published, and no `TurnResult`
    came back at all — while the identical failure a few lines lower got `ERROR`, an
    attributed provenance and `TurnResult(is_completed=False)`.
    """
    store = _ExplodingStore(tmp_path)
    agent = _threshold_agent(500, llm=_mock_llm(), store=store)
    agent.load_history(_bulky_dialogue(), turn_counter=9)
    store.armed = True

    result = await agent.execute_turn("the question")

    assert result.is_completed is False
    assert result.error is not None
    assert agent.state is AgentState.ERROR


@pytest.mark.asyncio
async def test_a_failing_compaction_leaves_memory_and_disk_agreeing(tmp_path: Path) -> None:
    """Compaction is destructive and unrecoverable, so it commits atomically or not at all.

    The previous order replaced `live.messages` before persisting, so one failed `save`
    left the process believing a 25-message session was 6 messages long while the record
    still held 25 — with nothing in the exception to say so.
    """
    store = _ExplodingStore(tmp_path)
    agent = _threshold_agent(60_000, store=store)
    agent.load_history(_bulky_dialogue(), turn_counter=9)
    agent.persist_session()
    before = len(agent.history)
    on_disk_before = store.load("sess_agent-compact")
    assert on_disk_before is not None

    store.armed = True
    with pytest.raises(OSError):
        await agent.compact_session()

    assert len(agent.history) == before, "in-memory session was compacted despite the failure"
    on_disk_after = store.load("sess_agent-compact")
    assert on_disk_after is not None
    assert len(on_disk_after.messages) == len(on_disk_before.messages)


# --------------------------------------------------------------------------------------
# #214 review R2: the empty-id idiom, in the fifth mutating route
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_session_with_an_empty_id_refuses_rather_than_compacting_the_active_one() -> (
    None
):
    """Same shape as the `persist_session("")` finding on #211."""
    agent = _threshold_agent(60_000)
    agent.load_history(_bulky_dialogue(), turn_counter=9)
    before = len(agent.history)

    with pytest.raises(PathTraversalError):
        await agent.compact_session("")

    assert len(agent.history) == before


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Seam 4: the CLI and UI surfaces become clients of the Core (#183 requirement 4, P8)
# --------------------------------------------------------------------------------------


def test_the_traversal_guard_has_one_implementation_and_two_callers(tmp_path: Path) -> None:
    """`AgentSessionManager.get_session_path` and `SessionStore.session_path` now
    resolve through the same function. A duplicated security control is one that gets
    fixed in a single copy."""
    manager = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)

    assert manager.get_session_path("sess_a") == resolve_session_path(
        tmp_path / UI_TRANSCRIPT_SUBDIR, "sess_a"
    )
    for bad in ("../escape", "nested/child", "back\\slash", ""):
        with pytest.raises(PathTraversalError):
            manager.get_session_path(bad)


def test_the_ui_transcript_and_the_core_conversation_do_not_share_a_file(tmp_path: Path) -> None:
    """They have different schemas, so one filename would mean silent mutual overwrite.

    The transcript keeps `<storage_dir>/<session_id>.json` — the path already on disk
    for existing installs — and the Core conversation goes under `<storage_dir>/core/`.
    """
    manager = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    manager.save_session_record(
        session_id="sess_a",
        agent_id="agent-a",
        messages=[{"sender": "user", "content": "a UI presentation record"}],
        turns=1,
    )
    manager.core_store.save(
        SessionState(session_id="sess_a", agent_id="agent-a", messages=_dialogue(), turn_counter=1)
    )

    assert manager.get_session_path("sess_a") != manager.core_store.session_path("sess_a")
    # Both survive.
    record = manager.load_session_record("sess_a")
    core = manager.core_store.load("sess_a")
    assert record is not None and record["messages"][0]["sender"] == "user"
    assert core is not None and core.messages == _dialogue()


@pytest.mark.asyncio
async def test_clearing_a_ui_session_goes_through_the_core_reset(tmp_path: Path) -> None:
    """`clear_session_history` re-seeded the system prompt itself, in two duplicated
    blocks in one method body. It now delegates to `BaseAgent.reset_session`."""
    manager = AgentSessionManager(
        storage_dir=tmp_path, llm=MockLLMConnector(), fallback_to_mock=True
    )
    agent = await manager.get_or_create_agent(agent_id="agent-a", session_id="sess_a")
    agent.load_history(_dialogue(), turn_counter=4, session_id="sess_a")

    manager.clear_session_history(agent_id="agent-a", session_id="sess_a")

    state = agent.get_session("sess_a")
    assert state.turn_counter == 0
    assert len(state.messages) == 1
    assert state.messages[0].role == MessageRole.SYSTEM
    assert state.messages[0].content == agent.config.system_prompt
    await manager.stop_agent("agent-a", "sess_a")


def test_clearing_a_session_with_no_live_agent_deletes_the_persisted_record(
    tmp_path: Path,
) -> None:
    """Otherwise a cleared transcript would sit beside an intact conversation.

    The record is *deleted* rather than reset in place: `SessionState.reset` needs the
    agent's `config.system_prompt` to re-seed, and with no agent constructed there is
    nothing authoritative to read it from. Resetting with the empty default would
    persist a session carrying no system prompt, which is worse than no record —
    deleting lets the next `get_or_create_agent` seed correctly from config.
    """
    manager = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    manager.core_store.save(
        SessionState(session_id="sess_a", agent_id="agent-a", messages=_dialogue(), turn_counter=6)
    )

    manager.clear_session_history(agent_id="agent-a", session_id="sess_a")

    assert manager.core_store.load("sess_a") is None


def test_clearing_refuses_a_traversal_before_deleting_anything(tmp_path: Path) -> None:
    manager = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    with pytest.raises(PathTraversalError):
        manager.clear_session_history(agent_id="agent-a", session_id="../escape")


# --------------------------------------------------------------------------------------
# The two UI endpoints delegate to the Core rather than reimplementing it
# --------------------------------------------------------------------------------------


def _ui_client(tmp_path: Path) -> TestClient:
    app = create_ui_app(
        static_dir=tmp_path / "static",
        llm=MockLLMConnector(),
        storage_dir=tmp_path / "sessions",
    )
    return TestClient(app)


def _post_json(
    client: TestClient, path: str, payload: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    """Typed wrapper over `TestClient.post`, confining one untyped boundary.

    `TestClient`'s request methods are untyped under `pyright --strict`, which is why
    `tests/unit/test_ui_server.py` carries a file-level
    `reportUnknownMemberType=false` header. This file deliberately does not: that
    suppression would also cover the Core-level tests above, where an unknown type is
    exactly what should be caught. Three pragmas in one helper, rather than sixteen
    spread across the endpoint tests or a blanket header over all of them.
    """
    response = client.post(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        path, json=payload
    )
    status = response.status_code  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    body = cast(dict[str, Any], response.json())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return status, body


def test_the_cli_repl_no_longer_reaches_into_private_agent_state() -> None:
    """The three `reportPrivateUsage` pragmas in `cli/commands/run.py` were the
    falsifiable proof that no Core session API existed. They are gone."""
    source = Path(run_module.__file__ or "").read_text(encoding="utf-8")
    assert "agent._history" not in source.replace("# Was: `agent._history", "# Was: `agent_history")
    assert "reportPrivateUsage" not in source.replace("`reportPrivateUsage` pragma", "pragma")


def test_cli_session_id_resumes_the_conversation_it_had(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--session-id` would be decorative if the REPL did not persist on exit."""
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    runner = CliRunner()

    first = runner.invoke(
        cli_main.app,
        [
            "run",
            "cli_bot",
            "--provider",
            "mock",
            "--prompt",
            "remember this",
            "--session-id",
            "sess_x",
        ],
    )
    assert first.exit_code == 0

    # `SessionStore()` with no argument, so the test resolves the path the way the CLI
    # does rather than reconstructing it — a reconstruction is what hid the collision.
    store = SessionStore()
    saved = store.load("sess_x")
    assert saved is not None
    assert saved.turn_counter == 1
    assert any("remember this" == (m.content or "") for m in saved.messages)


def test_cli_reset_flag_clears_the_persisted_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    runner = CliRunner()
    store = SessionStore()
    store.save(
        SessionState(session_id="sess_y", agent_id="cli_bot", messages=_dialogue(), turn_counter=7)
    )

    result = runner.invoke(
        cli_main.app,
        [
            "run",
            "cli_bot",
            "--provider",
            "mock",
            "--prompt",
            "hi",
            "--session-id",
            "sess_y",
            "--reset",
        ],
    )

    assert result.exit_code == 0
    assert "reset" in result.output
    saved = store.load("sess_y")
    assert saved is not None
    # Reset zeroed it, then the single prompt ran one turn.
    assert saved.turn_counter == 1


def test_cli_compact_flag_runs_a_compaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    runner = CliRunner()
    store = SessionStore()
    store.save(
        SessionState(
            session_id="sess_z",
            agent_id="cli_bot",
            messages=_bulky_dialogue(),
            turn_counter=12,
        )
    )

    result = runner.invoke(
        cli_main.app,
        [
            "run",
            "cli_bot",
            "--provider",
            "mock",
            "--prompt",
            "hi",
            "--session-id",
            "sess_z",
            "--compact",
        ],
    )

    assert result.exit_code == 0
    assert "Compacted" in result.output
    saved = store.load("sess_z")
    assert saved is not None
    assert any(m.compaction_ledger for m in saved.messages)


def test_the_cli_core_path_and_the_ui_transcript_path_are_never_the_same_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single highest-value pin in this seam: the two paths must not collide."""
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    cli_core = SessionStore()
    manager = AgentSessionManager(fallback_to_mock=True)

    cli_core_path = cli_core.session_path("sess_shared")
    ui_transcript_path = manager.get_session_path("sess_shared")

    assert cli_core_path != ui_transcript_path
    # And the UI's Core store is deliberately the *same* file as the CLI's — one store
    # per P8, rather than two isolated copies of one conversation.
    assert manager.core_store.session_path("sess_shared") == cli_core_path


def test_a_ui_transcript_write_does_not_destroy_the_cli_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured destruction, pinned end to end.

    Before: `ucx run` wrote a Core record to `<root>/<id>.json`, the UI wrote a UI
    transcript to the same file, and the CLI's next read returned `None` — a silent
    total loss of the conversation.
    """
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    cli_core = SessionStore()
    cli_core.save(
        SessionState(
            session_id="sess_shared",
            agent_id="agent-a",
            messages=_dialogue(),
            turn_counter=3,
        )
    )
    manager = AgentSessionManager(fallback_to_mock=True)

    manager.save_session_record(
        session_id="sess_shared",
        agent_id="agent-a",
        messages=[{"sender": "user", "content": "a UI presentation record"}],
        turns=1,
    )

    survived = cli_core.load("sess_shared")
    assert survived is not None, "the UI transcript write destroyed the CLI conversation"
    assert survived.turn_counter == 3
    assert survived.messages == _dialogue()
    # And the transcript is readable too — neither destroyed the other.
    record = manager.load_session_record("sess_shared")
    assert record is not None
    assert record["messages"][0]["sender"] == "user"


def test_a_cli_core_write_does_not_destroy_the_ui_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction, which was equally lossy."""
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    manager = AgentSessionManager(fallback_to_mock=True)
    manager.save_session_record(
        session_id="sess_shared",
        agent_id="agent-a",
        messages=[{"sender": "user", "content": "keep me"}],
        turns=1,
    )

    SessionStore().save(SessionState.seed("sess_shared", "agent-a", SYSTEM_PROMPT))

    record = manager.load_session_record("sess_shared")
    assert record is not None, "the CLI Core write destroyed the UI transcript"
    assert record["messages"][0]["content"] == "keep me"


def test_a_legacy_root_transcript_is_still_readable_and_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installs predating the namespacing have transcripts at the root, and some of
    those files are Core records or collision hybrids. They are read, never written."""
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    tmp_path.mkdir(parents=True, exist_ok=True)
    legacy = tmp_path / "sess_old.json"
    legacy.write_text(
        json.dumps({"session_id": "sess_old", "turns": 2, "messages": [{"sender": "user"}]}),
        encoding="utf-8",
    )
    legacy_bytes = legacy.read_bytes()
    manager = AgentSessionManager(fallback_to_mock=True)

    record = manager.load_session_record("sess_old")
    assert record is not None
    assert record["turns"] == 2

    manager.save_session_record(
        session_id="sess_old", agent_id="agent-a", messages=[{"sender": "agent"}], turns=3
    )

    # The legacy file is untouched; the new write went to the namespaced path.
    assert legacy.read_bytes() == legacy_bytes
    assert (tmp_path / UI_TRANSCRIPT_SUBDIR / "sess_old.json").is_file()


@pytest.mark.asyncio
async def test_a_failing_core_reset_leaves_the_transcript_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core is reset before the transcript is unlinked.

    The previous order deleted the transcript first, so a Core reset that then failed
    left `transcript False / core True` — the displayed history gone while the
    conversation the model reasons over was intact. That is the least recoverable of the
    four outcomes, because the user sees an empty pane and the model does not.
    """
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    manager = AgentSessionManager(llm=MockLLMConnector(), fallback_to_mock=True)
    agent = await manager.get_or_create_agent(agent_id="agent-a", session_id="sess_a")
    agent.load_history(_dialogue(), turn_counter=2, session_id="sess_a")
    manager.save_session_record(
        session_id="sess_a", agent_id="agent-a", messages=[{"sender": "user"}], turns=1
    )
    transcript = manager.get_session_path("sess_a")
    assert transcript.is_file()

    def _boom(session_id: str | None = None) -> object:
        raise OSError("simulated reset failure")

    monkeypatch.setattr(agent, "reset_session", _boom)

    with pytest.raises(OSError):
        manager.clear_session_history(agent_id="agent-a", session_id="sess_a")

    assert transcript.is_file(), "transcript was deleted before the Core reset succeeded"
    await manager.stop_agent("agent-a", "sess_a")


def test_the_autouse_fixture_keeps_session_writes_out_of_the_real_home() -> None:
    """Guard on the isolation fixture itself.

    `UCLONE_SESSION_DIR` was added so a headless run need not write into the invoking
    user's home, and then `test_cli_run.py` and `test_ui_server.py` set it **zero**
    times — running the former grew `~/.uclone/sessions/sess_repl-drain.json` by ~1 KB
    per run and hydrated the previous run's history, so the suite was neither hermetic
    nor idempotent. An opt-in whose failure mode is invisible is the wrong shape; the
    fixture in `tests/conftest.py` is autouse for that reason, and this asserts it.
    """
    store = SessionStore()
    manager = AgentSessionManager(fallback_to_mock=True)
    real_home_sessions = (Path.home() / ".uclone" / "sessions").resolve()

    for observed in (store.storage_dir, manager.storage_dir, manager.core_store.storage_dir):
        assert not observed.is_relative_to(real_home_sessions), (
            f"{observed} is inside the developer's real session directory"
        )


# --------------------------------------------------------------------------------------
# #215 review: the Core refusal must be translated at the HTTP boundary
#
# Every route caught `PathTraversalError` for its 400 and let everything else reach
# Starlette as a bare 500 with no body. So the mid-turn guard was implemented in the
# Core and, from a client's point of view, did not exist.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raised", "expected_status"),
    [
        (PathTraversalError("bad id"), 400),
        (SessionMutationDuringTurnError("turn in flight"), 409),
        (SessionStoreNotConfiguredError("no store"), 500),
        (OSError("read-only file system"), 500),
        (RuntimeError("something else"), 500),
    ],
)
def test_core_session_failures_are_translated_to_their_own_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    expected_status: int,
) -> None:
    """409 for a mid-turn refusal in particular: the request is legal, just not yet.

    A 500 tells the frontend the server is broken and gives it nothing to say, when the
    correct answer is "retry when the turn finishes".

    Ported from `/api/chat/reset` and `/api/chat/compact` when #1208 retired them. The
    translation is `_translate_session_error`, one function reached by every session
    route; `/api/session/history/truncate` is the surviving route that reaches it from a
    JSON body, so it is where the mapping is pinned now.
    """

    def _raise(*args: object, **kwargs: object) -> None:
        raise raised

    monkeypatch.setattr(AgentSessionManager, "truncate_session_history", _raise)

    status, _ = _post_json(
        _ui_client(tmp_path),
        "/api/session/history/truncate",
        {"agent_id": "agent-a", "session_id": "sess_a", "index": 0},
    )

    assert status == expected_status


def test_the_history_delete_route_translates_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`DELETE /api/session/history` is the route the frontend actually calls, so an
    untranslated refusal here is the one a user meets."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise SessionMutationDuringTurnError("turn in flight")

    monkeypatch.setattr(AgentSessionManager, "clear_session_history", _raise)
    client = _ui_client(tmp_path)

    response = client.delete(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/session/history?agent_id=agent-a&session_id=sess_a"
    )

    assert response.status_code == 409  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


@pytest.mark.asyncio
async def test_core_first_hydration_is_actually_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling Core-first hydration entirely (`if False and ...`) passed the suite.

    The legacy transcript path would then silently take over, which is the P8 violation
    the Core record exists to retire.
    """
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    SessionStore().save(
        SessionState(
            session_id="sess_a",
            agent_id="agent-a",
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="core system"),
                ChatMessage(role=MessageRole.USER, content="only in the core record"),
            ),
            turn_counter=7,
        )
    )
    manager = AgentSessionManager(llm=MockLLMConnector(), fallback_to_mock=True)

    agent = await manager.get_or_create_agent(agent_id="agent-a", session_id="sess_a")

    assert agent.get_session("sess_a").turn_counter == 7
    assert any("only in the core record" == (m.content or "") for m in agent.history)
    await manager.stop_agent("agent-a", "sess_a")


def test_the_ui_guard_resolves_under_the_transcript_subdir(tmp_path: Path) -> None:
    """Re-inlining the P3 guard passed the suite.

    An inline copy pasted from the pre-extraction code would resolve against
    `self._storage_dir` — the root — and so silently reintroduce the collision. Asserting
    the resolved *directory*, not just that hostile ids are refused, is what distinguishes
    the delegation from a duplicate.
    """
    manager = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    resolved = manager.get_session_path("sess_a")
    assert resolved.parent == (tmp_path / UI_TRANSCRIPT_SUBDIR).resolve()
    assert resolved.parent != tmp_path.resolve()


def test_the_repl_persists_the_session_on_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The `finally` persist: removing it passed the suite, so `--session-id` was decorative.

    This asserts only what `/exit` proves — that the session is durable **after** a
    normal exit. It says nothing about a REPL that dies, and an earlier version of this
    docstring claimed it did; see
    `test_the_repl_session_is_durable_before_it_exits` for that half.
    """
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    inputs = iter(["Hello world", "/exit"])
    monkeypatch.setattr(
        "rich.prompt.Prompt.ask",
        lambda *args, **kwargs: next(inputs),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app, ["run", "repl_bot", "--provider", "mock", "--session-id", "sess_persist"]
    )

    assert result.exit_code == 0
    saved = SessionStore().load("sess_persist")
    assert saved is not None, "the REPL exited without persisting its session"
    assert saved.turn_counter == 1


def test_the_repl_session_is_durable_before_it_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The per-turn persist's *stated purpose*, which the exit test cannot reach.

    The claim is "a REPL that dies mid-session keeps its turns". Driving `/exit` reaches
    the `finally`, so it proves the exit persist and nothing about a death — the test
    written for the per-turn persist was verifying the other control, which is why
    removing either one alone escaped: they masked each other.

    Nothing short of `os._exit` skips a `finally`, so instead of killing the process this
    observes the store **at a moment mid-session**: the next prompt reads the record
    before answering. If only the `finally` persisted, there would be nothing there yet.
    """
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    observed: list[int | None] = []

    def _prompt(*args: object, **kwargs: object) -> str:
        if not observed:
            observed.append(None)  # first call: no turn has run yet
            return "Hello world"
        # Second call: one turn has completed and the REPL has NOT exited.
        mid = SessionStore().load("sess_midflight")
        observed.append(None if mid is None else mid.turn_counter)
        return "/exit"

    monkeypatch.setattr("rich.prompt.Prompt.ask", _prompt)
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app, ["run", "repl_bot", "--provider", "mock", "--session-id", "sess_midflight"]
    )

    assert result.exit_code == 0
    assert observed[1] == 1, (
        "after one completed turn and before exit the session was not durable "
        f"(observed {observed[1]!r}); the per-turn persist is not running"
    )


def test_a_transient_per_turn_persist_failure_is_recovered_at_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The one case only the `finally` persist covers, so both halves of the pair are
    independently observable rather than masking each other.

    A per-turn persist that fails is logged and the turn still returns — deliberately,
    since a disk hiccup must not discard an answer a model produced. The exit persist is
    what makes the session durable anyway.
    """
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    calls = {"n": 0}
    real_save = SessionStore.save

    def _fail_first(
        self: SessionStore, state: SessionState, *args: Any, **kwargs: Any
    ) -> SessionState:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return real_save(self, state, *args, **kwargs)

    monkeypatch.setattr(SessionStore, "save", _fail_first)
    inputs = iter(["Hello world", "/exit"])
    monkeypatch.setattr(
        "rich.prompt.Prompt.ask",
        lambda *args, **kwargs: next(inputs),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app, ["run", "repl_bot", "--provider", "mock", "--session-id", "sess_transient"]
    )

    assert result.exit_code == 0
    assert "not saved" in result.output, "the failed per-turn persist was not reported"
    monkeypatch.undo()
    saved = SessionStore(storage_dir=tmp_path / CORE_RECORD_SUBDIR).load("sess_transient")
    assert saved is not None, "the exit persist did not recover the transient failure"
    assert saved.turn_counter == 1


def test_a_traversal_session_id_refuses_cleanly_instead_of_tracebacking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The name rule fired correctly in the Core and the CLI printed a stack trace."""
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app,
        [
            "run",
            "cli_bot",
            "--provider",
            "mock",
            "--prompt",
            "hi",
            "--session-id",
            "../../etc/evil",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid --session-id" in result.output
    assert "Traceback" not in result.output


def test_an_empty_cli_session_id_is_refused_not_redirected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--session-id ""` silently became `sess_<agent>`; the same `or` idiom again."""
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app,
        ["run", "cli_bot", "--provider", "mock", "--prompt", "hi", "--session-id", ""],
    )

    assert result.exit_code == 2
    assert SessionStore().load("sess_cli_bot") is None


def test_a_ui_turn_makes_the_core_conversation_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The freeze, pinned.

    `execute_turn` deliberately does not persist — a disk failure must not discard an
    answer a model already produced — so the caller owns the cadence. The UI did not,
    which meant the Core record froze at whatever it held when it was created while the
    transcript kept growing: measured at agent history 6 against a displayed transcript
    of 12, and turn counters 6 against 3, silently.

    This was the one control in this seam that survived its own mutation check, so it is
    pinned rather than reported.
    """
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    client = _ui_client(tmp_path)

    for _ in range(2):
        status, _ = _post_json(
            client,
            "/api/turn",
            {"message": "hello", "agent_id": "agent-orchestrator", "session_id": "sess_durable"},
        )
        assert status == 200

    # Resolved through a manager rooted the same way the app is, rather than by
    # reconstructing the path — reconstruction is what hid the collision.
    core = AgentSessionManager(storage_dir=tmp_path / "sessions", fallback_to_mock=True).core_store
    on_disk = core.load("sess_durable")
    assert on_disk is not None, "two UI turns left no Core record: the conversation is not durable"
    assert on_disk.turn_counter == 2, (
        f"Core record froze at {on_disk.turn_counter} turn(s) while the transcript grew"
    )
    # And the Core record holds the actual conversation, not just its seed.
    assert any("hello" == (m.content or "") for m in on_disk.messages)


# --------------------------------------------------------------------------------------
# The P3 guard's single-implementation property, pinned structurally (#215 review, B)
#
# This escaped three consecutive reviews, and the reason is not that the pins were weak
# but that they were the **wrong shape**: every one asserted an outcome ("hostile ids are
# refused") or a location ("the resolved path is under `ui/`"), and an inlined duplicate
# satisfies both. "The guard exists in exactly one place" is not an outcome any single
# call can observe, so no behavioural assertion can ever catch its re-duplication.
#
# A structural claim needs a structural pin. This uses the AST-sweep idiom already
# carrying P6's static guards in `test_provenance.py` and `test_p6_failover_compliance.py`,
# including their `scanned` sanity assertion so a broken glob fails loudly instead of
# passing vacuously.
# --------------------------------------------------------------------------------------


#: Every function in `src/` permitted to perform a path-containment check, and why.
#:
#: `agent.session.resolve_session_path` is the session-storage guard: one implementation,
#: called by `SessionStore.session_path` and by both of `AgentSessionManager`'s path
#: resolvers. Re-inlining it in `ui/app.py` is what silently reintroduced the collision,
#: because a pasted copy resolves against the storage root rather than the transcript
#: subdirectory.
#:
#: `sandbox.path_validator` guards a **different boundary** with different semantics — a
#: caller-supplied workspace root and an arbitrary target path, rather than a filename
#: derived from a session id — so it is deliberately separate rather than an oversight.
#: Consolidating the two is a real question and not this issue's; it is noted on #183.
_CONTAINMENT_ALLOWLIST = frozenset(
    {
        "agent.session.resolve_session_path",
        "sandbox.path_validator.resolve_safe_path",
    }
)


def _functions_performing_containment_checks() -> tuple[set[str], int]:
    """Sweep `src/uclone_x` for functions that call `Path.is_relative_to`.

    Returns the qualified names and the number of modules scanned, so a glob that stops
    matching cannot make this assertion pass by finding nothing.
    """
    import ast

    src_root = Path(__file__).resolve().parents[2] / "src" / "uclone_x"
    found: set[str] = set()
    scanned = 0
    for path in sorted(src_root.rglob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = str(path.relative_to(src_root).with_suffix("")).replace("/", ".")
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if any(
                isinstance(inner, ast.Attribute) and inner.attr == "is_relative_to"
                for inner in ast.walk(node)
            ):
                found.add(f"{module}.{node.name}")
    return found, scanned


def test_the_containment_check_exists_in_exactly_one_session_storage_function() -> None:
    """Re-inlining the guard anywhere in `src/` fails here, which no behavioural test can do."""
    found, scanned = _functions_performing_containment_checks()

    assert scanned > 50, f"sweep only reached {scanned} modules; the glob is wrong"
    unexpected = found - _CONTAINMENT_ALLOWLIST
    assert not unexpected, (
        "path containment is checked in a function outside the allow-list: "
        f"{sorted(unexpected)}. The session-storage guard has exactly one "
        "implementation, `agent.session.resolve_session_path`; a second copy is the "
        "defect that reintroduced the CLI/UI collision, because a pasted copy resolves "
        "against the storage root rather than the transcript subdirectory. If a new "
        "boundary genuinely needs its own check, add it to _CONTAINMENT_ALLOWLIST with "
        "the reason it is not this one."
    )
    missing = _CONTAINMENT_ALLOWLIST - found
    assert not missing, (
        f"allow-listed guards have disappeared: {sorted(missing)}. An allow-list that "
        "names functions which no longer exist stops constraining anything."
    )


def test_the_ui_path_resolvers_delegate_rather_than_reimplement() -> None:
    """The complementary half: the UI's resolvers contain no containment logic at all.

    Asserted on `ui.app` specifically because that is where the duplicate lived, and
    because the sweep above would also pass if the UI stopped resolving paths entirely.
    """
    import ast

    src = (Path(__file__).resolve().parents[2] / "src" / "uclone_x" / "ui" / "app.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name not in {"get_session_path", "legacy_session_path"}:
            continue
        calls = {
            inner.func.id
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
        }
        assert "resolve_session_path" in calls, (
            f"ui.app.AgentSessionManager.{node.name} no longer delegates to resolve_session_path"
        )
        raises = {
            inner.exc.func.id
            for inner in ast.walk(node)
            if isinstance(inner, ast.Raise)
            and isinstance(inner.exc, ast.Call)
            and isinstance(inner.exc.func, ast.Name)
        }
        assert "PathTraversalError" not in raises, (
            f"ui.app.AgentSessionManager.{node.name} raises PathTraversalError itself, "
            "which means it is deciding containment rather than delegating it"
        )


@pytest.mark.asyncio
async def test_the_cli_and_the_ui_no_longer_lose_each_others_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cross-layer lost update, which this seam promoted from impossible to routine —
    and which #219 now refuses instead of applying.

    Before the collision fix the two layers wrote different roots, so cross-layer
    contention could not happen at all. Sharing one Core record is correct — it is what
    P8's single store means, and seam 1 rejected the alternative — but combined with the
    UI persisting on *every* turn it makes the read-modify-write window ordinary rather
    than rare. No fault injection here and no contrived timing.

    A **fourth** pin on the same defect, alongside the three in the "Atomicity is not
    isolation" block above, and the only one that goes through `AgentSessionManager` and
    `BaseAgent.persist_session` rather than the store directly. It is the one that shows
    the fix reaches the real call path and not just the store's own API: the UI layer
    commits, and the CLI's stale write is refused rather than silently discarding it.
    """
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    ui = AgentSessionManager(llm=MockLLMConnector(), fallback_to_mock=True)
    ui_agent = await ui.get_or_create_agent(agent_id="agent-a", session_id="sess_shared")
    ui_agent.load_history(_dialogue(), turn_counter=1, session_id="sess_shared")
    ui_agent.persist_session("sess_shared")

    # A CLI process opens the same session and reads the same base state.
    cli_store = SessionStore()
    cli_view = cli_store.load("sess_shared")
    assert cli_view is not None and cli_view.turn_counter == 1

    # The UI records its turn first and commits.
    ui_agent.load_history(
        (*_dialogue(), ChatMessage(role=MessageRole.USER, content="ui turn 2")),
        turn_counter=2,
        session_id="sess_shared",
    )
    ui_agent.persist_session("sess_shared")

    # The CLI writes back onto the revision it read and is refused.
    with pytest.raises(StaleSessionWriteError) as refusal:
        cli_store.save(
            cli_view.with_messages(
                (*cli_view.messages, ChatMessage(role=MessageRole.USER, content="cli turn 2")),
                turn_counter=2,
            )
        )
    assert refusal.value.session_id == "sess_shared"

    final = cli_store.load("sess_shared")
    assert final is not None
    assert final.turn_counter == 2
    contents = [m.content or "" for m in final.messages]
    # The UI's turn is on disk and the CLI's stale write did not overwrite it.
    assert "ui turn 2" in contents
    assert "cli turn 2" not in contents

    # And the UI keeps persisting per turn afterwards, which is the regression that would
    # bite hardest: the live session has to adopt the revision its own save wrote, or
    # every turn after the first would be refused against a record only it had written.
    ui_agent.load_history(
        (*_dialogue(), ChatMessage(role=MessageRole.USER, content="ui turn 3")),
        turn_counter=3,
        session_id="sess_shared",
    )
    third = ui_agent.persist_session("sess_shared")
    assert third.turn_counter == 3
    assert third.revision > final.revision
    await ui.stop_agent("agent-a", "sess_shared")


# --------------------------------------------------------------------------------------
# #256 (#219 defect 1): a session id is a filename, and APFS/NTFS fold it
#
# Two shapes of test here, deliberately, because one shape alone is unsafe:
#
# * **Filesystem-independent**, by writing a record file whose *name* is one id and whose
#   `session_id` field is another. That is byte-for-byte the state a folded filesystem
#   produces, so these pin the rule itself and they run identically on ext4 — where the
#   defect cannot be provoked at all — and on APFS.
# * **Filesystem-dependent**, provoking the real fold through `save`/`load`. These are the
#   only ones that show the defect as a user meets it, and they are worthless on a
#   case-sensitive volume, so they skip there rather than passing vacuously (§6.9 case 2:
#   an instrument must assert it observed what it claims to observe).
#
# Each test takes pytest's per-test `tmp_path` and its own session ids, so no case reads
# state another case created (#250 — a sequential probe of these very doors under-reported
# by two of three, in the reassuring direction, because each case moved the record the
# next was measured against).
# --------------------------------------------------------------------------------------


def _filesystem_folds_case(directory: Path) -> bool:
    """Measure whether `directory`'s filesystem compares names case-insensitively.

    Asserts the probe file actually landed before drawing any conclusion from the
    lookup — an unwritten probe would make every volume look case-sensitive, which is
    the reassuring answer and would silently skip the tests that matter.
    """
    probe = directory / "_FoldProbe.json"
    probe.write_text("{}", encoding="utf-8")
    assert probe.is_file(), f"probe was never written to {directory}; the measurement is void"
    try:
        return (directory / "_foldprobe.json").is_file()
    finally:
        probe.unlink()


def _filesystem_folds_normalization(directory: Path) -> bool:
    """Measure whether `directory`'s filesystem ignores Unicode normalization.

    Separate from the case measurement because the two are separate properties: a volume
    could fold one and not the other, and a test skipped for the wrong reason is a test
    that reports nothing.
    """
    nfc = unicodedata.normalize("NFC", "_FoldProbeé.json")
    nfd = unicodedata.normalize("NFD", "_FoldProbeé.json")
    assert nfc != nfd, "the probe names are identical; this measures nothing"
    probe = directory / nfc
    probe.write_text("{}", encoding="utf-8")
    assert probe.is_file(), f"probe was never written to {directory}; the measurement is void"
    try:
        return (directory / nfd).is_file()
    finally:
        probe.unlink()


def _write_raw_record(path: Path, *, session_id: str, **extra: object) -> None:
    """Write a valid `SessionState` record at `path`, whatever `path` is named.

    The point is the mismatch: `path.stem` need not equal `session_id`, which is exactly
    the on-disk state a case- or normalization-folding filesystem produces when two
    distinct ids are saved. Built through `SessionState` rather than a hand-rolled dict so
    the record is one `load` would otherwise accept — a test that wrote an invalid record
    would be pinned by the unparseable-reads-as-absent path instead, and would pass
    against the defect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = SessionState(session_id=session_id, agent_id="agent-a", **cast(Any, extra))
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8")


def test_load_refuses_a_record_that_identifies_a_different_session(tmp_path: Path) -> None:
    """The read arm, filesystem-independently: `load(x)` must never return a record naming y.

    This is the whole defect in one assertion. On `e8b3e2f` this returned `SessA`'s state
    — 7 turns, `session_id='SessA'` — to a caller that asked for `SESSA`, which is
    cross-session information disclosure with nothing concurrent happening.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    _write_raw_record(store.storage_dir / "SESSA.json", session_id="SessA", turn_counter=7)

    with pytest.raises(SessionIdCollisionError) as excinfo:
        store.load("SESSA")

    assert excinfo.value.asked_session_id == "SESSA"
    assert excinfo.value.record_session_id == "SessA"
    assert excinfo.value.path == store.session_path("SESSA")


def test_load_refuses_rather_than_reporting_absent(tmp_path: Path) -> None:
    """`None` was the tempting answer and it is the wrong one (P6).

    A foreign record is not "no session here": the caller would take `None` as a new
    session, hold a whole conversation, and only discover at the closing `save` — where
    the fold is unavoidable — that the id was never usable. The refusal has to be at the
    first touch, so the distinction between `None` and a raise is itself pinned.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    _write_raw_record(store.storage_dir / "SESSA.json", session_id="SessA")

    with pytest.raises(SessionIdCollisionError):
        store.load("SESSA")
    # And the honest absence is still an absence, not a collision.
    assert store.load("some_other_id") is None


def test_save_refuses_before_it_reports_a_revision_conflict(tmp_path: Path) -> None:
    """#256's confusing side effect: the ordering inside `save` is the acceptance criterion.

    On `e8b3e2f` a *genuinely new* session colliding on case was refused with
    `StaleSessionWriteError` — "it was read at revision 0 and the record is now at
    revision 1, so this write would discard another writer's update" — naming a writer
    and a revision for a session the user had never heard of. The refusal was an accident
    of #240 and it described the wrong defect. The identity check must run first.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    _write_raw_record(store.storage_dir / "sessa.json", session_id="SessA", turn_counter=3)

    with pytest.raises(SessionIdCollisionError) as excinfo:
        store.save(SessionState(session_id="sessa", agent_id="agent-b"))

    # Not merely "some error": specifically not the revision story, which is the defect.
    assert not isinstance(excinfo.value, StaleSessionWriteError)
    assert "revision" not in str(excinfo.value)
    assert excinfo.value.record_session_id == "SessA"


def test_the_revision_precondition_still_reports_a_real_revision_conflict(
    tmp_path: Path,
) -> None:
    """The complementary half: #240 must not have been weakened to make room for #256.

    A same-id stale write is still a `StaleSessionWriteError` carrying the on-disk state
    to rebase onto. Without this, moving the identity check ahead of the precondition
    could have swallowed the precondition entirely and both tests above would still pass.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    first = store.save(SessionState(session_id="sess_a", agent_id="agent-a"))
    store.save(first.model_copy(update={"turn_counter": 1}))

    with pytest.raises(StaleSessionWriteError) as excinfo:
        store.save(first.model_copy(update={"turn_counter": 99}))

    assert excinfo.value.expected_revision == first.revision
    assert excinfo.value.current is not None
    assert excinfo.value.current.turn_counter == 1


def test_delete_refuses_a_record_that_identifies_a_different_session(tmp_path: Path) -> None:
    """The most destructive of the variant-id doors, and one #256's table does not list.

    Measured on `e8b3e2f`: `delete("SESSA")` returned `True` having unlinked
    `SessA.json`. A caller annihilated a conversation it had never named and was told it
    succeeded. Found by enumerating the store's surface rather than the card's cases.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    record = store.storage_dir / "SESSA.json"
    _write_raw_record(record, session_id="SessA", turn_counter=11)

    with pytest.raises(SessionIdCollisionError):
        store.delete("SESSA")

    # The assertion that matters is not the exception but the survival of the record.
    assert record.is_file()
    assert json.loads(record.read_text(encoding="utf-8"))["turn_counter"] == 11


def test_an_unparseable_record_is_still_deletable(tmp_path: Path) -> None:
    """The carve-out in `delete`, stated so it cannot be tightened away by accident.

    A record too damaged to name its session holds no id to compare, so refusing would
    make it impossible to clear — a guard whose failure mode is an undeletable file.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    junk = store.storage_dir / "sess_junk.json"
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_text("{not json at all", encoding="utf-8")

    assert store.delete("sess_junk") is True
    assert not junk.exists()


def test_hydrate_session_refuses_a_record_that_identifies_a_different_session(
    tmp_path: Path,
) -> None:
    """The sharpest door: the adopted state's id was rewritten to the asked-for one.

    On `e8b3e2f`, `hydrate_session("SESSA")` against a stored `"SessA"` returned that
    session's conversation *and* keyed it into `BaseAgent._sessions` under `"SESSA"`, so
    `get_session("SESSA")` reported `session_id='SESSA'` over another session's history —
    the evidence of whose conversation it was had been erased before any caller could
    look. The following `persist_session("SESSA")` then wrote it back onto `SessA.json`
    with a matching revision.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    _write_raw_record(store.storage_dir / "SESSA.json", session_id="SessA", turn_counter=42)
    agent = BaseAgent(config=AgentConfig(agent_id="agent-b", name="B"), store=store)

    with pytest.raises(SessionIdCollisionError):
        agent.hydrate_session("SESSA")

    # Nothing in memory was replaced, and no state claiming 42 turns exists anywhere.
    assert "SESSA" not in agent.session_ids


def test_the_ui_transcript_refuses_a_record_that_identifies_a_different_session(
    tmp_path: Path,
) -> None:
    """The Core store is not the only artifact keyed by a session id in a filename.

    Measured on `e8b3e2f`: after `save_session_record("SessA", ...)`,
    `load_session_record("SESSA")` returned `SessA`'s transcript and `get_session_history`
    cached it under `"SESSA"`, so the dashboard showed one session's conversation as
    another's. #256 names only the Core store; this door was found by enumerating the
    id-bearing surface.
    """
    manager = AgentSessionManager(storage_dir=tmp_path, llm=MockLLMConnector())
    transcript = manager.get_session_path("SessA")
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        json.dumps({"session_id": "SessA", "agent_id": "agent-a", "turns": 3, "messages": []}),
        encoding="utf-8",
    )
    # Rename onto the variant so the *file* is the variant and the *content* is not,
    # which is what the fold produces. Done explicitly so this runs on ext4 too.
    variant = manager.get_session_path("SESSA")
    if variant != transcript:
        transcript.rename(variant)

    with pytest.raises(SessionIdCollisionError):
        manager.load_session_record("SESSA")
    with pytest.raises(SessionIdCollisionError):
        manager.get_session_history("agent-b", "SESSA")
    with pytest.raises(SessionIdCollisionError):
        manager.clear_session_history(agent_id="agent-b", session_id="SESSA")

    # `clear_session_history` refused *before* destroying anything, which is the half a
    # raised exception alone does not establish.
    assert variant.is_file()


def test_a_transcript_that_cannot_name_its_session_is_still_readable(tmp_path: Path) -> None:
    """The UI check fires on positive evidence only, so a legacy file is not locked out.

    A pre-namespacing transcript with no `session_id` key has no owner to compare, and
    refusing it would take working sessions away from the dashboard to fix a defect it
    does not have.
    """
    manager = AgentSessionManager(storage_dir=tmp_path, llm=MockLLMConnector())
    path = manager.get_session_path("sess_legacy")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"messages": [], "turns": 0}), encoding="utf-8")

    record = manager.load_session_record("sess_legacy")

    assert record is not None
    assert "session_id" not in record


def test_the_collision_error_names_which_fold_explains_it() -> None:
    """The normalization message reads as a self-contradiction without this.

    NFC `"séance"` and its NFD spelling render identically in a terminal and under
    `repr`, so an unqualified message says *"for session 'séance': it identifies session
    'séance'"* and the user cannot see what is being complained about.
    """
    nfc = unicodedata.normalize("NFC", "séance")
    nfd = unicodedata.normalize("NFD", "séance")
    assert nfc != nfd

    # Each clause is asserted **exclusively**, not as a substring, and that is the whole
    # reason this block is written the way it is. The first version asserted only
    # `"differs only by letter case" in message`, and a mutation disabling the case-only
    # branch **escaped**: control fell through to the case-and-normalization branch, whose
    # message *contains* that phrase, so the assertion passed against the defect. That is
    # §6.9's coincidence case — the fixed and broken code produced messages this test could
    # not distinguish. Naming the clause each input must NOT produce is what separates them.
    with pytest.raises(SessionIdCollisionError) as case_only:
        verify_record_identity("SESSA", "SessA", Path("/tmp/SESSA.json"))
    case_message = str(case_only.value)
    assert "differs only by letter case" in case_message
    assert "Unicode normalization)" not in case_message
    assert "case and Unicode normalization" not in case_message

    with pytest.raises(SessionIdCollisionError) as norm_only:
        verify_record_identity(nfd, nfc, Path("/tmp/x.json"))
    norm_message = str(norm_only.value)
    assert "differs only by Unicode normalization" in norm_message
    assert "letter case" not in norm_message
    # The escaped forms, because the printable ones are indistinguishable — this is the
    # assertion that the message is actually usable rather than merely present.
    assert "s\\xe9ance" in norm_message
    assert "se\\u0301ance" in norm_message

    # Both folds at once, which is the case a real macOS tree produces most easily:
    # a differently-cased id typed on a keyboard that emits NFD.
    with pytest.raises(SessionIdCollisionError) as both:
        verify_record_identity(unicodedata.normalize("NFD", "SÉANCE"), nfc, Path("/tmp/x.json"))
    both_message = str(both.value)
    assert "differs only by letter case and Unicode normalization" in both_message

    # And a mismatch no fold explains does not get a fabricated cause (P6).
    with pytest.raises(SessionIdCollisionError) as unrelated:
        verify_record_identity("sess_a", "sess_b", Path("/tmp/sess_a.json"))
    assert "differs only" not in str(unrelated.value)


def test_a_matching_id_is_not_refused() -> None:
    """The complementary half: a guard that refused everything would pass every test above."""
    verify_record_identity("sess_a", "sess_a", Path("/tmp/sess_a.json"))
    nfd = unicodedata.normalize("NFD", "séance")
    verify_record_identity(nfd, nfd, Path("/tmp/x.json"))


def test_the_ui_translates_a_collision_to_409_not_500(tmp_path: Path) -> None:
    """A refusal implemented in the Core and not translated at the boundary does not exist.

    409 rather than 400: #256 states the P3 guard is not implicated, so the id is legal
    and the request well-formed. What makes it unserviceable is the current state of the
    store, which is what 409 describes.
    """
    manager = AgentSessionManager(storage_dir=tmp_path, llm=MockLLMConnector())
    core = manager.core_store
    _write_raw_record(core.storage_dir / "SESSA.json", session_id="SessA", turn_counter=4)
    transcript = manager.get_session_path("SESSA")
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(json.dumps({"session_id": "SessA", "messages": []}), encoding="utf-8")

    app = create_ui_app(static_dir=tmp_path / "static", session_manager=manager)
    with TestClient(app) as client:
        # Pragmas rather than a file-level header, matching `_post_json` above: this file
        # deliberately keeps unknown types fatal for the Core-level tests.
        response = client.delete(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            "/api/session/history?agent_id=agent-b&session_id=SESSA"
        )

    assert response.status_code == 409, response.text  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    body = cast(dict[str, Any], response.json())  # pyright: ignore[reportUnknownMemberType]
    assert "SessA" in body["detail"]


def test_the_cli_reports_an_unusable_session_id_instead_of_resuming_another_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`ucx run --session-id SESSA` printed "Resumed session 'SessA'" and overwrote it.

    The same usage-error class as a traversal id and a different cause: this id is legal
    and this filesystem cannot tell it from one that already exists.
    """
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
    _write_raw_record(
        tmp_path / CORE_RECORD_SUBDIR / "SESSA.json", session_id="SessA", turn_counter=9
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app,
        ["run", "cli_bot", "--provider", "mock", "--prompt", "hi", "--session-id", "SESSA"],
    )

    assert result.exit_code == 2
    assert "Unusable --session-id" in result.output
    assert "Traceback" not in result.output
    assert "Resumed session" not in result.output
    # The other session's record is untouched, which is the consequence under test.
    on_disk = json.loads((tmp_path / CORE_RECORD_SUBDIR / "SESSA.json").read_text("utf-8"))
    assert on_disk["turn_counter"] == 9
    assert on_disk["session_id"] == "SessA"


def test_list_session_ids_reports_real_ids_because_nothing_is_encoded(tmp_path: Path) -> None:
    """The encoding decision, pinned where it is observable.

    #256 requires `list_session_ids()` to report the real ids "under whatever encoding is
    chosen". The encoding chosen is **none** — the identity check reads the `session_id`
    inside the record, so the filename stays the plain id and no record on disk is
    renamed. A percent-encoding or a hash would have made every stem here wrong, which is
    the migration this test exists to say was not performed.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    for sid in ("SessA", "sess_b", unicodedata.normalize("NFC", "séance")):
        store.save(SessionState(session_id=sid, agent_id="agent-a"))

    ids = store.list_session_ids()

    assert ids == tuple(sorted(("SessA", "sess_b", unicodedata.normalize("NFC", "séance"))))
    # Stems, not encodings: every id is directly loadable under the name it reports.
    for sid in ids:
        assert (store.storage_dir / f"{sid}.json").is_file()
        loaded = store.load(sid)
        assert loaded is not None and loaded.session_id == sid


# --------------------------------------------------------------------------------------
# The filesystem-dependent arm: the defect as a user actually meets it.
#
# These provoke the real fold through the public API instead of planting a mismatched
# record, so they are the only ones that show `save` and `load` colliding on their own.
# They are also the only ones that can pass vacuously, so each skips on a volume that
# does not fold rather than reporting a pass it did not earn.
# --------------------------------------------------------------------------------------


def test_two_ids_differing_only_by_case_cannot_silently_share_one_record(
    tmp_path: Path,
) -> None:
    """Path A/D on APFS: `save("SessA")` then `load("SESSA")` returned SessA's conversation."""
    root = tmp_path / "core"
    root.mkdir(parents=True, exist_ok=True)
    if not _filesystem_folds_case(root):
        pytest.skip("filesystem is case-sensitive; this defect is unobservable here")
    store = SessionStore(storage_dir=root)
    store.save(SessionState(session_id="SessA", agent_id="agent-a", turn_counter=7))

    with pytest.raises(SessionIdCollisionError):
        store.load("SESSA")
    with pytest.raises(SessionIdCollisionError):
        store.save(SessionState(session_id="sessa", agent_id="agent-b", turn_counter=1))

    # The first session is intact and is the only one enumerated.
    survivor = store.load("SessA")
    assert survivor is not None and survivor.turn_counter == 7
    assert store.list_session_ids() == ("SessA",)


def test_two_ids_differing_only_by_unicode_normalization_cannot_share_one_record(
    tmp_path: Path,
) -> None:
    """The NFC/NFD arm, which is a separate filesystem property from the case arm."""
    root = tmp_path / "core"
    root.mkdir(parents=True, exist_ok=True)
    if not _filesystem_folds_normalization(root):
        pytest.skip("filesystem preserves Unicode normalization; unobservable here")
    nfc = unicodedata.normalize("NFC", "séance")
    nfd = unicodedata.normalize("NFD", "séance")
    store = SessionStore(storage_dir=root)
    store.save(SessionState(session_id=nfc, agent_id="agent-a", turn_counter=5))

    with pytest.raises(SessionIdCollisionError):
        store.load(nfd)
    with pytest.raises(SessionIdCollisionError):
        store.save(SessionState(session_id=nfd, agent_id="agent-b"))

    survivor = store.load(nfc)
    assert survivor is not None and survivor.turn_counter == 5


def test_the_read_modify_write_arm_cannot_overwrite_the_other_session(tmp_path: Path) -> None:
    """Path B, the arm the product actually uses and the one #240 cannot see.

    `ucx run` and the dashboard both hydrate before writing. On `e8b3e2f` the caller
    asked for `SESSA`, was handed a state whose `session_id` was `SessA`, appended a turn
    and persisted what it was handed — accepted, because a caller that read first holds a
    current revision. The turn counter went to 99 on a session the caller never named.
    """
    root = tmp_path / "core"
    root.mkdir(parents=True, exist_ok=True)
    if not _filesystem_folds_case(root):
        pytest.skip("filesystem is case-sensitive; this defect is unobservable here")
    store = SessionStore(storage_dir=root)
    store.save(SessionState(session_id="SessA", agent_id="agent-a", turn_counter=1))

    with pytest.raises(SessionIdCollisionError):
        held = store.load("SESSA")
        # Unreachable while the fix holds. Written out so this test describes the whole
        # sequence rather than only its first step: if `load` ever stops refusing, the
        # write below is what silently destroys the other session, and this test then
        # fails on the *write* instead of passing because the read succeeded.
        assert held is not None
        store.save(held.model_copy(update={"turn_counter": 99}))

    final = store.load("SessA")
    assert final is not None
    assert final.turn_counter == 1, "the other session's record was overwritten"


#: Every function in `src/` permitted to decide that a record belongs to another session.
#:
#: One entry, deliberately. `agent.session.verify_record_identity` is the whole rule, and
#: the reason it needs a structural pin is the same reason `_CONTAINMENT_ALLOWLIST` does:
#: the check is three lines long and its call sites are spread across two layers, so
#: re-inlining `if record["session_id"] != session_id: raise` at a fourth door is the
#: cheapest possible mistake to make. Two copies of this rule would be two things to fix,
#: and the copy that got missed would be the disclosure.
#:
#: `resolve_session_path` deliberately does **not** appear: it cannot host the rule, since
#: the rule needs the record's contents and a path resolver has none. That is why #256 is
#: not fixed inside the existing P3 guard.
_IDENTITY_ALLOWLIST = frozenset({"agent.session.verify_record_identity"})


def _functions_raising_session_id_collision() -> tuple[set[str], int]:
    """Sweep `src/uclone_x` for functions that raise `SessionIdCollisionError`.

    Returns the qualified names and the module count, so a glob that stops matching
    cannot make the assertion below pass by finding nothing (§6.9 case 2).
    """
    import ast

    src_root = Path(__file__).resolve().parents[2] / "src" / "uclone_x"
    found: set[str] = set()
    scanned = 0
    for path in sorted(src_root.rglob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = str(path.relative_to(src_root).with_suffix("")).replace("/", ".")
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Raise) or inner.exc is None:
                    continue
                exc = inner.exc.func if isinstance(inner.exc, ast.Call) else inner.exc
                if isinstance(exc, ast.Name) and exc.id == "SessionIdCollisionError":
                    found.add(f"{module}.{node.name}")
    return found, scanned


def test_the_record_identity_check_exists_in_exactly_one_function() -> None:
    """Re-inlining the ownership rule anywhere in `src/` fails here.

    No behavioural test can do this: a second copy that happens to be correct today
    passes every test in this file, and is discovered only when one copy is fixed and the
    other is not.
    """
    found, scanned = _functions_raising_session_id_collision()

    assert scanned > 50, f"sweep only reached {scanned} modules; the glob is wrong"
    unexpected = found - _IDENTITY_ALLOWLIST
    assert not unexpected, (
        "a record-ownership decision is made in a function outside the allow-list: "
        f"{sorted(unexpected)}. The rule has exactly one implementation, "
        "`agent.session.verify_record_identity`; call it rather than restating it. If a "
        "new door genuinely needs its own decision, add it to _IDENTITY_ALLOWLIST with "
        "the reason it cannot delegate."
    )
    missing = _IDENTITY_ALLOWLIST - found
    assert not missing, (
        f"allow-listed identity guards have disappeared: {sorted(missing)}. An allow-list "
        "naming functions that no longer exist stops constraining anything."
    )


def test_every_id_addressed_record_door_delegates_to_the_identity_check() -> None:
    """The complementary half: the sweep above also passes if no door checks at all.

    Enumerated from the id-bearing surface rather than from #256's table, which is how
    `SessionStore.delete` and the UI transcript doors were found in the first place. A
    door reaches the rule either by calling `verify_record_identity` or by routing through
    a door that does — the second column records which, so a reader can tell a delegation
    from an omission.
    """
    import ast

    src_root = Path(__file__).resolve().parents[2] / "src" / "uclone_x"
    #: door -> the callee through which it reaches `verify_record_identity`.
    expected: dict[tuple[str, str], str] = {
        ("agent/session.py", "load"): "verify_record_identity",
        ("agent/session.py", "save"): "load",
        ("agent/session.py", "delete"): "load",
        ("ui/app.py", "load_session_record"): "verify_record_identity",
        ("ui/app.py", "clear_session_history"): "load_session_record",
    }
    checked = 0
    for (relpath, func_name), callee in expected.items():
        tree = ast.parse((src_root / relpath).read_text(encoding="utf-8"))
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == func_name
        ]
        assert len(matches) == 1, f"{relpath}::{func_name} found {len(matches)} times, expected 1"
        names = {
            inner.func.id if isinstance(inner.func, ast.Name) else inner.func.attr
            for inner in ast.walk(matches[0])
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name | ast.Attribute)
        }
        assert callee in names, (
            f"{relpath}::{func_name} no longer reaches the record-ownership rule via "
            f"{callee!r}; a variant session id can read or destroy another session's "
            f"record through it (#256)"
        )
        checked += 1

    assert checked == len(expected), f"only {checked} doors were examined"


# --------------------------------------------------------------------------------------
# #263 review (a reviewer): enumeration must agree with loading.
#
# The first version of `list_session_ids` read `p.stem`, which made it **inverted** on a
# tree where a filename and its record disagree: it reported the one id `load` refuses and
# hid the one id `load` accepts. Behaviour was already correct; the enumeration was not.
# Fixed by reading the id out of each record, which is the rule every other door uses.
# --------------------------------------------------------------------------------------


def _assert_enumeration_agrees_with_loading(store: SessionStore, probes: Sequence[str]) -> None:
    """Assert `list_session_ids()` and `load()` agree, in both directions.

    Both directions, because either alone is satisfiable by a broken implementation: one
    that returns `()` satisfies "every listed id loads", and one that returns every
    conceivable string satisfies "every loadable id is listed".
    """
    listed = store.list_session_ids()
    for sid in listed:
        assert store.load(sid) is not None, f"listed {sid!r} but load returned no record"
    for probe in probes:
        try:
            loadable = store.load(probe) is not None
        except SessionIdCollisionError:
            loadable = False
        assert loadable == (probe in listed), (
            f"{probe!r}: loadable={loadable} but listed={probe in listed}"
        )


def test_list_session_ids_is_not_inverted_when_a_filename_and_its_record_disagree(
    tmp_path: Path,
) -> None:
    """The #263 finding, in the three lines the reviewer measured it in.

    A variant pair created on ext4 and carried onto APFS collapses to one file, and the
    survivor can perfectly well be named `SessA.json` while holding **`sessa`**'s record.
    Reading `p.stem` then reported `("SessA",)` — the id `load` *refuses* — while hiding
    `sessa`, the id `load` *accepts*. Precisely inverted, and reachable on plain APFS
    through any rename, restore, or archive extraction.

    **Not filesystem-independent, and an earlier version of this docstring claimed it
    was.** That claim was wrong in the same way the defect it pins is wrong — a statement
    about identity that a folding filesystem hides — and it made this test **fail** on a
    case-sensitive volume: a reviewer measured
    `AssertionError: assert () == ('sessa',)` there while its two neighbours skipped
    correctly. It blocks rather than annoys, because P8 makes verification local-only with
    no CI, so a contributor on Linux would meet a red gate on the very test that proves
    the review finding.

    `()` is the **correct** answer on ext4: the rename makes the record genuinely
    unreachable, because `sessa.json` and `SessA.json` are two different files there. The
    scenario this test describes — *one* file reachable under two spellings — cannot exist
    on a case-sensitive volume at all.

    So the split below is deliberate rather than a bare skip:

    * The **agreement property is asserted unguarded**, because it holds on both volumes
      and is the thing worth protecting everywhere. A bare `pytest.skip` at the top would
      silence enumeration entirely on ext4 — and would disarm the `samefile`-substitution
      mutation, which this test is the only one to catch.
    * Only the **inversion assertion** is guarded, since only it needs the fold to exist.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    store.save(SessionState(session_id="sessa", agent_id="agent-a", turn_counter=4))
    (store.storage_dir / "sessa.json").rename(store.storage_dir / "SessA.json")

    # Unguarded: true on every filesystem, and the property the fix actually promises.
    _assert_enumeration_agrees_with_loading(store, ("sessa", "SessA", "SESSA"))

    # Measured on a scratch subdirectory rather than on the store's own, so the probe
    # cannot be mistaken for a record by the enumeration under test.
    probe_dir = tmp_path / "fold_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    if not _filesystem_folds_case(probe_dir):
        # ext4 and friends: the record is genuinely unreachable, so reporting nothing is
        # right. Asserted rather than skipped, so this branch still says something.
        assert store.list_session_ids() == ()
        assert store.load("sessa") is None
        return

    # The fold exists, so exactly one file is reachable under two spellings — and the id
    # that loads is the one reported, while the stem is not.
    assert store.list_session_ids() == ("sessa",)
    loaded = store.load("sessa")
    assert loaded is not None and loaded.turn_counter == 4
    with pytest.raises(SessionIdCollisionError):
        store.load("SessA")

    # The files are still there: "skipped" must mean skipped, never deleted.
    assert sorted(p.name for p in store.storage_dir.glob("*.json")) == ["SessA.json"]


def test_list_session_ids_skips_a_record_no_load_can_reach(tmp_path: Path) -> None:
    """The bound, stated as narrowly as it holds rather than as perfect agreement.

    A record whose id resolves to a *different* file is unreachable through `load`, so
    listing it would re-create the same disagreement in the other direction. It is
    skipped. Needs a case-sensitive volume to construct, since on APFS `held_elsewhere`
    and its stem would be the same file — so it is built by planting two files whose
    stems differ from the id by more than case.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    _write_raw_record(store.storage_dir / "not_the_id.json", session_id="held_elsewhere")

    # `load("held_elsewhere")` resolves to `held_elsewhere.json`, which does not exist.
    assert store.load("held_elsewhere") is None
    # So it is not listed -- an enumeration naming it would disagree with `load`.
    assert store.list_session_ids() == ()
    _assert_enumeration_agrees_with_loading(store, ("held_elsewhere", "not_the_id"))


def test_list_session_ids_skips_records_load_reports_absent(tmp_path: Path) -> None:
    """Unparseable and unreadable records are skipped, for the same agreement reason.

    `load` reports them absent, so an enumeration that listed them would claim sessions
    exist that cannot be opened. Reading `p.stem` did exactly that.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    store.save(SessionState(session_id="sess_good", agent_id="agent-a"))
    (store.storage_dir / "sess_corrupt.json").write_text("{not json", encoding="utf-8")
    (store.storage_dir / "sess_wrong_shape.json").write_text('{"a": 1}', encoding="utf-8")

    assert store.list_session_ids() == ("sess_good",)
    _assert_enumeration_agrees_with_loading(
        store, ("sess_good", "sess_corrupt", "sess_wrong_shape")
    )


def test_list_session_ids_does_not_trip_over_an_illegal_id_inside_a_record(
    tmp_path: Path,
) -> None:
    """A hand-edited record can hold an id `validate_session_id` refuses.

    Enumeration must skip it rather than raise: it cannot be loaded under that id at all,
    so it is genuinely unreachable, and a listing that raises takes the whole directory
    down over one bad file.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    store.save(SessionState(session_id="sess_ok", agent_id="agent-a"))
    store.storage_dir.mkdir(parents=True, exist_ok=True)
    # Written raw: `SessionState` itself does not police the id, only the path resolver does.
    (store.storage_dir / "sess_evil.json").write_text(
        json.dumps(
            SessionState(session_id="../escape", agent_id="agent-a").model_dump(mode="json")
        ),
        encoding="utf-8",
    )

    assert store.list_session_ids() == ("sess_ok",)
    with pytest.raises(PathTraversalError):
        store.load("../escape")


def test_list_session_ids_logs_every_record_it_omits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A skipped record must leave a trace, matching what `load` already does.

    `list_session_ids` had **zero** logger calls while `load` warns
    `"…treating as absent"` for the same class of condition two methods above, so a
    record dropped from the listing was undiagnosable. Nothing is dropped from disk on
    this path — which is why this is a diagnosability fix and not a correctness one — so
    the test asserts both halves: a warning per omission, **and** the files still there.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    store.save(SessionState(session_id="sess_good", agent_id="agent-a"))
    (store.storage_dir / "sess_corrupt.json").write_text("{not json", encoding="utf-8")
    (store.storage_dir / "sess_wrong_shape.json").write_text('{"a": 1}', encoding="utf-8")
    # Candidate file does not exist -> `samefile` raises -> the OSError skip path.
    _write_raw_record(store.storage_dir / "not_the_id.json", session_id="held_elsewhere")
    # Candidate file DOES exist but is a different file -> the `not reachable` skip path.
    # A separate case because it is a separate branch, and an earlier version of this test
    # exercised only the one above: a mutation downgrading this branch's warning to
    # `logger.debug` **escaped**, since nothing reached it.
    _write_raw_record(store.storage_dir / "decoy.json", session_id="sess_good")

    with caplog.at_level(logging.WARNING, logger="uclone_x.agent.session"):
        assert store.list_session_ids() == ("sess_good",)

    omitted = [r.getMessage() for r in caplog.records if "omitting from listing" in r.getMessage()]
    assert len(omitted) == 4, f"expected one warning per omitted record, got: {omitted}"
    # Each warning names the file it is about, so the message is actionable rather than
    # merely present -- the whole point of preferring a log over silence.
    for stem in ("sess_corrupt", "sess_wrong_shape", "not_the_id", "decoy"):
        assert any(stem in m for m in omitted), f"no warning names {stem}: {omitted}"

    # And the omission is an omission, not a deletion.
    assert sorted(p.name for p in store.storage_dir.glob("*.json")) == [
        "decoy.json",
        "not_the_id.json",
        "sess_corrupt.json",
        "sess_good.json",
        "sess_wrong_shape.json",
    ]


def test_two_files_claiming_one_id_list_it_once_and_load_the_reachable_one(
    tmp_path: Path,
) -> None:
    """The duplicate-claim case, which `list_session_ids`'s docstring only bounded.

    Two records both naming `sess_good`, one at `sess_good.json` and one at `decoy.json`.
    `load("sess_good")` can only ever open the first, so the second is unreachable and is
    skipped — which keeps the "represented by exactly one id" half of the agreement
    property true rather than merely claimed.

    Filesystem-independent: no fold is involved, both files exist under distinct names on
    every filesystem. This case also reaches a skip branch the logging test missed, which
    is how the `not reachable` warning came to be exercised at all.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    store.save(SessionState(session_id="sess_good", agent_id="agent-a", turn_counter=9))
    _write_raw_record(store.storage_dir / "decoy.json", session_id="sess_good", turn_counter=1)

    assert store.list_session_ids() == ("sess_good",)
    loaded = store.load("sess_good")
    assert loaded is not None
    assert loaded.turn_counter == 9, "load returned the unreachable decoy"
    _assert_enumeration_agrees_with_loading(store, ("sess_good", "decoy"))
    # Both files survive: unreachable is not a licence to delete.
    assert sorted(p.name for p in store.storage_dir.glob("*.json")) == [
        "decoy.json",
        "sess_good.json",
    ]


def test_a_hardlinked_alias_neither_duplicates_nor_loses_a_session(tmp_path: Path) -> None:
    """One inode under two names must enumerate once, not twice and not zero times.

    Reachable through `cp -l`, some backup tools, and some archive extractions. It is the
    other case where `samefile` and `resolve() == resolve()` disagree — measured on both
    an APFS and a case-sensitive volume: `samefile=True`, `resolve()==resolve()=False`.

    **It does not arm the `samefile`-substitution mutation, and that is recorded rather
    than papered over.** Whenever `samefile(candidate, path)` is true under distinct
    names, the candidate exists and is itself enumerated under a matching name, so the id
    enters the set either way and the *output* is identical. The substitution is therefore
    observable **only** on a folding filesystem — which is a property of the code, not a
    gap in these tests, and is why the guarded assertion in the inversion test above is
    the only thing that can catch it.
    """
    store = SessionStore(storage_dir=tmp_path / "core")
    store.save(SessionState(session_id="sess_real", agent_id="agent-a", turn_counter=5))
    os.link(store.storage_dir / "sess_real.json", store.storage_dir / "sess_alias.json")

    assert store.list_session_ids() == ("sess_real",)
    loaded = store.load("sess_real")
    assert loaded is not None and loaded.turn_counter == 5
    # `sess_alias` is a name, not a session — and the refusal here is the *right* answer,
    # which my first draft of this test got wrong by asserting `None`. The alias file
    # genuinely exists (that is what a hardlink is) and the record inside it names
    # `sess_real`, so this is the ordinary identity refusal and not an absence. Worth
    # keeping as an assertion because it shows the rule is about record contents, not
    # about folding: no fold is involved here at all.
    with pytest.raises(SessionIdCollisionError) as excinfo:
        store.load("sess_alias")
    assert excinfo.value.record_session_id == "sess_real"
    # No fold explains this collision, so the message must not invent one (P6).
    assert "differs only" not in str(excinfo.value)
    _assert_enumeration_agrees_with_loading(store, ("sess_real", "sess_alias"))
