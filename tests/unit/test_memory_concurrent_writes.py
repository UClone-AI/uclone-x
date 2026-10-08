"""One agent, several processes: a write keeps what the others recorded (#1125, uGraph step 3).

The clone's facts are edges in one sqlite file, and one write is one transaction. These
tests pin what that gives that the whole-document merge did not: a second writer waits for
the first and then reads what it committed, so nothing is lost at any interleaving.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.memory.store import CrossSessionMemory


def _provenance() -> Provenance:
    return Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="memory.test", model="test-model"),
        served_by=ServiceRef(provider="memory.test", model="test-model"),
    )


def _record(memory: CrossSessionMemory, value: str, subject: str = "user") -> str:
    fact = memory.record_fact(
        subject=subject,
        predicate="prefers",
        object_value=value,
        provenance=_provenance(),
        source_session_id="sess-1",
        auto_retract_conflicts=False,
    )
    return fact.fact_id


def _values(path: Path) -> set[str]:
    """What a process that opens the file fresh reads: the truth on disk."""
    return {fact.object_value for fact in CrossSessionMemory(storage_path=path).list_facts()}


def test_two_writers_alternating_on_one_file_lose_nothing(tmp_path: Path) -> None:
    """The dashboard and `ucx run` are two holders of one agent's memory.

    Each opened the file before the other wrote. Under the whole-document store, every
    fact the other recorded in between was erased by the later replace, silently, which is
    the substitution P6 forbids. Here each write is its own transaction over the file as it
    is then, so the three facts all remain and each writer sees the others'.
    """
    path = tmp_path / "knowledge.sqlite3"

    dashboard = CrossSessionMemory(storage_path=path)
    cli = CrossSessionMemory(storage_path=path)
    _record(dashboard, "tabs")
    _record(cli, "spaces")
    _record(dashboard, "dark mode")

    assert _values(path) == {"tabs", "spaces", "dark mode"}
    assert {f.object_value for f in cli.list_facts()} == {"tabs", "spaces", "dark mode"}
    assert {f.object_value for f in dashboard.list_facts()} == {"tabs", "spaces", "dark mode"}


_CHILD = """
import sys
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.memory.store import CrossSessionMemory

path, tag, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
service = ServiceRef(provider="memory.test", model="test-model")
provenance = Provenance(path=ExecutionPath.PRIMARY, requested=service, served_by=service)
memory = CrossSessionMemory(storage_path=path)
for index in range(count):
    memory.record_fact(
        subject=f"{tag}-{index}",
        predicate="prefers",
        object_value="x",
        provenance=provenance,
        source_session_id="sess-1",
    )
"""


def test_two_processes_writing_at_once_lose_nothing(tmp_path: Path) -> None:
    """Two real processes, each writing twenty facts as fast as it can, against one file.

    A writer that finds the file locked waits for the other's transaction instead of
    failing or overwriting it, so all forty facts are there afterwards.

    Killed by: src/uclone_x/knowledge/sqlite_store.py :: BUSY_TIMEOUT_SECONDS = 30.0
    Becomes: BUSY_TIMEOUT_SECONDS = 0.0
    """
    path = tmp_path / "knowledge.sqlite3"
    CrossSessionMemory(storage_path=path)  # the file exists before either child starts
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(path), tag, "20"],
            env=env,
            stderr=subprocess.PIPE,
            text=True,
        )
        for tag in ("alpha", "beta")
    ]
    failures = [child.communicate(timeout=120)[1] for child in children]

    assert [child.returncode for child in children] == [0, 0], failures
    subjects = {fact.subject for fact in CrossSessionMemory(storage_path=path).list_facts()}
    assert subjects == {f"{tag}-{i}" for tag in ("alpha", "beta") for i in range(20)}


def test_a_writer_waits_for_an_open_transaction_and_then_sees_what_it_committed(
    tmp_path: Path,
) -> None:
    """The write lock is real: a second writer blocks, rather than writing past the first.

    The first writer holds its transaction open. The second's `record_fact` cannot finish
    while it does, and when the first commits, the second has read it: the fact the first
    wrote is there to conflict with. This is what the whole-document merge could only
    check for; it could not hold anyone back.

    Killed by: src/uclone_x/knowledge/sqlite_store.py :: conn.execute("BEGIN IMMEDIATE")  # the one write lock, held to COMMIT
    Becomes: conn.execute("BEGIN")  # the one write lock, held to COMMIT
    """
    path = tmp_path / "knowledge.sqlite3"
    first = CrossSessionMemory(storage_path=path)
    second = CrossSessionMemory(storage_path=path)
    finished = threading.Event()

    def second_writes() -> None:
        second.record_fact(
            subject="user",
            predicate="editor",
            object_value="vim",
            provenance=_provenance(),
            source_session_id="sess-2",
        )
        finished.set()

    writer = threading.Thread(target=second_writes)
    with first.knowledge.transaction() as tx:
        writer.start()
        assert not finished.wait(0.3), "the second writer got past an open transaction"
        from uclone_x.memory.graph import write_fact
        from uclone_x.memory.models import MemoryFact

        write_fact(
            tx,
            MemoryFact(
                subject="user",
                predicate="editor",
                object_value="emacs",
                provenance=_provenance(),
                source_session_id="sess-1",
            ),
        )
    writer.join(timeout=30)

    assert finished.is_set()
    facts = {f.object_value: f for f in first.list_facts(include_retracted=True)}
    assert facts["emacs"].retracted, "the second writer read the first's commit and superseded it"
    assert not facts["vim"].retracted
    assert facts["vim"].contradicts_fact_id == facts["emacs"].fact_id


def test_a_retraction_by_another_process_is_not_undone(tmp_path: Path) -> None:
    """Both hold the fact; only one knows it was withdrawn.

    Nothing is cached between calls, so the writer that did not retract it reads the
    withdrawal when it next acts, and cannot reinforce or correct a fact the person asked to
    be forgotten.

    Killed by: src/uclone_x/memory/store.py :: raise ValueError(f"Memory fact '{fact_id}' is retracted and cannot be reinforced")
    Becomes: pass
    """
    path = tmp_path / "knowledge.sqlite3"
    first = CrossSessionMemory(storage_path=path)
    fact_id = _record(first, "tabs")
    second = CrossSessionMemory(storage_path=path)

    second.retract_fact(
        fact_id, reason="the user asked for it to be forgotten", provenance=_provenance()
    )

    with pytest.raises(ValueError, match="retracted"):
        first.reinforce_fact(fact_id, "turn-1")
    with pytest.raises(ValueError, match="retracted"):
        first.correct_fact(fact_id, "spaces", _provenance())
    seen = first.get_fact(fact_id)
    assert seen is not None
    assert seen.retracted
    assert seen.retraction_reason == "the user asked for it to be forgotten"
    assert first.list_facts() == []


def test_a_file_damaged_while_open_fails_loudly_and_is_left_alone(tmp_path: Path) -> None:
    """A write never recreates a file it can no longer read.

    The store holds no connection between calls, so damage done under a running process
    shows on its next call, as the error SQLite raises. The bytes are not touched: they are
    the only copy, and the next process to start sets them aside (`load_failure`).
    """
    path = tmp_path / "knowledge.sqlite3"
    memory = CrossSessionMemory(storage_path=path)
    _record(memory, "tabs")
    for sidecar in tmp_path.glob("knowledge.sqlite3-*"):
        sidecar.unlink()
    damaged = b"this is not a sqlite database" * 100
    path.write_bytes(damaged)

    with pytest.raises(sqlite3.DatabaseError):
        _record(memory, "spaces")

    assert path.read_bytes() == damaged
