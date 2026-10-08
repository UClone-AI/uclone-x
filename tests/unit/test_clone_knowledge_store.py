"""A clone's facts live in its knowledge file (uGraph step 3, #2086).

What these pin: the one-time import of a `memory.json`, the scope a clone's file belongs
to, history kept when a fact is replaced, and a damaged file set aside, never recreated
over.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.knowledge.sqlite_store import ScopeMismatchError, SqliteKnowledgeStore
from uclone_x.memory.graph import fact_of
from uclone_x.memory.legacy_import import IMPORTED_META_KEY
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.store import CrossSessionMemory, set_aside_database


def _provenance() -> Provenance:
    return Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="memory.test", model="test-model"),
        served_by=ServiceRef(provider="memory.test", model="test-model"),
    )


def _old_fact(
    fact_id: str,
    value: str,
    created_at: str = "2026-01-01T00:00:00Z",
    *,
    retracted: bool = False,
) -> MemoryFact:
    return MemoryFact(
        fact_id=fact_id,
        subject="user",
        predicate="prefers",
        object_value=value,
        provenance=_provenance(),
        source_session_id="sess-0",
        created_at=created_at,
        updated_at=created_at,
        retracted=retracted,
    )


def _write_legacy(path: Path, facts: list[MemoryFact]) -> None:
    path.write_text(
        json.dumps({"facts": [f.model_dump(mode="json") for f in facts]}), encoding="utf-8"
    )


def test_a_memory_json_is_imported_once_and_set_aside(tmp_path: Path) -> None:
    """The first start after the upgrade copies the old facts, with their ids; a second adds none.

    A fact id is what the UI's edit and forget routes hold, so it must survive the move.
    The old file is renamed, not deleted, and is not read again.

    Killed by: src/uclone_x/memory/legacy_import.py :: copied = tx.meta(IMPORTED_META_KEY) is None
    Becomes: copied = True
    """
    legacy = tmp_path / "memory.json"
    _write_legacy(legacy, [_old_fact("mem_a", "tabs"), _old_fact("mem_b", "dark mode")])

    first = CrossSessionMemory(storage_path=tmp_path / "k.sqlite3", legacy_path=legacy)
    assert {f.fact_id for f in first.list_facts()} == {"mem_a", "mem_b"}
    assert not legacy.exists()
    assert len(list(tmp_path.glob("memory.json.imported-*"))) == 1

    # A memory.json that turns up later is not read: it is set aside, and nothing is added.
    _write_legacy(legacy, [_old_fact("mem_c", "spaces")])
    second = CrossSessionMemory(storage_path=tmp_path / "k.sqlite3", legacy_path=legacy)
    assert {f.fact_id for f in second.list_facts()} == {"mem_a", "mem_b"}
    assert not legacy.exists()
    assert len(list(tmp_path.glob("memory.json.imported-*"))) == 2
    assert second.knowledge.meta(IMPORTED_META_KEY) == "2"


def test_an_unreadable_memory_json_is_set_aside_and_reported(tmp_path: Path) -> None:
    """A legacy file that cannot be parsed is moved, not read as empty, and `load_failure` says so.

    Killed by: src/uclone_x/memory/store.py :: moved: str | None = str(set_aside_unreadable(legacy_path))
    Becomes: moved: str | None = None
    """
    legacy = tmp_path / "memory.json"
    legacy.write_text("{not json", encoding="utf-8")

    memory = CrossSessionMemory(storage_path=tmp_path / "k.sqlite3", legacy_path=legacy)

    assert memory.list_facts() == []
    assert memory.load_failure is not None
    assert not legacy.exists()
    kept = list(tmp_path.glob("memory.json.unreadable-*"))
    assert [p.read_text(encoding="utf-8") for p in kept] == ["{not json"]
    assert memory.knowledge.meta(IMPORTED_META_KEY) is None


def test_a_clone_file_refuses_another_scope(tmp_path: Path) -> None:
    """Two agents' facts never mix: a file made for one clone is not opened for another.

    Killed by: src/uclone_x/knowledge/sqlite_store.py :: if meta.get("scope") != self._scope:
    Becomes: if False:
    """
    path = tmp_path / "k.sqlite3"
    CrossSessionMemory(storage_path=path, agent_id="agt_one").record_fact(
        subject="user",
        predicate="likes",
        object_value="tea",
        provenance=_provenance(),
        source_session_id="s",
    )

    with pytest.raises(ScopeMismatchError):
        SqliteKnowledgeStore(path, "agt_two")
    assert [
        f.object_value
        for f in CrossSessionMemory(storage_path=path, agent_id="agt_one").list_facts()
    ] == ["tea"]


def test_a_replaced_fact_stays_as_history(tmp_path: Path) -> None:
    """A new value for the same subject and predicate ends the old edge; it does not delete it.

    The old edge keeps its record and gets `expired_at` equal to the new edge's `recorded_at`
    (`approved`, not `retracted`); the new fact names it in `contradicts_fact_id`.

    Killed by: src/uclone_x/memory/graph.py :: expired_at=at,
    Becomes: expired_at=None,
    """
    memory = CrossSessionMemory(storage_path=tmp_path / "k.sqlite3")
    old = memory.record_fact(
        subject="user",
        predicate="lives_in",
        object_value="Seoul",
        provenance=_provenance(),
        source_session_id="s",
    )
    new = memory.record_fact(
        subject="user",
        predicate="lives_in",
        object_value="Busan",
        provenance=_provenance(),
        source_session_id="s",
    )

    assert [f.object_value for f in memory.list_facts()] == ["Busan"]
    rows = {row.edge.id: row for row in memory.knowledge.rows()}
    assert rows[old.fact_id].edge.status == "approved"
    assert rows[old.fact_id].edge.expired_at is not None
    assert rows[old.fact_id].edge.expired_at == rows[new.fact_id].edge.recorded_at
    assert new.contradicts_fact_id == old.fact_id
    history = memory.get_fact(old.fact_id)
    assert history is not None
    assert history.retracted
    assert history.object_value == "Seoul"


def test_a_stated_end_fills_the_interval_and_blocks_supersession(tmp_path: Path) -> None:
    """ "Until last year" is `valid.end`; such a fact is history and replaces nothing.

    Killed by: src/uclone_x/memory/graph.py :: end=fact.valid_until
    Becomes: end=None
    """
    memory = CrossSessionMemory(storage_path=tmp_path / "k.sqlite3")
    current = memory.record_fact(
        subject="user",
        predicate="lives_in",
        object_value="Busan",
        provenance=_provenance(),
        source_session_id="s",
    )
    past = memory.record_fact(
        subject="user",
        predicate="lives_in",
        object_value="Seoul",
        provenance=_provenance(),
        source_session_id="s",
        valid_until="2025-06-30T00:00:00Z",
    )

    rows = {row.edge.id: row for row in memory.knowledge.rows()}
    assert rows[past.fact_id].edge.valid.end == "2025-06-30T00:00:00Z"
    held = memory.get_fact(current.fact_id)
    assert held is not None
    assert not held.retracted
    assert fact_of(rows[past.fact_id]).valid_until == "2025-06-30T00:00:00Z"
    # Not one of the statements that hold now, which the reasoner works from.
    assert current.fact_id in memory.statements_now()
    assert past.fact_id not in memory.statements_now()


def test_a_corrupt_knowledge_file_is_set_aside_and_a_new_one_made(tmp_path: Path) -> None:
    """Facts that cannot be read are kept under another name and the gap is reported.

    Killed by: src/uclone_x/memory/store.py :: moved = set_aside_database(self._storage_path)
    Becomes: moved = None
    """
    path = tmp_path / "k.sqlite3"
    damaged = b"this is not a sqlite database" * 100
    path.write_bytes(damaged)

    memory = CrossSessionMemory(storage_path=path)

    assert memory.load_failure is not None
    assert "moved to" in memory.load_failure
    assert memory.list_facts() == []
    kept = list(tmp_path.glob("k.sqlite3.unreadable-*"))
    assert [p.read_bytes() for p in kept] == [damaged]
    memory.record_fact(
        subject="user",
        predicate="likes",
        object_value="tea",
        provenance=_provenance(),
        source_session_id="s",
    )
    assert [f.object_value for f in CrossSessionMemory(storage_path=path).list_facts()] == ["tea"]


def test_the_write_ahead_files_go_with_aset_aside_database(tmp_path: Path) -> None:
    """A `-wal` left beside the new file would be replayed into it, so it moves too.

    Killed by: src/uclone_x/memory/store.py :: for suffix in ("-wal", "-shm"):
    Becomes: for suffix in ():
    """
    path = tmp_path / "k.sqlite3"
    path.write_bytes(b"main")
    (tmp_path / "k.sqlite3-wal").write_bytes(b"log")
    (tmp_path / "k.sqlite3-shm").write_bytes(b"shared")

    assert set_aside_database(path) is not None

    assert sorted(p.name.split(".unreadable-")[0] for p in tmp_path.iterdir()) == [
        "k.sqlite3",
        "k.sqlite3-shm",
        "k.sqlite3-wal",
    ]
    assert not path.exists()
    assert not (tmp_path / "k.sqlite3-wal").exists()


def test_a_lock_timeout_during_legacy_import_does_not_set_memory_json_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked sqlite store propagates as refusal; memory.json is kept to retry (#2097).

    Killed by: src/uclone_x/memory/store.py :: except sqlite3.OperationalError:
    Becomes: except ():
    """
    import sqlite3

    legacy = tmp_path / "memory.json"
    _write_legacy(legacy, [_old_fact("mem_a", "tabs")])

    def _locked(*args: object, **kwargs: object) -> bool:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("uclone_x.memory.store.import_legacy_memory", _locked)

    with pytest.raises(sqlite3.OperationalError):
        CrossSessionMemory(storage_path=tmp_path / "k.sqlite3", legacy_path=legacy)

    assert legacy.exists()
    assert list(tmp_path.glob("memory.json.unreadable-*")) == []
    assert list(tmp_path.glob("memory.json.imported-*")) == []


def test_an_imported_retracted_fact_stays_retracted_and_is_not_listed_or_recalled(
    tmp_path: Path,
) -> None:
    """A retracted fact in legacy memory.json stays retracted after import (#2097).

    Killed by: src/uclone_x/memory/graph.py :: status="retracted" if fact.retracted else tx.policy.initial_status,
    Becomes: status=tx.policy.initial_status,
    """
    legacy = tmp_path / "memory.json"
    active_fact = _old_fact("mem_active", "tabs")
    retracted_fact = _old_fact("mem_retracted", "spaces", retracted=True)

    _write_legacy(legacy, [active_fact, retracted_fact])

    memory = CrossSessionMemory(storage_path=tmp_path / "k.sqlite3", legacy_path=legacy)

    # Listed facts only include active facts
    assert [f.fact_id for f in memory.list_facts()] == ["mem_active"]

    # History can still get the retracted fact, and it is marked retracted
    retracted_got = memory.get_fact("mem_retracted")
    assert retracted_got is not None
    assert retracted_got.retracted

    # In the sqlite store edge rows, status is 'retracted'
    rows = {row.edge.id: row for row in memory.knowledge.rows()}
    assert rows["mem_retracted"].edge.status == "retracted"
    assert rows["mem_active"].edge.status == "approved"
