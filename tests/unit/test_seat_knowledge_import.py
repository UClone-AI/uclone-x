"""A retired seat knowledge record becomes its clone's facts, once (clone-knowledge-graph step 6).

Until step 6 each room seat saved its own rules engine to `knowledge/<session_id>.yaml`
(#1367). Step 6 retired that engine; `room/seat_knowledge_import.py` moves what a record
holds to where the clone now keeps what it learned (§3.7). What these pin:

* each relation is saved as a fact of the seat's clone, origin `saved`, with the record's
  session and room, and the record is renamed `.imported-`, never deleted;
* an imported fact is added beside a fact it conflicts with, never over it;
* a record already imported (or partly) saves nothing twice;
* an unreadable record is renamed `.unreadable-` and nothing is saved;
* a record named by no seat's session, or whose clone's memory will not open or read, keeps
  its name;
* the desktop head runs the import when it builds the room stack;
* a fact the person forgot or corrected is never brought back (#1872);
* two imports of one record at once leave each fact once, and the second says it found the
  record taken rather than that a rename failed (#1872);
* a failure is logged by its type and place, never by the record's text (#1872);
* a knowledge directory that cannot be listed is logged, and nothing raises (#1872);
* while the clone's memory has a set-aside copy the import has not waited on, a record with
  relations keeps its name for one start, since a fact forgotten inside the copy cannot be
  counted as held (#1879); the copy is noted in the pending record, and the next start
  goes ahead (#1892).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from uclone_x.core.provenance import Provenance
from uclone_x.core.set_aside import set_aside
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import OntologyRelation
from uclone_x.room.seat_knowledge_import import (
    IMPORTED_MARKER,
    SEAT_KNOWLEDGE_SUBDIR,
    WAITED_RECORD,
    SeatKnowledgeImport,
    import_seat_knowledge,
    imported_fact_id,
)

ROOM = "room_a"
CLONE = "scout"
SESSION = f"sess_room__{ROOM}__{CLONE}"


def _record(directory: Path, session_id: str, *relations: tuple[str, str, str]) -> Path:
    engine = OntologyEngine()
    for source, predicate, target in relations:
        engine.register_relation(
            OntologyRelation(source_entity=source, predicate=predicate, target_entity=target)
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.yaml"
    engine.save_to_yaml(path)
    return path


POSTGRES = ("Postgres", "is_a", "Database")


class _Memories:
    """One store per clone id under `root`, as the app's memory map hands them out."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.stores: dict[str, CrossSessionMemory] = {}

    def __call__(self, clone_id: str) -> CrossSessionMemory:
        if clone_id not in self.stores:
            self.stores[clone_id] = CrossSessionMemory(self.root / clone_id / "memory.json")
        return self.stores[clone_id]


def test_each_relation_becomes_a_saved_fact_of_the_seats_clone_and_the_file_is_kept(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: kept = set_aside(path, IMPORTED_MARKER)
    Becomes: kept = path
    """
    knowledge = tmp_path / "knowledge"
    path = _record(knowledge, SESSION, POSTGRES)
    memories = _Memories(tmp_path / "agents")

    result = import_seat_knowledge(knowledge, memories)

    assert result == SeatKnowledgeImport(imported=1, facts=1)
    (fact,) = memories(CLONE).list_facts()
    assert (fact.subject, fact.predicate, fact.object_value) == POSTGRES
    assert fact.origin == "saved"
    assert fact.source_session_id == SESSION
    assert fact.source_room_id == ROOM
    assert not path.exists()
    (kept,) = knowledge.iterdir()
    assert kept.name.startswith(f"{path.name}{IMPORTED_MARKER}")


def test_an_imported_fact_never_retracts_what_the_clone_already_holds(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: auto_retract_conflicts=False,
    Becomes: auto_retract_conflicts=True,
    """
    knowledge = tmp_path / "knowledge"
    _record(knowledge, SESSION, ("Busan", "is_in", "Korea"))
    memories = _Memories(tmp_path / "agents")
    memories(CLONE).record_fact(
        subject="Busan",
        predicate="is_in",
        object_value="Japan",
        provenance=Provenance.primary(provider="person"),
        source_session_id="sess_earlier",
    )

    import_seat_knowledge(knowledge, memories)

    assert {f.object_value for f in memories(CLONE).list_facts()} == {"Japan", "Korea"}


def test_a_second_pass_over_the_same_relations_saves_nothing_twice(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: if key in held or not all(key):
    Becomes: if not all(key):
    """
    knowledge = tmp_path / "knowledge"
    memories = _Memories(tmp_path / "agents")
    _record(knowledge, SESSION, POSTGRES)
    import_seat_knowledge(knowledge, memories)
    # The same relation again, as a record whose rename failed is read on the next start.
    _record(knowledge, SESSION, POSTGRES, ("wheel", "part_of", "car"))

    result = import_seat_knowledge(knowledge, memories)

    assert result == SeatKnowledgeImport(imported=1, facts=1)
    assert len(memories(CLONE).list_facts()) == 2


def test_an_unreadable_record_is_set_aside_and_nothing_is_saved(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: aside = set_aside_unreadable(path)
    Becomes: aside = path
    """
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    path = knowledge / f"{SESSION}.yaml"
    path.write_text("relations: [unterminated\n", encoding="utf-8")
    memories = _Memories(tmp_path / "agents")

    result = import_seat_knowledge(knowledge, memories)

    assert result == SeatKnowledgeImport(unreadable=1)
    assert memories.stores == {}
    (kept,) = knowledge.iterdir()
    assert kept.name.startswith(f"{path.name}.unreadable-")


def test_a_record_named_by_no_seats_session_is_left_where_it_is(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: if len(parts) != 3 or parts[0] != SESSION_ID_PREFIX or not parts[1] or not parts[2]:
    Becomes: if False:
    """
    knowledge = tmp_path / "knowledge"
    path = _record(knowledge, "sess_one_to_one", POSTGRES)
    memories = _Memories(tmp_path / "agents")

    assert import_seat_knowledge(knowledge, memories) == SeatKnowledgeImport(left=1)
    assert path.exists()
    assert memories.stores == {}


def test_a_record_whose_clone_memory_will_not_open_keeps_its_name(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: except Exception as exc:  # an id with no home, or a store that will not open
    Becomes: except OSError as exc:  # an id with no home, or a store that will not open
    """
    knowledge = tmp_path / "knowledge"
    path = _record(knowledge, SESSION, POSTGRES)

    def no_home(clone_id: str) -> CrossSessionMemory:
        raise LookupError(clone_id)

    assert import_seat_knowledge(knowledge, no_home) == SeatKnowledgeImport(left=1)
    assert path.exists()


def test_a_record_whose_clone_memory_cannot_be_read_keeps_its_name(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: if memory.load_failure is not None:
    Becomes: if memory.load_failure is None:
    """
    knowledge = tmp_path / "knowledge"
    path = _record(knowledge, SESSION, POSTGRES)
    store_file = tmp_path / "agents" / CLONE / "memory.json"
    store_file.parent.mkdir(parents=True)
    store_file.write_text("{not json", encoding="utf-8")

    memories = _Memories(tmp_path / "agents")
    result = import_seat_knowledge(knowledge, memories)

    assert result == SeatKnowledgeImport(left=1)
    assert path.exists()
    assert memories(CLONE).list_facts() == []


def _set_aside_memory(tmp_path: Path, knowledge: Path) -> Path:
    """An unreadable `memory.json`, set aside by the first start's open; the copy's path."""
    store_file = tmp_path / "agents" / CLONE / "memory.json"
    store_file.parent.mkdir(parents=True, exist_ok=True)
    store_file.write_text("{not json", encoding="utf-8")
    # The first open sets the unreadable file aside; this pass keeps every record.
    first = import_seat_knowledge(knowledge, _Memories(tmp_path / "agents"))
    assert (first.imported, first.left) == (0, len(list(knowledge.glob("*.yaml"))))
    return max(store_file.parent.glob("memory.json.unreadable-*"))


def test_a_record_waits_one_start_while_the_clones_memory_has_a_new_set_aside_copy(
    tmp_path: Path,
) -> None:
    """The person forgot the relation inside a memory file that was later set aside unread.

    The next start opens a fresh `memory.json` with no record of the forgetting, so an
    import then could bring the fact back. It waits one start instead, keeping the record
    and noting the copy in the pending record; a start after that goes ahead (#1879,
    #1892). Before #1892 the wait never ended: retention keeps the newest copies and this
    build cannot read them.

    Killed by: src/uclone_x/room/seat_knowledge_import.py :: if unseen:
    Becomes: if False:
    Killed by: src/uclone_x/room/seat_knowledge_import.py :: unseen = copies - waited.get(clone_id, frozenset())
    Becomes: unseen = copies - frozenset()
    Killed by: src/uclone_x/room/seat_knowledge_import.py :: newly_waited.setdefault(clone_id, set()).update(copies)
    Becomes: newly_waited.setdefault(clone_id, set()).update(())
    Killed by: src/uclone_x/room/seat_knowledge_import.py :: if newly_waited:
    Becomes: if False:
    """
    knowledge = tmp_path / "knowledge"
    path = _record(knowledge, SESSION, POSTGRES)
    copy = _set_aside_memory(tmp_path, knowledge)

    # The next start: a fresh store, readable and empty, beside a copy not waited on yet.
    memories = _Memories(tmp_path / "agents")
    assert import_seat_knowledge(knowledge, memories) == SeatKnowledgeImport(waiting=1)
    assert path.exists()
    assert memories(CLONE).list_facts() == []

    # The start after: the copy was already set aside when the pending record was written.
    after = import_seat_knowledge(knowledge, _Memories(tmp_path / "agents"))
    assert after == SeatKnowledgeImport(imported=1, facts=1)
    assert not path.exists()
    assert copy.exists()  # the copy itself is never touched


def test_every_record_of_the_clone_waits_in_the_pass_that_first_sees_the_copy(
    tmp_path: Path,
) -> None:
    """A copy first seen in this pass is not waited on yet for the pass's later records.

    Killed by: src/uclone_x/room/seat_knowledge_import.py :: unseen = copies - waited.get(clone_id, frozenset())
    Becomes: unseen = copies - waited.get(clone_id, frozenset()) - newly_waited.get(clone_id, set())
    """
    knowledge = tmp_path / "knowledge"
    _record(knowledge, SESSION, POSTGRES)
    _record(knowledge, f"sess_room__room_b__{CLONE}", ("Redis", "is_a", "Cache"))
    _set_aside_memory(tmp_path, knowledge)

    first = import_seat_knowledge(knowledge, _Memories(tmp_path / "agents"))

    assert first == SeatKnowledgeImport(waiting=2)


def test_a_copy_set_aside_after_the_wait_waits_once_in_turn(tmp_path: Path) -> None:
    """The pending record names the copies it waited on, so a later copy is new (#1892).

    Killed by: src/uclone_x/room/seat_knowledge_import.py :: unseen = copies - waited.get(clone_id, frozenset())
    Becomes: unseen = copies if clone_id not in waited else set()
    """
    knowledge = tmp_path / "knowledge"
    path = _record(knowledge, SESSION, POSTGRES)
    _set_aside_memory(tmp_path, knowledge)
    assert import_seat_knowledge(knowledge, _Memories(tmp_path / "agents")).waiting == 1

    # Before the next start the memory is set aside again: a second, newer copy.
    _set_aside_memory(tmp_path, knowledge)
    assert import_seat_knowledge(knowledge, _Memories(tmp_path / "agents")).waiting == 1
    assert path.exists()

    after = import_seat_knowledge(knowledge, _Memories(tmp_path / "agents"))
    assert after == SeatKnowledgeImport(imported=1, facts=1)


@pytest.mark.parametrize("body", ["{not json", '["a list"]', '{"clones": {"scout": "one"}}'])
def test_a_pending_record_that_cannot_be_read_waits_again(tmp_path: Path, body: str) -> None:
    """Unreadable or misshapen, the pending record counts as empty: the safe direction.

    Killed by: src/uclone_x/room/seat_knowledge_import.py :: except (OSError, ValueError) as exc:
    Becomes: except OSError as exc:
    Killed by: src/uclone_x/room/seat_knowledge_import.py :: cast(dict[str, Any], raw).get("clones") if isinstance(raw, dict) else None
    Becomes: cast(dict[str, Any], raw).get("clones")
    """
    knowledge = tmp_path / "knowledge"
    _record(knowledge, SESSION, POSTGRES)
    _set_aside_memory(tmp_path, knowledge)
    (knowledge / WAITED_RECORD).write_text(body, encoding="utf-8")

    with_bad_record = import_seat_knowledge(knowledge, _Memories(tmp_path / "agents"))

    assert with_bad_record == SeatKnowledgeImport(waiting=1)
    # Rewritten whole by that pass, it now lets the next start go ahead.
    assert import_seat_knowledge(knowledge, _Memories(tmp_path / "agents")).imported == 1


def test_the_pending_record_is_never_read_as_a_seat_record(tmp_path: Path) -> None:
    """A smoke check with no mutation of its own: the records are found by `*.yaml`."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / WAITED_RECORD).write_text('{"clones": {}}', encoding="utf-8")

    assert import_seat_knowledge(knowledge, _Memories(tmp_path)) == SeatKnowledgeImport()
    assert (knowledge / WAITED_RECORD).exists()


def test_no_knowledge_directory_is_nothing_to_do(tmp_path: Path) -> None:
    """A smoke check with no mutation of its own: `Path.glob` over an absent directory is
    empty too, so dropping the directory check changes nothing this can see."""
    assert import_seat_knowledge(tmp_path / "absent", _Memories(tmp_path)) == (
        SeatKnowledgeImport()
    )


def test_the_desktop_head_imports_when_it_builds_the_room_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/rooms.py :: import_seat_knowledge(
    Becomes: (lambda *_: None)(
    """
    from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
    from uclone_x.llm.connectors.mock import MockLLMConnector
    from uclone_x.ui.app import AgentSessionManager
    from uclone_x.ui.rooms import RoomStack

    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
    mgr = AgentSessionManager(
        storage_dir=tmp_path / "sessions", llm=MockLLMConnector(), workspace_dir=tmp_path / "ws"
    )
    path = _record(mgr.storage_dir / SEAT_KNOWLEDGE_SUBDIR, SESSION, POSTGRES)

    RoomStack(mgr)

    assert not path.exists()
    facts = mgr.memory_for(CLONE).list_facts()
    assert [(f.subject, f.object_value, f.source_room_id) for f in facts] == [
        ("Postgres", "Database", ROOM)
    ]


@pytest.mark.parametrize("took_back", ["forgot", "corrected"])
def test_a_fact_the_person_took_back_is_not_imported_again(tmp_path: Path, took_back: str) -> None:
    """The record's relation was forgotten, or corrected to another value, before the upgrade.

    Killed by: src/uclone_x/room/seat_knowledge_import.py :: for f in memory.list_facts(include_retracted=True)
    Becomes: for f in memory.list_facts(include_retracted=False)
    """
    knowledge = tmp_path / "knowledge"
    memories = _Memories(tmp_path / "agents")
    person = Provenance.primary(provider="person")
    held = memories(CLONE).record_fact(
        subject="Postgres",
        predicate="is_a",
        object_value="Database",
        provenance=person,
        source_session_id="sess_earlier",
    )
    if took_back == "forgot":
        memories(CLONE).retract_fact(held.fact_id, "forgotten by the user", person)
        expected: list[str] = []
    else:
        memories(CLONE).correct_fact(held.fact_id, "Relational database", person)
        expected = ["Relational database"]
    path = _record(knowledge, SESSION, POSTGRES)

    result = import_seat_knowledge(knowledge, memories)

    assert result == SeatKnowledgeImport(imported=1, facts=0)
    assert [f.object_value for f in memories(CLONE).list_facts()] == expected
    assert not path.exists()


def test_two_imports_of_one_record_at_once_leave_each_fact_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Process B has read the record and its clone's facts when process A imports it whole.

    Two stores on one `memory.json` stand for the two processes. B's first save runs A's
    whole import first, which is the interleaving that saved every fact twice.

    Killed by: src/uclone_x/room/seat_knowledge_import.py :: fact_id=imported_fact_id(clone_id, *key),
    Becomes: fact_id=None,
    """
    knowledge = tmp_path / "knowledge"
    path = _record(knowledge, SESSION, POSTGRES)
    first = _Memories(tmp_path / "agents")
    second = _Memories(tmp_path / "agents")
    store = second(CLONE)
    real_record_fact = store.record_fact
    raced: list[SeatKnowledgeImport] = []

    def record_fact_after_the_other_process(**kwargs: object) -> object:
        if not raced:
            raced.append(import_seat_knowledge(knowledge, first))
        return real_record_fact(**kwargs)  # type: ignore[arg-type]

    store.record_fact = record_fact_after_the_other_process  # type: ignore[method-assign]
    caplog.set_level(logging.INFO, logger="uclone_x.room.seat_knowledge_import")

    result = import_seat_knowledge(knowledge, second)

    assert raced == [SeatKnowledgeImport(imported=1, facts=1)]
    assert result == SeatKnowledgeImport()
    on_disk = CrossSessionMemory(tmp_path / "agents" / CLONE / "memory.json").list_facts()
    assert [(f.subject, f.object_value) for f in on_disk] == [("Postgres", "Database")]
    assert not path.exists()
    assert "taken by another import" in caplog.text
    assert "could not be renamed" not in caplog.text


def test_the_imported_id_is_the_same_in_every_process_and_differs_per_clone() -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: (clone_id, subject, predicate, object_value))
    Becomes: (subject, predicate, object_value))
    """
    ids = {imported_fact_id(c, *POSTGRES) for c in ("scout", "scout", "atlas")}
    assert len(ids) == 2
    assert all(i.startswith("mem_") and len(i) == len("mem_") + 12 for i in ids)


_SECRET = "the-record-says-this-privately"


@pytest.mark.parametrize(
    ("body", "where"),
    [
        # A field of the wrong type: pydantic quotes the value as `input_value`.
        (
            f"relations:\n- source_entity: [{_SECRET}]\n  predicate: p\n  target_entity: t\n",
            "ValidationError at source_entity",
        ),
        # Broken YAML: the parser quotes the line it stopped on.
        (f"relations: [{{source_entity: {_SECRET}\n", "at line"),
    ],
)
def test_an_unreadable_record_is_logged_without_its_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, body: str, where: str
) -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: why = _where(exc)
    Becomes: why = str(exc)
    """
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / f"{SESSION}.yaml").write_text(body, encoding="utf-8")
    caplog.set_level(logging.INFO, logger="uclone_x.room.seat_knowledge_import")

    assert import_seat_knowledge(knowledge, _Memories(tmp_path / "agents")) == (
        SeatKnowledgeImport(unreadable=1)
    )
    assert where in caplog.text
    assert _SECRET not in caplog.text


def test_a_knowledge_directory_that_cannot_be_listed_is_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Killed by: src/uclone_x/room/seat_knowledge_import.py :: except OSError as unlisted:
    Becomes: except LookupError as unlisted:
    """
    knowledge = tmp_path / "knowledge"
    path = _record(knowledge, SESSION, POSTGRES)

    def refused(self: Path, pattern: str) -> object:
        raise PermissionError(13, "Permission denied", str(self))

    with monkeypatch.context() as patch:  # undone before tmp_path's own cleanup globs
        patch.setattr(Path, "glob", refused)
        result = import_seat_knowledge(knowledge, _Memories(tmp_path / "agents"))

    assert result == SeatKnowledgeImport()
    assert path.exists()
    assert "could not be listed (PermissionError (EACCES))" in caplog.text


class TestSetAsideMarker:
    def test_a_marker_not_shaped_dot_word_dash_renames_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "record.yaml"
        path.write_text("x", encoding="utf-8")
        for marker in ("imported-", ".imported", ".-", ""):
            with pytest.raises(ValueError):
                set_aside(path, marker)
        assert path.exists()

    def test_the_kept_name_is_the_records_own_then_the_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "record.yaml"
        path.write_text("x", encoding="utf-8")
        kept = set_aside(path, ".imported-")
        assert kept.name.startswith("record.yaml.imported-")
        assert kept.read_text(encoding="utf-8") == "x"
