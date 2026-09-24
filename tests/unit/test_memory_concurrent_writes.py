"""One agent, several processes: a save keeps what the others recorded (#1125)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.memory import store
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.store import CrossSessionMemory


def _provenance() -> Provenance:
    return Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="memory.test", model="test-model"),
        served_by=ServiceRef(provider="memory.test", model="test-model"),
    )


def _record(memory: CrossSessionMemory, value: str) -> str:
    fact = memory.record_fact(
        subject="user",
        predicate="prefers",
        object_value=value,
        provenance=_provenance(),
        source_session_id="sess-1",
        auto_retract_conflicts=False,
    )
    return fact.fact_id


def _values_on_disk(path: Path) -> set[str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {fact["object_value"] for fact in document["facts"]}


def test_a_fact_another_process_recorded_survives_our_save(tmp_path: Path) -> None:
    """The dashboard and `ucx run` are two processes holding one agent's memory.

    Each loads the document when it starts and writes it whole when it saves, so every
    fact the other recorded in between was erased by the later `os.replace` -- silently,
    which is the substitution P6 forbids. The store simply had fewer facts than the agent
    had been told, and nothing anywhere reported a loss.

    Killed by: src/uclone_x/memory/store.py :: self._absorb_concurrent_writes()
    Becomes: pass
    """
    path = tmp_path / "memory.json"

    dashboard = CrossSessionMemory(storage_path=path)
    _record(dashboard, "tabs")
    dashboard.save()

    cli = CrossSessionMemory(storage_path=path)
    _record(cli, "spaces")
    cli.save()

    _record(dashboard, "dark mode")
    dashboard.save()

    assert _values_on_disk(path) == {"tabs", "spaces", "dark mode"}


def test_what_a_save_preserved_is_not_erased_by_the_next_one(tmp_path: Path) -> None:
    """Merging for the write alone would lose the same facts one save later.

    The absorbed facts have to become this object's own view: a merge that only reached
    the serialized document would leave the next `save` writing this process's facts over
    a file it had just agreed with, undoing its own repair.

    Killed by: src/uclone_x/memory/store.py :: self._facts[fact_id] = their_fact
    Becomes: pass
    """
    path = tmp_path / "memory.json"

    ours = CrossSessionMemory(storage_path=path)
    _record(ours, "tabs")
    ours.save()

    theirs = CrossSessionMemory(storage_path=path)
    _record(theirs, "spaces")
    theirs.save()

    ours.save()
    ours.save()

    assert _values_on_disk(path) == {"tabs", "spaces"}


def test_a_retraction_by_another_process_is_not_undone(tmp_path: Path) -> None:
    """Both writers hold the fact; only one of them knows it was withdrawn.

    A fact is never un-retracted, so the retracted copy is the later word whatever the two
    clocks say. Resolving by timestamp alone would let a writer holding a stale active copy
    reinstate a fact the user asked to be forgotten.

    Killed by: src/uclone_x/memory/store.py :: if candidate.retracted != held.retracted:
    Becomes: if False:
    """
    path = tmp_path / "memory.json"

    first = CrossSessionMemory(storage_path=path)
    fact_id = _record(first, "tabs")
    first.save()

    second = CrossSessionMemory(storage_path=path)
    second.retract_fact(
        fact_id,
        reason="the user asked for it to be forgotten",
        provenance=_provenance(),
    )
    # The retraction is given a strictly *earlier* `updated_at` than the copy `first`
    # holds, so only the retracted-wins rule can decide this. `utc_now_iso` has
    # one-second granularity, so without this the two writes usually share a timestamp
    # and the mutant died on a tie rather than on the rule -- a kill that depended on
    # which side of a second boundary the test ran.
    withdrawn = second.get_fact(fact_id)
    assert withdrawn is not None
    second._facts[fact_id] = withdrawn.model_copy(  # pyright: ignore[reportPrivateUsage]
        update={"updated_at": "2000-01-01T00:00:00Z"}
    )
    second.save()

    first.save()

    document = json.loads(path.read_text(encoding="utf-8"))
    (stored,) = document["facts"]
    assert stored["retracted"] is True
    assert stored["retraction_reason"] == "the user asked for it to be forgotten"


def test_an_unreadable_document_is_quarantined_rather_than_overwritten(tmp_path: Path) -> None:
    """A file we cannot parse still holds facts, and a save would be their only grave.

    `load`'s quarantine is reached from the save path too, so the unreadable copy is kept
    beside the new document and `load_failure` says so, instead of the write silently
    taking its place.

    Killed by: src/uclone_x/memory/store.py :: except Exception as unreadable:
    Becomes: except ImportError:
    """
    path = tmp_path / "memory.json"

    ours = CrossSessionMemory(storage_path=path)
    _record(ours, "tabs")
    ours.save()

    path.write_text("{ this is not json", encoding="utf-8")
    ours.save()

    assert _values_on_disk(path) == {"tabs"}
    assert ours.load_failure is not None
    quarantined = list(tmp_path.glob("memory.json.unreadable-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{ this is not json"


def test_an_unchanged_document_is_not_re_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case -- one writer -- must not pay for the merge on every save.

    The file is identified by the digest of its bytes and compared against the document
    this object last agreed with, so a store nobody else is writing reads the file and
    stops there.

    Killed by: src/uclone_x/memory/store.py :: if _document_digest(content) == self._witness:
    Becomes: if False:
    """
    path = tmp_path / "memory.json"
    memory = CrossSessionMemory(storage_path=path)
    _record(memory, "tabs")
    memory.save()

    parsed: list[str] = []
    real_parse = store._parse_facts  # pyright: ignore[reportPrivateUsage]

    def counting_parse(content: str) -> dict[str, MemoryFact]:
        parsed.append(content)
        # The original, captured before the patch. Reaching for `store._parse_facts`
        # here called this function, and the recursion died inside the guard it was
        # meant to be measuring.
        return real_parse(content)

    monkeypatch.setattr(store, "_parse_facts", counting_parse)
    memory.save()

    assert parsed == [], "a document nobody else touched costs one read, not a merge"


def test_the_merge_is_a_check_not_a_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The docstring must not grow into a claim of serialisability.

    What the merge closes is the read-modify-write window spanning a whole session. It
    leaves the one inside `save`: two writers that have both read and not yet replaced
    still lose one of the two writes. Standing a writer in that window -- its read already
    done -- reproduces the surviving loss, so the limit is a recorded fact rather than a
    sentence nobody checked.
    """
    path = tmp_path / "memory.json"

    slow = CrossSessionMemory(storage_path=path)
    _record(slow, "tabs")
    slow.save()

    fast = CrossSessionMemory(storage_path=path)
    _record(fast, "interleaved")
    fast.save()

    # `slow` read the document before `fast` replaced it, and is now past that read.
    def already_read(_self: CrossSessionMemory) -> None:
        return None

    monkeypatch.setattr(CrossSessionMemory, "_absorb_concurrent_writes", already_read)
    slow.save()

    assert "interleaved" not in _values_on_disk(path), (
        "if this now passes, `save` has become mutually exclusive and its docstring, "
        "which says it is not, has to be corrected"
    )


def test_a_document_that_cannot_be_decoded_is_quarantined_too(tmp_path: Path) -> None:
    """Unreadable is unreadable: a file we cannot decode never reaches the parser.

    The read used to sit outside the guard, so a document that was not UTF-8 -- a
    truncated write, a file another tool put there -- sent `UnicodeDecodeError` out
    through `save`, and so out of `record_fact`, which had never raised for a bad file.
    Nothing was quarantined, `load_failure` stayed `None`, and every later write raised
    the same codec error: an agent that could not record anything again, reported to the
    operator as a traceback rather than as a corrupt store (P0, P6).

    Killed by: src/uclone_x/memory/store.py :: except Exception as unreadable:
    Becomes: except json.JSONDecodeError:
    """
    path = tmp_path / "memory.json"

    ours = CrossSessionMemory(storage_path=path)
    _record(ours, "tabs")
    ours.save()

    path.write_bytes(b"\xff\xfe{}")
    ours.save()

    assert _values_on_disk(path) == {"tabs"}
    assert ours.load_failure is not None
    quarantined = list(tmp_path.glob("memory.json.unreadable-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"\xff\xfe{}"


def test_quarantining_does_not_drop_the_facts_we_were_about_to_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unreadable branch must not re-read the file through `load`.

    `load`'s success path assigns `self._facts` outright. Delegating to it meant that a
    document which failed the first read and parsed on the second -- another process
    having replaced it in between -- left this object holding the file's facts and none
    of its own, which `save` then wrote: the fact this session had just been told,
    dropped silently, by the very method added to stop that happening.

    Killed by: src/uclone_x/memory/store.py :: self._quarantine_unreadable(unreadable)
    Becomes: self.load()
    """
    path = tmp_path / "memory.json"

    theirs = CrossSessionMemory(storage_path=path)
    _record(theirs, "spaces")

    ours = CrossSessionMemory(storage_path=path)

    # Written after `ours` last agreed with the document, so the digest no longer
    # matches and the merge actually reaches the parser. Nothing here knows "tabs":
    # a re-read through `load` can only come back without it.
    _record(theirs, "wheels")

    real_parse = store._parse_facts  # pyright: ignore[reportPrivateUsage]
    attempts: list[str] = []

    def fails_once(content: str) -> dict[str, MemoryFact]:
        attempts.append(content)
        if len(attempts) == 1:
            raise ValueError("unreadable on this read, readable on the next")
        return real_parse(content)

    monkeypatch.setattr(store, "_parse_facts", fails_once)
    ours_id = _record(ours, "tabs")

    assert ours.get_fact(ours_id) is not None, "our own fact must survive the quarantine"
    assert "tabs" in _values_on_disk(path)
