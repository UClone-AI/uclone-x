"""A relation a person declared one value at a time (#2332).

Declared as an asserted `functionalProperty` axiom in the clone's rules; nothing in code
lists relations (#2204). The learner then replaces every current value of it, and recall
shows only the value said last. A relation with no declaration keeps the #2328 behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.memory_seed import seed_fact
from tests.unit import test_knowledge_extractor as K
from tests.unit.test_knowledge_extractor import (
    _PROV,  # pyright: ignore[reportPrivateUsage]
    SCOUT,
    _fact,  # pyright: ignore[reportPrivateUsage]
    _memory,  # pyright: ignore[reportPrivateUsage]
    _Model,  # pyright: ignore[reportPrivateUsage]
    _room,  # pyright: ignore[reportPrivateUsage]
    _say,  # pyright: ignore[reportPrivateUsage]
    _seat,  # pyright: ignore[reportPrivateUsage]
    _Seats,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.recall import recall_prompt_section
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import OntologyTier


@pytest.fixture(autouse=True)
def _sessions_under_tmp(tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    """The seats keep their sessions under this test's `tmp_path`, not the working tree."""
    K._SESSIONS[0] = tmp_path / "sessions"  # pyright: ignore[reportPrivateUsage]


def _engine() -> OntologyEngine:
    return OntologyEngine(agent_id="scout", namespace_iri="ontology/clones/scout")


class TestTheDeclaration:
    def test_set_and_read_back_by_the_relation_as_it_is_filed(self) -> None:
        """
        Killed by: src/uclone_x/ontology/engine.py :: and normalize_axiom_kind(axiom.predicate) == FUNCTIONAL_PROPERTY
        Becomes: and True
        Killed by: src/uclone_x/ontology/engine.py :: for name in held:
        Becomes: for name in []:
        """
        engine = _engine()
        engine.teach_axiom(name="other", subject_entity="likes", predicate="transitiveProperty")

        assert engine.set_functional("Lives in", True) is True
        assert engine.set_functional("lives_in", True) is False  # already declared
        assert engine.functional_relations() == frozenset({"lives_in"})

        assert engine.set_functional("LIVES_IN", False) is True
        assert engine.functional_relations() == frozenset()
        assert engine.get_axiom("other") is not None

    def test_a_declaration_written_by_hand_counts_and_one_not_asserted_does_not(self) -> None:
        """`ucx ontology teach` writes the same axiom; a candidate one is not a declaration.

        Killed by: src/uclone_x/ontology/engine.py :: axiom.tier is OntologyTier.ASSERTED
        Becomes: True
        """
        engine = _engine()
        engine.teach_axiom(name="home", subject_entity="lives_in", predicate="owl:functional")
        engine.teach_axiom(
            name="guess",
            subject_entity="works_at",
            predicate="functionalProperty",
            tier=OntologyTier.INDUCED_CANDIDATE,
        )

        assert engine.functional_relations() == frozenset({"lives_in"})

    def test_a_blank_relation_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _engine().set_functional("  ", True)


def _said(memory: CrossSessionMemory, predicate: str, value: str, day: int) -> None:
    """A fact about the person, said on day `day` of September, so the order is not a tie."""
    at = f"2026-09-{day:02d}T00:00:00Z"
    seed_fact(
        memory,
        MemoryFact(
            subject="user",
            predicate=predicate,
            object_value=value,
            provenance=_PROV,
            source_session_id="s0",
            single_valued=False,
            created_at=at,
            updated_at=at,
        ),
    )


def _two_homes(memory: CrossSessionMemory) -> None:
    _said(memory, "lives_in", "Busan", 1)
    _said(memory, "lives_in", "Seoul", 2)


class TestRecallShowsOnlyTheCurrentValue:
    async def test_a_functional_relation_shows_the_value_said_last_alone(self) -> None:
        """
        Killed by: src/uclone_x/memory/recall.py :: if relation_key(latest.predicate) in functional:
        Becomes: if False:
        """
        memory = CrossSessionMemory()
        _two_homes(memory)
        _said(memory, "prefers", "jazz", 3)
        _said(memory, "prefers", "tea", 4)

        section = await recall_prompt_section(memory, "hi", functional=frozenset({"lives_in"}))

        lines = [line for line in section.splitlines() if line.startswith("- ")]
        assert "- user: lives_in -> Seoul" in lines
        assert not any("Busan" in line for line in lines)
        assert any("(said last)" in line and "prefers" in line for line in lines)

    async def test_the_turns_recall_reads_the_clones_own_rules(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/agent/prompt_assembler.py :: functional = frozenset[str]() if ontology is None else ontology.functional_relations()
        Becomes: functional = frozenset[str]()
        """
        memory = _memory(tmp_path)
        _two_homes(memory)
        engine = _engine()
        engine.set_functional("lives_in", True)
        seat = _seat(SCOUT, _Model(), memory)
        seat._ontology = engine  # pyright: ignore[reportPrivateUsage]

        section = await seat._prompt_assembler.recall_memory("hi")  # pyright: ignore[reportPrivateUsage]

        assert section is not None
        assert "Seoul" in section and "Busan" not in section


async def test_the_room_learns_under_the_seats_declaration(tmp_path: Path) -> None:
    """A move the model did not mark replaces the old home once `lives_in` is declared.

    Killed by: src/uclone_x/room/orchestrator.py :: functional=lambda: _functional_relations(agent),
    Becomes: functional=lambda: frozenset(),
    """
    memory = _memory(tmp_path)
    old = memory.record_fact("user", "lives_in", "Busan", _PROV, "s0", origin="told")
    engine = _engine()
    engine.set_functional("lives_in", True)
    seat = _seat(SCOUT, _Model(extractions=[[_fact("user", "lives_in", "Seoul")]]), memory)
    seat._ontology = engine  # pyright: ignore[reportPrivateUsage]
    orch = _room(tmp_path, _Seats((SCOUT, seat)))

    await _say(orch, "I live in Seoul now.")

    [new] = memory.list_facts()
    assert new.object_value == "Seoul"
    assert new.contradicts_fact_id == old.fact_id


class TestNearSubjectGraphMergeAndUndo:
    """Near-matching subjects merge when their graph attributes agree, and undo losslessly (uGraph §6 Q2)."""

    def test_near_subjects_stay_separate_until_attributes_overlap_without_conflict(
        self, tmp_path: Path
    ) -> None:
        """`지현` and `김지현` stay separate with disjoint attributes or conflicting relations,
        and merge once they share a fact and have no unapproved relation conflict.

        Killed by: src/uclone_x/memory/store.py :: if not any(rels1[r] & rels2[r] for r in shared):
        Becomes: if False:
        Killed by: src/uclone_x/memory/store.py :: return not any(r not in several and not (rels1[r] & rels2[r]) for r in shared)
        Becomes: return True
        Killed by: src/uclone_x/memory/store.py :: tx.set_entity_aliases(target_ent.id, aliases)
        Becomes: pass
        Killed by: src/uclone_x/knowledge/sqlite_store.py :: "UPDATE entities SET aliases = ? WHERE id = ?",
        Becomes: "UPDATE entities SET aliases = '[]' WHERE id = ?",
        Killed by: src/uclone_x/knowledge/sqlite_store.py :: self._conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        Becomes: pass
        Killed by: src/uclone_x/knowledge/sqlite_store.py :: "UPDATE edges SET subject_id = ?, expired_at = ?, record = ? WHERE id = ?",
        Becomes: "UPDATE edges SET subject_id = subject_id, expired_at = ?, record = ? WHERE id = ?",
        """
        memory = _memory(tmp_path)
        memory.record_fact("김지현", "lives_in", "Seoul", _PROV, "s0", single_valued=False)
        memory.record_fact("지현", "role", "designer", _PROV, "s0", single_valued=False)

        # No shared attribute yet: still two subjects.
        assert {f.subject for f in memory.list_facts()} == {"김지현", "지현"}

        # Conflicting `lives_in` (Busan vs Seoul) blocks the merge even with shared `works_at`.
        memory.record_fact("지현", "lives_in", "Busan", _PROV, "s0", single_valued=False)
        memory.record_fact("김지현", "works_at", "Hanbit", _PROV, "s0", single_valued=False)
        memory.record_fact("지현", "works_at", "Hanbit", _PROV, "s0", single_valued=False)
        assert {f.subject for f in memory.list_facts()} == {"김지현", "지현"}

        # Once `lives_in` is marked as holding several values, the conflict is lifted and the
        # next write merges `지현` into `김지현`.
        memory.mark_several_values("lives_in")
        memory.record_fact("지현", "prefers", "tea", _PROV, "s0", single_valued=False)

        facts = memory.list_facts()
        assert {f.subject for f in facts} == {"김지현"}
        by_pred = {(f.predicate, f.object_value): f for f in facts}
        assert by_pred[("works_at", "Hanbit")].metadata.get("merged_from") == "지현"
        assert by_pred[("role", "designer")].metadata.get("merged_from") == "지현"
        assert by_pred[("prefers", "tea")].metadata.get("merged_from") == "지현"
        assert by_pred[("lives_in", "Seoul")].metadata.get("merged_from") is None

        entities = memory.knowledge.entities()
        target = next(e for e in entities if e.name == "김지현")
        assert "지현" in target.aliases
        assert not any(e.name == "지현" for e in entities)

        # Later facts recorded under the alias `지현` land directly on `김지현`.
        later = memory.record_fact("지현", " hobby ", "climbing", _PROV, "s0", single_valued=False)
        assert later.subject == "김지현"
        assert later.metadata.get("merged_from") == "지현"

    def test_unmerge_restores_the_source_subject_and_blocks_future_auto_merge(
        self, tmp_path: Path
    ) -> None:
        """`unmerge_subject` splits `지현` back out losslessly and records the pair in
        `unmerged_subjects` so subsequent turns keep them separate.

        Killed by: src/uclone_x/memory/store.py :: if _pair_key(name1, name2) in unmerged:
        Becomes: if False:
        Killed by: src/uclone_x/memory/store.py :: clear_expired=revive,
        Becomes: clear_expired=False,
        """
        memory = _memory(tmp_path)
        memory.record_fact("김지현", "works_at", "Hanbit", _PROV, "s0", single_valued=False)
        memory.record_fact("지현", "role", "designer", _PROV, "s0", single_valued=False)
        memory.record_fact("지현", "works_at", "Hanbit", _PROV, "s0", single_valued=False)

        assert {f.subject for f in memory.list_facts()} == {"김지현"}

        restored = memory.unmerge_subject("지현", _PROV)
        assert {(f.subject, f.predicate, f.object_value) for f in restored} == {
            ("지현", "role", "designer"),
            ("지현", "works_at", "Hanbit"),
        }
        all_facts = memory.list_facts()
        assert {(f.subject, f.predicate, f.object_value) for f in all_facts} == {
            ("김지현", "works_at", "Hanbit"),
            ("지현", "role", "designer"),
            ("지현", "works_at", "Hanbit"),
        }
        assert all(f.metadata.get("merged_from") is None for f in all_facts)

        # Recording another shared fact does not merge them again.
        memory.record_fact("김지현", "likes", "jazz", _PROV, "s0", single_valued=False)
        memory.record_fact("지현", "likes", "jazz", _PROV, "s0", single_valued=False)
        assert {f.subject for f in memory.list_facts()} == {"김지현", "지현"}

    async def test_extractor_resolves_merged_alias_and_shows_known_facts(
        self, tmp_path: Path
    ) -> None:
        """After `지현` merges into `김지현`, a turn mentioning `지현` sees `김지현`'s known facts
        and files new facts about `지현` under `김지현` with `merged_from="지현"`.

        Killed by: src/uclone_x/memory/extractor.py :: and merged_folded in text
        Becomes: and False
        Killed by: src/uclone_x/memory/extractor.py :: aliases.append(merged.strip())
        Becomes: pass
        Killed by: src/uclone_x/memory/extractor.py :: fact_meta["merged_from"] = candidate.merged_from
        Becomes: pass
        """
        memory = _memory(tmp_path)
        memory.record_fact("김지현", "works_at", "Hanbit", _PROV, "s0", single_valued=False)
        memory.record_fact("지현", "works_at", "Hanbit", _PROV, "s0", single_valued=False)

        model = _Model(
            extractions=[
                [
                    _fact(
                        "지현",
                        "works_at",
                        "Hanbit",
                        quote="한빛에서 일하는 지현은",
                    ),
                    _fact(
                        "지현",
                        "role",
                        "editor",
                        quote="지현은 편집자야",
                    ),
                ]
            ]
        )
        seat = _seat(SCOUT, model, memory)
        orch = _room(tmp_path, _Seats((SCOUT, seat)))

        await _say(orch, "한빛에서 일하는 지현은 편집자야.")

        # The extraction prompt included the merged fact about 김지현 even though the span only said 지현.
        assert "김지현 | works_at | Hanbit" in str(model.extraction_requests[0].messages[0].content)
        facts = memory.list_facts()
        works_at = next(f for f in facts if f.predicate == "works_at")
        assert works_at.metadata.get("observations")
        editor = next(f for f in facts if f.predicate == "role")
        assert editor.subject == "김지현"
        assert editor.metadata.get("merged_from") == "지현"
