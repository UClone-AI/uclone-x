"""uGraph step 2: the story codex read through the shared fact model (`CodexYamlStore`)."""

from __future__ import annotations

from typing import Any

from uclone_x_story.codex_store import STORY_AXIS, CodexYamlStore
from uclone_x_story.context import CodexIndex, CodexItem, render_entry
from uclone_x_story.schemas import CharacterEntry, Outline, Proposal
from uclone_x_story.timeline import entry_snapshot, place_scenes

# Four scenes in reading order; s4 is a flashback that happens first in story time.
OUTLINE = Outline.model_validate(
    {
        "chapters": [
            {
                "id": "ch01",
                "title": "One",
                "scenes": [
                    {"id": "s1", "title": "Harbour", "story_time": 1},
                    {"id": "s2", "title": "Storm", "story_time": 2},
                    {"id": "s3", "title": "Lighthouse", "story_time": 3},
                    {"id": "s4", "title": "Long ago", "story_time": 0},
                ],
            }
        ]
    }
)


def _codex(*entries: dict[str, Any]) -> CodexIndex:
    return CodexIndex(
        items=tuple(
            CodexItem(kind="characters", entry=CharacterEntry.model_validate(e)) for e in entries
        )
    )


def _proposal(number: int, entry_id: str, change: dict[str, Any], **more: Any) -> Proposal:
    return Proposal.model_validate(
        {
            "id": f"p{number:03d}",
            "kind": "characters",
            "entry_id": entry_id,
            "change": change,
            "evidence": [{"scene_id": "s2", "quote": "라온은 끝내 눈을 뜨지 못했다"}],
            "proposed_at": "2026-10-07T00:00:00+00:00",
            "room_id": "room",
            "entry_digest": None if "new_entry" in change else "d1",
            **more,
        }
    )


RAON = {"id": "raon", "name": "라온", "state": {"status": "alive"}}
DEATH = {"progression": {"at": "s2", "set": {"status": "dead"}}}


def _status(
    store: CodexYamlStore, scene_id: str, *, ended: bool = True, proposed: bool
) -> set[str]:
    point = store.point(scene_id, ended=ended)
    return {
        value
        for subject, predicate, value in store.facts_at(STORY_AXIS, point, proposed=proposed)
        if subject == "raon" and predicate == "status"
    }


class TestAPendingProposalIsAProposedEdge:
    def test_a_pending_death_at_scene_k_is_seen_after_k_and_not_before(self) -> None:
        """The acceptance of #2206: a death written at s2 and not yet approved.

        Killed by: examples/story/src/uclone_x_story/codex_store.py :: status="proposed" if pending else "retracted",
        Becomes: status="approved" if pending else "retracted",
        Killed by: examples/story/src/uclone_x_story/codex_store.py :: start = scene_point(placements[progression.at], ended=True)
        Becomes: start = None
        """
        store = CodexYamlStore.read("moon", OUTLINE, _codex(RAON), [_proposal(1, "raon", DEATH)])
        # Approved and proposed: dead from the end of s2 on, in story time, and no longer
        # alive there (#2207: the pending change ends the value it replaces).
        assert _status(store, "s3", proposed=True) == {"dead"}
        assert _status(store, "s2", proposed=True) == {"dead"}
        assert _status(store, "s2", ended=False, proposed=True) == {"alive"}
        assert _status(store, "s1", proposed=True) == {"alive"}
        assert _status(store, "s4", proposed=True) == {"alive"}  # the flashback is earlier
        # The approved view alone does not believe it.
        assert _status(store, "s3", proposed=False) == {"alive"}
        [dead] = [e for e in store.edges() if e.value == "dead"]
        assert dead.status == "proposed"
        assert dead.evidence == (("s2", "라온은 끝내 눈을 뜨지 못했다"),)

    def test_a_proposed_change_ends_where_the_entry_next_changes_that_value(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/codex_store.py :: later = [p for p in changes.get((section, key), ()) if p > start]
        Becomes: later = []
        """
        risen = {**RAON, "progressions": [{"at": "s3", "set": {"status": "alive"}}]}
        store = CodexYamlStore.read("moon", OUTLINE, _codex(risen), [_proposal(1, "raon", DEATH)])
        assert "dead" in _status(store, "s2", proposed=True)
        assert _status(store, "s3", proposed=True) == {"alive"}

    def test_a_pending_change_ends_the_value_it_replaces_only_in_the_proposed_view(
        self,
    ) -> None:
        """#2207: the proposed view is the codex with the proposal made, one value per key.

        Killed by: examples/story/src/uclone_x_story/codex_store.py :: out.add(entry.id, section, key, held, None, ended)
        Becomes: out.add(entry.id, section, key, held, None)
        Killed by: src/uclone_x/knowledge/store.py :: and edge.valid_in(proposed=proposed).contains(axis, point)
        Becomes: and edge.valid.contains(axis, point)
        """
        store = CodexYamlStore.read("moon", OUTLINE, _codex(RAON), [_proposal(1, "raon", DEATH)])
        [alive] = [e for e in store.edges() if e.value == "alive"]
        assert alive.status == "approved" and alive.valid.end is None
        assert alive.proposed_valid is not None
        assert alive.proposed_valid.end == store.point("s2")
        assert _status(store, "s3", proposed=True) == {"dead"}
        assert _status(store, "s3", proposed=False) == {"alive"}

    def test_later_pending_changes_follow_one_another(self) -> None:
        """A second pending change of one value ends the first, at its own scene, as
        approving the two in turn would.

        Killed by: examples/story/src/uclone_x_story/codex_store.py :: ended = _ended_by_pending(pending.get((section, key), ()), start, order=order)
        Becomes: ended = None
        """
        rises = {"progression": {"at": "s3", "set": {"status": "alive"}}}
        store = CodexYamlStore.read(
            "moon",
            OUTLINE,
            _codex(RAON),
            [_proposal(1, "raon", DEATH), _proposal(2, "raon", rises)],
        )
        assert _status(store, "s2", proposed=True) == {"dead"}
        assert _status(store, "s3", proposed=True) == {"alive"}

    def test_a_pending_change_to_a_pending_new_entry_ends_its_first_value(self) -> None:
        """A cast still waiting for approval dies in a scene: both are proposals, and the
        proposed view has the death, not the cast's first status beside it."""
        harin = {"id": "harin", "name": "하린", "state": {"status": "alive"}}
        dies = {"progression": {"at": "s2", "set": {"status": "dead"}}}
        store = CodexYamlStore.read(
            "moon",
            OUTLINE,
            _codex(RAON),
            [_proposal(1, "harin", {"new_entry": harin}), _proposal(2, "harin", dies)],
        )
        statuses = {
            v
            for s, p, v in store.facts_at(STORY_AXIS, store.point("s3"), proposed=True)
            if s == "harin" and p == "status"
        }
        assert statuses == {"dead"}

    def test_a_rejected_proposal_is_retracted_and_in_neither_view(self) -> None:
        rejected = _proposal(1, "raon", DEATH, status="rejected", decided_at="2026-10-07T01:00:00")
        store = CodexYamlStore.read("moon", OUTLINE, _codex(RAON), [rejected])
        [dead] = [e for e in store.edges() if e.value == "dead"]
        assert dead.status == "retracted" and dead.expired_at == "2026-10-07T01:00:00"
        assert _status(store, "s3", proposed=True) == {"alive"}

    def test_a_pending_new_entry_is_an_entity_with_proposed_edges(self) -> None:
        harin = {"id": "harin", "name": "하린", "state": {"status": "alive"}}
        store = CodexYamlStore.read(
            "moon", OUTLINE, _codex(RAON), [_proposal(1, "harin", {"new_entry": harin})]
        )
        assert [e.id for e in store.entities()] == ["raon", "harin"]
        point = store.point("s1")
        assert ("harin", "status", "alive") in store.facts_at(STORY_AXIS, point, proposed=True)
        assert ("harin", "status", "alive") not in store.facts_at(STORY_AXIS, point)


class TestTheCodexIsReadAsEdges:
    def test_a_progression_ends_one_edge_and_starts_the_next_at_its_scene(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/codex_store.py :: out.add(entry.id, section, key, held, point, _earliest(point, ended))
        Becomes: out.add(entry.id, section, key, held, None, ended)
        """
        dies = {**RAON, "progressions": [{"at": "s2", "set": {"status": "dead"}}]}
        store = CodexYamlStore.read("moon", OUTLINE, _codex(dies))
        assert _status(store, "s1", proposed=False) == {"alive"}
        assert _status(store, "s2", ended=False, proposed=False) == {"alive"}
        assert _status(store, "s2", proposed=False) == {"dead"}
        assert _status(store, "s3", proposed=False) == {"dead"}
        assert all(e.status == "approved" for e in store.edges())
        assert store.scope == "story:moon"

    def test_relations_are_edges_between_two_entities(self) -> None:
        """A relation is `(entry, word, other entry)`, not a state key with the other's id.

        Killed by: examples/story/src/uclone_x_story/codex_store.py :: object_id=key if section == "relations" else None,
        Becomes: object_id=None,
        """
        doyun = {"id": "doyun", "name": "도윤", "relations": {"raon": "parent"}}
        changes = {**doyun, "progressions": [{"at": "s3", "relations": {"raon": None}}]}
        store = CodexYamlStore.read("moon", OUTLINE, _codex(RAON, changes))
        [edge] = [e for e in store.edges() if e.subject_id == "doyun"]
        assert (edge.predicate, edge.object_id, edge.value) == ("parent", "raon", None)
        assert ("doyun", "parent", "raon") in store.facts_at(STORY_AXIS, store.point("s2"))
        assert ("doyun", "parent", "raon") not in store.facts_at(STORY_AXIS, store.point("s3"))

    def test_a_list_is_one_edge_per_element_and_a_structured_value_is_marked(self) -> None:
        state: dict[str, Any] = {"possesses": ["lantern", "key"], "x": {}}
        mara: dict[str, Any] = {"id": "mara", "name": "Mara", "state": state}
        store = CodexYamlStore.read("moon", OUTLINE, _codex(mara))
        held = {(r.edge.predicate, r.edge.value, r.record["single"]) for r in store.rows()}
        assert held == {
            ("possesses", "lantern", True),
            ("possesses", "key", True),
            ("x", "{}", False),
        }
        assert store.resolve("mara").entity_id == "mara"


def test_the_writer_sees_the_relations_the_scene_starts_from() -> None:
    """A relation a progression adds is in the snapshot and the entry the Writer reads.

    Killed by: examples/story/src/uclone_x_story/timeline.py :: relations[other] = word
    Becomes: pass
    """
    doyun = CharacterEntry.model_validate(
        {
            "id": "doyun",
            "name": "도윤",
            "relations": {"raon": "sibling"},
            "progressions": [{"at": "s1", "relations": {"harin": "spouse", "raon": None}}],
        }
    )
    placements = place_scenes(OUTLINE)
    item = CodexItem(kind="characters", entry=doyun)
    before = entry_snapshot(doyun, placements, "s1", through_scene=False)
    after = entry_snapshot(doyun, placements, "s2", through_scene=False)
    assert render_entry(item, before)["relations"] == {"raon": "sibling"}
    assert render_entry(item, after)["relations"] == {"harin": "spouse"}


class TestAPendingRemovalEndsTheValue:
    """F2 of #2210's review: a pending `null` is a removal, and the proposed view sees it."""

    def test_a_pending_relation_removal_ends_the_relation_in_the_proposed_view(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/codex_store.py :: out.add(entry.id, section, key, held, None, ended)
        Becomes: out.add(entry.id, section, key, held, None)
        """
        doyun = {"id": "doyun", "name": "도윤", "relations": {"raon": "parent"}}
        dropped = {"progression": {"at": "s2", "relations": {"raon": None}}}
        store = CodexYamlStore.read(
            "moon", OUTLINE, _codex(RAON, doyun), [_proposal(1, "doyun", dropped)]
        )
        after, before = store.point("s3"), store.point("s2", ended=False)
        assert "raon" not in store.relations_at("doyun", after, proposed=True)
        assert "raon" in store.relations_at("doyun", before, proposed=True)
        assert "raon" in store.relations_at("doyun", after, proposed=False)

    def test_a_pending_state_removal_ends_the_value_in_the_proposed_view(self) -> None:
        held = {**RAON, "state": {"status": "alive", "possesses": ["lamp"]}}
        lost = {"progression": {"at": "s2", "set": {"possesses": None}}}
        store = CodexYamlStore.read("moon", OUTLINE, _codex(held), [_proposal(1, "raon", lost)])
        after = store.point("s3")
        assert store.values_at("raon", "possesses", after, proposed=True) == []
        [still] = store.values_at("raon", "possesses", after, proposed=False)
        assert still.edge.value == "lamp"


class TestAnAppliedProposalKeepsItsQuotesOnItsProgression:
    """uGraph §6 Q3 (#2346): a progression that names the proposal it came from carries its
    evidence on the codex edge, while a progression written by hand without `proposal` has
    none and a missing proposal file does not break reading."""

    def test_a_progression_naming_an_applied_proposal_carries_its_evidence(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/codex_store.py :: evidence = self.applied_evidence.get(cell.proposal, ())
        Becomes: evidence = ()
        Killed by: examples/story/src/uclone_x_story/codex_store.py :: cells[(section, key)] = _Cell(value, point, progression.at, progression.proposal)
        Becomes: cells[(section, key)] = _Cell(value, point, progression.at)
        """
        applied = _proposal(1, "raon", DEATH, status="applied", decided_at="2026-10-07T01:00:00")
        dies = {
            **RAON,
            "progressions": [
                {"at": "s2", "set": {"status": "dead"}, "proposal": "p001"},
                {"at": "s3", "set": {"mood": "calm"}},
            ],
        }
        store = CodexYamlStore.read("moon", OUTLINE, _codex(dies), [applied])
        [dead] = [r for r in store.rows() if r.edge.value == "dead"]
        assert dead.edge.status == "approved"
        assert dead.edge.evidence == (("s2", "라온은 끝내 눈을 뜨지 못했다"),)
        assert dead.record["origin"]["proposal"] == "p001"
        [calm] = [r for r in store.rows() if r.edge.value == "calm"]
        assert calm.edge.evidence == ()
        assert "proposal" not in calm.record["origin"]
        # When the proposal file is no longer there, the codex still reads cleanly.
        without_file = CodexYamlStore.read("moon", OUTLINE, _codex(dies), [])
        [dead_without] = [r for r in without_file.rows() if r.edge.value == "dead"]
        assert dead_without.edge.evidence == ()
        assert dead_without.record["origin"]["proposal"] == "p001"
