"""uGraph step 1: the shared fold, the near-match measure, the model and the resolver."""

from __future__ import annotations

import unicodedata

import pytest

from uclone_x.knowledge import (
    Edge,
    Entity,
    EntityRef,
    EntityResolver,
    Interval,
    bigram_overlap,
    edges_at,
    facts_at,
    fold,
    holds_at,
)
from uclone_x.knowledge.models import EdgeStatus
from uclone_x.memory.models import fold_name
from uclone_x.story.quotes import folded

SCOPE = "story:s1"


def _entities(*rows: tuple[str, str, tuple[str, ...]]) -> list[Entity]:
    return [Entity(id=i, scope=SCOPE, kind="characters", name=n, aliases=a) for i, n, a in rows]


def test_fold_is_nfc_whitespace_and_case() -> None:
    """
    Killed by: src/uclone_x/knowledge/fold.py :: return " ".join(unicodedata.normalize("NFC", text).split()).casefold()
    Becomes: return " ".join(text.split()).casefold()
    """
    assert fold("  José \t Ríos\n") == "josé ríos"
    assert fold(unicodedata.normalize("NFD", "José")) == fold("José")


@pytest.mark.parametrize("text", ["José  Ríos", " 라온\t씨 ", "STRASSE", "a b"])
def test_the_two_old_folds_are_the_shared_one(text: str) -> None:
    assert fold_name(text) == fold(text) == folded(text)


def test_bigram_overlap_keeps_its_old_behaviour() -> None:
    assert bigram_overlap("안개 속 등대", "안개 속 등대") == 1.0
    assert bigram_overlap("안개 속 등대", "우주 정거장의 반란") < 0.1
    assert bigram_overlap("a", "ab") == 0.0
    assert bigram_overlap("지현", "김지현") == 0.5


def test_bigram_overlap_treats_composed_and_decomposed_as_one_text() -> None:
    """
    Killed by: src/uclone_x/knowledge/fold.py :: squeezed = _NO_SPACE.sub("", unicodedata.normalize("NFC", text).casefold())
    Becomes: squeezed = _NO_SPACE.sub("", text.casefold())
    """
    assert bigram_overlap(unicodedata.normalize("NFD", "한글 이름"), "한글 이름") == 1.0
    assert bigram_overlap(unicodedata.normalize("NFD", "José Ríos"), "José Ríos") == 1.0


def test_an_entity_ref_cannot_be_made_by_hand() -> None:
    with pytest.raises(TypeError):
        EntityRef(kind="new", name="x")


def test_exact_match_by_id_name_or_alias_in_any_form() -> None:
    resolver = EntityResolver(
        SCOPE, _entities(("raon", "라온", ("Raon",)), ("vane", "Lord Vane", ()))
    )
    for name in ("라온", "RAON", unicodedata.normalize("NFD", "라온"), "  lord   vane "):
        ref = resolver.resolve(name)
        assert ref.kind == "existing"
        assert ref.stage == 2
    assert resolver.resolve("RAON").entity_id == "raon"
    assert resolver.resolve("lord vane").entity_id == "vane"


def test_parenthetical_note_quotes_and_honorific_do_not_hide_a_name() -> None:
    resolver = EntityResolver(SCOPE, _entities(("raon", "라온", ())))
    for name in ("라온(주인공)", "“라온”", "라온 씨", "라온 님"):
        assert resolver.resolve(name).entity_id == "raon", name


def test_a_name_two_entities_share_is_ambiguous_and_not_guessed() -> None:
    """
    Killed by: src/uclone_x/knowledge/resolver.py :: if len(ids) == 1:
    Becomes: if len(ids) >= 1:
    """
    resolver = EntityResolver(SCOPE, _entities(("a", "하린", ("린",)), ("b", "하은", ("린",))))
    ref = resolver.resolve("린")
    assert ref.kind == "ambiguous"
    assert ref.entity_id is None
    assert ref.ambiguous_ids == ("a", "b")


def test_near_matches_are_candidates_over_the_whole_scope() -> None:
    resolver = EntityResolver(
        SCOPE, _entities(("jh", "김지현", ()), ("vane", "Lord Vane", ()), ("moon", "달의 인장", ()))
    )
    short = resolver.resolve("지현")
    assert short.kind == "new"
    assert short.stage == 3
    assert [c[0] for c in short.candidates] == ["jh"]
    assert [c[0] for c in resolver.resolve("Vane 경").candidates] == ["vane"]
    unrelated = resolver.resolve("우주 정거장")
    assert unrelated.kind == "new"
    assert unrelated.candidates == ()
    assert unrelated.stage == 0


def test_containment_needs_a_real_share_of_the_longer_name() -> None:
    """
    Killed by: src/uclone_x/knowledge/resolver.py :: CONTAINMENT_MIN_RATIO = 0.5
    Becomes: CONTAINMENT_MIN_RATIO = 0.0
    """
    resolver = EntityResolver(SCOPE, _entities(("d", "daniel", ())))
    assert resolver.resolve("an").candidates == ()
    assert EntityResolver(SCOPE, _entities(("j", "김지현", ()))).resolve("지현").candidates


def test_raising_jaccard_threshold_drops_near_candidate() -> None:
    """A near candidate by bigram Jaccard alone, neither containing the other (#2088).

    Killed by: src/uclone_x/knowledge/resolver.py :: JACCARD_THRESHOLD = 0.5
    Becomes: JACCARD_THRESHOLD = 0.9
    """
    resolver = EntityResolver(
        SCOPE,
        _entities(("haje", "라온하제", ())),
    )
    # Exactly at 0.5 threshold: neither contains the other, bigram Jaccard is 2/4 = 0.5.
    at_threshold = resolver.resolve("라온하늘")
    assert [c[0] for c in at_threshold.candidates] == ["haje"]


def test_lowering_jaccard_threshold_admits_subthreshold_candidate() -> None:
    """A sub-threshold name must not be a candidate (#2088).

    Killed by: src/uclone_x/knowledge/resolver.py :: JACCARD_THRESHOLD = 0.5
    Becomes: JACCARD_THRESHOLD = 0.1
    """
    resolver = EntityResolver(
        SCOPE,
        _entities(("minjun", "김민준", ())),
    )
    # Under threshold: bigram Jaccard is 1/3 = 0.333 < 0.5.
    under_threshold = resolver.resolve("이민준")
    assert under_threshold.candidates == ()


def test_a_scope_is_not_read_across() -> None:
    other = [Entity(id="x", scope="clone:c1", kind="people", name="라온")]
    ref = EntityResolver(SCOPE, other).resolve("라온")
    assert ref.kind == "new"
    assert ref.candidates == ()


def test_an_empty_name_is_refused() -> None:
    with pytest.raises(ValueError):
        EntityResolver(SCOPE, []).resolve(" (note) ")


def _edge(
    i: str,
    valid: Interval,
    *,
    object_id: str | None = None,
    value: str | None = None,
    status: EdgeStatus = "approved",
    expired_at: str | None = None,
) -> Edge:
    return Edge(
        id=i,
        scope=SCOPE,
        subject_id="raon",
        predicate="state",
        valid=valid,
        recorded_at="t",
        object_id=object_id,
        value=value,
        status=status,
        expired_at=expired_at,
    )


def test_facts_at_reads_approved_unexpired_edges_valid_at_the_point() -> None:
    """
    Killed by: src/uclone_x/knowledge/store.py :: return edge.status in wanted and edge.expired_at is None and edge.valid.contains(axis, point)
    Becomes: return edge.status != "retracted" and edge.expired_at is None and edge.valid.contains(axis, point)
    """
    edges = [
        _edge("1", Interval("story", None, (2, 0)), value="alive"),
        _edge("2", Interval("story", (2, 0), None), value="dead"),
        _edge("3", Interval("story", None, None), object_id="harin", status="proposed"),
        _edge("4", Interval("story", None, None), value="old", expired_at="t2"),
        _edge("5", Interval("wall", None, None), value="wall-only"),
    ]
    assert facts_at(edges, "story", (1, 5)) == [("raon", "state", "alive")]
    assert facts_at(edges, "story", (2, 0)) == [("raon", "state", "dead")]


def test_the_proposed_view_adds_proposed_edges_and_never_retracted_or_expired_ones() -> None:
    """Approved and proposed, each kept as it is; retracted and expired in neither view.

    Killed by: src/uclone_x/knowledge/store.py :: wanted = APPROVED_OR_PROPOSED if proposed else APPROVED
    Becomes: wanted = APPROVED
    """
    edges = [
        _edge("alive", Interval("story", None, None), value="alive"),
        _edge("dead", Interval("story", (2, 0), None), value="dead", status="proposed"),
        _edge("lost", Interval("story", None, None), value="lost", status="retracted"),
        _edge(
            "old", Interval("story", None, None), value="old", expired_at="t2", status="proposed"
        ),
    ]
    assert facts_at(edges, "story", (3, 0)) == [("raon", "state", "alive")]
    assert facts_at(edges, "story", (3, 0), proposed=True) == [
        ("raon", "state", "alive"),
        ("raon", "state", "dead"),
    ]
    assert facts_at(edges, "story", (1, 0), proposed=True) == [("raon", "state", "alive")]
    held = edges_at(edges, "story", (3, 0), proposed=True)
    assert [(e.id, e.status) for e in held] == [("alive", "approved"), ("dead", "proposed")]
    assert not holds_at(edges[2], "story", (3, 0), proposed=True)


def test_an_edge_has_one_object_or_one_value() -> None:
    with pytest.raises(ValueError):
        _edge("x", Interval("wall"))
    with pytest.raises(ValueError):
        _edge("x", Interval("wall"), object_id="a", value="b")
