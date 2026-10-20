"""uGraph: the shared fold, the near-match measure, the model, the resolver, and what an
extraction keeps of a model's facts (`verify_facts`)."""

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
from uclone_x.knowledge.extraction import (
    ALIAS_MARKER,
    AliasCandidate,
    ExtractedFact,
    alias_prompt,
    confirmed_aliases,
    read_presence,
    verify_facts,
)
from uclone_x.knowledge.models import EdgeStatus
from uclone_x.knowledge.quotes import folded
from uclone_x.memory.models import fold_name

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
    Killed by: src/uclone_x/knowledge/store.py :: edge.status in wanted
    Becomes: edge.status != "retracted"
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


# -- verify_facts: the names a fact gives ------------------------------------------------

CAST = _entities(
    ("kalrin_deu_raiteu", "칼린 드 라이트", ()),
    ("moraseu", "모라스", ()),
    ("harin", "하린", ("린",)),
    ("haeun", "하은", ("린",)),
)
TEXT = "Kalrin fell in the snow and did not rise. 린은 문 앞에 서 있었다. Moras watched."


def _fact(subject: str, quote: str, named: str = "") -> ExtractedFact:
    return ExtractedFact(subject=subject, key="status", value="dead", quote=quote, named=named)


def test_a_name_two_entities_share_drops_the_fact_and_guesses_neither() -> None:
    """#2216 review (M7): a fact about "린", which 하린 and 하은 both go by, is dropped and
    counted, never filed under the first match.

    Killed by: src/uclone_x/knowledge/extraction.py :: if len(ids) > 1:
    Becomes: if len(ids) > 9:
    """
    verified = verify_facts(
        [_fact("린", "린은 문 앞에 서 있었다")], TEXT, EntityResolver(SCOPE, CAST)
    )
    assert verified.kept == []
    assert verified.dropped == {"ambiguous_name": 1}


def test_an_id_answer_resolves_a_name_written_in_another_script_and_is_an_alias() -> None:
    """#2217: the cast is in Hangul and the prose in Latin letters. The reply's id resolves
    the subject, and the words the quote calls it by come back as a candidate alias -- no
    rule in code maps one script to the other.

    Killed by: src/uclone_x/knowledge/extraction.py :: return named, False
    Becomes: return None, False
    """
    verified = verify_facts(
        [_fact("kalrin_deu_raiteu", "Kalrin fell in the snow", named="Kalrin")],
        TEXT,
        EntityResolver(SCOPE, CAST),
    )
    [read] = verified.kept
    assert (read.subject, read.alias) == ("kalrin_deu_raiteu", "Kalrin")
    assert verified.dropped == {}


def test_a_name_the_subject_has_or_the_quote_lacks_is_no_alias() -> None:
    """A name the entity already has is no news, and words not in the quote were not
    copied from the text.

    Killed by: src/uclone_x/knowledge/extraction.py :: if fold(named) not in fold(fact.quote):
    Becomes: if False:
    """
    resolver = EntityResolver(SCOPE, CAST)
    known = verify_facts([_fact("moraseu", "Moras watched", named="모라스")], TEXT, resolver)
    absent = verify_facts(
        [_fact("kalrin_deu_raiteu", "Kalrin fell in the snow", named="Ilsa")], TEXT, resolver
    )
    assert [(r.subject, r.alias) for r in known.kept] == [("moraseu", None)]
    assert [(r.subject, r.alias) for r in absent.kept] == [("kalrin_deu_raiteu", None)]


def test_a_name_that_is_another_entitys_drops_the_fact() -> None:
    """The reply's id says Kalrin and its words say 모라스: they disagree, and nothing is
    guessed.

    Killed by: src/uclone_x/knowledge/extraction.py :: return None, found.entity_id != subject
    Becomes: return None, found.entity_id == subject
    """
    text = "모라스 watched Kalrin fall."
    verified = verify_facts(
        [_fact("kalrin_deu_raiteu", "모라스 watched Kalrin fall", named="모라스")],
        text,
        EntityResolver(SCOPE, CAST),
    )
    assert verified.kept == []
    assert verified.dropped == {"named_other": 1}


def test_a_name_two_other_entities_share_drops_the_fact() -> None:
    """ "린" is 하린's and 하은's, and the reply's id says Kalrin: the words name someone else,
    whichever of the two they are (#2226).

    Killed by: src/uclone_x/knowledge/extraction.py :: return None, subject not in found.ambiguous_ids
    Becomes: return None, False
    """
    verified = verify_facts(
        [_fact("kalrin_deu_raiteu", "린은 문 앞에 서 있었다", named="린")],
        TEXT,
        EntityResolver(SCOPE, CAST),
    )
    assert verified.kept == []
    assert verified.dropped == {"named_other": 1}


def test_the_alias_question_lists_each_candidate_and_only_a_yes_confirms() -> None:
    """Stage 4 (§3.4): a candidate is a name only when the answer says so; an answer that
    cannot be read confirms none.

    Killed by: src/uclone_x/knowledge/extraction.py :: if said is True or (isinstance(said, str) and said.strip().lower() in ("true", "yes")):
    Becomes: if said or (isinstance(said, str) and said.strip().lower() in ("true", "yes")):
    """
    kalrin = CAST[0]
    asked = alias_prompt(
        [
            AliasCandidate(kalrin, "Kalrin", "Kalrin fell"),
            AliasCandidate(kalrin, "she", "she fell"),
        ]
    )
    assert asked.startswith(ALIAS_MARKER)
    assert '1. 칼린 드 라이트 -- words: "Kalrin" -- quote: "Kalrin fell"' in asked
    assert '2. 칼린 드 라이트 -- words: "she"' in asked
    reply = '{"1": true, "2": false, "3": "no", "4": "yes"}'
    assert confirmed_aliases(reply, 4) == {0, 3}
    assert confirmed_aliases("I think the first one.", 2) == set()


# --------------------------------------------------------------------------------------
# A fact's object stands on a quote that names it, in any script (#2238, #2249)
# --------------------------------------------------------------------------------------

#: One holder and one item per script family. The item's name is never the word the quote
#: uses whole: a noun-first name (French), names with no spaces (Japanese, Chinese), a
#: one-character word (키), so a rule on a name's last word or its spacing fails somewhere.
SCRIPTS = {
    "ko": ("에이든", "도시 중심가의 키", "에이든은 키를 주머니에 넣었다", "키"),
    "en": ("Aidan", "the old relic of the vale", "Aidan held the relic close", "relic"),
    "fr": ("Aidan", "la clé du centre-ville", "Aidan sortit la clé de sa poche", "la clé"),
    "ja": ("ジノ", "壊れた懐中時計", "ジノは時計をポケットに入れた", "時計"),
    "zh": ("艾登", "城市中心的钥匙", "艾登把钥匙放进口袋", "钥匙"),
}


def _holding(script: str) -> tuple[EntityResolver, str, str, str]:
    holder, item, quote, words = SCRIPTS[script]
    resolver = EntityResolver(
        SCOPE,
        [
            Entity("holder", SCOPE, "characters", holder),
            Entity("item", SCOPE, "items", item),
            Entity("place", SCOPE, "places", "도시 중심가"),
        ],
    )
    return resolver, quote, words, item


def _held(quote: str, *, value: str = "item", object_named: str = "") -> ExtractedFact:
    return ExtractedFact(
        subject="holder", key="possesses", value=value, quote=quote, object_named=object_named
    )


@pytest.mark.parametrize("script", list(SCRIPTS))
def test_the_words_the_reading_gives_for_the_object_hold_it_in_every_script(script: str) -> None:
    """The reading model says which words of the quote name the object (`object_named`);
    they hold it when they are in the quote and in the object's name, as characters --
    no word order, spacing or ending is assumed (#2249).

    Killed by: src/uclone_x/knowledge/extraction.py :: return words in _squeezed(name)
    Becomes: return False
    """
    resolver, quote, words, _ = _holding(script)
    verified = verify_facts([_held(quote, object_named=words)], quote, resolver)
    assert [r.subject for r in verified.kept] == ["holder"]


@pytest.mark.parametrize("script", list(SCRIPTS))
def test_without_those_words_only_the_whole_name_holds_the_object(script: str) -> None:
    """With no `object_named`, the quote must hold one of the object's names or aliases
    whole; a part of the name does not, whatever the script.

    Killed by: src/uclone_x/knowledge/extraction.py :: if any(_squeezed(name) in said for name in names):
    Becomes: if True:
    """
    resolver, quote, _, item = _holding(script)
    holder = SCRIPTS[script][0]
    whole = f"{holder} {item}"
    partial = verify_facts([_held(quote)], quote, resolver)
    named = verify_facts([_held(whole)], whole, resolver)
    assert partial.kept == [] and partial.dropped == {"object_not_in_quote": 1}
    assert [r.subject for r in named.kept] == ["holder"]


def test_object_words_that_are_not_in_the_quote_or_name_another_entity_hold_nothing() -> None:
    """The words must be the quote's own, and not another entity's name: "도시 중심가" is
    in "도시 중심가의 키" as characters, and it is a place (#2238's false finding).

    Killed by: src/uclone_x/knowledge/extraction.py :: if not words or words not in said:
    Becomes: if not words:
    Killed by: src/uclone_x/knowledge/extraction.py :: if found.kind != "new" and found.entity_id != entity_id:
    Becomes: if False:
    """
    resolver, quote, _, _ = _holding("ko")
    cafe = "사라가 도시 중심가의 한 카페에 앉아 있었다"
    elsewhere = "에이든은 열쇠를 주머니에 넣었다"
    verified = verify_facts(
        [
            _held(elsewhere, object_named="키"),  # in the item's name, not in the quote
            _held(cafe, object_named="도시 중심가"),  # another entity's name
            _held(quote, object_named="주머니"),  # in the quote, not in the item's name
        ],
        f"{quote} {cafe} {elsewhere}",
        resolver,
    )
    assert verified.kept == []
    assert verified.dropped == {"object_not_in_quote": 3}


def test_a_relation_stands_on_a_quote_that_names_the_other() -> None:
    """`of` is held to its quote as the value is: kin named there is kept, kin not named
    is dropped (#2249 item 2).

    Killed by: src/uclone_x/knowledge/extraction.py :: objects = [e for e in (other, _object_of(resolver, fact, subject)) if e is not None]
    Becomes: objects = [e for e in (_object_of(resolver, fact, subject),) if e is not None]
    """
    resolver = EntityResolver(SCOPE, CAST)
    said = "칼린 드 라이트는 모라스의 딸이었다"
    elsewhere = "칼린 드 라이트는 눈 속에 쓰러졌다"
    facts = [
        ExtractedFact(
            subject="kalrin_deu_raiteu", key="family", value="daughter", quote=said, of="moraseu"
        ),
        ExtractedFact(
            subject="kalrin_deu_raiteu", key="family", value="sister", quote=elsewhere, of="moraseu"
        ),
    ]
    verified = verify_facts(facts, f"{said} {elsewhere}", resolver)
    assert [r.fact.value for r in verified.kept] == ["daughter"]
    assert verified.dropped == {"object_not_in_quote": 1}


def test_a_new_person_a_relation_names_stands_on_a_quote_that_names_them() -> None:
    """The clone files a name it does not hold as a new subject (#2235); a relation to such
    a person is kept when the quote names them, not dropped for being no entity yet
    (#2249 item 3).

    Killed by: src/uclone_x/knowledge/extraction.py :: names = resolver.names_of(entity_id) or (entity_id,)
    Becomes: names = resolver.names_of(entity_id)
    """
    resolver = EntityResolver("clone:c", [Entity("user", "clone:c", "person", "user")])
    said = "My sister Mina works in Seoul"
    facts = [
        ExtractedFact(subject="user", key="sister", value="yes", quote=said, of="Mina"),
        ExtractedFact(subject="user", key="brother", value="yes", quote=said, of="Jo"),
    ]
    verified = verify_facts(facts, said, resolver, new_names=True)
    assert [(r.fact.key, r.of) for r in verified.kept] == [("sister", "Mina")]
    assert verified.dropped == {"object_not_in_quote": 1}


def test_object_words_that_are_only_a_bracket_or_quotation_marks_drop_that_fact_alone() -> None:
    """`object_named` of `(그것)` is in the quote, and is no name once its brackets go: it
    was resolved as one, which raised and lost the whole reading. It holds nothing; the
    reading's other facts stand (#2254 review).

    Killed by: src/uclone_x/knowledge/extraction.py :: if not name_key(fact.object_named):
    Becomes: if False:
    """
    resolver, quote, _, _ = _holding("ko")
    bracketed = "에이든은 (그것)을 주머니에 넣었다"
    verified = verify_facts(
        [
            _held(bracketed, object_named="(그것)"),
            _held(quote, object_named="키"),
        ],
        f"{bracketed} {quote}",
        resolver,
    )
    assert [r.fact.quote for r in verified.kept] == [quote]
    assert verified.dropped == {"object_not_in_quote": 1}


def test_names_are_folded_by_one_function() -> None:
    """`quotes.folded`, the name the story example imports, is `fold` itself, not a second
    copy of the rule (#2240 item 5).

    Killed by: src/uclone_x/knowledge/quotes.py :: folded = fold
    Becomes: folded = str.casefold
    """
    from uclone_x.knowledge import quotes

    assert quotes.folded is fold


# -- read_presence: who a reply says is there at the text's own time (#2311) -------------


def test_a_presence_map_is_read_by_entity_and_keeps_only_its_words() -> None:
    """Names and ids both resolve; a word not among the four and a name no entity has are
    left out, not guessed.

    Killed by: src/uclone_x/knowledge/extraction.py :: if presence not in PRESENCES:
    Becomes: if False:
    """
    resolver = EntityResolver(SCOPE, _entities(("harin", "하린", ()), ("jinho", "이진호", ())))

    read = read_presence(
        {"harin": "Acting", "이진호": "remembered", "누군가": "acting", "jinho ": "ghost"},
        resolver,
    )

    assert read == {"harin": "acting", "jinho": "remembered"}


def test_one_entity_given_twice_is_acting_when_either_says_so() -> None:
    """
    Killed by: src/uclone_x/knowledge/extraction.py :: if out.get(entity) != ACTING:
    Becomes: if True:
    """
    resolver = EntityResolver(SCOPE, _entities(("jinho", "이진호", ())))

    assert read_presence({"jinho": "acting", "이진호": "mentioned"}, resolver) == {
        "jinho": "acting"
    }


def test_no_map_is_no_answer_and_an_empty_map_names_nobody() -> None:
    resolver = EntityResolver(SCOPE, _entities(("jinho", "이진호", ())))

    assert read_presence(None, resolver) is None
    assert read_presence(["jinho"], resolver) is None
    assert read_presence({}, resolver) == {}
