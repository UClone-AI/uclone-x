"""Continuity after writing is a query of the story's store, over one reading per scene.

Each scene is read once (`enrich_scene`): its facts become proposals for what is new or
changed, and the same facts are checked (`findings_from`) against the store's approved-and-
proposed view at the scene's end. Nothing is approved in these stories, as in a run that
only ever says "네": the cast and the clock are pending proposals, and so is every death,
holder and kinship a scene writes. What these pin, in order of what it would cost to get
wrong:

* **#2014's live cases are caught by the query**, from the scenes qwen3:8b wrote on
  2026-09-29 (`tests/fixtures/story_yes_only_live`): the clock that changes hands between
  scenes with no word of how, and 이진호 dead in one scene and acting in a later one.
* **A handover the scene states, and a death of someone else, are not findings**: the
  false positive of the closed prose reader ("진호는 살아남았지만 하린이 죽었다") does
  not fire, because the reading names who died.
* **A family relation an earlier scene wrote holds a later scene to it.**

The model's reading is the input: a fake gives each scene's facts, as qwen3:8b read them
or as a reader would, each with a quote from the real scene.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from uclone_x.knowledge.extraction import ALIAS_MARKER, parse_extraction, verify_facts
from uclone_x_story.continuity import Finding, checked_store, findings_from
from uclone_x_story.enrich import PRESENCE_MARKER, Enrichment, enrich_scene
from uclone_x_story.library import StoryError, StoryLibrary
from uclone_x_story.proposals import apply_proposal, preview_proposal, reject_proposal
from uclone_x_story.schemas import Outline, Proposal, dump_file
from uclone_x_story.timeline import place_scenes
from uclone_x_story.work import StoryWork

ROOM = "room_a"
LIVE = json.loads(
    (
        Path(__file__).resolve().parents[1] / "fixtures" / "story_yes_only_live" / "scenes.json"
    ).read_text("utf-8")
)["scenes"]


class Reading:
    """A fake model: one reply per scene reading, in turn; the alias question answered
    with `names` (every candidate a name by default), the words it was asked of kept."""

    def __init__(self, replies: list[dict[str, Any]], names: bool = True) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.names = names
        self.asked: list[str] = []

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        del system, temperature, max_tokens
        if prompt.startswith(PRESENCE_MARKER):
            return "{}"  # not answered: the reading's own labels decide (#2317)
        if prompt.startswith(ALIAS_MARKER):
            words = re.findall(r'^\d+\. .*? -- words: "(.*?)"', prompt, re.MULTILINE)
            self.asked += words
            return json.dumps({str(n): self.names for n in range(1, len(words) + 1)})
        self.prompts.append(prompt)
        reply: dict[str, Any] = self.replies.pop(0) if self.replies else {"facts": [], "new": []}
        return json.dumps(reply, ensure_ascii=False)


def said(subject: str, key: str, value: str, quote: str, **more: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "key": key,
        "value": value,
        "quote": quote,
        "time": "now",
        "polarity": "asserted",
        **more,
    }


def _story(
    workspace: Path,
    chapters: dict[str, list[str]],
    pending: list[tuple[str, dict[str, Any]]],
) -> StoryWork:
    """A story with these scenes, nothing written, and `pending` new entries as proposals
    (kind, entry): a run that never approved anything."""
    library = StoryLibrary(workspace)
    record = library.create("Live", conversation_id=ROOM)
    work = StoryWork(library, record.story_id)
    outline = Outline.model_validate(
        {
            "chapters": [
                {
                    "id": chapter,
                    "title": chapter,
                    "scenes": [{"id": f"{chapter}.{s}", "title": f"{chapter}.{s}"} for s in scenes],
                }
                for chapter, scenes in chapters.items()
            ]
        }
    )
    work.write("outline.yaml", dump_file(outline), room_id=ROOM, expected_digest=None)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for kind, entry in pending:
        work.add_proposal(
            Proposal.model_validate(
                {
                    "id": "p000",
                    "kind": kind,
                    "entry_id": entry["id"],
                    "change": {"new_entry": entry},
                    "proposed_at": now,
                    "room_id": ROOM,
                }
            ),
            room_id=ROOM,
        )
    return work


async def _write_and_check(
    work: StoryWork, scenes: dict[str, str], readings: list[dict[str, Any]]
) -> tuple[list[Finding], dict[str, Enrichment]]:
    """Each scene written and read once, in order; then each checked against the store, as
    `story_start` does after writing a chapter."""
    model = Reading(readings)
    read: dict[str, Enrichment] = {}
    for scene_id, text in scenes.items():
        work.write_scene(
            scene_id, text, room_id=ROOM, expected_digest=None, agent_id=None, turn_index=None
        )
        read[scene_id] = await enrich_scene(
            work, model=model, scene_id=scene_id, text=text, room_id=ROOM
        )
    assert len(model.prompts) == len(scenes)  # one reading per scene, and no other call
    outline, _ = work.require_outline()
    store = checked_store(work.story_id, outline, work.codex(), [p for p, _ in work.proposals()[0]])
    findings: list[Finding] = []
    for scene_id, text in scenes.items():
        findings += findings_from(
            store=store,
            outline=outline,
            axioms=[],
            axioms_are_defaults=False,
            scene_id=scene_id,
            text=text,
            facts=read[scene_id].facts,
        )
    return findings, read


JINHO = {"id": "ijinho", "name": "이진호", "state": {"status": "alive"}}
ELDER = {"id": "maeul_jangro", "name": "마을 장로", "state": {"status": "alive"}}
CLOCK = {"id": "jogaknan_sigye", "name": "조각난 시계"}
LIVE_CAST = [("characters", JINHO), ("characters", ELDER), ("items", CLOCK)]


class TestTheLiveCasesAreCaughtByTheQuery:
    """#2014: what the graph grader found in the live run and the tool missed."""

    async def test_the_clock_changing_hands_between_scenes_is_caught(self, tmp_path: Path) -> None:
        """The elder holds the clock in the first scene; 이진호 holds it two scenes on, and
        no scene says it changed hands.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: return "possesses", [*held, item]
        Becomes: return None
        """
        work = _story(tmp_path, {"ch01": ["s01", "s02", "s03"]}, LIVE_CAST)
        scenes = {s: LIVE[s] for s in ("ch01.s01", "ch01.s02", "ch01.s03")}
        findings, read = await _write_and_check(
            work,
            scenes,
            [
                {
                    "facts": [
                        said(
                            "마을 장로",
                            "possesses",
                            CLOCK["id"],
                            "조각난 시계를 가지고 있는 장로의 유산을 찾기 위해",
                        )
                    ]
                },
                {
                    "facts": [
                        said("ijinho", "status", "alive", "이진호는 무덤 앞에 서서 숨을 참았다")
                    ]
                },
                {
                    "facts": [
                        said(
                            "ijinho",
                            "possesses",
                            "조각난 시계",
                            "이진호는 무덤에서 빼앗아온 조각난 시계를 다시 집어 들었다",
                        )
                    ]
                },
            ],
        )
        # The first scene's holder is a proposal; the third scene's is not proposed.
        [gained] = read["ch01.s01"].proposed
        assert gained.entry_id == "maeul_jangro"
        assert read["ch01.s03"].proposed == []
        [flip] = findings
        assert flip.scene_id == "ch01.s03" and flip.kind == "holder_changed"
        assert flip.note == (
            "‘ch01.s03’ 장면의 “이진호는 무덤에서 빼앗아온 조각난 시계를 다시 집어 들었다” — "
            "‘조각난 시계’는 앞의 ‘ch01.s01’ 장면에서 마을 장로가 가지고 있었는데, "
            "이 장면에서는 건네받았다는 말 없이 이진호가 가지고 있습니다."
        )

    async def test_jinho_dead_and_then_acting_is_caught(self, tmp_path: Path) -> None:
        """The grader read 이진호's possession as his death, and him at work in the grave in
        the next chapter. The live scene says the possession through pronouns only ("그는
        자신의 정체성을 잃고"), which a proposal is never made from (a quote must name whom
        it is about); so, as #2022 seeded the elder's death, a line that names him is put
        at the end of that scene.

        Killed by: examples/story/src/uclone_x_story/continuity.py :: and folded(there[2]) == "dead"
        Becomes: and folded(there[2]) == "missing"
        """
        died = "그날 밤 이진호는 무덤 앞에서 숨을 거두었다"
        work = _story(tmp_path, {"ch02": ["s03"], "ch03": ["s01"]}, LIVE_CAST)
        scenes = {"ch02.s03": f"{LIVE['ch02.s03']}\n\n{died}.", "ch03.s01": LIVE["ch03.s01"]}
        acting = "이진호는 무덤의 어두운 공간에서 시계를 손으로 들어 올리며"
        findings, read = await _write_and_check(
            work,
            scenes,
            [
                {
                    "facts": [
                        said(
                            "이진호", "status", "alive", "이진호는 마을을 떠나야 한다는 생각을 했다"
                        ),
                        said("이진호", "status", "dead", died),
                    ]
                },
                {"facts": [said("이진호", "status", "alive", acting)]},
            ],
        )
        [death] = read["ch02.s03"].proposed
        assert death.change.progression is not None
        assert death.change.progression.set == {"status": "dead"}
        [back] = findings
        assert back.scene_id == "ch03.s01" and back.kind == "dead_then_acting"
        assert back.note == (
            f"‘ch03.s01’ 장면의 “{acting}” — 이진호는 앞의 ‘ch02.s03’ 장면에서 죽었는데, "
            "이 장면에서 다시 움직입니다."
        )

    async def test_a_pronoun_only_death_is_read_and_not_proposed(self, tmp_path: Path) -> None:
        """The live possession line as it was written: the check still reads it in its own
        scene, but nothing is proposed from a quote that names no one, so nothing later is
        held to it. The precision rule of the codex growth, kept (`subject_not_named`)."""
        work = _story(tmp_path, {"ch02": ["s03"]}, LIVE_CAST)
        pronoun = "그는 자신의 정체성을 잃고, 마을의 힘에 점령당했다"
        _, read = await _write_and_check(
            work,
            {"ch02.s03": LIVE["ch02.s03"]},
            [{"facts": [said("이진호", "status", "dead", pronoun)]}],
        )
        assert read["ch02.s03"].proposed == []
        assert read["ch02.s03"].dropped == {"subject_not_named": 1}
        assert [r.fact.quote for r in read["ch02.s03"].facts] == [pronoun]

    async def test_the_seeded_second_holder_is_caught_and_the_seeded_handover_is_not(
        self, tmp_path: Path
    ) -> None:
        """#2022's seeds on the same run: the elder carries the clock 이진호 held in the scene
        before -- once with no word of how, once handed over.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: return "possesses", [v for v in held if v != item]
        Becomes: return None
        """
        lifted = "이진호는 무덤의 어두운 공간에서 시계를 손으로 들어 올리며"
        # The words naming the clock, as the reading is asked for them (#2249): the quote
        # writes only "시계", not the whole "조각난 시계".
        jinho_holds = said("ijinho", "possesses", CLOCK["id"], lifted, object_named="시계")
        anchor = "마을 장로는 그를 지켜보며"
        seeds = {
            "second_owner": (
                "마을 장로는 조각난 시계를 품에 넣고 다녔다. ",
                [
                    said(
                        "마을 장로",
                        "possesses",
                        CLOCK["id"],
                        "마을 장로는 조각난 시계를 품에 넣고 다녔다",
                    )
                ],
            ),
            "handover": (
                "이진호가 남긴 조각난 시계는 마을 장로에게 건네졌고, 장로는 그 시계를 품에 넣고 다녔다. ",
                [
                    said(
                        "ijinho",
                        "loses",
                        CLOCK["id"],
                        "이진호가 남긴 조각난 시계는 마을 장로에게 건네졌고",
                    ),
                    said(
                        "마을 장로",
                        "possesses",
                        CLOCK["id"],
                        "장로는 그 시계를 품에 넣고 다녔다",
                        object_named="시계",
                    ),
                ],
            ),
        }
        caught: dict[str, list[str]] = {}
        for name, (seed, facts) in seeds.items():
            work = _story(tmp_path / name, {"ch03": ["s01", "s02"]}, LIVE_CAST)
            scenes = {
                "ch03.s01": LIVE["ch03.s01"],
                "ch03.s02": LIVE["ch03.s02"].replace(anchor, seed + anchor, 1),
            }
            findings, _ = await _write_and_check(
                work, scenes, [{"facts": [jinho_holds]}, {"facts": facts}]
            )
            caught[name] = [f.kind for f in findings]
        assert caught == {"second_owner": ["holder_changed"], "handover": []}


class TestWhatIsNotAFinding:
    async def test_one_who_survives_beside_one_who_dies_is_not_called_dead(
        self, tmp_path: Path
    ) -> None:
        """The false positive of #2022's review: "진호는 살아남았지만 하린이 죽었다", and 진호
        later. The reading names who died, so nothing reads a death into 진호.

        Killed by: examples/story/src/uclone_x_story/continuity.py :: if not fact.checked or store.kind_of(read.subject) != "characters":
        Becomes: if store.kind_of(read.subject) != "characters":
        """
        jinho = {"id": "jinho", "name": "진호", "state": {"status": "alive"}}
        harin = {"id": "harin", "name": "하린", "state": {"status": "alive"}}
        work = _story(
            tmp_path, {"ch01": ["s01", "s02"]}, [("characters", jinho), ("characters", harin)]
        )
        scenes = {
            "ch01.s01": "다리가 무너졌다. 진호는 살아남았지만 하린이 죽었다.",
            "ch01.s02": "다음 날 아침, 진호는 다시 길을 나섰다. 하린은 꿈에서 웃고 있었다.",
        }
        findings, read = await _write_and_check(
            work,
            scenes,
            [
                {
                    "facts": [
                        said("진호", "status", "alive", "진호는 살아남았지만"),
                        said("하린", "status", "dead", "하린이 죽었다"),
                    ]
                },
                {
                    "facts": [
                        said("진호", "status", "alive", "진호는 다시 길을 나섰다"),
                        # A dream is not her present: read, and not checked.
                        {
                            **said("하린", "status", "alive", "하린은 꿈에서 웃고 있었다"),
                            "time": "past",
                        },
                    ]
                },
            ],
        )
        assert findings == []
        [death] = read["ch01.s01"].proposed
        assert death.entry_id == "harin"


class TestAFamilyRelationHoldsLaterScenes:
    async def test_a_relation_an_earlier_scene_wrote_holds_a_later_one(
        self, tmp_path: Path
    ) -> None:
        """도윤 is 하린's father in the first scene (a proposal), and her husband in the next.

        Killed by: examples/story/src/uclone_x_story/continuity.py :: keys |= {f.predicate for f in facts if f.predicate.startswith(RELATION_PREFIX)}
        Becomes: pass
        """
        doyun = {"id": "doyun", "name": "도윤", "state": {"status": "alive"}}
        harin = {"id": "harin", "name": "하린", "state": {"status": "alive"}}
        work = _story(
            tmp_path, {"ch01": ["s01", "s02"]}, [("characters", doyun), ("characters", harin)]
        )
        father, husband = "하린의 아버지 도윤은 등대를 지켰다", "도윤은 하린의 남편이었다"
        scenes = {"ch01.s01": f"{father}.", "ch01.s02": f"폭풍이 왔다. {husband}."}
        findings, read = await _write_and_check(
            work,
            scenes,
            [
                {"facts": [said("도윤", "family", "father", father, of="하린")]},
                {"facts": [said("도윤", "family", "husband", husband, of="하린")]},
            ],
        )
        [kin] = read["ch01.s01"].proposed
        assert kin.change.progression is not None
        assert kin.change.progression.relations == {"harin": "parent"}
        assert read["ch01.s02"].proposed == []  # it contradicts: the check's to report
        [finding] = findings
        assert finding.scene_id == "ch01.s02"
        assert finding.note == (
            f"‘ch01.s02’ 장면의 “{husband}” — 이 문장에서는 도윤이 하린의 배우자인 것으로 "
            "나오지만, 앞의 ‘ch01.s01’ 장면에서는 도윤이 하린의 부모인 것으로 나왔습니다."
        )
        for said_text in (finding.note,):
            assert "doyun" not in said_text and "relation" not in said_text


def test_the_reading_is_verified_once_in_knowledge() -> None:
    """A fact whose quote is not in the scene, or whose name is no one's, never reaches the
    check; a status said twice is kept where the scene says it last."""
    from uclone_x.knowledge.models import Entity
    from uclone_x.knowledge.resolver import EntityResolver

    text = "이진호는 걸었다. 그는 쓰러졌다. 이진호는 다시 일어섰다."
    resolver = EntityResolver("story:s", [Entity("ijinho", "story:s", "characters", "이진호")])
    facts = parse_extraction(
        json.dumps(
            {
                "facts": [
                    said("이진호", "status", "alive", "이진호는 걸었다"),
                    said("이진호", "status", "dead", "그는 쓰러졌다"),
                    said("ijinho", "status", "alive", "이진호는 다시 일어섰다"),
                    said("이진호", "status", "dead", "이진호는 묻혔다"),
                    said("박씨", "status", "dead", "그는 쓰러졌다"),
                ]
            },
            ensure_ascii=False,
        )
    )
    verified = verify_facts(facts, text, resolver)
    assert verified.dropped == {"duplicate": 1, "quote_not_in_text": 1, "not_an_entity": 1}
    alive = [r for r in verified.kept if r.fact.value == "alive"]
    assert [r.fact.quote for r in alive] == ["이진호는 다시 일어섰다"]


KALRIN = {"id": "kalrin_deu_raiteu", "name": "칼린 드 라이트", "state": {"status": "alive"}}
MORAS = {"id": "moraseu", "name": "모라스", "state": {"status": "alive"}}
#: The item's prose name recorded as its alias, as the cast stage records the names the prose
#: will use (#2223): a fact about an item stands on a quote that names it (#2238).
RELIC = {"id": "yumul", "name": "유물", "aliases": ["relic"]}
LATIN_CAST = [("characters", KALRIN), ("characters", MORAS), ("items", RELIC)]


class TestACastInHangulAndProseInLatinLetters:
    """#2217, from #2216's live run: the cast was proposed in Hangul and the prose wrote the
    same names in Latin letters, so every change was dropped as ``subject_not_named``. The
    reading model is shown each entry's id with its names and answers with the id and the
    name the quote uses; that name becomes a pending alias, and the scene's changes resolve
    through it. Nothing in code maps one script to the other."""

    async def test_a_handover_and_a_death_resolve_and_dead_then_acting_is_caught(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/enrich.py :: n for n in (item.entry.name, *item.entry.aliases, *self.called.get(subject, ())) if n
        Becomes: n for n in (item.entry.name, *item.entry.aliases) if n
        """
        work = _story(tmp_path, {"ch01": ["s01", "s02", "s03"]}, LATIN_CAST)
        held = "Moras held the old relic close to his chest"
        gave = "At the gate, Moras gave the old relic to Kalrin"
        took = "Kalrin took the relic and ran"
        died = "An arrow struck Kalrin in the back, and Kalrin fell dead in the snow"
        acting = "Kalrin walked into the tavern and ordered a drink"
        scenes = {
            "ch01.s01": f"{held}.",
            "ch01.s02": f"{gave}. {took}. {died}.",
            "ch01.s03": f"{acting}.",
        }
        findings, read = await _write_and_check(
            work,
            scenes,
            [
                {"facts": [said("moraseu", "possesses", "yumul", held, named="Moras")]},
                {
                    "facts": [
                        said("moraseu", "loses", "yumul", gave, named="Moras"),
                        said("kalrin_deu_raiteu", "possesses", "yumul", took, named="Kalrin"),
                        said("kalrin_deu_raiteu", "status", "dead", died, named="Kalrin"),
                    ]
                },
                {"facts": [said("kalrin_deu_raiteu", "status", "alive", acting, named="Kalrin")]},
            ],
        )

        def changes(scene_id: str) -> list[tuple[str, Any]]:
            out: list[tuple[str, Any]] = []
            for p in read[scene_id].proposed:
                progression = p.change.progression
                out.append((p.entry_id, p.change.aliases or (progression and progression.set)))
            return out

        assert changes("ch01.s01") == [
            ("moraseu", ["Moras"]),
            ("moraseu", {"possesses": ["yumul"]}),
        ]
        assert changes("ch01.s02") == [
            ("kalrin_deu_raiteu", ["Kalrin"]),
            ("moraseu", {"possesses": []}),
            ("kalrin_deu_raiteu", {"possesses": ["yumul"]}),
            ("kalrin_deu_raiteu", {"status": "dead"}),
        ]
        # The pending alias is one of Kalrin's names now: the next reading shows it, and
        # proposes it no more.
        assert read["ch01.s03"].proposed == []
        [back] = findings  # the stated handover is no finding
        assert back.scene_id == "ch01.s03" and back.kind == "dead_then_acting"
        assert "칼린 드 라이트는 앞의 ‘ch01.s02’ 장면에서 죽었는데" in back.note

    async def test_words_the_alias_question_does_not_confirm_are_no_name(
        self, tmp_path: Path
    ) -> None:
        """The reading model also gives a pronoun as the name (qwen3:8b, 2026-10-08): asked
        whether it is a name and told no, nothing is proposed from it, and a quote that
        names the subject by it alone still names no one.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: confirmed = {key for index, key in enumerate(asked) if index in yes}
        Becomes: confirmed = set(asked)
        """
        work = _story(tmp_path, {"ch01": ["s01"]}, LATIN_CAST)
        died = "She fell dead in the snow"
        model = Reading(
            [{"facts": [said("kalrin_deu_raiteu", "status", "dead", died, named="She")]}],
            names=False,
        )
        work.write_scene(
            "ch01.s01",
            f"{died}.",
            room_id=ROOM,
            expected_digest=None,
            agent_id=None,
            turn_index=None,
        )
        read = await enrich_scene(
            work, model=model, scene_id="ch01.s01", text=f"{died}.", room_id=ROOM
        )
        assert model.asked == ["She"]
        assert read.proposed == []
        assert read.dropped == {"alias_not_confirmed": 1, "subject_not_named": 1}

    async def test_the_next_reading_lists_a_pending_alias_and_an_approved_one_is_in_the_file(
        self, tmp_path: Path
    ) -> None:
        """A pending alias is one of the entry's names for the next reading, and approving
        it writes it into the entry file and leaves the death proposed beside it approvable.

        Killed by: examples/story/src/uclone_x_story/codex_store.py :: item = with_pending_aliases(CodexItem(kind=proposal.kind, entry=entry), called)
        Becomes: item = CodexItem(kind=proposal.kind, entry=entry)
        Killed by: examples/story/src/uclone_x_story/proposals.py :: if _names_only(proposal.change):
        Becomes: if not _names_only(proposal.change):
        """
        work = _story(tmp_path, {"ch01": ["s01", "s02"]}, LATIN_CAST)
        died = "Kalrin fell dead in the snow"
        model = Reading(
            [{"facts": [said("kalrin_deu_raiteu", "status", "dead", died, named="Kalrin")]}]
        )
        for scene_id, text in (("ch01.s01", f"{died}."), ("ch01.s02", "Snow fell.")):
            work.write_scene(
                scene_id, text, room_id=ROOM, expected_digest=None, agent_id=None, turn_index=None
            )
            await enrich_scene(work, model=model, scene_id=scene_id, text=text, room_id=ROOM)
        assert "- kalrin_deu_raiteu: 칼린 드 라이트, Kalrin" in model.prompts[1]
        pending = {p.id: p for p, _ in work.proposals()[0] if p.status == "pending"}
        for proposal in pending.values():  # the cast first, then what the scene proposed
            apply_proposal(work, proposal.id, room_id=ROOM, decided_in="story_view")
        found = work.entry("characters", "kalrin_deu_raiteu")
        assert found is not None
        assert found[0].aliases == ["Kalrin"]
        assert found[0].progressions[0].set == {"status": "dead"}


class TestANameAndAChangeFromOneReadingApproveInEitherOrder:
    """#2223 review: a reading proposes a name for an entry beside what happens to it, both
    against the same entry file. Approving the change first rewrote the file, so the name
    was refused as made against an older entry, and the refusal said to reject it -- and a
    rejected name is never read as the entry's again, so the cross-script fix was off for
    that character for good. A name changes nothing else an entry says, so it applies to
    the entry as it is now."""

    DIED = "Kalrin fell dead in the snow"
    ACTING = "Kalrin walked into the tavern and ordered a drink"

    async def _read_the_death(
        self, tmp_path: Path
    ) -> tuple[StoryWork, Reading, Proposal, Proposal]:
        work = _story(tmp_path, {"ch01": ["s01", "s02"]}, LATIN_CAST)
        for cast, _ in work.proposals()[0]:
            apply_proposal(work, cast.id, room_id=ROOM, decided_in="story_view")
        model = Reading(
            [
                {"facts": [said("kalrin_deu_raiteu", "status", "dead", self.DIED, named="Kalrin")]},
                {
                    "facts": [
                        said("kalrin_deu_raiteu", "status", "alive", self.ACTING, named="Kalrin")
                    ]
                },
            ]
        )
        await _write_and_read(work, model, "ch01.s01", f"{self.DIED}.")
        pending = [p for p, _ in work.proposals()[0] if p.status == "pending"]
        [name] = [p for p in pending if p.change.aliases]
        [death] = [p for p in pending if p.change.progression]
        return work, model, name, death

    async def test_the_name_approved_after_the_death_still_names_the_next_scene(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/proposals.py :: if digest != proposal.entry_digest and not _names_only(proposal.change):
        Becomes: if digest != proposal.entry_digest:
        Killed by: examples/story/src/uclone_x_story/proposals.py :: if digest == proposal.entry_digest or _names_only(proposal.change)
        Becomes: if digest == proposal.entry_digest
        """
        work, model, name, death = await self._read_the_death(tmp_path)
        apply_proposal(work, death.id, room_id=ROOM, decided_in="story_view")
        outline, _ = work.require_outline()
        preview = preview_proposal(work, work.proposal(name.id)[0], place_scenes(outline))
        assert preview.blocked is None
        apply_proposal(work, name.id, room_id=ROOM, decided_in="story_view")
        found = work.entry("characters", "kalrin_deu_raiteu")
        assert found is not None
        assert found[0].aliases == ["Kalrin"]
        assert found[0].progressions[0].set == {"status": "dead"}

        read = await _write_and_read(work, model, "ch01.s02", f"{self.ACTING}.")
        # Named by the approved name (no `subject_not_named`), and not proposed as a change:
        # alive after an approved death is what the check reports, not a proposal.
        assert read.dropped == {"contradicts_codex": 1}
        store = checked_store(
            work.story_id, outline, work.codex(), [p for p, _ in work.proposals()[0]]
        )
        [back] = findings_from(
            store=store,
            outline=outline,
            axioms=[],
            axioms_are_defaults=False,
            scene_id="ch01.s02",
            text=f"{self.ACTING}.",
            facts=read.facts,
        )
        assert back.kind == "dead_then_acting"

    async def test_a_name_a_person_rejects_is_not_asked_or_read_again(self, tmp_path: Path) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/enrich.py :: if (subject, folded(name)) in self.rejected:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/enrich.py :: if key not in rejected and key not in asked:
        Becomes: if key not in asked:
        """
        work, model, name, _ = await self._read_the_death(tmp_path)
        reject_proposal(work, name.id, room_id=ROOM, reason=None, decided_in="story_view")
        read = await _write_and_read(work, model, "ch01.s02", f"{self.ACTING}.")
        assert model.asked == ["Kalrin"]  # the first reading's question, and no other
        assert read.proposed == []
        assert read.dropped == {"alias_rejected": 1, "subject_not_named": 1}

    async def test_approving_the_name_moves_only_what_waits_on_the_same_file(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/proposals.py :: or other.entry_digest != before
        Becomes: or False
        Killed by: examples/story/src/uclone_x_story/proposals.py :: other.status != "pending"
        Becomes: False
        """
        work, _, name, death = await self._read_the_death(tmp_path)
        older = _proposal(
            work, {"progression": {"at": "ch01.s01", "set": {"wounded": True}}}, "older"
        )
        refused = _proposal(work, {"progression": {"at": "ch01.s01", "set": {"lost": True}}}, None)
        reject_proposal(work, refused.id, room_id=ROOM, reason=None, decided_in="story_view")
        before = death.entry_digest
        apply_proposal(work, name.id, room_id=ROOM, decided_in="story_view")
        after = {p.id: p.entry_digest for p, _ in work.proposals()[0]}
        assert after[death.id] not in (before, None)
        assert after[older.id] == "older"
        assert after[refused.id] == before
        apply_proposal(work, death.id, room_id=ROOM, decided_in="story_view")


async def _write_and_read(work: StoryWork, model: Reading, scene_id: str, text: str) -> Enrichment:
    work.write_scene(
        scene_id, text, room_id=ROOM, expected_digest=None, agent_id=None, turn_index=None
    )
    return await enrich_scene(work, model=model, scene_id=scene_id, text=text, room_id=ROOM)


def _proposal(work: StoryWork, change: dict[str, Any], digest: str | None) -> Proposal:
    """A pending change to Kalrin, made against the entry file with `digest` (`None`: the
    file as it is now)."""
    found = work.entry("characters", "kalrin_deu_raiteu")
    assert found is not None
    return work.add_proposal(
        Proposal.model_validate(
            {
                "id": "p000",
                "kind": "characters",
                "entry_id": "kalrin_deu_raiteu",
                "change": change,
                "proposed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "room_id": ROOM,
                "entry_digest": digest or found[1],
            }
        ),
        room_id=ROOM,
    )


class TestANamesBoundaries:
    """#2226, from the #2223 reviews: what a name may not do."""

    async def test_a_pending_name_of_an_approved_entry_is_listed_and_not_proposed_again(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/codex_store.py :: items: list[CodexItem] = [with_pending_aliases(item, called) for item in codex.items]
        Becomes: items: list[CodexItem] = list(codex.items)
        """
        work = _story(tmp_path, {"ch01": ["s01", "s02"]}, LATIN_CAST)
        for cast, _ in work.proposals()[0]:
            apply_proposal(work, cast.id, room_id=ROOM, decided_in="story_view")
        died = "Kalrin fell dead in the snow"
        model = Reading(
            [
                {"facts": [said("kalrin_deu_raiteu", "status", "dead", died, named="Kalrin")]},
                {"facts": [said("kalrin_deu_raiteu", "status", "dead", died, named="Kalrin")]},
            ]
        )
        await _write_and_read(work, model, "ch01.s01", f"{died}.")
        read = await _write_and_read(work, model, "ch01.s02", f"{died}.")
        assert "- kalrin_deu_raiteu: 칼린 드 라이트, Kalrin" in model.prompts[1]
        assert [p.change.aliases for p in read.proposed] == []
        assert model.asked == ["Kalrin"]  # asked once, at the first reading

    def test_a_name_the_entry_has_in_another_case_is_not_added_again(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/schemas.py :: known = {fold(n) for n in (entry.name, *have)}
        Becomes: known = set(have) | {entry.name}
        """
        from uclone_x_story.schemas import CharacterEntry, ProposalChange, changed_entry

        entry = CharacterEntry.model_validate({**KALRIN, "aliases": ["Kalrin"]})
        data = changed_entry(entry, ProposalChange(aliases=["kalrin", "KALRIN ", "Kal"]))
        assert data["aliases"] == ["Kalrin", "Kal"]

    async def test_a_name_another_entry_has_is_refused_in_plain_words(self, tmp_path: Path) -> None:
        """Approving it would make the name two entries', which a reading then drops as
        ambiguous with no one told: the preview says so, and approving is refused.

        Killed by: examples/story/src/uclone_x_story/proposals.py :: shared = alias_clash(work, proposal)
        Becomes: shared = None
        Killed by: examples/story/src/uclone_x_story/proposals.py :: clash = alias_clash(work, proposal)
        Becomes: clash = None
        """
        work = _story(tmp_path, {"ch01": ["s01"]}, LATIN_CAST)
        for cast, _ in work.proposals()[0]:
            apply_proposal(work, cast.id, room_id=ROOM, decided_in="story_view")
        name = _proposal(work, {"aliases": ["모라스"]}, None)
        outline, _ = work.require_outline()
        preview = preview_proposal(work, name, place_scenes(outline))
        sentence = (
            "It cannot be approved: '모라스' is already the name of 모라스. Reject it, or "
            "give the name to one entry only."
        )
        assert preview.blocked == sentence
        with pytest.raises(StoryError) as refused:
            apply_proposal(work, name.id, room_id=ROOM, decided_in="story_view")
        assert str(refused.value) == sentence
        found = work.entry("characters", "kalrin_deu_raiteu")
        assert found is not None and found[0].aliases == []

    @pytest.mark.parametrize(
        "change",
        [
            {"aliases": ["Kalrin"], "progression": {"at": "ch01.s01", "set": {"wounded": True}}},
            {"aliases": ["Kalrin"], "visual": {"tags": ["scar"]}},
        ],
        ids=["with a progression", "with a visual change"],
    )
    async def test_a_name_with_another_change_is_still_held_to_the_entry_it_was_made_on(
        self, tmp_path: Path, change: dict[str, Any]
    ) -> None:
        """Only a name alone applies to an entry changed since; a name beside any other
        change is refused as every change is.

        Killed by: examples/story/src/uclone_x_story/proposals.py :: return change.aliases is not None and all(
        Becomes: return change.aliases is not None or all(
        Killed by: examples/story/src/uclone_x_story/proposals.py :: getattr(change, field) is None for field in type(change).model_fields if field != "aliases"
        Becomes: getattr(change, field) is None for field in type(change).model_fields if field not in ("aliases", "visual")
        """
        work = _story(tmp_path, {"ch01": ["s01"]}, LATIN_CAST)
        for cast, _ in work.proposals()[0]:
            apply_proposal(work, cast.id, room_id=ROOM, decided_in="story_view")
        mixed = _proposal(work, change, None)
        other = _proposal(work, {"progression": {"at": "ch01.s01", "set": {"lost": True}}}, None)
        apply_proposal(work, other.id, room_id=ROOM, decided_in="story_view")
        with pytest.raises(StoryError, match="changed after proposal"):
            apply_proposal(work, mixed.id, room_id=ROOM, decided_in="story_view")


ROMANCE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "story_romance_false_findings"
        / "run.json"
    ).read_text("utf-8")
)


class _Replayed:
    """The run's own scene readings, as the model gave them, one per scene in turn."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        del system, temperature, max_tokens
        if prompt.startswith((ALIAS_MARKER, PRESENCE_MARKER)):
            return "{}"  # the run asked neither question
        return self.replies.pop(0)


class TestTheAcceptanceRunsFalseFindings:
    """#2014's acceptance run (qwen3:8b on main 607edf8a, 2026-10-11): a romance with no
    death and no item handed over, in which the store query made 9 findings, every one
    false. Replayed from the run's own scenes and readings, it makes none (#2238, #2239)."""

    async def test_the_run_replayed_makes_no_finding(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/knowledge/extraction.py :: if any(not _anchored(resolver, entity, fact) for entity in objects):
        Becomes: if False:
        """
        work = _story(tmp_path, ROMANCE["chapters"], [tuple(c) for c in ROMANCE["cast"]])
        model = _Replayed(ROMANCE["replies"])
        read: dict[str, Enrichment] = {}
        for scene_id, text in ROMANCE["scenes"].items():
            work.write_scene(
                scene_id, text, room_id=ROOM, expected_digest=None, agent_id=None, turn_index=None
            )
            read[scene_id] = await enrich_scene(
                work, model=model, scene_id=scene_id, text=text, room_id=ROOM
            )
        outline, _ = work.require_outline()
        store = checked_store(
            work.story_id, outline, work.codex(), [p for p, _ in work.proposals()[0]]
        )
        found = [
            finding
            for scene_id, text in ROMANCE["scenes"].items()
            for finding in findings_from(
                store=store,
                outline=outline,
                axioms=[],
                axioms_are_defaults=False,
                scene_id=scene_id,
                text=text,
                facts=read[scene_id].facts,
            )
        ]
        assert len(ROMANCE["false_findings"]) == 9  # what the run reported
        assert found == []

    def test_a_quote_that_shows_no_item_holds_none(self) -> None:
        """The two quotes #2238 names: a line about 에이든, and one about 루이스, neither
        showing the item the reading said each held.

        Killed by: src/uclone_x/knowledge/extraction.py :: if found.kind == "existing" and found.entity_id != subject:
        Becomes: if False:
        """
        cast = _cast_entities()
        said = [
            "에이든은 손바닥을 탁 치며 말했다.",
            "루이스는 포도원의 풍경을 배경으로 사라에게 다가섰다.",
        ]
        verified = verify_facts(
            [
                _read_fact("eideun", "possesses", "도시 중심가의 키", said[0]),
                _read_fact("ruiseu", "possesses", "포도원의 편지", said[1]),
            ],
            " ".join(said),
            cast,
        )
        assert verified.kept == []
        assert verified.dropped == {"object_not_in_quote": 2}

    def test_words_that_name_another_entity_drop_the_fact_though_the_quote_lacks_them(
        self,
    ) -> None:
        """ch04.s02: a fact with `subject: sara` and `named: 에이든`, its quote naming
        neither, made 사라 male beside female (#2239).

        Killed by: src/uclone_x/knowledge/extraction.py :: if not fold(named):
        Becomes: if not fold(named) or fold(named) not in fold(fact.quote):
        """
        quote = "창가에 앉아 커피를 마시며 그를 바라보는 그녀의 마음은 어쩔 수 없이 흔들렸다."
        verified = verify_facts(
            [_read_fact("sara", "gender", "male", quote, named="에이든")], quote, _cast_entities()
        )
        assert verified.kept == []
        assert verified.dropped == {"named_other": 1}


def _cast_entities() -> Any:
    from uclone_x.knowledge.models import Entity
    from uclone_x.knowledge.resolver import EntityResolver

    entities = [
        Entity(
            entry["id"],
            "story:s",
            kind,
            entry["name"],
            tuple(entry.get("aliases", ())),
        )
        for kind, entry in ROMANCE["cast"]
    ]
    return EntityResolver("story:s", entities)


def _read_fact(subject: str, key: str, value: str, quote: str, named: str = "") -> Any:
    from uclone_x.knowledge.extraction import ExtractedFact

    return ExtractedFact(subject=subject, key=key, value=value, quote=quote, named=named)


async def test_the_reading_is_asked_for_the_words_that_name_a_facts_object(
    tmp_path: Path,
) -> None:
    """The core holds a fact's object to the words of its quote that name it (#2249); the
    story's reading is asked for them, or a quote naming the clock "시계" would not hold it.

    Killed by: examples/story/src/uclone_x_story/enrich.py :: "facts": [{{"subject": "<id>", "named": "<name in the quote>", "key": "<key>", "value": "<value>", "object_named": "<words in the quote>", "quote": "<exact words>", "time": "now", "polarity": "asserted"}}],
    Becomes: "facts": [{{"subject": "<id>", "named": "<name in the quote>", "key": "<key>", "value": "<value>", "quote": "<exact words>", "time": "now", "polarity": "asserted"}}],
    """
    work = _story(tmp_path, {"ch01": ["s01"]}, LIVE_CAST)
    model = Reading([{"facts": []}])
    await _write_and_read(work, model, "ch01.s01", "이진호는 시계를 품에 넣었다.")
    assert '"object_named": "<words in the quote>"' in model.prompts[0]
