# pyright: reportPrivateUsage=false
"""Stage 3 (Narrative Spine and act-by-act Chapter Outline generation and revision)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from uclone_x.knowledge.fold import bigram_overlap
from uclone_x_story.arc import (
    Arc,
    Spine,
    arc_for,
    chapter_problems,
    chapter_text,
    problem_reasons,
)
from uclone_x_story.schemas import Outline
from uclone_x_story.skill_data import StructureTemplate
from uclone_x_story.start_cast import _cast_ids
from uclone_x_story.start_premise import _feedback_with_brief
from uclone_x_story.start_state import (
    MAX_CHAPTERS,
    MAX_SCENES_PER_CHAPTER,
    SCENES_PER_CHAPTER,
    PremiseEmbedder,
    _arc_of,
    _ask_json,
    _brief_lines,
    _ids,
    _items,
    _RunBase,
    _text,
    _texts,
    cosine,
    scenes_for,
    structure_for,
)

#: Cosine similarity threshold above which two multi-beat scene summaries are repetitive.
MAX_SCENE_COSINE = 0.88
#: Character-bigram Jaccard threshold above which two scene bodies are repetitive.
MAX_SCENE_BIGRAM = 0.58
#: Minimum characters for non-trivial scene body comparison (leaves tiny test stubs alone).
MIN_SCENE_COMPARE_CHARS = 40

#: The first words of the call that plans what a story turns on, before its chapters.
#: The first words of the call that revises the outline by the person's feedback.
REVISE_MARK = "Revise this story outline as the person asks"


def _chapter_dicts(outline: Outline, stakes: Sequence[str]) -> list[dict[str, Any]]:
    """The outline's chapters as the model writes them: names, not ids."""
    return [
        {
            "title": chapter.title,
            **({"stakes": stakes[n]} if n < len(stakes) and stakes[n] else {}),
            "scenes": [
                {"title": s.title, "summary": s.summary, "beats": list(s.beats)}
                for s in chapter.scenes
            ],
        }
        for n, chapter in enumerate(outline.chapters)
    ]


def _outline_json(outline: Outline, names: Mapping[str, str], stakes: Sequence[str]) -> str:
    """The outline as JSON the model can revise, with the cast by name."""
    chapters = _chapter_dicts(outline, stakes)
    for chapter, kept in zip(chapters, outline.chapters, strict=True):
        for scene, source in zip(chapter["scenes"], kept.scenes, strict=True):
            scene["characters"] = [names[i] for i in source.characters if i in names]
            scene["places"] = [names[i] for i in source.places if i in names]
    return json.dumps({"chapters": chapters}, ensure_ascii=False, indent=1)


SPINE_MARK = "Before any chapter is outlined, decide what this story turns on"
#: The first words of the call that outlines one chapter, as in "... chapter 2 of 3".
CHAPTER_PROMPT_MARK = "Outline only chapter"

_SCENE_SHAPE = (
    "Every scene has a title, a one-sentence summary of what happens, 2 to 4 beats "
    "(concrete physical actions or discoveries, in order; never repeat a scene's summary, "
    "realization, or beats within the chapter or from an earlier chapter), "
    "and the names of the characters and places in it, "
    "exactly as listed above. Work every 'must include' element into the story."
)


def _scene_body_text(sc: Mapping[str, Any]) -> str:
    summary = _text(sc.get("summary"))
    beats = _texts(sc.get("beats"))
    return "\n".join([*([summary] if summary else ()), *beats]).strip()


async def repetitive_scenes(
    scenes: Sequence[Mapping[str, Any]],
    prior_scenes: Sequence[Mapping[str, Any]] = (),
    embedder: PremiseEmbedder | None = None,
) -> list[tuple[str, str]]:
    """Pairs `(earlier_title, current_title)` where a scene in `scenes` semantically or
    lexically repeats an earlier scene (in `prior_scenes` or earlier in `scenes`).

    Uses `embedder` (`MAX_SCENE_COSINE`) when available and falls back to `bigram_overlap`
    (`MAX_SCENE_BIGRAM`). Only scenes with at least `MIN_SCENE_COMPARE_CHARS` characters of
    summary + beats are compared.
    """
    prior_items = [
        (scene_title(_text(s.get("title")), _text(s.get("summary"))) or f"scene-{idx + 1}", body)
        for idx, s in enumerate(prior_scenes)
        if len(body := _scene_body_text(s)) >= MIN_SCENE_COMPARE_CHARS
    ]
    cur_items = [
        (scene_title(_text(s.get("title")), _text(s.get("summary"))) or f"scene-{idx + 1}", body)
        for idx, s in enumerate(scenes)
        if len(body := _scene_body_text(s)) >= MIN_SCENE_COMPARE_CHARS
    ]
    if not cur_items or len(prior_items) + len(cur_items) < 2:
        return []
    all_items = [*prior_items, *cur_items]
    texts = [body for _, body in all_items]
    vectors: Sequence[Sequence[float]] | None = None
    if embedder is not None:
        try:
            embedded = await embedder.embed(texts)
            if len(embedded) == len(texts):
                vectors = embedded
        except Exception:  # noqa: BLE001 - fallback to bigram overlap
            vectors = None

    pairs: list[tuple[str, str]] = []
    offset = len(prior_items)
    for cur_idx, (cur_title, cur_body) in enumerate(cur_items):
        global_i = offset + cur_idx
        for global_j in range(global_i):
            prev_title, prev_body = all_items[global_j]
            is_rep = bigram_overlap(prev_body, cur_body) >= MAX_SCENE_BIGRAM
            if not is_rep and vectors is not None:
                is_rep = cosine(vectors[global_j], vectors[global_i]) >= MAX_SCENE_COSINE
            if is_rep:
                pairs.append((prev_title, cur_title))
                break
    return pairs


def _spine_prompt(title: str, count: int, arc: Arc) -> str:
    return (
        f"The story is built on the structure '{title}', {count} chapters.\n"
        f"{SPINE_MARK}. Write each in one sentence, with THIS premise and cast:\n"
        "- central_conflict: who wants what, against whom or what.\n"
        "- twist: a turn the reader does not see coming, which changes what the conflict "
        f"is about. The reader learns it in chapter {arc.twist_chapter}.\n"
        "- clue: a small, concrete thing an earlier chapter can show that points at the "
        "twist without stating it.\n"
        f"- resolution: how chapter {arc.climax_chapter} decides the conflict, and what it "
        "costs.\n"
        'Reply as JSON: {"central_conflict": ..., "twist": ..., "clue": ..., '
        '"resolution": ...}'
    )


def _has_scenes(reply: Mapping[str, Any] | None) -> bool:
    """Whether a chapter reply holds at least one scene the outline can keep."""
    if reply is None:
        return False
    scenes = [cast(dict[str, Any], s) for s in _items(reply.get("scenes")) if isinstance(s, dict)]
    return any(_text(s.get("title")) or _text(s.get("summary")) for s in scenes)


def _chapters(found: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The chapter objects of an outline reply, in order."""
    raw = found.get("chapters")
    items = cast(list[object], raw) if isinstance(raw, list) else []
    return [cast(dict[str, Any], c) for c in items if isinstance(c, dict)]


def outline_from(
    data: Mapping[str, Any],
    cast_lines: Sequence[Mapping[str, Any]],
    *,
    acts: Sequence[str] = (),
) -> Outline:
    """An outline from the model's JSON: ids by code, names mapped to the cast's ids.

    With `acts`, the chapters are the structure's acts in order: the n-th chapter kept is
    act n, and chapters past the last act are dropped.
    """
    by_name = _cast_ids(cast_lines)
    chapters: list[dict[str, Any]] = []
    raw_chapters = data.get("chapters")
    items = cast(list[object], raw_chapters) if isinstance(raw_chapters, list) else []
    for chapter in items[: len(acts) or MAX_CHAPTERS]:
        if not isinstance(chapter, dict):
            continue
        ch = cast(dict[str, Any], chapter)
        number = len(chapters) + 1
        scenes = _scene_dicts(ch.get("scenes"), f"ch{number:02d}", by_name)
        if not scenes:
            continue
        chapters.append(
            {
                "id": f"ch{number:02d}",
                "title": _text(ch.get("title")) or f"{number}장",
                **({"act": acts[number - 1]} if number <= len(acts) else {}),
                "scenes": scenes,
            }
        )
    return Outline.model_validate({"chapters": chapters})


#: A title that ends like this is a clause cut short, not a name: a comma, a sentence's
#: end, or a Korean connective ending ("느끼며", "알았지만").
_CLAUSE_END = re.compile(r"(?:[,.!?…:;]|(?:며|고|서|지만|는데|면서|으나|니까|다가|하여))$")
#: The longest a scene title may run before it reads as a sentence.
MAX_SCENE_TITLE = 20
#: "X의 Y" -- the shape most scene titles have ("장로의 속삭임"); X is not a pronoun.
_OF_PHRASE = re.compile(r"(?<!\S)([^\s,.]+)의\s+([^\s,.]+)")
_PRONOUN_OWNERS = frozenset({"자신", "자기", "그", "그녀", "나", "너", "우리", "그들", "저"})
#: Particles taken off the end of the phrase's noun, longest first.
_PARTICLES = tuple("에서 에게 으로 을 를 은 는 이 가 에 로 와 과 도 만".split())


def _clause_title(title: str, summary: str) -> bool:
    """Whether the model's title is a clause rather than a name: cut off mid-sentence,
    too long, or the opening of the summary itself."""
    if _CLAUSE_END.search(title) or len(title) > MAX_SCENE_TITLE:
        return True
    return len(title) >= 8 and summary.startswith(title) and title != summary


def _noun_of(word: str) -> str:
    for particle in _PARTICLES:
        if word.endswith(particle) and len(word) - len(particle) >= 2:
            return word[: -len(particle)]
    return word


def scene_title(title: str, summary: str) -> str:
    """A short title for a planned scene. The model's own title is kept unless it is a
    clause (`_clause_title`) or missing; then a "X의 Y" phrase from the summary or the
    clause is taken, or failing that the clause's first words, cut at the first break.
    Empty only when both title and summary are."""
    if title and not _clause_title(title, summary):
        return title
    source = summary or title
    for text in (summary, title):
        for m in _OF_PHRASE.finditer(text):
            if m.group(1) not in _PRONOUN_OWNERS:
                return f"{m.group(1)}의 {_noun_of(m.group(2))}"
    head = re.split(r"[,.!?…:;]", source, maxsplit=1)[0].split()
    if len(head) > 1 and re.search(r"(?:은|는)$", head[0]):
        head = head[1:]  # the subject ("이진호는") names the scene no better than its cast
    short = " ".join(head[:3])
    return _CLAUSE_END.sub("", short)[:MAX_SCENE_TITLE].strip()


def _scene_dicts(
    value: object, chapter_id: str, by_name: Mapping[tuple[str, str], str]
) -> list[dict[str, Any]]:
    """A chapter's scenes from the model's JSON: ids by code (`<chapter_id>.s01`), and
    the names of the cast as its ids; a scene with neither title nor summary, or one that
    repeats an earlier scene's summary and beats word for word (or by high bigram overlap),
    is dropped."""
    scenes: list[dict[str, Any]] = []
    seen_body: set[tuple[str, tuple[str, ...]]] = set()
    seen_texts: list[str] = []
    for scene in _items(value)[:MAX_SCENES_PER_CHAPTER]:
        if not isinstance(scene, dict):
            continue
        sc = cast(dict[str, Any], scene)
        summary = _text(sc.get("summary"))
        beats = _texts(sc.get("beats"))
        title = scene_title(_text(sc.get("title")), _text(sc.get("summary")))
        if not title:
            continue
        if summary and beats:
            body_key = (summary.casefold(), tuple(b.casefold() for b in beats))
            if body_key in seen_body:
                continue
            body_text = _scene_body_text(sc)
            if len(body_text) >= MIN_SCENE_COMPARE_CHARS and any(
                bigram_overlap(prev, body_text) >= 0.65 for prev in seen_texts
            ):
                continue
            seen_body.add(body_key)
            if len(body_text) >= MIN_SCENE_COMPARE_CHARS:
                seen_texts.append(body_text)
        raw_trans = (
            [cast(dict[str, Any], t) for t in _items(sc.get("transitions")) if isinstance(t, dict)]
            if chapter_id == "plan"
            else []
        )
        scenes.append(
            {
                "id": f"{chapter_id}.s{len(scenes) + 1:02d}",
                "title": title,
                "summary": summary,
                "beats": beats,
                "characters": _ids(sc.get("characters"), "characters", by_name),
                "places": _ids(sc.get("places"), "places", by_name),
                **({"transitions": raw_trans} if raw_trans else {}),
            }
        )
    return scenes


class _RunOutlineMixin(_RunBase):
    """Stage 3 outline generation and feedback revision."""

    def _template(self) -> StructureTemplate | None:
        state = self.state
        if state.structure and state.structure in self.structures:
            return self.structures[state.structure]
        picked = structure_for(state.request, state.brief.genre, self.structures)
        if picked is None:
            return None
        self._save(structure=picked[0], structure_source=picked[1])
        return self.structures[picked[0]]

    async def _outline(self) -> None:
        """Stage 3: the chosen structure's beats, filled with this premise and cast.

        One chapter per act of the structure (`structure_for`: the person's named one, else
        the genre's). Code decides what each chapter does and where the twist is revealed
        (`arc_for`). The model first says what the story turns on (`Spine`), and then
        outlines one chapter per call, in order (`_outline_by_chapter`). Asked for the
        whole outline in one reply and then again with its problems, qwen3:8b fixed one of
        four missing twists and settled conflicts early (#1991).
        """
        state = self.state
        premise = state.premise()
        cast_text = "\n".join(
            f"- {c['name']} ({c['kind']}): {c.get('profile', '')}" for c in state.cast
        )
        template = self._template()
        head = (
            f"Story premise:\n{premise.title}: {premise.premise}\n\n"
            f"Direction:\n{_brief_lines(state.brief)}\n\nCast and places:\n{cast_text}\n\n"
        )
        kept_acts = template.acts[:MAX_CHAPTERS] if template is not None else ()
        acts = [act.id for act in kept_acts]
        arc = arc_for([[beat.id for beat in act.beats] for act in kept_acts])
        check: dict[str, Any] | None = None
        spine: Spine | None = None
        if template is None or arc is None:
            found = await self._outline_at_once(head, template)
        else:
            found, spine, check = await self._outline_by_chapter(head, template, arc)
        outline = outline_from(found, state.cast, acts=acts)
        notes = list(state.notes)
        if not outline.chapters:
            notes.append("The outline came back empty; one chapter was made from the premise.")
            outline = outline_from(
                {
                    "chapters": [
                        {
                            "title": premise.title,
                            "scenes": [{"title": premise.title, "summary": premise.premise}],
                        }
                    ]
                },
                state.cast,
                acts=acts,
            )
        elif acts and len(outline.chapters) < len(acts):
            notes.append(
                f"The outline covers {len(outline.chapters)} of the structure's {len(acts)} acts."
            )
        current = self.work.outline()
        self.work.save_outline(
            outline, room_id=self.room_id, expected_digest=current[1] if current else None
        )
        self._save(
            stage="review_outline",
            outline_saved=True,
            central_conflict=_text(found.get("central_conflict")) or None,
            twist=_text(found.get("twist")) or None,
            spine=spine.record() if spine else None,
            arc=arc.record() if arc else None,
            stakes=[_text(c.get("stakes")) for c in _chapters(found)][: len(acts) or None],
            outline_check=check,
            notes=notes,
        )

    async def _outline_at_once(
        self, head: str, template: StructureTemplate | None
    ) -> dict[str, Any]:
        """The whole outline in one reply: with no structure, or one under three acts."""
        plan = ""
        if template is not None:
            plan = f" on the structure '{template.title}', one chapter per act, in order:\n" + (
                "\n".join(
                    f"Chapter {number} (act '{act.title}'):\n"
                    + "\n".join(f"  - {beat.title}: {beat.purpose}" for beat in act.beats)
                    for number, act in enumerate(template.acts[:MAX_CHAPTERS], start=1)
                )
            )
        prompt = (
            f"{head}Outline the story in 3 to {MAX_CHAPTERS} chapters{plan}\n"
            f"Each chapter has {SCENES_PER_CHAPTER} scenes. {_SCENE_SHAPE}\n"
            'Reply as JSON: {"chapters": [{"title": ..., "scenes": [{"title": ..., '
            '"summary": ..., "beats": [...], "characters": [names], "places": [names]}]}]}'
        )
        return await _ask_json(self.model, prompt, temperature=0.6, max_tokens=3500) or {}

    async def _outline_by_chapter(
        self, head: str, template: StructureTemplate, arc: Arc
    ) -> tuple[dict[str, Any], Spine, dict[str, Any]]:
        """The outline one chapter per call, each checked and asked for once more alone.

        First the spine (conflict, twist, clue, resolution), held by code; then, act by
        act, a call told the chapters so far, its act's beats, its function, what it must
        not do, and only its part of the spine (`Spine.told`: no twist before the twist
        chapter). Each chapter is checked as it comes (`chapter_problems`); with a problem,
        that chapter alone is asked for again with the reasons, and the draft with fewer
        problems is kept. A chapter that comes back with no scene ends the outline there.
        """
        kept_acts = template.acts[:MAX_CHAPTERS]
        found = (
            await _ask_json(
                self.model,
                head + _spine_prompt(template.title, len(kept_acts), arc),
                temperature=0.6,
                max_tokens=1200,
            )
            or {}
        )
        spine = Spine(
            conflict=_text(found.get("central_conflict")),
            twist=_text(found.get("twist")),
            clue=_text(found.get("clue")),
            resolution=_text(found.get("resolution")),
        )
        chapters: list[dict[str, Any]] = []
        record: list[dict[str, Any]] = []
        asked = 0
        for number, act in enumerate(kept_acts, start=1):
            so_far = "\n\n".join(
                f"Chapter {n}:\n{chapter_text(c)}" for n, c in enumerate(chapters, start=1)
            )
            must_not: list[str] = []
            if number < arc.twist_chapter:
                must_not.append(
                    "reveal the twist: no character learns, says, rightly guesses or confirms it"
                )
            if number < arc.climax_chapter:
                must_not.append(
                    "settle the conflict: no win, loss, forgiveness, reconciliation, "
                    "acceptance or final decision; the main character ends it worse off "
                    "than at its start"
                )
            prompt = (
                f"{head}The story is built on the structure '{template.title}', one chapter "
                f"per act, {len(kept_acts)} chapters. What it turns on (the writer's plan, "
                f"which the reader does not know yet):\n{spine.told(arc, number)}\n\n"
                + (f"The chapters so far:\n{so_far}\n\n" if so_far else "")
                + f"{CHAPTER_PROMPT_MARK} {number} of {len(kept_acts)} (act '{act.title}'). "
                "It covers these beats, in order, with events of THIS premise and cast:\n"
                + "\n".join(f"  - {beat.title}: {beat.purpose}" for beat in act.beats)
                + f"\nThis chapter must {arc.does(number)}.\n"
                + (f"It must not {'; nor '.join(must_not)}.\n" if must_not else "")
                + "It does not restate the premise or an earlier chapter's events: it starts "
                "where the chapter before ended. Each scene must move the plot to a distinct "
                "situation or action; do not repeat an earlier scene's summary or beats.\n"
                f"It has {scenes_for(arc, number)} scenes. {_SCENE_SHAPE}\n"
                'Reply as JSON: {"title": ..., "stakes": what can be lost at its end, '
                '"scenes": [{"title": ..., "summary": ..., "beats": [...], '
                '"characters": [names], "places": [names]}]}'
            )
            previous = chapter_text(chapters[-1]) if chapters else None
            reply = await _ask_json(self.model, prompt, temperature=0.6, max_tokens=2500)
            asked += 1
            if not _has_scenes(reply):
                break
            chapter = cast(dict[str, Any], reply)
            prior_scenes = [
                cast(Mapping[str, Any], s)
                for c in chapters
                for s in _items(c.get("scenes"))
                if isinstance(s, Mapping)
            ]
            draft_scenes = [
                cast(Mapping[str, Any], s)
                for s in _items(chapter.get("scenes"))[: scenes_for(arc, number)]
                if isinstance(s, Mapping)
            ]
            first = await chapter_problems(
                self.model, arc, number, chapter_text(chapter), spine=spine, previous=previous
            )
            rep_pairs = await repetitive_scenes(draft_scenes, prior_scenes, self.embedder)
            for left_title, right_title in rep_pairs:
                first.append(
                    {
                        "kind": "repetitive_scene",
                        "chapter": number,
                        "left": left_title,
                        "right": right_title,
                    }
                )
            entry: dict[str, Any] = {"chapter": number, "first": first, "asked": 1, "final": first}
            if first:
                reasons = list(problem_reasons(arc, first))
                for left_title, right_title in rep_pairs:
                    reasons.append(
                        f"scene '{right_title}' repeats the situation or events of '{left_title}'; "
                        "replace its summary, beats, and setting with a completely different event "
                        "that advances the plot"
                    )
                again = (
                    f"{prompt}\n\nYour last draft of this chapter broke the story's shape:\n"
                    + "\n".join(f"- {r}" for r in reasons)
                    + "\nWrite this chapter again, fixing these."
                )
                second = await _ask_json(self.model, again, temperature=0.6, max_tokens=2500)
                asked += 1
                entry["asked"] = 2
                if _has_scenes(second):
                    retried = cast(dict[str, Any], second)
                    later = await chapter_problems(
                        self.model,
                        arc,
                        number,
                        chapter_text(retried),
                        spine=spine,
                        previous=previous,
                    )
                    retried_scenes = [
                        cast(Mapping[str, Any], s)
                        for s in _items(retried.get("scenes"))[: scenes_for(arc, number)]
                        if isinstance(s, Mapping)
                    ]
                    later_rep = await repetitive_scenes(retried_scenes, prior_scenes, self.embedder)
                    for left_title, right_title in later_rep:
                        later.append(
                            {
                                "kind": "repetitive_scene",
                                "chapter": number,
                                "left": left_title,
                                "right": right_title,
                            }
                        )
                    entry["second"] = later
                    if len(later) < len(first):
                        chapter = retried
                        entry["final"] = later
            # The chapter's weight decides its scenes, not the reply's length.
            chapter = {
                **chapter,
                "scenes": _items(chapter.get("scenes"))[: scenes_for(arc, number)],
            }
            chapters.append(chapter)
            record.append(entry)
        head_problems: list[dict[str, Any]] = []
        if not spine.twist:
            head_problems.append({"kind": "no_twist", "chapter": arc.twist_chapter})
        if len(chapters) < len(kept_acts):
            head_problems.append({"kind": "missing_chapters", "chapter": len(chapters) + 1})
        check = {
            "first": head_problems + [p for e in record for p in e["first"]],
            "final": head_problems + [p for e in record for p in e["final"]],
            "asked": asked,
            "chapters": record,
        }
        outline = {
            "central_conflict": spine.conflict,
            "twist": spine.twist,
            "chapters": chapters,
        }
        return outline, spine, check

    # -- stage 3b: feedback on the outline, and settling the plot --------------------------

    def _acts(self) -> list[str]:
        template = self._template()
        return [act.id for act in template.acts[:MAX_CHAPTERS]] if template is not None else []

    async def _revise_outline(self, feedback: str) -> None:
        """The outline revised by the person's `feedback`: one call, then the code's checks.

        The model is given the outline as it stands and the feedback, and asked to change
        only what the feedback asks. The reply goes through `outline_from` as the first
        outline did, and each chapter is checked against the arc (`chapter_problems`);
        what the check finds is shown, not fixed behind the person's back. A chapter the
        person asked to remove is not reported missing. An unreadable reply keeps the
        outline as it was, and the reply says so.
        """
        state = self.state
        current = self.work.outline()
        names = {str(c.get("id")): str(c.get("name", "")) for c in state.cast}
        shown = _outline_json(current[0] if current else Outline(), names, state.stakes)
        premise = state.premise()
        cast_text = "\n".join(
            f"- {c['name']} ({c['kind']}): {c.get('profile', '')}" for c in state.cast
        )
        asked_feedback = _feedback_with_brief(feedback, state.brief)
        prompt = (
            f"Story premise:\n{premise.title}: {premise.premise}\n\n"
            f"Direction:\n{_brief_lines(state.brief)}\n\nCast and places:\n{cast_text}\n\n"
            f"Central conflict: {state.central_conflict or ''}\n"
            f"Twist: {state.twist or ''}\n\n"
            f"{REVISE_MARK}. The outline as it stands, as JSON:\n{shown}\n\n"
            f"The person's feedback, word for word:\n{asked_feedback}\n\n"
            "Change what the feedback asks for and keep everything else as it is. If it "
            "asks to remove a chapter, leave that chapter out. If it changes the conflict "
            "or the twist, give the new one.\n"
            "Keep each chapter's number of scenes unless the feedback asks otherwise, and "
            f"at most {MAX_SCENES_PER_CHAPTER}. {_SCENE_SHAPE}\n"
            'Reply as JSON: {"central_conflict": ..., "twist": ..., "chapters": '
            '[{"title": ..., "stakes": ..., "scenes": [{"title": ..., "summary": ..., '
            '"beats": [...], "characters": [names], "places": [names]}]}]}'
        )
        found = await _ask_json(self.model, prompt, temperature=0.4, max_tokens=3500) or {}
        said = [*state.outline_feedback, feedback]
        notes = list(state.notes)
        acts = self._acts()
        raw = _chapters(found)
        outline = outline_from(found, state.cast, acts=acts if len(raw) == len(acts) else ())
        if not outline.chapters:
            notes.append("The revised outline could not be read; the outline was kept.")
            self._save(outline_feedback=said, revision_failed=True, notes=notes)
            return
        if acts and len(outline.chapters) != len(acts):
            notes.append(
                f"After the feedback the outline has {len(outline.chapters)} chapters for "
                f"the structure's {len(acts)} acts, so chapters are not tied to acts."
            )
        conflict = _text(found.get("central_conflict")) or state.central_conflict or ""
        twist = _text(found.get("twist")) or state.twist or ""
        stakes = [_text(c.get("stakes")) for c in raw][: len(outline.chapters)]
        check = state.outline_check
        arc = _arc_of(state.arc)
        if arc is not None:
            spine = Spine(
                conflict=conflict,
                twist=twist,
                clue=(state.spine or {}).get("clue", ""),
                resolution=(state.spine or {}).get("resolution", ""),
            )
            problems: list[dict[str, Any]] = (
                [] if twist else [{"kind": "no_twist", "chapter": arc.twist_chapter}]
            )
            previous: str | None = None
            for number, chapter in enumerate(_chapter_dicts(outline, stakes), start=1):
                text = chapter_text(chapter)
                problems += await chapter_problems(
                    self.model, arc, number, text, spine=spine, previous=previous
                )
                previous = text
            check = {**(state.outline_check or {}), "final": problems, "revised": len(said)}
        current = self.work.outline()
        self.work.save_outline(
            outline, room_id=self.room_id, expected_digest=current[1] if current else None
        )
        self._save(
            outline_feedback=said,
            revision_failed=False,
            central_conflict=conflict or None,
            twist=twist or None,
            stakes=stakes,
            outline_check=check,
            notes=notes,
        )


__all__ = [
    "CHAPTER_PROMPT_MARK",
    "MAX_SCENE_BIGRAM",
    "MAX_SCENE_COSINE",
    "MAX_SCENE_TITLE",
    "MIN_SCENE_COMPARE_CHARS",
    "REVISE_MARK",
    "SPINE_MARK",
    "_CLAUSE_END",
    "_OF_PHRASE",
    "_PARTICLES",
    "_PRONOUN_OWNERS",
    "_RunOutlineMixin",
    "_SCENE_SHAPE",
    "_chapter_dicts",
    "_chapters",
    "_clause_title",
    "_has_scenes",
    "_noun_of",
    "_outline_json",
    "_scene_dicts",
    "_spine_prompt",
    "outline_from",
    "repetitive_scenes",
    "scene_title",
]
