# pyright: reportPrivateUsage=false
"""Stage 3b (Settled Plot KG extraction and trajectory validation) and Stage 4b (Chapter scene planning)."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from pydantic import ValidationError

from uclone_x.knowledge.quotes import folded
from uclone_x_story.arc import Spine, chapter_text
from uclone_x_story.kg_plan import (
    effective_codex,
    entering_world_state,
    format_chronological_kg_log,
    merge_transitions,
    parse_detail_transitions,
    parse_plot_transitions,
    thread_transitions,
    unplaced_transitions,
    validate_kg_trajectory,
)
from uclone_x_story.library import StoryError
from uclone_x_story.names import CodexNames, id_for_name
from uclone_x_story.proposals import new_entry_clash
from uclone_x_story.schemas import Chapter, Outline, Proposal
from uclone_x_story.start_cast import (
    _NEW_CHARACTERS_ASK,
    _NEW_CHARACTERS_SHAPE,
    _cast_ids,
    _new_characters,
    _RunCastMixin,
)
from uclone_x_story.start_notes import _scene_labels, _waiting_scene_words, _where
from uclone_x_story.start_outline import (
    _SCENE_SHAPE,
    _chapter_dicts,
    _outline_json,
    _scene_dicts,
)
from uclone_x_story.start_premise import _feedback_with_brief
from uclone_x_story.start_state import (
    MAX_SCENES_PER_CHAPTER,
    Language,
    StartState,
    _arc_of,
    _ask_json,
    _brief_lines,
    _ids,
    _text,
    _texts,
    candidate_names,
    scenes_for,
)
from uclone_x_story.tools import NewEntryParams, new_entry_draft

#: The first words of the call that reads the settled outline for the codex.
KG_MARK = "Read this settled story outline for its story bible"
#: How many threads and items settling the plot proposes, beside the conflict and twist.
MAX_PLOT_THREADS = 4
MAX_PLOT_ITEMS = 3
#: Words that name an idea, a feeling or a motif, not a thing a character can hold: an
#: item proposed at the plot lock whose name is one of these, or ends in one, is dropped
#: (`abstract_item`). "용기" is not here: it is also a container.
ABSTRACT_ITEM_WORDS: frozenset[str] = frozenset(
    {
        "기억",
        "추억",
        "희망",
        "진실",
        "비밀",
        "사랑",
        "두려움",
        "공포",
        "과거",
        "미래",
        "약속",
        "죄책감",
        "믿음",
        "신뢰",
        "복수",
        "운명",
        "자유",
        "상실",
        "슬픔",
        "분노",
        "우정",
        "정의",
        "명예",
        "꿈",
        "욕망",
        "배신",
        "후회",
        "용서",
        "증오",
        "외로움",
        "고독",
        "트라우마",
        "그리움",
        "침묵",
        "유산",
        "memory",
        "memories",
        "hope",
        "truth",
        "secret",
        "secrets",
        "love",
        "fear",
        "past",
        "future",
        "promise",
        "guilt",
        "trust",
        "faith",
        "courage",
        "revenge",
        "destiny",
        "fate",
        "freedom",
        "loss",
        "grief",
        "anger",
        "friendship",
        "justice",
        "honor",
        "dream",
        "dreams",
        "desire",
        "betrayal",
        "regret",
        "forgiveness",
        "hatred",
        "loneliness",
        "silence",
        "legacy",
    }
)
#: The first words of the call that plans a later chapter before it is written, and of
#: the call that revises that plan by the person's feedback: "... chapter 2".
DETAILS_MARK = "Plan the scenes of chapter"
DETAILS_REVISE_MARK = "Revise the plan of chapter"


#: The name a plot thread takes when its own names a character, place or item (#2380):
#: names are one key across kinds (#1808), so the thread is told apart in the story's
#: language rather than lost.
_OWN_THREAD_NAME: Mapping[Language, str] = {"ko": "{name} (줄거리)", "en": "{name} (thread)"}


def _as_thread_of_its_own(params: NewEntryParams, language: Language) -> NewEntryParams:
    """`params` under a thread's own name (`_OWN_THREAD_NAME`)."""
    return params.model_copy(update={"name": _OWN_THREAD_NAME[language].format(name=params.name)})


def _kg_prompt(state: StartState, outline: Outline) -> str:
    characters = ", ".join(str(c["name"]) for c in state.cast if c.get("kind") == "characters")
    places = ", ".join(str(c["name"]) for c in state.cast if c.get("kind") == "places")
    places_line = f"Places: {places}\n" if places else ""
    shown = _chapter_dicts(outline, state.stakes)
    # Each scene under the label code wrote, which a transition names it by (#2374).
    for chapter, kept in zip(shown, outline.chapters, strict=True):
        for scene, source in zip(chapter["scenes"], kept.scenes, strict=True):
            scene["title"] = f"[{source.id}] {scene['title']}"
    chapters = "\n\n".join(f"Chapter {n}:\n{chapter_text(c)}" for n, c in enumerate(shown, start=1))
    # The reply's transitions are keyed by these labels, so code wrote each scene (#2378).
    slots = ", ".join(f'"[{s.id}]": []' for c in outline.chapters for s in c.scenes)
    return (
        f"{KG_MARK}. The person has settled this plot.\n\n"
        f"Characters: {characters}\n"
        f"{places_line}"
        f"Central conflict: {state.central_conflict or ''}\nTwist: {state.twist or ''}\n\n"
        f"{chapters}\n\n"
        "List what the story bible should track from this plot, beyond the characters, "
        "places, conflict and twist above:\n"
        f"- threads: up to {MAX_PLOT_THREADS} key events, secrets or promises that one "
        "chapter sets up and a later one pays off, and relationships between characters "
        "that are not family (a rivalry, a debt, a friendship, a love), each with the "
        "chapter that sets it up and the chapter that pays it off (never give a thread "
        "the exact same name as an item);\n"
        f"- items: up to {MAX_PLOT_ITEMS} physical objects the plot turns on, each named "
        "in the outline above: a thing a character can hold, carry, lose or hand over (a "
        "letter, a key, a lantern). Not an idea, feeling, memory, secret, promise, "
        "motif, room, building or place: those are threads, places or nothing;\n"
        "- transitions: key state or relationship changes across chapters (who acquires, "
        "loses or uses an item, whose status or location changes, or how a relationship "
        "or thread shifts in a specific scene), each listed under the label of the scene it "
        "happens in, as shown above; a scene with none keeps an empty list. Make each state "
        "value a concrete state delta (a specific new stance, goal, or relationship shift "
        "caused by that scene), not a repeated generic word.\n"
        "Write names, profiles, and transition values in the language of the outline.\n"
        'Reply as JSON: {"threads": [{"name": a short name, "profile": one sentence, '
        '"characters": [names from the list above], "planted_chapter": number, '
        '"pay_off_chapter": number}], "items": [{"name": ..., "profile": one sentence}], '
        f'"transitions": {{{slots}}}}}, each scene\'s list holding {{"subject": a character, '
        'item or thread name, "set": {"location" or "possesses" or "stance" or "goal" or '
        '"status": "new value"}, "relations": {"other name": "relation"}, '
        '"note": one short sentence}'
    )


def abstract_item(name: str) -> bool:
    """Whether an item's `name` names an idea or a motif ("기억", "아버지의 비밀") rather
    than a physical object.

    The name, or its last word, is one of `ABSTRACT_ITEM_WORDS`; a Korean last word also
    counts when it ends in a Korean one of two or more letters ("잃어버린기억"), while an
    English last word must be one exactly ("the locket" is kept, "a promise" is not).
    """
    words = re.findall(r"\w+", name.casefold())
    if not words:
        return False
    whole, last = " ".join(words), words[-1]
    if whole in ABSTRACT_ITEM_WORDS or last in ABSTRACT_ITEM_WORDS:
        return True
    return any(len(w) >= 2 and last.endswith(w) for w in ABSTRACT_ITEM_WORDS if not w.isascii())


def _kg_params(
    found: Mapping[str, Any],
    cast_lines: Sequence[Mapping[str, Any]],
    scene_of: Callable[[object], str | None],
) -> tuple[list[tuple[NewEntryParams, dict[str, str | None]]], list[str]]:
    """The threads and items of the settled-plot reply, as new entries to propose, and
    the names of the items dropped as abstract (`abstract_item`), which do not count
    toward `MAX_PLOT_ITEMS`."""
    by_name = _cast_ids(cast_lines)
    cast_names = {k for (_, k) in by_name}
    raw_items = found.get("items")
    item_dicts = [
        cast(dict[str, Any], it)
        for it in (cast(list[object], raw_items) if isinstance(raw_items, list) else [])
        if isinstance(it, dict)
    ]
    physical_item_names = {
        n.casefold()
        for it in item_dicts
        if (n := _text(it.get("name")))
        and not abstract_item(n)
        and not any(c.casefold() in cast_names for c in candidate_names(n))
    }
    wanted: list[tuple[NewEntryParams, dict[str, str | None]]] = []
    dropped: list[str] = []
    for kind, key, most in (
        ("threads", "threads", MAX_PLOT_THREADS),
        ("items", "items", MAX_PLOT_ITEMS),
    ):
        raw = found.get(key)
        items = cast(list[object], raw) if isinstance(raw, list) else []
        taken = 0
        for item in items:
            if taken >= most:
                break
            if not isinstance(item, dict):
                continue
            it = cast(dict[str, Any], item)
            name = _text(it.get("name"))
            if not name:
                continue
            if kind == "items" and any(c.casefold() in cast_names for c in candidate_names(name)):
                continue
            if kind == "items" and abstract_item(name):
                dropped.append(name)
                continue
            if kind == "threads" and name.casefold() in physical_item_names:
                continue
            taken += 1
            fields: dict[str, str | None] = {}
            state: dict[str, Any] = {}
            if kind == "threads":
                involves = _ids(it.get("characters"), "characters", by_name)
                if involves:
                    state["involves"] = involves
                fields = {
                    "planted_in": scene_of(it.get("planted_chapter")),
                    "pay_off_by": scene_of(it.get("pay_off_chapter")),
                }
            params = NewEntryParams(
                kind="threads" if kind == "threads" else "items",
                name=name,
                profile=_text(it.get("profile")),
                state=state,
            )
            wanted.append((params, fields))
    return wanted, dropped


def _with_entry_fields(draft: Proposal, fields: Mapping[str, str | None]) -> Proposal:
    """`draft` with fields `NewEntryParams` has no place for (a thread's scenes, notes)."""
    given = {k: v for k, v in fields.items() if v}
    if not given:
        return draft
    data = draft.model_dump(mode="json")
    data["change"] = {"new_entry": {**data["change"]["new_entry"], **given}}
    try:
        return Proposal.model_validate(data)
    except ValidationError as exc:
        # Named as the person reads it, never by its id (#2382).
        name = str(data["change"]["new_entry"].get("name") or "")
        raise StoryError(f"'{name}' was not proposed: it does not fit a codex entry.") from exc


def _detail_scenes(
    found: Mapping[str, Any] | None,
    cast_lines: Sequence[Mapping[str, Any]],
    limit: int = MAX_SCENES_PER_CHAPTER,
) -> list[dict[str, Any]]:
    """A chapter plan's scenes from the model's JSON, the cast by name as the person reads
    it; a name not in the cast is left out, and scenes past `limit` (the chapter's count,
    `scenes_for`) are too. Empty when the reply holds no scene."""
    if found is None:
        return []
    by_name = _cast_ids(cast_lines)
    names = {str(c.get("id")): str(c.get("name", "")) for c in cast_lines}
    return [
        {
            "title": scene["title"],
            "summary": scene["summary"],
            "beats": scene["beats"],
            "characters": [names[i] for i in scene["characters"]],
            "places": [names[i] for i in scene["places"]],
            **({"transitions": scene["transitions"]} if scene.get("transitions") else {}),
        }
        for scene in _scene_dicts(found.get("scenes"), "plan", by_name)[:limit]
    ]


class _RunDetailsMixin(_RunCastMixin):
    """Stage 3b plot lock and Stage 4b chapter detail planning."""

    async def _rewrite_asked(self, edits: Sequence[Mapping[str, str]], feedback: str) -> None:
        raise NotImplementedError

    def _kinds_named(self, name: str, pending: Sequence[Proposal]) -> set[str]:
        """The kinds of the codex entries, and of the new entries `pending` proposes, that
        `name` already names (`CodexNames`, as `new_entry_clash` compares them)."""
        kinds = {item.kind for item in CodexNames(self.work.codex()).matches(name)}
        wanted = folded(name)
        for proposal in pending:
            if proposal.status != "pending" or proposal.change.new_entry is None:
                continue
            try:
                other = proposal.new_entry()
            except ValueError:
                continue
            if wanted in {folded(k) for k in (other.id, other.name, *other.aliases)}:
                kinds.add(proposal.kind)
        return kinds

    async def _lock_plot(self) -> None:
        """Settle the plot: record it, and propose its key events for the codex.

        Code proposes the central conflict and the twist as threads, planted and paid off
        at the scenes the arc gives them. One call reads the settled outline for the
        rest: key events, secrets and relationships between characters that are not
        family (threads, with the characters they involve) and the objects the plot
        turns on (items). Each is a new-entry proposal, pending like the cast: nothing is
        approved here (§5.3). An unreadable reply leaves the conflict and the twist.
        """
        state = self.state
        current = self.work.outline()
        outline = current[0] if current else Outline()
        notes = list(state.notes)
        firsts = [c.scenes[0].id for c in outline.chapters if c.scenes]

        def scene_of(number: object) -> str | None:
            if isinstance(number, bool) or not isinstance(number, int):
                return None
            return firsts[number - 1] if 1 <= number <= len(firsts) else None

        arc = _arc_of(state.arc)
        spine = state.spine or {}
        wanted: list[tuple[NewEntryParams, dict[str, str | None]]] = []
        if state.central_conflict:
            climax = arc.climax_chapter if arc else len(firsts)
            wanted.append(
                (
                    NewEntryParams(
                        kind="threads",
                        id="central_conflict",
                        name="중심 갈등",
                        profile=state.central_conflict,
                    ),
                    {
                        "planted_in": scene_of(1),
                        "pay_off_by": scene_of(min(climax, len(firsts))),
                        "notes": spine.get("resolution") or None,
                    },
                )
            )
        if state.twist:
            reveal = arc.twist_chapter if arc else len(firsts)
            wanted.append(
                (
                    NewEntryParams(kind="threads", id="twist", name="반전", profile=state.twist),
                    {
                        "pay_off_by": scene_of(min(reveal, len(firsts))),
                        "notes": spine.get("clue") or None,
                    },
                )
            )
        dropped = 0
        found = await _ask_json(
            self.model, _kg_prompt(state, outline), temperature=0.3, max_tokens=3000
        )
        if found is None:
            notes.append("The settled outline's key events could not be read.")
        else:
            read, abstract = _kg_params(found, state.cast, scene_of)
            wanted += read
            dropped = len(abstract)
            if abstract:
                notes.append(
                    f"{dropped} item(s) named an idea, not an object, and were not proposed: "
                    + ", ".join(abstract)
                    + "."
                )
        pending = [p for p, _ in self.work.proposals()[0] if p.status == "pending"]
        saved: list[Proposal] = []
        added: list[dict[str, str]] = []
        with_fields_fn = getattr(
            sys.modules.get("uclone_x_story.start"), "_with_entry_fields", _with_entry_fields
        )
        for index, (params, fields) in enumerate(wanted):
            try:
                draft = with_fields_fn(
                    new_entry_draft(params, room_id=self.room_id, context=self.context), fields
                )
            except StoryError as exc:
                notes.append(str(exc))
                continue
            clash = new_entry_clash(
                self.work, draft.kind, draft.new_entry(), pending=[*pending, *saved]
            )
            taken_by = self._kinds_named(params.name, [*pending, *saved])
            if clash is not None and params.kind == "threads" and "threads" not in taken_by:
                # A thread named like a character, place or item takes a name of its own,
                # so it keeps its lifecycle rather than losing it to the other (#2380).
                params = _as_thread_of_its_own(params, state.brief.language)
                wanted[index] = (params, fields)
                try:
                    draft = with_fields_fn(
                        new_entry_draft(params, room_id=self.room_id, context=self.context),
                        fields,
                    )
                except StoryError as exc:
                    notes.append(str(exc))
                    continue
                clash = new_entry_clash(
                    self.work, draft.kind, draft.new_entry(), pending=[*pending, *saved]
                )
            if clash is not None:
                notes.append(f"'{params.name}' was not proposed: {clash}.")
                continue
            proposal = self.work.add_proposal(draft, room_id=self.room_id)
            saved.append(proposal)
            added.append(
                {
                    "proposal": proposal.id,
                    "kind": proposal.kind,
                    "entry_id": proposal.entry_id,
                    "name": params.name,
                }
            )
        all_proposals = [p for p, _ in self.work.proposals()[0]]
        codex_now = effective_codex(self.work.codex(), all_proposals)
        base_transitions = thread_transitions(
            wanted, outline, id_for_name_fn=id_for_name, language=state.brief.language
        )
        llm_transitions = (
            parse_plot_transitions(found, outline, state.cast, codex_now)
            if found is not None
            else []
        )
        raw_transitions = merge_transitions(base_transitions, llm_transitions)
        kept_transitions, report = validate_kg_trajectory(
            outline, codex_now, raw_transitions, arc=arc, cast_lines=state.cast
        )
        self._save(
            plot_locked=True,
            plot_kg=added,
            plot_kg_dropped=dropped,
            plot_kg_unplaced=unplaced_transitions(found, outline),
            planned_transitions=[t.record() for t in kept_transitions],
            kg_trajectory=report.record(),
            notes=notes,
        )

    def _details_head(self, number: int, outline: Outline) -> str:
        """What a chapter's plan is made from: the premise, the cast, the story's plan for
        this chapter, the chapter as outlined, and how the chapter before it ended."""
        state = self.state
        names = {str(c.get("id")): str(c.get("name", "")) for c in state.cast}
        chapter = outline.chapters[number - 1]
        outlined = _outline_json(Outline(chapters=[chapter]), names, [])
        arc = _arc_of(state.arc)
        plan = ""
        if arc is not None and number <= len(arc.functions):
            spine = Spine(
                conflict=state.central_conflict or "",
                twist=state.twist or "",
                clue=(state.spine or {}).get("clue", ""),
                resolution=(state.spine or {}).get("resolution", ""),
            )
            plan = (
                f"What the story turns on (the writer's plan, which the reader does not know "
                f"yet):\n{spine.told(arc, number)}\nThis chapter must {arc.does(number)}.\n"
            )
        elif state.central_conflict:
            plan = f"Central conflict: {state.central_conflict}\n"
        stakes = state.stakes[number - 1] if number <= len(state.stakes) else ""
        if stakes:
            plan += f"What can be lost at its end: {stakes}\n"
        tracked = ", ".join(k["name"] for k in state.plot_kg)
        prior_context = ""
        if number > 1:
            prior_lines: list[str] = []
            for ch in outline.chapters[: number - 1]:
                for sc in ch.scenes:
                    if sc.summary:
                        prior_lines.append(f"- [{sc.id}] {sc.title}: {sc.summary}")
            if prior_lines:
                prior_context += (
                    "Earlier chapters' scenes (do not repeat their events or realizations):\n"
                    + "\n".join(prior_lines)
                    + "\n\n"
                )
            if chapter.scenes:
                all_proposals = [p for p, _ in self.work.proposals()[0]]
                rejected = {p.id for p in all_proposals if p.status == "rejected"}
                codex_now = effective_codex(
                    self.work.codex(),
                    all_proposals,
                    state.planned_transitions,
                    rejected=rejected,
                )
                entering = entering_world_state(codex_now, outline, chapter.scenes[0].id)
                if entering:
                    prior_context += (
                        f"Entering world state at start of chapter {number}:\n{entering}\n\n"
                    )
                chrono_log = format_chronological_kg_log(
                    codex_now, outline, before_scene_id=chapter.scenes[0].id
                )
                if chrono_log:
                    prior_context += f"{chrono_log}\n\n"
        before = self._last_text(outline, number - 1)
        ending = (
            f"How chapter {number - 1} ended, in its last lines:\n{before[-1200:]}\n\n"
            if before
            else ""
        )
        premise = state.premise()
        return (
            f"Story premise:\n{premise.title}: {premise.premise}\n\n"
            f"Direction:\n{_brief_lines(state.brief)}\n\nCast and places:\n"
            f"{self._cast_text()}\n"
            + (f"The story bible also tracks: {tracked}\n" if tracked else "")
            + f"\n{plan}\n{prior_context}"
            f"Chapter {number} of {len(outline.chapters)} as the settled outline "
            f"has it, as JSON:\n{outlined}\n\n{ending}"
        )

    async def _propose_details(self, number: int) -> None:
        """The plan of chapter `number`, one call, shown to the person before it is written.

        The plan keeps the settled outline's events for the chapter and makes its scenes
        and beats concrete, starting where the chapter before ended. A reply that cannot
        be read shows the outline's scenes for the chapter as the plan, and says so.
        """
        outline, _ = self.work.require_outline()
        count = scenes_for(_arc_of(self.state.arc), number)
        start = (
            "The first scene opens the story."
            if number == 1
            else "The first scene starts where the chapter before ended."
        )
        prompt = (
            f"{self._details_head(number, outline)}"
            f"{DETAILS_MARK} {number}, before it is written. Keep the outline's events for "
            "this chapter, in order, and make each scene concrete: who does what, where, "
            f"and what changes. {start} Do not settle or reveal anything the plan above "
            "says this chapter must not.\n"
            f"It has {count} scenes. {_SCENE_SHAPE}\n"
            "For each scene, include transitions that record concrete state or relationship "
            "changes (a new location, an item acquired/lost, or a specific multi-word "
            "stance/goal/relationship shift caused by that scene's action; do not repeat "
            "the same stance or goal value from an earlier scene).\n"
            f"{_NEW_CHARACTERS_ASK}"
            "Write every text value (including transition values, relations, and notes) in "
            "the language of the outline.\n"
            'Reply as JSON: {"scenes": [{"title": ..., "summary": ..., "beats": [...], '
            '"characters": [names], "places": [names], "transitions": [{"subject": name, '
            '"set": {"location" or "possesses" or "stance" or "goal" or "status": "value"}, '
            '"relations": {"other name": "relation"}, '
            f'"note": what changes}}]}}], {_NEW_CHARACTERS_SHAPE}}}'
        )
        found = await _ask_json(self.model, prompt, temperature=0.6, max_tokens=3500)
        scenes = _detail_scenes(found, self.state.cast, count)
        new = _new_characters(found, self._known_characters()) if scenes else []
        notes = list(self.state.notes)
        from_outline = not scenes
        if from_outline:
            notes.append(
                f"The plan of chapter {number} could not be read; the outline's scenes are shown."
            )
            names = {str(c.get("id")): str(c.get("name", "")) for c in self.state.cast}
            shown = json.loads(
                _outline_json(Outline(chapters=[outline.chapters[number - 1]]), names, [])
            )
            scenes = cast(list[dict[str, Any]], shown["chapters"][0]["scenes"])
        self._save(
            stage="review_chapter",
            details_chapter=number,
            chapter_details=scenes,
            details_new=new,
            details_feedback=[],
            details_revision_failed=False,
            details_scene_edits=[],
            details_scene_edits_failed=[],
            details_scene_edits_ambiguous=[],
            details_from_outline=from_outline,
            notes=notes,
        )

    async def _revise_details(self, feedback: str) -> bool:
        """The chapter's plan revised by the person's `feedback`, in one call; whether the
        feedback asked instead to change scenes already written.

        Only what the feedback asks is changed. An unreadable reply keeps the plan as it
        was, and the reply says so. The same call says which written scenes, of those it is
        shown by title, the feedback asks to change (`edits_written`): those are matched to
        the outline's ids, the plan is kept, and the flow says to edit them (#2303). At this
        stop the person's words once revised the next plan when they asked to fix a scene
        already written, and that scene stayed as it was.
        """
        state = self.state
        number = state.details_chapter or state.chapter_written + 1
        outline, _ = self.work.require_outline()
        # After "which one?" the answer is often only a place ("1장 2번째"): the words the
        # person first asked with are kept with the question, and read with the answer, so
        # the scene is matched by both and rewritten as first asked (#2366 review).
        pending = state.details_scene_edits_ambiguous
        asked = "\n".join([*(a["words"] for a in pending), feedback])
        shown_new = [
            {k: c[k] for k in ("name", "profile", "gender")}
            | {"family": [{"relative": f["relative"], "is": f["is"]} for f in c["family"]]}
            for c in state.details_new
        ]
        plan = json.dumps(
            {"scenes": state.chapter_details, "new_characters": shown_new},
            ensure_ascii=False,
            indent=1,
        )
        count = scenes_for(_arc_of(state.arc), number)
        written = self._written_titles(outline)
        labels = _scene_labels(outline, written)
        shown_written = "".join(f"- {label}\n" for label in labels.values())
        prompt = (
            f"{self._details_head(number, outline)}"
            f"{DETAILS_REVISE_MARK} {number} as the person asks. The plan as it stands, as "
            f"JSON:\n{plan}\n\nThe person's feedback, word for word:\n"
            f"{_feedback_with_brief(feedback, state.brief)}\n\n"
            f"{_waiting_scene_words(pending)}"
            "Change what the feedback asks for and keep everything else as it is. If it "
            "asks to remove a scene, leave that scene out.\n"
            + (
                "Scenes already written, which this plan does not change (if the feedback "
                "asks to change one of these written scenes instead of this chapter's plan, "
                f"put its title in edits_written and leave scenes as []):\n{shown_written}"
                if written
                else ""
            )
            + f"It has at most {count} scenes. {_SCENE_SHAPE}\n"
            "For each scene, include transitions that record concrete state or relationship "
            "changes caused by that scene's action.\n"
            f"{_NEW_CHARACTERS_ASK}"
            "Write every text value (including transition values, relations, and notes) in "
            "the language of the outline.\n"
            'Reply as JSON: {"edits_written": [the titles of the scenes already written that '
            "the feedback asks to change, as listed; [] when it asks only about this plan], "
            '"scenes": [{"title": ..., "summary": ..., "beats": [...], '
            '"characters": [names], "places": [names], "transitions": [{"subject": name, '
            '"set": {"location" or "possesses" or "stance" or "goal" or "status": "value"}, '
            '"relations": {"other name": "relation"}, '
            f'"note": what changes}}]}}], {_NEW_CHARACTERS_SHAPE}}}'
        )
        found = await _ask_json(self.model, prompt, temperature=0.4, max_tokens=3500)
        by_label = {label: scene_id for scene_id, label in labels.items()}
        named = list(dict.fromkeys(_texts((found or {}).get("edits_written"))))
        # A bare title two written scenes share names neither: the flow overwrites the
        # scene it picks, so it does not pick (#2365).
        ambiguous = [
            _where(outline, written, title) | {"words": asked}
            for title in named
            if title not in by_label and sum(1 for w in written.values() if w == title) > 1
        ]
        if ambiguous:
            self._save(details_scene_edits=[], details_scene_edits_ambiguous=ambiguous)
            return True
        edits = [
            {"id": by_label[label], "title": written[by_label[label]]}
            for label in named
            if label in by_label
        ]
        if edits:
            self._save(details_scene_edits=edits, details_scene_edits_ambiguous=[])
            await self._rewrite_asked(edits, asked)
            return True
        scenes = _detail_scenes(found, state.cast, count)
        said = [*state.details_feedback, feedback]
        if not scenes:
            notes = [*state.notes, f"The revised plan of chapter {number} could not be read."]
            self._save(
                details_feedback=said,
                details_revision_failed=True,
                details_scene_edits=[],
                details_scene_edits_failed=[],
                details_scene_edits_ambiguous=[],
                notes=notes,
            )
            return False
        self._save(
            details_feedback=said,
            details_revision_failed=False,
            details_from_outline=False,
            details_scene_edits=[],
            details_scene_edits_failed=[],
            details_scene_edits_ambiguous=[],
            chapter_details=scenes,
            details_new=_new_characters(found, self._known_characters()),
        )
        return False

    def _written_titles(self, outline: Outline) -> dict[str, str]:
        """Each written scene's id and the title the person knows it by, in story order."""
        written = set(self.work.written_scenes())
        return {
            scene.id: scene.title or scene.id
            for _, scene in outline.scenes_in_order()
            if scene.id in written
        }

    async def _apply_details(self, number: int) -> None:
        """The chapter's plan put into the outline, as its scenes, before it is written.

        The chapter keeps its id, title and act; its scenes get ids by code, as the
        outline's did. With no plan, the outline is left as it is.
        """
        details = self.state.chapter_details
        if not details:
            return
        outline, digest = self.work.require_outline()
        if not 1 <= number <= len(outline.chapters):
            return
        kept = outline.chapters[number - 1]
        scenes = _scene_dicts(details, kept.id, _cast_ids(self.state.cast))
        if not scenes:
            return
        chapter = Chapter.model_validate(
            {"id": kept.id, "title": kept.title, "act": kept.act, "scenes": scenes}
        )
        chapters = list(outline.chapters)
        chapters[number - 1] = chapter
        updated_outline = Outline(chapters=chapters)
        self.work.save_outline(updated_outline, room_id=self.room_id, expected_digest=digest)
        all_proposals = [p for p, _ in self.work.proposals()[0]]
        codex_now = effective_codex(self.work.codex(), all_proposals)
        detail_trans = parse_detail_transitions(
            {"scenes": details}, updated_outline, number, self.state.cast, codex_now
        )
        if self.state.planned_transitions or detail_trans:
            replace_scenes = {s.id for s in chapter.scenes} if detail_trans else None
            merged = merge_transitions(
                self.state.planned_transitions,
                detail_trans,
                replace_scenes=replace_scenes,
            )
            validated, report = validate_kg_trajectory(
                updated_outline,
                codex_now,
                merged,
                arc=_arc_of(self.state.arc),
                cast_lines=self.state.cast,
            )
            self._save(
                planned_transitions=[t.record() for t in validated],
                kg_trajectory=report.record(),
            )


__all__ = [
    "ABSTRACT_ITEM_WORDS",
    "DETAILS_MARK",
    "DETAILS_REVISE_MARK",
    "KG_MARK",
    "MAX_PLOT_ITEMS",
    "MAX_PLOT_THREADS",
    "_RunDetailsMixin",
    "_detail_scenes",
    "_kg_params",
    "_kg_prompt",
    "_with_entry_fields",
    "abstract_item",
]
