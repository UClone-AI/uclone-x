# pyright: reportPrivateUsage=false
"""RPG-style scene-by-scene interactive story building with dynamic temporal KG updates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import ValidationError

from uclone_x.knowledge.fold import bigram_overlap
from uclone_x.knowledge.quotes import folded
from uclone_x_story.arc import Spine
from uclone_x_story.context import CodexIndex
from uclone_x_story.kg_plan import (
    SceneTransition,
    chronological_kg_log,
    effective_codex,
    entering_world_state,
    format_chronological_kg_log,
    merge_transitions,
    parse_scene_transitions,
    validate_kg_trajectory,
)
from uclone_x_story.library import StoryError
from uclone_x_story.names import id_for_name
from uclone_x_story.proposals import new_entry_clash
from uclone_x_story.schemas import (
    Chapter,
    CodexKind,
    Outline,
    Proposal,
    Scene,
    StoryFileError,
    status_of,
)
from uclone_x_story.start_cast import _cast_ids
from uclone_x_story.start_details import (
    _as_thread_of_its_own,
    _with_entry_fields,
    abstract_item,
)
from uclone_x_story.start_outline import _SCENE_SHAPE, scene_title
from uclone_x_story.start_premise import _feedback_with_brief
from uclone_x_story.start_scene import _RunSceneMixin
from uclone_x_story.start_state import (
    LANGUAGE_NAMES,
    _arc_of,
    _ask_json,
    _brief_lines,
    _ids,
    _items,
    _text,
    _texts,
)
from uclone_x_story.tools import NewEntryParams, new_entry_draft

#: First words of the prompt that generates 3 RPG scene choices grounded in the temporal KG.
SCENE_CHOICES_MARK = "Propose three distinct RPG choices for scene"
#: First words of the prompt that adapts a scene plan and KG transitions to the person's input.
SCENE_CUSTOM_MARK = "Adapt the scene plan and KG transitions for scene"

_DEFAULT_APPROACHES: Mapping[str, tuple[str, str, str]] = {
    "ko": ("정면 돌파 / 행동", "탐색 / 단서 추적", "대화 / 관계 변화"),
    "en": ("Direct / Tactical", "Investigation / Discovery", "Dialogue / Relational"),
}

_VALID_DYNAMIC_KINDS: frozenset[str] = frozenset({"characters", "places", "items", "threads"})


def _normalize_dynamic_entries(raw: object) -> list[dict[str, Any]]:
    """Parse optional `new_entries` proposed alongside an RPG scene choice or custom action."""
    out: list[dict[str, Any]] = []
    for item in _items(raw):
        if not isinstance(item, dict):
            continue
        entry = cast("dict[str, Any]", item)
        name = _text(entry.get("name"))
        kind = _text(entry.get("kind")).casefold()
        if kind in ("character", "person"):
            kind = "characters"
        elif kind in ("place", "location"):
            kind = "places"
        elif kind in ("item", "object"):
            kind = "items"
        elif kind in ("thread", "event", "secret"):
            kind = "threads"
        if not name or kind not in _VALID_DYNAMIC_KINDS:
            continue
        if kind == "items" and abstract_item(name):
            continue
        profile = _text(entry.get("profile") or entry.get("summary") or name)
        out.append(
            {
                "kind": kind,
                "name": name,
                "profile": profile or name,
                **(
                    {"characters": _texts(entry.get("characters"))}
                    if entry.get("characters")
                    else {}
                ),
            }
        )
    return out


def _normalize_choice_dict(
    raw: Mapping[str, Any],
    number: int,
    fallback_scene: Scene,
    cast_names: Mapping[str, str],
    default_approach: str,
) -> dict[str, Any]:
    """Normalize one RPG choice object from the model's JSON reply."""
    summary = _text(raw.get("summary")) or fallback_scene.summary
    title = scene_title(_text(raw.get("title")), summary) or fallback_scene.title
    raw_approach = _text(raw.get("approach"))
    approach = raw_approach if 0 < len(raw_approach) <= 28 else default_approach
    beats = _texts(raw.get("beats"))[:4] or list(fallback_scene.beats)
    if len(beats) == 1:
        beats = [beats[0], summary or title]
    chars = _texts(raw.get("characters")) or [
        cast_names.get(cid, cid) for cid in fallback_scene.characters if cid in cast_names
    ]
    places = _texts(raw.get("places")) or [
        cast_names.get(pid, pid) for pid in fallback_scene.places if pid in cast_names
    ]
    raw_trans = _items(raw.get("transitions"))
    transitions = [cast("dict[str, Any]", t) for t in raw_trans if isinstance(t, dict)]
    new_entries = _normalize_dynamic_entries(raw.get("new_entries"))
    return {
        "number": number,
        "title": title,
        "approach": approach,
        "summary": summary,
        "beats": beats,
        "characters": chars,
        "places": places,
        "transitions": transitions,
        **({"new_entries": new_entries} if new_entries else {}),
    }


def _fallback_choices(
    scene: Scene,
    cast_names: Mapping[str, str],
    language: str,
    planned_transitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Three deterministic fallback choices built from the outline's scene when a model reply
    cannot be parsed."""
    approaches = _DEFAULT_APPROACHES.get(language, _DEFAULT_APPROACHES["en"])
    chars = [cast_names.get(cid, cid) for cid in scene.characters if cid in cast_names]
    places = [cast_names.get(pid, pid) for pid in scene.places if pid in cast_names]
    scene_trans = [
        {
            "subject": cast_names.get(str(t.get("entity_id")), str(t.get("entity_id"))),
            "set": dict(cast("dict[str, Any]", t.get("set") or {})),
            "relations": dict(cast("dict[str, Any]", t.get("relations") or {})),
            **({"note": str(t["note"])} if t.get("note") else {}),
        }
        for t in planned_transitions
        if t.get("scene_id") == scene.id
    ]
    base_beats = list(scene.beats) if len(scene.beats) >= 2 else [scene.summary, scene.title]
    return [
        {
            "number": idx,
            "title": scene.title,
            "approach": approaches[idx - 1],
            "summary": scene.summary,
            "beats": base_beats,
            "characters": chars,
            "places": places,
            "transitions": scene_trans,
        }
        for idx in (1, 2, 3)
    ]


def is_unplanned_death_proposal(
    proposal: Proposal,
    outline: Outline | None,
    planned_transitions: Sequence[Mapping[str, Any]],
) -> bool:
    """Whether `proposal` is an unapproved (`pending`) bottom-up `status=dead` extraction
    that was not planned in `planned_transitions` and contradicts `outline` (the character
    was not in that scene's cast or is scheduled to appear in a later scene)."""
    prog = proposal.change.progression
    if proposal.status != "pending" or prog is None:
        return False
    if status_of(prog.set.get("status")) != "dead":
        return False
    at_scene = prog.at
    eid = proposal.entry_id
    for tr in planned_transitions:
        if tr.get("scene_id") == at_scene and tr.get("entity_id") == eid:
            raw_set = tr.get("set")
            if isinstance(raw_set, Mapping):
                typed_set = cast(Mapping[str, object], raw_set)
                if status_of(typed_set.get("status")) == "dead":
                    return False
    if outline is None:
        return False
    ordered = [sc for _, sc in outline.scenes_in_order()]
    at_idx = next((idx for idx, sc in enumerate(ordered) if sc.id == at_scene), None)
    if at_idx is None:
        return False
    in_current_cast = eid in ordered[at_idx].characters
    in_later_cast = any(eid in sc.characters for sc in ordered[at_idx + 1 :])
    return (not in_current_cast) or in_later_cast


def ensure_choice_transition(
    scene_trans: Sequence[SceneTransition],
    *,
    scene_id: str,
    lead_id: str | None,
    title: str,
    summary: str,
) -> list[SceneTransition]:
    """Ensure the person's enacted RPG choice (`title` / `summary`) is recorded in
    `scene_trans` on `lead_id` at `scene_id` so every scene decision persists in the KG."""
    decision = (
        f"{title} — {summary}" if title and summary and title not in summary else (summary or title)
    ).strip()
    if not lead_id or not decision:
        return list(scene_trans)
    out: list[SceneTransition] = []
    found_lead = False
    for tr in scene_trans:
        if tr.scene_id == scene_id and tr.entity_id == lead_id:
            found_lead = True
            updated_set = dict(tr.set)
            updated_set.setdefault("milestone", decision)
            updated_note = tr.note or decision
            out.append(
                SceneTransition(
                    scene_id=tr.scene_id,
                    entity_id=tr.entity_id,
                    kind=tr.kind,
                    set=updated_set,
                    relations=dict(tr.relations),
                    note=updated_note,
                )
            )
        else:
            out.append(tr)
    if not found_lead:
        out.append(
            SceneTransition(
                scene_id=scene_id,
                entity_id=lead_id,
                kind="characters",
                set={"milestone": decision},
                note=decision,
            )
        )
    return out


def _remaining_pre_rpg_beats(
    scene: Scene,
    written_prose: str,
    cast_names: Mapping[str, str],
) -> list[str]:
    """Filter `scene.beats` for an unwritten scene $s_t$ ($t > 1$) so any pre-RPG beat
    whose entities and text were already enacted in `written_prose` is pruned."""
    if not written_prose.strip():
        return list(scene.beats)
    folded_prose = folded(written_prose)
    paragraphs = [p.strip() for p in written_prose.split("\n") if p.strip()]
    unseen_names = [
        name
        for eid in (*scene.characters, *scene.places)
        if (name := cast_names.get(eid, "")) and folded(name) not in folded_prose
    ]
    kept: list[str] = []
    for beat in scene.beats:
        folded_beat = folded(beat)
        if any(folded(name) in folded_beat for name in unseen_names):
            kept.append(beat)
            continue
        max_overlap = max((bigram_overlap(beat, p) for p in paragraphs), default=0.0)
        if max_overlap < 0.08:
            kept.append(beat)
    return kept


class _RunRpgMixin(_RunSceneMixin):
    """Scene-by-scene RPG interactive story building with dynamic KG updates."""

    def _next_unwritten_scene(self, outline: Outline) -> tuple[int, Scene] | None:
        """Return `(chapter_number, scene)` for the first unwritten scene in reading order."""
        written = set(self.state.scenes_written) | set(self.work.written_scenes())
        for ch_idx, chapter in enumerate(outline.chapters, start=1):
            for scene in chapter.scenes:
                if scene.id not in written:
                    return ch_idx, scene
        return None

    def _previous_scene_text(self, outline: Outline, scene_id: str) -> str | None:
        """Return the prose text of the scene immediately preceding `scene_id`, if written."""
        ordered = [sc.id for _, sc in outline.scenes_in_order()]
        if scene_id not in ordered:
            return None
        at = ordered.index(scene_id)
        for prev_id in reversed(ordered[:at]):
            found = self.work.manuscript(prev_id)
            if found is not None and found.text.strip():
                return found.text
        return None

    def _excluded_proposals(
        self, all_proposals: Sequence[Proposal], outline: Outline | None = None
    ) -> set[str]:
        return {
            p.id
            for p in all_proposals
            if p.status == "rejected"
            or is_unplanned_death_proposal(p, outline, self.state.planned_transitions)
        }

    def _effective_codex_now(self, outline: Outline | None = None) -> CodexIndex:
        if outline is None:
            loaded = self.work.outline()
            outline = loaded[0] if loaded is not None else None
        all_proposals = [p for p, _ in self.work.proposals()[0]]
        rejected = self._excluded_proposals(all_proposals, outline)
        return effective_codex(
            self.work.codex(),
            all_proposals,
            self.state.planned_transitions,
            rejected=rejected,
        )

    def _scene_rpg_context(
        self, outline: Outline, chapter_number: int, scene: Scene
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Build `(prompt_header, entering_kg_str, kg_timeline_list)` for `scene`."""
        state = self.state
        codex_now = self._effective_codex_now(outline)
        entering = entering_world_state(codex_now, outline, scene.id)
        timeline = chronological_kg_log(codex_now, outline, before_scene_id=scene.id)
        chrono_block = format_chronological_kg_log(codex_now, outline, before_scene_id=scene.id)

        arc = _arc_of(state.arc)
        stakes = state.stakes[chapter_number - 1] if 0 < chapter_number <= len(state.stakes) else ""
        arc_note = ""
        if arc is not None and 1 <= chapter_number <= len(arc.functions):
            spine = Spine(
                conflict=state.central_conflict or "",
                twist=state.twist or "",
                clue=(state.spine or {}).get("clue", ""),
                resolution=(state.spine or {}).get("resolution", ""),
            )
            arc_note = (
                f"What the story turns on:\n{spine.told(arc, chapter_number)}\n"
                f"This chapter must {arc.does(chapter_number)}.\n"
                f"Chapter {chapter_number} macro goal (stakes): {stakes}\n"
            )
        elif state.central_conflict:
            arc_note = f"Central conflict: {state.central_conflict}\n"

        completed_lines: list[str] = []
        written_prose_parts: list[str] = []
        for _, sc in outline.scenes_in_order():
            if sc.id == scene.id:
                break
            ms = self.work.manuscript(sc.id)
            ms_text = ms.text.strip() if ms is not None else ""
            if ms_text:
                written_prose_parts.append(ms_text)
            if sc.summary or sc.beats or ms_text:
                beats_str = f" (Beats enacted: {'; '.join(sc.beats)})" if sc.beats else ""
                last_para = (
                    [p.strip() for p in ms_text.split("\n") if p.strip()][-1] if ms_text else ""
                )
                outcome_str = f" [Prose ended at: {last_para[-220:]}]" if last_para else ""
                completed_lines.append(
                    f"- [{sc.id}] {sc.title}: {sc.summary}{beats_str}{outcome_str}"
                )
        completed_block = (
            "Completed scenes & enacted decisions so far (immutable canon — these actions, "
            "departures, and reminiscences have ALREADY happened; do NOT re-ask or repeat them):\n"
            + "\n".join(completed_lines)
            + "\n\n"
            if completed_lines
            else ""
        )

        previous = self._previous_scene_text(outline, scene.id)
        ending = (
            f"End of previous scene (the story is ALREADY at this exact physical moment; "
            f"start all choices AFTER this moment without repeating what just happened):\n"
            f"{previous[-1200:]}\n\n"
            if previous
            else "This is the opening scene of the story.\n\n"
        )
        entering_block = (
            f"Current World State entering [{scene.id}] (do NOT contradict locations, "
            f"possessions, statuses, milestones, or relationships):\n{entering}\n\n"
            if entering
            else ""
        )
        chrono_section = f"{chrono_block}\n\n" if chrono_block else ""
        ch_scenes = (
            outline.chapters[chapter_number - 1].scenes
            if 1 <= chapter_number <= len(outline.chapters)
            else []
        )
        scene_idx = next((idx for idx, s in enumerate(ch_scenes, start=1) if s.id == scene.id), 1)
        ch_total = max(len(ch_scenes), 1)
        cast_names = {str(c.get("id")): str(c.get("name", "")) for c in state.cast}
        if completed_lines:
            written_prose = "\n".join(written_prose_parts)
            rem_beats = _remaining_pre_rpg_beats(scene, written_prose, cast_names)
            char_labels = ", ".join(
                cast_names.get(cid, cid) for cid in scene.characters if cid in cast_names
            )
            place_labels = ", ".join(
                cast_names.get(pid, pid) for pid in scene.places if pid in cast_names
            )
            hooks_line = (
                f"- Remaining chapter hooks not yet enacted: {'; '.join(rem_beats)}\n"
                if rem_beats
                else "- Advance directly to the next obstacle, discovery, or encounter.\n"
            )
            slot_block = (
                f"Target scene slot [{scene.id}] "
                f"(scene {scene_idx} of {ch_total} in chapter {chapter_number}):\n"
                f"- Characters in reach for this scene: {char_labels or 'any active cast'}\n"
                f"- Locations in reach for this scene: {place_labels or 'current or next location'}\n"
                f"{hooks_line}\n"
            )
        else:
            slot_block = (
                f"Baseline outline slot for scene [{scene.id}] "
                f"(scene {scene_idx} of {ch_total} in chapter {chapter_number}):\n"
                f"- Title: {scene.title}\n"
                f"- Baseline summary: {scene.summary}\n"
                f"- Baseline beats: {'; '.join(scene.beats)}\n\n"
            )
        premise = state.premise()
        lang_name = LANGUAGE_NAMES[state.brief.language]
        header = (
            f"Story premise:\n{premise.title}: {premise.premise}\n\n"
            f"Direction (write all natural-language text in {lang_name}):\n"
            f"{_brief_lines(state.brief)}\n\n"
            f"Cast and places:\n{self._cast_text()}\n\n"
            f"{arc_note}\n"
            f"{completed_block}"
            f"{chrono_section}"
            f"{entering_block}"
            f"{ending}"
            f"{slot_block}"
        )
        return header, entering, timeline

    async def _propose_scene_choices(self, scene_id: str | None = None) -> None:
        """Generate 3 RPG scene choices for `scene_id` (or the next unwritten scene) and stop
        at `review_scene`."""
        outline, _ = self.work.require_outline()
        target_chapter: int | None = None
        target_scene: Scene | None = None
        if scene_id is not None:
            for ch_idx, chapter in enumerate(outline.chapters, start=1):
                for sc in chapter.scenes:
                    if sc.id == scene_id:
                        target_chapter, target_scene = ch_idx, sc
                        break
                if target_scene is not None:
                    break
        if target_scene is None or target_chapter is None:
            nxt = self._next_unwritten_scene(outline)
            if nxt is None:
                self._save(
                    stage="done",
                    scene_chapter=None,
                    scene_id=None,
                    scene_choices=[],
                )
                return
            target_chapter, target_scene = nxt

        header, entering, timeline = self._scene_rpg_context(outline, target_chapter, target_scene)
        approaches = _DEFAULT_APPROACHES.get(self.state.brief.language, _DEFAULT_APPROACHES["en"])
        prompt = (
            f"{header}"
            f"{SCENE_CHOICES_MARK} [{target_scene.id}] in chapter {target_chapter}.\n"
            "Act as a narrative Game Master: offer 3 distinct, compelling choices for how "
            "this scene unfolds while steering toward the chapter's macro goal and advancing "
            "open plot threads:\n"
            f"- Choice 1 (approach='{approaches[0]}'): follows the main momentum with direct "
            "action, arrival at the next destination, or confronting an immediate obstacle.\n"
            f"- Choice 2 (approach='{approaches[1]}'): uncovers a concrete clue, secret, or "
            "environmental detail that deepens the mystery or world.\n"
            f"- Choice 3 (approach='{approaches[2]}'): shifts character dynamics (meeting a "
            "key character, alliance, confrontation, negotiation, or emotional turning point).\n"
            "CRITICAL CONTINUITY RULE: The previous scene has ALREADY ended at the exact "
            "physical moment shown in `End of previous scene` (if a character already boarded "
            "a boat, departed, or recalled childhood memories there, they are ALREADY underway). "
            "Never re-propose a decision, departure, or memory that was already enacted in "
            "Completed scenes; start every choice AFTER the end of the previous scene at the "
            "next distinct event, encounter, obstacle, or discovery.\n"
            "Every choice MUST obey the Current World State entering this scene (no character "
            "uses an item they do not possess or acts from an impossible location) and MUST "
            "include concrete KG `transitions` recording what changes in this scene (a new "
            "location, item acquired/used, thread stage, or specific stance/goal/relationship "
            "shift). If a choice introduces a new physical item, place, character, or plot "
            "thread needed for the story, list it in `new_entries`.\n"
            f"{_SCENE_SHAPE}\n"
            'Reply as JSON: {"choices": ['
            '{"number": 1, "title": ..., "approach": ..., "summary": ..., "beats": [...], '
            '"characters": [names], "places": [names], '
            '"transitions": [{"subject": name, "set": {"location" or "possesses" or "stance" '
            'or "goal" or "status": "value"}, "relations": {"other name": "relation"}, '
            '"note": what changes}], '
            '"new_entries": [{"kind": "items" or "places" or "threads" or "characters", '
            '"name": ..., "profile": one sentence}]}]}'
        )
        found = await _ask_json(self.model, prompt, temperature=0.65, max_tokens=3500)
        cast_names = {str(c.get("id")): str(c.get("name", "")) for c in self.state.cast}
        raw_choices = [
            cast("dict[str, Any]", c)
            for c in _items((found or {}).get("choices"))
            if isinstance(c, dict)
        ]
        choices: list[dict[str, Any]] = []
        for idx in (1, 2, 3):
            raw_c = raw_choices[idx - 1] if idx - 1 < len(raw_choices) else {}
            if raw_c:
                choices.append(
                    _normalize_choice_dict(
                        raw_c,
                        idx,
                        target_scene,
                        cast_names,
                        approaches[idx - 1],
                    )
                )
        if len(choices) < 3:
            fallbacks = _fallback_choices(
                target_scene,
                cast_names,
                self.state.brief.language,
                self.state.planned_transitions,
            )
            for idx in range(len(choices) + 1, 4):
                choices.append(fallbacks[idx - 1])

        self._save(
            stage="review_scene",
            scene_chapter=target_chapter,
            scene_id=target_scene.id,
            scene_choices=choices,
            scene_entering_kg=entering,
            scene_kg_timeline=timeline,
        )

    async def _adapt_scene_choice(
        self,
        outline: Outline,
        chapter_number: int,
        scene: Scene,
        base_choice: Mapping[str, Any],
        feedback: str,
    ) -> dict[str, Any]:
        """Adapt `base_choice` (or create a custom scene plan) from the person's `feedback`,
        including any new KG entities or transitions requested."""
        header, _, _ = self._scene_rpg_context(outline, chapter_number, scene)
        base_json = json.dumps(base_choice, ensure_ascii=False, indent=1)
        prompt = (
            f"{header}"
            f"{SCENE_CUSTOM_MARK} [{scene.id}] in chapter {chapter_number} according to the "
            "person's choice or custom instruction.\n\n"
            f"Base option selected:\n{base_json}\n\n"
            f"The person's direction, word for word:\n"
            f"{_feedback_with_brief(feedback, self.state.brief)}\n\n"
            "Build the scene's concrete plan (`title`, `summary`, `beats`, `characters`, "
            "`places`, `transitions`, and optional `new_entries`) so it carries out what the "
            "person asked for without contradicting the Current World State entering this "
            "scene, while keeping the story moving toward the chapter's macro goal and "
            "open plot threads. If the person's instruction introduces a new item, place, "
            "character, or plot thread, include it in `new_entries` and reference it in "
            "`transitions`.\n"
            f"{_SCENE_SHAPE}\n"
            'Reply as JSON: {"title": ..., "summary": ..., "beats": [...], '
            '"characters": [names], "places": [names], '
            '"transitions": [{"subject": name, "set": {"location" or "possesses" or "stance" '
            'or "goal" or "status": "value"}, "relations": {"other name": "relation"}, '
            '"note": what changes}], '
            '"new_entries": [{"kind": "items" or "places" or "threads" or "characters", '
            '"name": ..., "profile": one sentence}]}'
        )
        found = await _ask_json(self.model, prompt, temperature=0.5, max_tokens=2500)
        cast_names = {str(c.get("id")): str(c.get("name", "")) for c in self.state.cast}
        if not found or not (_text(found.get("summary")) or _texts(found.get("beats"))):
            return dict(base_choice)
        return _normalize_choice_dict(
            found,
            int(base_choice.get("number") or 1),
            scene,
            cast_names,
            _text(base_choice.get("approach")) or "Custom",
        )

    def _apply_dynamic_new_entries(
        self,
        new_entries: Sequence[Mapping[str, Any]],
        outline: Outline,
        scene_id: str,
        chapter_number: int,
    ) -> list[dict[str, str]]:
        """Propose any new codex entries (`items`, `places`, `threads`, `characters`) introduced
        during RPG scene progression so the KG grows dynamically before the scene is written."""
        if not new_entries:
            return []
        proposals = [p for p, _ in self.work.proposals()[0]]
        codex_now = self._effective_codex_now()
        cast_lines = list(self.state.cast)
        cast_props = list(self.state.cast_proposals)
        plot_kg = list(self.state.plot_kg)
        known_folded = {folded(str(c.get("name", ""))) for c in cast_lines if c.get("name")} | {
            folded(item.entry.name) for item in codex_now.items
        }
        added_records: list[dict[str, str]] = []
        by_char_name = _cast_ids(cast_lines)

        for raw in new_entries:
            kind_str = _text(raw.get("kind"))
            name = _text(raw.get("name"))
            profile = _text(raw.get("profile")) or name
            if not name or kind_str not in _VALID_DYNAMIC_KINDS:
                continue
            if folded(name) in known_folded:
                continue
            entry_id = id_for_name(name)
            if not entry_id:
                continue
            kind = cast(CodexKind, kind_str)
            state_dict: dict[str, Any] = {}
            extra_fields: dict[str, str | None] = {}
            if kind == "characters":
                state_dict["status"] = "alive"
            elif kind == "threads":
                involved = _ids(raw.get("characters"), "characters", by_char_name)
                if involved:
                    state_dict["involves"] = involved
                last_ch = outline.chapters[-1] if outline.chapters else None
                payoff_sid = last_ch.scenes[-1].id if last_ch and last_ch.scenes else scene_id
                extra_fields = {"planted_in": scene_id, "pay_off_by": payoff_sid}

            params = NewEntryParams(
                kind=kind,
                name=name,
                profile=profile,
                state=state_dict,
            )
            try:
                draft = _with_entry_fields(
                    new_entry_draft(params, room_id=self.room_id, context=self.context),
                    extra_fields,
                )
            except (StoryError, ValidationError):
                continue
            clash = new_entry_clash(self.work, draft.kind, draft.new_entry(), pending=proposals)
            if clash is not None and kind == "threads":
                params = _as_thread_of_its_own(params, self.state.brief.language)
                try:
                    draft = _with_entry_fields(
                        new_entry_draft(params, room_id=self.room_id, context=self.context),
                        extra_fields,
                    )
                except (StoryError, ValidationError):
                    continue
                clash = new_entry_clash(self.work, draft.kind, draft.new_entry(), pending=proposals)
            if clash is not None:
                continue

            proposal = self.work.add_proposal(draft, room_id=self.room_id)
            proposals.append(proposal)
            known_folded.add(folded(params.name))
            record = {
                "proposal": proposal.id,
                "kind": proposal.kind,
                "entry_id": proposal.entry_id,
                "name": params.name,
            }
            added_records.append(record)
            if kind in ("characters", "places"):
                cast_lines.append(
                    {
                        "id": proposal.entry_id,
                        "kind": kind,
                        "name": params.name,
                        "profile": profile,
                        **({"status": "alive"} if kind == "characters" else {}),
                    }
                )
                cast_props.append(proposal.id)
                by_char_name = _cast_ids(cast_lines)
            else:
                plot_kg.append(record)

        if added_records:
            self._save(
                cast=cast_lines,
                cast_proposals=cast_props,
                plot_kg=plot_kg,
            )
        return added_records

    async def _write_single_scene(self, scene_id: str) -> bool:
        """Write only `scene_id` through the manuscript path, run bottom-up KG enrichment
        (`_grow`), and run continuity/plan checks (`_check`). Return True if written."""
        outline, _ = self.work.require_outline()
        located = outline.find(scene_id)
        if located is None:
            return False
        chapter, scene = located
        chapter_no = next(
            (idx for idx, ch in enumerate(outline.chapters, start=1) if ch.id == chapter.id),
            1,
        )
        notes = list(self.state.notes)
        written = list(self.state.scenes_written)
        previous = self._previous_scene_text(outline, scene_id)
        cast_text = self._cast_text()
        text, refused = await self._draft(
            self._scene_prompt(outline, scene, previous, cast_text),
            previous,
            scene_id=scene.id,
            notes=notes,
        )
        if refused is not None:
            notes.append(f"Scene '{scene.title}' was not written after {3} drafts: {refused}")
            self._save(notes=notes, chapter_failed=True)
            return False

        existing = self.work.manuscript(scene.id)
        self.work.write_scene(
            scene.id,
            text,
            room_id=self.room_id,
            expected_digest=existing.digest if existing else None,
            agent_id=self.context.agent_id,
            turn_index=self.context.turn_index,
        )
        try:
            self.work.note_session(self.room_id, scene_written=scene.id)
        except (StoryError, StoryFileError):
            notes.append(f"Scene '{scene.title}' was written, but the session was not noted.")
        if scene.id not in written:
            written.append(scene.id)
        await self._grow(scene.id, text, notes)
        await self._illustrate_written_scene(scene.id, text)
        ch_scenes = [s.id for s in chapter.scenes if s.id in written]
        self._save(
            scenes_written=written,
            notes=notes,
            chapter_failed=False,
            chapter_written=chapter_no,
            chapter_scenes=ch_scenes,
            scene_written_last=scene.id,
        )
        await self._check([scene.id])
        return True

    async def _apply_scene_choice(
        self,
        *,
        choice: int | None,
        changes: str | None,
        done: list[str],
    ) -> None:
        """Apply the chosen RPG option (or custom action in `changes`) to the current scene,
        update the outline and temporal KG trajectory, write the single scene, and propose
        choices for the next scene."""
        outline, digest = self.work.require_outline()
        scene_id = self.state.scene_id
        located = outline.find(scene_id) if scene_id else None
        if located is None:
            nxt = self._next_unwritten_scene(outline)
            if nxt is None:
                self._save(stage="done", scene_id=None, scene_choices=[])
                return
            chapter_no, scene = nxt
            scene_id = scene.id
        else:
            chapter, scene = located
            chapter_no = next(
                (idx for idx, ch in enumerate(outline.chapters, start=1) if ch.id == chapter.id),
                1,
            )

        choices = self.state.scene_choices
        picked_num = choice if choice is not None and 1 <= choice <= len(choices) else 1
        base_choice: dict[str, Any] = (
            dict(choices[picked_num - 1])
            if choices and 1 <= picked_num <= len(choices)
            else {
                "number": 1,
                "title": scene.title,
                "approach": "Direct",
                "summary": scene.summary,
                "beats": list(scene.beats),
                "characters": [],
                "places": [],
                "transitions": [],
            }
        )

        feedback = (changes or "").strip()
        if feedback:
            chosen = await self._adapt_scene_choice(
                outline, chapter_no, scene, base_choice, feedback
            )
            done.append("scene_adapted")
        else:
            chosen = base_choice

        # 1. Dynamically propose any new KG entities introduced by this choice/action
        new_entries = _normalize_dynamic_entries(chosen.get("new_entries"))
        if new_entries:
            added_kg = self._apply_dynamic_new_entries(new_entries, outline, scene.id, chapter_no)
            if added_kg:
                done.append("kg_updated")

        # 2. Update the scene in the outline with the chosen title, summary, beats, cast, places
        by_name = _cast_ids(self.state.cast)
        char_ids = _ids(chosen.get("characters"), "characters", by_name) or list(scene.characters)
        place_ids = _ids(chosen.get("places"), "places", by_name) or list(scene.places)
        beats = _texts(chosen.get("beats"))[:4] or list(scene.beats)
        summary = _text(chosen.get("summary")) or scene.summary
        title = scene_title(_text(chosen.get("title")), summary) or scene.title

        updated_scene = scene.model_copy(
            update={
                "title": title,
                "summary": summary,
                "beats": beats,
                "characters": char_ids,
                "places": place_ids,
            }
        )
        chapters: list[Chapter] = []
        for ch in outline.chapters:
            if any(s.id == scene.id for s in ch.scenes):
                new_scenes = [updated_scene if s.id == scene.id else s for s in ch.scenes]
                chapters.append(ch.model_copy(update={"scenes": new_scenes}))
            else:
                chapters.append(ch)
        updated_outline = Outline(chapters=chapters)
        self.work.save_outline(updated_outline, room_id=self.room_id, expected_digest=digest)

        # 3. Parse and merge the scene's KG transitions and validate the chronological trajectory
        all_proposals = [p for p, _ in self.work.proposals()[0]]
        rejected = self._excluded_proposals(all_proposals, updated_outline)
        codex_base = effective_codex(self.work.codex(), all_proposals, rejected=rejected)
        parsed_trans = parse_scene_transitions(
            chosen.get("transitions"),
            scene.id,
            self.state.cast,
            codex_base,
        )
        lead_id = (
            char_ids[0]
            if char_ids
            else next(
                (
                    str(c.get("id"))
                    for c in self.state.cast
                    if c.get("kind") == "characters" and c.get("id")
                ),
                None,
            )
        )
        scene_trans = ensure_choice_transition(
            parsed_trans,
            scene_id=scene.id,
            lead_id=lead_id,
            title=title,
            summary=summary,
        )
        if self.state.planned_transitions or scene_trans:
            merged = merge_transitions(
                self.state.planned_transitions,
                scene_trans,
                replace_scenes={scene.id} if scene_trans else None,
            )
            validated, report = validate_kg_trajectory(
                updated_outline,
                codex_base,
                merged,
                arc=_arc_of(self.state.arc),
                cast_lines=self.state.cast,
            )
            self._save(
                planned_transitions=[t.record() for t in validated],
                kg_trajectory=report.record(),
            )

        # 4. Write this single scene and run bottom-up KG enrichment & continuity check
        wrote = await self._write_single_scene(scene.id)
        if not wrote:
            return
        done.append("scene")
        done.append("continuity")

        # 5. Move to the next unwritten scene's RPG choices, or finish when all scenes are written
        nxt = self._next_unwritten_scene(updated_outline)
        if nxt is not None:
            await self._propose_scene_choices(nxt[1].id)
            done.append("scene_choices")
        else:
            self._save(
                stage="done",
                scene_chapter=None,
                scene_id=None,
                scene_choices=[],
            )


__all__ = [
    "SCENE_CHOICES_MARK",
    "SCENE_CUSTOM_MARK",
    "_RunRpgMixin",
    "_fallback_choices",
    "_normalize_choice_dict",
    "_normalize_dynamic_entries",
    "ensure_choice_transition",
    "is_unplanned_death_proposal",
]
