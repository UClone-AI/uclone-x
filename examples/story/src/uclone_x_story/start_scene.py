# pyright: reportPrivateUsage=false
"""Stage 4 (Scene prose writing, bottom-up KG enrichment) and Stage 5 (Continuity/plan check and rewrite)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from uclone_x.i18n.scripts import letter_scripts, script_of, words_outside
from uclone_x_story.context import CodexIndex, scene_context
from uclone_x_story.continuity import Finding, checked_store, findings_from
from uclone_x_story.enrich import grow_scene
from uclone_x_story.kg_plan import effective_codex, kg_scene_guidance
from uclone_x_story.library import StoryError
from uclone_x_story.plan_check import (
    absent_cast,
    asks_model,
    plan_findings,
    plan_prompt,
    scene_threads,
)
from uclone_x_story.prose import REPEAT_SHARE, prose_refusal, repeated_share
from uclone_x_story.scene_illustration import illustrate_scene
from uclone_x_story.schemas import Chapter, Outline, Proposal, Scene, StoryFileError
from uclone_x_story.start_details import _RunDetailsMixin
from uclone_x_story.start_notes import _shown_findings
from uclone_x_story.start_premise import _said
from uclone_x_story.start_state import (
    _SYSTEM,
    LANGUAGE_NAMES,
    MAX_DRAFTS,
    MAX_SCENES_WRITTEN,
    REPEAT_MARK,
    _arc_of,
    _brief_lines,
    _text,
    _texts,
    story_scripts,
)


def _without_names(text: str, names: Sequence[str]) -> str:
    """`text` with each of `names` taken out where it stands as a whole name: longest
    first, case folded, and only where neither neighbour is a letter of the name's own
    script, so "Ann" leaves "Annual" whole while "Harin" leaves "Harin은" (#2381)."""
    for name in sorted(names, key=len, reverse=True):
        if not name.strip():
            continue  # an empty name matches everywhere and would split every word
        own = letter_scripts(name)

        def whole(found: re.Match[str], own: set[str] = own) -> str:
            at, end = found.start(), found.end()
            before = found.string[at - 1] if at > 0 else ""
            after = found.string[end] if end < len(found.string) else ""
            if any(c and script_of(c) in own for c in (before, after)):
                return found.group(0)
            return " "

        text = re.sub(re.escape(name), whole, text, flags=re.IGNORECASE)
    return text


def _still_foreign(title: str, words: Sequence[str]) -> str:
    """The note a scene saved with words outside the story's scripts gets (#2380)."""
    return f"Scene '{title}' still has words outside the story's language: {', '.join(words)}."


def _stale_at(proposal: Proposal, scene_id: str, own: set[str]) -> bool:
    """Whether `proposal` is a pending change an earlier reading of scene `scene_id` made:
    placed at that scene, and not among what its latest reading proposed (`own`)."""
    progression = proposal.change.progression
    return (
        proposal.status == "pending"
        and progression is not None
        and progression.at == scene_id
        and proposal.id not in own
        and any(e.scene_id == scene_id for e in proposal.evidence)
    )


class _RunSceneMixin(_RunDetailsMixin):
    """Stage 4 scene writing, bottom-up codex enrichment, continuity checks, and rewrites."""

    async def _write_chapter(self, number: int, done: list[str]) -> None:
        """Chapter `number` written and checked; then the next chapter's plan, or done.

        A chapter that saved no scene leaves the stage where it was, for the person to
        try again. Otherwise the scenes are checked against the codex (the notes are
        shown, never acted on), and the next chapter of the outline is planned and shown
        (`review_chapter`); after the outline's last chapter the start is done.
        """
        await self._chapter(number)
        if self.state.chapter_failed:
            return
        done.append("chapter")
        await self._check(self.state.chapter_scenes)
        done.append("continuity")
        outline, _ = self.work.require_outline()
        if number < len(outline.chapters):
            await self._propose_details(number + 1)
            done.append("details")
        else:
            self._save(
                stage="done",
                details_chapter=None,
                chapter_details=[],
                details_new=[],
                details_feedback=[],
                details_revision_failed=False,
                details_scene_edits=[],
                details_scene_edits_failed=[],
                details_scene_edits_ambiguous=[],
                details_from_outline=False,
            )

    def _scene_prompt(
        self, outline: Outline, scene: Scene, previous: str | None, cast_text: str
    ) -> str:
        state = self.state
        meta = self.work.record().model_dump(mode="json", exclude={"lease"})
        all_proposals = [p for p, _ in self.work.proposals()[0]]
        rejected = {p.id for p in all_proposals if p.status == "rejected"}
        codex = effective_codex(
            self.work.codex(), all_proposals, state.planned_transitions, rejected=rejected
        )
        bundle = scene_context(
            outline, scene.id, codex, previous_text=previous, story=meta, request=None
        )
        earlier_lines: list[str] = []
        for _, sc in outline.scenes_in_order():
            if sc.id == scene.id:
                break
            if sc.summary:
                earlier_lines.append(f"- [{sc.id}] {sc.title}: {sc.summary}")
        earlier_block = (
            "Earlier scenes already written (do not repeat their realizations or inner "
            "monologues):\n" + "\n".join(earlier_lines) + "\n\n"
            if earlier_lines
            else ""
        )
        guidance = kg_scene_guidance(bundle, codex)
        kg_block = f"{guidance}\n\n" if guidance else ""
        return (
            f"Direction:\n{_brief_lines(state.brief)}\n\n"
            f"Premise: {state.premise().premise}\n\nCast and places:\n"
            f"{cast_text}\n\nWhat to know before writing this scene (JSON):\n"
            f"{json.dumps(bundle, ensure_ascii=False)[:6000]}\n\n"
            f"{earlier_block}{kg_block}"
            f"Write scene '{scene.title}' in full as prose: {scene.summary} "
            f"Beats, in order: {'; '.join(scene.beats)}. Write only the scene's text, "
            "with no headings and no notes about the writing. The scene before is already "
            "written (end_of_previous_scene is how it ends): go on from there, and do not "
            "retell it. Enact this scene's new actions, dialogue, and planned state changes "
            "through concrete events; do not repeat inner monologues or realizations from "
            "earlier scenes."
        )

    def _scene_title(self, scene_id: str) -> str:
        """Scene `scene_id` as a note names it for the person: its title, never its id."""
        try:
            current = self.work.outline()
        except StoryFileError:
            return "a scene"
        located = current[0].find(scene_id) if current else None
        return located[1].title if located is not None else "a scene"

    def _story_names(self) -> list[str]:
        """Every name and alias the story's cast and codex give, in whatever script: a
        name the cast stage gave in another script (#2217) is the story's own (#2380)."""
        names: list[str] = []
        for line in self.state.cast:
            names += [_text(line.get("name")), *_texts(line.get("aliases"))]
        for item in self.work.codex().items:
            names += [item.entry.name, *item.entry.aliases]
        return [n for n in names if n]

    async def _draft(
        self,
        prompt: str,
        previous: str | None = None,
        *,
        scene_id: str,
        notes: list[str],
    ) -> tuple[str, str | None]:
        """A scene's text and `None`, or the last draft and why the prose check refused it.

        A draft that repeats `previous`, the scene before (`repeated_share` at
        `REPEAT_SHARE` or more), is drafted once more, told to go on from where that scene
        ended (#2307: qwen3:8b pasted the scene before whole into 7 of 36 scenes). The new
        draft is kept when the prose check passes it and it repeats less; otherwise the
        first is.
        """
        text = ""
        refused: str | None = "empty"
        asked = prompt
        for _ in range(MAX_DRAFTS):
            text = (
                await self.model.complete(asked, system=_SYSTEM, temperature=0.8, max_tokens=3000)
            ).strip()
            refused = prose_refusal(text) if text else "empty"
            if refused is None:
                break
            asked = (
                f"{prompt}\n\nYour last draft was not saved. Fix this and write the "
                f"whole scene again:\n{refused if text else 'The draft was empty.'}"
            )
        scripts = story_scripts(self.state.brief.language, _said(self.state))
        # The story's own names and aliases in another script (#2217) are not the words
        # this check is for; only they are taken out, so a sentence beside them is still
        # caught (#2380).
        names = [n for n in self._story_names() if words_outside(n, scripts)]
        foreign = words_outside(_without_names(text, names), scripts) if refused is None else []
        if refused is None and foreign:
            # A draft with words in another script is drawn once more, told the words, as a
            # premise is (#2301, #2378); the one with fewer such words is kept.
            again = (
                await self.model.complete(
                    f"{prompt}\n\nA draft of this scene had words outside "
                    f"{LANGUAGE_NAMES[self.state.brief.language]}: {', '.join(foreign)}. "
                    "Write every word in it, and write the whole scene again.",
                    system=_SYSTEM,
                    temperature=0.8,
                    max_tokens=3000,
                )
            ).strip()
            if (
                again
                and prose_refusal(again) is None  # a redraft the prose check refuses is not kept
                and len(words_outside(_without_names(again, names), scripts)) < len(foreign)
            ):
                text = again
            still = words_outside(_without_names(text, names), scripts)
            if still:
                notes.append(_still_foreign(self._scene_title(scene_id), still))
        if refused is not None or not previous:
            return text, refused
        share = repeated_share(previous, text)
        if share < REPEAT_SHARE:
            return text, None
        again = (
            await self.model.complete(
                f"{prompt}\n\n{REPEAT_MARK} That scene is already written. Write this "
                "scene's own events, going on from where the scene before ended, without "
                "retelling it.",
                system=_SYSTEM,
                temperature=0.8,
                max_tokens=3000,
            )
        ).strip()
        if again and prose_refusal(again) is None and repeated_share(previous, again) < share:
            return again, None
        return text, None

    async def _grow(self, scene_id: str, text: str, notes: list[str]) -> None:
        """The scene just saved, read for what it adds to the codex (§1.1 row 7).

        The same reading `story_manuscript write` does (`grow_scene`), and the only one:
        its facts are what the check after writing reads (`_check`). The additions are
        pending proposals, never applied here; a "yes" that carries the flow on leaves
        them pending. A reading that fails proposes nothing and stops nothing: the scene
        stays written, and the note says so to the model, not to the person.
        """
        so_far = self.grown or []
        self.grown = so_far
        grown = await grow_scene(
            self.work,
            model=self.model,
            scene_id=scene_id,
            text=text,
            room_id=self.room_id,
            agent_id=self.context.agent_id,
        )
        self.readings[scene_id] = grown
        if grown is None:
            notes.append(
                f"Scene '{self._scene_title(scene_id)}' was saved, but it could not be read for what it adds "
                "to the codex or checked against it."
            )
            return
        self.grown = [*so_far, *(p.id for p in grown.proposed)]

    async def _illustrate_written_scene(self, scene_id: str, text: str) -> Outline:
        """Generate a key-frame illustration for `scene_id` when `state.illustrate` is enabled
        and an image engine is ready, persisting the URL in `outline.yaml` and `start.yaml`."""
        outline, outline_digest = self.work.require_outline()
        if not self.state.illustrate:
            return outline
        all_proposals = [p for p, _ in self.work.proposals()[0]]
        rejected = {p.id for p in all_proposals if p.status == "rejected"}
        codex = effective_codex(
            self.work.codex(),
            all_proposals,
            self.state.planned_transitions,
            rejected=rejected,
        )
        url = await illustrate_scene(
            model=self.model,
            image_tool=self.image_tool,
            context=self.context,
            story_id=self.work.story_id,
            outline=outline,
            codex=codex,
            cast_lines=self.state.cast,
            scene_id=scene_id,
            text=text,
            genre=self.state.brief.genre,
            style_notes=self.state.brief.tone,
        )
        if not url:
            return outline
        updated_chapters: list[Chapter] = []
        for ch in outline.chapters:
            new_scenes = [
                sc.model_copy(update={"illustration": url}) if sc.id == scene_id else sc
                for sc in ch.scenes
            ]
            updated_chapters.append(ch.model_copy(update={"scenes": new_scenes}))
        updated_outline = outline.model_copy(update={"chapters": updated_chapters})
        try:
            self.work.save_outline(
                updated_outline, room_id=self.room_id, expected_digest=outline_digest
            )
            outline = updated_outline
        except (StoryError, StoryFileError):
            pass
        self.state = self.state.model_copy(
            update={"scene_illustrations": {**self.state.scene_illustrations, scene_id: url}}
        )
        return outline

    async def _chapter(self, number: int = 1) -> None:
        """Stage 4: chapter `number`'s scenes, written through the manuscript path.

        The first scene is written after the end of the chapter before, so the story
        runs on from where it stopped.
        """
        state = self.state
        outline, _ = self.work.require_outline()
        chapter = outline.chapters[number - 1]
        notes = list(state.notes)
        written = list(state.scenes_written)
        previous: str | None = self._last_text(outline, number - 1)
        cast_text = self._cast_text()
        for scene in chapter.scenes[:MAX_SCENES_WRITTEN]:
            if scene.id in written:
                found = self.work.manuscript(scene.id)
                previous = found.text if found else previous
                continue
            text, refused = await self._draft(
                self._scene_prompt(outline, scene, previous, cast_text),
                previous,
                scene_id=scene.id,
                notes=notes,
            )
            if refused is not None:
                notes.append(
                    f"Scene '{scene.title}' was not written after {MAX_DRAFTS} drafts: {refused}"
                )
                break
            self.work.write_scene(
                scene.id,
                text,
                room_id=self.room_id,
                expected_digest=None,
                agent_id=self.context.agent_id,
                turn_index=self.context.turn_index,
            )
            try:
                self.work.note_session(self.room_id, scene_written=scene.id)
            except (StoryError, StoryFileError):
                notes.append(f"Scene '{scene.title}' was written, but the session was not noted.")
            written.append(scene.id)
            await self._grow(scene.id, text, notes)
            outline = await self._illustrate_written_scene(scene.id, text)
            previous = text
        this_chapter = [s.id for s in chapter.scenes if s.id in written]
        if not this_chapter:
            self._save(scenes_written=written, notes=notes, chapter_failed=True)
            return
        self._save(
            scenes_written=written,
            notes=notes,
            chapter_failed=False,
            chapter_written=number,
            chapter_scenes=this_chapter,
        )

    async def _check(self, scene_ids: Sequence[str]) -> None:
        """Each scene's facts checked against the story, then the scene held to its plan
        and to the settled plot (`_held_to_plan`); nothing is rewritten.

        The facts are the scene's one reading (`_grow`), so this asks the model nothing
        more than the plan check does. The check is a query of the story's store in its
        approved-and-proposed view (`continuity.findings_from`): what earlier scenes wrote
        and the cast the chapter went on with are pending proposals there, so a death an
        earlier scene wrote, or an item it gave someone, holds this scene to it before a
        person approves it. A scene whose reading could not be read is listed as not
        checked, never as consistent.
        """
        outline, _ = self.work.require_outline()
        axioms, are_defaults = self.work.axioms()
        kept = [f for f in self.state.continuity if f.get("scene_id") not in scene_ids]
        checked = [s for s in self.state.continuity_checked if s not in scene_ids]
        unread = [s for s in self.state.continuity_unread if s not in scene_ids]
        not_held = [s for s in self.state.plan_unread if s not in scene_ids]
        codex = self.work.codex()
        proposals = [p for p, _ in self.work.proposals()[0]]
        store = checked_store(self.work.story_id, outline, codex, proposals)
        for scene_id in scene_ids:
            found = self.work.manuscript(scene_id)
            if found is None:
                continue
            reading = self.readings.get(scene_id)
            if reading is None or reading.unread:
                unread.append(scene_id)
            else:
                # What an earlier reading of this scene proposed describes a text that was
                # replaced; the scene is held to what the rest of the story says.
                own = {p.id for p in reading.proposed}
                current = [p for p in proposals if not _stale_at(p, scene_id, own)]
                scene_store = (
                    store
                    if len(current) == len(proposals)
                    else checked_store(self.work.story_id, outline, codex, current)
                )
                findings: list[Finding] = findings_from(
                    store=scene_store,
                    outline=outline,
                    axioms=axioms,
                    axioms_are_defaults=are_defaults,
                    scene_id=scene_id,
                    text=found.text,
                    facts=reading.facts,
                )
                kept += [f.record() for f in findings]
                checked.append(scene_id)
            held, read = await self._held_to_plan(outline, store.codex(), scene_id, found.text)
            kept += [f.record() for f in held]
            if not read:
                not_held.append(scene_id)
        self._save(
            continuity=kept,
            continuity_checked=checked,
            continuity_unread=unread,
            plan_unread=not_held,
        )

    async def _held_to_plan(
        self, outline: Outline, codex: CodexIndex, scene_id: str, text: str
    ) -> tuple[list[Finding], bool]:
        """What scene `scene_id` leaves out of its plan, adds to it, or tells too early.

        Code finds a planned character the scene never names (`absent_cast`). One call
        (temperature 0) reads the planned beats, the twist while it must stay hidden, and
        unplanned turns, and code keeps only what the reply shows (`plan_findings`). A
        scene with no beats and no twist to keep is not sent. The flag is false when the
        reply could not be read: code's findings are kept, and the scene is listed as not
        read against its plan, never as keeping it.
        """
        located = outline.find(scene_id)
        if located is None:
            return [], True
        chapter, scene = located
        number = next(n for n, c in enumerate(outline.chapters, start=1) if c.id == chapter.id)
        arc = _arc_of(self.state.arc)
        where: dict[str, Any] = {
            "chapter": number,
            "twist": self.state.twist,
            "twist_chapter": arc.twist_chapter if arc else None,
        }
        findings = absent_cast(codex, scene, text)
        if not asks_model(scene, **where):
            return findings, True
        names = {item.entry.id: item.entry.name for item in codex.items}
        scene_transitions: list[str] = []
        for tr in self.state.planned_transitions:
            if tr.get("scene_id") != scene_id:
                continue
            raw_set = tr.get("set")
            set_map = cast("dict[str, Any]", raw_set) if isinstance(raw_set, dict) else {}
            raw_rel = tr.get("relations")
            rel_map = cast("dict[str, Any]", raw_rel) if isinstance(raw_rel, dict) else {}
            parts = (
                [f"{k}={v}" for k, v in set_map.items()]
                + [f"relation({names.get(k, k)})={v}" for k, v in rel_map.items()]
                + ([str(tr["note"])] if tr.get("note") else [])
            )
            entity_label = names.get(str(tr.get("entity_id")), str(tr.get("entity_id")))
            scene_transitions.append(f"{entity_label}: {', '.join(parts)}")
        reply = await self.model.complete(
            plan_prompt(
                scene,
                text,
                transitions=scene_transitions,
                threads=scene_threads(codex, scene_id),
                **where,
            ),
            system="You compare fiction with its plan. Reply with JSON only.",
            temperature=0.0,
            max_tokens=1500,
        )
        try:
            return findings + plan_findings(scene, text, reply, **where), True
        except ValueError:
            return findings, False

    async def _rewrite_asked(self, edits: Sequence[Mapping[str, str]], feedback: str) -> None:
        """The written scenes the person's words ask to change, each written again with
        those words and saved over itself, then checked again (#2361).

        A live Writer, told to edit such a scene by hand, failed to replace it (no digest,
        then a stale one) and wrote its new text into the next chapter's first scene
        instead. Here the scene is the one the words were matched to by title, and it is
        replaced at the digest it was read at. A draft the prose check refuses leaves the
        scene as it was (`details_scene_edits_failed`).
        """
        outline, _ = self.work.require_outline()
        cast_text = self._cast_text()
        notes = list(self.state.notes)
        order = [s.id for _, s in outline.scenes_in_order()]
        done: list[str] = []
        failed: list[str] = []
        for edit in edits:
            scene_id = edit["id"]
            located = outline.find(scene_id)
            current = self.work.manuscript(scene_id)
            if located is None or current is None:
                failed.append(scene_id)
                continue
            _, scene = located
            at = order.index(scene_id)
            before = self.work.manuscript(order[at - 1]) if at > 0 else None
            prompt = (
                f"{self._scene_prompt(outline, scene, before.text if before else None, cast_text)}"
                f"\n\nThis scene was written before. The person asks for it to be written "
                f"again, in their words:\n{feedback}\nWrite the whole scene again as they ask, "
                f"keeping what they did not ask to change.\n\nThe scene as written:\n"
                f"{current.text}"
            )
            text, refused = await self._draft(
                prompt, before.text if before else None, scene_id=scene_id, notes=notes
            )
            if refused is not None:
                notes.append(f"Scene '{scene.title}' was not written again: {refused}")
                failed.append(scene_id)  # every draft refused: the scene stays as it was
                continue
            self.work.write_scene(
                scene_id,
                text,
                room_id=self.room_id,
                expected_digest=current.digest,
                agent_id=self.context.agent_id,
                turn_index=self.context.turn_index,
            )
            done.append(scene_id)
            await self._grow(scene_id, text, notes=notes)  # read again, as any rewrite is
            outline = await self._illustrate_written_scene(scene_id, text)
        rewritten = list(self.state.rewritten)
        self._save(
            rewritten=[*rewritten, *(s for s in done if s not in rewritten)],
            details_scene_edits_failed=failed,
            notes=notes,
        )
        if done:
            await self._check(done)  # the scenes asked about, checked again

    async def _rewrite(self) -> None:
        """The flagged scenes written again, told what disagreed, then checked again.

        Run only when the person asks for it after reading the findings. The scene
        replaced is kept in the manuscript's history by `write_scene`.
        """
        outline, _ = self.work.require_outline()
        cast_text = self._cast_text()
        notes = list(self.state.notes)
        shown = _shown_findings(self.state)
        flagged = list(dict.fromkeys(f["scene_id"] for f in shown))
        rewritten = list(self.state.rewritten)
        done: list[str] = []
        for scene_id in flagged:
            located = outline.find(scene_id)
            current = self.work.manuscript(scene_id)
            if located is None or current is None:
                continue
            _, scene = located
            order = [s.id for _, s in outline.scenes_in_order()]
            at = order.index(scene_id)
            before = self.work.manuscript(order[at - 1]) if at > 0 else None
            disagree = "\n".join(f"- {f['note']}" for f in shown if f["scene_id"] == scene_id)
            prompt = (
                f"{self._scene_prompt(outline, scene, before.text if before else None, cast_text)}"
                f"\n\nThis scene was written before, but these lines disagree with the "
                f"story's codex or with the scene's plan:\n{disagree}\nWrite the whole scene "
                f"again so that it agrees with the codex and carries out the plan above, "
                f"keeping its events otherwise.\n\nThe scene as written:\n"
                f"{current.text}"
            )
            text, refused = await self._draft(
                prompt, before.text if before else None, scene_id=scene_id, notes=notes
            )
            if refused is not None:
                notes.append(f"Scene '{scene.title}' was not rewritten: {refused}")
                continue
            self.work.write_scene(
                scene_id,
                text,
                room_id=self.room_id,
                expected_digest=current.digest,
                agent_id=self.context.agent_id,
                turn_index=self.context.turn_index,
            )
            done.append(scene_id)
            await self._grow(scene_id, text, notes)
            outline = await self._illustrate_written_scene(scene_id, text)
        self._save(rewritten=[*rewritten, *(s for s in done if s not in rewritten)], notes=notes)
        if done:
            await self._check(done)


__all__ = [
    "_RunSceneMixin",
    "_stale_at",
    "_still_foreign",
    "_without_names",
]
