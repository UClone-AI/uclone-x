# pyright: reportPrivateUsage=false
"""`story_start`: a person's request made into a story, one fixed stage after another.

The Writer started a story by improvising a chain of tool calls -- `story_library create`,
`muse_spark`, `story_outline init`, `story_codex create`, `story_manuscript write` -- and
a small model skipped steps, left the outline as the template's placeholders, and wrote
the first scene before anyone chose what the story was about. Here the order is code:

0. direction -- genre, tone, audience, length, what must be in it and what must not,
   read from the request; only what the request leaves open is filled in, and the
   brief says which fields the request set. A genre the request leaves open is not
   filled silently: code draws one from the muse tables at random, with alternatives,
   and the flow stops for the person to take it or name another;
1. premises -- three, each from its own model call around its own story engine (a
   secret, a rival, a hard choice...) drawn by code, with its own muse cards; each is
   checked for the request's must-haves and rewritten once when it drops one, and the
   three are measured against each other (`premise_similarity`); the flow stops for the
   person to choose one by number. Words that turn all three down, or say what the person
   wants instead, are read by the model (`PREMISE_ANSWER_MARK`) and three new premises are
   drawn from them; the flow stops at the premises again (#2262);
2. cast -- the characters and places of the chosen premise, proposed as new codex entries
   (the `story_codex create` path), which the person approves as any other proposal; each
   character starts with the status the premise gives them (alive, missing, dead, unknown);
3. outline -- chapters and scenes with real beats, from the premise and the cast; each
   chapter has as many scenes as its weight in the arc gives it (`scenes_for`: two for a
   chapter that sets up or winds down, four for the twist or the climax, three otherwise).
   Agreeing settles the plot and proposes its key events -- happenings, and the physical
   objects the outline names, with abstract nouns ("기억", "진실") dropped by code and
   counted (`abstract_item`) -- for the codex;
4. chapter -- once the proposed cast is decided or the person says to go on without
   deciding it, the first chapter's plan is shown (4b), and on "네" its scenes are written
   through `StoryWork.write_scene`, then checked against the codex. Each scene saved (and each scene rewritten after the check)
   is read for what it adds to the codex, as `story_manuscript write` reads one
   (`uclone_x_story.enrich.grow_scene`): the additions wait as proposals for the person,
   and a reading that fails leaves the chapter as written;
4b. chapter plan -- before each chapter is written, the first included, its scenes and
   beats are planned from the locked outline, what the arc says that chapter must do, the
   cast, and how the chapter before it ended; the flow stops to show the plan. Passing
   `changes` revises it (one model call) and it is shown again; `go_on=True` writes that
   chapter the same way and plans the next, until the outline's last chapter is written.
   A new character the accepted plan introduces with a stated relation to one of the cast
   is proposed as a new entry, with the kinship kept on both sides as the cast's is (a
   progression on an approved relative).

The flow stops after stages 1, 2 and 3 for the person, unless the request asks for the
story at once ("바로 써줘"), which runs every stage in the same order with the first
premise and writes the chapter over the cast as proposed -- and says so to the person.
Even then one call writes at most `CHAPTERS_PER_CALL` chapters, and the reply says how to
go on.
At every stop, `go_on=True` takes what the flow recommends -- the genre proposed, the
recommended premise, the cast as proposed -- so agreeing carries a bare request through
each chapter's plan to a written chapter; `choice`, `genre`, `changes`, `must_have`,
`avoid`, and `rewrite` carry the person's explicit decisions without parsing their raw
utterance inside the tool. Every call returns the stage it reached and what it needs next.
What the flow knows is kept in `start.yaml` in the story's folder, not in memory, so a
later call -- in the same conversation or after a restart -- continues where the last one
stopped.

While a start waits at a stop before its chapter loop (`flow_stop`), the outline and the
scenes are this flow's to make: `story_outline` and `story_manuscript` refuse to write them
by hand and name the `story_start` call that goes on (`by_hand_refusal`, #2262).

Each stage asks the model through `StoryModel`, a provider-neutral seam: the tool's
default goes through `invoke_auxiliary_model` on the calling clone's own connector and
budget (P5), and a test passes a fake.
"""

from __future__ import annotations

import logging
import random
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

from uclone_x.agent.reply_lines import NOTE_ASKS_KEY, NOTE_LISTS_KEY
from uclone_x.knowledge.fold import bigram_overlap
from uclone_x.tools.base import BaseTool, artifact_path_from_url
from uclone_x.tools.models import REPLY_NOTE_KEY, ToolContext, ToolResult
from uclone_x.tools.protocols import ToolProtocol
from uclone_x_story import OPEN_STORY_KEY
from uclone_x_story.library import StoryError, StoryLibrary
from uclone_x_story.scene_illustration import SCENE_VISUAL_MARK
from uclone_x_story.schemas import StoryFileError
from uclone_x_story.skill_data import (
    BUNDLED_DATA_ROOT,
    MuseTable,
    SkillDataError,
    StructureTemplate,
    data_roots,
    load_muse_tables_sourced,
    load_structure_templates_sourced,
)
from uclone_x_story.start_details import (
    ABSTRACT_ITEM_WORDS,
    DETAILS_MARK,
    DETAILS_REVISE_MARK,
    KG_MARK,
    _as_thread_of_its_own,
    _with_entry_fields,
    abstract_item,
)
from uclone_x_story.start_notes import (
    _SAME_TURN,
    _kept_note,
    _listed_genres,
    _next_of,
    _not_settled,
    _plain_check,
    _reply_note,
    _shown_findings,
    _style_notes,
    by_hand_refusal,
    flow_stop,
    start_waiting,
    stop_words,
)
from uclone_x_story.start_outline import (
    CHAPTER_PROMPT_MARK,
    REVISE_MARK,
    SPINE_MARK,
    _RunOutlineMixin,
    outline_from,
    scene_title,
)
from uclone_x_story.start_premise import (
    _direction,
    _genre_choice,
    _premises,
    _wanted_from,
    _with_wanted,
    _without_avoided_cast,
)
from uclone_x_story.start_rpg import (
    SCENE_CHOICES_MARK,
    SCENE_CUSTOM_MARK,
    _RunRpgMixin,
)
from uclone_x_story.start_scene import _still_foreign, _without_names
from uclone_x_story.start_state import (
    CHAPTERS_PER_CALL,
    DEFAULTS,
    ENGINES,
    GENRE_LABELS,
    MAX_CAST_TRIES,
    MAX_DRAFTS,
    MAX_PREMISE_BIGRAM,
    MAX_PREMISE_COSINE,
    REPEAT_MARK,
    START_FILE,
    AuxiliaryStoryModel,
    Brief,
    EmbedderFactory,
    ModelFactory,
    PremiseEmbedder,
    Similarity,
    StartState,
    StoryModel,
    StoryStartParams,
    _default_embedder,
    _default_model,
    _dump,
    _genre_turned_down,
    _load_state,
    _texts,
    brief_from,
    cosine,
    genre_label,
    genre_options,
    json_object,
    names_genre,
    premise_similarity,
    scenes_for,
    story_scripts,
    structure_for,
    turned_down_genres,
    wants_interactive,
    wants_it_now,
)
from uclone_x_story.tools import new_entry_draft
from uclone_x_story.work import StoryWork

logger = logging.getLogger(__name__)

__all__ = [
    "ABSTRACT_ITEM_WORDS",
    "AuxiliaryStoryModel",
    "Brief",
    "CHAPTERS_PER_CALL",
    "CHAPTER_PROMPT_MARK",
    "DEFAULTS",
    "DETAILS_MARK",
    "DETAILS_REVISE_MARK",
    "ENGINES",
    "GENRE_LABELS",
    "KG_MARK",
    "MAX_CAST_TRIES",
    "MAX_DRAFTS",
    "MAX_PREMISE_BIGRAM",
    "MAX_PREMISE_COSINE",
    "PremiseEmbedder",
    "REPEAT_MARK",
    "REVISE_MARK",
    "SCENE_CHOICES_MARK",
    "SCENE_CUSTOM_MARK",
    "SCENE_VISUAL_MARK",
    "SPINE_MARK",
    "START_FILE",
    "Similarity",
    "StartState",
    "StoryModel",
    "StoryStartParams",
    "StoryStartTool",
    "_as_thread_of_its_own",
    "_reply_note",
    "_still_foreign",
    "_with_entry_fields",
    "_without_names",
    "abstract_item",
    "bigram_overlap",
    "brief_from",
    "by_hand_refusal",
    "cosine",
    "flow_stop",
    "genre_label",
    "genre_options",
    "json_object",
    "names_genre",
    "new_entry_draft",
    "outline_from",
    "premise_similarity",
    "scene_title",
    "scenes_for",
    "start_waiting",
    "stop_words",
    "story_scripts",
    "structure_for",
    "wants_interactive",
    "wants_it_now",
]


class StoryStartTool(BaseTool[StoryStartParams]):
    """A new story from a person's request, through fixed stages, stopping for the person."""

    name = "story_start"
    description = (
        "Start a new story from what the person asks for, and take it forward one stage at "
        "a time: direction (a genre is proposed when the request names none), three "
        "premises for the person to choose from, cast and places (proposed for approval), "
        "outline (the person's changes revise it; agreeing settles the plot and proposes "
        "its key events for the codex), then each chapter in turn: its scene plan is shown "
        "for the person to change or accept, and accepting writes it. 'start' with the "
        "request word for word; when the person decides at a stop, 'continue' with structured "
        "parameters ('choice', 'genre', 'go_on', 'changes', 'must_have', 'avoid', 'rewrite'). "
        "If the person only asks a question without deciding, answer in chat without calling "
        "this tool. If the person asks to have it written at once, the whole start runs in "
        "one call, one chapter per call."
    )
    params_type = StoryStartParams
    writes_files: ClassVar[bool] = True
    opens_story: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"status"})
    needs_room: ClassVar[bool] = True
    not_run_note: ClassVar[str] = "No story was started or changed."

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        data_roots: Sequence[Path] = (BUNDLED_DATA_ROOT,),
        embedder_factory: EmbedderFactory | None = None,
        rng: random.Random | None = None,
        image_tool: ToolProtocol | None = None,
    ) -> None:
        self._model_factory = model_factory or _default_model
        self._embedder_factory = embedder_factory or _default_embedder
        # Draws the genre proposed when the request names none; a test passes a seeded one.
        self._rng = rng or secrets.SystemRandom()
        self._data_roots = tuple(data_roots)
        self._image_tool = image_tool
        super().__init__(name=self.name, description=self.description, params_type=StoryStartParams)

    async def run(self, params: StoryStartParams, context: ToolContext) -> dict[str, Any]:
        if params.action == "start":
            return await self._start(params, context)
        if params.action == "status":
            work = StoryWork.open_in(context)
            state, _ = _load_state(work)
            return self._result(work, state, done=[])
        story_id, room_id = StoryLibrary.writer_of(context)
        work = StoryWork(StoryLibrary(context.require_workspace()), story_id)
        state, digest = _load_state(work)
        if context.turn_id is not None and context.turn_id == state.stop_turn_id:
            raise StoryError(_SAME_TURN)
        state = state.model_copy(update={"stop_turn_id": context.turn_id})
        if params.action == "leave":
            return self._leave(work, room_id, state, digest, params.request)
        if state.left_request is not None:
            raise StoryError(
                "The person left this story's start to write the story themselves, so it "
                "does not go on. Nothing was changed."
            )
        if params.changes and params.request and params.request.strip():
            # The person's words, word for word, in place of a summary or a placeholder
            # ("premises rejected", live #2370): a redraw or a revision follows what they
            # said (#2371).
            params = params.model_copy(update={"changes": params.request.strip()})
        if params.changes and wants_it_now(params.changes):
            state = state.model_copy(update={"quick": True})
        if params.interactive is not None:
            state = state.model_copy(update={"interactive": params.interactive})
        elif params.changes and wants_interactive(params.changes):
            state = state.model_copy(update={"interactive": True})
        if params.illustrate is not None:
            state = state.model_copy(update={"illustrate": params.illustrate})
        wanted = _wanted_from(params)
        structures = self._structures(context)
        model = self._model_factory(context)
        embedder = self._embedder_factory(context)
        if state.stage == "choose_genre":
            genre = _genre_choice(state, params, wanted)
            run = _Run(
                work,
                room_id,
                context,
                model,
                state,
                digest,
                structures,
                embedder,
                image_tool=self._image_tool,
            )
            await run.take_genre(genre, self._tables(context), embedder, wanted)
            done = ["genre", "premises"]
            if run.state.quick:
                done += await run.forward(choice=None, go_on=True)
            return self._result(work, run.state, done=done)
        if params.changes and not state.outline_saved:
            # A structure the person names before the outline is built wins over the
            # genre's; words that name none change nothing.
            named = structure_for(params.changes, "", structures)
            if named is not None and named[1] == "request":
                state = state.model_copy(
                    update={"structure": named[0], "structure_source": "request"}
                )
        run = _Run(
            work,
            room_id,
            context,
            model,
            state,
            digest,
            structures,
            embedder,
            image_tool=self._image_tool,
        )
        at_premises = state.stage == "choose_premise"
        if at_premises and state.cast_failed is None and params.choice is None and not state.quick:
            changes = (params.changes or "").strip() or ", ".join(wanted["avoid"])
            if not changes and params.go_on:
                # A decision at the premises with wants and no number is what the person
                # wants instead (#2360): refused for want of a number, the Writer sent the
                # same `go_on` and `must_have` 12 times and the stop never moved. New
                # premises are drawn with them, at the premises only: at any other stop a
                # yes with a request settles it. A yes with a request comes with its number
                # and takes that premise. Wants with no `go_on` are still a question (#2270).
                changes = ", ".join(wanted["must_have"])
            if changes:
                genre_raw = (params.genre or "").strip()
                genre = (
                    genre_label(genre_raw, state.brief.language)
                    if genre_raw
                    and not _genre_turned_down(genre_raw, turned_down_genres(wanted["avoid"]))
                    else ""
                )
                tables = self._tables(context)
                await run.redraw_premises(changes, {"genre": genre, **wanted}, tables, embedder)
                return _not_settled(self._result(work, run.state, done=["premises"]), params)
        done = await run.forward(
            choice=params.choice,
            go_on=params.go_on,
            rewrite=params.rewrite,
            changes=params.changes,
            wanted=wanted,
        )
        return _not_settled(self._result(work, run.state, done=done, kept=run.kept), params)

    async def _start(self, params: StoryStartParams, context: ToolContext) -> dict[str, Any]:
        request = (params.request or "").strip()
        if not request:
            raise StoryError(
                "Pass the person's request word for word in 'request', so nothing was started."
            )
        room_id = context.room_id
        if room_id is None:
            raise StoryError("A story is started inside a conversation, so nothing was started.")
        library = StoryLibrary(context.require_workspace())
        model = self._model_factory(context)
        brief = await _direction(model, request)
        tables = self._tables(context)
        quick = wants_it_now(request)
        interactive = (
            params.interactive if params.interactive is not None else wants_interactive(request)
        )
        illustrate = params.illustrate if params.illustrate is not None else True
        # A genre the request names is kept (#1723); otherwise one is proposed, drawn by
        # code, and the flow stops for the person to take it or name another -- unless
        # they asked for the story at once, when the one drawn is used and they are told.
        options = (
            []
            if "genre" in brief.from_request
            else genre_options(sorted(tables), self._rng, brief.language)
        )
        if options:
            brief = brief.model_copy(update={"genre": options[0]})
        record = library.create(
            brief.title,
            room_id,
            genre=brief.genre,
            style_notes=_style_notes(brief),
        )
        released: dict[str, Any] = {}
        if context.story_id is not None and context.story_id != record.story_id:
            try:
                if library.release(context.story_id, room_id):
                    released["released"] = context.story_id
            except StoryError:
                released["previous_story_not_released"] = context.story_id
        work = StoryWork(library, record.story_id)
        structures = self._structures(context)
        picked = structure_for(request, brief.genre, structures)
        state = StartState(
            request=request,
            quick=quick,
            interactive=interactive,
            illustrate=illustrate,
            stage="choose_genre" if options and not quick else "choose_premise",
            brief=brief,
            genre_options=options,
            structure=picked[0] if picked else None,
            structure_source=picked[1] if picked else None,
            stop_turn_id=context.turn_id,
        )
        done = ["direction"]
        embedder = self._embedder_factory(context)
        if state.stage == "choose_premise":
            state = await _premises(model, state, tables, embedder)
            done.append("premises")
        digest = work.write(START_FILE, _dump(state), room_id=room_id, expected_digest=None)
        try:
            work.note_session(room_id)
        except (StoryError, StoryFileError):
            logger.warning("story_start: the session of %s was not recorded", record.story_id)
        run = _Run(
            work,
            room_id,
            context,
            model,
            state,
            digest,
            structures,
            embedder,
            image_tool=self._image_tool,
        )
        if state.quick:
            done.extend(await run.forward(choice=None, go_on=True))
        result = self._result(work, run.state, done=done)
        return {OPEN_STORY_KEY: record.story_id, **released, **result}

    async def execute(
        self,
        params: dict[str, Any] | ToolContext | None = None,
        context: ToolContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute `story_start` and attach any generated scene illustration paths as artifacts."""
        result = await super().execute(params=params, context=context, **kwargs)
        if result.success and isinstance(result.output, dict):
            output_map = cast("dict[str, object]", result.output)
            illus = output_map.get("scene_illustrations")
            if isinstance(illus, Mapping):
                illus_map = cast("Mapping[str, object]", illus)
                paths: list[str] = []
                for url in illus_map.values():
                    path = artifact_path_from_url(url)
                    if path is not None and path not in paths:
                        paths.append(path)
                if paths:
                    return result.model_copy(update={"artifacts": tuple(paths)})
        return result

    @staticmethod
    def _leave(
        work: StoryWork, room_id: str, state: StartState, digest: str, request: str | None
    ) -> dict[str, Any]:
        """The start stopped for good: from here the outline and scenes are written as the
        person asks, with the story tools (#2270). Only with the person's words asking for
        it, kept in `start.yaml`; whether they mean it is not judged here (#2204)."""
        asked_by = (request or "").strip()
        if not asked_by:
            raise StoryError(
                "'leave' stops the start only when the person asks to write the story "
                "themselves: pass their words, word for word, as 'request'. Nothing was "
                "changed."
            )
        left = state.model_copy(update={"left_request": asked_by})
        work.write(START_FILE, _dump(left), room_id=room_id, expected_digest=digest)
        return {
            "story_id": work.story_id,
            "stage": left.stage,
            "left": True,
            "left_request": asked_by,
            "next": "The start is stopped. Write the outline and the scenes as the person "
            "asks, with story_outline and story_manuscript.",
            REPLY_NOTE_KEY: {
                "ko": "단계별로 제안하던 이야기 만들기를 멈췄습니다. 이제부터 개요와 장면은 "
                "말씀하시는 대로 씁니다.",
                "en": "I have stopped the step-by-step start. From here the outline and the "
                "scenes are written as you ask.",
            },
        }

    def _tables(self, context: ToolContext) -> dict[str, MuseTable]:
        try:
            found = load_muse_tables_sourced(data_roots(context.skill_dirs, self._data_roots))
        except SkillDataError as exc:
            logger.warning("story_start could not load the muse tables: %s", exc)
            return {}
        return {genre: sourced.item for genre, sourced in found.items()}

    def _structures(self, context: ToolContext) -> dict[str, StructureTemplate]:
        """The bundled structure templates and those of the clone's active skills."""
        try:
            found = load_structure_templates_sourced(
                data_roots(context.skill_dirs, self._data_roots)
            )
        except SkillDataError as exc:
            logger.warning("story_start could not load the structures: %s", exc)
            return {}
        return {structure_id: sourced.item for structure_id, sourced in found.items()}

    @staticmethod
    def _result(
        work: StoryWork,
        state: StartState,
        *,
        done: list[str],
        kept: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "story_id": work.story_id,
            "stage": state.stage,
            "done_now": done,
            "brief": state.brief.model_dump(mode="json"),
            "premises": [
                {"number": p.number, "title": p.title, "premise": p.premise} for p in state.premises
            ],
        }
        if state.interactive:
            result["interactive"] = True
        if state.genre_options:
            result["genre_options"] = state.genre_options
        if state.premises:
            result["recommended"] = state.recommended
        if state.chosen is not None:
            result["chosen"] = state.chosen
        if state.cast_failed is not None:
            result["cast_failed"] = state.cast_failed
        if state.structure is not None:
            result["structure"] = {"id": state.structure, "source": state.structure_source}
        if state.central_conflict or state.twist:
            result["central_conflict"] = state.central_conflict
            result["twist"] = state.twist
        if state.spine is not None:
            result["spine"] = state.spine
        if state.arc is not None:
            result["arc"] = {**state.arc, "stakes": state.stakes}
        if state.outline_check is not None:
            result["outline_check"] = _plain_check(state.outline_check)
        if state.outline_feedback:
            result["outline_feedback"] = state.outline_feedback
        if state.plot_locked:
            result["plot_locked"] = True
            result["plot_kg"] = state.plot_kg
        if state.plot_kg_dropped:
            result["plot_kg_dropped"] = state.plot_kg_dropped
        if state.cast_proposals:
            result["cast_proposals"] = state.cast_proposals
        if state.premise_similarity is not None:
            result["premise_similarity"] = state.premise_similarity
        if state.scenes_written:
            result["scenes_written"] = state.scenes_written
        if state.scene_illustrations:
            result["scene_illustrations"] = dict(state.scene_illustrations)
        if state.cast_waiting:
            result["cast_waiting"] = state.cast_waiting
        if state.cast_assumed:
            result["cast_assumed"] = state.cast_assumed
        if state.introduced:
            result["introduced"] = state.introduced
        if state.continuity_checked or state.continuity_unread:
            result["continuity"] = state.continuity
            result["continuity_unread"] = state.continuity_unread
            result["plan_unread"] = state.plan_unread
        if state.rewritten:
            result["rewritten"] = state.rewritten
        if state.codex_growth:
            result["codex_growth"] = state.codex_growth
        if state.chapter_written:
            result["chapter_written"] = state.chapter_written
            result["chapter_scenes"] = state.chapter_scenes
        if state.details_chapter is not None:
            result["details_chapter"] = state.details_chapter
            result["chapter_details"] = state.chapter_details
            if state.details_new:
                result["details_new"] = state.details_new
            if state.details_feedback:
                result["details_feedback"] = state.details_feedback
            if state.details_scene_edits:
                result["scene_edits"] = state.details_scene_edits
        if state.scene_id is not None:
            result["scene_chapter"] = state.scene_chapter
            result["scene_id"] = state.scene_id
            result["scene_choices"] = state.scene_choices
            if state.scene_entering_kg:
                result["scene_entering_kg"] = state.scene_entering_kg
            if state.scene_kg_timeline:
                result["scene_kg_timeline"] = state.scene_kg_timeline
        if state.scene_written_last is not None:
            result["scene_written_last"] = state.scene_written_last
        if state.notes:
            result["notes"] = state.notes
        result["next"] = _next_of(state)
        shown = work.outline() if state.stage == "review_outline" else None
        note: dict[str, Any] = dict(_reply_note(state, done, shown[0] if shown else None))
        if state.stage == "choose_premise" and state.cast_failed is None and state.premises:
            # The note lists the premises by title: a reply that already names every one
            # of them gets only the question after the list (`NOTE_LISTS_KEY`, #2301).
            note[NOTE_LISTS_KEY] = [p.title for p in state.premises]
        if state.stage == "choose_genre":
            # The same for the genres, named in each language's own words (#2360: vague T1
            # showed the model's list, then the note's).
            note[NOTE_LISTS_KEY] = _listed_genres(state)
        if state.cast_failed is None and state.stage in (
            "choose_genre",
            "choose_premise",
            "review_scene",
        ):
            # The stop's question is the note's last line; a reply that already asks it,
            # in the same words, does not get it again (`NOTE_ASKS_KEY`, #2305).
            note[NOTE_ASKS_KEY] = [str(note[lang]).splitlines()[-1] for lang in ("ko", "en")]
        kept = kept or {}
        said = _kept_note(kept)
        if said is not None:
            # What taking a stop kept for later is said back first, on a line of its own,
            # so a change request read as taking it is seen (#2313); marked as a question
            # line is, so a reply that says it in the same words does not get it twice.
            ko, en = said
            note["ko"] = f"{ko}\n{note['ko']}" if note["ko"] else ko
            note["en"] = f"{en}\n{note['en']}" if note["en"] else en
            note[NOTE_ASKS_KEY] = [*note.get(NOTE_ASKS_KEY, []), ko, en]
            result["kept_for_later"] = {k: list(v) for k, v in kept.items() if v}
        result[REPLY_NOTE_KEY] = note
        return result


class _Run(_RunRpgMixin, _RunOutlineMixin):
    """One call's walk through the stages after the premises, saving after each."""

    async def take_genre(
        self,
        genre: str,
        tables: Mapping[str, MuseTable],
        embedder: PremiseEmbedder | None,
        wanted: Mapping[str, Any] | None = None,
    ) -> None:
        """The genre the person took: the brief, the story and the structure follow it,
        and then the premises are drawn in it, with what else the person asked for at the
        genre stop (`wanted`, #2305)."""
        state = self.state
        brief = _with_wanted(state.brief.model_copy(update={"genre": genre}), wanted or {})
        update: dict[str, Any] = {
            "brief": brief,
            "stage": "choose_premise",
            "genre_answered": True,
        }
        if state.structure_source != "request":
            picked = structure_for(state.request, genre, self.structures)
            update["structure"] = picked[0] if picked else None
            update["structure_source"] = picked[1] if picked else None
        library = StoryLibrary(self.context.require_workspace())
        library.set_genre(self.work.story_id, self.room_id, genre)
        self.state = await _premises(self.model, state.model_copy(update=update), tables, embedder)
        self._save()

    async def redraw_premises(
        self,
        words: str,
        answer: Mapping[str, Any],
        tables: Mapping[str, MuseTable],
        embedder: PremiseEmbedder | None,
    ) -> None:
        """Three new premises, drawn from what the person said when they turned the last
        three down (§1.1 1: a rejection gets new proposals). A genre they named replaces
        the story's; what they want is kept as a must-have, beside the request's."""
        state = self.state
        brief = state.brief
        genre = str(answer.get("genre") or "")
        must = [*brief.must_have, *cast("list[str]", answer.get("must_have") or [])]
        avoid = [*brief.avoid, *cast("list[str]", answer.get("avoid") or [])]
        stated = [k for k in ("must_have", "avoid") if answer.get(k)] + (["genre"] if genre else [])
        brief = brief.model_copy(
            update={
                "genre": genre or brief.genre,
                "must_have": list(dict.fromkeys(must)),
                "avoid": list(dict.fromkeys(avoid)),
                "from_request": list(dict.fromkeys([*brief.from_request, *stated])),
            }
        )
        update: dict[str, Any] = {
            "brief": brief,
            "premise_feedback": [*state.premise_feedback, words],
            "premises_turned_down": [
                *state.premises_turned_down,
                *(p.title for p in state.premises),
            ],
        }
        if genre and genre != state.brief.genre:
            StoryLibrary(self.context.require_workspace()).set_genre(
                self.work.story_id, self.room_id, genre
            )
            if state.structure_source != "request":
                picked = structure_for(state.request, genre, self.structures)
                update["structure"] = picked[0] if picked else None
                update["structure_source"] = picked[1] if picked else None
        self.state = await _premises(self.model, state.model_copy(update=update), tables, embedder)
        self._save()

    def keep_wanted(self, answer: Mapping[str, Any]) -> None:
        """What the person asked for when they took a premise, the outline, a chapter's plan
        or the waiting cast, kept in the brief for every stage after it (#2305), and in
        `kept` for this call's reply (#2313)."""
        for key in ("must_have", "avoid"):
            # Only what the brief does not hold yet is said back: the Writer passes the same
            # `must_have` at every stop, and the reply said it was added each time (#2360).
            known = set(cast("list[str]", getattr(self.state.brief, key)))
            new = [t for t in _texts(answer.get(key)) if t not in known]
            self.kept[key] = list(dict.fromkeys([*self.kept[key], *new]))
        self._save(brief=_with_wanted(self.state.brief, answer))
        kept_cast = _without_avoided_cast(self.state.cast, self.state.brief.avoid)
        if len(kept_cast) != len(self.state.cast):
            kept_ids = {c.get("id") for c in kept_cast}
            proposals = [
                p
                for c, p in zip(self.state.cast, self.state.cast_proposals, strict=False)
                if c.get("id") in kept_ids
            ]
            self._save(cast=kept_cast, cast_proposals=proposals)

    async def forward(
        self,
        *,
        choice: int | None,
        go_on: bool = False,
        rewrite: bool = False,
        changes: str | None = None,
        wanted: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Run stages until one stops for the person, or all are done; the stages run.

        The chapter waits while a proposed cast entry is still pending, unless the person
        asked for the story at once or said to go on (`go_on`): then it is written over
        the cast as proposed, and the entries it assumed are kept for the reply. A cast
        stage that proposes no character stops the flow before the outline, even when the
        story was asked for at once. Once the chapter is written, the scenes the check
        flagged are rewritten only when the person asks for it (`rewrite`). Agreeing
        (`go_on`) takes the recommended premise; a number the person gave wins over it.

        At the outline stop, `changes` revises the outline and the flow stops there again.
        Settling the plot proposes its key events for the codex (`_lock_plot`), then plans
        the first chapter and shows the plan (or proposes RPG choices for the first scene
        when `interactive=True`).

        Before each chapter, the first included, its plan waits (`review_chapter`): agreeing
        writes it; `changes` revises the plan and the flow stops again, except in quick mode,
        where the revised plan is written in the same call. No call writes more than
        `CHAPTERS_PER_CALL` chapters.
        """
        done: list[str] = []
        while True:
            stage = self.state.stage
            if stage == "choose_premise":
                retry = self.state.chosen if self.state.cast_failed else None
                number = choice if choice is not None else retry
                # A premise is taken only by its number (#2360): `go_on` alone came from a
                # person who turned all three down, and premise 1 was taken. Only a story
                # asked for at once takes the recommended one without being told.
                if number is None and self.state.quick:
                    number = self.state.recommended
                if number is None:
                    raise StoryError(
                        "The person has not chosen a premise yet, and a premise is taken only "
                        "by its number: ask which of the three, and pass it as 'choice' (a "
                        f"bare yes takes the recommended one, {self.state.recommended}). If "
                        "they turned them down, pass what they want instead in 'changes' "
                        "(never 'request'), and new premises are drawn. Nothing was changed."
                    )
                if (changes or "").strip() and not (
                    wanted and (wanted.get("must_have") or wanted.get("avoid"))
                ):
                    wanted = {"must_have": [(changes or "").strip()], "avoid": []}
                if wanted and (wanted.get("must_have") or wanted.get("avoid")):
                    self.keep_wanted(wanted)
                    wanted = None
                self._save(chosen=number)
                await self._cast()
                if self.state.cast_failed:
                    return done
                done.append("cast")
                if not self.state.quick:
                    return done
            elif stage == "review_cast":
                if (changes or "").strip() and not (
                    wanted and (wanted.get("must_have") or wanted.get("avoid"))
                ):
                    wanted = {"must_have": [(changes or "").strip()], "avoid": []}
                if wanted and (wanted.get("must_have") or wanted.get("avoid")):
                    self.keep_wanted(wanted)
                    wanted = None
                await self._outline()
                done.append("outline")
                if not self.state.quick:
                    return done
            elif stage == "review_outline":
                if wanted and (wanted.get("must_have") or wanted.get("avoid")):
                    self.keep_wanted(wanted)
                    wanted = None
                if not self.state.plot_locked:
                    feedback = (changes or "").strip()
                    if feedback and not self.state.quick:
                        await self._revise_outline(feedback)
                        done.append("outline_revised")
                        return done
                    await self._lock_plot()
                    done.append("plot")
                pending = self._pending_cast()
                if pending and not (go_on or self.state.quick):
                    self._save(cast_waiting=pending)
                    return done
                self._save(cast_waiting=[], cast_assumed=pending)
                if self.state.interactive and not self.state.quick:
                    await self._propose_scene_choices()
                    done.append("scene_choices")
                    return done
                # The first chapter is planned and shown before it is written, as every
                # later one is (§1.1 4). A story asked for at once writes it in this call.
                await self._propose_details(1)
                done.append("details")
                if not self.state.quick:
                    return done
                changes = None
            elif stage == "review_chapter":
                if wanted and (wanted.get("must_have") or wanted.get("avoid")):
                    self.keep_wanted(wanted)
                    wanted = None
                if self.state.interactive and not self.state.quick:
                    await self._propose_scene_choices()
                    done.append("scene_choices")
                    return done
                if "chapter" not in done:
                    flagged = _shown_findings(self.state)
                    if rewrite and flagged:
                        await self._rewrite()
                        done.append("rewrite")
                        return done
                    said = (changes or "").strip()
                    if said:
                        if await self._revise_details(said):
                            done.append("scene_edit")
                            return done
                        done.append("details_revised")
                        if not self.state.quick:  # the revised plan is shown first
                            return done
                number = self.state.details_chapter or self.state.chapter_written + 1
                if self.state.details_scene_edits:
                    self._save(
                        details_scene_edits=[],
                        details_scene_edits_failed=[],
                        details_scene_edits_ambiguous=[],
                    )  # answered
                await self._apply_details(number)
                if self._introduce(number):
                    done.append("introduced")
                await self._write_chapter(number, done)
                if not self._writes_on(done):
                    return done
            elif stage == "review_scene":
                if wanted and (wanted.get("must_have") or wanted.get("avoid")):
                    self.keep_wanted(wanted)
                    wanted = None
                if not self.state.interactive:
                    number = self.state.scene_chapter or self.state.chapter_written + 1
                    await self._propose_details(number)
                    done.append("details")
                    return done
                flagged = _shown_findings(self.state)
                if flagged and rewrite:
                    await self._rewrite()
                    done.append("rewrite")
                    if self.state.scene_id:
                        await self._propose_scene_choices(self.state.scene_id)
                    return done
                await self._apply_scene_choice(
                    choice=choice,
                    changes=changes,
                    done=done,
                )
                return done
            else:
                if rewrite and _shown_findings(self.state):
                    await self._rewrite()
                    done.append("rewrite")
                return done

    def _writes_on(self, done: Sequence[str]) -> bool:
        """Whether this call goes on to the next chapter without stopping for the person.

        Only a story asked for at once does, and only up to `CHAPTERS_PER_CALL` chapters
        in one call; the next call writes the next one.
        """
        return (
            self.state.quick
            and self.state.stage == "review_chapter"
            and not self.state.chapter_failed
            and done.count("chapter") < CHAPTERS_PER_CALL
        )
