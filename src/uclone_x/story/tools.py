"""The Writer's story tools: outline, codex, manuscript and context (#1556).

Each works on the story the call's conversation has open (`ToolContext.story_id`) and
takes no path from the model: a file is named by what it is -- a scene id, a codex entry
id -- and the tool computes where it lives. Reads need only an open story; writes need the
conversation to hold the story's lease, and go through `StoryLibrary.write_file`, so a
file changed by hand after it was read is not overwritten.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from uclone_x.story.context import (
    CodexIndex,
    UnreadableFile,
    last_and_next,
    recap,
    scene_context,
)
from uclone_x.story.library import StoryError, StoryLibrary
from uclone_x.story.proposals import apply_proposal, check_applies, reject_proposal
from uclone_x.story.quotes import MIN_QUOTE_CHARACTERS, quote_found, quote_too_short
from uclone_x.story.schemas import (
    CODEX_KINDS,
    ENTRY_ID_PATTERN,
    CodexEntry,
    CodexKind,
    Outline,
    Proposal,
    SessionsFile,
    StoryFileError,
    describe_invalid,
)
from uclone_x.story.skill_data import (
    SkillDataError,
    Sourced,
    StructureTemplate,
    data_roots,
    load_structure_templates_sourced,
)
from uclone_x.story.work import (
    OUTLINE_FILE,
    SESSIONS_FILE,
    StoryWork,
    manuscript_file,
    proposal_file,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

logger = logging.getLogger(__name__)

__all__ = [
    "StoryAuditTool",
    "StoryCodexTool",
    "StoryContextTool",
    "StoryManuscriptTool",
    "StoryOutlineTool",
    "propose_visual",
]

#: A model's JSON arguments: unknown keys are refused by name; not strict, since a JSON
#: decoder gives lists where a strict tuple would be refused (#1375).
_ARGUMENTS = ConfigDict(frozen=True, extra="forbid")
_ENTRY_ID = re.compile(ENTRY_ID_PATTERN)

#: What an unanswered `apply` says (#1557). The desktop app does not ask during a
#: conversation, so there every apply ends with this sentence; a person decides the
#: proposal in the story's view instead (#1560). No story or file tool reaches that view;
#: a persona with an unconfined shell (Clone's `bash_run`) can call its local API (#1589).
#:
#: The runtime says it before the tool runs, for every shell and every proposal id, so it
#: claims nothing that depends on either (#1584): not that the proposal exists and is
#: kept -- the id may name none -- and not that this shell is the desktop app.
APPLY_NOT_APPROVED_NOTE = (
    "The change was not applied, and nothing in the story changed: applying a proposal "
    "needs a person to approve the call, and no one did. Where the app does not ask during "
    "a conversation, as the desktop app does not, a person approves or rejects proposals "
    "in the story's view, under Files."
)


def _meta(work: StoryWork, room_id: str | None) -> dict[str, Any]:
    record = work.record()
    meta: dict[str, Any] = {"story_id": record.story_id, "title": record.title}
    for key in ("genre", "logline", "style_notes"):
        value = getattr(record, key)
        if value:
            meta[key] = value
    meta["writable_here"] = record.lease is not None and record.lease.holder == room_id
    return meta


def _writer(context: ToolContext) -> tuple[StoryWork, str]:
    """The open story and the conversation writing it; refuses before anything is read."""
    story_id, room_id = StoryLibrary.writer_of(context)
    return StoryWork(StoryLibrary(context.require_workspace()), story_id), room_id


# -- story_outline -----------------------------------------------------------------------


class StoryOutlineParams(BaseModel):
    """What to do with the story's outline."""

    model_config = _ARGUMENTS

    action: Literal["get", "init", "set_scene", "move", "structures"] = Field(
        description="'get' the outline; 'init' a new one from 'chapter_titles' or from a "
        "'structure'; 'structures' lists the structures 'init' can start from; 'set_scene' "
        "adds a scene (give 'chapter_id' and 'title') or changes one (give 'scene_id' and "
        "the fields to change); 'move' puts 'scene_id' in 'chapter_id', before 'before' or "
        "at the end."
    )
    chapter_titles: list[str] | None = Field(
        default=None, description="For 'init': the chapters' titles, in order."
    )
    structure: str | None = Field(
        default=None,
        description="For 'init', instead of 'chapter_titles': a structure's id from "
        "'structures'. The outline gets one chapter per act and one scene per beat, titled "
        "with the beat and summarised with what it is for, to be rewritten as the story's "
        "own.",
    )
    scene_id: str | None = Field(
        default=None,
        description="The scene to change or move. Leave it out to add a scene: an id like "
        "'ch02.s03' is given to it.",
    )
    chapter_id: str | None = Field(
        default=None, description="The chapter a new scene goes in, or 'move' moves it to."
    )
    chapter_title: str | None = Field(
        default=None,
        description="For 'set_scene': the title of a new chapter 'chapter_id', when it does "
        "not exist yet.",
    )
    title: str | None = Field(default=None, description="The scene's title.")
    summary: str | None = Field(default=None, description="What happens in the scene.")
    characters: list[str] | None = Field(
        default=None, description="Codex character ids of who is in the scene."
    )
    places: list[str] | None = Field(
        default=None, description="Codex place ids of where the scene happens."
    )
    beats: list[str] | None = Field(default=None, description="The scene's beats, in order.")
    story_time: str | int | None = Field(
        default=None, description="When the scene happens in the story."
    )
    before: str | None = Field(default=None, description="For 'move': the scene to put it before.")


def _scene_number(chapter_id: str, taken: set[str]) -> str:
    number = 1
    while f"{chapter_id}.s{number:02d}" in taken:
        number += 1
    return f"{chapter_id}.s{number:02d}"


class StoryOutlineTool(BaseTool[StoryOutlineParams]):
    """The story's chapters and scenes, in reading order."""

    name = "story_outline"
    description = (
        "Read or change the open story's outline: its chapters and scenes in reading order. "
        "Each scene has a fixed id, a title, a summary, beats and the codex ids of its "
        "characters and places. Start with 'init', from chapter titles or from a structure "
        "such as three acts ('structures' lists them); add and change scenes with "
        "'set_scene'."
    )
    params_type = StoryOutlineParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"get"})
    needs_room: ClassVar[bool] = True
    not_run_note: ClassVar[str] = "The outline was not changed."

    def __init__(self) -> None:
        super().__init__(
            name=self.name, description=self.description, params_type=StoryOutlineParams
        )

    async def run(self, params: StoryOutlineParams, context: ToolContext) -> dict[str, Any]:
        if params.action == "structures":
            return {
                "structures": [
                    {
                        "id": template_id,
                        "title": found.item.title,
                        "description": found.item.description,
                        "beats": sum(len(act.beats) for act in found.item.acts),
                        "source": found.source,
                    }
                    for template_id, found in sorted(_structures(context).items())
                ]
            }
        if params.action == "get":
            work = StoryWork.open_in(context)
            outline, _ = work.require_outline()
            written = set(work.written_scenes())
            return {
                "chapters": [
                    {
                        **chapter.model_dump(mode="json", exclude={"scenes"}, exclude_none=True),
                        "scenes": [
                            {
                                **scene.model_dump(mode="json", exclude_defaults=True),
                                "written": scene.id in written,
                            }
                            for scene in chapter.scenes
                        ],
                    }
                    for chapter in outline.chapters
                ]
            }
        work, room_id = _writer(context)
        current = work.outline()
        if params.action == "init":
            if current is not None:
                raise StoryError(
                    "The story already has an outline, so it was not replaced. Change it "
                    "with 'set_scene' and 'move'."
                )
            if params.structure is not None:
                if params.chapter_titles:
                    raise StoryError(
                        "Give either 'chapter_titles' or a 'structure', not both, so the "
                        "outline was not created."
                    )
                found = self._structure(params.structure, context)
                outline = self._validated(_outline_from(found.item))
                work.save_outline(outline, room_id=room_id, expected_digest=None)
                return {
                    **self._saved(work, outline, "created"),
                    "structure": found.item.id,
                    "source": found.source,
                }
            titles = [t.strip() for t in params.chapter_titles or [] if t.strip()]
            if not titles:
                raise StoryError(
                    "Give the chapters' titles in 'chapter_titles', or a 'structure' to start from."
                )
            data: dict[str, Any] = {
                "chapters": [
                    {"id": f"ch{i:02d}", "title": t, "scenes": []}
                    for i, t in enumerate(titles, start=1)
                ]
            }
            outline = self._validated(data)
            work.save_outline(outline, room_id=room_id, expected_digest=None)
            return self._saved(work, outline, "created")
        if current is None:
            raise StoryError(
                "The story has no outline yet. Start one with 'init', giving the chapters' titles."
            )
        outline, digest = current
        data = outline.model_dump(mode="json")
        if params.action == "set_scene":
            done = self._set_scene(data, params)
        else:
            done = self._move(data, params)
        changed = self._validated(data)
        work.save_outline(changed, room_id=room_id, expected_digest=digest)
        return self._saved(work, changed, done)

    @staticmethod
    def _structure(requested: str, context: ToolContext) -> Sourced[StructureTemplate]:
        templates = _structures(context)
        found = templates.get(_structure_id(requested))
        if found is None:
            raise StoryError(
                f"There is no structure '{requested}', so the outline was not created. "
                f"Structures: {', '.join(sorted(templates))}."
            )
        return found

    @staticmethod
    def _validated(data: dict[str, Any]) -> Outline:
        try:
            return Outline.model_validate(data)
        except ValidationError as exc:
            raise StoryError(describe_invalid("The changed outline", exc)) from exc

    @staticmethod
    def _saved(work: StoryWork, outline: Outline, done: str) -> dict[str, Any]:
        return {
            "outline": done,
            "chapters": len(outline.chapters),
            "scenes": len(outline.scenes_in_order()),
            "path": work.workspace_path(OUTLINE_FILE),
        }

    @staticmethod
    def _chapters(data: dict[str, Any]) -> list[dict[str, Any]]:
        return data["chapters"]

    def _find_scene(
        self, data: dict[str, Any], scene_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        for chapter in self._chapters(data):
            for scene in chapter["scenes"]:
                if scene["id"] == scene_id:
                    return chapter, scene
        return None

    def _set_scene(self, data: dict[str, Any], params: StoryOutlineParams) -> str:
        fields = {
            key: getattr(params, key)
            for key in ("title", "summary", "characters", "places", "beats", "story_time")
            if getattr(params, key) is not None
        }
        found = self._find_scene(data, params.scene_id) if params.scene_id else None
        if found is not None:
            chapter, scene = found
            if params.chapter_id is not None and params.chapter_id != chapter["id"]:
                raise StoryError(
                    f"Scene '{scene['id']}' is in chapter '{chapter['id']}'. Use 'move' to "
                    "put it in another chapter."
                )
            if not fields:
                raise StoryError(f"Say what to change in scene '{scene['id']}'.")
            scene.update(fields)
            return f"scene '{scene['id']}' changed"
        if params.chapter_id is None:
            if params.scene_id is not None:
                raise StoryError(
                    f"The outline has no scene '{params.scene_id}'. To add it, also give "
                    "the 'chapter_id' it goes in."
                )
            raise StoryError("A new scene needs the 'chapter_id' it goes in.")
        if "title" not in fields:
            raise StoryError("A new scene needs a 'title'.")
        chapters = self._chapters(data)
        chapter: dict[str, Any] | None = next(
            (c for c in chapters if c["id"] == params.chapter_id), None
        )
        if chapter is None:
            if not params.chapter_title:
                raise StoryError(
                    f"The outline has no chapter '{params.chapter_id}'. Give 'chapter_title' "
                    "to add it as a new chapter at the end."
                )
            chapter = {"id": params.chapter_id, "title": params.chapter_title, "scenes": []}
            chapters.append(chapter)
        taken = {s["id"] for c in chapters for s in c["scenes"]}
        scene_id = params.scene_id or _scene_number(chapter["id"], taken)
        chapter["scenes"].append({"id": scene_id, **fields})
        return f"scene '{scene_id}' added to chapter '{chapter['id']}'"

    def _move(self, data: dict[str, Any], params: StoryOutlineParams) -> str:
        if params.scene_id is None or params.chapter_id is None:
            raise StoryError(
                "'move' needs the 'scene_id' to move and the 'chapter_id' to move it to."
            )
        found = self._find_scene(data, params.scene_id)
        if found is None:
            raise StoryError(f"The outline has no scene '{params.scene_id}'.")
        target = next((c for c in self._chapters(data) if c["id"] == params.chapter_id), None)
        if target is None:
            raise StoryError(f"The outline has no chapter '{params.chapter_id}'.")
        source, scene = found
        if params.before is not None:
            if params.before == params.scene_id:
                raise StoryError("A scene cannot be moved before itself.")
            if not any(s["id"] == params.before for s in target["scenes"]):
                raise StoryError(
                    f"Chapter '{target['id']}' has no scene '{params.before}' to put it before."
                )
        source["scenes"].remove(scene)
        if params.before is None:
            target["scenes"].append(scene)
            where = "at the end"
        else:
            at = next(i for i, s in enumerate(target["scenes"]) if s["id"] == params.before)
            target["scenes"].insert(at, scene)
            where = f"before '{params.before}'"
        return f"scene '{scene['id']}' moved to chapter '{target['id']}', {where}"


def _structure_id(requested: str) -> str:
    """`"Save the Cat"` and `"save_the_cat"` name the template `save-the-cat`."""
    return re.sub(r"[\s_]+", "-", requested.strip().lower())


def _structures(context: ToolContext) -> dict[str, Sourced[StructureTemplate]]:
    """The bundled structure templates and those of the calling agent's active skills.

    Read on every call, like the muse tables: a skill approved since the last call counts,
    and a damaged template is refused by file and field rather than skipped (P6).
    """
    try:
        return load_structure_templates_sourced(data_roots(context.skill_dirs))
    except SkillDataError as exc:
        logger.warning("story_outline could not load its structures: %s", exc)
        raise StoryError(
            f"The story structures could not be loaded, so the outline was not changed: "
            f"{exc.plain}."
        ) from exc


def _outline_from(template: StructureTemplate) -> dict[str, Any]:
    """One chapter per act, one scene per beat: the beat's title, and its purpose as summary."""
    return {
        "chapters": [
            {
                "id": f"ch{number:02d}",
                "title": act.title,
                "act": act.id,
                "scenes": [
                    {
                        "id": f"ch{number:02d}.s{beat_number:02d}",
                        "title": beat.title,
                        "summary": beat.purpose,
                    }
                    for beat_number, beat in enumerate(act.beats, start=1)
                ],
            }
            for number, act in enumerate(template.acts, start=1)
        ]
    }


# -- story_codex -------------------------------------------------------------------------


class StoryCodexParams(BaseModel):
    """What to look up in the story's codex, or what change to propose, apply or reject."""

    model_config = _ARGUMENTS

    action: Literal["get", "search", "proposals", "propose", "apply", "reject"] = Field(
        description="'get' one entry by 'entry_id'; 'search' the entries by 'query', or list "
        "them all without one; 'proposals' lists the proposed changes; 'propose' a change to "
        "an entry that a scene shows ('entry_id', 'at', 'quote', and 'set' or tags); 'apply' "
        "or 'reject' a proposal by 'proposal_id'. Applying runs only once the person approves "
        "it when asked; saying it in chat is not approval."
    )
    entry_id: str | None = Field(
        default=None, description="For 'get' and 'propose': the entry's id."
    )
    kind: CodexKind | None = Field(
        default=None,
        description="Only entries of this kind: characters, places, items or threads.",
    )
    query: str | None = Field(
        default=None,
        description="For 'search': text to find in an entry's id, name, aliases or profile.",
    )
    at: str | None = Field(
        default=None,
        description="For 'propose': the scene after which the change is true.",
    )
    set: dict[str, JsonValue] | None = Field(
        default=None,
        description='For \'propose\': state values the change sets, e.g. {"status": "dead"}; '
        "null removes one.",
    )
    add_tags: list[str] | None = Field(
        default=None, description="For 'propose' on a character: visual tags it gains."
    )
    remove_tags: list[str] | None = Field(
        default=None, description="For 'propose' on a character: visual tags it loses."
    )
    note: str | None = Field(default=None, description="For 'propose': why, in a sentence.")
    quote: str | None = Field(
        default=None,
        description="For 'propose': the words of scene 'at' that show the change, copied "
        "exactly as whole words. A proposal without one is refused.",
    )
    proposal_id: str | None = Field(
        default=None, description="For 'apply' and 'reject': the proposal, e.g. 'p003'."
    )
    reason: str | None = Field(default=None, description="For 'reject': why, in a sentence.")


_DECIDE = (
    "A person decides: they approve or reject it in the story's view, under Files, or "
    "story_codex 'apply' asks them where the app can ask, and 'reject' drops it."
)


def _proposal_line(proposal: Proposal) -> dict[str, Any]:
    return proposal.model_dump(mode="json", exclude_defaults=True, exclude={"entry_digest"})


class StoryCodexTool(BaseTool[StoryCodexParams]):
    """The story's characters, places, items and threads, and the changes proposed to them."""

    name = "story_codex"
    description = (
        "Look up the open story's codex: its characters, places, items and threads, each "
        "kept in the story as its own file. 'get' reads one entry in full; 'search' finds "
        "entries by name, alias or profile. When a scene changes an entry (a death, a lost "
        "sword, a new scar), 'propose' the change with the quote that shows it; a person "
        "decides: 'apply' asks them for approval, and 'reject' drops it."
    )
    params_type = StoryCodexParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"get", "search", "proposals"})
    needs_room: ClassVar[bool] = True
    approval_actions: ClassVar[frozenset[str]] = frozenset({"apply"})
    approval_timeout_note: ClassVar[str | None] = APPLY_NOT_APPROVED_NOTE
    not_run_note: ClassVar[str] = "Nothing was looked up or changed."

    def __init__(self) -> None:
        super().__init__(name=self.name, description=self.description, params_type=StoryCodexParams)

    async def run(self, params: StoryCodexParams, context: ToolContext) -> dict[str, Any]:
        if params.action == "propose":
            return self._propose(params, context)
        if params.action == "apply":
            return self._apply(params, context)
        if params.action == "reject":
            return self._reject(params, context)
        if params.action == "proposals":
            return self._list_proposals(context)
        codex = StoryWork.open_in(context).codex()
        result: dict[str, Any]
        if params.action == "get":
            if params.entry_id is None or not params.entry_id.strip():
                raise StoryError("Say which entry to get, in 'entry_id'.")
            found = codex.find(params.entry_id.strip(), params.kind)
            if not found:
                kinds = params.kind or "codex"
                raise StoryError(
                    f"The story's {kinds} has no entry '{params.entry_id.strip()}'."
                    + self._unreadable_hint(codex)
                )
            result = {
                "entries": [
                    {"kind": item.kind, **item.entry.model_dump(mode="json", exclude_defaults=True)}
                    for item in found
                ]
            }
        else:
            needle = (params.query or "").strip().casefold()
            matches = [
                item
                for item in codex.items
                if (params.kind is None or item.kind == params.kind)
                and (
                    not needle
                    or any(
                        needle in text.casefold()
                        for text in (
                            item.entry.id,
                            item.entry.name,
                            item.entry.profile,
                            *item.entry.aliases,
                        )
                    )
                )
            ]
            result = {
                "entries": [
                    {
                        "id": item.entry.id,
                        "kind": item.kind,
                        "name": item.entry.name,
                        **({"aliases": list(item.entry.aliases)} if item.entry.aliases else {}),
                    }
                    for item in matches
                ],
                "kinds": list(CODEX_KINDS),
            }
        if codex.unreadable:
            result["unreadable_files"] = [
                {"file": u.file, "reason": u.reason} for u in codex.unreadable
            ]
        return result

    @staticmethod
    def _unreadable_hint(codex: CodexIndex) -> str:
        if not codex.unreadable:
            return ""
        return (
            f" {len(codex.unreadable)} codex file(s) or folder(s) could not be read, and it "
            "may be one of them: search the codex to see which and why."
        )

    # -- proposals ------------------------------------------------------------------

    def _list_proposals(self, context: ToolContext) -> dict[str, Any]:
        work = StoryWork.open_in(context)
        proposals, unreadable = work.proposals()
        result: dict[str, Any] = {
            "pending": [_proposal_line(p) for p, _ in proposals if p.status == "pending"],
            "decided": [
                {"id": p.id, "status": p.status, "entry_id": p.entry_id}
                for p, _ in proposals
                if p.status != "pending"
            ],
        }
        if unreadable:
            result["unreadable_files"] = [{"file": u.file, "reason": u.reason} for u in unreadable]
        return result

    def _propose(self, params: StoryCodexParams, context: ToolContext) -> dict[str, Any]:
        work, room_id = _writer(context)
        entry_id = (params.entry_id or "").strip()
        if not entry_id:
            raise StoryError("Say which entry the change is to, in 'entry_id'.")
        kind, entry, digest = _one_entry(work, entry_id, params.kind)
        scene_id = (params.at or "").strip()
        if not scene_id:
            raise StoryError(
                "Say after which scene the change is true, in 'at', so nothing was proposed."
            )
        outline, _ = work.require_outline()
        if outline.find(scene_id) is None:
            raise StoryError(f"The outline has no scene '{scene_id}', so nothing was proposed.")
        quote = (params.quote or "").strip()
        if not quote:
            raise StoryError(
                "A proposal needs a quote from the scene that shows the change, so nothing "
                "was proposed."
            )
        if quote_too_short(quote):
            raise StoryError(
                f"The quote is too short to show the change: it needs at least "
                f"{MIN_QUOTE_CHARACTERS} letters, so nothing was proposed."
            )
        text = work.manuscript(scene_id)
        if text is None:
            raise StoryError(f"Scene '{scene_id}' has no text yet, so nothing was proposed.")
        if not quote_found(quote, text.text):
            raise StoryError(
                f"The quote is not in scene '{scene_id}', so nothing was proposed. Copy "
                "whole words exactly as the scene has them."
            )
        change: dict[str, Any] = {}
        if params.set:
            change["progression"] = {"at": scene_id, "set": params.set, "note": params.note}
        if params.add_tags or params.remove_tags:
            change["visual_progression"] = {
                "at": scene_id,
                "add_tags": params.add_tags or [],
                "remove_tags": params.remove_tags or [],
                "note": params.note,
            }
        if not change:
            raise StoryError(
                "Say what changes, in 'set' or in 'add_tags' and 'remove_tags', so nothing "
                "was proposed."
            )
        draft = _draft(
            kind=kind,
            entry_id=entry.id,
            change=change,
            evidence=[{"scene_id": scene_id, "quote": quote}],
            room_id=room_id,
            context=context,
            entry_digest=digest,
        )
        check_applies(kind, entry, draft)
        saved = work.add_proposal(draft, room_id=room_id)
        return {
            "proposed": saved.id,
            "proposal": _proposal_line(saved),
            "path": work.workspace_path(proposal_file(saved.id)),
            "next": _DECIDE,
        }

    def _apply(self, params: StoryCodexParams, context: ToolContext) -> dict[str, Any]:
        proposal_id = self._proposal_id(params)
        if not context.approved_by_person:
            raise StoryError(
                f"Applying proposal '{proposal_id}' needs a person's approval, and this call "
                "was not approved, so the codex was not changed."
            )
        work, room_id = _writer(context)
        applied = apply_proposal(work, proposal_id, room_id=room_id, decided_in="conversation")
        result: dict[str, Any] = {
            "applied": proposal_id,
            "entry": {"kind": applied.kind, "id": applied.entry_id},
            "path": work.workspace_path(applied.entry_path),
        }
        if applied.notes:
            result["notes"] = applied.notes
        return result

    def _reject(self, params: StoryCodexParams, context: ToolContext) -> dict[str, Any]:
        proposal_id = self._proposal_id(params)
        work, room_id = _writer(context)
        relative = reject_proposal(
            work, proposal_id, room_id=room_id, reason=params.reason, decided_in="conversation"
        )
        return {"rejected": proposal_id, "path": work.workspace_path(relative)}

    @staticmethod
    def _proposal_id(params: StoryCodexParams) -> str:
        if params.proposal_id is None or not params.proposal_id.strip():
            raise StoryError("Say which proposal, in 'proposal_id'.")
        return params.proposal_id.strip()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _one_entry(
    work: StoryWork, entry_id: str, kind: CodexKind | None
) -> tuple[CodexKind, CodexEntry, str]:
    """The one entry `entry_id` (of `kind` when given), with its kind and digest."""
    if not _ENTRY_ID.fullmatch(entry_id):
        raise StoryError(
            f"'{entry_id}' is not an entry id: use lowercase letters and digits, joined by "
            "'.', '_' or '-'. Nothing was proposed."
        )
    found: list[tuple[CodexKind, CodexEntry, str]] = []
    for candidate in (kind,) if kind is not None else CODEX_KINDS:
        loaded = work.entry(candidate, entry_id)
        if loaded is not None:
            found.append((candidate, loaded[0], loaded[1]))
    if not found:
        where = f"{kind}" if kind is not None else "codex"
        raise StoryError(
            f"The story's {where} has no entry '{entry_id}', so nothing was proposed. Add the "
            "entry to the codex first."
        )
    if len(found) > 1:
        kinds = ", ".join(k for k, _, _ in found)
        raise StoryError(
            f"'{entry_id}' is an entry in more than one kind ({kinds}). Say which, in 'kind'."
        )
    return found[0]


def propose_visual(
    context: ToolContext, character_id: str, visual: dict[str, Any]
) -> dict[str, Any]:
    """Propose new values for a codex character's `visual` block; `character_sheet save`.

    Needs the story's lease, like every story write. A person applies the proposal with
    story_codex 'apply', as any other.
    """
    work, room_id = _writer(context)
    if not visual:
        raise StoryError(
            "Say what to change about how the character looks, so nothing was proposed."
        )
    kind, entry, digest = _one_entry(work, character_id, "characters")
    draft = _draft(
        kind=kind,
        entry_id=entry.id,
        change={"visual": visual},
        evidence=[],
        room_id=room_id,
        context=context,
        entry_digest=digest,
    )
    check_applies(kind, entry, draft)
    saved = work.add_proposal(draft, room_id=room_id)
    return {
        "proposed": saved.id,
        "proposal": _proposal_line(saved),
        "path": work.workspace_path(proposal_file(saved.id)),
        "next": _DECIDE,
    }


def _draft(
    *,
    kind: CodexKind,
    entry_id: str,
    change: dict[str, Any],
    evidence: list[dict[str, Any]],
    room_id: str,
    context: ToolContext,
    entry_digest: str,
) -> Proposal:
    """A proposal to save, numbered when it is saved; a change that does not fit is refused."""
    data: dict[str, Any] = {
        "id": "p000",
        "kind": kind,
        "entry_id": entry_id,
        "change": change,
        "evidence": evidence,
        "proposed_at": _now(),
        "room_id": room_id,
        "agent_id": context.agent_id,
        "entry_digest": entry_digest,
    }
    try:
        return Proposal.model_validate(data)
    except ValidationError as exc:
        raise StoryError(describe_invalid("The proposal", exc)) from exc


# -- story_audit -------------------------------------------------------------------------


class AuditFact(BaseModel):
    """One fact read from the scene."""

    model_config = _ARGUMENTS

    subject: str = Field(description="Who or what it is about, e.g. 'Vane'.")
    predicate: str = Field(
        description="What is said of it, e.g. 'status', 'possesses', 'located_in'."
    )
    object: str = Field(description="The value, e.g. 'alive', 'the moon sword'.")
    quote: str | None = Field(
        default=None,
        description="The words of the scene the fact is read from, copied exactly as whole "
        "words. A fact without one is not checked.",
    )


class StoryAuditParams(BaseModel):
    """Which scene to check, and the facts read from it."""

    model_config = _ARGUMENTS

    action: Literal["check"] = Field(
        description="'check' the facts read from 'scene_id' against the codex and the "
        "story's rules."
    )
    scene_id: str = Field(description="The scene the facts are read from.")
    facts: list[AuditFact] = Field(
        default_factory=list[AuditFact],
        description="The facts the scene states or shows, each with its quote.",
    )


class StoryAuditTool(BaseTool[StoryAuditParams]):
    """Checks a scene's facts against what the story says was true when it happens."""

    name = "story_audit"
    description = (
        "Check a written scene for continuity errors. Read the scene, list the facts it "
        "states or shows (who is alive or dead, who has what, where someone is), each with "
        "the exact words it comes from, and 'check' them: they are compared with the codex "
        "as it stands at that point of the story, under the story's rules. A contradiction "
        "names every fact involved and where it came from. Nothing is changed."
    )
    params_type = StoryAuditParams
    writes_files: ClassVar[bool] = False
    needs_room: ClassVar[bool] = True
    not_run_note: ClassVar[str] = "Nothing was checked."

    def __init__(self) -> None:
        super().__init__(name=self.name, description=self.description, params_type=StoryAuditParams)

    async def run(self, params: StoryAuditParams, context: ToolContext) -> dict[str, Any]:
        from uclone_x.story.audit import SubmittedFact, audit_scene  # the reasoner, when asked

        work = StoryWork.open_in(context)
        scene_id = params.scene_id.strip()
        outline, _ = work.require_outline()
        if outline.find(scene_id) is None:
            raise StoryError(f"The outline has no scene '{scene_id}', so nothing was checked.")
        text = work.manuscript(scene_id)
        if text is None:
            raise StoryError(f"Scene '{scene_id}' has no text yet, so there is nothing to check.")
        if not params.facts:
            raise StoryError(
                "Give the facts the scene states, each with its quote, in 'facts'. Nothing "
                "was checked."
            )
        axioms, defaults = work.axioms()
        return audit_scene(
            story_id=work.story_id,
            outline=outline,
            scene_id=scene_id,
            scene_text=text.text,
            codex=work.codex(),
            facts=[SubmittedFact(f.subject, f.predicate, f.object, f.quote) for f in params.facts],
            axioms=axioms,
            axioms_are_defaults=defaults,
        )


# -- story_manuscript --------------------------------------------------------------------


class StoryManuscriptParams(BaseModel):
    """What to do with a scene's text."""

    model_config = _ARGUMENTS

    action: Literal["write", "read", "list"] = Field(
        description="'write' a scene's text; 'read' it back with its digest; 'list' the "
        "scenes in reading order and which have text."
    )
    scene_id: str | None = Field(
        default=None, description="The outline scene, for 'write' and 'read'."
    )
    text: str | None = Field(default=None, description="For 'write': the scene's full text.")
    digest: str | None = Field(
        default=None,
        description="For 'write' over a scene that has text: the digest 'read' gave, so an "
        "edit made since is not lost. Leave it out for a scene with no text yet.",
    )
    session_summary: str | None = Field(
        default=None,
        description="For 'write': one paragraph on where the story stands now, for the next "
        "conversation's recap.",
    )


class StoryManuscriptTool(BaseTool[StoryManuscriptParams]):
    """The text of the story's scenes, one file per scene, with the text each write replaced."""

    name = "story_manuscript"
    description = (
        "Write, read or list the open story's scenes. Each outline scene's text is its own "
        "file; writing over a scene keeps the text it replaces in the scene's history. To "
        "rewrite a scene, 'read' it first and pass its digest to 'write'."
    )
    params_type = StoryManuscriptParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"read", "list"})
    needs_room: ClassVar[bool] = True
    not_run_note: ClassVar[str] = "No scene was written."

    def __init__(self) -> None:
        super().__init__(
            name=self.name, description=self.description, params_type=StoryManuscriptParams
        )

    async def run(self, params: StoryManuscriptParams, context: ToolContext) -> dict[str, Any]:
        if params.action == "write":
            return self._write(params, context)
        work = StoryWork.open_in(context)
        if params.action == "read":
            scene_id = self._scene_id(params)
            found = work.manuscript(scene_id)
            if found is None:
                raise StoryError(f"Scene '{scene_id}' has no text yet.")
            return {"scene_id": scene_id, "text": found.text, "digest": found.digest}
        written = work.written_scenes()
        current = work.outline()
        order = [s.id for _, s in current[0].scenes_in_order()] if current else []
        result: dict[str, Any] = {
            "scenes": [{"scene_id": s, "written": s in set(written)} for s in order]
        }
        stray = [s for s in written if s not in set(order)]
        if stray:
            result["written_but_not_in_outline"] = stray
        return result

    @staticmethod
    def _scene_id(params: StoryManuscriptParams) -> str:
        if params.scene_id is None or not params.scene_id.strip():
            raise StoryError("Say which scene, in 'scene_id'.")
        return params.scene_id.strip()

    def _write(self, params: StoryManuscriptParams, context: ToolContext) -> dict[str, Any]:
        work, room_id = _writer(context)
        scene_id = self._scene_id(params)
        if params.text is None or not params.text.strip():
            raise StoryError("Give the scene's text, so nothing was written.")
        outline, _ = work.require_outline()
        if outline.find(scene_id) is None:
            raise StoryError(
                f"The outline has no scene '{scene_id}', so nothing was written. Add it with "
                "story_outline 'set_scene' first."
            )
        digest, notes = work.write_scene(
            scene_id,
            params.text,
            room_id=room_id,
            expected_digest=params.digest,
            agent_id=context.agent_id,
            turn_index=context.turn_index,
        )
        try:
            work.note_session(room_id, scene_written=scene_id, summary=params.session_summary)
        except (StoryError, StoryFileError) as exc:
            notes.append(f"The scene was written, but this conversation's record was not: {exc}")
        result: dict[str, Any] = {
            "scene_id": scene_id,
            "digest": digest,
            "path": work.workspace_path(manuscript_file(scene_id)),
        }
        if notes:
            result["notes"] = notes
        return result


# -- story_context -----------------------------------------------------------------------


class StoryContextParams(BaseModel):
    """Which context to gather."""

    model_config = _ARGUMENTS

    action: Literal["for_scene", "recap"] = Field(
        description="'for_scene' gathers what to know before writing 'scene_id'; 'recap' "
        "says where the story stands and what to write next, for the start of a "
        "conversation."
    )
    scene_id: str | None = Field(default=None, description="For 'for_scene': the scene.")


class StoryContextTool(BaseTool[StoryContextParams]):
    """What to read before writing: one scene's context, or the whole story's recap."""

    name = "story_context"
    description = (
        "Gather the open story's context. 'recap' at the start of a conversation: what the "
        "story is, what earlier conversations wrote, and the next scene's context. "
        "'for_scene' before writing a scene: the scene, its neighbours, the end of the scene "
        "before, and the codex entries it needs, with a manifest of what was included, what "
        "was left out, and why."
    )
    params_type = StoryContextParams
    writes_files: ClassVar[bool] = False
    needs_room: ClassVar[bool] = True
    not_run_note: ClassVar[str] = "Nothing was gathered."

    def __init__(self) -> None:
        super().__init__(
            name=self.name, description=self.description, params_type=StoryContextParams
        )

    async def run(self, params: StoryContextParams, context: ToolContext) -> dict[str, Any]:
        work = StoryWork.open_in(context)
        meta = _meta(work, context.room_id)
        if params.action == "for_scene":
            if params.scene_id is None or not params.scene_id.strip():
                raise StoryError("Say which scene, in 'scene_id'.")
            scene_id = params.scene_id.strip()
            outline, _ = work.require_outline()
            if outline.find(scene_id) is None:
                raise StoryError(f"The outline has no scene '{scene_id}'.")
            return self._bundle(work, outline, scene_id, work.codex(), meta)
        current = work.outline()
        outline = current[0] if current else None
        written = work.written_scenes()
        problems: list[UnreadableFile] = []
        try:
            sessions, _ = work.sessions()
        except (StoryError, StoryFileError) as exc:
            # The recap still says the rest; the record that did not load is named in it.
            sessions = SessionsFile()
            problems.append(UnreadableFile(SESSIONS_FILE, str(exc)))
        last_text: str | None = None
        next_context: dict[str, Any] | None = None
        if outline is not None:
            last, following = last_and_next(outline, written)
            if last is not None:
                found = work.manuscript(last[1].id)
                last_text = found.text if found else None
            if following is not None:
                next_context = self._bundle(work, outline, following[1].id, work.codex(), meta)
                next_context.pop("story", None)  # the recap says it once, at the top
        return recap(
            story=meta,
            outline=outline,
            written=written,
            sessions=sessions,
            last_text=last_text,
            next_context=next_context,
            problems=problems,
        )

    @staticmethod
    def _bundle(
        work: StoryWork,
        outline: Outline,
        scene_id: str,
        codex: CodexIndex,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        ordered = [s.id for _, s in outline.scenes_in_order()]
        index = ordered.index(scene_id)
        previous = work.manuscript(ordered[index - 1]) if index > 0 else None
        return scene_context(
            outline,
            scene_id,
            codex,
            previous_text=previous.text if previous else None,
            story=meta,
        )
