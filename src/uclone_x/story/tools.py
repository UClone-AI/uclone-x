"""The Writer's story tools: outline, codex, manuscript and context (#1556).

Each works on the story the call's conversation has open (`ToolContext.story_id`) and
takes no path from the model: a file is named by what it is -- a scene id, a codex entry
id -- and the tool computes where it lives. Reads need only an open story; writes need the
conversation to hold the story's lease, and go through `StoryLibrary.write_file`, so a
file changed by hand after it was read is not overwritten.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uclone_x.story.context import (
    CodexIndex,
    UnreadableFile,
    last_and_next,
    recap,
    scene_context,
)
from uclone_x.story.library import StoryError, StoryLibrary
from uclone_x.story.schemas import (
    CODEX_KINDS,
    CodexKind,
    Outline,
    SessionsFile,
    StoryFileError,
    describe_invalid,
)
from uclone_x.story.work import OUTLINE_FILE, SESSIONS_FILE, StoryWork, manuscript_file
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

__all__ = [
    "StoryCodexTool",
    "StoryContextTool",
    "StoryManuscriptTool",
    "StoryOutlineTool",
]

#: A model's JSON arguments: unknown keys are refused by name; not strict, since a JSON
#: decoder gives lists where a strict tuple would be refused (#1375).
_ARGUMENTS = ConfigDict(frozen=True, extra="forbid")


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

    action: Literal["get", "init", "set_scene", "move"] = Field(
        description="'get' the outline; 'init' a new one from 'chapter_titles'; 'set_scene' "
        "adds a scene (give 'chapter_id' and 'title') or changes one (give 'scene_id' and "
        "the fields to change); 'move' puts 'scene_id' in 'chapter_id', before 'before' or "
        "at the end."
    )
    chapter_titles: list[str] | None = Field(
        default=None, description="For 'init': the chapters' titles, in order."
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
        "characters and places. Start with 'init'; add and change scenes with 'set_scene'."
    )
    params_type = StoryOutlineParams
    writes_files: ClassVar[bool] = True
    needs_room: ClassVar[bool] = True
    not_run_note: ClassVar[str] = "The outline was not changed."

    def __init__(self) -> None:
        super().__init__(
            name=self.name, description=self.description, params_type=StoryOutlineParams
        )

    async def run(self, params: StoryOutlineParams, context: ToolContext) -> dict[str, Any]:
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
            titles = [t.strip() for t in params.chapter_titles or [] if t.strip()]
            if not titles:
                raise StoryError("Give the chapters' titles in 'chapter_titles'.")
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


# -- story_codex -------------------------------------------------------------------------


class StoryCodexParams(BaseModel):
    """What to look up in the story's codex."""

    model_config = _ARGUMENTS

    action: Literal["get", "search"] = Field(
        description="'get' one entry by 'entry_id'; 'search' the entries by 'query', or list "
        "them all without one."
    )
    entry_id: str | None = Field(default=None, description="For 'get': the entry's id.")
    kind: CodexKind | None = Field(
        default=None,
        description="Only entries of this kind: characters, places, items or threads.",
    )
    query: str | None = Field(
        default=None,
        description="For 'search': text to find in an entry's id, name, aliases or profile.",
    )


class StoryCodexTool(BaseTool[StoryCodexParams]):
    """The story's characters, places, items and threads."""

    name = "story_codex"
    description = (
        "Look up the open story's codex: its characters, places, items and threads, each "
        "kept in the story as its own file. 'get' reads one entry in full; 'search' finds "
        "entries by name, alias or profile."
    )
    params_type = StoryCodexParams
    writes_files: ClassVar[bool] = False
    needs_room: ClassVar[bool] = True
    not_run_note: ClassVar[str] = "Nothing was looked up."

    def __init__(self) -> None:
        super().__init__(name=self.name, description=self.description, params_type=StoryCodexParams)

    async def run(self, params: StoryCodexParams, context: ToolContext) -> dict[str, Any]:
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
            f" {len(codex.unreadable)} codex file(s) could not be read, and it may be one "
            "of them: search the codex to see which and why."
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
