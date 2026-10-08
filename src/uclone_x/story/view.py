"""A story's view: its outline, its codex, and the changes waiting for a person (#1560).

The Files screen lists a story as a folder. This service reads one story whole, for a
person: the outline in reading order and in the order its scenes happen, the codex
grouped by kind, and every proposed codex change with what approving it would do.

**The outline is a story board** (§1.1 row 5 of the novel-writer design). Each chapter
carries its function and the twist and climax marks from `start.yaml`'s arc, but only when
the arc has as many chapters as the outline, since otherwise the two do not line up. Each
scene carries the codex entries in it (listed by the outline, planted or paid off there as a
thread, or named in its title, summary, beats or manuscript text by name or alias, a plain
case-folded substring match), the changes the codex records at it, what the check after
writing found there (`continuity`, `continuity_checked`, `continuity_unread`), and the
pending proposals that rest on it (a change placed at it, or evidence quoting it). A cast
member the outline lists but the codex does not hold yet is shown by its proposed name and
marked not in the codex. A `start.yaml` that does not read gives `board_note`; the board then
shows no functions and no checks, rather than guessing.

**Where a person decides a proposal.** The model proposes a change (`story_codex
propose`); a person decides it. In a conversation the runtime asks them when the model
calls `story_codex apply`; the desktop app does not ask there, so this view is where they
decide. `approve` and `reject` are called by the view's buttons, over the local routes in
`uclone_x.ui.artifacts`. No story or file tool reaches them. A persona with an
unconfined shell (Clone's `bash_run`) can still call those routes over the local API, so
until #1589 is closed `decided_in: story_view` is not proof that a person decided.
Both go through `uclone_x.story.proposals`, the code `story_codex` uses, so the checks are
the same wherever the decision is made:

* the proposal must still be undecided, and its file must be the one the person was shown
  (`seen_digest`);
* the entry must be the version the proposal was made against (`entry_digest`);
* the write is made as the conversation holding the story's writing lease, through
  `StoryLibrary.write_file`, so the lease is honoured. A story nobody is writing, or whose
  writer no longer exists, is refused with the remedy (open it in a conversation); so is
  one whose writer is answering right now, since the decision would change the story under
  that turn.

This module is the view's Core; `uclone_x.ui.artifacts` only maps its refusals to status
codes. Each note it shows a person (`decide_code`, `outline_code`, `board_code`) and each
refusal (`code` on the error) carries a code with its English sentence; the head words the
code from its `dock.story` catalog (`StoryNoteCode`, `StoryRefusalCode`) and falls back to
the sentence.

**What cannot be read is said, never skipped** (P6): an outline, codex entry or proposal
file that does not load is listed with its reason, and a proposal that cannot be approved
now says why before anyone presses a button.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from uclone_x.errors import RoomNotFoundError, UnreadableRoomRecordError
from uclone_x.room.service import RoomService
from uclone_x.story.context import CodexIndex
from uclone_x.story.library import (
    StoryError,
    StoryLibrary,
    StoryRecord,
    UnknownStoryError,
)
from uclone_x.story.proposals import (
    ChangeLine,
    apply_proposal,
    preview_proposal,
    reject_proposal,
)
from uclone_x.story.quotes import passage_in
from uclone_x.story.schemas import (
    CODEX_KINDS,
    CharacterEntry,
    CodexEntry,
    CodexKind,
    Outline,
    Proposal,
    Scene,
    StoryFileError,
    ThreadEntry,
)
from uclone_x.story.start import START_FILE, StartState
from uclone_x.story.timeline import Placement, assumptions, place_scenes
from uclone_x.story.work import StoryWork

__all__ = [
    "ConversationRef",
    "ProposalDecided",
    "StoryNotFoundError",
    "StoryNotWritableError",
    "StoryOverview",
    "StoryView",
    "WriterBusyError",
]

_MODEL = ConfigDict(frozen=True, extra="forbid")

_KIND_LABELS: dict[CodexKind, str] = {
    "characters": "Characters",
    "places": "Places",
    "items": "Items",
    "threads": "Threads",
}


#: Why the view shows a note in place of part of the story, as a code the head words in
#: the reader's language (`dock.story.notes`); the English sentence beside it is the
#: fallback for a head that does not know the code.
StoryNoteCode = Literal[
    "no_writer", "writer_gone", "start_unreadable", "no_outline", "outline_unreadable"
]
#: Why a request on the view was refused, as a code the head words (`dock.story.refusals`).
StoryRefusalCode = Literal["story_gone", "no_writer", "writer_gone", "writer_busy"]


class StoryNotFoundError(StoryError):
    """The story is not in the library (any more)."""

    code: StoryRefusalCode = "story_gone"


class StoryNotWritableError(StoryError):
    """No existing conversation holds the story's writing lease, so nothing can be saved."""

    def __init__(self, message: str, code: StoryRefusalCode) -> None:
        super().__init__(message)
        self.code: StoryRefusalCode = code


class WriterBusyError(StoryError):
    """The conversation writing the story is answering now; deciding would change it under
    that turn."""

    code: StoryRefusalCode = "writer_busy"


# -- values ---------------------------------------------------------------------------


class ConversationRef(BaseModel):
    """A conversation by id, with its title while it exists."""

    model_config = _MODEL

    room_id: str
    title: str | None
    exists: bool


#: How a codex entry is in a scene on the board: the outline lists it for the scene, the
#: scene's plan or text names it, or the scene plants or pays off a thread.
SceneLink = Literal["listed", "mentioned", "planted", "paid_off"]


class SceneEntry(BaseModel):
    """A codex entry the scene takes part in, for the board's link to the codex."""

    model_config = _MODEL

    kind: CodexKind
    id: str
    name: str
    how: SceneLink
    #: False for an entry the outline lists that is not in the codex -- only proposed, or
    #: missing. The board shows its name and does not link it.
    in_codex: bool


class SceneChange(BaseModel):
    """A change the codex records at a scene: true once the scene has ended (§5.1)."""

    model_config = _MODEL

    kind: CodexKind
    entry_id: str
    entry_name: str
    #: The state keys the change sets (`null` clears one); empty for a change in looks.
    set: dict[str, JsonValue]
    #: Looks added and removed, for a character's visual change.
    add_looks: list[str]
    remove_looks: list[str]
    note: str | None


class SceneFinding(BaseModel):
    """A contradiction `story_start`'s check after writing found in the scene (§4.3 step 5)."""

    model_config = _MODEL

    quote: str
    #: The sentence the person is told; the reasoner's kind of contradiction is not here.
    note: str


class SceneView(BaseModel):
    model_config = _MODEL

    id: str
    title: str
    summary: str
    story_time: str | int | None
    characters: list[str]
    places: list[str]
    #: Whether the scene has text in the manuscript.
    written: bool
    #: The codex entries in the scene, listed ones first.
    entries: list[SceneEntry]
    #: What the codex records changing at this scene.
    changes: list[SceneChange]
    #: Whether `story_start`'s check after writing read the scene (`checked`), could not
    #: read it (`unread`), or never ran on it (`None`). `checked` with no findings means the
    #: facts it read agreed with the codex, not that the scene was proven consistent.
    continuity: Literal["checked", "unread"] | None
    findings: list[SceneFinding]
    #: The ids of the pending proposals that rest on this scene: a quote from it, or a
    #: change placed at it.
    pending: list[str]


class ChapterView(BaseModel):
    model_config = _MODEL

    id: str
    title: str
    act: str | None
    scenes: list[SceneView]
    #: What `story_start`'s arc gave the chapter to do (`uclone_x.story.arc.FUNCTIONS`);
    #: empty when the story has no arc, or the outline no longer has the arc's chapters.
    functions: list[str]
    #: The chapter the arc reveals the twist in, and the one it decides the conflict in.
    twist: bool
    climax: bool


class OutlineView(BaseModel):
    model_config = _MODEL

    chapters: list[ChapterView]
    #: Every scene id, in the order the scenes happen in the story.
    story_order: list[str]
    #: Scenes placed in story time without a time of their own, and where, as sentences.
    assumptions: list[str]


class Unreadable(BaseModel):
    model_config = _MODEL

    file: str
    reason: str


class EntryView(BaseModel):
    model_config = _MODEL

    id: str
    name: str
    aliases: list[str]
    profile: str
    state: dict[str, JsonValue]
    #: The changes the entry records, each true once scene `at` has ended.
    progressions: list[dict[str, JsonValue]]
    #: How a character looks at the start; `None` for an entry with no looks.
    looks: list[str] | None


class CodexGroup(BaseModel):
    model_config = _MODEL

    kind: CodexKind
    label: str
    entries: list[EntryView]


class EvidenceView(BaseModel):
    model_config = _MODEL

    scene_id: str
    scene_title: str | None
    quote: str
    #: Whether the quoted words are still in the scene's text; `None` when the scene has no
    #: text. Word edges and length are not checked again (`quotes.passage_in`).
    still_in_scene: bool | None


class ChangeView(BaseModel):
    model_config = _MODEL

    what: str
    at: str | None
    before: JsonValue
    after: JsonValue
    placed: bool


class ProposalView(BaseModel):
    model_config = _MODEL

    id: str
    status: str
    kind: CodexKind
    entry_id: str
    entry_name: str | None
    proposed_at: str
    #: The conversation that proposed it, while it exists; `exists` false once deleted.
    proposed_in: ConversationRef
    agent_id: str | None
    note: str | None
    evidence: list[EvidenceView]
    changes: list[ChangeView]
    #: A line diff of the entry file, now and as approving would write it.
    diff: list[str]
    #: Why approving it now would be refused, or `None`.
    blocked: str | None
    #: The proposal file's digest, sent back with a decision so it is about what was shown.
    digest: str
    decided_at: str | None
    decided_in: str | None
    reason: str | None


class StoryOverview(BaseModel):
    model_config = _MODEL

    story_id: str
    title: str
    logline: str | None
    genre: str | None
    writer: ConversationRef | None
    #: Why a decision cannot be saved now, with the remedy; `None` when it can.
    decide_note: str | None
    decide_code: StoryNoteCode | None = None
    outline: OutlineView | None
    #: Why there is no outline to show: none yet, or one that does not read.
    outline_note: str | None
    outline_code: StoryNoteCode | None = None
    #: Why the board shows no chapter functions or continuity notes although the story was
    #: started with `story_start`: its `start.yaml` does not read.
    board_note: str | None
    board_code: StoryNoteCode | None = None
    codex: list[CodexGroup]
    codex_unreadable: list[Unreadable]
    pending: list[ProposalView]
    decided: list[ProposalView]
    proposals_unreadable: list[Unreadable]


class ProposalDecided(BaseModel):
    model_config = _MODEL

    proposal_id: str
    decision: str
    #: What the person should know about how it went, e.g. comments not kept.
    notes: list[str]


# -- the service ----------------------------------------------------------------------


class StoryView:
    """One workspace's stories, read whole, with their proposals decided by a person."""

    def __init__(
        self,
        workspace: Path,
        rooms: RoomService,
        *,
        turn_busy: Callable[[str], bool],
    ) -> None:
        """`turn_busy(room_id)` says whether that conversation is answering right now.

        Required, not defaulted, for the reason `ArtifactLibrary` gives: deciding while the
        writer is mid-turn changes the story under that turn.
        """
        self._library = StoryLibrary(workspace.resolve())
        self._rooms = rooms
        self._turn_busy = turn_busy

    # -- reading ------------------------------------------------------------------------

    def _record(self, story_id: str) -> StoryRecord:
        try:
            return self._library.load(story_id)
        except UnknownStoryError as exc:
            raise StoryNotFoundError(
                f"There is no story called '{story_id}' any more. Refresh the list to see the "
                "stories there are."
            ) from exc

    def _conversation(self, room_id: str) -> ConversationRef:
        try:
            title = self._rooms.get(room_id).title
        except RoomNotFoundError:
            return ConversationRef(room_id=room_id, title=None, exists=False)
        except UnreadableRoomRecordError:
            return ConversationRef(room_id=room_id, title=None, exists=True)
        return ConversationRef(room_id=room_id, title=title, exists=True)

    def show(self, story_id: str) -> StoryOverview:
        """The story whole: outline, codex, and proposals with what each would change."""
        record = self._record(story_id)
        work = StoryWork(self._library, story_id)
        writer = self._conversation(record.lease.holder) if record.lease is not None else None

        index = work.codex()
        loaded, unreadable = work.proposals()
        start, board_note = _start_state(work)
        decide = _decide_note(record, writer)
        outline_view, outline_note, placements, scene_titles = self._outline(
            work, index, [p for p, _ in loaded if p.status == "pending"], start
        )
        groups: list[CodexGroup] = []
        for kind in CODEX_KINDS:
            entries = [_entry_view(item.entry) for item in index.items if item.kind == kind]
            groups.append(CodexGroup(kind=kind, label=_KIND_LABELS[kind], entries=entries))

        pending: list[ProposalView] = []
        decided: list[ProposalView] = []
        for proposal, digest in loaded:
            view = self._proposal_view(work, proposal, digest, placements, scene_titles)
            (pending if proposal.status == "pending" else decided).append(view)
        decided.reverse()  # newest first

        return StoryOverview(
            story_id=story_id,
            title=record.title,
            logline=record.logline,
            genre=record.genre,
            writer=writer,
            decide_note=decide[1] if decide else None,
            decide_code=decide[0] if decide else None,
            outline=outline_view,
            outline_note=outline_note[1] if outline_note else None,
            outline_code=outline_note[0] if outline_note else None,
            board_note=board_note,
            board_code="start_unreadable" if board_note else None,
            codex=groups,
            codex_unreadable=[Unreadable(file=u.file, reason=u.reason) for u in index.unreadable],
            pending=pending,
            decided=decided,
            proposals_unreadable=[Unreadable(file=u.file, reason=u.reason) for u in unreadable],
        )

    @staticmethod
    def _outline(
        work: StoryWork,
        index: CodexIndex,
        pending: Sequence[Proposal],
        start: StartState | None,
    ) -> tuple[
        OutlineView | None,
        tuple[StoryNoteCode, str] | None,
        Mapping[str, Placement] | None,
        dict[str, str],
    ]:
        try:
            found = work.outline()
        except StoryFileError as exc:
            return None, ("outline_unreadable", f"The outline could not be read: {exc}"), None, {}
        if found is None:
            return None, ("no_outline", "This story has no outline yet."), None, {}
        outline: Outline = found[0]
        placements = place_scenes(outline)
        written = set(work.written_scenes())
        proposed = _proposed_names(pending)
        arc = (start.arc or {}) if start is not None else {}
        functions = _chapter_functions(arc, len(outline.chapters))
        chapters: list[ChapterView] = []
        for number, chapter in enumerate(outline.chapters, start=1):
            scenes: list[SceneView] = []
            for scene in chapter.scenes:
                text = work.manuscript(scene.id) if scene.id in written else None
                haystack = " ".join(
                    [scene.title, scene.summary, *scene.beats, text.text if text else ""]
                ).casefold()
                scenes.append(
                    SceneView(
                        id=scene.id,
                        title=scene.title,
                        summary=scene.summary,
                        story_time=scene.story_time,
                        characters=list(scene.characters),
                        places=list(scene.places),
                        written=scene.id in written,
                        entries=_scene_entries(scene, index, proposed, haystack),
                        changes=_scene_changes(scene.id, index),
                        continuity=_checked(scene.id, start),
                        findings=[
                            SceneFinding(quote=f.get("quote", ""), note=f.get("note", ""))
                            for f in (start.continuity if start is not None else [])
                            if f.get("scene_id") == scene.id and f.get("note")
                        ],
                        pending=[p.id for p in pending if _rests_on(p, scene.id)],
                    )
                )
            chapters.append(
                ChapterView(
                    id=chapter.id,
                    title=chapter.title,
                    act=chapter.act,
                    scenes=scenes,
                    functions=functions[number - 1] if functions else [],
                    twist=bool(functions) and arc.get("twist_chapter") == number,
                    climax=bool(functions) and arc.get("climax_chapter") == number,
                )
            )
        order = [p.scene_id for p in sorted(placements.values(), key=lambda p: p.position)]
        placed = [f"{a['scene_id']} is placed {a['placed']}." for a in assumptions(placements)]
        titles = {scene.id: scene.title for _, scene in outline.scenes_in_order()}
        view = OutlineView(chapters=chapters, story_order=order, assumptions=placed)
        return view, None, placements, titles

    def _proposal_view(
        self,
        work: StoryWork,
        proposal: Proposal,
        digest: str,
        placements: Mapping[str, Placement] | None,
        scene_titles: dict[str, str],
    ) -> ProposalView:
        evidence: list[EvidenceView] = []
        for item in proposal.evidence:
            text = work.manuscript(item.scene_id)
            evidence.append(
                EvidenceView(
                    scene_id=item.scene_id,
                    scene_title=scene_titles.get(item.scene_id),
                    quote=item.quote,
                    still_in_scene=passage_in(item.quote, text.text) if text else None,
                )
            )
        entry_name: str | None = None
        changes: list[ChangeLine] = []
        diff: list[str] = []
        blocked: str | None = None
        if proposal.status == "pending":
            preview = preview_proposal(work, proposal, placements)
            entry_name, changes, diff, blocked = (
                preview.entry_name,
                preview.changes,
                preview.diff,
                preview.blocked,
            )
        else:
            try:
                current = work.entry(proposal.kind, proposal.entry_id)
            except (StoryError, StoryFileError):
                current = None
            entry_name = current[0].name if current is not None else _new_name(proposal)
        return ProposalView(
            id=proposal.id,
            status=proposal.status,
            kind=proposal.kind,
            entry_id=proposal.entry_id,
            entry_name=entry_name,
            proposed_at=proposal.proposed_at,
            proposed_in=self._conversation(proposal.room_id),
            agent_id=proposal.agent_id,
            note=_note(proposal),
            evidence=evidence,
            changes=[
                ChangeView(what=c.what, at=c.at, before=c.before, after=c.after, placed=c.placed)
                for c in changes
            ],
            diff=diff,
            blocked=blocked,
            digest=digest,
            decided_at=proposal.decided_at,
            decided_in=proposal.decided_in,
            reason=proposal.reason,
        )

    # -- deciding -----------------------------------------------------------------------

    def _decider(self, story_id: str) -> tuple[StoryWork, str]:
        """The story and the conversation a decision is written as; refused with a remedy."""
        record = self._record(story_id)
        writer = self._conversation(record.lease.holder) if record.lease is not None else None
        note = _decide_note(record, writer)
        if note is not None or writer is None:
            code, sentence = note or ("no_writer", _NO_WRITER.format(title=record.title))
            raise StoryNotWritableError(sentence, code)
        if self._turn_busy(writer.room_id):
            name = f"“{writer.title}”" if writer.title else "writing this story"
            raise WriterBusyError(
                f"The conversation {name} is answering right now, so the decision was not "
                "saved. Wait for the answer to finish, then try again."
            )
        return StoryWork(self._library, story_id), writer.room_id

    def approve(self, story_id: str, proposal_id: str, *, seen_digest: str) -> ProposalDecided:
        """Apply `proposal_id` as the person approved it, as it was shown to them."""
        work, room_id = self._decider(story_id)
        applied = apply_proposal(
            work,
            proposal_id,
            room_id=room_id,
            decided_in="story_view",
            expected_proposal_digest=seen_digest,
        )
        return ProposalDecided(proposal_id=proposal_id, decision="applied", notes=applied.notes)

    def reject(
        self, story_id: str, proposal_id: str, *, seen_digest: str, reason: str | None
    ) -> ProposalDecided:
        """Mark `proposal_id` rejected, as it was shown to the person."""
        work, room_id = self._decider(story_id)
        reject_proposal(
            work,
            proposal_id,
            room_id=room_id,
            reason=reason.strip() if reason and reason.strip() else None,
            decided_in="story_view",
            expected_proposal_digest=seen_digest,
        )
        return ProposalDecided(proposal_id=proposal_id, decision="rejected", notes=[])


_NO_WRITER = (
    "No conversation is writing “{title}”, so a decision cannot be saved. Open the story in "
    "a conversation, then decide here."
)


def _decide_note(
    record: StoryRecord, writer: ConversationRef | None
) -> tuple[Literal["no_writer", "writer_gone"], str] | None:
    """Why a decision cannot be saved now, as a code and its English sentence."""
    if writer is None:
        return "no_writer", _NO_WRITER.format(title=record.title)
    if not writer.exists:
        return "writer_gone", (
            f"The conversation that was writing “{record.title}” no longer exists, so a "
            "decision cannot be saved. Open the story in a conversation, then decide here."
        )
    return None


def _new_name(proposal: Proposal) -> str | None:
    """The name a rejected new entry would have had, so the list does not show its id (#1808)."""
    new = proposal.change.new_entry
    name = new.get("name") if new is not None else None
    return name if isinstance(name, str) else None


def _note(proposal: Proposal) -> str | None:
    change = proposal.change
    for part in (change.progression, change.visual_progression):
        if part is not None and part.note:
            return part.note
    return None


# -- the board: scenes linked to the codex (§1.1 row 5) ---------------------------------

#: A name shorter than this is not looked for in a scene: one letter matches everything.
#: The same floor as `uclone_x.story.context`, whose lorebook match this follows.
_MIN_NAME = 2


def _start_state(work: StoryWork) -> tuple[StartState | None, str | None]:
    """`start.yaml` when the story was started with `story_start`; a note when it will not read."""
    found = work.read(START_FILE)
    if found is None:
        return None, None
    try:
        return StartState.model_validate(yaml.safe_load(found.text)), None
    except (yaml.YAMLError, ValidationError):
        return None, (
            "The record of how this story was started could not be read, so the board does not "
            "show what each chapter does or what the check after writing found."
        )


def _chapter_functions(arc: Mapping[str, object], chapters: int) -> list[list[str]]:
    """Each chapter's functions from the arc; none when the outline's chapters are not the arc's."""
    raw = arc.get("functions")
    if not isinstance(raw, list) or len(raw) != chapters:  # pyright: ignore[reportUnknownArgumentType]
        return []
    functions: list[list[str]] = []
    for item in raw:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(item, list):
            return []
        functions.append([str(f) for f in item])  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    return functions


def _proposed_names(pending: Sequence[Proposal]) -> dict[tuple[str, str], str]:
    """The names pending new-entry proposals give, by (kind, id): the cast before approval."""
    names: dict[tuple[str, str], str] = {}
    for proposal in pending:
        name = _new_name(proposal)
        if name is not None:
            names[(proposal.kind, proposal.entry_id)] = name
    return names


def _named_in(entry: CodexEntry, haystack: str) -> bool:
    return any(
        len(name) >= _MIN_NAME and name.casefold() in haystack
        for name in (entry.name, *entry.aliases)
    )


def _scene_entries(
    scene: Scene,
    index: CodexIndex,
    proposed: Mapping[tuple[str, str], str],
    haystack: str,
) -> list[SceneEntry]:
    """The entries the outline lists for the scene, then those it plants, pays off or names."""
    found: list[SceneEntry] = []
    seen: set[tuple[str, str]] = set()
    listed: tuple[tuple[CodexKind, list[str]], ...] = (
        ("characters", scene.characters),
        ("places", scene.places),
    )
    for kind, ids in listed:
        for entry_id in ids:
            if (kind, entry_id) in seen:
                continue
            seen.add((kind, entry_id))
            match = index.find(entry_id, kind)
            name = match[0].entry.name if match else proposed.get((kind, entry_id), entry_id)
            found.append(
                SceneEntry(kind=kind, id=entry_id, name=name, how="listed", in_codex=bool(match))
            )
    for item in index.items:
        key = (item.kind, item.entry.id)
        if key in seen:
            continue
        how: SceneLink | None = None
        entry = item.entry
        if isinstance(entry, ThreadEntry) and entry.planted_in == scene.id:
            how = "planted"
        elif isinstance(entry, ThreadEntry) and entry.paid_off_in == scene.id:
            how = "paid_off"
        elif _named_in(entry, haystack):
            how = "mentioned"
        if how is not None:
            seen.add(key)
            found.append(
                SceneEntry(kind=item.kind, id=entry.id, name=entry.name, how=how, in_codex=True)
            )
    return found


def _scene_changes(scene_id: str, index: CodexIndex) -> list[SceneChange]:
    """Every progression the codex places at the scene, in codex order."""
    changes: list[SceneChange] = []
    for item in index.items:
        entry = item.entry
        for progression in entry.progressions:
            if progression.at == scene_id:
                changes.append(
                    SceneChange(
                        kind=item.kind,
                        entry_id=entry.id,
                        entry_name=entry.name,
                        set=dict(progression.set),
                        add_looks=[],
                        remove_looks=[],
                        note=progression.note,
                    )
                )
        if isinstance(entry, CharacterEntry) and entry.visual is not None:
            for looks in entry.visual.progressions:
                if looks.at == scene_id:
                    changes.append(
                        SceneChange(
                            kind=item.kind,
                            entry_id=entry.id,
                            entry_name=entry.name,
                            set={},
                            add_looks=list(looks.add_tags),
                            remove_looks=list(looks.remove_tags),
                            note=looks.note,
                        )
                    )
    return changes


def _checked(scene_id: str, start: StartState | None) -> Literal["checked", "unread"] | None:
    if start is None:
        return None
    if scene_id in start.continuity_checked:
        return "checked"
    if scene_id in start.continuity_unread:
        return "unread"
    return None


def _rests_on(proposal: Proposal, scene_id: str) -> bool:
    """Whether a proposal quotes the scene or places its change at it."""
    change = proposal.change
    placed = [p.at for p in (change.progression, change.visual_progression) if p is not None]
    return scene_id in placed or any(e.scene_id == scene_id for e in proposal.evidence)


def _entry_view(entry: CodexEntry) -> EntryView:
    looks = entry.visual.tags if isinstance(entry, CharacterEntry) and entry.visual else None
    return EntryView(
        id=entry.id,
        name=entry.name,
        aliases=list(entry.aliases),
        profile=entry.profile,
        state=dict(entry.state),
        progressions=[p.model_dump(mode="json", exclude_defaults=True) for p in entry.progressions],
        looks=list(looks) if looks is not None else None,
    )
