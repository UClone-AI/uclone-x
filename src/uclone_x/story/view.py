"""A story's view: its outline, its codex, and the changes waiting for a person (#1560).

The Files screen lists a story as a folder. This service reads one story whole, for a
person: the outline in reading order and in the order its scenes happen, the codex
grouped by kind, and every proposed codex change with what approving it would do.

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
codes.

**What cannot be read is said, never skipped** (P6): an outline, codex entry or proposal
file that does not load is listed with its reason, and a proposal that cannot be approved
now says why before anyone presses a button.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from uclone_x.errors import RoomNotFoundError, UnreadableRoomRecordError
from uclone_x.room.service import RoomService
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
    StoryFileError,
)
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


class StoryNotFoundError(StoryError):
    """The story is not in the library (any more)."""


class StoryNotWritableError(StoryError):
    """No existing conversation holds the story's writing lease, so nothing can be saved."""


class WriterBusyError(StoryError):
    """The conversation writing the story is answering now; deciding would change it under
    that turn."""


# -- values ---------------------------------------------------------------------------


class ConversationRef(BaseModel):
    """A conversation by id, with its title while it exists."""

    model_config = _MODEL

    room_id: str
    title: str | None
    exists: bool


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


class ChapterView(BaseModel):
    model_config = _MODEL

    id: str
    title: str
    act: str | None
    scenes: list[SceneView]


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
    outline: OutlineView | None
    #: Why there is no outline to show: none yet, or one that does not read.
    outline_note: str | None
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

        outline_view, outline_note, placements, scene_titles = self._outline(work)
        index = work.codex()
        groups: list[CodexGroup] = []
        for kind in CODEX_KINDS:
            entries = [_entry_view(item.entry) for item in index.items if item.kind == kind]
            groups.append(CodexGroup(kind=kind, label=_KIND_LABELS[kind], entries=entries))

        loaded, unreadable = work.proposals()
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
            decide_note=_decide_note(record, writer),
            outline=outline_view,
            outline_note=outline_note,
            codex=groups,
            codex_unreadable=[Unreadable(file=u.file, reason=u.reason) for u in index.unreadable],
            pending=pending,
            decided=decided,
            proposals_unreadable=[Unreadable(file=u.file, reason=u.reason) for u in unreadable],
        )

    @staticmethod
    def _outline(
        work: StoryWork,
    ) -> tuple[OutlineView | None, str | None, Mapping[str, Placement] | None, dict[str, str]]:
        try:
            found = work.outline()
        except StoryFileError as exc:
            return None, f"The outline could not be read: {exc}", None, {}
        if found is None:
            return None, "This story has no outline yet.", None, {}
        outline: Outline = found[0]
        placements = place_scenes(outline)
        written = set(work.written_scenes())
        chapters = [
            ChapterView(
                id=chapter.id,
                title=chapter.title,
                act=chapter.act,
                scenes=[
                    SceneView(
                        id=scene.id,
                        title=scene.title,
                        summary=scene.summary,
                        story_time=scene.story_time,
                        characters=list(scene.characters),
                        places=list(scene.places),
                        written=scene.id in written,
                    )
                    for scene in chapter.scenes
                ],
            )
            for chapter in outline.chapters
        ]
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
            entry_name = current[0].name if current is not None else None
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
            raise StoryNotWritableError(note or _NO_WRITER.format(title=record.title))
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


def _decide_note(record: StoryRecord, writer: ConversationRef | None) -> str | None:
    if writer is None:
        return _NO_WRITER.format(title=record.title)
    if not writer.exists:
        return (
            f"The conversation that was writing “{record.title}” no longer exists, so a "
            "decision cannot be saved. Open the story in a conversation, then decide here."
        )
    return None


def _note(proposal: Proposal) -> str | None:
    change = proposal.change
    for part in (change.progression, change.visual_progression):
        if part is not None and part.note:
            return part.note
    return None


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
