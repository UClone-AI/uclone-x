"""Deciding a proposed codex change: what it would do, and applying or rejecting it (#1557, #1560).

A proposal is made by the model (`story_codex propose`, `character_sheet save`) and decided
by a person. It is decided in one of two places, and both come here, so the checks are the
same wherever the decision is made:

* **in a conversation**: `story_codex apply`, which the runtime runs only once a person
  answered its approval request (`ToolContext.approved_by_person`);
* **in the story view**: the person presses Approve or Reject on the proposal
  (`uclone_x.story.view`). No story or file tool reaches that path; a persona with an
  unconfined shell (Clone's `bash_run`) can call its local API (#1589), so there
  `decided_in` is not proof that a person decided.

Every write goes through `StoryWork.write`, so the lease and the digest checks apply. A
proposal is applied only to the entry it was made against: its `entry_digest` must still be
the entry file's digest, so a person never approves a change against a version of the entry
they did not see.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import ValidationError

from uclone_x.story.library import StoryError
from uclone_x.story.schemas import (
    CharacterEntry,
    CodexEntry,
    CodexKind,
    Proposal,
    StoryFileError,
    changed_entry,
    describe_invalid,
    dump_file,
    entry_model,
)
from uclone_x.story.timeline import Placement, entry_snapshot
from uclone_x.story.work import StoryWork, entry_file, proposal_file

__all__ = [
    "AppliedProposal",
    "ChangeLine",
    "DecidedIn",
    "ProposalPreview",
    "apply_proposal",
    "check_applies",
    "pending_proposal",
    "preview_proposal",
    "reject_proposal",
]

#: Where a proposal was decided: in a conversation, or in the story view.
DecidedIn = Literal["conversation", "story_view"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def check_applies(kind: CodexKind, entry: CodexEntry, proposal: Proposal) -> CodexEntry:
    """`entry` with `proposal`'s change made, validated as its file would be.

    Raises:
        StoryError: the changed entry would not fit its file.
        StoryFileError: the change is to how a character looks, and `entry` is not one.
    """
    data = changed_entry(entry, proposal.change)
    try:
        return entry_model(kind).model_validate(data)
    except ValidationError as exc:
        raise StoryError(describe_invalid(entry_file(kind, entry.id), exc)) from exc


def pending_proposal(
    work: StoryWork, proposal_id: str, expected_digest: str | None = None
) -> tuple[Proposal, str]:
    """The proposal `proposal_id` and its file's digest, if it is still undecided.

    With `expected_digest` -- the digest of the proposal a person was shown -- a proposal
    whose file changed since is refused, so the decision is about what they saw.

    Raises:
        StoryError: there is no such proposal, it was already decided, or it changed.
        StoryFileError: its file does not fit.
    """
    proposal, digest = work.proposal(proposal_id)
    if proposal.status != "pending":
        raise StoryError(
            f"Proposal '{proposal_id}' was already {proposal.status}, so nothing was changed."
        )
    if expected_digest is not None and digest != expected_digest:
        raise StoryError(
            f"Proposal '{proposal_id}' changed after it was shown, so nothing was changed. "
            "Look at it again before deciding."
        )
    return proposal, digest


@dataclass(frozen=True)
class AppliedProposal:
    """A proposal applied: the entry it changed, and anything the person should know."""

    proposal_id: str
    kind: CodexKind
    entry_id: str
    #: The entry file, relative to the story folder.
    entry_path: str
    notes: list[str] = field(default_factory=list[str])


def apply_proposal(
    work: StoryWork,
    proposal_id: str,
    *,
    room_id: str,
    decided_in: DecidedIn,
    expected_proposal_digest: str | None = None,
) -> AppliedProposal:
    """Make the change `proposal_id` proposes, writing as the conversation `room_id`.

    The caller has a person's approval; this function does not ask for one. It refuses a
    proposal already decided, one whose entry changed after it was made, and -- when
    `expected_proposal_digest` is given -- one whose file changed after the person saw it.

    Raises:
        StoryError: refused, with the reason; nothing was written.
        StoryReadOnlyError: `room_id` does not hold the story's lease.
    """
    proposal, proposal_digest = pending_proposal(work, proposal_id, expected_proposal_digest)
    current = work.entry(proposal.kind, proposal.entry_id)
    if current is None:
        raise StoryError(
            f"The entry '{proposal.entry_id}' is no longer in the codex, so proposal "
            f"'{proposal_id}' was not applied."
        )
    entry, digest = current
    if digest != proposal.entry_digest:
        raise StoryError(
            f"The entry '{proposal.entry_id}' changed after proposal '{proposal_id}' was "
            "made, so it was not applied. Reject it and propose the change again."
        )
    changed = check_applies(proposal.kind, entry, proposal)
    relative = entry_file(proposal.kind, proposal.entry_id)
    old = work.read(relative)
    work.write(relative, dump_file(changed, compact=True), room_id=room_id, expected_digest=digest)
    notes: list[str] = []
    if old is not None and "#" in old.text:
        notes.append(
            f"{relative} was rewritten with the change, and the comments it had were not kept."
        )
    decided = proposal.model_copy(
        update={"status": "applied", "decided_at": _now(), "decided_in": decided_in}
    )
    try:
        work.write(
            proposal_file(proposal_id),
            dump_file(decided, compact=True),
            room_id=room_id,
            expected_digest=proposal_digest,
        )
    except StoryError as exc:
        notes.append(
            f"The change was applied, but proposal '{proposal_id}' was not marked as applied: {exc}"
        )
    return AppliedProposal(proposal_id, proposal.kind, proposal.entry_id, relative, notes)


def reject_proposal(
    work: StoryWork,
    proposal_id: str,
    *,
    room_id: str,
    reason: str | None,
    decided_in: DecidedIn,
    expected_proposal_digest: str | None = None,
) -> str:
    """Mark `proposal_id` rejected, writing as `room_id`; return the proposal's file.

    Raises:
        StoryError: there is no such proposal, or it was already decided.
        StoryReadOnlyError: `room_id` does not hold the story's lease.
    """
    proposal, digest = pending_proposal(work, proposal_id, expected_proposal_digest)
    decided = proposal.model_copy(
        update={
            "status": "rejected",
            "decided_at": _now(),
            "reason": reason,
            "decided_in": decided_in,
        }
    )
    relative = proposal_file(proposal_id)
    work.write(relative, dump_file(decided, compact=True), room_id=room_id, expected_digest=digest)
    return relative


# -- what a proposal would do, for a person deciding it ---------------------------------


@dataclass(frozen=True)
class ChangeLine:
    """One value a proposal changes, as it is and as it would be.

    `at` is the scene after which the change holds, or `None` for a change to how a
    character looks from the start. `before` and `after` are the value at that moment of
    the story; `None` in `before` means the entry has no such value yet. When the scene is
    not in the outline the moment cannot be placed, and `before` is the entry's starting
    value with `placed` false.
    """

    what: str
    at: str | None
    before: Any
    after: Any
    placed: bool = True


@dataclass(frozen=True)
class ProposalPreview:
    """What approving a proposal would do, shown before a person decides.

    `changes` states each value before and after. `diff` is a line diff of the entry file as
    it is and as approving would write it -- the bytes, not a paraphrase. `blocked` says why
    approving it now would be refused, or is `None`.
    """

    entry_name: str | None
    entry_path: str
    changes: list[ChangeLine]
    diff: list[str]
    blocked: str | None


def _moment(
    entry: CodexEntry, placements: Mapping[str, Placement] | None, at: str
) -> tuple[dict[str, Any], list[str] | None, bool]:
    """The entry's state and visual tags once scene `at` has ended, and whether it placed."""
    if placements is None or at not in placements:
        visual = entry.visual if isinstance(entry, CharacterEntry) else None
        return dict(entry.state), (list(visual.tags) if visual is not None else None), False
    snapshot = entry_snapshot(entry, placements, at, through_scene=True)
    return snapshot.state, snapshot.visual_tags, True


def _visual_fields(entry: CodexEntry) -> dict[str, Any]:
    if isinstance(entry, CharacterEntry) and entry.visual is not None:
        return entry.visual.model_dump(mode="json")
    return {}


def _change_lines(
    proposal: Proposal,
    entry: CodexEntry,
    changed: CodexEntry,
    placements: Mapping[str, Placement] | None,
) -> list[ChangeLine]:
    # The change is added after the entry's own progressions, so once its scene has ended
    # it holds: `after` is the value it sets, on top of the entry as it stands then.
    lines: list[ChangeLine] = []
    change = proposal.change
    if change.progression is not None:
        at = change.progression.at
        before, _, placed = _moment(entry, placements, at)
        for key, value in change.progression.set.items():
            lines.append(ChangeLine(key, at, before.get(key), value, placed))
    if change.visual_progression is not None:
        vp = change.visual_progression
        _, before_tags, placed = _moment(entry, placements, vp.at)
        kept = [t for t in before_tags or [] if t not in set(vp.remove_tags)]
        after_tags = kept + [t for t in vp.add_tags if t not in kept]
        lines.append(ChangeLine("looks", vp.at, before_tags, after_tags, placed))
    if change.visual is not None:
        old = _visual_fields(entry)
        new = _visual_fields(changed)
        for key in change.visual.model_dump(exclude_none=True):
            lines.append(ChangeLine(f"looks: {key}", None, old.get(key), new.get(key)))
    return lines


def preview_proposal(
    work: StoryWork, proposal: Proposal, placements: Mapping[str, Placement] | None
) -> ProposalPreview:
    """What approving `proposal` would change, or why it cannot be approved now.

    `placements` is the outline's story-time order (`place_scenes`), or `None` when the
    story has no outline that reads.
    """
    relative = entry_file(proposal.kind, proposal.entry_id)
    try:
        current = work.entry(proposal.kind, proposal.entry_id)
    except (StoryError, StoryFileError) as exc:
        return ProposalPreview(
            None, relative, [], [], f"It cannot be approved: the entry does not read. {exc}"
        )
    if current is None:
        return ProposalPreview(
            None,
            relative,
            [],
            [],
            f"It cannot be approved: the entry '{proposal.entry_id}' is no longer in the "
            "codex. Reject it.",
        )
    entry, digest = current
    blocked = (
        None
        if digest == proposal.entry_digest
        else (
            f"It cannot be approved: the entry '{proposal.entry_id}' changed after this was "
            "proposed. Reject it, and ask for the change again if it is still wanted."
        )
    )
    try:
        changed = check_applies(proposal.kind, entry, proposal)
    except (StoryError, StoryFileError) as exc:
        return ProposalPreview(
            entry.name, relative, [], [], blocked or f"It cannot be approved: {exc}"
        )
    before = work.read(relative)
    diff = list(
        difflib.unified_diff(
            (before.text if before is not None else "").splitlines(),
            dump_file(changed, compact=True).splitlines(),
            fromfile=f"{relative} (now)",
            tofile=f"{relative} (if approved)",
            lineterm="",
            n=2,
        )
    )
    return ProposalPreview(
        entry.name, relative, _change_lines(proposal, entry, changed, placements), diff, blocked
    )
