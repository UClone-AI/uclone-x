"""The head's clone surface: `GET /api/clones`.

**Why this exists.** Nothing enumerated the clones that are *installed*. `GET /api/agents`
returned `session_mgr.list_agents(...)`, which is live instances: on an install where
nothing has been spawned it is empty, and that is exactly the first screen a new user
sees. (2026-09-27: `/api/agents` removed, #1775.) The clones themselves are directories under the agents root, and by P8 a second
head attached to the same Core would need the same enumeration -- so the Core owns it
(`uclone_x.core.agent_home.list_agent_homes`) and no head walks that directory itself.

**Why `clone` here and `AgentHome` there.** Design §3.1.1 puts the boundary between the
two vocabularies at the wire: what crosses `/api/` is the product's word, what lives
inside the Core keeps the Core's. So `AgentHome` and `agent_id` are unchanged
and this module says `clone` (`AgentInfo` went with `/api/agents`, 2026-09-27, #1775). It renames what is read, not what is typed.

**The open question this module decides (design §6.9).**

*Is a clone with a home that has never run shown as present, or as dormant?* **Dormant.**
Every row in this list is present -- that is what being in it means -- so `present` would
encode nothing, and a surface that wanted to draw a running clone differently would be
back to inferring it from a second endpoint. `dormant` names the one thing that differs
from `live`: installed, and not running. The reason string says so in words.

*What does a row say when its home is unreadable?* **It says so, in place.** The row keeps
its position with `status: "unreadable"`, `id: null`, and a reason naming the path and the
fault. Dropping it would render "this clone's home is damaged" identically to "this clone
is not installed" -- and of those two it is the damaged home the reader can act on, while
the name stays taken either way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from uclone_x.core.agent_home import (
    AGENTS_DIR_ENV_VAR,
    AgentHomeEntry,
    AgentHomeFault,
    AgentHomeState,
    list_agent_homes,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle; the app imports this module
    from uclone_x.ui.app import AgentSessionManager
    from uclone_x.ui.rooms import RoomStack

__all__ = [
    "CloneCatalog",
    "CloneListing",
    "CloneRootState",
    "CloneStatus",
    "CloneSummary",
    "merged_listing",
    "register_clone_routes",
]


class CloneStatus(StrEnum):
    """What is true of one clone, decided by the Core and never inferred by a surface."""

    #: An instance is running right now.
    LIVE = "live"
    #: Installed, and not running. See this module's docstring for why not `present`.
    DORMANT = "dormant"
    #: The name is taken by something that cannot serve as this clone's home.
    UNREADABLE = "unreadable"


class CloneRootState(StrEnum):
    """What was true of the directory the clones live in.

    Carried separately from the list because an empty list is the same value for three
    different facts, and only two of them are something a reader can act on.
    """

    READABLE = "readable"
    MISSING = "missing"
    UNREADABLE = "unreadable"


class CloneSummary(BaseModel):
    """One clone, in the shape a surface renders.

    Typed rather than a bare dict because this is a wire contract: a second head reads
    every field, and `dict[str, object]` pushes the checking onto whoever consumes it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The clone's name. It is also the directory name under the clones root, which is
    #: what makes it unique without a second index that could disagree.
    name: str
    #: The opaque identifier recorded in the home, minted the first time the clone runs.
    #: `null` before that, and for a home that could not be read.
    id: str | None
    status: CloneStatus
    #: Rendered verbatim. It states what is true of this clone, so that a row never
    #: depends on the reader working out why it looks the way it does.
    reason: str


class CloneListing(BaseModel):
    """Every clone installed here, and what was true of the directory holding them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    clones: tuple[CloneSummary, ...]
    #: The directory the clones were read from, named so a reader can check it.
    root: str
    root_state: CloneRootState
    #: Rendered before the list and never collapsed into it: "nothing is installed",
    #: "the directory could not be read" and "the directory is not there" are three
    #: facts that a bare empty list renders as one.
    reason: str


#: The remedy both root faults share. The folder can be redirected, and a mistyped
#: override is the commonest way it comes to be missing or unreadable -- so the variable
#: is named rather than left for the reader to remember.
_CHECK_THE_OVERRIDE = (
    f"If {AGENTS_DIR_ENV_VAR} is set, check that it names the folder your clones were "
    f"installed into."
)


def _root_reason(state: CloneRootState, root: Path, cause: str, count: int) -> str:
    """The sentence shown above the list, which is never allowed to be empty.

    `count` is the number of rows the list actually holds, and every branch here agrees
    with it. Counting anything else -- the homes on disk, say, while the list also carries
    running clones that have none -- is the two-predicate defect of #1088: a surface
    obeying `CloneListing.reason`'s contract then prints "No clone is installed yet."
    directly above a clone. Design §3.2.6: one predicate, not two.

    Nothing the Core wrote is concatenated here. `cause` is the operating system's own
    words, which belong to neither vocabulary, and everything else is stated in the
    product's (design §3.1.1).
    """
    if count == 0:
        if state is CloneRootState.MISSING:
            return (
                f"No clone is installed yet, and the folder that would hold them is not "
                f"there: {root}. {_CHECK_THE_OVERRIDE}"
            )
        if state is CloneRootState.UNREADABLE:
            return (
                f"Your clones could not be listed, so this is unknown rather than empty: "
                f"{root} could not be read ({cause}). Check that folder's permissions. "
                f"{_CHECK_THE_OVERRIDE}"
            )
        return f"No clone is installed yet. Clones live in {root}, and it is empty."

    counted = "One clone" if count == 1 else f"{count} clones"
    if state is CloneRootState.MISSING:
        return (
            f"{counted} on this machine. The folder that would hold your clones is not "
            f"there: {root}."
        )
    if state is CloneRootState.UNREADABLE:
        return (
            f"{counted} on this machine. {root} could not be read ({cause}), so more may "
            f"be installed than are listed here."
        )
    return f"{counted} on this machine, in {root}."


def _damage_reason(entry: AgentHomeEntry) -> str:
    """What a row says when its files cannot serve as a clone's, in the product's words.

    Rendered from `AgentHomeEntry.fault`, which is a value rather than a sentence, so the
    Core's nouns cannot reach the wire beside this module's (design §3.1.1). `cause`
    crosses verbatim, and it is written to be safe to: it is the operating system's own
    text, or the Core's rule-level explanation with this layer's nouns left off.

    A name fault carries `cause` for the same reason the others do. Without it `Scout/`
    and `my clone/` render identically, so the row says a name is wrong and never says
    which rule it broke -- an absence stating no cause (P6), and one the Core had already
    computed and this branch was throwing away.
    """
    if entry.fault is AgentHomeFault.UNUSABLE_NAME:
        return (
            f"This clone cannot be started: {entry.username!r} is not a name a clone can "
            f"have, so nothing can be installed under it. {entry.cause} Its folder is "
            f"{entry.path}."
        )
    if entry.fault is AgentHomeFault.NOT_A_DIRECTORY:
        return (
            f"This clone cannot be started: {entry.path} is not a folder, so the name is "
            f"taken by something that holds no clone."
        )
    if entry.fault is AgentHomeFault.EMPTY_ID:
        return (
            f"This clone cannot be started: the identity recorded at {entry.id_path} is "
            f"empty. That identity is how everything else refers to this clone, so a "
            f"replacement is not invented -- restore the file, or remove the clone and "
            f"set it up again."
        )
    if entry.fault is AgentHomeFault.UNREADABLE_CLONE_FILE:
        return (
            "This clone cannot be started: its settings file is missing or could not be "
            "read, so it has no name to be called by. Restore the file from a backup, or "
            "remove the clone and set it up again."
        )
    if entry.fault is AgentHomeFault.DUPLICATE_HANDLE:
        return (
            f"This clone cannot be started: another clone is also called "
            f"'{entry.username}', so that name reaches neither of them. Rename one of them."
        )
    # `UNREADABLE_ID`, and the `None` that `state is UNREADABLE` makes unreachable. Left
    # as the fallback rather than a raise so that every branch here is one a test reaches.
    return (
        f"This clone cannot be started: the identity recorded at {entry.id_path} could "
        f"not be read ({entry.cause}). Check that file's permissions."
    )


def _clone_from_home(entry: AgentHomeEntry, running: frozenset[str]) -> CloneSummary:
    """Turn one home into the row a surface draws, in the product's vocabulary."""
    if entry.state is AgentHomeState.UNREADABLE:
        return CloneSummary(
            name=entry.username,
            id=None,
            status=CloneStatus.UNREADABLE,
            reason=_damage_reason(entry),
        )
    # Seats are keyed by clone id (clone-data-scopes §4 step 3); a seat of a name that is no
    # clone is keyed by that name, which is why both are looked for.
    if entry.agent_id in running or entry.username in running:
        return CloneSummary(
            name=entry.username,
            id=entry.agent_id,
            status=CloneStatus.LIVE,
            reason=f"Running now. Its files are in {entry.path}.",
        )
    return CloneSummary(
        name=entry.username,
        id=entry.agent_id,
        status=CloneStatus.DORMANT,
        reason=_dormant_reason(entry),
    )


def _dormant_reason(entry: AgentHomeEntry) -> str:
    """What an installed clone that is not running says about itself.

    Two sentences, not one. A clone that has never run has no recorded identity, and a
    row that said only "installed and not running" would leave `id: null` looking like
    the damage the row below it actually is -- when it is the ordinary state of something
    nobody has started yet.
    """
    if entry.agent_id is None:
        return (
            f"Installed, and has never been started -- it is given an identity the first "
            f"time it runs. Its files are in {entry.path}."
        )
    return f"Installed and not running. Its files are in {entry.path}."


def _running_clone_names(session_mgr: AgentSessionManager, room_stack: RoomStack) -> frozenset[str]:
    """Every clone with a live instance behind it: the room seats.

    Every head is a one-seat room (design §1.3, D1), so a clone taking part in a
    conversation is built and cached by that room's `RoomAgentResolver`, and that cache is
    the only place a live clone exists. The manager's separate chat map was never written
    once rooms took over and was removed (#1899); reading it reported nothing.
    """
    del session_mgr  # kept in the signature; the manager holds no agent
    return room_stack.seated_agent_ids()


def clone_listing(
    session_mgr: AgentSessionManager,
    room_stack: RoomStack,
    *,
    unhomed: Iterable[str] = (),
) -> CloneListing:
    """Every installed clone, with the running ones marked as running.

    Public, and separate from the route, because the join is the part worth a test of its
    own: the Core supplies what is on disk, the live seats supply what is running, and the
    surface is handed the answer rather than the halves.

    `room_stack` is required rather than optional. It is the seat the default path uses,
    so a caller allowed to omit it would get a listing that is wrong in exactly the
    ordinary case, and silently.

    `unhomed` names clones the persona catalog holds with no folder behind them (one
    registered while the app runs). Each is a row, counted by the sentence above the list.
    """
    listing = list_agent_homes()
    running = _running_clone_names(session_mgr, room_stack)
    on_disk = frozenset(entry.username for entry in listing.homes) | frozenset(
        entry.agent_id for entry in listing.homes if entry.agent_id is not None
    )

    clones = [_clone_from_home(entry, running) for entry in listing.homes]
    for name in unhomed:
        if name in on_disk:
            continue
        clones.append(
            CloneSummary(
                name=name,
                # Addressed by its handle: it has no directory, so it was never given an id.
                id=name,
                status=CloneStatus.LIVE if name in running else CloneStatus.DORMANT,
                reason=(
                    "Defined only while this app runs, with no files of its own. Nothing "
                    "it learns will be kept after it stops."
                ),
            )
        )
    on_disk |= frozenset(unhomed)
    # A clone the user is talking to must not be missing from the list of clones. The
    # home is minted on bring-up, so this is the case where it was removed underneath a
    # running instance -- rare, and silent in a listing that reported only the disk.
    for username in sorted(running - on_disk):
        clones.append(
            CloneSummary(
                name=username,
                id=None,
                status=CloneStatus.LIVE,
                reason=(
                    f"Running now, with no files under {listing.root}. Nothing it learns "
                    f"will be kept after it stops."
                ),
            )
        )

    root_state = CloneRootState(listing.root_state.value)
    return CloneListing(
        clones=tuple(clones),
        root=str(listing.root),
        root_state=root_state,
        reason=_root_reason(root_state, listing.root, listing.cause, len(clones)),
    )


#: The persona side of a clone entry, as the head's persona payload writes it: a list of
#: entries each carrying `id` and `handle`, and the fields that go beside the list (the
#: tools an editor offers, where a save lands).
CloneCatalog = tuple[list[dict[str, Any]], dict[str, Any]]


def merged_listing(listing: CloneListing, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each row of `listing` with its persona entry's fields beside its status.

    `/api/clones` is the one clone resource (clone-data-scopes §3.7): every row carries
    `id`, `handle` and `display_name`, and the persona fields where the clone has a
    readable definition. A row joins its entry by id, else by handle; an unreadable row
    joins none, since its definition is exactly what could not be read.
    """
    by_id = {str(entry["id"]): entry for entry in entries}
    by_handle = {str(entry["handle"]): entry for entry in entries}
    rows: list[dict[str, Any]] = []
    for clone in listing.clones:
        row: dict[str, Any] = {"handle": clone.name, "display_name": {}}
        entry = None
        if clone.status is not CloneStatus.UNREADABLE:
            entry = by_id.get(clone.id) if clone.id is not None else None
            entry = entry if entry is not None else by_handle.get(clone.name)
        if entry is not None:
            row.update(entry)
        # The row's own facts win: they are what the disk held when it was listed, and the
        # sentence above the list was written from them.
        row.update(name=clone.name, id=clone.id, status=clone.status.value, reason=clone.reason)
        rows.append(row)
    return rows


def register_clone_routes(
    app: FastAPI,
    session_mgr: AgentSessionManager,
    room_stack: RoomStack,
    catalog: Callable[[], CloneCatalog] | None = None,
    *,
    refuse_cross_origin: Callable[[Request], None],
) -> None:
    """Mount `GET /api/clones` on `app`.

    With a `catalog`, each row also carries the clone's persona fields and the answer the
    catalog's extra fields, which is what the head reads; without one, the bare listing.

    Answers 200 for every state it can describe, including the ones that hold no clones:
    a root that cannot be read is a fact about the installation, not a failure of the
    request, and a 500 would leave the surface with nothing to say beyond "error". The
    reason carries the remedy instead, including the name of the variable that redirects
    the folder.
    """

    @app.get("/api/clones")
    async def list_clones(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """List the clones installed here, running or not, with what each is."""
        refuse_cross_origin(request)  # another site must not list clones (#2146)
        if catalog is None:
            return clone_listing(session_mgr, room_stack).model_dump(mode="json")
        # The disk is read before the catalog: loading the catalog installs and migrates
        # clones (clone-data-scopes §3.8), and the listing reports what it found, not what
        # its own read made.
        listing = clone_listing(session_mgr, room_stack)
        entries, extras = catalog()
        unhomed = [str(entry["handle"]) for entry in entries if entry["id"] == entry["handle"]]
        if unhomed:
            listing = clone_listing(session_mgr, room_stack, unhomed=unhomed)
        rows = merged_listing(listing, entries)
        return {
            **extras,
            "status": "ok",
            "clones": rows,
            "count": len(rows),
            "root": listing.root,
            "root_state": listing.root_state.value,
            "reason": listing.reason,
        }
