"""The head's clone surface: `GET /api/clones`.

**Why this exists.** Nothing enumerated the clones that are *installed*. `GET /api/agents`
returns `session_mgr.list_agents(...)`, which is live instances: on an install where
nothing has been spawned it is empty, and that is exactly the first screen a new user
sees. The clones themselves are directories under the agents root, and by P8 a second
head attached to the same Core would need the same enumeration -- so the Core owns it
(`uclone_x.core.agent_home.list_agent_homes`) and no head walks that directory itself.

**Why `clone` here and `AgentHome` there.** Design §3.1.1 puts the boundary between the
two vocabularies at the wire: what crosses `/api/` is the product's word, what lives
inside the Core keeps the Core's. So `AgentInfo`, `AgentHome` and `agent_id` are unchanged
and this module says `clone`. It renames what is read, not what is typed.

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

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
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

__all__ = ["CloneListing", "CloneRootState", "CloneStatus", "CloneSummary", "register_clone_routes"]


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
    if entry.username in running:
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
    """Every clone with a live instance behind it, wherever it is seated.

    Two seats, not one. `AgentSessionManager.list_agents()` returns `_agents`, the chat
    map; a clone taking part in a conversation is built and cached by that room's
    `RoomAgentResolver` and never written back into it. Under D1 (design §1.3) every new
    conversation is a room, single-clone ones included, so the conversation seat is the
    *default* path -- reading only the chat map reports the ordinary running clone as
    "Installed and not running" while the user is reading its reply.

    The same shape as `AgentSessionManager.memory_for`, whose docstring already records
    the rule this restores: the ids collide by design, because an install's clones are
    both its chat clones and the seats a conversation puts them in.
    """
    chatting = frozenset(agent.agent_id for agent in session_mgr.list_agents())
    return chatting | room_stack.seated_agent_ids()


def clone_listing(session_mgr: AgentSessionManager, room_stack: RoomStack) -> CloneListing:
    """Every installed clone, with the running ones marked as running.

    Public, and separate from the route, because the join is the part worth a test of its
    own: the Core supplies what is on disk, the live seats supply what is running, and the
    surface is handed the answer rather than the halves.

    `room_stack` is required rather than optional. It is the seat the default path uses,
    so a caller allowed to omit it would get a listing that is wrong in exactly the
    ordinary case, and silently.
    """
    listing = list_agent_homes()
    running = _running_clone_names(session_mgr, room_stack)
    on_disk = frozenset(entry.username for entry in listing.homes)

    clones = [_clone_from_home(entry, running) for entry in listing.homes]
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


def register_clone_routes(
    app: FastAPI, session_mgr: AgentSessionManager, room_stack: RoomStack
) -> None:
    """Mount `GET /api/clones` on `app`.

    Answers 200 for every state it can describe, including the ones that hold no clones:
    a root that cannot be read is a fact about the installation, not a failure of the
    request, and a 500 would leave the surface with nothing to say beyond "error". The
    reason carries the remedy instead, including the name of the variable that redirects
    the folder.
    """

    @app.get("/api/clones")
    async def list_clones() -> CloneListing:  # pyright: ignore[reportUnusedFunction]
        """List the clones installed here, running or not."""
        return clone_listing(session_mgr, room_stack)
