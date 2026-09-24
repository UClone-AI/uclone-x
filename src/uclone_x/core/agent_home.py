"""One directory per agent, named by the agent's username.

An agent's durable state used to be spread by convention: memory at
`~/.uclone/memory/<sanitized id>.json`, with the sanitizer folding every character it
did not like into `_`. That derivation is not injective -- `a.b`, `a/b`, `a b` and
`a_b` all land on `a_b.json`, which is four agents sharing one file with nothing
reporting it. Here the name is refused instead of repaired, so a username either
addresses exactly one directory or it is not a username.

The layout:

    <agents root>/<username>/
        id              an opaque, stable identifier, written once
        memory.json     that agent's cross-session memory

The directory name is the username, so the filesystem is what enforces uniqueness --
there is no second index that can disagree with it.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "AGENTS_DIR_ENV_VAR",
    "AGENT_ID_PREFIX",
    "DEFAULT_AGENTS_ROOT",
    "AgentHome",
    "AgentHomeEntry",
    "AgentHomeError",
    "AgentHomeFault",
    "AgentHomeListing",
    "AgentHomeRootState",
    "AgentHomeState",
    "default_agents_root",
    "list_agent_homes",
    "refuse_an_unusable_username",
]

#: Redirects every agent's home, as `UCLONE_SESSION_DIR` redirects the session store.
#: It replaces `UCLONE_MEMORY_DIR`, which named a directory of loose files that no
#: longer exists.
AGENTS_DIR_ENV_VAR = "UCLONE_AGENTS_DIR"

DEFAULT_AGENTS_ROOT = Path.home() / ".uclone" / "agents"

#: Distinguishes an agent's opaque identifier from its username at a glance, so a value
#: read out of a log or a record says which of the two it is.
AGENT_ID_PREFIX = "agt_"

#: `\Z` rather than `$`: `$` also matches immediately before a trailing newline, so
#: `scout\n` would pass and then become a second directory that renders as `scout` in
#: every listing and log. That is the ambiguity this whole rule exists to prevent.
_USERNAME_RE = re.compile(r"\A[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?\Z")

_MAX_USERNAME_LENGTH = 64

#: Windows resolves these to devices wherever they appear as a path component, so `con/`
#: is not a directory there. The rule's whole argument is that a username must mean one
#: directory on every machine, and these mean none. (Windows also resolves `con.txt`; the
#: name rule already refuses '.', so the bare spellings are the whole reachable set.)
_RESERVED_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{digit}" for digit in range(1, 10)}
    | {f"lpt{digit}" for digit in range(1, 10)}
)


#: The allowed set, named once. It is stated in both the refusal a caller in this layer
#: reads and the explanation a head renders, and two spellings of one rule is the drift
#: this constant exists to prevent.
_ALLOWED_IN_A_NAME = (
    "Allowed: lowercase letters, digits, '-' and '_', starting and ending with a letter or digit."
)


class AgentHomeError(Exception):
    """A username cannot name a directory, or a home could not be established.

    `explanation` carries the part of the refusal that names *which* rule the value broke,
    written without this module's nouns. The message itself is for a caller in this layer
    and says `agent username`; a head renders the same refusal in the product's own words
    (design §3.1.1), and without this it has only the fault's name -- so two names broken
    by two different rules arrive at a reader as one sentence, with the remedy the Core
    already computed thrown away.

    Empty for the failures that have no rule-level explanation to give, which is every
    raise outside `refuse_an_unusable_username`.
    """

    def __init__(self, message: str, *, explanation: str = "") -> None:
        super().__init__(message)
        self.explanation = explanation


def default_agents_root() -> Path:
    """Resolve the root every agent home sits under, honouring `UCLONE_AGENTS_DIR`."""
    override = os.environ.get(AGENTS_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_AGENTS_ROOT


def refuse_an_unusable_username(username: str) -> None:
    """Refuse a username that cannot become one unambiguous directory name.

    Modelled on `room.service._refuse_an_id_the_derivation_cannot_carry`: the failure is
    raised against the name that caused it, rather than surfacing later as one agent
    reading another's facts with no error anywhere.

    Uppercase is refused rather than lowercased, which is the rule most likely to look
    arbitrary. It is not: macOS and Windows filesystems are case-insensitive, so
    `Scout/` and `scout/` are the *same* directory there and two different ones on
    Linux. Accepting both spellings would make an agent's identity depend on which
    machine it runs on.

    Raises:
        AgentHomeError: The username is empty, too long, reserved by a platform, or
            holds a character that cannot appear in a directory name this code is
            willing to derive.
    """
    if not username:
        raise AgentHomeError(
            "an agent username is empty. It names the directory holding that agent's "
            "identity and memory, so it has to be a name.",
            explanation="The name is empty, and a folder has to be called something.",
        )
    if len(username) > _MAX_USERNAME_LENGTH:
        raise AgentHomeError(
            f"agent username {username!r} is {len(username)} characters; the limit is "
            f"{_MAX_USERNAME_LENGTH}. It becomes a directory name, and filesystems "
            f"differ on where they stop accepting one.",
            explanation=(
                f"It is {len(username)} characters long, and the limit is {_MAX_USERNAME_LENGTH}."
            ),
        )
    if not _USERNAME_RE.match(username):
        offenders = sorted({char for char in username if not re.match(r"[a-z0-9_-]", char)})
        detail = (
            f"It holds {', '.join(repr(char) for char in offenders)}."
            if offenders
            else "It starts or ends with '-' or '_'."
        )
        raise AgentHomeError(
            f"agent username {username!r} is not usable as a directory name. {detail} "
            f"{_ALLOWED_IN_A_NAME} Uppercase is refused rather than folded because a "
            f"case-insensitive filesystem would give two spellings one directory, and a "
            f"case-sensitive one would give them two.",
            explanation=f"{detail} {_ALLOWED_IN_A_NAME}",
        )
    if username in _RESERVED_DEVICE_NAMES:
        raise AgentHomeError(
            f"agent username {username!r} is a reserved device name on Windows, where it "
            f"names a device rather than a directory. Refusing it everywhere keeps one "
            f"username meaning one directory on every machine.",
            explanation=(
                f"{username!r} is a reserved device name on Windows, where it names a "
                f"device rather than a folder."
            ),
        )


@dataclass(frozen=True)
class AgentHome:
    """The directory holding one agent's identity and durable state."""

    username: str
    path: Path

    @classmethod
    def for_username(cls, username: str, root: Path | None = None) -> AgentHome:
        """Locate the home for `username`, refusing a name a directory cannot carry.

        Performs no I/O: a caller that only wants to know where an agent's state would
        live should not create it as a side effect of asking.
        """
        refuse_an_unusable_username(username)
        base = root if root is not None else default_agents_root()
        path = base / username
        # The name rule already excludes every separator, so this can only fire if that
        # rule is weakened. It is here because the rule is the *only* thing standing
        # between a username and `mkdir(parents=True)`, and a hole in it would otherwise
        # become a write outside the root rather than a refusal (cf. `session.py`, which
        # validates the id and then resolves the path).
        if path.parent != base or path.name != username:
            raise AgentHomeError(
                f"agent username {username!r} does not resolve to a directory directly "
                f"inside {base}. Refusing rather than writing outside the agents root."
            )
        return cls(username=username, path=path)

    @property
    def id_path(self) -> Path:
        """Path of the file holding this agent's opaque identifier."""
        return self.path / "id"

    @property
    def memory_path(self) -> Path:
        """Path of this agent's cross-session memory document."""
        return self.path / "memory.json"

    def recorded_agent_id(self) -> str | None:
        """This agent's identifier if one has been written, and None before that.

        Public because enumerating what is installed has to read an id *without* bringing
        the agent into being: `agent_id()` mints and writes, which would turn a listing
        into an installer. Same rule as `for_username` -- asking must not create.

        Raises:
            AgentHomeError: The `id` file exists and holds nothing, which is damage this
                module refuses to repair; see `_read_id_if_present`.
        """
        return self._read_id_if_present()

    def agent_id(self) -> str:
        """Return this agent's opaque identifier, minting it on first use.

        The value is written to a temporary file, flushed, and only then linked into
        place under its real name. A reader therefore sees either no `id` file or a
        complete one -- never the empty file that `O_CREAT | O_EXCL` leaves visible
        between the create and the write, which a concurrently starting process would
        have read as damage and refused.

        `os.link` fails if the name already exists, so two processes reaching a fresh
        home at once cannot both believe they assigned the id: the loser discards its
        candidate and reads the winner's value.
        """
        try:
            self.path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Including the `FileExistsError` a regular file at `<root>/<username>` raises.
            # Re-raised as this module's error so the UI reports it as an agent-home fault
            # rather than letting a bare `OSError` fall through to a 500 labelled "Session
            # operation failed" -- the mis-attribution this module exists to stop, arriving
            # through a different door (P6).
            raise AgentHomeError(
                f"agent {self.username!r} has no usable home at {self.path}: {exc}"
            ) from exc
        recorded = self._read_id_if_present()
        if recorded is not None:
            return recorded

        self._publish_id(f"{AGENT_ID_PREFIX}{uuid.uuid4().hex}")

        settled = self._read_id_if_present()
        if settled is None:
            raise AgentHomeError(
                f"{self.id_path} is missing immediately after being written, so agent "
                f"{self.username!r} has no identity this process can report."
            )
        return settled

    def _publish_id(self, candidate: str) -> None:
        """Land `candidate` at `id_path` unless another process got there first."""
        descriptor, staged = tempfile.mkstemp(dir=self.path, prefix=".id-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(candidate + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(staged, self.id_path)
            except FileExistsError:
                # Another process minted first. Its value is the agent's id; ours was
                # never anyone's, so there is nothing to reconcile.
                return
            except OSError as link_failure:
                # A root on a filesystem with no hard links (exFAT, FAT32, some network
                # mounts) fails here with `EPERM` or `ENOTSUP`. Named as what it is, for
                # the reason the `mkdir` above is.
                raise AgentHomeError(
                    f"{self.id_path} could not be written, so agent {self.username!r} has "
                    f"no identity: {link_failure}"
                ) from link_failure
            self._fsync_directory()
        finally:
            os.unlink(staged)

    def _fsync_directory(self) -> None:
        """Make the new directory entry durable, not just the bytes it points at.

        Without this the file can survive a crash with no name, or the name with no
        bytes -- and an empty `id` is refused and never repaired, so that outcome would
        strand the agent permanently.
        """
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _read_id_if_present(self) -> str | None:
        """This agent's recorded id, or None if it has not been written yet.

        Raises:
            AgentHomeError: The file exists but holds nothing. Minting a replacement
                would make every system still holding the old id wrong, so the damage is
                reported instead of covered.
        """
        try:
            recorded = self.id_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not recorded:
            raise AgentHomeError(
                f"{self.id_path} is empty, so agent {self.username!r} has a home but no "
                f"identity. Refusing to mint a replacement: an agent whose id changes is "
                f"a different agent to everything that recorded the old one."
            )
        return recorded


class AgentHomeRootState(StrEnum):
    """What was true of the agents root itself when it was listed.

    A listing returns an empty tuple of homes in three of these four situations, so the
    tuple alone tells a caller nothing. This is the value that does (P6).
    """

    #: The root was read. It may still hold nothing, which an empty `homes` says.
    READABLE = "readable"
    #: The root does not exist. Ordinary before anything has been installed, and also
    #: what a mistyped `UCLONE_AGENTS_DIR` looks like from here.
    MISSING = "missing"
    #: The root exists and could not be read -- permissions, or a path that is not a
    #: directory. Whatever is installed under it is unknown, not absent.
    UNREADABLE = "unreadable"


class AgentHomeState(StrEnum):
    """What was true of one directory under the agents root."""

    #: The directory names one agent and can be read. It may hold no `id` yet, which is
    #: the ordinary state of an agent nothing has brought up.
    INSTALLED = "installed"
    #: The directory occupies a name and cannot serve as that agent's home. Reported in
    #: place rather than skipped: the name is taken either way, and a row that vanishes
    #: reads as "not installed" -- the one thing it is not.
    UNREADABLE = "unreadable"


class AgentHomeFault(StrEnum):
    """Why a directory under the agents root cannot serve as one agent's home.

    A *value*, not a sentence, and that is the point. A head renders this layer's findings
    in the product's own vocabulary (design §3.1.1), so a sentence composed here would
    arrive on the wire carrying this layer's nouns -- `agent`, `agent home`, `id file` --
    beside the head's word for the same object. Naming the fault and leaving the wording
    to the boundary is what makes the two halves impossible to mix.
    """

    #: The directory's name cannot be a username, so nothing can be installed under it.
    UNUSABLE_NAME = "unusable_name"
    #: The name is taken by something that is not a directory.
    NOT_A_DIRECTORY = "not_a_directory"
    #: The `id` file exists and holds nothing. Damage this module refuses to repair,
    #: because minting a replacement makes every system holding the old id wrong.
    EMPTY_ID = "empty_id"
    #: The `id` file could not be read at all -- a permission, a directory in its place.
    UNREADABLE_ID = "unreadable_id"


@dataclass(frozen=True)
class AgentHomeEntry:
    """One directory under the agents root, and what could be learned about it.

    Carries facts and paths, never a rendered sentence: a head states them in its own
    vocabulary (design §3.1.1). `fault` is set exactly when `state` is `UNREADABLE`, and
    `cause` holds the words a head may show as they are -- the operating system's, which
    belong to nobody's vocabulary, or, where the refusal is this module's own, the
    `AgentHomeError.explanation` written without this module's nouns. A head still
    composes the sentence; what `cause` adds is the half of it only this layer knows,
    which for a name is *which rule it broke*.
    """

    username: str
    path: Path
    #: Where this agent's identifier is recorded. Named here because the layout is this
    #: module's: a head that composed `path / "id"` itself would be deciding a filename
    #: it does not own.
    id_path: Path
    state: AgentHomeState
    agent_id: str | None
    fault: AgentHomeFault | None
    #: The underlying error text, or empty when the fault has no underlying error and
    #: when there is no fault at all.
    cause: str


@dataclass(frozen=True)
class AgentHomeListing:
    """Every agent home under one root, and what was true of the root itself."""

    root: Path
    root_state: AgentHomeRootState
    #: The operating system's own words for why the root could not be read. Empty for a
    #: root that was read and for one that is simply not there, neither of which has an
    #: error behind it.
    cause: str
    homes: tuple[AgentHomeEntry, ...]


def list_agent_homes(root: Path | None = None) -> AgentHomeListing:
    """Enumerate the agent homes under `root`, defaulting to the resolved agents root.

    Nothing else enumerates them. `list_agents` on the head's session manager returns
    *live instances*, which is empty on an install where nothing has been spawned -- so a
    surface built on it shows nothing on exactly the first screen a new user sees.

    This function never raises for a fault it can describe. A root that cannot be read
    and a root holding nothing both produce an empty `homes`, and a damaged single home
    would, if its refusal escaped, empty the whole list; each is instead reported as the
    state it is (P6). The only exception left is a caller passing a root it cannot even
    name, which is a programming error rather than an installation one.

    Performs no writes: asking what is installed must not install anything, for the same
    reason `AgentHome.for_username` performs no I/O.
    """
    agents_root = root if root is not None else default_agents_root()
    try:
        entries = sorted(agents_root.iterdir(), key=lambda entry: entry.name)
    except FileNotFoundError:
        # No `cause`: "the directory is not there" is the state itself, and the operating
        # system's `ENOENT` text adds nothing a reader could act on that `root` does not.
        return AgentHomeListing(
            root=agents_root,
            root_state=AgentHomeRootState.MISSING,
            cause="",
            homes=(),
        )
    except OSError as unreadable:
        return AgentHomeListing(
            root=agents_root,
            root_state=AgentHomeRootState.UNREADABLE,
            cause=str(unreadable),
            homes=(),
        )

    homes = tuple(
        _describe_agent_home(entry)
        for entry in entries
        # A leading dot cannot be a username under `_USERNAME_RE`, so such an entry never
        # competes for one. Skipping it rather than reporting it as damage keeps
        # `.DS_Store` and editor droppings out of a list of agents.
        if not entry.name.startswith(".")
    )
    return AgentHomeListing(
        root=agents_root,
        root_state=AgentHomeRootState.READABLE,
        cause="",
        homes=homes,
    )


def _describe_agent_home(path: Path) -> AgentHomeEntry:
    """Report one directory under the agents root, never raising for what it finds."""
    name = path.name
    home = AgentHome(username=name, path=path)
    try:
        refuse_an_unusable_username(name)
    except AgentHomeError as unusable:
        return AgentHomeEntry(
            username=name,
            path=path,
            id_path=home.id_path,
            state=AgentHomeState.UNREADABLE,
            agent_id=None,
            fault=AgentHomeFault.UNUSABLE_NAME,
            # The explanation, not `str(unusable)`: the message says `agent username`,
            # which is this layer's noun and may not cross the wire beside the head's
            # word for the same object. The explanation states the same rule in words
            # that name no object at all. Even for the character-class refusal it is not
            # a suffix of the message -- the opening clause and the trailing sentence
            # are both gone, so `str(e).endswith(e.explanation)` is False; the other three are
            # independent paraphrases -- `directory` becomes `folder`, the sentences
            # arguing the rule are dropped, and the empty-name pair share no wording at
            # all -- so a head can name which rule the name broke.
            cause=unusable.explanation,
        )

    if not path.is_dir():
        # The `FileExistsError` case `AgentHome.agent_id` documents, seen from outside:
        # the name is taken and every bring-up under it fails. No underlying error to
        # quote -- the fault is the whole fact.
        return AgentHomeEntry(
            username=name,
            path=path,
            id_path=home.id_path,
            state=AgentHomeState.UNREADABLE,
            agent_id=None,
            fault=AgentHomeFault.NOT_A_DIRECTORY,
            cause="",
        )

    try:
        recorded = home.recorded_agent_id()
    except AgentHomeError as damage:
        return AgentHomeEntry(
            username=name,
            path=path,
            id_path=home.id_path,
            state=AgentHomeState.UNREADABLE,
            agent_id=None,
            fault=AgentHomeFault.EMPTY_ID,
            cause=str(damage),
        )
    except OSError as unreadable:
        return AgentHomeEntry(
            username=name,
            path=path,
            id_path=home.id_path,
            state=AgentHomeState.UNREADABLE,
            agent_id=None,
            fault=AgentHomeFault.UNREADABLE_ID,
            cause=str(unreadable),
        )

    return AgentHomeEntry(
        username=name,
        path=path,
        id_path=home.id_path,
        state=AgentHomeState.INSTALLED,
        agent_id=recorded,
        fault=None,
        cause="",
    )
