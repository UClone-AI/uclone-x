"""One directory per clone, named by the clone's opaque id.

An agent's durable state used to be spread by convention: memory at
`~/.uclone/memory/<sanitized id>.json`, with the sanitizer folding every character it
did not like into `_`. That derivation is not injective -- `a.b`, `a/b`, `a b` and
`a_b` all land on `a_b.json`, which is four agents sharing one file with nothing
reporting it. Here the name is refused instead of repaired, so a handle either
addresses exactly one clone or it is not a handle.

The layout (clone-data-scopes §3.2):

    <agents root>/<agt_id>/
        id              the directory name, kept for the reader that predates it
        clone.yaml      handle, display_name and the persona fields
        knowledge.sqlite3  that clone's knowledge graph: its cross-session facts (uGraph)
        memory.json     an earlier build's memory, imported once and then set aside

The directory is named by the id, not the handle, so a handle can change -- or be
written in any script, through `display_name` -- without moving anything that refers to
the clone. The handle lives in `clone.yaml`, and `resolve_handle` is the one reader that
turns a handle into an id. Two clones claiming one handle are refused when written and
reported when read (`DuplicateHandleError`), never settled by directory order.

A directory named by a handle rather than an id is the layout before 2026-09-27; the
migration in `uclone_x.agent.clone_store` renames it.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import uuid
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX (Windows): clone writes go unserialised
    fcntl = None

__all__ = [
    "AGENTS_DIR_ENV_VAR",
    "AGENT_ID_PREFIX",
    "CLONE_FILE_NAME",
    "ONTOLOGY_FILE_NAME",
    "DEFAULT_AGENTS_ROOT",
    "AgentHome",
    "AgentHomeEntry",
    "AgentHomeError",
    "AgentHomeFault",
    "AgentHomeListing",
    "AgentHomeRootState",
    "AgentHomeState",
    "CloneNotFoundError",
    "DuplicateHandleError",
    "clone_handles",
    "clone_id_of",
    "clone_root_lock",
    "create_clone",
    "default_agents_root",
    "is_agent_id",
    "list_agent_homes",
    "handle_of",
    "mint_agent_id",
    "peer_handles",
    "read_clone_file",
    "refuse_an_unusable_username",
    "replace_clone_file",
    "resolve_handle",
    "seat_id_for",
]

#: Redirects every agent's home, as `UCLONE_SESSION_DIR` redirects the session store.
#: It replaces `UCLONE_MEMORY_DIR`, which named a directory of loose files that no
#: longer exists.
AGENTS_DIR_ENV_VAR = "UCLONE_AGENTS_DIR"

DEFAULT_AGENTS_ROOT = Path.home() / ".uclone" / "agents"

#: Distinguishes an agent's opaque identifier from its username at a glance, so a value
#: read out of a log or a record says which of the two it is.
AGENT_ID_PREFIX = "agt_"

#: The file in a clone's directory that holds its handle, display name and persona.
CLONE_FILE_NAME = "clone.yaml"

#: The file in a clone's directory that holds its rules: the asserted concepts, relations
#: and axioms a person gave it (clone-data-scopes §3.2, clone-knowledge-graph §3.1).
ONTOLOGY_FILE_NAME = "ontology.yaml"

#: What `mint_agent_id` produces and what a clone directory is named. Checked before a
#: path is built from an id, for the reason `_USERNAME_RE` is checked before a handle is.
_AGENT_ID_RE = re.compile(r"\Aagt_[0-9a-f]{32}\Z")

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


class CloneNotFoundError(AgentHomeError):
    """No clone has the handle asked for. The message is plain, for a person to read."""


class DuplicateHandleError(AgentHomeError):
    """More than one clone claims one handle, so it names none of them. Plain message."""


def is_agent_id(value: str) -> bool:
    """Whether `value` is an id `mint_agent_id` could have produced, and so a directory name."""
    return _AGENT_ID_RE.match(value) is not None


def mint_agent_id() -> str:
    """A new opaque clone id: `agt_` and 32 lowercase hex digits."""
    return f"{AGENT_ID_PREFIX}{uuid.uuid4().hex}"


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


#: The sidecar every clone write and the migration serialise on (design §3.8). A dot name,
#: so no listing or migration ever mistakes it for a clone.
_LOCK_FILE_NAME = ".lock"

#: Roots this thread already holds the lock on, with a depth. `flock` locks belong to an
#: open file, so a second `open` + `flock` of the same sidecar in one thread would wait on
#: itself; the migration takes the lock and then creates clones, which take it again.
_held_locks = threading.local()


@contextmanager
def clone_root_lock(root: Path | None = None) -> Generator[Path]:
    """Hold the exclusive lock over the clone root `root`, waiting for it if needed.

    Reentrant within one thread. Between processes -- a CLI and a server starting together
    -- the second waits for the first and then finds nothing left to do, because every
    step under the lock reads what is on disk before it writes. Yields the root.
    """
    base = root if root is not None else default_agents_root()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AgentHomeError(f"the clone root {base} could not be created: {exc}") from exc
    # Keyed on the root as it resolves once it exists, so every spelling of one directory
    # -- through `..` or a symlinked parent such as macOS's `/tmp` -- is one key. Keyed
    # before the `mkdir`, a first start spelled either way held the lock under one key and
    # its nested call asked under another, and `flock` waited on itself (#1949 review D3).
    key = str(base.resolve())
    depths: dict[str, int] = getattr(_held_locks, "depths", None) or {}
    _held_locks.depths = depths
    if depths.get(key, 0) > 0:
        depths[key] += 1
        try:
            yield base
        finally:
            depths[key] -= 1
        return
    try:
        handle = (base / _LOCK_FILE_NAME).open("a+b")
    except OSError as exc:
        raise AgentHomeError(f"the clone root {base} could not be locked: {exc}") from exc
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        depths[key] = 1
        try:
            yield base
        finally:
            depths[key] = 0
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def read_clone_file(directory: Path) -> dict[str, Any] | None:
    """The mapping in `directory`'s `clone.yaml`, or None when there is no such file.

    Raises:
        AgentHomeError: The file is there and is not a readable YAML mapping. Carried as
            this module's error so a listing reports it in place rather than failing.
    """
    path = directory / CLONE_FILE_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AgentHomeError(f"{path} could not be read: {exc}") from exc
    try:
        loaded: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise AgentHomeError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise AgentHomeError(f"{path} does not hold a mapping of fields.")
    return {str(key): value for key, value in cast(dict[object, object], loaded).items()}


def _handle_in(fields: Mapping[str, Any]) -> str | None:
    """The handle a clone file names, or None when it names no usable one."""
    handle = fields.get("handle")
    if not isinstance(handle, str):
        return None
    try:
        refuse_an_unusable_username(handle)
    except AgentHomeError:
        return None
    return handle


def clone_handles(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Every handle under `root`, each with the ids of the clones claiming it.

    More than one id under a handle is the duplicate `resolve_handle` refuses. Clone
    directories whose `clone.yaml` cannot be read are left out here; `list_agent_homes`
    reports them in place.
    """
    base = root if root is not None else default_agents_root()
    claims: dict[str, list[str]] = {}
    try:
        entries = sorted(base.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return {}
    for entry in entries:
        if not is_agent_id(entry.name) or not entry.is_dir():
            continue
        try:
            fields = read_clone_file(entry)
        except AgentHomeError:
            continue
        handle = _handle_in(fields) if fields is not None else None
        if handle is not None:
            claims.setdefault(handle, []).append(entry.name)
    return {handle: tuple(ids) for handle, ids in claims.items()}


def handle_of(agent_id: str, root: Path | None = None) -> str | None:
    """The handle the clone `agent_id` carries, or None when there is no such clone."""
    if not is_agent_id(agent_id):
        return None
    base = root if root is not None else default_agents_root()
    try:
        fields = read_clone_file(base / agent_id)
    except AgentHomeError:
        return None
    return _handle_in(fields) if fields is not None else None


def resolve_handle(handle: str, root: Path | None = None) -> str:
    """The id of the one clone called `handle`.

    Raises:
        AgentHomeError: `handle` is not a name a clone can have.
        CloneNotFoundError: No clone is called `handle`.
        DuplicateHandleError: More than one clone is, so it names none of them.
    """
    refuse_an_unusable_username(handle)  # before a lookup could call it merely absent
    claims = clone_handles(root)
    ids = claims.get(handle)
    if not ids:
        known = ", ".join(sorted(claims))
        raise CloneNotFoundError(
            f"There is no clone called '{handle}'."
            + (f" The clones are: {known}." if known else " There are no clones yet."),
            explanation="Check the name in the clone list.",
        )
    if len(ids) > 1:
        raise DuplicateHandleError(
            f"More than one clone is called '{handle}', so none of them was used. "
            "Rename one of them.",
        )
    return next(iter(ids))


def clone_id_of(ref: str, root: Path | None = None) -> str:
    """The id of the clone `ref` names: an id is its own, a handle is resolved.

    Everything keyed by clone -- a room seat, the memory and rules-engine maps -- is keyed
    by id (clone-data-scopes §4 step 3), and a caller still holding a handle goes through
    here, so one clone never gets two keys.

    Raises:
        CloneNotFoundError: `ref` is an id with no directory, or a handle no clone carries.
        DuplicateHandleError: More than one clone is called `ref`.
        AgentHomeError: `ref` cannot be a clone's name.
    """
    if is_agent_id(ref):
        base = root if root is not None else default_agents_root()
        if not (base / ref).is_dir():
            raise CloneNotFoundError(
                f"There is no clone with the id '{ref}'.",
                explanation="Check the clone list.",
            )
        return ref
    return resolve_handle(ref, root)


def seat_id_for(ref: str, root: Path | None = None) -> str:
    """The key `ref` is seated and remembered by: its clone's id, else `ref` itself.

    For a map or a seat that also takes an agent no clone backs (`ucx run --agent x` with
    no clone called `x` speaks as the host's fallback). A handle two clones claim is left
    as it is: it names neither, and the persona lookup reports that.
    """
    if is_agent_id(ref):
        return ref
    try:
        return resolve_handle(ref, root)
    except AgentHomeError:
        return ref


def peer_handles(peers: Iterable[str], root: Path | None = None) -> tuple[str, ...]:
    """`peers` as handles: an id is read back to its clone's handle, a handle stays.

    A clone stores its peers by id (design §3.3), so a rename rewrites only the renamed
    clone's own file. Everything that addresses a peer still speaks in handles, which is
    what this translates to. An id naming no clone is left out: there is nobody to call.
    """
    base = root if root is not None else default_agents_root()
    resolved: list[str] = []
    for peer in peers:
        handle = handle_of(peer, base) if is_agent_id(peer) else peer
        if handle is not None and handle not in resolved:
            resolved.append(handle)
    return tuple(resolved)


def _refuse_a_clone_file_without(handle: str, clone_yaml_text: str) -> None:
    """Refuse clone text whose `handle` is not `handle`: the caller composed it wrongly."""
    try:
        loaded: object = yaml.safe_load(clone_yaml_text)
    except yaml.YAMLError as exc:
        raise AgentHomeError(f"the clone file for {handle!r} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict) or cast(dict[object, object], loaded).get("handle") != handle:
        raise AgentHomeError(f"the clone file for {handle!r} does not carry that handle.")


def _refuse_an_unusable_handle(handle: str) -> None:
    refuse_an_unusable_username(handle)
    if is_agent_id(handle):
        raise AgentHomeError(
            f"{handle!r} has the shape of a clone id, so it cannot also be a handle.",
            explanation="That name has the shape of a clone's id. Choose another name.",
        )


def _write_durably(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_clone(
    handle: str,
    clone_yaml_text: str,
    *,
    files: Mapping[str, bytes] | None = None,
    root: Path | None = None,
    agent_id: str | None = None,
) -> AgentHome:
    """Make a clone in one step: its directory, `id`, `clone.yaml` and `files` together.

    Everything is written into a temporary directory beside the others and then renamed
    to the new id, so a reader sees either no clone or a whole one (design §3.4). Taken
    under `clone_root_lock`, so two creates of one handle cannot both succeed.

    Raises:
        AgentHomeError: `handle` is not a usable handle, or the directory could not be
            written.
        DuplicateHandleError: A clone of that handle already exists.
    """
    _refuse_an_unusable_handle(handle)
    _refuse_a_clone_file_without(handle, clone_yaml_text)
    new_id = agent_id if agent_id is not None else mint_agent_id()
    if not is_agent_id(new_id):
        raise AgentHomeError(f"{new_id!r} is not a clone id.")
    for name in files or {}:
        if name in {"id", CLONE_FILE_NAME} or Path(name).name != name or name.startswith("."):
            raise AgentHomeError(f"{name!r} cannot be written into a clone's directory.")
    with clone_root_lock(root) as base:
        if clone_handles(base).get(handle):
            raise DuplicateHandleError(
                f"A clone called '{handle}' already exists. Choose another name."
            )
        target = base / new_id
        if target.exists():
            raise AgentHomeError(f"{target} already exists, so no clone was created there.")
        try:
            staged = Path(tempfile.mkdtemp(dir=base, prefix=".new-"))
        except OSError as exc:
            raise AgentHomeError(f"a clone could not be created under {base}: {exc}") from exc
        try:
            _write_durably(staged / "id", f"{new_id}\n".encode())
            _write_durably(staged / CLONE_FILE_NAME, clone_yaml_text.encode("utf-8"))
            for name, data in (files or {}).items():
                _write_durably(staged / name, data)
            _fsync_dir(staged)
            os.rename(staged, target)
            _fsync_dir(base)
        except OSError as exc:
            raise AgentHomeError(f"a clone could not be created under {base}: {exc}") from exc
        finally:
            if staged.exists():
                for leftover in staged.iterdir():
                    leftover.unlink()
                staged.rmdir()
    return AgentHome(username=handle, path=target)


def replace_clone_file(agent_id: str, clone_yaml_text: str, root: Path | None = None) -> Path:
    """Replace the `clone.yaml` of the clone `agent_id` in one step, and return its path.

    The new text may carry a new handle, which is how a clone is renamed: its directory,
    and everything recorded under its id, stay where they are.

    Raises:
        AgentHomeError: There is no clone `agent_id`, the text names no usable handle, or
            the file could not be written.
        DuplicateHandleError: Another clone already has the new handle.
    """
    with clone_root_lock(root) as base:
        home = AgentHome.for_clone(agent_id, base)
        if not home.path.is_dir():
            raise CloneNotFoundError("That clone is no longer here.")
        try:
            loaded: object = yaml.safe_load(clone_yaml_text)
        except yaml.YAMLError as exc:
            raise AgentHomeError(f"the clone file for {agent_id} is not valid YAML: {exc}") from exc
        fields = (
            {str(k): v for k, v in cast(dict[object, object], loaded).items()}
            if isinstance(loaded, dict)
            else {}
        )
        raw_handle = fields.get("handle")
        if not isinstance(raw_handle, str):
            raise AgentHomeError(f"the clone file for {agent_id} carries no handle.")
        _refuse_an_unusable_handle(raw_handle)
        others = [other for other in clone_handles(base).get(raw_handle, ()) if other != agent_id]
        if others:
            raise DuplicateHandleError(
                f"A clone called '{raw_handle}' already exists. Choose another name."
            )
        descriptor, staged = tempfile.mkstemp(dir=home.path, prefix=".clone-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(clone_yaml_text.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged, home.clone_path)
        except OSError as exc:
            raise AgentHomeError(f"{home.clone_path} could not be written: {exc}") from exc
        finally:
            if os.path.exists(staged):
                os.unlink(staged)
        return home.clone_path


@dataclass(frozen=True)
class AgentHome:
    """The directory holding one agent's identity and durable state."""

    username: str
    path: Path

    @classmethod
    def for_clone(cls, agent_id: str, root: Path | None = None) -> AgentHome:
        """Locate the directory of the clone `agent_id`. Performs no I/O.

        Raises:
            AgentHomeError: `agent_id` is not a clone id, so no path is built from it.
        """
        if not is_agent_id(agent_id):
            raise AgentHomeError(
                f"{agent_id!r} is not a clone id, so it names no clone's directory."
            )
        base = root if root is not None else default_agents_root()
        return cls(username=agent_id, path=base / agent_id)

    @classmethod
    def for_handle(cls, handle: str, root: Path | None = None) -> AgentHome:
        """Locate the directory of the one clone called `handle`, creating nothing.

        The door every caller holding a name goes through (design §3.3). A name no clone
        carries is refused, never given a home: there is no lazy home any more (§3.4).

        Raises:
            AgentHomeError: `handle` is not a name a clone can have.
            CloneNotFoundError: No clone is called `handle`.
            DuplicateHandleError: More than one clone is.
        """
        base = root if root is not None else default_agents_root()
        agent_id = resolve_handle(handle, base)
        return cls(username=handle, path=base / agent_id)

    @classmethod
    def for_ref(cls, ref: str, root: Path | None = None) -> AgentHome:
        """Locate the directory of the clone `ref` names, by id or by handle (`clone_id_of`).

        Raises:
            CloneNotFoundError: No clone is `ref`.
            DuplicateHandleError: More than one clone is called `ref`.
            AgentHomeError: `ref` cannot be a clone's name.
        """
        base = root if root is not None else default_agents_root()
        agent_id = clone_id_of(ref, base)
        return cls(username=agent_id, path=base / agent_id)

    @classmethod
    def for_legacy_name(cls, username: str, root: Path | None = None) -> AgentHome:
        """Locate `<root>/<username>`, the layout before 2026-09-27. Performs no I/O.

        Only the migration reads this layout (`uclone_x.agent.clone_store`); every other
        caller holds a handle and goes through `for_handle`.
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
    def clone_path(self) -> Path:
        """Path of this clone's `clone.yaml`: handle, display name and persona."""
        return self.path / CLONE_FILE_NAME

    @property
    def id_path(self) -> Path:
        """Path of the file holding this agent's opaque identifier."""
        return self.path / "id"

    @property
    def ontology_path(self) -> Path:
        """Path of this clone's rules file, which its one rules engine loads."""
        return self.path / ONTOLOGY_FILE_NAME

    @property
    def knowledge_path(self) -> Path:
        """Path of this agent's knowledge file: its cross-session facts, as a graph."""
        return self.path / "knowledge.sqlite3"

    @property
    def memory_path(self) -> Path:
        """Path of the memory document an earlier build wrote, which the knowledge file replaced.

        Read only by the one-time import (`memory.legacy_import`); it is set aside afterwards.
        """
        return self.path / "memory.json"

    def recorded_agent_id(self) -> str | None:
        """This agent's identifier if one has been written, and None before that.

        Public because enumerating what is installed has to read an id *without* bringing
        the agent into being: `agent_id()` mints and writes, which would turn a listing
        into an installer. Same rule as `for_clone` -- asking must not create.

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
    #: A clone directory's `clone.yaml` is missing, unreadable, or names no usable handle.
    UNREADABLE_CLONE_FILE = "unreadable_clone_file"
    #: Another clone directory claims the same handle, so the handle names neither.
    DUPLICATE_HANDLE = "duplicate_handle"


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

    Nothing else enumerates them. The seated clones a head reports (`seated_agent_ids`
    on its room stack) are *live instances*, which is empty on an install where no
    conversation has started -- so a surface built on them shows nothing on exactly the
    first screen a new user sees.

    This function never raises for a fault it can describe. A root that cannot be read
    and a root holding nothing both produce an empty `homes`, and a damaged single home
    would, if its refusal escaped, empty the whole list; each is instead reported as the
    state it is (P6). The only exception left is a caller passing a root it cannot even
    name, which is a programming error rather than an installation one.

    Performs no writes: asking what is installed must not install anything, for the same
    reason `AgentHome.for_clone` performs no I/O.
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

    claims = clone_handles(agents_root)
    described = [
        _describe_agent_home(entry, claims)
        for entry in entries
        # A leading dot cannot be a username under `_USERNAME_RE`, so such an entry never
        # competes for one. Skipping it rather than reporting it as damage keeps
        # `.DS_Store`, editor droppings, the lock and the migration's records out of a
        # list of agents.
        if not entry.name.startswith(".")
    ]
    # By handle, which is what a reader recognises; the id directory names sort randomly.
    homes = tuple(sorted(described, key=lambda home: (home.username, home.path.name)))
    return AgentHomeListing(
        root=agents_root,
        root_state=AgentHomeRootState.READABLE,
        cause="",
        homes=homes,
    )


def _describe_agent_home(path: Path, claims: Mapping[str, tuple[str, ...]]) -> AgentHomeEntry:
    """Report one directory under the agents root, never raising for what it finds.

    A directory named by an id reports the handle its `clone.yaml` carries. One named by
    anything else is the layout before 2026-09-27, which the migration renames; it is
    described as it always was, so a listing taken before the migration still reads.
    """
    if is_agent_id(path.name) and path.is_dir():
        return _describe_clone_directory(path, claims)
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

    return _installed_entry(name, path, home.id_path, recorded)


def _installed_entry(
    username: str, path: Path, id_path: Path, agent_id: str | None
) -> AgentHomeEntry:
    """A home that reads: installed, with its id when it has one, and no fault."""
    return AgentHomeEntry(
        username=username,
        path=path,
        id_path=id_path,
        state=AgentHomeState.INSTALLED,
        agent_id=agent_id,
        fault=None,
        cause="",
    )


def _describe_clone_directory(path: Path, claims: Mapping[str, tuple[str, ...]]) -> AgentHomeEntry:
    """Report one id-named clone directory: its handle, or why it has none."""
    home = AgentHome(username=path.name, path=path)
    try:
        fields = read_clone_file(path)
    except AgentHomeError as broken:
        fields, cause = None, str(broken)
    else:
        cause = "" if fields is not None else f"{home.clone_path} is missing."
    handle = _handle_in(fields) if fields is not None else None
    if handle is None:
        return AgentHomeEntry(
            username=path.name,
            path=path,
            id_path=home.id_path,
            state=AgentHomeState.UNREADABLE,
            agent_id=path.name,
            fault=AgentHomeFault.UNREADABLE_CLONE_FILE,
            cause=cause or f"{home.clone_path} names no usable handle.",
        )
    if len(claims.get(handle, ())) > 1:
        return AgentHomeEntry(
            username=handle,
            path=path,
            id_path=home.id_path,
            state=AgentHomeState.UNREADABLE,
            agent_id=path.name,
            fault=AgentHomeFault.DUPLICATE_HANDLE,
            cause="",
        )
    return _installed_entry(handle, path, home.id_path, path.name)
