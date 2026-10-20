"""Clones as directories: the persona store over `<agents root>/<agt_id>/clone.yaml`.

A persona and a clone are one thing (owner ruling, 2026-09-27), so a clone's definition
lives in its own directory beside its memory and picture, not in whichever directory the
server was launched from (clone-data-scopes §3.2). This module is the
store `PersonaRegistry` reads and writes through, and the start-up pass that brings an
install to that layout:

1. **Migration** (§3.8 step 1). `agents/<name>/` directories from before 2026-09-27 are
   given a `clone.yaml` in place and then renamed to their id.
2. **Import** (§3.8 step 2). `<workspace>/.uclone/personas/*.yaml` become clones, pictures
   included. A handle that already has a clone keeps it; a differing file is reported
   once and recorded in that clone's directory, so later starts stay quiet.
3. **Install** (§3.4). Each package persona not yet recorded in `agents/.installed` is
   installed, or only recorded when a clone of that handle already exists.
4. **Peers** (§3.8 step 2b). Every `a2a_peers` handle is rewritten to an id; a handle
   naming no clone yet is moved to `pending_peers` and reported, and becomes an id, in the
   place it held, at the
   first start after a clone of that name is installed (#2245). A stored peer is never a
   handle, and a pending one is never offered as a peer.

Every move goes to `agents/.migration-<date>.log`. Nothing is deleted. The whole pass runs
under the clone root's lock, so two processes starting together cannot both run it; the
second waits and then finds nothing to do.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import hashlib
import logging
import os
import tempfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from uclone_x.agent.models import PersonaDefinition
from uclone_x.agent.persona_store import (
    DEFAULT_PERSONA_NAME,
    DEFAULT_WORKSPACE_PERSONAS_SUBDIR,
    PersonaDraft,
    PersonaLoadError,
    PersonaNotFound,
    PersonaWriteConflict,
    PersonaWriteRefused,
    display_name_from_mapping,
    dump_persona_yaml,
    persona_file_contents,
    persona_from_mapping,
    read_persona_mapping,
    refuse_an_unwritable_display_name,
)
from uclone_x.core.agent_home import (
    CLONE_FILE_NAME,
    AgentHome,
    AgentHomeError,
    DuplicateHandleError,
    clone_handles,
    clone_root_lock,
    create_clone,
    default_agents_root,
    is_agent_id,
    peer_handles,
    read_clone_file,
    refuse_an_unusable_username,
    replace_clone_file,
)
from uclone_x.core.models import persona_model_problem

logger = logging.getLogger(__name__)

__all__ = [
    "AVATAR_SUFFIXES",
    "INSTALLED_FILE_NAME",
    "CloneDirectoryPersonaStore",
    "CloneRecord",
    "CloneStoreReport",
    "UnreadableClone",
    "append_report",
    "ensure_clone_store",
    "import_workspace_personas",
    "install_builtin_personas",
    "migrate_legacy_homes",
    "peers_with_pending",
    "read_imports",
    "record_import",
    "translate_stored_peers",
    "with_pending_peers",
]

#: Records which package personas this install has installed, one handle per line.
#: Recording, not presence, is what stops a builtin the user deleted from coming back.
INSTALLED_FILE_NAME = ".installed"

#: In a clone's directory: which workspace files were imported into it or conflicted with
#: it, by path and digest, so a conflict is reported once (§3.8 step 2).
IMPORTS_FILE_NAME = "imports.yaml"

#: The picture formats a clone's directory may hold, as `avatar<suffix>`.
AVATAR_SUFFIXES: tuple[str, ...] = (".png", ".webp", ".jpg", ".jpeg", ".gif")

#: In `clone.yaml`: the peers a clone names by a handle no clone carries yet -- a shipped
#: default naming a builtin an extension not yet installed ships (#2245). A mapping of
#: each handle to the place it held among the clone's peers, pending ones counted, so it
#: goes back there (#2256). Not a persona field, so nothing offers them; the start-up pass
#: stores each by id once the builtin of that handle is installed.
PENDING_PEERS_KEY = "pending_peers"

#: A clone's own keys in `clone.yaml`, written before the persona fields.
_CLONE_KEYS: tuple[str, ...] = ("handle", "display_name", "template", PENDING_PEERS_KEY)

#: The persona fields a home with no persona of its own takes when the package's `clone`
#: builtin cannot be read -- a registry pointed at an empty package directory.
_FALLBACK_PERSONA: dict[str, Any] = {
    "role": "Personal AI Clone",
    "description": "",
    "system_prompt": "You are a personal AI clone.",
    "append_default_prompt": True,
}


@dataclass
class CloneStoreReport:
    """What one start-up pass did, one plain line per move, for the migration log."""

    lines: list[str] = field(default_factory=list[str])

    def add(self, line: str) -> None:
        self.lines.append(line)
        logger.info("clone store: %s", line)

    def warn(self, line: str) -> None:
        """`add`, for a move a person has to follow up on, so the server log says so too."""
        self.lines.append(line)
        logger.warning("clone store: %s", line)


@dataclass(frozen=True)
class CloneRecord:
    """One clone directory as the store read it."""

    agent_id: str
    handle: str
    #: Per locale; empty when the clone has none, and a reader falls back to `handle`.
    display_name: dict[str, str]
    #: The package persona this clone was installed from, or None.
    template: str | None
    path: Path

    @property
    def clone_path(self) -> Path:
        return self.path / CLONE_FILE_NAME


@dataclass(frozen=True)
class UnreadableClone:
    """A clone directory whose persona the store could not read (#2220).

    It is not loaded: `GET /api/clones` lists it as unreadable with `reason`, and a turn
    that names it is refused with it. The loader's own words, which name the file and the
    field, go to the log only.
    """

    agent_id: str
    handle: str
    #: Plain words for a person: what is wrong and what to do, with no path or field name.
    #: The English fallback: a head words `reason_code` in the reader's language (#2241).
    reason: str
    #: Which reason: `bare_model` or `unreadable_file`.
    reason_code: str
    #: The name a person gave it, when that much of the file can be read (a bare model
    #: leaves it readable); empty otherwise, and a head shows the handle (#2275).
    display_name: dict[str, str] = field(default_factory=dict[str, str])


#: Why a clone whose model names no connection cannot be loaded.
_BARE_MODEL_REASON = (
    "This clone's model does not say which connection it is on, so the clone cannot be "
    "loaded. In the clone's settings file, write the model the way Settings shows it, "
    "starting with its connection, or remove the model so the clone uses the default one."
)

#: Why any other clone the loader refuses cannot be loaded.
_UNREADABLE_REASON = (
    "This clone's settings file holds a value this version cannot read, so the clone "
    "cannot be loaded. Restore the file from a backup, or remove the clone and set it up "
    "again."
)

#: The model fields 0.3.1 saved in a persona, each a bare model id, with the words a
#: person reads for each. 0.3.1 had no picture model.
_EARLIER_MODEL_FIELDS: tuple[tuple[str, str], ...] = (
    ("model_name", "model"),
    ("fast_model", "fast model"),
)


# --- reading and writing one clone file -------------------------------------------------


def _clone_text(
    handle: str,
    persona_fields: dict[str, Any],
    *,
    display_name: dict[str, str] | None,
    template: str | None,
    pending_peers: Mapping[str, int] | None = None,
) -> str:
    """The text of a `clone.yaml`: the clone's own keys, then its persona fields."""
    contents: dict[str, Any] = {"handle": handle}
    if display_name:
        contents["display_name"] = dict(display_name)
    if template:
        contents["template"] = template
    if pending_peers:
        contents[PENDING_PEERS_KEY] = dict(pending_peers)
    for key, value in persona_fields.items():
        if key not in (*_CLONE_KEYS, "name", "id"):
            contents[key] = value
    return dump_persona_yaml(contents)


def _pending_peers(fields: dict[str, Any]) -> dict[str, int]:
    """The handles a clone file keeps pending, each with its place (`PENDING_PEERS_KEY`)."""
    raw: object = fields.get(PENDING_PEERS_KEY)
    if not isinstance(raw, dict):
        return {}
    return {
        str(peer): place
        for peer, place in cast(dict[object, object], raw).items()
        if isinstance(place, int) and not isinstance(place, bool) and place >= 0
    }


def with_pending_peers(peers: Sequence[str], pending: Mapping[str, int]) -> list[str]:
    """`peers` with each pending handle put back in the place it held (#2256).

    In order of place, so each place counts the ones before it, as when it was recorded.
    A place past the end -- peers removed since -- is the end.
    """
    merged = list(peers)
    for peer, place in sorted(pending.items(), key=lambda item: item[1]):
        merged.insert(min(place, len(merged)), peer)
    return merged


def peers_with_pending(record: CloneRecord, peers: Sequence[str]) -> list[str]:
    """`peers` -- the clone at `record`'s, as handles -- with its pending ones in place.

    What the clone was given, which a comparison with the persona it shipped as needs: a
    peer that waits for its builtin was not taken away (#2256).
    """
    try:
        fields = read_clone_file(record.path) or {}
    except AgentHomeError:
        fields = {}
    return with_pending_peers(peers, _pending_peers(fields))


def _builtin_clone(handle: str, claims: Mapping[str, Sequence[str]], base: Path) -> str | None:
    """The id of the clone installed from the builtin `handle`, if that clone has it.

    Not any clone of that handle: a `writer` a person made is not the Writer a shipped
    peer list names, and taking it would give a clone a peer nobody chose (#2256).
    """
    owners = claims.get(handle, ())
    if len(owners) != 1:
        return None
    try:
        fields = read_clone_file(base / owners[0])
    except AgentHomeError:
        return None
    return owners[0] if fields is not None and fields.get("template") == handle else None


def _persona_fields(mapping: dict[str, Any]) -> dict[str, Any]:
    """`mapping` without the keys that say who the clone is rather than how it behaves."""
    return {key: value for key, value in mapping.items() if key not in (*_CLONE_KEYS, "name", "id")}


def _raw_models(fields: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """The model, fast model and picture model a persona mapping names, as written."""
    raw: object = fields.get("llm_config") or fields.get("llm")
    llm = cast(dict[object, object], raw) if isinstance(raw, dict) else {}

    def text(key: str) -> str | None:
        value = llm.get(key)
        return value if isinstance(value, str) else None

    return text("model_name"), text("fast_model"), text("image_model")


def _unreadable_reason(fields: dict[str, Any]) -> tuple[str, str]:
    """Why a clone file the loader refused cannot be loaded: plain words, and their code."""
    if persona_model_problem(*_raw_models(fields)) is not None:
        return _BARE_MODEL_REASON, "bare_model"
    return _UNREADABLE_REASON, "unreadable_file"


def _readable_display_name(fields: dict[str, Any], source: Path) -> dict[str, str]:
    """`fields`' display name if it can be read on its own, else empty (#2275)."""
    try:
        return display_name_from_mapping(fields, source)
    except PersonaLoadError:
        return {}


def _load_problem(fields: dict[str, Any], handle: str, folder: Path) -> str | None:
    """Why the clone at `folder` cannot be loaded, in the loader's words; None if it can.

    The store's own check, asked by a start-up pass that is about to rewrite the file, so it
    does not rewrite one it cannot parse (#2241). Tools are not checked here: a pass that
    runs before the tool registry exists cannot, so a clone unreadable only for an unknown
    tool still has its peers stored by id.
    """
    source = folder / CLONE_FILE_NAME
    try:
        display_name_from_mapping(fields, source)
        if _persona_fields(fields):
            persona_from_mapping(_clone_persona_mapping(fields, handle), source)
    except (ValidationError, PersonaLoadError) as exc:
        return str(exc)
    return None


#: A model taken off, or kept, by `_earlier_models`: the words for its field, and its value.
_Model = tuple[str, str]


def _earlier_models(
    mapping: dict[str, Any], *, keep_refs: bool
) -> tuple[dict[str, Any], list[_Model], list[_Model]]:
    """`mapping` without the models an earlier version saved: (mapping, taken off, kept).

    0.3.1 stored a model as a bare id, which says nothing about the connection it is on.
    An id with a `/` in it (`Qwen/Qwen3-8B`, as vLLM names models) reads as the ref
    `<connection>/<model>` and points the clone at a connection that does not exist; one
    without fails the ref rule (#2220). Either way the clone follows the default model
    once it is taken off, until its model is chosen again.

    `keep_refs=False` takes every model off. Step 1 asks for it: it reads the workspace file
    only beside an old `agents/<name>/` home, which exists only on an install that has never
    started a build with the clone store (that start renames it, #1949). Every build that
    reads refs (#2158) came after, so the file was last used by one that could not, and a
    `/` in its model is a 0.3.1 id's, not a connection's (clone-data-scopes §3.8 2a, #2241).
    `keep_refs=True` takes off only what is not a ref, and lists the refs it kept: step 2
    reads any directory that becomes a workspace, whose file may have been written for
    today's build, where `ollama/custom-model` is exactly the right spelling.
    """
    result = dict(mapping)
    dropped: list[_Model] = []
    kept: list[_Model] = []
    for key in ("llm_config", "llm"):
        raw: object = mapping.get(key)
        if not isinstance(raw, dict):
            continue
        llm = {str(k): v for k, v in cast(dict[object, object], raw).items()}
        for field_name, label in _EARLIER_MODEL_FIELDS:
            value = llm.get(field_name)
            if not isinstance(value, str) or not value.strip():
                continue
            if keep_refs and persona_model_problem(value, None, None) is None:
                kept.append((label, value.strip()))
                continue
            del llm[field_name]
            dropped.append((label, value.strip()))
        if llm:
            result[key] = llm
        else:
            del result[key]
    return result, dropped, kept


def _report_earlier_models(
    report: CloneStoreReport,
    handle: str,
    dropped: Sequence[_Model],
    kept: Sequence[_Model] = (),
) -> None:
    """Say, in the migration log and the server log, what became of a clone's saved models.

    A model taken off is said to be gone. A ref kept from a workspace file is said as it
    is read, so a 0.3.1 id that happens to hold a `/` is not carried over in silence.
    """
    for label, value in dropped:
        report.warn(
            f"clone '{handle}': its {label} '{value}' was saved by an earlier version, which "
            "did not record the connection it is on, so it was not kept and the clone uses "
            "the default. Open the clone in Settings and choose its model again."
        )
    for label, value in kept:
        connection, _, model = value.partition("/")
        report.warn(
            f"clone '{handle}': its {label} '{value}' is read as the model '{model.strip()}' "
            f"on the connection '{connection.strip()}'. If it was saved by an earlier "
            f"version, or '{connection.strip()}' is not one of your connections, open the "
            "clone in Settings and choose its model again."
        )


def _write_text_atomically(path: Path, text: str) -> None:
    descriptor, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)


def _avatar_files(folder: Path, stem: str) -> dict[str, bytes]:
    """The chosen and previous pictures `<stem>.<ext>` / `<stem>.prev.<ext>` in `folder`,
    keyed by the names they take in a clone directory."""
    found: dict[str, bytes] = {}
    for kind, source_stem in (("avatar", stem), ("avatar.prev", f"{stem}.prev")):
        for suffix in AVATAR_SUFFIXES:
            candidate = folder / f"{source_stem}{suffix}"
            if candidate.is_file():
                found[f"{kind}{suffix}"] = candidate.read_bytes()
                break
    return found


def _package_mappings(
    builtin_dir: Path | None, extension_dirs: Sequence[Path] = ()
) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Every package persona by handle, with its file and raw mapping.

    `extension_dirs` are the builtin folders installed extensions ship (#2208), read after
    the package's own. One handle shipped twice is refused: a builtin installed from
    whichever folder happened to be read last would be a clone nobody chose.
    """
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for folder in (builtin_dir, *extension_dirs):
        if folder is None or not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.yaml")) + sorted(folder.glob("*.yml")):
            mapping = read_persona_mapping(path.read_text(encoding="utf-8"), path)
            handle = str(mapping.get("name") or mapping.get("id") or path.stem)
            if handle in found:
                raise PersonaLoadError(
                    f"{path}: the builtin persona {handle!r} is already shipped by "
                    f"{found[handle][0]}; two builtins cannot share a name."
                )
            found[handle] = (path, mapping)
    return found


def _clone_persona_mapping(fields: dict[str, Any], handle: str) -> dict[str, Any]:
    """The persona a clone's `clone.yaml` holds, as the mapping `persona_from_mapping` reads.

    The clone's label rides on its persona, as a package file's does (#1947), so the store's
    reading and the import's comparison with a workspace file see the same persona.
    """
    data = _persona_fields(fields)
    data["name"] = handle
    if "display_name" in fields:
        data["display_name"] = fields["display_name"]
    return data


def _workspace_mappings(
    workspace_root: Path | None, report: CloneStoreReport | None = None
) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Every readable persona file in `<workspace>/.uclone/personas/`, by handle.

    A file that cannot be read as a persona is left out and, given `report`, named in it
    with the reason, so a mistyped key is not a clone that quietly never appears (P6).
    """
    if workspace_root is None:
        return {}
    folder = workspace_root / DEFAULT_WORKSPACE_PERSONAS_SUBDIR
    if not folder.is_dir():
        return {}
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(folder.glob("*.yaml")) + sorted(folder.glob("*.yml")):
        try:
            mapping = read_persona_mapping(path.read_text(encoding="utf-8"), path)
            # Read as the clone it becomes: without a model that is not a ref, which the
            # import does not keep, so a 0.3.1 bare id does not leave the persona out.
            stripped, _, _ = _earlier_models(mapping, keep_refs=True)
            persona = persona_from_mapping(stripped, path)
        except (OSError, PersonaLoadError, ValueError) as exc:
            logger.warning("clone store: %s cannot be imported as a clone: %s", path, exc)
            if report is not None:
                report.add(f"{path} was not imported: {exc}")
            continue
        found.setdefault(persona.name, (path, mapping))
    return found


# --- the store --------------------------------------------------------------------------


class CloneDirectoryPersonaStore:
    """The persona store over every clone directory under one agents root.

    A persona's name is its clone's handle. Two clones claiming one handle serve neither:
    they are left out and logged, and `GET /api/clones` shows both as damaged.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        tool_names: Collection[str] | None = None,
    ) -> None:
        self._root = root
        self._tool_names: frozenset[str] | None = (
            frozenset(tool_names) if tool_names is not None else None
        )
        self._personas: dict[str, PersonaDefinition] = {}
        self._records: dict[str, CloneRecord] = {}
        self._unreadable: dict[str, UnreadableClone] = {}
        self.reload()

    @property
    def root(self) -> Path:
        """The agents root, read when asked so `UCLONE_AGENTS_DIR` is honoured."""
        return self._root if self._root is not None else default_agents_root()

    @property
    def directory(self) -> Path:
        return self.root

    @property
    def read_only(self) -> bool:
        return False

    def reload(self) -> None:
        self._personas.clear()
        self._records.clear()
        self._unreadable.clear()
        root = self.root
        try:
            entries = sorted(root.iterdir(), key=lambda entry: entry.name)
        except OSError:
            return
        claims = clone_handles(root)
        for entry in entries:
            if not is_agent_id(entry.name) or not entry.is_dir():
                continue
            try:
                fields = read_clone_file(entry)
            except AgentHomeError as damage:
                logger.warning("clone store: %s is skipped: %s", entry, damage)
                continue
            if fields is None:
                continue
            handle = fields.get("handle")
            if not isinstance(handle, str) or handle not in claims:
                logger.warning("clone store: %s names no usable handle and is skipped", entry)
                continue
            if len(claims[handle]) > 1:
                logger.warning(
                    "clone store: handle %r is claimed by %s, so it names none of them",
                    handle,
                    ", ".join(claims[handle]),
                )
                continue
            source = entry / CLONE_FILE_NAME
            try:
                display_name = display_name_from_mapping(fields, source)
                # A clone whose file says only who it is has no persona of its own: it is
                # a clone (its handle resolves, it has memory) and it speaks as its host's
                # fallback, as an agent with no persona always has (`build_clone`).
                if _persona_fields(fields):
                    self._personas[handle] = self._parse(fields, handle, source)
            except (PersonaLoadError, ValidationError) as exc:
                # This clone's fault, not the store's: it is listed as unreadable and the
                # others load (#2220).
                logger.warning(
                    "clone store: clone '%s' (%s) cannot be loaded and is listed as unreadable: %s",
                    handle,
                    source,
                    exc,
                )
                reason, code = _unreadable_reason(fields)
                self._unreadable[handle] = UnreadableClone(
                    agent_id=entry.name,
                    handle=handle,
                    reason=reason,
                    reason_code=code,
                    display_name=_readable_display_name(fields, source),
                )
                continue
            template = fields.get("template")
            self._records[handle] = CloneRecord(
                agent_id=entry.name,
                handle=handle,
                display_name=display_name,
                template=template if isinstance(template, str) else None,
                path=entry,
            )

    def _parse(self, fields: dict[str, Any], handle: str, source: Path) -> PersonaDefinition:
        data = _clone_persona_mapping(fields, handle)
        try:
            persona = persona_from_mapping(data, source)
        except ValidationError as exc:
            # The loader's own refusals only: any other exception is a defect, and it
            # fails loudly rather than reading as one unreadable clone (#2241).
            raise PersonaLoadError(f"{source}: {exc}") from exc
        self._validate_tools(persona, source=source)
        return persona

    def _validate_tools(self, persona: PersonaDefinition, *, source: Path) -> None:
        if self._tool_names is None or not persona.allowed_tools:
            return
        unknown = [name for name in persona.allowed_tools if name not in self._tool_names]
        if not unknown:
            return
        known = sorted(self._tool_names)
        details = "; ".join(
            f"{name!r}"
            + (
                f" (did you mean {', '.join(repr(m) for m in matches)}?)"
                if (matches := difflib.get_close_matches(name, known, n=2, cutoff=0.5))
                else ""
            )
            for name in unknown
        )
        raise PersonaLoadError(
            f"{source}: persona {persona.name!r} declares tool(s) that are not registered: "
            f"{details}. Registered: {', '.join(known)}"
        )

    def get_persona(self, name: str) -> PersonaDefinition | None:
        return self._personas.get(self.handle_for(name))

    def unreadable_clone(self, ref: str) -> UnreadableClone | None:
        """The clone `ref` (a handle or a clone id) names, when its file could not be read."""
        found = self._unreadable.get(ref)
        if found is None and is_agent_id(ref):
            found = next((c for c in self._unreadable.values() if c.agent_id == ref), None)
        return found

    def unreadable_clones(self) -> tuple[UnreadableClone, ...]:
        """Every clone directory whose file could not be read, by handle."""
        return tuple(self._unreadable[handle] for handle in sorted(self._unreadable))

    def has_persona(self, name: str) -> bool:
        return self.handle_for(name) in self._personas

    def list_personas(self) -> Sequence[PersonaDefinition]:
        return sorted(
            self._personas.values(),
            key=lambda p: (0 if p.name == DEFAULT_PERSONA_NAME else 1, p.name),
        )

    def source_of(self, name: str) -> Path | None:
        record = self.record_of(name)
        return record.clone_path if record is not None else None

    def record_of(self, name: str) -> CloneRecord | None:
        """The clone directory behind the persona `name` (a handle or a clone id), or None."""
        record = self._records.get(name)
        if record is None and is_agent_id(name):
            record = next((r for r in self._records.values() if r.agent_id == name), None)
        return record

    def handle_for(self, ref: str) -> str:
        """`ref` as the handle a persona is named by: an id loaded here is read back to it."""
        if is_agent_id(ref):
            record = self.record_of(ref)
            if record is not None:
                return record.handle
        return ref

    def save_persona(
        self, draft: PersonaDraft, *, create: bool, allow_override: bool = False
    ) -> PersonaDefinition:
        """Create a clone from `draft` in one step, or rewrite an existing one's file.

        An edit keeps the clone's display name and its "Learn from conversations" switch
        unless the draft carries them, and always keeps its template. Peers are written as
        ids.
        """
        del allow_override  # a clone is only ever edited in place
        try:
            refuse_an_unusable_username(draft.name)
        except AgentHomeError as exc:
            raise PersonaWriteRefused(
                f"{draft.name!r} cannot be a persona name: {exc} The name is the clone's "
                f"handle, so choose one that fits that rule."
            ) from exc
        refuse_an_unwritable_display_name(draft)
        record = self._records.get(draft.name)
        if create and (record is not None or clone_handles(self.root).get(draft.name)):
            raise PersonaWriteConflict(
                f"a persona named {draft.name!r} already exists. Edit it, or choose another name."
            )
        if not create and record is None:
            raise PersonaNotFound(f"no persona named {draft.name!r} is loaded.")

        fields = persona_file_contents(draft)
        fields["a2a_peers"] = self._peer_ids(draft.a2a_peers)
        if not fields["a2a_peers"]:
            del fields["a2a_peers"]
        # An edit that leaves the switch out keeps the clone's own setting (#1840).
        current = self._personas.get(draft.name) if record is not None else None
        learns = (
            draft.learn_from_conversations
            if "learn_from_conversations" in draft.model_fields_set or current is None
            else current.learn_from_conversations
        )
        fields.pop("learn_from_conversations", None)
        if not learns:
            fields["learn_from_conversations"] = False
        display_name = (
            draft.display_name
            if "display_name" in draft.model_fields_set or record is None
            else record.display_name
        )
        template = record.template if record is not None else None
        # A pending peer was never shown, so an edit cannot have meant to drop it (#2245).
        pending = _pending_peers(read_clone_file(record.path) or {}) if record is not None else {}
        text = _clone_text(
            draft.name,
            fields,
            display_name=display_name,
            template=template,
            pending_peers=pending,
        )
        label = record.clone_path if record is not None else self.root / draft.name
        try:
            checked = read_persona_mapping(text, label)
            display_name_from_mapping(checked, label)
            self._parse(checked, draft.name, label)
        except (PersonaLoadError, Exception) as exc:
            raise PersonaWriteRefused(str(exc)) from exc

        try:
            if record is None:
                create_clone(draft.name, text, root=self.root)
            else:
                replace_clone_file(record.agent_id, text, root=self.root)
        except DuplicateHandleError as exc:
            raise PersonaWriteConflict(str(exc)) from exc
        except AgentHomeError as exc:
            raise PersonaWriteRefused(str(exc)) from exc
        self.reload()
        persona = self._personas.get(draft.name)
        if persona is None:  # pragma: no cover - the write above just made it
            raise PersonaWriteRefused(f"persona {draft.name!r} was written and cannot be read.")
        return persona

    def _peer_ids(self, peers: Iterable[str]) -> list[str]:
        """`peers` as clone ids, refusing a name no clone carries."""
        claims = clone_handles(self.root)
        ids: list[str] = []
        for peer in peers:
            if is_agent_id(peer):
                found = peer
            else:
                owners = claims.get(peer, ())
                if len(owners) != 1:
                    raise PersonaWriteRefused(
                        f"There is no clone called '{peer}' to name as a peer. "
                        "Choose a clone from the list."
                    )
                found = owners[0]
            if found not in ids:
                ids.append(found)
        return ids


# --- the start-up pass ------------------------------------------------------------------


def ensure_clone_store(
    workspace_root: Path | None,
    *,
    builtin_dir: Path | None,
    extension_dirs: Sequence[Path] = (),
    root: Path | None = None,
    install: bool = True,
) -> CloneStoreReport:
    """Bring the clone root to the one-directory-per-clone layout; a no-op once it is.

    Runs at start, before anything reads a clone (design §3.8): migration, the import of
    `workspace_root`'s personas, install of `builtin_dir`'s and `extension_dirs`' (the
    builtins installed extensions ship), and the peer translation, in
    that order, under the clone root's lock. The report is appended to
    `agents/.migration-<date>.log` when it says anything.

    A root that cannot be created, locked or written is logged and left as it is, never
    raised: the app still opens, and `GET /api/clones` is what says the root is unusable.
    """
    report = CloneStoreReport()
    try:
        with clone_root_lock(root) as base:  # the whole migration, one process at a time
            migrate_legacy_homes(
                base,
                workspace_root=workspace_root,
                builtin_dir=builtin_dir,
                extension_dirs=extension_dirs,
                report=report,
            )
            import_workspace_personas(workspace_root, root=base, report=report)
            if install:
                install_builtin_personas(
                    builtin_dir, extension_dirs=extension_dirs, root=base, report=report
                )
            translate_stored_peers(root=base, report=report)
            if report.lines:
                append_report(base, report)
    except (AgentHomeError, OSError) as exc:
        logger.warning("clone store: the clone root was left as it is: %s", exc)
        report.add(f"the clone root was left as it is: {exc}")
    return report


def append_report(base: Path, report: CloneStoreReport) -> None:
    """Append `report` to the clone root's `.migration-<date>.log` (§3.8 step 4)."""
    now = _dt.datetime.now(_dt.UTC)
    path = base / f".migration-{now.date().isoformat()}.log"
    stamp = now.isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        for line in report.lines:
            handle.write(f"{stamp} {line}\n")


def migrate_legacy_homes(
    root: Path,
    *,
    workspace_root: Path | None,
    builtin_dir: Path | None,
    extension_dirs: Sequence[Path] = (),
    report: CloneStoreReport,
) -> None:
    """§3.8 step 1: give each `agents/<name>/` a `clone.yaml`, then rename it to its id."""
    try:
        entries = sorted(root.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return
    package = _package_mappings(builtin_dir, extension_dirs)
    workspace: dict[str, tuple[Path, dict[str, Any]]] | None = None
    for entry in entries:
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        if is_agent_id(entry.name):
            _repair_an_id_directory(entry, report)
            continue
        try:
            refuse_an_unusable_username(entry.name)
        except AgentHomeError:
            # `GET /api/clones` shows it as a name no clone can have; nothing to migrate.
            continue
        if workspace is None:
            workspace = _workspace_mappings(workspace_root)
        _migrate_one(
            entry,
            root,
            workspace_root=workspace_root,
            package=package,
            workspace=workspace,
            report=report,
        )


def _repair_an_id_directory(entry: Path, report: CloneStoreReport) -> None:
    """An id-named directory whose `id` file is missing gets one; a mismatch is reported."""
    id_path = entry / "id"
    try:
        recorded = id_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        _write_text_atomically(id_path, f"{entry.name}\n")
        report.add(f"agents/{entry.name}: wrote its missing id file.")
        return
    except OSError:
        return
    if recorded != entry.name:
        logger.warning("clone store: %s records the id %r; its name is used", entry, recorded)


def _migrate_one(
    entry: Path,
    root: Path,
    *,
    workspace_root: Path | None,
    package: dict[str, tuple[Path, dict[str, Any]]],
    workspace: dict[str, tuple[Path, dict[str, Any]]],
    report: CloneStoreReport,
) -> None:
    name = entry.name
    legacy = AgentHome.for_legacy_name(name, root)
    try:
        agent_id = legacy.agent_id()
    except AgentHomeError as damage:
        report.add(f"agents/{name}: not migrated, its identity could not be read ({damage}).")
        return
    if not is_agent_id(agent_id):
        report.add(f"agents/{name}: not migrated, its id {agent_id!r} is not a clone id.")
        return
    target = root / agent_id
    if target.exists():
        report.add(f"agents/{name}: not migrated, agents/{agent_id} already exists.")
        return

    if (entry / CLONE_FILE_NAME).is_file():
        # A crash between writing clone.yaml and the rename: the handle is on disk.
        os.rename(entry, target)
        report.add(f"agents/{name} -> agents/{agent_id}: finished an interrupted move.")
        return

    if clone_handles(root).get(name):
        report.add(
            f"agents/{name}: not migrated, a clone called '{name}' already exists. "
            "Its files were left where they are."
        )
        return

    # The picture chosen for a clone lived in the workspace personas directory as
    # `<name>.<ext>`, a built-in's included; a shipped picture stays in the package, where
    # the avatar store finds it by the clone's template.
    files: dict[str, bytes] = {}
    if workspace_root is not None:
        files = _avatar_files(workspace_root / DEFAULT_WORKSPACE_PERSONAS_SUBDIR, name)
    source: Path | None = None
    dropped: list[_Model] = []
    if name in workspace:
        source, mapping = workspace[name]
        # Read only beside an old home, so last used by a build that read no ref: none is
        # kept, `/` or not (see `_earlier_models`).
        mapping, dropped, _ = _earlier_models(mapping, keep_refs=False)
        origin = f"persona from {source}"
        template = None
    elif name in package:
        source, mapping = package[name]
        origin = "builtin persona"
        template = name
    else:
        mapping = package[DEFAULT_PERSONA_NAME][1] if DEFAULT_PERSONA_NAME in package else {}
        mapping = _persona_fields(mapping) or dict(_FALLBACK_PERSONA)
        mapping.pop("a2a_peers", None)
        origin = f"no persona of its own; given the '{DEFAULT_PERSONA_NAME}' builtin's fields"
        template = None
    # The label a person gave it (#1947) moves with it, from a workspace file as from a
    # package one; `_persona_fields` below would drop it, so it is carried separately.
    display_name = {} if source is None else display_name_from_mapping(mapping, source)
    text = _clone_text(name, _persona_fields(mapping), display_name=display_name, template=template)
    _write_text_atomically(entry / CLONE_FILE_NAME, text)
    for file_name, data in files.items():
        if not (entry / file_name).exists():
            (entry / file_name).write_bytes(data)
    os.rename(entry, target)
    report.add(f"agents/{name} -> agents/{agent_id}: handle '{name}', {origin}.")
    _report_earlier_models(report, name, dropped)


def read_imports(folder: Path) -> dict[str, str]:
    """The files imported into the clone at `folder`, by path, with the digest taken."""
    path = folder / IMPORTS_FILE_NAME
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(k): str(v) for k, v in cast(dict[object, object], loaded).items()}


def record_import(folder: Path, source: Path, digest: str) -> None:
    """Record in the clone at `folder` that `source` was taken in at `digest`."""
    records = read_imports(folder)
    records[str(source)] = digest
    _write_text_atomically(folder / IMPORTS_FILE_NAME, dump_persona_yaml(records))


def _comparable(persona: PersonaDefinition, root: Path) -> PersonaDefinition:
    return persona.model_copy(update={"a2a_peers": peer_handles(persona.a2a_peers, root)})


def import_workspace_personas(
    workspace_root: Path | None,
    *,
    root: Path | None = None,
    report: CloneStoreReport | None = None,
) -> CloneStoreReport:
    """§3.8 step 2: make a clone of each workspace persona whose handle has none.

    Public because a conversation that switches workspace runs it on the new directory
    (design §4 step 4). A handle that already has a clone keeps it; a file that differs
    from it is reported once and recorded in that clone's directory.
    """
    report = report if report is not None else CloneStoreReport()
    workspace = _workspace_mappings(workspace_root, report)
    if not workspace:
        return report
    with clone_root_lock(root) as base:
        for handle, (source, raw_mapping) in workspace.items():
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            mapping, earlier, refs = _earlier_models(raw_mapping, keep_refs=True)
            owners = clone_handles(base).get(handle, ())
            if not owners:
                display_name = display_name_from_mapping(mapping, source)
                text = _clone_text(
                    handle, _persona_fields(mapping), display_name=display_name, template=None
                )
                home = create_clone(
                    handle, text, files=_avatar_files(source.parent, source.stem), root=base
                )
                record_import(home.path, source, digest)
                report.add(f"imported {source} as clone '{handle}' (agents/{home.path.name}).")
                _report_earlier_models(report, handle, earlier, refs)
                continue
            folder = base / owners[0]
            if read_imports(folder).get(str(source)) == digest:
                continue
            data = _clone_persona_mapping(read_clone_file(folder) or {}, handle)
            try:
                held = _comparable(persona_from_mapping(data, folder / CLONE_FILE_NAME), base)
                offered = _comparable(persona_from_mapping(mapping, source), base)
            except (PersonaLoadError, ValidationError):
                # A clone made without a persona (only a handle) is not one to compare with;
                # it stays as it is, and a start must not fail on it.
                continue
            record_import(folder, source, digest)
            if held != offered:
                report.add(
                    f"kept clone '{handle}' (agents/{folder.name}); {source} defines it "
                    "differently and was not imported."
                )
    return report


def _read_installed(base: Path) -> list[str]:
    try:
        text = (base / INSTALLED_FILE_NAME).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def install_builtin_personas(
    builtin_dir: Path | None,
    *,
    extension_dirs: Sequence[Path] = (),
    root: Path | None = None,
    report: CloneStoreReport | None = None,
) -> CloneStoreReport:
    """§3.4: install each package persona not yet recorded in `agents/.installed`.

    `extension_dirs` are the builtin folders installed extensions ship (#2208), installed
    the same way.
    """
    report = report if report is not None else CloneStoreReport()
    package = _package_mappings(builtin_dir, extension_dirs)
    if not package:
        return report
    with clone_root_lock(root) as base:
        installed = _read_installed(base)
        added: list[str] = []
        for handle, (source, mapping) in package.items():
            if handle in installed:
                continue
            added.append(handle)
            if clone_handles(base).get(handle):
                report.add(f"builtin '{handle}': a clone of that handle exists; recorded only.")
                continue
            text = _clone_text(
                handle,
                _persona_fields(mapping),
                display_name=display_name_from_mapping(mapping, source),
                template=handle,
            )
            # No picture is copied: the shipped one is found in the package by template,
            # so resetting a chosen picture falls back to it.
            home = create_clone(handle, text, root=base)
            report.add(f"installed builtin '{handle}' as agents/{home.path.name}.")
        if added:
            _write_text_atomically(
                base / INSTALLED_FILE_NAME, "".join(f"{h}\n" for h in [*installed, *added])
            )
    return report


def translate_stored_peers(
    *, root: Path | None = None, report: CloneStoreReport | None = None
) -> CloneStoreReport:
    """§3.8 step 2b: rewrite every stored `a2a_peers` handle to the id it names.

    A handle naming no clone yet is moved to `pending_peers` with its place and reported
    once, so no stored peer is a handle and none is offered while nobody answers it. A
    pending handle is stored by id, back in its place, at the first start after the builtin
    of that handle is installed -- not any clone a person gives that handle (#2245, #2256).
    A clone whose file cannot be parsed is skipped, handles and all, and its peers are
    stored by id at the first start after the file is mended (#2241). Tools are not checked
    (see `_load_problem`).
    """
    report = report if report is not None else CloneStoreReport()
    with clone_root_lock(root) as base:
        claims = clone_handles(base)
        for handle, owners in sorted(claims.items()):
            if len(owners) != 1:
                continue
            folder = base / owners[0]
            try:
                fields = read_clone_file(folder)
            except AgentHomeError:
                continue
            if fields is None:
                continue
            raw: object = fields.get("a2a_peers")
            live = [str(peer) for peer in cast(list[object], raw)] if isinstance(raw, list) else []
            pending = _pending_peers(fields)
            # Written by #2245's build as a list, which says no place: one start drops it,
            # and says so, rather than reading a place it never recorded.
            unreadable = PENDING_PEERS_KEY in fields and not isinstance(
                fields[PENDING_PEERS_KEY], dict
            )
            ready = [peer for peer in pending if _builtin_clone(peer, claims, base) is not None]
            if all(is_agent_id(peer) for peer in live) and not ready and not unreadable:
                continue
            if _load_problem(fields, handle, folder) is not None:
                logger.warning(
                    "clone store: clone '%s' cannot be loaded, so its peers were left as written",
                    handle,
                )
                continue
            if unreadable:
                report.warn(
                    f"clone '{handle}': its waiting peers were saved by an earlier build in a "
                    "form this one does not read, so they were dropped."
                )
            translated: list[str] = []
            still_pending: dict[str, int] = {}
            for peer in with_pending_peers(live, pending):
                if is_agent_id(peer):
                    found: str | None = peer
                elif peer in pending:
                    found = _builtin_clone(peer, claims, base)
                    if found is not None:
                        report.add(f"clone '{handle}': pending peer '{peer}' is installed now.")
                else:
                    named = claims.get(peer, ())
                    found = named[0] if len(named) == 1 else None
                    if found is None and peer not in still_pending:
                        report.add(
                            f"clone '{handle}': peer '{peer}' names no clone yet; it is "
                            "kept pending and becomes a peer once that clone is installed."
                        )
                if found is None:
                    still_pending.setdefault(peer, len(translated) + len(still_pending))
                elif found not in translated:
                    translated.append(found)
            updated = dict(fields)
            if translated:
                updated["a2a_peers"] = translated
            else:
                updated.pop("a2a_peers", None)
            if still_pending:
                updated[PENDING_PEERS_KEY] = still_pending
            else:
                updated.pop(PENDING_PEERS_KEY, None)
            replace_clone_file(owners[0], dump_persona_yaml(updated), root=base)
            report.add(f"clone '{handle}': peers stored by id.")
    return report
