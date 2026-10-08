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
   naming no clone is dropped and reported. A stored peer is never a handle.

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
from collections.abc import Collection, Iterable, Sequence
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

logger = logging.getLogger(__name__)

__all__ = [
    "AVATAR_SUFFIXES",
    "INSTALLED_FILE_NAME",
    "CloneDirectoryPersonaStore",
    "CloneRecord",
    "CloneStoreReport",
    "append_report",
    "ensure_clone_store",
    "import_workspace_personas",
    "install_builtin_personas",
    "migrate_legacy_homes",
    "read_imports",
    "record_import",
    "translate_stored_peers",
]

#: Records which package personas this install has installed, one handle per line.
#: Recording, not presence, is what stops a builtin the user deleted from coming back.
INSTALLED_FILE_NAME = ".installed"

#: In a clone's directory: which workspace files were imported into it or conflicted with
#: it, by path and digest, so a conflict is reported once (§3.8 step 2).
IMPORTS_FILE_NAME = "imports.yaml"

#: The picture formats a clone's directory may hold, as `avatar<suffix>`.
AVATAR_SUFFIXES: tuple[str, ...] = (".png", ".webp", ".jpg", ".jpeg", ".gif")

#: A clone's own keys in `clone.yaml`, written before the persona fields.
_CLONE_KEYS: tuple[str, ...] = ("handle", "display_name", "template")

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


# --- reading and writing one clone file -------------------------------------------------


def _clone_text(
    handle: str,
    persona_fields: dict[str, Any],
    *,
    display_name: dict[str, str] | None,
    template: str | None,
) -> str:
    """The text of a `clone.yaml`: the clone's own keys, then its persona fields."""
    contents: dict[str, Any] = {"handle": handle}
    if display_name:
        contents["display_name"] = dict(display_name)
    if template:
        contents["template"] = template
    for key, value in persona_fields.items():
        if key not in (*_CLONE_KEYS, "name", "id"):
            contents[key] = value
    return dump_persona_yaml(contents)


def _persona_fields(mapping: dict[str, Any]) -> dict[str, Any]:
    """`mapping` without the keys that say who the clone is rather than how it behaves."""
    return {key: value for key, value in mapping.items() if key not in (*_CLONE_KEYS, "name", "id")}


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


def _package_mappings(builtin_dir: Path | None) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Every package persona by handle, with its file and raw mapping."""
    if builtin_dir is None or not builtin_dir.is_dir():
        return {}
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(builtin_dir.glob("*.yaml")) + sorted(builtin_dir.glob("*.yml")):
        mapping = read_persona_mapping(path.read_text(encoding="utf-8"), path)
        handle = str(mapping.get("name") or mapping.get("id") or path.stem)
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
            persona = persona_from_mapping(mapping, path)
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
            # A clone whose file says only who it is has no persona of its own: it is a
            # clone (its handle resolves, it has memory) and it speaks as its host's
            # fallback, as an agent with no persona always has (`build_clone`).
            if _persona_fields(fields):
                self._personas[handle] = self._parse(fields, handle, source)
            template = fields.get("template")
            self._records[handle] = CloneRecord(
                agent_id=entry.name,
                handle=handle,
                display_name=display_name_from_mapping(fields, source),
                template=template if isinstance(template, str) else None,
                path=entry,
            )

    def _parse(self, fields: dict[str, Any], handle: str, source: Path) -> PersonaDefinition:
        data = _clone_persona_mapping(fields, handle)
        try:
            persona = persona_from_mapping(data, source)
        except PersonaLoadError:
            raise
        except Exception as exc:
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

        An edit keeps the clone's display name unless the draft carries one, and always
        keeps its template. Peers are written as ids.
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
        display_name = (
            draft.display_name
            if "display_name" in draft.model_fields_set or record is None
            else record.display_name
        )
        template = record.template if record is not None else None
        text = _clone_text(draft.name, fields, display_name=display_name, template=template)
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
    root: Path | None = None,
    install: bool = True,
) -> CloneStoreReport:
    """Bring the clone root to the one-directory-per-clone layout; a no-op once it is.

    Runs at start, before anything reads a clone (design §3.8): migration, the import of
    `workspace_root`'s personas, install of `builtin_dir`'s, and the peer translation, in
    that order, under the clone root's lock. The report is appended to
    `agents/.migration-<date>.log` when it says anything.

    A root that cannot be created, locked or written is logged and left as it is, never
    raised: the app still opens, and `GET /api/clones` is what says the root is unusable.
    """
    report = CloneStoreReport()
    try:
        with clone_root_lock(root) as base:  # the whole migration, one process at a time
            migrate_legacy_homes(
                base, workspace_root=workspace_root, builtin_dir=builtin_dir, report=report
            )
            import_workspace_personas(workspace_root, root=base, report=report)
            if install:
                install_builtin_personas(builtin_dir, root=base, report=report)
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
    report: CloneStoreReport,
) -> None:
    """§3.8 step 1: give each `agents/<name>/` a `clone.yaml`, then rename it to its id."""
    try:
        entries = sorted(root.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return
    package = _package_mappings(builtin_dir)
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
    if name in workspace:
        source, mapping = workspace[name]
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
        for handle, (source, mapping) in workspace.items():
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
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
    root: Path | None = None,
    report: CloneStoreReport | None = None,
) -> CloneStoreReport:
    """§3.4: install each package persona not yet recorded in `agents/.installed`."""
    report = report if report is not None else CloneStoreReport()
    package = _package_mappings(builtin_dir)
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

    A handle naming no clone is dropped and reported, so no stored peer is a handle.
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
            raw: object = (fields or {}).get("a2a_peers")
            if not isinstance(raw, list) or not raw:
                continue
            peers = [str(peer) for peer in cast(list[object], raw)]
            if all(is_agent_id(peer) for peer in peers):
                continue
            translated: list[str] = []
            for peer in peers:
                if is_agent_id(peer):
                    found: str | None = peer
                else:
                    named = claims.get(peer, ())
                    found = named[0] if len(named) == 1 else None
                    if found is None:
                        report.add(
                            f"clone '{handle}': peer '{peer}' names no clone and was dropped."
                        )
                if found is not None and found not in translated:
                    translated.append(found)
            updated = dict(fields or {})
            if translated:
                updated["a2a_peers"] = translated
            else:
                updated.pop("a2a_peers", None)
            replace_clone_file(owners[0], dump_persona_yaml(updated), root=base)
            report.add(f"clone '{handle}': peers stored by id.")
    return report
