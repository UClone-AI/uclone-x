"""The settings file every head reads, its one writer, and the default model it names.

The dashboard's Settings panel, ``ucx run``, ``ucx room``, ``ucx loop`` and setup all keep
their choices in ``settings.json`` under the session root (``default_session_root()``,
which honours ``UCLONE_SESSION_DIR``):

* :func:`update_settings_file` is the one writer (settings-single-source S1): it locks,
  re-reads and merges, so no writer erases keys it did not set.
* The model choice is the file's ``connections`` and ``default_models``
  (``uclone_x.llm.connections``, model-gateway §3.2). :func:`read_saved_choice` reads the
  default deep model's connection as a :class:`SavedChoice`, which is what a terminal
  command builds its one connector from.
* :func:`remember_choice_if_unset` fills in a first connection and default model **only
  where none is saved** (setup); :func:`save_choice` sets them on request (``ucx llm use``).
* :func:`save_api_key` / :func:`delete_api_key` keep one key per *connection* (S3 as
  revised): saving one connection's key never touches another's.

The pre-gateway keys (``llm_provider``, ``llm_model``, ``llm_model_fast``, ``llm_base_url``,
``llm_api_keys``) are never read (owner ruling 2026-09-28: no users yet, no migration). A
file holding only them reads as having no model saved.

Keys live in this file and nowhere else: not in ``.env``, not in the OS keychain.
Environment variables still override the file (a CI job has to be able to), but nothing
here writes one.

Reading a saved choice is configuration, not substitution (P6): the person, or setup on
their behalf, named this model. What stays the caller's job is saying so -- a head that
acts on a saved choice reports that it came from here, via :func:`describe_saved_choice`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, cast

from uclone_x.core.set_aside import set_aside_unreadable
from uclone_x.llm.connections import (
    ADDABLE_KINDS,
    CONNECTIONS_KEY,
    DEFAULT_COMFYUI_ADDRESS,
    DEFAULT_MODELS_KEY,
    DEFAULT_OLLAMA_ADDRESS,
    SLOTS,
    Connection,
    ConnectionError_,
    ModelRef,
    ModelRefError,
    connection_file_rows,
    refuse_bare_model,
    saved_connections,
    saved_default_models,
    slug_connection_id,
)
from uclone_x.llm.providers import PROVIDERS, canonical_provider, chat_kind

logger = logging.getLogger(__name__)

#: The set-aside's log line: the two paths, never the contents -- the file holds API keys.
_SET_ASIDE_LOG = (
    "Settings file %s could not be read; it was moved aside, unchanged, to %s before this save"
)

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX (Windows): writes go unserialised
    fcntl = None

#: The file the dashboard has always written; the name is shared, not new.
SETTINGS_FILE_NAME = "settings.json"

#: Providers a saved choice may name, by id or alias (``uclone_x.llm.providers``).
#: Anything else is refused by the factory, naming the file, rather than ignored -- an
#: ignored choice would read as "none saved".
SAVED_PROVIDERS: frozenset[str] = frozenset(
    name
    for spec in PROVIDERS.values()
    if "chat" in spec.capabilities
    for name in (spec.id, *spec.aliases)
)


@dataclass(frozen=True)
class SavedChoice:
    """The default deep model as a terminal command builds from it: its connection and model.

    Read from ``default_models.deep`` and the connection its ref names. ``provider`` is the
    connection's kind, ``model`` the model id on it.
    """

    provider: str
    model: str | None
    base_url: str | None
    path: Path
    #: Kept out of ``repr`` so a logged or printed choice never carries the credential.
    #: Only the key saved *for this connection*.
    api_key: str | None = field(default=None, repr=False)
    #: The default fast model's id, when it is on the same connection; else ``None``.
    model_fast: str | None = None
    #: The connection's id (``gemini``, ``gpu-box``).
    connection_id: str | None = None


def _provider_id(provider: str) -> str:
    """``provider``'s id in the provider table, or the lowered name when it is not in it."""
    return canonical_provider(provider) or provider.strip().lower()


def same_provider(first: str | None, second: str | None) -> bool:
    """Whether two provider names are the same service (``gemini`` and ``google`` are)."""
    return (
        first is not None
        and second is not None
        and bool(first.strip())
        and _provider_id(first) == _provider_id(second)
    )


def api_keys(data: Mapping[str, Any]) -> dict[str, str]:
    """Every saved connection's key, by connection id (S3: one key per connection)."""

    return {conn.id: conn.key for conn in saved_connections(data) if conn.key}


def api_key_for(data: Mapping[str, Any], provider: str | None) -> str | None:
    """The key saved for the connection ``provider`` names, else ``None``.

    ``provider`` is a connection id, or a kind: a kind finds only the row whose id is that
    kind (S3). Never another row's key, of another kind or of the same one: a second row of
    a kind has its own address, and its key is its own.
    """

    if provider is None or not provider.strip():
        return None
    name = provider.strip()
    rows = saved_connections(data)
    by_id = next((conn for conn in rows if conn.id == name), None)
    if by_id is not None:
        return by_id.key
    kind = canonical_provider(name)
    if kind is None:
        return None
    exact = next((conn for conn in rows if conn.id == kind), None)
    return exact.key if exact is not None and exact.kind == kind else None


def settings_file() -> Path:
    """Where the choice lives: ``<session root>/settings.json``.

    Resolved through ``default_session_root()`` rather than a second ``Path.home()``
    expression, so the dashboard and the terminal cannot disagree about the location.
    The resolver lives in `uclone_x.core.session`, below both this package and the agent
    package (#1734). Imported here rather than at module level only so that loading the
    connector package adds nothing to the modules it already pulls in.
    """
    from uclone_x.core.session import default_session_root

    return default_session_root() / SETTINGS_FILE_NAME


def _read_settings(path: Path) -> dict[str, Any] | None:
    """The file's top-level object, or ``None`` when absent, unreadable or not an object."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return cast(dict[str, Any], raw)


def settings_data(path: Path | None = None) -> dict[str, Any]:
    """The settings file's contents, or ``{}`` when it is absent or cannot be read.

    For readers that need more than the saved choice -- the connector factory reads the
    key saved for whichever provider it resolved. Never written back: an unreadable file
    reads as empty here, and only :func:`update_settings_file` decides what to do with one.
    """
    data = _read_settings(path if path is not None else settings_file())
    return data if data is not None else {}


def _clean(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def read_saved_choice(path: Path | None = None) -> SavedChoice | None:
    """The saved default deep model and its connection, or ``None`` when there is none.

    ``None`` covers a missing or unreadable file, no ``default_models.deep``, and a deep ref
    whose connection is not saved. Only saved rows are read: the environment's overrides
    are the factory's precedence steps, not a saved choice. :func:`saved_choice_note`
    tells the cases apart for the refusal message.
    """

    target = path if path is not None else settings_file()
    data = _read_settings(target)
    if data is None:
        return None
    defaults = saved_default_models(data)
    if defaults.deep is None:
        return None
    deep = ModelRef.parse(defaults.deep)
    conn = next((c for c in saved_connections(data) if c.id == deep.connection_id), None)
    if conn is None:
        return None
    fast = ModelRef.parse(defaults.fast) if defaults.fast else None
    return SavedChoice(
        provider=conn.kind,
        model=deep.model,
        base_url=conn.base_url,
        path=target,
        api_key=conn.key,
        model_fast=fast.model if fast is not None and fast.connection_id == conn.id else None,
        connection_id=conn.id,
    )


def saved_choice_note(path: Path | None = None) -> str:
    """One plain sentence on why no saved choice applies, for the "not configured" refusal."""
    target = path if path is not None else settings_file()
    if not target.exists():
        return f"No model has been saved yet (setup and Settings save one to {target})."
    if _read_settings(target) is None:
        return f"The saved settings at {target} could not be read."
    return f"The saved settings at {target} do not name a default model on a saved connection yet."


def describe_saved_choice(
    choice: SavedChoice, model: str | None = None, model_from: str | None = None
) -> str:
    """The line a head prints when it acts on a saved choice: what, and from where.

    ``model`` is the model actually asked for, when the caller knows it; ``model_from``
    names the variable it came from when that, not the file, chose it. A model other than
    the saved one is not called "the model saved", only its provider is.
    """
    shown = model or choice.model
    what = f"{shown} ({choice.provider})" if shown else choice.provider
    if shown is None or shown == choice.model:
        return f"Using {what}, the model saved in {choice.path}."
    if model_from is not None:
        return (
            f"Using {what}: the provider saved in {choice.path}, with the model {model_from} names."
        )
    return f"Using {what}, with the provider saved in {choice.path}."


def lock_file(target: Path) -> Path:
    """The sidecar file writers of ``target`` lock, next to it in the same directory."""
    return target.parent / f".{target.name}.lock"


@contextmanager
def _locked(target: Path) -> Generator[None]:
    """Hold the settings lock across processes for one read-merge-replace.

    The dashboard, setup and ``ucx llm use`` can write at the same moment; without the
    lock each reads the old file, and the later replace drops the earlier writer's keys.
    A sidecar is locked rather than the file itself, because the file is replaced (a new
    inode) on every write. Where ``fcntl`` does not exist, writes are not serialised.

    The sidecar is opened read-only (``flock`` needs no write access) and created ``0600``.
    When it cannot be opened at all -- a read-only directory, or a lock file left owned by
    root after a ``sudo`` run -- the write goes ahead unlocked: an unserialised save is the
    behaviour before the lock existed, and refusing the save over it would be worse.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:  # pragma: no cover - not POSIX
        yield
        return
    try:
        fd = os.open(lock_file(target), os.O_RDONLY | os.O_CREAT, 0o600)
    except OSError:
        yield
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def _current(target: Path) -> dict[str, Any]:
    """The file's contents for a merge; ``ValueError`` when it exists but cannot be read."""
    if not target.exists():
        return {}
    current = _read_settings(target)
    if current is not None:
        return current
    raise ValueError(f"{target} could not be read, so it was left as it is")


def _set_aside_settings(target: Path) -> Path:
    """Move the unreadable settings file aside, never writing over it (#1844); log where.

    It may be a newer build's settings, API keys included, that this build cannot parse. If
    the rename fails, the ``OSError`` stops the write and the file stays where it was. The
    log names the two paths and nothing the file holds: it holds keys (#1860).
    """
    aside = set_aside_unreadable(target)
    logger.warning(_SET_ASIDE_LOG, target, aside)
    return aside


def _merge(target: Path, data: dict[str, Any], updates: Mapping[str, Any]) -> None:
    """Write ``data`` with ``updates`` applied, atomically. The caller holds the lock."""
    data.update(updates)
    fd, temp = tempfile.mkstemp(dir=target.parent, prefix=".settings-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def update_settings_file(
    updates: Mapping[str, Any] | Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    path: Path | None = None,
    replace_unreadable_with: Mapping[str, Any] | None = None,
) -> Path | None:
    """Merge ``updates`` into the settings file, keeping every key they do not name.

    The one writer of the file, for setup and for the dashboard alike. It re-reads the
    file under a lock shared across processes, so a writer changes only the keys it
    names: a dashboard started before ``ucx install`` used to rewrite the whole file from
    memory on any Settings save, putting ``"llm_provider": null`` back over the model setup
    had just saved. The file is replaced atomically, so a reader never sees half of it.

    A file that exists but cannot be read raises ``ValueError`` and is left as it is,
    unless the caller passes ``replace_unreadable_with`` -- the whole state it holds, which
    the dashboard does because a Settings save is the person stating every value there.
    Even then the unreadable file is first renamed to ``settings.json.unreadable-<UTC
    time>`` beside it, so its contents are kept rather than written over; that new path is
    returned, so the caller can tell the person (#1860). ``None`` when nothing was set aside.
    Raises ``OSError`` when the file cannot be written or set aside.

    ``updates`` may be a function of the file's current contents, called under the lock: a
    writer that changes one row of a list (a connection) computes the new list from what is
    in the file at that moment, so a row another process saved meanwhile is kept. An
    exception it raises leaves the file as it is.
    """
    target = path if path is not None else settings_file()
    with _locked(target):
        aside: Path | None = None
        try:
            current = _current(target)
        except ValueError:
            if replace_unreadable_with is None:
                raise
            aside = _set_aside_settings(target)
            current = dict(replace_unreadable_with)
        _merge(target, current, updates(current) if callable(updates) else updates)
        return aside


def remember_choice_if_unset(
    *, provider: str, model: str | None, base_url: str | None, path: Path | None = None
) -> tuple[bool, SavedChoice | None]:
    """Save a first connection and default model, filling in only what is not saved yet.

    Returns ``(written, before)``: whether anything was written, and the choice saved
    before this call (``None`` when no default deep model was saved). Nothing a person saved
    is replaced:

    * no connection whose id is ``provider``'s kind -- one is added, at ``base_url``;
    * no default deep model -- ``<kind>/<model>`` becomes it, when ``model`` is given;
    * otherwise nothing is written.

    Every other key in the file survives. Raises ``OSError`` when the file cannot be written
    and ``ValueError`` when it exists but cannot be read; it is left as it is then.
    """
    target = path if path is not None else settings_file()
    kind = _known_provider(provider)
    before = read_saved_choice(target)
    written: list[bool] = []

    def change(current: Mapping[str, Any]) -> dict[str, Any]:

        updates: dict[str, Any] = {}
        rows = saved_connections(current)
        if not any(row.id == kind for row in rows):
            added = Connection(id=kind, kind=kind, base_url=_clean(base_url))
            updates[CONNECTIONS_KEY] = connection_file_rows([*rows, added], current)
        if model is not None and saved_default_models(current).deep is None:
            raw = current.get(DEFAULT_MODELS_KEY)
            stored = dict(cast(Mapping[str, Any], raw)) if isinstance(raw, Mapping) else {}
            stored["deep"] = f"{kind}/{model.strip()}"
            updates[DEFAULT_MODELS_KEY] = stored
        written.append(bool(updates))
        return updates

    # Decided without the lock first: when there is nothing to fill in (the common case on
    # a re-install), setup writes nothing and so touches neither the file nor its lock,
    # which a read-only or root-owned session directory would refuse.
    if not change(_current(target)):
        return False, before
    written.clear()
    update_settings_file(change, path=target)
    return written[-1], before


def save_choice(
    *, provider: str, model: str, base_url: str | None, path: Path | None = None
) -> None:
    """Make ``<kind>/<model>`` the default deep model -- what ``ucx llm use`` does on request.

    The connection whose id is the kind is added, or its address replaced; its key and
    every other connection are kept (S3).
    """
    kind = _known_provider(provider)

    def change(current: Mapping[str, Any]) -> dict[str, Any]:

        rows = saved_connections(current)
        found = next((row for row in rows if row.id == kind), None)
        if found is not None:
            _refuse_unsupported(found)
        if found is None:
            rows.append(Connection(id=kind, kind=kind, base_url=_clean(base_url)))
        else:
            rows = [
                replace(row, base_url=_clean(base_url)) if row.id == kind else row for row in rows
            ]
        raw = current.get(DEFAULT_MODELS_KEY)
        stored = dict(cast(Mapping[str, Any], raw)) if isinstance(raw, Mapping) else {}
        stored["deep"] = f"{kind}/{model.strip()}"
        return {CONNECTIONS_KEY: connection_file_rows(rows, current), DEFAULT_MODELS_KEY: stored}

    update_settings_file(change, path=path)


def _known_provider(provider: str) -> str:
    """``provider``'s id; ``ValueError`` naming it when the provider table does not know it."""
    provider_id = canonical_provider(provider)
    if provider_id is None or not chat_kind(provider_id):
        # An image engine (`comfyui`) is a connection too, but it holds no conversation
        # and takes no key, so it is never a chat choice or a key's home.
        known = ", ".join(sorted(p for p in PROVIDERS if chat_kind(p)))
        raise ValueError(f"There is no provider called {provider.strip()!r}. Known: {known}.")
    return provider_id


def _connection_for_key(data: Mapping[str, Any], name: str) -> tuple[str, str]:
    """``(connection id, kind)`` a key saved under ``name`` goes to.

    ``name`` is a saved connection's id, or a kind: a kind names the row whose id is that
    kind, made when ``ucx key set`` is the first to name it.
    """

    clean = name.strip()
    found = next((row for row in saved_connections(data) if row.id == clean), None)
    if found is not None:
        return found.id, found.kind
    kind = _known_provider(clean)
    return kind, kind


def _change_key(target: Path, name: str, key: str | None) -> None:
    """Set (or, with ``None``, remove) one connection's key, keeping every other row."""

    def change(current: Mapping[str, Any]) -> dict[str, Any]:

        conn_id, kind = _connection_for_key(current, name)
        rows = saved_connections(current)
        if not any(row.id == conn_id for row in rows):
            if key is None:
                return {}
            rows.append(Connection(id=conn_id, kind=kind))
        for row in rows:
            if row.id == conn_id:
                _refuse_unsupported(row)
        rows = [replace(row, key=key) if row.id == conn_id else row for row in rows]
        return {CONNECTIONS_KEY: connection_file_rows(rows, current)}

    update_settings_file(change, path=target)


def save_api_key(provider: str, key: str, *, path: Path | None = None) -> None:
    """Save ``key`` as one connection's key, leaving every other connection's key as it is.

    ``provider`` is a saved connection's id, or a kind (``google`` saves under ``gemini``),
    which names the row whose id is that kind and adds it when there is none. Saving a key
    is not choosing a model. Raises ``ValueError`` for an empty key or a name that is
    neither, and when the file exists but cannot be read (it is left as it is);
    ``OSError`` when it cannot be written.
    """
    clean = _clean(key)
    if clean is None:
        raise ValueError("The key is empty, so nothing was saved.")
    _change_key(settings_file() if path is None else path, provider, clean)


def delete_api_key(provider: str, *, path: Path | None = None) -> None:
    """Remove one connection's saved key, leaving every other connection's key as it is.

    Removing a key that is not saved is not an error. A key set in the environment is not
    touched: the environment is read, never written.
    """
    _change_key(settings_file() if path is None else path, provider, None)


# -- Connections and default models (model-gateway §3.2), through the one writer above. --


def _refuse_unsupported(conn: Connection) -> None:
    """Refuse a change to a row of a kind this build does not know: it is kept as written."""
    if conn.unsupported:
        raise ConnectionError_(
            f"The connection {conn.id} is of a kind this version does not support "
            f"({conn.kind!r}), so it cannot be changed here. Remove it in "
            f"Settings and add it again, or use a version that supports it."
        )


def add_connection(
    kind: str,
    *,
    label: str | None = None,
    base_url: str | None = None,
    key: str | None = None,
    path: Path | None = None,
) -> Connection:
    """Save a new connection and return it with its id (§3.1).

    The first connection of a kind takes the kind as its id; another takes its label (or
    the kind) slugged and made unique. Refuses a kind the person cannot add.
    """
    clean_kind = canonical_provider(kind)
    if clean_kind is None or clean_kind not in ADDABLE_KINDS:
        raise ConnectionError_(
            f"There is no kind of connection called {kind.strip()!r}. "
            f"Choose one of: {', '.join(ADDABLE_KINDS)}."
        )
    clean_base = _clean(base_url)
    if clean_kind == "vllm" and clean_base is None:
        raise ConnectionError_("A vLLM connection needs the server's address.")
    if clean_kind == "remote_gpu" and clean_base is None:
        raise ConnectionError_("A GPU server connection needs the worker's address.")
    if clean_kind == "ollama" and clean_base is None:
        clean_base = DEFAULT_OLLAMA_ADDRESS
    if clean_kind == "comfyui" and clean_base is None:
        clean_base = DEFAULT_COMFYUI_ADDRESS
    created: list[Connection] = []

    def change(current: Mapping[str, Any]) -> dict[str, Any]:
        rows = saved_connections(current)
        taken = [row.id for row in rows]
        conn_id = (
            clean_kind
            if clean_kind not in taken and not _clean(label)
            else slug_connection_id(_clean(label) or clean_kind, taken)
        )
        conn = Connection(
            id=conn_id, kind=clean_kind, base_url=clean_base, key=_clean(key), label=_clean(label)
        )
        created.append(conn)
        return {CONNECTIONS_KEY: connection_file_rows([*rows, conn], current)}

    update_settings_file(change, path=path)
    return created[0]


_UNCHANGED: Final = object()


def update_connection(
    conn_id: str,
    *,
    label: object = _UNCHANGED,
    base_url: object = _UNCHANGED,
    key: object = _UNCHANGED,
    kind: str | None = None,
    create: bool = False,
    path: Path | None = None,
) -> Connection:
    """Change one saved row's label, address or key; every other row is left as it is.

    An empty string clears a field. With ``create``, a row of ``kind`` is made under
    ``conn_id`` when none is saved (``ucx key set gemini``). Saving one connection's key
    never touches another's (S3).
    """
    changed: list[Connection] = []

    def change(current: Mapping[str, Any]) -> dict[str, Any]:
        rows = saved_connections(current)
        index = next((i for i, row in enumerate(rows) if row.id == conn_id), None)
        if index is None:
            if not create or kind is None:
                raise ConnectionError_(f"There is no connection called {conn_id!r}.")
            rows.append(Connection(id=conn_id, kind=kind))
            index = len(rows) - 1
        _refuse_unsupported(rows[index])
        updates: dict[str, Any] = {}
        for name, value in (("label", label), ("base_url", base_url), ("key", key)):
            if value is not _UNCHANGED:
                updates[name] = _clean(value)
        rows[index] = replace(rows[index], **updates)
        changed.append(rows[index])
        return {CONNECTIONS_KEY: connection_file_rows(rows, current)}

    update_settings_file(change, path=path)
    return changed[0]


def remove_connection(conn_id: str, *, path: Path | None = None) -> None:
    """Remove one saved row. Removing a row that is not saved is refused, naming it."""

    def change(current: Mapping[str, Any]) -> dict[str, Any]:
        rows = saved_connections(current)
        kept = [row for row in rows if row.id != conn_id]
        if len(kept) == len(rows):
            raise ConnectionError_(f"There is no connection called {conn_id!r}.")
        return {CONNECTIONS_KEY: connection_file_rows(kept, current)}

    update_settings_file(change, path=path)


def save_default_models(changes: Mapping[str, str | None], *, path: Path | None = None) -> None:
    """Set the default refs ``changes`` names, keeping the others; ``None`` clears a slot."""
    for slot, value in changes.items():
        if slot not in SLOTS:
            raise ModelRefError(f"There is no default model called {slot!r}.")
        refuse_bare_model(value, slot=slot, allow_auto=slot == "image")

    def change(current: Mapping[str, Any]) -> dict[str, Any]:
        raw = current.get(DEFAULT_MODELS_KEY)
        stored: dict[str, Any] = (
            dict(cast(Mapping[str, Any], raw)) if isinstance(raw, Mapping) else {}
        )
        for slot, value in changes.items():
            if value is None:
                stored.pop(slot, None)
            else:
                stored[slot] = value.strip()
        return {DEFAULT_MODELS_KEY: stored}

    update_settings_file(change, path=path)
