"""The model choice a person saved, read by every head from one file.

The dashboard's Settings panel has always kept its provider/model selection in
``settings.json`` under the session root (``default_session_root()``, which honours
``UCLONE_SESSION_DIR``). The terminal commands never read it: ``ucx install --yes`` ended
with ``setup llm: ready model=qwen3:8b ...`` and the very next ``ucx run --prompt ...``
refused with "No LLM provider is configured", because the factory looked only at
arguments and environment variables. The install had set nothing any of them read.

This module is the one reader and the one first-time writer of that choice, so the
dashboard, ``ucx run``, ``ucx room`` and ``ucx loop`` agree on it:

* :func:`read_saved_choice` returns what was saved, or ``None``.
* :func:`remember_choice_if_unset` fills in a choice **only where none is saved**,
  keeping every other key in the file. It is what setup calls: a model the person picked
  in Settings is theirs, and a later install must not replace it.
* :func:`save_choice` replaces it on the person's instruction (``ucx llm use``).
* :func:`update_settings_file` is the one writer underneath both, and the dashboard's:
  it re-reads the file and merges, so no writer erases keys it did not set.

Reading a saved choice is configuration, not substitution (P6): the person, or setup on
their behalf, named this provider. What stays the caller's job is saying so -- a head that
acts on a saved choice reports that it came from here, via :func:`describe_saved_choice`.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX (Windows): writes go unserialised
    fcntl = None

#: The file the dashboard has always written; the name is shared, not new.
SETTINGS_FILE_NAME = "settings.json"

#: Providers a saved choice may name. Anything else is refused by the factory, naming
#: the file, rather than ignored -- an ignored choice would read as "none saved".
SAVED_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "anthropic", "gemini", "google", "ollama", "vllm", "mock"}
)


@dataclass(frozen=True)
class SavedChoice:
    """One saved provider selection, with the file it came from."""

    provider: str
    model: str | None
    base_url: str | None
    path: Path
    #: Kept out of ``repr`` so a logged or printed choice never carries the credential.
    #: Only a key saved *for this provider*; see :func:`key_belongs_to`.
    api_key: str | None = field(default=None, repr=False)


def _family(provider: str) -> str:
    """Gemini is reached under two names; a key for one is a key for the other."""
    name = provider.strip().lower()
    return "gemini" if name == "google" else name


def same_provider(first: str | None, second: str | None) -> bool:
    """Whether two provider names are the same service (``gemini`` and ``google`` are)."""
    return first is not None and second is not None and _family(first) == _family(second)


def key_owner(data: Mapping[str, Any]) -> str | None:
    """The provider the saved API key was saved for.

    ``llm_api_key_provider`` says so. A key saved before that field existed belongs to the
    provider saved alongside it, which is the only provider it can have been saved for.
    """
    return _clean(data.get("llm_api_key_provider")) or _clean(data.get("llm_provider"))


def key_belongs_to(data: Mapping[str, Any], provider: str | None) -> str | None:
    """The saved API key when it was saved for ``provider``, else ``None``.

    The file keeps one key while the provider changes (`ucx llm use` to Ollama and back
    to OpenAI must not lose the OpenAI key), so the key is applied only where it belongs:
    an OpenAI key is never sent to Anthropic.
    """
    key = _clean(data.get("llm_api_key"))
    if key is None or not same_provider(key_owner(data), provider):
        return None
    return key


def settings_file() -> Path:
    """Where the choice lives: ``<session root>/settings.json``.

    Resolved through ``default_session_root()`` rather than a second ``Path.home()``
    expression, so the dashboard and the terminal cannot disagree about the location.
    Imported here, not at module level: this module sits under the connector factory,
    which the agent package imports while it is still initialising.
    """
    from uclone_x.agent.session import default_session_root

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


def _clean(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def read_saved_choice(path: Path | None = None) -> SavedChoice | None:
    """The saved choice, or ``None`` when no provider is saved.

    ``None`` covers a missing file, one that cannot be parsed, and one the dashboard wrote
    before anything was selected (``"llm_provider": null``). :func:`saved_choice_note`
    tells those apart for the refusal message.
    """
    target = path if path is not None else settings_file()
    data = _read_settings(target)
    if data is None:
        return None
    provider = _clean(data.get("llm_provider"))
    if provider is None:
        return None
    return SavedChoice(
        provider=provider.lower(),
        model=_clean(data.get("llm_model")),
        base_url=_clean(data.get("llm_base_url")),
        path=target,
        api_key=key_belongs_to(data, provider),
    )


def saved_choice_note(path: Path | None = None) -> str:
    """One plain sentence on why no saved choice applies, for the "not configured" refusal."""
    target = path if path is not None else settings_file()
    if not target.exists():
        return f"No model has been saved yet (setup and Settings save one to {target})."
    if _read_settings(target) is None:
        return f"The saved settings at {target} could not be read."
    return f"The saved settings at {target} do not name a model yet."


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


def _current(
    target: Path, replace_unreadable_with: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The file's contents for a merge; ``ValueError`` when it exists but cannot be read."""
    if not target.exists():
        return {}
    current = _read_settings(target)
    if current is not None:
        return current
    if replace_unreadable_with is not None:
        return dict(replace_unreadable_with)
    raise ValueError(f"{target} could not be read, so it was left as it is")


def _merge(target: Path, data: dict[str, Any], updates: Mapping[str, Any]) -> None:
    """Write ``data`` with ``updates`` applied, atomically. The caller holds the lock."""
    old_provider = _clean(data.get("llm_provider"))
    if (
        old_provider is not None
        and "llm_provider" in updates
        and "llm_api_key_provider" not in updates
        and _clean(data.get("llm_api_key")) is not None
        and _clean(data.get("llm_api_key_provider")) is None
    ):
        # The key was saved before keys were tagged, so it belongs to the provider saved
        # with it. Record that before the provider changes, or the key would silently
        # follow the new provider -- and be sent to a service it was never meant for.
        data["llm_api_key_provider"] = old_provider
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
    updates: Mapping[str, Any],
    *,
    path: Path | None = None,
    replace_unreadable_with: Mapping[str, Any] | None = None,
) -> None:
    """Merge ``updates`` into the settings file, keeping every key they do not name.

    The one writer of the file, for setup and for the dashboard alike. It re-reads the
    file under a lock shared across processes, so a writer changes only the keys it
    names: a dashboard started before ``ucx install`` used to rewrite the whole file from
    memory on any Settings save, putting ``"llm_provider": null`` back over the model setup
    had just saved. The file is replaced atomically, so a reader never sees half of it.

    A file that exists but cannot be read raises ``ValueError`` and is left as it is,
    unless the caller passes ``replace_unreadable_with`` -- the whole state it holds, which
    the dashboard does because a Settings save is the person stating every value there.
    Raises ``OSError`` when the file cannot be written.
    """
    target = path if path is not None else settings_file()
    with _locked(target):
        _merge(target, _current(target, replace_unreadable_with), updates)


def remember_choice_if_unset(
    *, provider: str, model: str | None, base_url: str | None, path: Path | None = None
) -> tuple[bool, SavedChoice | None]:
    """Save a first choice, filling in only what is not saved yet.

    Returns ``(written, before)``: whether anything was written, and the choice saved
    before this call (``None`` when no provider was saved). Nothing a person saved is
    replaced:

    * no provider saved -- ``provider`` is saved, and ``model``/``base_url`` wherever the
      file has none;
    * the same provider saved -- only an empty model or address is filled in;
    * another provider saved -- nothing is written.

    Every other key in the file (the dashboard keeps read roots and the ComfyUI address
    there) survives. Raises ``OSError`` when the file cannot be written and ``ValueError``
    when it exists but cannot be read; it is left as it is then, because replacing a file
    this code cannot parse would discard whatever the person had in it.
    """
    target = path if path is not None else settings_file()
    # Decided without the lock first: when there is nothing to fill in (the common case on
    # a re-install), setup writes nothing and so touches neither the file nor its lock,
    # which a read-only or root-owned session directory would refuse.
    _, pending, before = _fill_empty(target, provider, model, base_url)
    if not pending:
        return False, before
    with _locked(target):
        # Decided again under the lock: another writer may have saved since the first read.
        current, updates, before = _fill_empty(target, provider, model, base_url)
        if not updates:
            return False, before
        _merge(target, current, updates)
        return True, before


def _fill_empty(
    target: Path, provider: str, model: str | None, base_url: str | None
) -> tuple[dict[str, Any], dict[str, Any], SavedChoice | None]:
    """The file's contents, what :func:`remember_choice_if_unset` would add, and the choice before."""
    current = _current(target)
    before = read_saved_choice(target)
    if before is not None and before.provider != provider.lower():
        return current, {}, before
    # Each field is judged on its own: a model or address saved without a provider
    # (the dashboard can write one before a provider is picked) is still the person's.
    updates: dict[str, Any] = {}
    if before is None:
        updates["llm_provider"] = provider
    if model is not None and _clean(current.get("llm_model")) is None:
        updates["llm_model"] = model
    if base_url is not None and _clean(current.get("llm_base_url")) is None:
        updates["llm_base_url"] = base_url
    return current, updates, before


def save_choice(
    *, provider: str, model: str, base_url: str | None, path: Path | None = None
) -> None:
    """Replace the saved choice outright -- what ``ucx llm use`` does on request.

    Unlike :func:`remember_choice_if_unset`, this is the person's own instruction, so it
    overwrites the provider, model and address. A saved API key is kept, with the
    provider it was saved for (:func:`key_belongs_to`): switching to Ollama and back to
    OpenAI finds the OpenAI key again, and no other provider is ever sent it.
    """
    update_settings_file(
        {"llm_provider": provider, "llm_model": model, "llm_base_url": base_url}, path=path
    )
