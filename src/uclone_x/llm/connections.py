"""Connections and model refs: where a model is, and which one (model-gateway §3.1-3.2).

A **connection** is one reachable model source -- a cloud API with its key, a local Ollama,
a vLLM box on the network -- kept as one row of ``connections`` in ``settings.json``. Several
are kept side by side, so connecting one never disconnects another (G1).

A **model ref** names one model on one connection: ``<connection id>/<model id>``, split at
the *first* ``/`` because a model id may contain one (``meta-llama/Llama-3.3-70B``) and a
connection id may not. A bare id, with no connection, is refused wherever a ref is stored
(a persona, a default model), so a model never silently runs on whichever connection
happens to be first.

The **default models** (``deep``, ``fast``, ``image``) are refs in ``default_models``.

The environment still overrides the file and is never written (settings-single-source S4,
as revised by model-gateway §3.2):

* ``LLM_PROVIDER`` or an endpoint variable (``OLLAMA_HOST``, ``VLLM_BASE_URL``) makes an
  ephemeral connection whose id is the kind, shadowing a saved row of that id;
* a key variable on its own (``GEMINI_API_KEY``) supplies the key of the connection whose id
  is that kind, or makes an ephemeral one when none is saved;
* a model variable (``OPENAI_MODEL``) holds a bare id, read as ``<kind>/<id>``, and
  overrides ``default_models.deep`` for the process. It is the one place a bare id is read.

This module only parses: the environment's overrides are applied by the gateway
(`uclone_x.llm.gateway.connections_in_effect`), and every write goes through the settings
file's one writer, `uclone_x.llm.connectors.saved_choice` (S1).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

from uclone_x.errors import LLMCredentialsNotConfiguredError, PlainRefusalError
from uclone_x.llm.providers import PROVIDERS, canonical_provider, spec_for

#: The settings-file field holding the connection rows.
CONNECTIONS_KEY: Final = "connections"

#: The settings-file field holding the default model refs.
DEFAULT_MODELS_KEY: Final = "default_models"

#: The image default that picks a ready engine on its own (image doc §3.2; step 5).
IMAGE_AUTO: Final = "auto"

#: The chat kinds a person may add. `mock` stays readable (tests, demos) but is not offered.
CHAT_KINDS: Final[tuple[str, ...]] = ("gemini", "openai", "anthropic", "ollama", "vllm")

#: The image engines a person may add (model-gateway §3.5): a ComfyUI daemon, and the remote
#: GPU worker. They draw pictures and hold no conversation.
IMAGE_KINDS: Final[tuple[str, ...]] = ("comfyui", "remote_gpu")

#: Every kind Settings › Connections offers, in the order it lists them.
ADDABLE_KINDS: Final[tuple[str, ...]] = (*CHAT_KINDS, *IMAGE_KINDS)

#: Kinds whose address the person types (a local or remote server).
BASE_URL_KINDS: Final = frozenset({"ollama", "vllm", "comfyui", "remote_gpu"})

#: The one model a remote GPU worker connection offers: the worker takes no checkpoint, so
#: it cannot be told which model to use, and only ``<connection>/auto`` is accepted (§3.5).
REMOTE_GPU_MODEL: Final = "auto"

#: The address Ollama answers on when nothing else is said.
DEFAULT_OLLAMA_ADDRESS: Final = "http://127.0.0.1:11434"

#: The address a ComfyUI daemon answers on when nothing else is said.
DEFAULT_COMFYUI_ADDRESS: Final = "http://127.0.0.1:8188"

ConnectionSource = Literal["settings", "env"]
Slot = Literal["deep", "fast", "image"]
SLOTS: Final[tuple[Slot, ...]] = ("deep", "fast", "image")

logger = logging.getLogger(__name__)

#: The unreadable-row contents already warned about, so a file read on every call warns once.
_WARNED_UNREADABLE: set[str] = set()

_SLUG_INVALID = re.compile(r"[^a-z0-9-]+")


class ModelRefError(PlainRefusalError, ValueError):
    """A model name that is not a ref, or names a connection that does not exist.

    Its text is written for the person: what was wrong, and how to write it instead.
    """


#: The key a connection with no key of its own is built with, for a kind whose server takes
#: none (vLLM, Ollama): the connector sends no credential and reads no key variable, so a
#: key saved on another row, or the kind's variable, never reaches this row's address (S3).
NO_KEY: Final = ""


class ConnectionKeyMissingError(LLMCredentialsNotConfiguredError):
    """A connection of a kind that needs a key, with none of its own (S3).

    Refused before anything is built, as a connector refuses a missing key: borrowing the
    key of another row of the same kind, or the kind's variable, would send that credential
    to this row's address.
    """

    def __init__(self, conn_id: str) -> None:
        super().__init__(
            f"The connection {conn_id} has no key, so its models cannot be used. "
            f"Add a key to the connection {conn_id}."
        )
        self.connection_id = conn_id


def connection_key(conn: Connection) -> str:
    """The key ``conn``'s connector is built with: its own, and nobody else's (S3).

    The environment's key reaches only the row whose id is its kind, and it is already in
    ``conn.key`` there (`connections_in_effect`). A kind that needs a key and has none is
    refused (`ConnectionKeyMissingError`); one whose server takes none gets `NO_KEY`.
    """
    if conn.key:
        return conn.key
    spec = spec_for(conn.kind)
    if spec is not None and spec.requires_key:
        raise ConnectionKeyMissingError(conn.id)
    return NO_KEY


class ConnectionError_(PlainRefusalError, ValueError):  # noqa: N801 - `ConnectionError` is a builtin
    """A connection change Settings refuses: an unknown kind or id, or an env-set row."""


@dataclass(frozen=True)
class ModelRef:
    """``<connection id>/<model id>``: one model on one connection."""

    connection_id: str
    model: str

    def __str__(self) -> str:
        return f"{self.connection_id}/{self.model}"

    @classmethod
    def parse(cls, text: str) -> ModelRef:
        """The ref ``text`` spells, split at its first ``/``; a plain refusal otherwise."""
        clean = text.strip()
        head, sep, tail = clean.partition("/")
        if not sep or not head.strip() or not tail.strip():
            raise ModelRefError(
                f"The model {clean!r} does not say which connection it is on. "
                f"Name the connection first, for example gemini/{clean or 'model-name'}."
            )
        return cls(connection_id=head.strip(), model=tail.strip())


def is_model_ref(text: str | None) -> bool:
    """Whether ``text`` is a well-formed ref (``None`` and ``""`` are not)."""
    if not text:
        return False
    try:
        ModelRef.parse(text)
    except ModelRefError:
        return False
    return True


def refuse_bare_model(text: str | None, *, slot: str, allow_auto: bool = False) -> None:
    """Refuse ``text`` unless it is empty, a ref, or (for a picture slot) ``auto``."""
    if text is None or (allow_auto and text.strip() == IMAGE_AUTO):
        return
    try:
        ModelRef.parse(text)
    except ModelRefError as exc:
        raise ModelRefError(f"{slot}: {exc}") from exc


#: The row fields ``Connection`` reads; every other field of a row is kept as written.
_KNOWN_ROW_FIELDS = frozenset({"id", "kind", "base_url", "key", "label"})


@dataclass(frozen=True)
class Connection:
    """One reachable model source, as the settings file (or the environment) holds it."""

    id: str
    kind: str
    base_url: str | None = None
    #: Kept out of ``repr``, so a logged connection never carries the credential.
    key: str | None = field(default=None, repr=False)
    label: str | None = None
    source: ConnectionSource = "settings"
    #: The variable the key came from, when the environment supplied it.
    key_env_var: str | None = None
    #: The variable behind a row whose ``source`` is ``"env"`` (``LLM_PROVIDER``,
    #: ``OLLAMA_HOST``, ``OPENAI_API_KEY``): the one to change or unset.
    env_var: str | None = None
    #: The row names a kind this build does not know (``kind`` is kept as written). It is
    #: listed with that said, never called, and saved back exactly as it was read (P6).
    unsupported: bool = False
    #: The row as read. An unsupported one is written back unchanged, since this build
    #: cannot know which of its fields matter; a supported one keeps the fields this build
    #: does not know (a newer version's), and its known fields come from the attributes.
    #: Kept out of ``repr`` (it may hold a key).
    raw: Mapping[str, object] | None = field(default=None, repr=False, compare=False)

    @property
    def display_label(self) -> str:
        """What a person reads: the label, else the kind's name (``Google``)."""
        if self.label:
            return self.label
        spec = spec_for(self.kind)
        return spec.display_name if spec is not None else self.id

    def file_row(self) -> dict[str, Any]:
        """The row this connection is saved as; empty fields are left out.

        An unsupported row is saved exactly as it was read. A supported one keeps every
        field this build does not know, as read (P6: a writer never deletes what it could
        not understand), and writes its known fields from the attributes, so a cleared key
        stays cleared.
        """
        if self.unsupported and self.raw is not None:
            return dict(self.raw)
        unknown = {
            name: value for name, value in (self.raw or {}).items() if name not in _KNOWN_ROW_FIELDS
        }
        row: dict[str, Any] = {"id": self.id, "kind": self.kind, **unknown}
        if self.base_url:
            row["base_url"] = self.base_url
        if self.key:
            row["key"] = self.key
        if self.label:
            row["label"] = self.label
        return row


@dataclass(frozen=True)
class DefaultModels:
    """The system default refs; ``None`` is unset, and an unset fast follows deep."""

    deep: str | None = None
    fast: str | None = None
    image: str = IMAGE_AUTO
    #: slot -> the variable that set it, for a default the environment overrides.
    env_vars: Mapping[str, str] = field(default_factory=dict[str, str])

    def get(self, slot: Slot) -> str | None:
        """The ref saved for ``slot`` (``fast`` falls back to ``deep``)."""
        if slot == "deep":
            return self.deep
        if slot == "fast":
            return self.fast or self.deep
        return self.image


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _read_rows(data: Mapping[str, Any]) -> tuple[list[Connection], list[object]]:
    """``connections`` as connections, and the items that cannot be one (kept as written)."""
    raw = data.get(CONNECTIONS_KEY)
    if not isinstance(raw, list):
        return [], []
    rows: list[Connection] = []
    unreadable: list[object] = []
    seen: set[str] = set()
    for item in cast(list[object], raw):
        if not isinstance(item, Mapping):
            unreadable.append(item)
            continue
        row = cast(Mapping[str, object], item)
        conn_id = _text(row.get("id"))
        if conn_id is None or "/" in conn_id or conn_id in seen:
            unreadable.append(row)
            continue
        seen.add(conn_id)
        written_kind = _text(row.get("kind"))
        kind = canonical_provider(written_kind)
        rows.append(
            Connection(
                id=conn_id,
                kind=kind if kind is not None else (written_kind or ""),
                base_url=_text(row.get("base_url")),
                key=_text(row.get("key")),
                label=_text(row.get("label")),
                unsupported=kind is None,
                raw=dict(row),
            )
        )
    seen_text = repr(unreadable)
    if unreadable and seen_text not in _WARNED_UNREADABLE:
        # Not a connection anyone can address (no id, a "/" in it, or an id used twice), so
        # nothing can list it; said here (once per content: the file is read on every
        # call), and kept in the file by every writer.
        _WARNED_UNREADABLE.add(seen_text)
        logger.warning(
            "Saved connections with no usable id (%d) were kept in the settings file as "
            "written, and are not listed.",
            len(unreadable),
        )
    return rows, unreadable


def saved_connections(data: Mapping[str, Any]) -> list[Connection]:
    """The rows in a settings file's ``connections``.

    A row is read with an id (no ``/``); the first row of an id wins. A row whose kind the
    provider table does not know is read too, marked ``unsupported``: it is listed with
    that said and every turn that names it is refused in plain words, never dropped (P6).
    Nothing else in the file is consulted: the pre-gateway keys (``llm_provider``,
    ``llm_api_keys``, ...) are never read (no migration, by ruling).
    """
    return _read_rows(data)[0]


def connection_file_rows(
    connections: Sequence[Connection], current: Mapping[str, Any]
) -> list[object]:
    """``connections`` as the file's rows, followed by the items of ``current`` no build
    could read, unchanged: a writer never deletes what it could not understand."""
    return [*(conn.file_row() for conn in connections), *_read_rows(current)[1]]


def saved_default_models(data: Mapping[str, Any]) -> DefaultModels:
    """``default_models`` as saved; a value that is not a ref is read as unset."""
    raw = data.get(DEFAULT_MODELS_KEY)
    stored: Mapping[str, object] = (
        cast(Mapping[str, object], raw) if isinstance(raw, Mapping) else {}
    )
    deep = _text(stored.get("deep"))
    fast = _text(stored.get("fast"))
    image = _text(stored.get("image")) or IMAGE_AUTO
    return DefaultModels(
        deep=deep if is_model_ref(deep) else None,
        fast=fast if is_model_ref(fast) else None,
        image=image if image == IMAGE_AUTO or is_model_ref(image) else IMAGE_AUTO,
    )


def slug_connection_id(text: str, taken: Sequence[str]) -> str:
    """A connection id from ``text``: lower case, ``a-z0-9-``, unique among ``taken``."""
    base = _SLUG_INVALID.sub("-", text.strip().lower()).strip("-") or "connection"
    candidate, number = base, 2
    while candidate in taken:
        candidate, number = f"{base}-{number}", number + 1
    return candidate


def kind_rows() -> list[dict[str, Any]]:
    """Every kind a person may add, as ``GET /api/connections`` lists them (§3.7.1).

    Each kind says what its models can do (``capabilities``, from the provider table), so a
    head can tell a picture engine from a chat connection without a list of its own.
    """
    defaults = {"ollama": DEFAULT_OLLAMA_ADDRESS, "comfyui": DEFAULT_COMFYUI_ADDRESS}
    rows: list[dict[str, Any]] = []
    for kind in ADDABLE_KINDS:
        spec = PROVIDERS[kind]
        rows.append(
            {
                "kind": kind,
                "label": spec.display_name,
                "needs_key": spec.requires_key,
                "needs_base_url": kind in BASE_URL_KINDS,
                "default_base_url": defaults.get(kind),
                "key_url": spec.console_url,
                "capabilities": list(spec.capabilities),
            }
        )
    return rows


def image_ref_problem(ref: ModelRef, conn: Connection | None) -> str | None:
    """Why ``ref`` cannot be a picture model, in plain words; ``None`` when it can be.

    A ref on no connection, on a connection that draws nothing (a chat-only kind), on an own
    engine whose connection has no address, or on a remote GPU worker naming anything but
    ``auto`` -- the worker cannot be told which model
    to load, so a pin to one would be a promise it cannot keep (§3.5).
    """
    if conn is None:
        return (
            f"The picture model {ref} cannot be used: there is no connection called "
            f"{ref.connection_id}."
        )
    spec = spec_for(conn.kind)
    if conn.unsupported or spec is None:
        return (
            f"The picture model {ref} cannot be used: the connection {conn.id} is of a kind "
            "this version does not support."
        )
    if "image_create" not in spec.capabilities:
        return (
            f"The picture model {ref} cannot be used: {conn.display_label} does not draw "
            "pictures. Choose a picture model from another connection."
        )
    if conn.kind in BASE_URL_KINDS and not (conn.base_url or "").strip():
        # An own engine with its address cleared (`PATCH /api/connections/comfyui
        # {"base_url": ""}`, a hand-edited file) is not connected anywhere. Its engine's
        # built-in address is not used in its place: that daemon may load another
        # checkpoint, and the picture would carry this ref's name (#2176, P6).
        return (
            f"The picture model {ref} cannot be used: the connection {conn.id} has no "
            f"address. Add its address to {conn.id} in Settings › Models."
        )
    if conn.kind == "remote_gpu" and ref.model != REMOTE_GPU_MODEL:
        return (
            f"The picture model {ref} cannot be used: the GPU server cannot be told which "
            f"model to draw with, so choose {conn.id}/{REMOTE_GPU_MODEL}, whatever it has loaded."
        )
    return None
