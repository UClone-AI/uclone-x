"""The model gateway: many connections, one model set, a connector per ref (model-gateway §3.3).

`ModelGateway` reads the connections and default models from the settings file on every
call (the file is the one source, S1), so a connection added in Settings or by `ucx key set`
while a head runs is seen at the next call.

* `connections()` -- the saved rows with the environment's overrides (S4);
* `defaults()` -- the default refs (`deep`, `fast`, `image`);
* `resolve(own, slot)` -- the one rule for which model a call uses: a clone's own ref wins,
  an empty slot takes the default, an empty fast default means deep;
* `connector_for(ref)` -- a connector for the ref's connection, built through
  `create_llm_connector` so `gate_if_paid` wraps every paid one (llm-token-gateway G8),
  cached per connection and dropped when that connection's row changes;
* `model_set(capability)` -- the union of every connection's live listing. A connection that
  cannot list appears as a group with its status and reason, never with a remembered list
  (provider-model-catalog G3).

A ref that cannot be served -- its connection removed -- is never replaced by the default
(§3.6, G7): the seat gets a connector that refuses every call with a plain sentence naming
the model and the cause.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

from uclone_x.core.models import AgentLLMConfig
from uclone_x.errors import ProviderFailureError, ProviderFailureKind
from uclone_x.llm.catalog import CatalogCache, CatalogEntry, CatalogResult, key_fingerprint
from uclone_x.llm.connections import (
    BASE_URL_KINDS,
    IMAGE_AUTO,
    REMOTE_GPU_MODEL,
    Connection,
    DefaultModels,
    ModelRef,
    Slot,
    connection_key,
    saved_connections,
    saved_default_models,
)
from uclone_x.llm.connectors.base import is_local_endpoint
from uclone_x.llm.connectors.ollama import OLLAMA_ENDPOINT_ENV_VARS, resolve_ollama_base_url
from uclone_x.llm.connectors.saved_choice import settings_data, settings_file
from uclone_x.llm.connectors.vllm import VLLM_ENDPOINT_ENV_VARS
from uclone_x.llm.model_listing import (
    CATALOG_PROVIDERS,
    LocalKeyRefusedError,
    list_local_models,
    list_ollama_entries,
    read_provider_catalog,
    vllm_request_headers,
)
from uclone_x.llm.model_policy import recommend
from uclone_x.llm.models import LLMRequest, ModelResponse, StreamChunk
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.llm.providers import (
    IMAGE_ENGINE_KINDS,
    PROVIDERS,
    canonical_provider,
    chat_kind,
    env_key,
    env_model,
    spec_for,
)
from uclone_x.llm.usage.gate import UsageGate

logger = logging.getLogger(__name__)

Capability = Literal["chat", "image"]
ConnectionStatus = Literal[
    "connected", "no_key", "key_rejected", "unreachable", "unchecked", "unsupported"
]

#: A catalogue status as a connection status (§3.7.1). `no_listing` -- the server answered,
#: but not with a model list -- is a connection that works and lists nothing.
_STATUS_OF: Final[dict[str, ConnectionStatus]] = {
    "live": "connected",
    "no_listing": "connected",
    "no_key": "no_key",
    "key_rejected": "key_rejected",
    "unreachable": "unreachable",
}

_UNREACHABLE_LOCAL: Final = (
    "Couldn't get an answer from {label}. Check that it is running and that the address is right."
)
_KEY_REFUSED_LOCAL: Final = "{label} is running, but it did not accept the key."
_UNSUPPORTED: Final = (
    "This kind of connection is not supported: {label} is saved as {kind}, which this "
    "version cannot use. Remove it, or use a version that supports it."
)


def unsupported_listing(conn: Connection) -> Listing:
    """What a row of a kind this build does not know says, in place of a listing (P6)."""
    return Listing(
        "unsupported",
        _UNSUPPORTED.format(
            label=conn.display_label, kind=repr(conn.kind) if conn.kind else "no kind"
        ),
    )


# -- The environment's overrides (settings-single-source S4, as revised by §3.2) --


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


#: The variables that name the remote GPU worker's address, in the order they are read.
REMOTE_GPU_ENV_VARS: Final = ("UCX_IMAGE_REMOTE_URL", "UCX_MEDIA_REMOTE_URL")


def _env_endpoint(kind: str) -> str | None:
    """The address the environment names for ``kind``, if any."""
    if kind == "remote_gpu":
        return next((v for name in REMOTE_GPU_ENV_VARS if (v := _text(os.getenv(name)))), None)
    if kind == "ollama":
        if any(_text(os.getenv(name)) for name in OLLAMA_ENDPOINT_ENV_VARS):
            return resolve_ollama_base_url()
        return None
    if kind == "vllm":
        for name in VLLM_ENDPOINT_ENV_VARS:
            if (value := _text(os.getenv(name))) is not None:
                return value
        return None
    spec = spec_for(kind)
    if spec is not None and spec.base_url_env:
        return _text(os.getenv(spec.base_url_env))
    return None


def _env_endpoint_variable(kind: str) -> str | None:
    """The variable that names ``kind``'s address, when one is set."""
    names: tuple[str, ...]
    if kind == "remote_gpu":
        names = REMOTE_GPU_ENV_VARS
    elif kind == "ollama":
        names = OLLAMA_ENDPOINT_ENV_VARS
    elif kind == "vllm":
        names = VLLM_ENDPOINT_ENV_VARS
    else:
        spec = spec_for(kind)
        names = (spec.base_url_env,) if spec is not None and spec.base_url_env else ()
    return next((name for name in names if _text(os.getenv(name)) is not None), None)


def env_provider_kind() -> str | None:
    """The chat kind ``LLM_PROVIDER`` names, or ``None`` (an image engine holds no chat)."""
    kind = canonical_provider(os.getenv("LLM_PROVIDER"))
    return kind if chat_kind(kind) else None


def connections_in_effect(data: Mapping[str, Any]) -> list[Connection]:
    """The saved rows with the environment's overrides applied (S4, §3.2).

    The environment shadows a row field by field, so ``LLM_PROVIDER=gemini`` with a key
    saved for the ``gemini`` row keeps that key; a row the environment made or shadowed is
    reported with ``source="env"`` and cannot be edited or removed in Settings.
    """
    rows = {conn.id: conn for conn in saved_connections(data)}
    order = list(rows)
    provider_kind = env_provider_kind()
    for kind in PROVIDERS:
        endpoint = _env_endpoint(kind) if kind in BASE_URL_KINDS else None
        key = env_key(kind)
        named = provider_kind == kind
        if not (named or endpoint or key):
            continue
        saved = rows.get(kind)
        if saved is not None and saved.kind != kind:
            continue  # a row of another kind under this id: the environment does not own it
        base = saved or Connection(id=kind, kind=kind, source="env")
        changes: dict[str, Any] = {}
        if named or endpoint:
            changes["source"] = "env"
            cloud_endpoint = None if kind in BASE_URL_KINDS else _env_endpoint(kind)
            if endpoint or cloud_endpoint:
                changes["base_url"] = endpoint or cloud_endpoint
            elif kind == "ollama" and base.base_url is None:
                changes["base_url"] = resolve_ollama_base_url()
        if key is not None:
            changes["key"], changes["key_env_var"] = key
        if changes.get("source", base.source) == "env":
            # The one variable to change or unset for this row (§3.7.1 `env_var`).
            changes["env_var"] = (
                "LLM_PROVIDER"
                if named
                else _env_endpoint_variable(kind) or (key[1] if key is not None else None)
            )
        rows[kind] = replace(base, **changes)
        if kind not in order:
            order.append(kind)
    return [rows[conn_id] for conn_id in order]


def default_models_in_effect(
    data: Mapping[str, Any], connections: Sequence[Connection] | None = None
) -> DefaultModels:
    """The saved defaults, with a model variable overriding ``deep`` (S4, §3.2).

    The variable read is ``LLM_PROVIDER``'s kind's, else that of the saved deep ref's
    connection, else the first kind with a connection whose model variable is set. Its
    bare id is read as ``<kind>/<id>``.
    """
    saved = saved_default_models(data)
    rows = connections if connections is not None else connections_in_effect(data)
    by_id = {conn.id: conn for conn in rows}
    candidates: list[str] = []
    if (named := env_provider_kind()) is not None:
        candidates.append(named)
    if saved.deep is not None:
        deep_conn = by_id.get(ModelRef.parse(saved.deep).connection_id)
        if deep_conn is not None:
            candidates.append(deep_conn.kind)
    candidates.extend(kind for kind in PROVIDERS if kind in by_id)
    for kind in candidates:
        found = env_model(kind)
        if found is not None:
            model, variable = found
            return replace(saved, deep=f"{kind}/{model}", env_vars={"deep": variable})
    return saved


class ModelRefUnavailableError(ProviderFailureError):
    """A ref whose connection is not there any more (§3.6): refused, never re-routed."""

    kind = ProviderFailureKind.MODEL_UNAVAILABLE
    retryable = False

    def __init__(self, ref: ModelRef) -> None:
        super().__init__(
            f"The model {ref} cannot be used: there is no connection called "
            f"{ref.connection_id} any more.",
            provider=ref.connection_id,
            model=ref.model,
        )
        self.ref = ref


class UnsupportedConnectionError(ProviderFailureError):
    """A ref on a connection of a kind this build does not know: refused, never re-routed."""

    kind = ProviderFailureKind.MODEL_UNAVAILABLE
    retryable = False

    def __init__(self, ref: ModelRef, conn: Connection) -> None:
        super().__init__(
            f"The model {ref} cannot be used: the connection {conn.id} is of a kind this "
            f"version does not support ({conn.kind or 'no kind given'}).",
            provider=ref.connection_id,
            model=ref.model,
        )
        self.ref = ref


class RefusingConnector:
    """A seat's connector for a ref that cannot be served: every call raises `error`.

    The turn then fails through the ordinary provider-failure path, with the error's plain
    sentence, and nothing is sent anywhere else in its place (G7).
    """

    def __init__(self, error: Exception, provider: str) -> None:
        self._error = error
        self._provider = provider
        self.paid = False

    @property
    def provider_name(self) -> str:
        return self._provider

    async def generate(self, request: LLMRequest) -> ModelResponse:
        raise self._error

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        raise self._error
        yield  # pragma: no cover - makes this an async generator


@dataclass(frozen=True)
class ModelEntry:
    """One model in the set, under its ref."""

    ref: str
    id: str
    display_name: str | None
    capabilities: tuple[str, ...]
    context_window: int | None

    def as_json(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "id": self.id,
            "display_name": self.display_name,
            "capabilities": list(self.capabilities),
            "context_window": self.context_window,
        }


@dataclass(frozen=True)
class ModelGroup:
    """One connection's part of the set: its models, or why there are none."""

    connection_id: str
    label: str
    kind: str
    status: ConnectionStatus
    detail: str | None
    models: tuple[ModelEntry, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "label": self.label,
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "models": [entry.as_json() for entry in self.models],
        }


@dataclass(frozen=True)
class ModelSet:
    """The union of every connection's live listing (G2)."""

    groups: tuple[ModelGroup, ...]

    def find(self, ref: str) -> ModelEntry | None:
        return next((m for g in self.groups for m in g.models if m.ref == ref), None)

    def group(self, connection_id: str) -> ModelGroup | None:
        return next((g for g in self.groups if g.connection_id == connection_id), None)


@dataclass(frozen=True)
class Listing:
    """What a connection's own listing said, as last read."""

    status: ConnectionStatus
    detail: str | None
    entries: tuple[CatalogEntry, ...] = ()
    recommended: str | None = None


@dataclass(frozen=True)
class DefaultBinding:
    """A connector that answers every slot following the system default, with its models.

    For a head that resolved its own connector (a terminal command's `--provider`, an
    environment variable, a test's fake): such a clone runs there, and only a clone that
    names its own ref goes through the gateway's connections.
    """

    connector: LLMProviderProtocol | None
    deep: str | None = None
    fast: str | None = None


@dataclass(frozen=True)
class SeatBinding:
    """What one clone runs on: its connector, and its config carrying bare model ids."""

    llm: LLMProviderProtocol | None
    llm_config: AgentLLMConfig
    #: The clone's own deep ref when it names one; `None` when it follows the default.
    pinned_ref: str | None = None
    #: The ref the turns go to, when one resolved.
    deep_ref: str | None = None


def connection_paid(conn: Connection) -> bool:
    """Whether calls on ``conn`` may cost money (§3.8): never a local or private host."""
    if conn.unsupported or conn.kind in ("ollama", "mock") or conn.kind in IMAGE_ENGINE_KINDS:
        # The image engines are the person's own machines (a ComfyUI, the GPU server).
        return False
    if conn.kind == "vllm":
        return not is_local_endpoint(conn.base_url)
    return not is_local_endpoint(conn.base_url) if conn.base_url else True


def _row_fingerprint(conn: Connection) -> tuple[str, ...]:
    return (
        conn.kind,
        conn.base_url or "",
        key_fingerprint(conn.key),
        conn.source,
        conn.key_env_var or "",
    )


ConnectorFactory = Callable[..., Any]


class ModelGateway:
    """Many connections, one model set, a connector per ref (model-gateway §3.3)."""

    def __init__(
        self,
        settings_path: Path | None = None,
        *,
        usage_gate: UsageGate | None = None,
        catalog: CatalogCache | None = None,
        default_binding: DefaultBinding | None = None,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        self._settings_path = settings_path
        self._usage_gate = usage_gate
        self._catalog = catalog if catalog is not None else CatalogCache()
        self._default_binding = default_binding
        self._factory = connector_factory
        #: connection id -> (its row's fingerprint, model id -> connector).
        self._connectors: dict[str, tuple[tuple[str, ...], dict[str, LLMProviderProtocol]]] = {}
        #: connection id -> (its row's fingerprint, the last listing read for it).
        self._listings: dict[str, tuple[tuple[str, ...], Listing]] = {}

    # -- What is saved --

    @property
    def settings_path(self) -> Path:
        return self._settings_path if self._settings_path is not None else settings_file()

    @property
    def default_binding(self) -> DefaultBinding | None:
        return self._default_binding

    def set_default_binding(self, binding: DefaultBinding | None) -> None:
        self._default_binding = binding

    def _data(self) -> dict[str, Any]:
        return settings_data(self.settings_path)

    def connections(self) -> tuple[Connection, ...]:
        """Every connection, saved rows first, with the environment's overrides."""
        return tuple(connections_in_effect(self._data()))

    def connection(self, conn_id: str) -> Connection | None:
        return next((c for c in self.connections() if c.id == conn_id), None)

    def defaults(self) -> DefaultModels:
        """The default refs, with a model variable overriding `deep` (S4)."""
        data = self._data()
        return default_models_in_effect(data, connections_in_effect(data))

    # -- The one rule --

    def resolve(self, own: str | None, slot: Slot) -> ModelRef | None:
        """The ref a call uses: ``own`` when it names one, else the default for ``slot``.

        An empty fast default means deep. ``None`` when nothing names a model (no default
        saved), and for the picture default ``auto``, which step 5 resolves. ``own`` must be
        a ref: a bare id raises `ModelRefError` (it is refused when a persona loads, so only
        a caller that skipped that check can reach this).
        """
        if own and own.strip():
            if slot == "image" and own.strip() == IMAGE_AUTO:
                return None
            return ModelRef.parse(own)
        chosen = self.defaults().get(slot)
        if chosen is None or chosen == IMAGE_AUTO:
            return None
        return ModelRef.parse(chosen)

    # -- Connectors --

    def connector_for(self, ref: ModelRef) -> LLMProviderProtocol:
        """A connector for ``ref``'s connection; `ModelRefUnavailableError` when it has none.

        Built through `create_llm_connector` (so a paid one is gated), with the connection's
        own key and address and ``ref.model`` as its default model. Kept per connection,
        and dropped as soon as that connection's row changes.
        """
        conn = self.connection(ref.connection_id)
        if conn is None:
            self._connectors.pop(ref.connection_id, None)
            raise ModelRefUnavailableError(ref)
        if conn.unsupported:
            raise UnsupportedConnectionError(ref, conn)
        fingerprint = _row_fingerprint(conn)
        kept = self._connectors.get(conn.id)
        if kept is None or kept[0] != fingerprint:
            fresh: dict[str, LLMProviderProtocol] = {}
            kept = (fingerprint, fresh)
            self._connectors[conn.id] = kept
        built = kept[1].get(ref.model)
        if built is None:
            built = self._build(conn, ref.model)
            kept[1][ref.model] = built
        return built

    def _build(self, conn: Connection, model: str) -> LLMProviderProtocol:
        factory = self._factory
        if factory is None:
            from uclone_x.llm.connectors.factory import create_llm_connector

            factory = create_llm_connector
        connector: LLMProviderProtocol = factory(
            provider=conn.kind,
            # Its own key only (S3); `NO_KEY` for a keyless server, so neither the factory nor
            # the connector looks for one elsewhere.
            api_key=connection_key(conn),
            base_url=conn.base_url,
            model=model,
            usage_gate=self._usage_gate,
            saved_choice_file=self._settings_path,
        )
        return connector

    def _connector_or_refusal(self, ref: ModelRef) -> LLMProviderProtocol:
        try:
            return self.connector_for(ref)
        except (ModelRefUnavailableError, UnsupportedConnectionError) as exc:
            return RefusingConnector(exc, ref.connection_id)
        except Exception as exc:  # a connector that refused construction (no key, no address)
            logger.info("The connector for %s could not be built: %s", ref, exc)
            return RefusingConnector(exc, ref.connection_id)

    def default_fast(self) -> tuple[LLMProviderProtocol | None, str | None]:
        """The connector and model for calls no single clone owns (room routing, §3.4)."""
        binding = self._default_binding
        if binding is not None:
            return binding.connector, binding.fast or binding.deep
        ref = self.resolve(None, "fast")
        if ref is None:
            return None, None
        return self._connector_or_refusal(ref), ref.model

    def default_deep(self) -> tuple[LLMProviderProtocol | None, str | None]:
        """The connector and model of the default deep ref, for a call on the person's behalf."""
        binding = self._default_binding
        if binding is not None:
            return binding.connector, binding.deep
        ref = self.resolve(None, "deep")
        if ref is None:
            return None, None
        return self._connector_or_refusal(ref), ref.model

    def bind(self, own: AgentLLMConfig) -> SeatBinding:
        """The connector and models a clone with config ``own`` runs on (§3.4).

        Its own deep ref wins, else the default deep. Its fast model is its own ref, else
        the default fast; it is carried only when it is on the deep model's connection,
        since one seat holds one connector. A ref that cannot be served gets a refusing
        connector, never the default's (§3.6).
        """
        binding = self._default_binding
        pinned = own.model_name.strip() if own.model_name and own.model_name.strip() else None
        deep_conn_id: str | None = None
        deep_ref: str | None = None
        llm: LLMProviderProtocol | None
        if pinned is not None and binding is not None and "/" not in pinned:
            # A bare id is a terminal command's `--model` for its own connector: not a ref,
            # and never looked up on a connection.
            llm, deep_model, pinned = binding.connector, pinned, None
        elif pinned is not None:
            ref = ModelRef.parse(pinned)
            llm, deep_model, deep_conn_id, deep_ref = (
                self._connector_or_refusal(ref),
                ref.model,
                ref.connection_id,
                str(ref),
            )
        elif binding is not None:
            llm, deep_model = binding.connector, binding.deep
        else:
            ref = self.resolve(None, "deep")
            if ref is None:
                llm, deep_model = None, None
            else:
                llm, deep_model, deep_conn_id, deep_ref = (
                    self._connector_or_refusal(ref),
                    ref.model,
                    ref.connection_id,
                    str(ref),
                )
        fast_model: str | None
        own_fast = own.fast_model.strip() if own.fast_model and own.fast_model.strip() else None
        if own_fast is not None:
            fast_ref = ModelRef.parse(own_fast)
            fast_model = fast_ref.model if fast_ref.connection_id == deep_conn_id else None
        elif binding is not None and pinned is None:
            fast_model = binding.fast or binding.deep
        else:
            fast_ref = self.resolve(None, "fast")
            fast_model = (
                fast_ref.model
                if fast_ref is not None and fast_ref.connection_id == deep_conn_id
                else None
            )
        config = own.model_copy(update={"model_name": deep_model, "fast_model": fast_model})
        return SeatBinding(llm=llm, llm_config=config, pinned_ref=pinned, deep_ref=deep_ref)

    # -- The model set --

    async def listing(self, conn: Connection, *, refresh: bool = False) -> Listing:
        """What ``conn``'s own listing says now (cloud listings are cached, §5)."""
        if refresh:
            self._catalog.clear()
        label = conn.display_label
        if conn.unsupported:
            listing = unsupported_listing(conn)
        elif conn.kind in CATALOG_PROVIDERS:
            result = await read_provider_catalog(
                conn.kind,
                base_url=conn.base_url,
                api_key=conn.key,
                cache=self._catalog,
                cache_id=conn.id,
            )
            listing = _listing_of(result) if result is not None else Listing("unreachable", None)
        elif conn.kind == "ollama":
            # Each entry says whether it chats (Ollama's embedders do not, #2167); `_group`
            # leaves the ones that cannot out of the conversation set.
            entries = await list_ollama_entries(conn.base_url)
            unreachable = _UNREACHABLE_LOCAL.format(label=label)
            listing = (
                Listing("unreachable", unreachable)
                if entries is None
                else Listing("connected", None, tuple(entries))
            )
        elif conn.kind == "vllm":
            try:
                found = await list_local_models(
                    conn.kind,
                    conn.base_url,
                    vllm_headers=vllm_request_headers(conn.key, env_fallback=False),
                    raise_on_refused_key=True,
                )
            except LocalKeyRefusedError:
                listing = Listing("key_rejected", _KEY_REFUSED_LOCAL.format(label=label))
            else:
                listing = (
                    Listing("unreachable", _UNREACHABLE_LOCAL.format(label=label))
                    if found is None
                    else Listing(
                        "connected", None, tuple(CatalogEntry(id=model) for model in found)
                    )
                )
        elif conn.kind == "comfyui":
            listing = await _comfyui_listing(conn)
        elif conn.kind == "remote_gpu":
            listing = await _remote_gpu_listing(conn)
        elif conn.kind == "mock":
            found = await list_local_models("mock")
            listing = Listing("connected", None, tuple(CatalogEntry(id=m) for m in found or ()))
        else:
            listing = Listing("unreachable", _UNREACHABLE_LOCAL.format(label=conn.display_label))
        self._listings[conn.id] = (_row_fingerprint(conn), listing)
        return listing

    def last_listing(self, conn: Connection) -> Listing | None:
        """The listing last read for ``conn``, while its row is unchanged; else ``None``."""
        kept = self._listings.get(conn.id)
        if kept is None or kept[0] != _row_fingerprint(conn):
            return None
        return kept[1]

    async def model_set(
        self, capability: Capability = "chat", *, refresh: bool = False
    ) -> ModelSet:
        """Every connection's live listing for ``capability``, one group per connection."""
        # A connection is in the set for what its kind can do: a picture engine is never
        # asked for conversation models, and a chat-only one never for pictures.
        wanted = "chat" if capability == "chat" else "image_create"
        conns = tuple(
            conn
            for conn in self.connections()
            if conn.unsupported
            or wanted in (spec.capabilities if (spec := spec_for(conn.kind)) else ())
        )
        if refresh:
            self._catalog.clear()  # every connection is asked again
        listings = await asyncio.gather(*(self.listing(conn) for conn in conns))
        groups = tuple(
            _group(conn, listing, capability) for conn, listing in zip(conns, listings, strict=True)
        )
        return ModelSet(groups=groups)

    def recommended(self, models: ModelSet) -> dict[str, str | None]:
        """`model_policy`'s pick over the union (§3.7.1): shown, never saved unasked."""
        deep: str | None = None
        for group in models.groups:
            if group.status != "connected" or not group.models:
                continue
            entries = [CatalogEntry(id=m.id) for m in group.models]
            pick = recommend(group.kind, entries)
            if pick is not None:
                deep = f"{group.connection_id}/{pick}"
                break
        return {"deep": deep, "fast": None}


_COMFYUI_UNREACHABLE: Final = (
    "Couldn't get an answer from {label}. Check that ComfyUI is running and that the "
    "address is right."
)
_REMOTE_GPU_UNREACHABLE: Final = (
    "Couldn't get an answer from {label}. Check that the GPU server's picture worker is "
    "running and that the address is right."
)
#: What the remote GPU worker's one entry is called: it cannot say which model it loaded.
REMOTE_GPU_ENTRY_NAME: Final = "Whatever the GPU server has loaded"


async def _comfyui_listing(conn: Connection) -> Listing:
    """The checkpoints a ComfyUI daemon can load: its own listing (G6), never assumed."""
    from uclone_x.tools.builtin.comfy_client import ComfyClient

    client = ComfyClient(base_url=conn.base_url)
    try:
        found = await client.checkpoints()
    except Exception:  # an address the client cannot use reads as "not there"
        found = None
    finally:
        await client.aclose()
    if found is None:
        return Listing("unreachable", _COMFYUI_UNREACHABLE.format(label=conn.display_label))
    return Listing("connected", None, tuple(CatalogEntry(id=name) for name in found))


async def _remote_gpu_listing(conn: Connection) -> Listing:
    """Whether the remote GPU worker answers; it lists no models, so its one entry is auto."""
    from uclone_x.tools.builtin.image import RemoteCudaImageEngine

    try:
        alive = await RemoteCudaImageEngine(base_url=conn.base_url).is_available()
    except Exception:
        alive = False
    if not alive:
        return Listing("unreachable", _REMOTE_GPU_UNREACHABLE.format(label=conn.display_label))
    return Listing("connected", None, (CatalogEntry(id=REMOTE_GPU_MODEL),))


def _image_models(conn: Connection, listing: Listing) -> tuple[ModelEntry, ...]:
    """A connection's picture models (§3.5): declared by the registry and in its listing.

    * ComfyUI: each registered own model whose checkpoint file the daemon lists;
    * the GPU server: its one entry, ``<id>/auto``, since it cannot be told which model;
    * Google: each cloud picture model the registry declares that the listing names.
    """
    from uclone_x.tools.builtin.media_registry import COMFYUI_ENGINE_TYPES, ModelRegistry

    listed = {entry.id for entry in listing.entries}
    if conn.kind == "remote_gpu":
        if REMOTE_GPU_MODEL not in listed:
            return ()
        return (
            ModelEntry(
                ref=f"{conn.id}/{REMOTE_GPU_MODEL}",
                id=REMOTE_GPU_MODEL,
                display_name=REMOTE_GPU_ENTRY_NAME,
                capabilities=("image_create",),
                context_window=None,
            ),
        )
    profiles = ModelRegistry().profiles()
    if conn.kind == "comfyui":
        files = {Path(name).name for name in listed}
        chosen = [
            p
            for p in profiles
            if p.engine_type in COMFYUI_ENGINE_TYPES and p.filename and p.filename in files
        ]
    elif conn.kind == "gemini":
        chosen = [p for p in profiles if p.engine_type == "gemini" and p.model_id in listed]
    else:
        chosen = []
    return tuple(
        ModelEntry(
            ref=f"{conn.id}/{p.model_id}",
            id=p.model_id,
            display_name=p.display_name,
            capabilities=tuple(f"image_{c}" for c in p.capabilities),
            context_window=None,
        )
        for p in chosen
    )


def _listing_of(result: CatalogResult) -> Listing:
    return Listing(
        status=_STATUS_OF.get(result.status, "unreachable"),
        detail=result.detail,
        entries=result.entries,
        recommended=result.recommended,
    )


def _group(conn: Connection, listing: Listing, capability: Capability) -> ModelGroup:
    entries: Sequence[CatalogEntry]
    if capability == "chat":
        entries = [e for e in listing.entries if e.chat_capable]
        models = tuple(
            ModelEntry(
                ref=f"{conn.id}/{e.id}",
                id=e.id,
                display_name=e.display_name,
                capabilities=("chat", "image_input") if e.accepts_images else ("chat",),
                context_window=e.context_window,
            )
            for e in entries
        )
    else:
        models = _image_models(conn, listing)
    return ModelGroup(
        connection_id=conn.id,
        label=conn.display_label,
        kind=conn.kind,
        status=listing.status,
        detail=listing.detail,
        models=models,
    )


__all__ = [
    "Capability",
    "ConnectionStatus",
    "DefaultBinding",
    "Listing",
    "ModelEntry",
    "ModelGateway",
    "ModelGroup",
    "ModelRefUnavailableError",
    "ModelSet",
    "RefusingConnector",
    "SeatBinding",
    "UnsupportedConnectionError",
    "connection_paid",
    "unsupported_listing",
]
