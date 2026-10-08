"""LLM connector factory resolving live providers from configuration (Principle 5 & Principle 6)."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from uclone_x.errors import LLMProviderError, LLMProviderNotConfiguredError
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.base import BaseLLMConnector, named_model
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.connectors.ollama import (
    OLLAMA_ENDPOINT_ENV_VARS,
    OllamaConnector,
    has_configured_ollama_endpoint,
)
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.connectors.saved_choice import (
    SAVED_PROVIDERS,
    SavedChoice,
    api_key_for,
    describe_saved_choice,
    read_saved_choice,
    same_provider,
    saved_choice_note,
    settings_data,
)
from uclone_x.llm.connectors.vllm import (
    VLLM_ENDPOINT_ENV_VARS,
    VLLMConnector,
    has_configured_vllm_endpoint,
)
from uclone_x.llm.providers import (
    IMAGE_ENGINE_KINDS,
    PROVIDERS,
    canonical_provider,
    env_key,
    env_model,
)
from uclone_x.llm.usage.gate import UsageGate, gate_if_paid

if TYPE_CHECKING:
    from uclone_x.tools.builtin.image import ImageEngineChoice

#: The hosted providers precedence step 3 auto-detects from a key variable, in its order.
_AUTO_DETECTED: tuple[str, ...] = ("openai", "anthropic", "gemini")

#: The credential variables precedence step 3 auto-detects from, in its order.
_CREDENTIAL_ENV_VARS: tuple[str, ...] = tuple(
    name for provider in _AUTO_DETECTED for name in PROVIDERS[provider].key_env_vars
)


def _set(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and bool(value.strip())


def _env_value(name: str) -> str:
    """A variable :func:`model_env_override` found set, stripped."""
    return os.environ[name].strip()


def model_env_override(provider: str | None) -> str | None:
    """The variable naming ``provider``'s model, when one is set, else ``None``.

    A variable outranks the saved model, as it outranks the saved provider: the dashboard
    has always let ``OLLAMA_MODEL`` win over its Settings file, and the factory and ``ucx
    run`` agree with it. The variables are the provider table's (``GEMINI_MODEL``,
    ``OPENAI_MODEL``, ``ANTHROPIC_MODEL``, ``VLLM_MODEL``, and Ollama's three).
    """
    found = env_model(provider)
    return found[1] if found is not None else None


def what_outranks_saved_choice(
    provider: str | None = None, base_url: str | None = None
) -> str | None:
    """What precedence steps 1-4 found ahead of the saved choice, named plainly, or ``None``.

    ``None`` means the saved choice is what the factory builds from. Otherwise the answer
    names the argument or the environment variable that wins, so a head can say which one
    to change rather than claim the saved choice is in use.
    """
    if provider is not None and provider.strip():
        return "the provider it was given"
    if _set("LLM_PROVIDER"):
        return "LLM_PROVIDER"
    for name in _CREDENTIAL_ENV_VARS:
        if _set(name):
            return name
    for name in (*OLLAMA_ENDPOINT_ENV_VARS, *VLLM_ENDPOINT_ENV_VARS):
        if _set(name):
            return name
    if has_configured_ollama_endpoint(base_url) or has_configured_vllm_endpoint(base_url):
        return "the address it was given"
    return None


def saved_choice_in_effect(
    provider: str | None = None, base_url: str | None = None, *, path: Path | None = None
) -> SavedChoice | None:
    """The saved choice ``create_llm_connector`` would build from, or ``None``.

    Precedence step 5 applies only when steps 1-4 name nothing: no ``provider`` argument,
    no ``LLM_PROVIDER``, no credential variable, no endpoint variable and no ``base_url``
    (:func:`what_outranks_saved_choice`). The factory and every head that reports *where*
    its model came from call this one function, so the report and the connector cannot
    disagree about whether the saved choice was used. ``path`` reads another settings file
    than the session root's (the dashboard keeps its own storage directory).
    """
    if what_outranks_saved_choice(provider, base_url) is not None:
        return None
    return read_saved_choice(path)


def saved_choice_notice(provider: str | None = None, model: str | None = None) -> str | None:
    """The line a head prints when its connector comes from the saved choice, else ``None``.

    Read before the connector is built, so a saved provider that then fails (an OpenAI
    choice with no key) has already been named as the reason that provider was tried.
    """
    saved = saved_choice_in_effect(provider)
    if saved is None:
        return None
    if model is None:
        variable = model_env_override(saved.provider)
        if variable is not None:
            return describe_saved_choice(saved, _env_value(variable), model_from=variable)
    return describe_saved_choice(saved, model)


def resolve_api_key(
    provider: str, api_key: str | None = None, data: Mapping[str, Any] | None = None
) -> str | None:
    """The key a connector for ``provider`` is built with, or ``None`` to let it refuse.

    The argument, then the provider's key variable, then the key saved for that provider
    in the settings file (``data``, the file's contents). The variable outranks the file
    (the environment is the override, never the storage); ``None`` is returned when a
    variable is set, so the connector reads it itself and names it in any refusal. A key
    saved for another provider is never returned.
    """
    if api_key is not None:
        return api_key
    if env_key(provider) is not None or data is None:
        return None
    return api_key_for(data, provider)


def resolve_deep_model(
    provider: str, model: str | None = None, data: Mapping[str, Any] | None = None
) -> str | None:
    """The deep model a connector for ``provider`` is built with, or ``None``.

    The argument, then the provider's model variable, then the settings file's default
    deep model -- only when its connection is of this kind, since a saved Gemini model means
    nothing to OpenAI. ``None`` leaves the connector with no model of its own: a request
    that names none is then refused before the network, in plain words.
    """
    from uclone_x.llm.connections import ModelRef, saved_connections, saved_default_models

    named = named_model(model)
    if named is not None:
        return named
    found = env_model(provider)
    if found is not None:
        return found[0]
    if data is None:
        return None
    deep = saved_default_models(data).deep
    if deep is None:
        return None
    ref = ModelRef.parse(deep)
    conn = next((c for c in saved_connections(data) if c.id == ref.connection_id), None)
    if conn is None or not same_provider(conn.kind, provider):
        return None
    return ref.model


def _connector_for_saved_choice(
    choice: SavedChoice,
    api_key: str | None,
    *,
    model: str | None,
    data: Mapping[str, Any],
    **kwargs: Any,
) -> BaseLLMConnector:
    """Build the provider a saved choice names, with its saved endpoint, key and model.

    A key or model the caller passed outranks the saved one, as an argument outranks the
    file everywhere else in this precedence; a model variable outranks the saved model.
    """
    if choice.provider not in SAVED_PROVIDERS:
        # Refused, naming the file: ignoring it would report "nothing is configured" to a
        # person who did configure something, and point them away from the real defect.
        raise LLMProviderError(
            f"The model choice saved in {choice.path} names a provider this version does "
            f"not support: {choice.provider}. Pick a model again in the dashboard's Settings."
        )
    key = api_key
    kind_row = choice.connection_id is None or choice.connection_id == choice.provider
    if key is None and not (kind_row and env_key(choice.provider)):
        # The connection's own key (S3): another row of the same kind keeps its own. A key
        # variable overrides only the row whose id is its kind (S4).
        key = choice.api_key
    if key is None and not kind_row:
        # A second row of a kind with no key of its own: refused, or built to send none,
        # but never handed the kind row's key or the kind's variable (S3).
        from uclone_x.llm.connections import Connection, connection_key

        key = connection_key(Connection(id=str(choice.connection_id), kind=choice.provider))
    return _construct(choice.provider, key, choice.base_url, model=model, data=data, **kwargs)


def _construct(
    provider: str,
    api_key: str | None,
    base_url: str | None,
    *,
    model: str | None,
    data: Mapping[str, Any],
    **kwargs: Any,
) -> BaseLLMConnector:
    """Build ``provider``'s connector with its resolved key and deep model.

    ``provider`` is already a name the provider table knows. The key and model come from
    :func:`resolve_api_key` and :func:`resolve_deep_model`, so every route into the
    factory -- an explicit name, ``LLM_PROVIDER``, a key variable, the saved choice --
    applies the same rule. The vLLM connector refuses construction when nothing names an
    endpoint, naming ``VLLM_BASE_URL``: that refusal is the point of naming the provider
    explicitly, rather than deferring it to a refused connection on the first turn (P6,
    #533).
    """
    provider_id = canonical_provider(provider) or provider
    if provider_id in IMAGE_ENGINE_KINDS:
        # An image engine is a connection, but it holds no conversation: refused rather
        # than built as anything that would answer (P6).
        raise LLMProviderError(
            f"{PROVIDERS[provider_id].display_name} draws pictures and cannot hold a "
            "conversation. Choose a conversation model from another connection."
        )
    key = resolve_api_key(provider_id, api_key, data)
    deep = resolve_deep_model(provider_id, model, data)
    if provider_id == "openai":
        return OpenAIConnector(api_key=key, base_url=base_url, model=deep, **kwargs)
    if provider_id == "anthropic":
        return AnthropicConnector(api_key=key, base_url=base_url, model=deep, **kwargs)
    if provider_id == "gemini":
        return GeminiConnector(api_key=key, base_url=base_url, model=deep, **kwargs)
    if provider_id == "ollama":
        return OllamaConnector(base_url=base_url, model=deep, **kwargs)
    if provider_id == "vllm":
        return VLLMConnector(api_key=key, base_url=base_url, model=deep, **kwargs)
    return MockLLMConnector(api_key=key, base_url=base_url, model=deep, **kwargs)


def create_llm_connector(
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    fallback_to_mock: bool = False,
    *,
    usage_gate: UsageGate | None = None,
    model: str | None = None,
    saved_choice_file: Path | None = None,
    **kwargs: Any,
) -> BaseLLMConnector:
    """Create an LLM provider connector from an explicit name or the environment.

    A connector whose ``paid`` is true comes back passed through the usage gate
    (``uclone_x.llm.usage.gate``): each call is checked against the user's limits on
    paid-model tokens first, and its tokens are recorded after. Every connector comes from
    here, so no caller can reach a paid provider around the gate
    (the token-gateway design). ``usage_gate`` names the gate, for a head that keeps its
    limits and usage outside the session root; by default the session root's are used.

    ``model`` is the deep model a request naming none is sent to; without it the provider's
    model variable, then the settings file's model, is used (``resolve_deep_model``).
    ``saved_choice_file`` is the settings file read for the saved choice and the key, for a
    head that keeps its own (the dashboard's storage directory); by default the session
    root's. Resolution is ``_build_connector``'s.
    """
    return gate_if_paid(
        _build_connector(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            fallback_to_mock=fallback_to_mock,
            model=model,
            saved_choice_file=saved_choice_file,
            **kwargs,
        ),
        usage_gate,
    )


def _build_connector(
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    fallback_to_mock: bool = False,
    *,
    model: str | None = None,
    saved_choice_file: Path | None = None,
    **kwargs: Any,
) -> BaseLLMConnector:
    """Resolve and construct the connector ``create_llm_connector`` returns, ungated.

    Resolution precedence:

    1. Explicit provider name (``openai``, ``anthropic``, ``gemini``/``google``,
       ``ollama``, ``vllm``, ``mock``).
    2. ``LLM_PROVIDER``.
    3. Auto-detection from a credential variable (``OPENAI_API_KEY``,
       ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``/``GOOGLE_API_KEY``).
    4. Auto-detection from a self-hosted endpoint variable: Ollama's
       (``OLLAMA_BASE_URL``, ``OLLAMA_FAST_BASE_URL``, ``LOCAL_LLM_BASE_URL``,
       ``OLLAMA_HOST``) or vLLM's (``VLLM_BASE_URL``), or from an explicit ``base_url``.
    5. The default deep model the person saved -- in the dashboard's Settings, or by
       ``ucx install`` / ``ucx start`` on a first setup -- and the connection its ref names
       (``default_models.deep`` and ``connections``), read from ``<session root>/settings.json``, or
       from ``saved_choice_file`` when one is given (``saved_choice.py``). Every step above outranks it, so a flag or a variable still
       wins; see ``saved_choice_in_effect``. A head that uses it says so. Its model is
       what a request naming no model gets, unless a model variable is set
       (``OLLAMA_MODEL``/``OLLAMA_INDEPTH_MODEL``/``OLLAMA_FAST_MODEL`` for Ollama,
       ``VLLM_MODEL`` for vLLM): a variable outranks the file for the model too, in the
       same order the dashboard applies (environment first, then its Settings file).
    6. When nothing above names a provider: ``MockLLMConnector`` if ``fallback_to_mock``,
       otherwise **``LLMProviderNotConfiguredError``**.

    Whichever step names the provider, its key and model are resolved the same way: the
    key from the argument, then the provider's key variable, then the key saved *for that
    connection* in the settings file; the deep model from ``model``, then the provider's
    model variable, then the saved default deep model when its connection is this kind. A
    variable outranks the file, as the dashboard has always applied it.

    Step 4 previously said "or the default ``http://localhost:11434``", and the code did not
    read those variables at all — it fell through to ``OllamaConnector`` unconditionally, and
    the connector applied the default itself. Both halves were wrong in the same direction
    (#533): an unconfigured installation built a connector that could not work, and the
    resulting failure arrived at turn time as a refused connection rather than at composition
    as "no provider is configured". The factory now performs the detection step 4 describes,
    and refuses when it finds nothing.

    There is no ``try``/``except`` around the construction, and that absence is the point
    (P6, #397). The removed handler did two things, and both defeated #385/PR #393:

    * With ``fallback_to_mock=True`` it returned a ``MockLLMConnector`` when construction
      raised — a **substituted implementation** answering for a provider the caller
      actually asked for, announced by nothing but a ``logger.warning``. P6's exemption is
      in-band attribution via ``provenance`` on the result envelope, and
      ``MockLLMConnector`` cannot supply it: it stamps
      ``path=PRIMARY, requested=served_by={provider: "mock"}, degraded=False``, so it does
      not merely fail to record the substitution — it overwrites the record of what was
      requested. A log line is not in band, and there is no version of a mock standing in
      for OpenAI that P6 permits.
    * With ``fallback_to_mock=False``, an intervening handler or redundant pre-checks
      risked misclassifying connector errors or duplicating connector validation.
      ``LLMCredentialsNotConfiguredError`` is a **sibling** of ``LLMProviderError`` under
      ``LLMError``, not a subclass. Each keyed connector validates its own credentials
      in ``__init__`` and names the variables it consulted. By removing both the wrapping
      handler and the pre-checks, the connector's error now reaches the caller unaltered.

    The per-provider "key is required" pre-checks are gone for the same reason: each keyed
    connector validates its own credentials in ``__init__`` and names the variables it
    consulted, so the pre-checks duplicated connector logic and added no information.

    ``fallback_to_mock`` survives at exactly one site — precedence step 6 — and it is not a
    fallback there. Nothing has failed and nothing is substituted for a value some failed
    operation was supposed to produce: the environment names no provider and holds no
    credential, so the caller's flag selects, in advance, which connector to build in place
    of the ``OllamaConnector`` default. It cannot be reached by a credential failure, an
    unsupported provider name, or a connector that refused to construct, because those all
    raise before this point.
    """
    resolved_provider = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()
    if not resolved_provider:
        # Any set variable selects its provider, blanks included: the person named that
        # provider, so its connector refuses the blank key rather than a mock answering.
        resolved_provider = next(
            (p for p in _AUTO_DETECTED if any(os.getenv(v) for v in PROVIDERS[p].key_env_vars)),
            "",
        )
    data = settings_data(saved_choice_file)

    if canonical_provider(resolved_provider) is not None:
        return _construct(resolved_provider, api_key, base_url, model=model, data=data, **kwargs)

    if resolved_provider:
        # An unmappable provider name is refused, naming the offending value, whatever
        # `fallback_to_mock` says. A mock returned here answers a request this factory
        # could not understand, which is the substitution P6 forbids rather than the
        # unconfigured-environment default below.
        raise LLMProviderError(f"Unsupported LLM provider: {resolved_provider}")

    # Step 4 before step 5, as the precedence above states. #539 added this detection below
    # the flag, so a caller with `OLLAMA_BASE_URL` set and `fallback_to_mock=True` received
    # a mock in place of the provider they had configured — a substituted implementation
    # with no in-band attribution, which is exactly what the paragraphs above forbid. The
    # flag is the unconfigured-case default; a configured endpoint is not the unconfigured
    # case.
    if has_configured_ollama_endpoint(base_url):
        return _construct("ollama", api_key, base_url, model=model, data=data, **kwargs)

    # After Ollama, deliberately. Both detectors answer True for any explicit `base_url`,
    # so a caller who passes one without naming a provider would otherwise change provider
    # with this line's position — and Ollama is what that call has always built. An
    # environment that sets both `OLLAMA_BASE_URL` and `VLLM_BASE_URL` is ambiguous, and the
    # factory does not resolve an ambiguity by guessing: it keeps the pre-existing answer,
    # and `LLM_PROVIDER=vllm` is how the other one is chosen.
    if has_configured_vllm_endpoint(base_url):
        return _construct("vllm", api_key, base_url, model=model, data=data, **kwargs)

    # Step 5, above the flag for the reason step 4 is: a saved choice is the person's
    # configuration, not the unconfigured case the flag decides.
    saved = saved_choice_in_effect(provider, base_url, path=saved_choice_file)
    if saved is not None:
        return _connector_for_saved_choice(saved, api_key, model=model, data=data, **kwargs)

    if fallback_to_mock:
        return MockLLMConnector(api_key=api_key, base_url=base_url, model=model, **kwargs)

    # Nothing named a provider, so nothing is built. Returning an `OllamaConnector` here
    # would succeed — it has no credential to validate — and defer the failure to the first
    # turn, where "connection refused" names a socket instead of the configuration defect
    # that caused it (P6). See `LLMProviderNotConfiguredError`.
    raise LLMProviderNotConfiguredError(
        "No LLM provider is configured. Name one explicitly, or set one of: "
        "LLM_PROVIDER=openai|anthropic|gemini|ollama|vllm; "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY or GOOGLE_API_KEY; "
        f"an Ollama endpoint in one of {', '.join(OLLAMA_ENDPOINT_ENV_VARS)}; "
        f"or a vLLM endpoint in {', '.join(VLLM_ENDPOINT_ENV_VARS)}. "
        f"{saved_choice_note(saved_choice_file)} "
        "`ucx install` sets up a local model and saves it; in the dashboard (`ucx start`), "
        "pick a model in Settings. From a terminal, `ucx llm status` reports what is reachable."
    )


def cloud_image_models() -> list[str]:
    """The cloud picture models the model registry declares, in its order (§3.5).

    A registry entry is what makes a Google model a picture model; a model a listing names
    without one is never offered or picked by `auto`.
    """
    from uclone_x.tools.builtin.media_registry import ModelRegistry

    return [p.model_id for p in ModelRegistry().profiles() if p.engine_type == "gemini"]


def image_engine_choice(
    settings_path: Path | None = None,
    own: str | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    gpu_tunnel_comfy_port: int | None = None,
) -> ImageEngineChoice:
    """The picture model one draw follows, read from ``settings_path`` now (model-gateway §3.5).

    ``own`` is the asking clone's picture model (a ref or `auto`); `None` or empty follows
    ``default_models.image``. The image engines are the connections: the first `comfyui`
    connection's address and the first `remote_gpu` one's are where `auto` looks, and a ref
    names one connection. Under `auto` the cloud model is the first one the registry
    declares, on the first Google connection that has a key of its own (S3) -- whatever
    the default chat model is. A ref that cannot be served is carried as ``refusal``, in
    plain words, and never replaced (G7). ``gpu_tunnel_comfy_port`` is the local port the
    connected remote-GPU tunnel forwards to its ComfyUI; `None` for a head without one.
    """
    from uclone_x.llm.connections import ModelRef, image_ref_problem, is_model_ref
    from uclone_x.llm.gateway import connections_in_effect, default_models_in_effect
    from uclone_x.tools.builtin.image import IMAGE_AUTO, ImageEngineChoice
    from uclone_x.tools.builtin.media_registry import COMFYUI_ENGINE_TYPES, ModelRegistry

    data = settings_data(settings_path)
    conns = connections_in_effect(data)
    clean_own = own.strip() if own and own.strip() else None
    chosen = clean_own or default_models_in_effect(data, conns).image or IMAGE_AUTO
    chosen_by: Any = "clone" if clean_own else "default"
    comfy = next((c for c in conns if c.kind == "comfyui" and c.base_url), None)
    remote = next((c for c in conns if c.kind == "remote_gpu" and c.base_url), None)
    base: dict[str, Any] = {
        "chosen": chosen,
        "chosen_by": chosen_by,
        "from_connections": True,
        "gpu_tunnel_comfy_port": gpu_tunnel_comfy_port,
    }
    if chosen == IMAGE_AUTO:
        cloud = next((c for c in conns if c.kind == "gemini" and c.key), None)
        models = cloud_image_models()
        gemini = None
        if cloud is not None and models:
            gemini = GeminiConnector(
                api_key=cloud.key, base_url=cloud.base_url, http_client=http_client
            )
        return ImageEngineChoice(
            **base,
            comfyui_base_url=comfy.base_url if comfy is not None else None,
            remote_url=remote.base_url if remote is not None else None,
            gemini=gemini,
            gemini_model=models[0] if gemini is not None else None,
            gemini_connection=cloud.id if gemini is not None and cloud is not None else None,
        )
    if not is_model_ref(chosen):
        # Unreachable through Settings, which refuses it; read from a hand-edited file.
        return ImageEngineChoice(
            **base,
            refusal=(
                f"The picture model {chosen!r} does not say which connection it is on. "
                "Choose the picture model again in Settings › Models."
            ),
        )
    ref = ModelRef.parse(chosen)
    conn = next((c for c in conns if c.id == ref.connection_id), None)
    problem = image_ref_problem(ref, conn)
    if problem is not None or conn is None:
        return ImageEngineChoice(**base, refusal=problem)
    if conn.kind == "gemini":
        if not conn.key:
            return ImageEngineChoice(
                **base,
                pin="gemini",
                refusal=(
                    f"The picture model {ref} cannot be used: the connection {conn.id} has "
                    f"no key. Add the key to {conn.id} in Settings › Models."
                ),
            )
        client = GeminiConnector(api_key=conn.key, base_url=conn.base_url, http_client=http_client)
        return ImageEngineChoice(
            **base, pin="gemini", gemini=client, gemini_model=ref.model, gemini_connection=conn.id
        )
    if conn.kind == "remote_gpu":
        return ImageEngineChoice(**base, pin="remote_gpu", remote_url=conn.base_url)
    profile = ModelRegistry().resolve(ref.model)
    if profile.model_id != ref.model or profile.engine_type not in COMFYUI_ENGINE_TYPES:
        return ImageEngineChoice(
            **base,
            pin="comfyui",
            refusal=(
                f"The picture model {ref} cannot be used: {ref.model} is not a picture "
                "model this version knows. Choose another picture model in Settings › Models."
            ),
        )
    return ImageEngineChoice(
        **base, pin="comfyui", pinned_profile=profile.model_id, comfyui_base_url=conn.base_url
    )


def bind_image_engine_settings(
    tool: object,
    settings_path: Path | None = None,
    gpu_tunnel_comfy_port: Callable[[], int | None] | None = None,
) -> None:
    """Point ``tool``'s picture settings at ``settings_path``, re-read on every draw.

    ``tool`` is whatever a registry holds as ``generate_image``; anything else is left as
    it is. Each draw passes the asking clone's own picture model, so a clone that names one
    draws with it and every other follows the default (§3.4). ``gpu_tunnel_comfy_port``
    answers the local port of the remote-GPU tunnel's ComfyUI while it is connected; only
    the dashboard has that tunnel.
    """
    from uclone_x.tools.builtin.image import GenerateImageTool

    if not isinstance(tool, GenerateImageTool):
        return

    def current(own: str | None) -> ImageEngineChoice:
        tunnel_port = gpu_tunnel_comfy_port() if gpu_tunnel_comfy_port is not None else None
        return image_engine_choice(settings_path, own, gpu_tunnel_comfy_port=tunnel_port)

    tool.bind_engine_settings(current)
