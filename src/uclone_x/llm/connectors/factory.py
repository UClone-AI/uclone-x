"""LLM connector factory resolving live providers from configuration (Principle 5 & Principle 6)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from typing import Any

from uclone_x.errors import LLMProviderError, LLMProviderNotConfiguredError
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.base import BaseLLMConnector
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
    describe_saved_choice,
    read_saved_choice,
    saved_choice_note,
)
from uclone_x.llm.connectors.vllm import (
    VLLM_ENDPOINT_ENV_VARS,
    VLLM_MODEL_ENV_VAR,
    VLLMConnector,
    has_configured_vllm_endpoint,
)
from uclone_x.llm.models import LLMRequest, ModelResponse, StreamChunk

#: The credential variables precedence step 3 auto-detects from, in its order.
_CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


#: Variables that name the model a self-hosted connector asks for, in the order the
#: connector reads them. The other connectors read no model variable.
_MODEL_ENV_VARS: dict[str, tuple[str, ...]] = {
    "ollama": ("OLLAMA_MODEL", "OLLAMA_INDEPTH_MODEL", "OLLAMA_FAST_MODEL"),
    "vllm": (VLLM_MODEL_ENV_VAR,),
}


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
    run`` now agree with it.
    """
    if provider is None:
        return None
    for name in _MODEL_ENV_VARS.get(provider.strip().lower(), ()):
        if _set(name):
            return name
    return None


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


def _names_no_model(model: str | None) -> bool:
    """Whether a request leaves the model to the connector (``resolve_ollama_model``'s test)."""
    return model is None or model.strip() in ("", "default")


@cache
def _saved_model_class(base: type[BaseLLMConnector]) -> type[BaseLLMConnector]:
    """``base``, sending a request that names no model to the saved model instead.

    A subclass rather than a wrapper, so the connector keeps its type and every attribute
    the heads read off it (``base_url``, ``context_windows``, ``isinstance`` checks).
    """

    class _SavedModelDefault(base):
        _saved_model: str

        @property
        def _default_model(self) -> str:
            return self._saved_model

        def _with_saved_model(self, request: LLMRequest) -> LLMRequest:
            if _names_no_model(request.model):
                return request.model_copy(update={"model": self._saved_model})
            return request

        # `base` is always a concrete connector; pyright sees only the abstract bound.
        async def generate(self, request: LLMRequest) -> ModelResponse:
            return await super().generate(  # pyright: ignore[reportAbstractUsage]
                self._with_saved_model(request)
            )

        def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
            return super().stream(  # pyright: ignore[reportAbstractUsage]
                self._with_saved_model(request)
            )

    _SavedModelDefault.__name__ = base.__name__
    _SavedModelDefault.__qualname__ = base.__qualname__
    return _SavedModelDefault


def _default_to_saved_model(connector: BaseLLMConnector, model: str) -> BaseLLMConnector:
    """Make ``model`` what ``connector`` asks for when a request names none.

    Without this a saved ``qwen3:1.7b`` reached every caller that does not pass a model --
    ACP, A2A, the eval answerer, a dashboard started before setup -- as an Ollama connector
    asking for its built-in ``qwen3:8b``, which setup never pulled, so every turn failed.
    """
    connector.__class__ = _saved_model_class(type(connector))
    connector._saved_model = model  # pyright: ignore[reportAttributeAccessIssue]
    return connector


def _connector_for_saved_choice(
    choice: SavedChoice, api_key: str | None, **kwargs: Any
) -> BaseLLMConnector:
    """Build the provider a saved choice names, with its saved endpoint, key and model.

    A key the caller passed outranks the saved one, as an argument outranks the file
    everywhere else in this precedence. The saved model becomes the connector's default,
    so a caller that names a model still gets that model.
    """
    if choice.provider not in SAVED_PROVIDERS:
        # Refused, naming the file: ignoring it would report "nothing is configured" to a
        # person who did configure something, and point them away from the real defect.
        raise LLMProviderError(
            f"The model choice saved in {choice.path} names a provider this version does "
            f"not support: {choice.provider}. Pick a model again in the dashboard's Settings."
        )
    connector = create_llm_connector(
        provider=choice.provider,
        api_key=api_key or choice.api_key,
        base_url=choice.base_url,
        **kwargs,
    )
    if choice.model is None or model_env_override(choice.provider) is not None:
        # `OLLAMA_MODEL` / `VLLM_MODEL` outrank the saved model, as they do in the
        # dashboard: the connector already reads the variable, so it is left to.
        return connector
    return _default_to_saved_model(connector, choice.model)


def create_llm_connector(
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    fallback_to_mock: bool = False,
    **kwargs: Any,
) -> BaseLLMConnector:
    """Create an LLM provider connector from an explicit name or the environment.

    Resolution precedence:

    1. Explicit provider name (``openai``, ``anthropic``, ``gemini``/``google``,
       ``ollama``, ``vllm``, ``mock``).
    2. ``LLM_PROVIDER``.
    3. Auto-detection from a credential variable (``OPENAI_API_KEY``,
       ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``/``GOOGLE_API_KEY``).
    4. Auto-detection from a self-hosted endpoint variable: Ollama's
       (``OLLAMA_BASE_URL``, ``OLLAMA_FAST_BASE_URL``, ``LOCAL_LLM_BASE_URL``,
       ``OLLAMA_HOST``) or vLLM's (``VLLM_BASE_URL``), or from an explicit ``base_url``.
    5. The choice the person saved -- in the dashboard's Settings, or by ``ucx install`` /
       ``ucx start`` on a first setup -- read from ``<session root>/settings.json``
       (``saved_choice.py``). Every step above outranks it, so a flag or a variable still
       wins; see ``saved_choice_in_effect``. A head that uses it says so. Its model is
       what a request naming no model gets, unless a model variable is set
       (``OLLAMA_MODEL``/``OLLAMA_INDEPTH_MODEL``/``OLLAMA_FAST_MODEL`` for Ollama,
       ``VLLM_MODEL`` for vLLM): a variable outranks the file for the model too, in the
       same order the dashboard applies (environment first, then its Settings file).
    6. When nothing above names a provider: ``MockLLMConnector`` if ``fallback_to_mock``,
       otherwise **``LLMProviderNotConfiguredError``**.

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

    if resolved_provider == "openai" or (not resolved_provider and os.getenv("OPENAI_API_KEY")):
        return OpenAIConnector(api_key=api_key, base_url=base_url, **kwargs)

    if resolved_provider == "anthropic" or (
        not resolved_provider and os.getenv("ANTHROPIC_API_KEY")
    ):
        return AnthropicConnector(api_key=api_key, base_url=base_url, **kwargs)

    if resolved_provider in ("gemini", "google") or (
        not resolved_provider and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    ):
        return GeminiConnector(api_key=api_key, base_url=base_url, **kwargs)

    if resolved_provider == "mock":
        return MockLLMConnector(api_key=api_key, base_url=base_url, **kwargs)

    if resolved_provider == "ollama":
        return OllamaConnector(base_url=base_url, **kwargs)

    if resolved_provider == "vllm":
        # No `base_url` default and no credential: `VLLMConnector` refuses construction when
        # nothing names an endpoint, naming `VLLM_BASE_URL`. That refusal is the point of
        # naming the provider explicitly — it says where the failure is, at composition,
        # rather than deferring it to a refused connection on the first turn (P6, #533).
        return VLLMConnector(api_key=api_key, base_url=base_url, **kwargs)

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
        return OllamaConnector(base_url=base_url, **kwargs)

    # After Ollama, deliberately. Both detectors answer True for any explicit `base_url`,
    # so a caller who passes one without naming a provider would otherwise change provider
    # with this line's position — and Ollama is what that call has always built. An
    # environment that sets both `OLLAMA_BASE_URL` and `VLLM_BASE_URL` is ambiguous, and the
    # factory does not resolve an ambiguity by guessing: it keeps the pre-existing answer,
    # and `LLM_PROVIDER=vllm` is how the other one is chosen.
    if has_configured_vllm_endpoint(base_url):
        return VLLMConnector(api_key=api_key, base_url=base_url, **kwargs)

    # Step 5, above the flag for the reason step 4 is: a saved choice is the person's
    # configuration, not the unconfigured case the flag decides.
    saved = saved_choice_in_effect(provider, base_url)
    if saved is not None:
        return _connector_for_saved_choice(saved, api_key, **kwargs)

    if fallback_to_mock:
        return MockLLMConnector(api_key=api_key, base_url=base_url, **kwargs)

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
        f"{saved_choice_note()} "
        "`ucx install` sets up a local model and saves it; in the dashboard (`ucx start`), "
        "pick a model in Settings. From a terminal, `ucx llm status` reports what is reachable."
    )
