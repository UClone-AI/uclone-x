"""LLM connector factory resolving live providers from configuration (Principle 5 & Principle 6)."""

from __future__ import annotations

import os
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
from uclone_x.llm.connectors.vllm import (
    VLLM_ENDPOINT_ENV_VARS,
    VLLMConnector,
    has_configured_vllm_endpoint,
)


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
    5. When nothing above names a provider: ``MockLLMConnector`` if ``fallback_to_mock``,
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

    ``fallback_to_mock`` survives at exactly one site — precedence step 5 — and it is not a
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
        "In the desktop app, use the Settings panel. From a terminal, "
        "`./ucx llm status` reports what is reachable."
    )
