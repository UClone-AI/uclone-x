"""The one table of what the product knows about each LLM provider.

A provider's facts -- its id and the other names it answers to, the variables that carry
its key, endpoint and model, whether it needs a key at all, and where a person gets one --
used to be written out wherever they were needed: the saved-choice reader kept its own set
of names, the connector factory its own credential and model variables, ``ucx run`` a
key-variable map keyed by *display name*, and ``ucx key`` a fourth copy with console links.
The copies drifted. ``ucx run`` knew vLLM's key variable and ``ucx key`` did not; the
saved-choice reader knew ``google`` was Gemini and the key command did not.

Every other site now looks the provider up here. Adding a provider, or a variable, is one
edit to :data:`PROVIDERS`.

Nothing here reads a file or touches the network. :func:`env_key` reads the environment,
because the environment is the one override every head honours ahead of the settings file,
and it is read-only: nothing in the product writes a provider variable back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "PROVIDERS",
    "ProviderSpec",
    "canonical_provider",
    "env_key",
    "env_model",
    "spec_for",
]


@dataclass(frozen=True)
class ProviderSpec:
    """Everything the product knows about one provider, and nothing it has to ask it."""

    #: The id the settings file, the factory and every head use (``gemini``, not ``google``).
    id: str
    #: Other names a person or an old settings file may use for the same provider.
    aliases: tuple[str, ...]
    #: The name shown to a person, and the one connectors put on their errors.
    display_name: str
    #: Variables that carry the key, in the order they are read; the first is canonical.
    key_env_vars: tuple[str, ...]
    #: The variable naming the endpoint, when the provider has one.
    base_url_env: str | None
    #: Variables naming the model, in the order they are read; the first is canonical.
    model_env_vars: tuple[str, ...]
    #: Whether a request cannot be sent without a key.
    requires_key: bool
    #: Where a person creates a key, when there is such a page.
    console_url: str | None = None
    #: How a key for this provider usually starts. Advisory only: a key that does not
    #: match is still saved (a current Gemini key starts ``AQ.``, not ``AIzaSy``).
    key_hint: str | None = None

    @property
    def model_env(self) -> str | None:
        """The canonical model variable, or ``None`` when the provider reads none."""
        return self.model_env_vars[0] if self.model_env_vars else None

    @property
    def key_env(self) -> str | None:
        """The canonical key variable, or ``None`` when the provider reads none."""
        return self.key_env_vars[0] if self.key_env_vars else None


PROVIDERS: dict[str, ProviderSpec] = {
    spec.id: spec
    for spec in (
        ProviderSpec(
            id="openai",
            aliases=(),
            display_name="OpenAI",
            key_env_vars=("OPENAI_API_KEY",),
            base_url_env="OPENAI_BASE_URL",
            model_env_vars=("OPENAI_MODEL",),
            requires_key=True,
            console_url="https://platform.openai.com/api-keys",
            key_hint="sk-",
        ),
        ProviderSpec(
            id="anthropic",
            aliases=(),
            display_name="Anthropic",
            key_env_vars=("ANTHROPIC_API_KEY",),
            base_url_env="ANTHROPIC_BASE_URL",
            model_env_vars=("ANTHROPIC_MODEL",),
            requires_key=True,
            console_url="https://console.anthropic.com/settings/keys",
            key_hint="sk-ant-",
        ),
        ProviderSpec(
            id="gemini",
            aliases=("google",),
            display_name="Google",
            key_env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            base_url_env="GEMINI_BASE_URL",
            model_env_vars=("GEMINI_MODEL",),
            requires_key=True,
            console_url="https://aistudio.google.com/app/apikey",
            key_hint=None,
        ),
        ProviderSpec(
            id="ollama",
            aliases=(),
            display_name="Ollama",
            key_env_vars=(),
            base_url_env="OLLAMA_BASE_URL",
            model_env_vars=("OLLAMA_MODEL", "OLLAMA_INDEPTH_MODEL", "OLLAMA_FAST_MODEL"),
            requires_key=False,
        ),
        ProviderSpec(
            id="vllm",
            aliases=(),
            display_name="vLLM",
            key_env_vars=("VLLM_API_KEY",),
            base_url_env="VLLM_BASE_URL",
            model_env_vars=("VLLM_MODEL",),
            requires_key=False,
        ),
        ProviderSpec(
            id="mock",
            aliases=(),
            display_name="Mock",
            key_env_vars=(),
            base_url_env=None,
            model_env_vars=(),
            requires_key=False,
        ),
    )
}

_BY_NAME: dict[str, ProviderSpec] = {
    name: spec for spec in PROVIDERS.values() for name in (spec.id, *spec.aliases)
}


def spec_for(name: str | None) -> ProviderSpec | None:
    """The provider ``name`` means, by id or alias, case and whitespace ignored; else ``None``."""
    if name is None:
        return None
    return _BY_NAME.get(name.strip().lower())


def canonical_provider(name: str | None) -> str | None:
    """The id ``name`` means (``google`` gives ``gemini``), or ``None`` for an unknown name."""
    spec = spec_for(name)
    return spec.id if spec is not None else None


def _set_variable(names: tuple[str, ...]) -> tuple[str, str] | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip(), name
    return None


def env_key(provider: str | None) -> tuple[str, str] | None:
    """``(key, variable)`` when the environment carries ``provider``'s key, else ``None``.

    The variable is returned with the value so a head can say *which* variable is
    overriding the key saved in Settings, rather than only that one is.
    """
    spec = spec_for(provider)
    return _set_variable(spec.key_env_vars) if spec is not None else None


def env_model(provider: str | None) -> tuple[str, str] | None:
    """``(model, variable)`` when the environment names ``provider``'s model, else ``None``."""
    spec = spec_for(provider)
    return _set_variable(spec.model_env_vars) if spec is not None else None
