"""vLLM connector — an OpenAI-compatible endpoint the operator runs themselves (P5 & P6)."""

from __future__ import annotations

import os
from typing import ClassVar

import httpx

from uclone_x.errors import LLMProviderNotConfiguredError
from uclone_x.llm.connectors.openai import OpenAIConnector, named_model
from uclone_x.llm.models import LLMRequest

VLLM_ENDPOINT_ENV_VARS: tuple[str, ...] = ("VLLM_BASE_URL",)
"""Environment variables that name a vLLM endpoint, in precedence order.

A tuple with one name rather than a bare constant, for the reason
`OLLAMA_ENDPOINT_ENV_VARS` is one: the caller asking *whether* an endpoint is configured
and the caller asking *which* endpoint it is must read the same list. `create_llm_connector`
asks the first and `resolve_vllm_base_url` answers the second, and a second accepted name
added to only one of them is how those two answers start disagreeing (#539).
"""

VLLM_MODEL_ENV_VAR = "VLLM_MODEL"
"""The variable naming the model a vLLM server was started with.

Separate from the endpoint, because one server serves exactly one model and the two facts
are configured at different times: the endpoint when the server is placed, the model when
it is launched with `--model`.
"""


_UNCONFIGURED_ENDPOINT_MESSAGE = (
    "No vLLM endpoint is configured: pass base_url= or set "
    f"{' or '.join(VLLM_ENDPOINT_ENV_VARS)} (e.g. http://localhost:8000/v1). "
    "Unlike Ollama, vLLM has no endpoint this connector can assume: a vLLM server is "
    "started per model on a port chosen at launch, so a default would name a process that "
    "may never have existed and defer the failure to the first turn (P6)."
)
"""Why a missing endpoint is a refusal. A constant so the `raise` is a single line.

The lethality ratchet replaces one line and re-runs the tests, so a `raise` spread over six
lines can only be mutated into a syntax error — and a collection error is a non-zero exit
that reads as a kill (`Mutation syntax error = false kill`). Held on one line, the refusal
can actually be mutated into the default it refuses to be, and the test that fails is then
evidence.
"""

_UNCONFIGURED_MODEL_MESSAGE = (
    "No vLLM model is configured: set model= on the request or "
    f"{VLLM_MODEL_ENV_VAR} (the value passed to `vllm serve --model`). A vLLM server serves "
    "the one model it was started with and answers any other name with a 404, so there is "
    "no default this connector can supply that is not a guess at that argument (P6)."
)
"""Why a missing model is a refusal. One line at the `raise`, as above."""


def normalize_vllm_base_url(url: str) -> str:
    """Return `url` with exactly one trailing `/v1`, whether or not it arrived with one.

    vLLM mounts its OpenAI-compatible API at `/v1`, and `OpenAIConnector` builds
    `f"{self.base_url}/chat/completions"`, so the `/v1` has to be in the base URL. An
    operator reading vLLM's own startup banner (`http://0.0.0.0:8000`) will set the host
    and port without it; one copying a curl example will include it. Both are the same
    endpoint, and the difference between them must not be the difference between a working
    connector and a 404 — which is what the two spellings produced before this existed.

    Trailing `/v1` segments are stripped before one is appended, so `.../v1/v1` (a
    paste of a normalized value into a variable that is normalized again) resolves to
    `.../v1` rather than accumulating.
    """
    cleaned = url.strip().rstrip("/")
    while cleaned.endswith("/v1"):
        cleaned = cleaned[: -len("/v1")].rstrip("/")
    return f"{cleaned}/v1"


def has_configured_vllm_endpoint(base_url: str | None = None) -> bool:
    """Report whether a vLLM endpoint was named by argument or environment.

    False means the environment says nothing about vLLM — it does **not** mean a vLLM
    server is unreachable, which is a question only a request can answer. The distinction
    is what `create_llm_connector` needs: an unconfigured environment must not be answered
    with a guessed endpoint.
    """
    if base_url and base_url.strip():
        return True
    for name in VLLM_ENDPOINT_ENV_VARS:
        # Deliberately not `(os.getenv(name) or "").strip()`: substituting a literal for a
        # missing value is the shape `test_no_connector_substitutes_a_literal_for_a_missing_value`
        # sweeps this package for, and the sweep is right to refuse it even where the
        # substitution is immediately discarded.
        value = os.getenv(name)
        if value is not None and value.strip():
            return True
    return False


def resolve_vllm_base_url(base_url: str | None = None) -> str:
    """The vLLM endpoint named by `base_url` or the environment, normalized.

    **Why there is no default, when `resolve_ollama_base_url` has one.** Ollama is a daemon
    installed as a service on a fixed, well-known port, so `http://localhost:11434` names
    the thing an installation actually put there. A vLLM server is launched per model with
    a port chosen at the command line, so `http://localhost:8000` — its documented default
    — names a process that exists only if somebody started it with those arguments. A
    connector built on that guess succeeds at construction and fails at the first turn with
    a refused connection, which reports a socket instead of the configuration defect that
    caused it (P6, and #533 in the factory for exactly this shape).

    Raises:
        LLMProviderNotConfiguredError: nothing named an endpoint. The message names the
            variable to set.
    """
    candidates = (base_url, *(os.getenv(name) for name in VLLM_ENDPOINT_ENV_VARS))
    for candidate in candidates:
        if candidate and candidate.strip():
            return normalize_vllm_base_url(candidate)
    raise LLMProviderNotConfiguredError(_UNCONFIGURED_ENDPOINT_MESSAGE)


def resolve_vllm_model(model: str | None = None) -> str | None:
    """The model named by the argument or `VLLM_MODEL`, or `None` when neither names one.

    `None` rather than a default, because a vLLM server serves the one model it was
    started with and rejects any other name with a 404. There is no model string this
    connector could supply that is not a guess at somebody else's `--model` argument, and
    a guess that happens to be wrong arrives as `The model X does not exist` — a message
    about the model the caller never chose (P6, #385).
    """
    if model is not None:
        stripped = model.strip()
        if stripped and stripped != "default":
            return stripped

    configured = os.getenv(VLLM_MODEL_ENV_VAR)
    if configured is not None and configured.strip():
        return configured.strip()
    return None


class VLLMConnector(OpenAIConnector):
    """A vLLM server, addressed through its OpenAI-compatible `/v1` surface.

    Subclassing `OpenAIConnector` rather than copying it: vLLM's `/v1/chat/completions`
    **is** OpenAI's, down to the `finish_reason` values and the `usage` block, so a sibling
    module would be a second copy of the payload builder, the finish-reason mapper and the
    usage completion, free to drift from the first. What differs is declared, and the base
    class reads each of these at the point it used to hold a literal (#1304):

    * `provider_name` — `"vllm"`, so `Provenance` and `TokenUsage.provider` both name the
      service that actually answered. Before the base class read this property, a
      self-hosted endpoint was recorded as OpenAI.
    * `_display_name` — a failure against `http://localhost:8000/v1` says `vLLM`, not
      `OpenAI`, so the operator is not sent to check an API key and a status page for a
      service that was never involved.
    * `_requires_api_key = False` — `vllm serve` takes `--api-key` but does not require it,
      and a demanded credential would be a value the operator has to invent. No key means
      **no** `Authorization` header (`_auth_headers`), not an empty one: #385's empty bearer
      stays closed.
    * the endpoint and the model, neither of which is defaulted — see `resolve_vllm_base_url`
      and `resolve_vllm_model`.

    Not claimed here, and measured rather than assumed: `vllm serve` exposes tool calling only
    when started with `--enable-auto-tool-choice --tool-call-parser <parser>`. This connector
    emits `tools` exactly as `OpenAIConnector` does and sends no `tool_choice` at all, so vLLM
    applies its own default of `"auto"` — and a server started without those flags **refuses**
    the request rather than answering it: HTTP 400, `'"auto" tool choice requires
    --enable-auto-tool-choice and --tool-call-parser to be set'`. It does not fall back to a
    model describing the call in prose; the same server answers the identical request with
    `tools` omitted, so the refusal is per-request and its message names both missing flags.
    Starting the server correctly is the operator's to fix, not something a connector can
    correct, but no detection is owed here: vLLM already fails in the shape P6 asks for
    (measured against a live vLLM 0.30.0 server, #1304).
    """

    _display_name: ClassVar[str] = "vLLM"
    _api_key_env_var: ClassVar[str] = "VLLM_API_KEY"
    _base_url_env_var: ClassVar[str] = "VLLM_BASE_URL"
    _requires_api_key: ClassVar[bool] = False

    _default_base_url: ClassVar[str] = ""
    """Never read: `__init__` resolves the endpoint and refuses rather than default it.

    Declared all the same, because the value it would otherwise inherit is
    `https://api.openai.com/v1` — a base URL that would send a local deployment's traffic,
    and its credential, to OpenAI if any future path reached it.
    """

    _default_model: ClassVar[str] = ""
    """Never read: `_resolve_request_model` is overridden and consults `VLLM_MODEL`.

    Declared for the same reason as `_default_base_url`: the inherited value is `gpt-4o`,
    which no vLLM server serves.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=resolve_vllm_base_url(base_url),
            timeout=timeout,
            http_client=http_client,
        )

    @property
    def provider_name(self) -> str:
        return "vllm"

    def _resolve_request_model(self, request: LLMRequest) -> str:
        """The model to ask for: the caller's, else `VLLM_MODEL`, else a refusal.

        The base class defaults to `gpt-4o` here. Inheriting that would send a request no
        vLLM server can answer and report the operator's missing configuration as the
        model's non-existence.

        Raises:
            LLMProviderNotConfiguredError: neither the request nor the environment names a
                model.
        """
        named = named_model(request)
        if named is not None:
            return named

        configured = resolve_vllm_model()
        if configured is not None:
            return configured

        raise LLMProviderNotConfiguredError(_UNCONFIGURED_MODEL_MESSAGE)
