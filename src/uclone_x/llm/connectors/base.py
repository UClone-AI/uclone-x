"""Base provider connector abstraction (Principle 5 & Principle 6)."""

from __future__ import annotations

import ipaddress
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast
from urllib.parse import urlsplit

import httpx

from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    LLMModelNotConfiguredError,
    MalformedToolCallArgumentsError,
    StructuredOutputUnsupportedError,
)
from uclone_x.llm.compactor import estimate_reply_tokens, estimate_request_tokens
from uclone_x.llm.models import (
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenCountSource,
    ToolCallRequest,
)
from uclone_x.llm.protocols import LLMProviderProtocol


def refuse_response_schema(request: LLMRequest, provider: str) -> None:
    """Raise when `request` asks for structured output this connector cannot send.

    For connectors with no mapping for `LLMRequest.response_schema`: dropping it would
    answer a structured request with free text and nothing saying so (P6).
    """
    if request.response_schema is not None:
        raise StructuredOutputUnsupportedError(provider)


def reported_count(block: Mapping[str, Any] | None, key: str) -> int | None:
    """The count a provider reported under `key`, or `None` when it reported none.

    `None` is not zero. A connector reading an absent usage field as `0` claimed that the
    provider counted zero tokens, labelled `TokenCountSource.PROVIDER`, and a provider that
    omitted usage was uncharged by every ceiling (#939). Pass the result to
    `resolve_token_counts`, which says what stands in for a missing count and labels it.
    """
    if block is None:
        return None
    value: object = block.get(key)
    if value is None:
        return None
    return int(cast(int | str | float, value))


def resolve_token_counts(
    request: LLMRequest,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    reply: str | None,
    tool_calls: Sequence[ToolCallRequest] = (),
) -> tuple[int, int, TokenCountSource]:
    """Complete a provider's usage report, and say whether it had to be completed.

    The rule for every connector (#939). A count the provider reported is used as reported.
    A count it did not report is estimated with the shared estimator
    (`uclone_x.llm.compactor`): the input from the request's messages and tool definitions,
    the output from the reply's text and tool calls. If either count is estimated, the
    whole usage is labelled `ESTIMATE`, because a total built partly from an estimate is
    not a count.

    Refusing the response was rejected. The model's answer is real, and ending a
    conversation because an OpenAI-compatible server leaves out `usage` is the refusal #935
    already rejected for the stream path. Recording `0` was rejected: that is the silent
    substitution this replaces.

    What "not reported" means belongs to each provider's wire format, and the connector
    decides it. Gemini omits zero-valued fields inside a `usageMetadata` it did send, so
    only an absent block is unreported there.
    """
    estimated = input_tokens is None or output_tokens is None
    if input_tokens is None:
        input_tokens = estimate_request_tokens(request)
    if output_tokens is None:
        output_tokens = estimate_reply_tokens(reply, tool_calls)
    source = TokenCountSource.ESTIMATE if estimated else TokenCountSource.PROVIDER
    return input_tokens, output_tokens, source


#: Host names that stay on this machine or its local network.
_LOCAL_HOST_SUFFIXES: tuple[str, ...] = (".local", ".lan", ".internal", ".home.arpa", ".localhost")


def is_local_endpoint(url: str | None) -> bool:
    """Whether `url` points at this machine or a private network, not a hosted service.

    Loopback, private-range and link-local addresses, `localhost`, single-label host names
    (`gpu-box`) and the local-network suffixes count as local. Anything else, including an
    address that cannot be parsed, counts as hosted: an unknown endpoint is treated as one
    that may charge (P6, fail closed).
    """
    if not url:
        return False
    host = urlsplit(url if "//" in url else f"//{url}").hostname
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        name = host.lower().rstrip(".")
        return name == "localhost" or "." not in name or name.endswith(_LOCAL_HOST_SUFFIXES)
    return address.is_loopback or address.is_private or address.is_link_local


def parse_dict_payload(data: object) -> dict[str, Any]:
    """Safely parse a JSON string or mapping into a dict[str, Any], raising on malformed strings (P6)."""
    if data is None or data == "":
        return {}
    if isinstance(data, str):
        if not data.strip():
            return {}
        try:
            parsed = json.loads(data)
        except Exception as exc:
            raise MalformedToolCallArgumentsError(
                f"Failed to parse tool call arguments JSON: {exc}. Raw payload: {data!r}"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise MalformedToolCallArgumentsError(
                f"Expected JSON object for tool call arguments, got {type(parsed).__name__}: {data!r}"
            )
        data = cast(object, parsed)
    if isinstance(data, Mapping):
        mapping_data = cast(Mapping[object, object], data)
        result: dict[str, Any] = {}
        for key in mapping_data:
            result[str(key)] = mapping_data[key]
        return result
    raise MalformedToolCallArgumentsError(
        f"Invalid tool call arguments type: {type(data).__name__}. Expected JSON string or mapping."
    )


def named_model(model: str | None) -> str | None:
    """``model`` stripped, or ``None`` when it names no model (absent, blank or ``default``)."""
    if model is None:
        return None
    stripped = model.strip()
    if not stripped or stripped == "default":
        return None
    return stripped


def resolve_model(requested: str | None, configured: str | None, provider: str) -> str:
    """The model a request is sent to: the one it names, else the connector's own.

    ``configured`` is the model the connector was built with -- the model a person chose,
    handed down by the connector factory. With neither, the request is refused before the
    network with :class:`LLMModelNotConfiguredError`, naming ``provider`` (a display name).
    A connector never fills in a model id from its source: those ids go stale, and the
    person who chose a model is owed a refusal rather than a call to one they did not pick.

    One function for the request builder, the streaming path and the ``Provenance.requested``
    each connector reports, so no copy of the rule can drift from another.
    """
    chosen = named_model(requested) or named_model(configured)
    if chosen is None:
        raise LLMModelNotConfiguredError(provider)
    return chosen


class BaseLLMConnector(ABC, LLMProviderProtocol):
    """Abstract base connector for foundation model providers."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._http_client = http_client

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider service."""
        ...

    @property
    def paid(self) -> bool:
        """Whether calls through this connector may cost the user money.

        A paid connector's calls count against the user's usage limits
        (the token-gateway design). The default is `True`, so a connector that
        does not say otherwise is counted rather than left unlimited (P6). A local model
        overrides it.
        """
        return True

    def _require_api_key(self) -> str:
        """Return `api_key` as a `str`, raising if it is absent or blank.

        The keyed connectors validate their key in `__init__`, so this cannot fail on
        one built through its constructor. It exists because the auth header dicts have
        to narrow `api_key` — declared `str | None` here — to a `str`, and the way they
        did it was `self.api_key or ""`, which put an empty credential on the wire and
        made a missing key look like a provider `401` (P6, #385). A `cast` would silence
        the type error while leaving that reachable, since `api_key` is public and
        mutable: anything that assigns to it after construction bypasses the `__init__`
        check, and this is where that is caught.
        """
        key = self.api_key
        if key is None or not key.strip():
            raise LLMCredentialsNotConfiguredError(
                f"{type(self).__name__}.api_key is {key!r} at request time, so there is "
                "no credential to authenticate with. It is not sent as an empty header, "
                "because the provider would answer 401 and the caller would see a "
                "transport fault rather than the configuration defect that caused it "
                "(P6, #385)."
            )
        return key

    @abstractmethod
    async def generate(self, request: LLMRequest) -> ModelResponse:
        """Generate a complete model response."""
        ...

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream normalized chunks carrying deltas, tool calls and usage."""
        ...

    def _get_client(self) -> httpx.AsyncClient:
        """Return the injected HTTP client or create a new client context."""
        if self._http_client is not None:
            return self._http_client
        return httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned."""
        if self._http_client is not None:
            await self._http_client.aclose()
