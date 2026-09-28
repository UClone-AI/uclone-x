"""What a hosted provider's failed call is, for the person who has to act on it (#1630).

The hosted connectors -- Gemini, Anthropic, and the OpenAI-compatible family vLLM belongs
to -- raised every failure as `LLMProviderError(f"<Provider> error {status}: {body}")`. A
retired model, a rejected key, a spent quota and a provider outage all reached the user as
the same line of JSON, or, where a head declined to show raw text, as "something went
wrong". Neither says whose side the problem is on, which is the first thing someone new to
model providers needs to know.

Each function here is the one place a connector's failure becomes a
`ProviderFailureError`. The status decides first; a body phrase is read only where a
provider is known to answer a status that does not say enough (Gemini's rejected key is a
400). A response none of the rules places is a `ProviderResponseError`, never a guess at
one of the others. The raw response is logged here, because the error's own message is
written to be shown and carries none of it. It is logged below WARNING, for the reason
`_RAW_RESPONSE_LOG_LEVEL` gives.
"""

from __future__ import annotations

import logging

import httpx

from uclone_x.errors import (
    ModelNotAvailableError,
    ProviderAuthError,
    ProviderFailureError,
    ProviderOutageError,
    ProviderQuotaError,
    ProviderResponseError,
    ProviderUnreachableError,
)
from uclone_x.llm.connectors.ollama import describe_transport_error

__all__ = ["failed_request", "failed_status", "unusable_response"]

logger = logging.getLogger(__name__)

#: The level the raw response is logged at. Below WARNING, because Python's fallback handler
#: prints WARNING and above to stderr when no head has configured logging -- and none does --
#: which would put the status and body on the user's terminal (#1630 review).
_RAW_RESPONSE_LOG_LEVEL = logging.INFO

#: How much of a response body the log keeps. An HTML error page from a proxy can be long,
#: and the part that identifies it is at the start.
_LOGGED_BODY_CHARS = 2000

#: Phrases that place a response the status alone does not, lowercased, each from a body a
#: provider is documented to send. Gemini rejects a bad key with a 400 whose body names
#: `API_KEY_INVALID`; Anthropic reports a spent credit balance as a 400; OpenAI names a spent
#: quota `insufficient_quota`; Gemini's spent quota is `RESOURCE_EXHAUSTED`.
_AUTH_PHRASES = ("api_key_invalid", "api key not valid")
_QUOTA_PHRASES = ("credit balance", "insufficient_quota", "resource_exhausted")
_MODEL_MISSING_PHRASES = ("model_not_found",)

#: What makes a 403 about the key. A 403 alone is not: OpenAI answers a 403 for an
#: unsupported country, and a proxy in front of a custom endpoint answers its own "Forbidden"
#: page, and neither is cured by a new key. Gemini says `PERMISSION_DENIED` for a key that may
#: not call the API; Anthropic's `permission_error` says "Your API key does not have
#: permission".
_FORBIDDEN_KEY_PHRASES = ("api key", "api_key", "permission_denied", "permission_error")

#: What makes a 404 about the model, from each provider's documented answer: Gemini's "is not
#: found for API version", OpenAI's and vLLM's "The model `x` does not exist", Anthropic's
#: `"message": "model: <id>"`. Not the word "model" alone: Google's HTML 404 page for a
#: mistyped endpoint echoes the URL, and that URL holds `/models/<id>:generateContent`.
_MODEL_404_PHRASES = ("is not found for api version", "does not exist", '"model: ')


def failed_status(
    *, provider: str, model: str, status_code: int, body: str
) -> ProviderFailureError:
    """The error for a response whose status is not 200, with the raw response logged.

    A 404 is a missing model only when its body says the model is missing, and a 403 is a
    key problem only when its body names the key or a permission. A 404 from a mistyped
    custom endpoint, or a 403 from a proxy or a region block, is neither, and telling the
    user their model was retired or their key refused would send them to fix the wrong thing.
    """
    logger.log(
        _RAW_RESPONSE_LOG_LEVEL,
        "%s answered %d for model %s: %s",
        provider,
        status_code,
        model,
        body[:_LOGGED_BODY_CHARS],
    )
    text = body.lower()
    if (
        status_code == 401
        or (status_code == 403 and any(phrase in text for phrase in _FORBIDDEN_KEY_PHRASES))
        or any(phrase in text for phrase in _AUTH_PHRASES)
    ):
        return ProviderAuthError(provider=provider, model=model)
    if (status_code == 404 and any(phrase in text for phrase in _MODEL_404_PHRASES)) or any(
        phrase in text for phrase in _MODEL_MISSING_PHRASES
    ):
        return ModelNotAvailableError(provider=provider, model=model)
    if status_code in (402, 429) or any(phrase in text for phrase in _QUOTA_PHRASES):
        return ProviderQuotaError(provider=provider, model=model)
    if status_code >= 500 or "overloaded" in text:
        return ProviderOutageError(provider=provider, model=model)
    return ProviderResponseError(provider=provider, model=model)


def failed_request(*, provider: str, model: str, exc: httpx.RequestError) -> ProviderFailureError:
    """The error for a request that got no answer at all, with the transport error logged."""
    logger.log(
        _RAW_RESPONSE_LOG_LEVEL,
        "%s could not be reached for model %s: %s",
        provider,
        model,
        describe_transport_error(exc),
    )
    return ProviderUnreachableError(provider=provider, model=model)


def unusable_response(*, provider: str, model: str, detail: str) -> ProviderFailureError:
    """The error for a 200 whose body is not a reply -- not JSON, or no candidates in it."""
    logger.log(
        _RAW_RESPONSE_LOG_LEVEL,
        "%s answered model %s with an unusable reply: %s",
        provider,
        model,
        detail,
    )
    return ProviderResponseError(provider=provider, model=model)
