"""How large a model's context window is, and where that number came from (P6).

This module exists because the token ring on the composer needs a denominator, and the
obvious sources for one are wrong in ways that are invisible on screen.

**The Ollama trap, measured.** `GET /api/show` reports the window the model was *trained*
for -- `llama.context_length: 131072` for `llama3.2:1b` on this machine. `GET /api/ps`
reports the window the daemon *actually gave* the same model when it loaded it:
`context_length: 32768`. Unless an agent configures `context_limit` -- which the Ollama
connector then sends as `num_ctx` (#1372) -- nothing in this runtime chooses the window,
so the second number is the server's own choice and it is the one a turn is truncated
against. Drawing a ring
against the first would have shown a seat holding 30,000 tokens as a quarter full while it
was in fact about to lose its earliest turns. That is the plausible substituted value P6
forbids, arrived at from a real endpoint returning a real number.

So the rule here is: **a locally served model's window is read from the server, for the
model it has loaded, or it is not known.** There is no table of local models, because a
table cannot know what the daemon decided.

**Hosted providers are the opposite case.** An Anthropic or OpenAI model's window is a
published figure the API enforces exactly; there is no per-installation choice to observe,
and no endpoint that reports one. For those a declared table *is* the measurement.

**The compaction trigger reads the same figures (#1372).** `compaction_window` is the one
resolver `BaseAgent` uses for the trigger and for the step result budget. It used to read
`MODEL_CONTEXT_WINDOWS` for every provider, so a `llama3*` model was compacted at 70% of
128,000 while the daemon served 32,768, and between the two every turn was cut from the
front by the daemon with no compaction and no ledger.

Neither path has a provider default: a context window that is somewhat wrong is a ring
drawn to the wrong fraction, and the reader cannot tell. An unrecognised model returns
`None`, and the surface says it does not know.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

import httpx

from uclone_x.llm.compactor import resolve_model_context_limit

logger = logging.getLogger(__name__)

__all__ = [
    "PUBLISHED_CONTEXT_WINDOWS",
    "ContextWindow",
    "OllamaContextWindows",
    "OLLAMA_CONTEXT_WINDOWS",
    "SERVED_WINDOW_PROVIDERS",
    "compaction_window",
    "ollama_model_key",
    "published_context_window",
]

#: Where a window figure came from. The head turns these into a sentence; the Core does
#: not write the sentence, because which words a reader sees is the head's (P8).
WindowSource = Literal["loaded", "published"]


class ContextWindow:
    """A window figure and its provenance, never one without the other (P6)."""

    __slots__ = ("tokens", "source")

    def __init__(self, tokens: int, source: WindowSource) -> None:
        self.tokens = tokens
        self.source = source

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ContextWindow)
            and other.tokens == self.tokens
            and other.source == self.source
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"ContextWindow(tokens={self.tokens}, source={self.source!r})"


#: Context windows the provider publishes and its API enforces, in tokens.
#:
#: Deliberately no `"default"` key on any provider, unlike `PRICING_TABLE`. See the module
#: docstring: an unknown model's window is unknown, and a ring has no honest way to show a
#: guess as a guess.
#:
#: `ollama` and `vllm` are absent on purpose and must stay absent. Both serve whatever the
#: operator loaded, at whatever window that server chose, so any figure written here would
#: be a claim about someone else's machine.
PUBLISHED_CONTEXT_WINDOWS: dict[str, dict[str, int]] = {
    "openai": {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-4-turbo": 128_000,
        "gpt-3.5-turbo": 16_385,
    },
    "anthropic": {
        "claude-3-5-sonnet": 200_000,
        "claude-3-opus": 200_000,
        "claude-3-haiku": 200_000,
        "claude-3-7-sonnet": 200_000,
    },
    "gemini": {
        "gemini-1.5-pro": 2_097_152,
        "gemini-1.5-flash": 1_048_576,
        "gemini-2.0-flash": 1_048_576,
    },
    "google-genai": {
        "gemini-1.5-pro": 2_097_152,
        "gemini-1.5-flash": 1_048_576,
        "gemini-2.0-flash": 1_048_576,
    },
}

#: Delimiters a model tag is split on: a dated or versioned tag (`gpt-4o-mini-2024-07-18`) must match `gpt-4o-mini` and not `gpt-4o`.
_DELIMITERS: frozenset[str] = frozenset({"-", ".", ":", "_", "/"})


def _normalize_model_name(model: str | None) -> str:
    """Strip gateway and fine-tune prefixes."""
    if not model:
        return ""
    name = model.strip().lower()
    if "/" in name:
        name = name.split("/")[-1]
    if name.startswith("ft:"):
        parts = name.split(":")
        if len(parts) > 1 and parts[1]:
            name = parts[1]
    return name


def published_context_window(provider: str | None, model: str | None) -> int | None:
    """The window a hosted provider publishes for `model`, or `None` if unrecognised.

    Exact match first, then the longest delimited prefix, and then nothing -- there is no
    provider fallback here by design.
    """
    if not provider:
        return None
    table = PUBLISHED_CONTEXT_WINDOWS.get(provider.strip().lower())
    if not table:
        return None
    norm = _normalize_model_name(model)
    if not norm:
        return None
    if norm in table:
        return table[norm]
    best_len = -1
    found: int | None = None
    for key, tokens in table.items():
        key_len = len(key)
        if norm.startswith(key) and (len(norm) == key_len or norm[key_len] in _DELIMITERS):
            if key_len > best_len:
                best_len = key_len
                found = tokens
    return found


def ollama_model_key(model: str) -> str:
    """`model` as the daemon names it in `/api/ps`: a name with no tag is `:latest`.

    Ollama resolves `llama3.2` to `llama3.2:latest` and reports the loaded model under the
    full name, so a store keyed by the name as configured never finds an untagged model's
    window (#1372). The tag is looked for in the last path segment only, because a
    registry host may carry a port (`localhost:5000/team/model`).
    """
    name = model.strip()
    if name and ":" not in name.rsplit("/", 1)[-1]:
        return name + ":latest"
    return name


class OllamaContextWindows:
    """What the Ollama daemon says it gave each model it has loaded.

    **Observations are remembered for the life of the process.** A model unloads after its
    keep-alive expires and drops out of `/api/ps`, but the window the daemon chose for it
    does not change when it does. Forgetting on unload would make the composer's ring swing
    between a token fraction and a turn fraction while the reader sat still, which reads as
    a broken ring rather than as a model going idle. A remembered figure is still a figure
    this daemon reported for this model; it is never a default and never a guess.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], int] = {}

    @staticmethod
    def _key(base_url: str, model: str) -> tuple[str, str]:
        return (base_url.rstrip("/"), ollama_model_key(model))

    def remember(self, base_url: str, model: str, tokens: int) -> None:
        """Record one observation. Used by `refresh` and directly by tests."""
        if model.strip() and tokens > 0:
            self._seen[self._key(base_url, model)] = tokens

    def get(self, base_url: str, model: str | None) -> int | None:
        """The window last observed for `model`, or `None` if it was never loaded here."""
        if not model:
            return None
        return self._seen.get(self._key(base_url, model))

    async def refresh(
        self,
        base_url: str,
        *,
        timeout: float = 2.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Read `GET /api/ps` and remember every window it reports.

        Never raises. A daemon that is down, slow or too old to report `context_length`
        leaves the store as it was, and the caller then has no figure -- which is the
        honest outcome and the one the surface is built to say out loud.
        """
        url = f"{base_url.rstrip('/')}/api/ps"
        client = http_client if http_client is not None else httpx.AsyncClient(timeout=timeout)
        should_close = http_client is None
        try:
            resp = await client.get(url, timeout=timeout)
            if resp.status_code != 200:
                logger.debug(
                    "Ollama /api/ps returned %s; context windows unchanged", resp.status_code
                )
                return
            payload: object = resp.json()
            if not isinstance(payload, dict):
                return
            raw = cast(dict[str, Any], payload).get("models")
            if not isinstance(raw, list):
                return
            for item in cast(list[object], raw):
                if not isinstance(item, dict):
                    continue
                entry = cast(dict[str, Any], item)
                name = entry.get("model") or entry.get("name")
                tokens = entry.get("context_length")
                if (
                    isinstance(name, str)
                    and isinstance(tokens, int)
                    and not isinstance(tokens, bool)
                ):
                    self.remember(base_url, name, tokens)
        except Exception as exc:  # never break a readout over a window nobody promised
            logger.debug("Could not read Ollama context windows from %s: %s", url, exc)
        finally:
            if should_close:
                await client.aclose()


#: The store the head reads. One per process: the observations are about the daemon this
#: process talks to, not about any conversation, so nothing here belongs in a session.
OLLAMA_CONTEXT_WINDOWS = OllamaContextWindows()


#: Providers whose window is the server's choice and is read from the server. vLLM is not
#: here yet: it has no reader in this module, and it refuses an over-length request with an
#: error rather than cutting it silently, so its table guess fails loudly rather than
#: quietly. Adding it means adding a reader, never a table row.
SERVED_WINDOW_PROVIDERS: frozenset[str] = frozenset({"ollama"})


def compaction_window(
    provider: str | None,
    model: str | None,
    *,
    base_url: str | None,
    configured: int | None,
    store: OllamaContextWindows | None = None,
) -> int | None:
    """The window an agent's compaction trigger and step budget count against (#1372).

    For a provider in `SERVED_WINDOW_PROVIDERS`:

    * `configured` (the agent's `context_limit`) when set. The connector sends it as
      `num_ctx`, so the daemon loads the model at that window and the request and the limit
      agree by construction. The one exception is a smaller figure the daemon reports for
      the model: Ollama clamps `num_ctx` to the window the model was trained for, and then
      the reported figure is what a turn is cut against.
    * otherwise the window the daemon reported for the model it loaded, or `None` when it
      has not reported one. **Never the model table**: a table figure here is a claim about
      the operator's machine, and it is the figure that let the daemon truncate silently.

    For any other provider, `configured` when set, else `MODEL_CONTEXT_WINDOWS`, as before.
    """
    configured_tokens = configured if configured is not None and configured > 0 else None
    if (provider or "").strip().lower() in SERVED_WINDOW_PROVIDERS:
        windows = store if store is not None else OLLAMA_CONTEXT_WINDOWS
        served = windows.get(base_url, model) if base_url else None
        if configured_tokens is None:
            return served  # the daemon's figure, or unknown: never the table
        if served is not None and served < configured_tokens:
            return served  # the daemon clamped `num_ctx` to the trained window
        return configured_tokens  # sent as num_ctx, so it is what the daemon serves
    if configured_tokens is not None:
        return configured_tokens
    return resolve_model_context_limit(model)
