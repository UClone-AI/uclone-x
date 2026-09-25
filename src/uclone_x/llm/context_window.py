"""How large a model's context window is, and where that number came from (P6).

This module exists because the token ring on the composer needs a denominator, and the
obvious sources for one are wrong in ways that are invisible on screen.

**The Ollama trap, measured.** `GET /api/show` reports the window the model was *trained*
for -- `llama.context_length: 131072` for `llama3.2:1b` on this machine. `GET /api/ps`
reports the window the daemon *actually gave* the same model when it loaded it:
`context_length: 32768`. The second number is the one a turn is truncated against. Drawing
a ring against the first would have shown a seat holding 30,000 tokens as a quarter full
while it was in fact about to lose its earliest turns. That is the plausible substituted
value P6 forbids, arrived at from a real endpoint returning a real number.

**This runtime chooses the window it asks for.** The Ollama connector sends `num_ctx` on
every request: an agent's `context_limit` when one is configured (#1372), else the
daemon's own `OLLAMA_CONTEXT_LENGTH` when it is set where UClone-X runs, and otherwise
`DEFAULT_OLLAMA_NUM_CTX`. Left to itself the daemon picks from the machine's VRAM -- 4096
tokens under 24 GB -- and a fresh-machine test on a 16 GB GPU found one persona's request
alone at 5,186 tokens, so the first tool result of a turn had nowhere to go.

So the rule here is: **a locally served model's window is what the daemon serves, which is
the `num_ctx` sent, clamped by the window the model was trained for -- and the clamp is read
from the server, for the model it has loaded.** There is no table of local models, because
a table cannot know what the daemon decided.

**Hosted providers are the opposite case.** An Anthropic or OpenAI model's window is a
published figure the API enforces exactly; there is no per-installation choice to observe,
and no endpoint that reports one. For those a declared table *is* the measurement.

**The compaction trigger reads the same figures (#1372).** `compaction_window` is the one
resolver `BaseAgent` uses for the trigger and for the step result budget. It used to read
`MODEL_CONTEXT_WINDOWS` for every provider, so a `llama3*` model was compacted at 70% of
128,000 while the daemon served 32,768, and between the two every turn was cut from the
front by the daemon with no compaction and no ledger.

The hosted path has no provider default: a context window that is somewhat wrong is a ring
drawn to the wrong fraction, and the reader cannot tell. An unrecognised hosted model
returns `None`, and the surface says it does not know. `DEFAULT_OLLAMA_NUM_CTX` is not such
a default: it is the figure this runtime sends, so it is what the daemon is asked to serve.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, cast

import httpx

from uclone_x.llm.compactor import resolve_model_context_limit

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OLLAMA_NUM_CTX",
    "PUBLISHED_CONTEXT_WINDOWS",
    "ContextWindow",
    "OllamaContextWindows",
    "OLLAMA_CONTEXT_WINDOWS",
    "SERVED_WINDOW_PROVIDERS",
    "compaction_window",
    "default_ollama_num_ctx",
    "ollama_model_key",
    "published_context_window",
]

#: The `num_ctx` the Ollama connector sends when no agent configures `context_limit`.
#:
#: The daemon's own choice is 4096 on a machine with under 24 GB of VRAM, and the artist
#: persona's request alone measured 5,186 tokens on such a machine, so one tool result
#: failed the turn. 16384 holds that request, a 1024-token reply reserve and several tool
#: results. Its cost is KV cache, which grows linearly with the window: `qwen3:8b` (36
#: layers, 8 KV heads of 128, float16) holds 144 KiB per token, so 2.25 GiB at 16384 on top
#: of about 5.2 GB of weights -- about 7.5 GB, inside a 12 GB GPU. `qwen3:1.7b`, the model
#: for 8 GB Macs, holds 112 KiB per token: 1.75 GiB, about 3.2 GB with its weights. A model
#: trained for less is clamped by the daemon, and `compaction_window` reads the clamp.
DEFAULT_OLLAMA_NUM_CTX = 16_384

#: The daemon's own variable for the window it loads models with. A request's `num_ctx`
#: overrides it, so it is read here as well, the way `OLLAMA_KEEP_ALIVE` is.
OLLAMA_CONTEXT_LENGTH_ENV = "OLLAMA_CONTEXT_LENGTH"


def default_ollama_num_ctx() -> int:
    """The `num_ctx` sent when no agent configures `context_limit`.

    `OLLAMA_CONTEXT_LENGTH` when it holds a positive integer, else
    `DEFAULT_OLLAMA_NUM_CTX`. Without this, an operator who set the variable for the
    daemon -- to 32768 for a long-context model, or to 8192 to fit a small GPU -- would
    have it replaced by 16384 on every request, the same trap `resolve_ollama_keep_alive`
    avoids for `OLLAMA_KEEP_ALIVE`. Anything else in the variable is ignored rather than
    sent, because the daemon would reject or misread it.
    """
    raw = (os.getenv(OLLAMA_CONTEXT_LENGTH_ENV) or "").strip()
    try:
        tokens = int(raw)
    except ValueError:
        return DEFAULT_OLLAMA_NUM_CTX
    return tokens if tokens > 0 else DEFAULT_OLLAMA_NUM_CTX


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

    For a provider in `SERVED_WINDOW_PROVIDERS`, the `num_ctx` the connector sends:
    `configured` (the agent's `context_limit`) when set, else `default_ollama_num_ctx()`
    (`OLLAMA_CONTEXT_LENGTH`, then `DEFAULT_OLLAMA_NUM_CTX`).
    The daemon loads the model at that window, so the request and the limit agree by
    construction. The one exception is a smaller figure the daemon reports for the model:
    Ollama clamps `num_ctx` to the window the model was trained for, and then the reported
    figure is what a turn is cut against. **Never the model table**: a table figure here is
    a claim about the operator's machine, and it is the figure that let the daemon truncate
    silently.

    For any other provider, `configured` when set, else `MODEL_CONTEXT_WINDOWS`, as before.
    """
    configured_tokens = configured if configured is not None and configured > 0 else None
    if (provider or "").strip().lower() in SERVED_WINDOW_PROVIDERS:
        windows = store if store is not None else OLLAMA_CONTEXT_WINDOWS
        served = windows.get(base_url, model) if base_url else None
        sent = configured_tokens if configured_tokens is not None else default_ollama_num_ctx()
        if served is not None and served < sent:
            return served  # the daemon clamped `num_ctx` to the trained window
        return sent  # sent as num_ctx, so it is what the daemon serves
    if configured_tokens is not None:
        return configured_tokens
    return resolve_model_context_limit(model)
