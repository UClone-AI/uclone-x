"""What `uclone_x.llm.context_window` may and may not claim about a model's window.

The subject here is not arithmetic. It is the one thing this module exists to refuse: a
window figure that was not measured. A window is the denominator of a ring, and a ring
drawn to the wrong fraction looks exactly like a ring drawn to the right one.
"""

from __future__ import annotations

import httpx
import pytest

from uclone_x.llm.context_window import (
    PUBLISHED_CONTEXT_WINDOWS,
    OllamaContextWindows,
    compaction_window,
    published_context_window,
)


class TestThePublishedTable:
    def test_a_dated_tag_matches_the_longest_prefix_and_not_the_shortest(self) -> None:
        """`gpt-4o-mini-2024-07-18` is a `gpt-4o-mini`, and both keys are in the table.

        A first-match loop over a dict is order-dependent, and the two models' windows happen to be equal today, so an
        order-dependent match would pass this file and be wrong the first time they differ.

        Killed by: src/uclone_x/llm/context_window.py :: if key_len > best_len:
        Becomes: if key_len < best_len:
        """
        assert published_context_window("openai", "gpt-4o-mini-2024-07-18") == 128_000
        assert published_context_window("openai", "gpt-3.5-turbo-0125") == 16_385

    def test_a_prefix_that_is_not_delimited_is_not_a_match(self) -> None:
        """`gpt-4omni` is not a `gpt-4o`, and a naive `startswith` says it is.

        Killed by: src/uclone_x/llm/context_window.py :: if norm.startswith(key) and (len(norm) == key_len or norm[key_len] in _DELIMITERS):
        Becomes: if norm.startswith(key):
        """
        assert published_context_window("openai", "gpt-4omni") is None

    def test_an_unrecognised_model_has_no_window_rather_than_a_typical_one(self) -> None:
        """The refusal this module exists for.

        A provider fallback here would put a plausible number under every ring on the
        screen, including the ones measuring a model nobody has ever priced. `None` is
        what the surface can say out loud.

        Killed by: src/uclone_x/llm/context_window.py :: found: int | None = None
        Becomes: found: int | None = 128_000
        """
        assert published_context_window("anthropic", "claude-9-unreleased") is None
        assert published_context_window("openai", "some-local-finetune") is None

    def test_no_provider_carries_a_default_entry(self) -> None:
        """Pinned as a property of the table, because the lookup cannot catch it.

        `published_context_window` has no `default` branch, so adding a `"default"` key to
        a provider would not fall back -- it would become a model literally named
        `default`, matching nothing, and the omission would look deliberate. The table is
        where the mistake would be made, so the table is where it is checked.

        Killed by: src/uclone_x/llm/context_window.py :: "gpt-3.5-turbo": 16_385,
        Becomes: "gpt-3.5-turbo": 16_385, "default": 128_000,
        """
        for provider, table in PUBLISHED_CONTEXT_WINDOWS.items():
            assert "default" not in table, (
                f"{provider} declares a default context window; an unknown model's window "
                f"is unknown and must be reported as such"
            )

    def test_no_locally_served_provider_is_in_the_table(self) -> None:
        """Ollama and vLLM serve at whatever window *that* machine chose.

        Measured on this repository's own machine: Ollama reports
        `llama.context_length: 131072` for `llama3.2:1b` from `/api/show`, and
        `context_length: 32768` for the same model from `/api/ps` once it has loaded it.
        A table entry would have to pick one of those for everybody, and the one it would
        pick is the wrong one by a factor of four.

        Killed by: src/uclone_x/llm/context_window.py :: "google-genai": {
        Becomes: "ollama": {"llama3.2": 131_072}, "google-genai": {
        """
        assert "ollama" not in PUBLISHED_CONTEXT_WINDOWS
        assert "vllm" not in PUBLISHED_CONTEXT_WINDOWS
        assert published_context_window("ollama", "llama3.2:1b") is None

    def test_a_gateway_prefix_does_not_hide_the_model(self) -> None:
        """`openai/gpt-4o` through a gateway is still `gpt-4o`.

        Killed by: src/uclone_x/llm/context_window.py :: name = name.split("/")[-1]
        Becomes: name = name.split("/")[0]
        """
        assert published_context_window("openai", "openai/gpt-4o") == 128_000


def _ps_client(payload: object, status: int = 200) -> httpx.AsyncClient:
    """A client answering `/api/ps` with `payload` and nothing else."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps", request.url
        return httpx.Response(status, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestWhatTheDaemonReports:
    @pytest.mark.asyncio
    async def test_the_window_comes_from_what_the_daemon_loaded(self) -> None:
        """`/api/ps` reports the window in force, which is the only honest denominator.

        Killed by: src/uclone_x/llm/context_window.py :: tokens = entry.get("context_length")
        Becomes: tokens = entry.get("size")
        """
        store = OllamaContextWindows()
        async with _ps_client(
            {"models": [{"model": "llama3.2:1b", "context_length": 32_768, "size": 2_571_559_239}]}
        ) as client:
            await store.refresh("http://ollama.test", http_client=client)

        assert store.get("http://ollama.test", "llama3.2:1b") == 32_768

    @pytest.mark.asyncio
    async def test_a_model_the_daemon_never_loaded_has_no_window(self) -> None:
        """Not zero, and not another model's figure.

        Killed by: src/uclone_x/llm/context_window.py :: return self._seen.get(self._key(base_url, model))
        Becomes: return self._seen.get(self._key(base_url, model)) or next(iter(self._seen.values()), None)
        """
        store = OllamaContextWindows()
        async with _ps_client(
            {"models": [{"model": "llama3.2:1b", "context_length": 32_768}]}
        ) as c:
            await store.refresh("http://ollama.test", http_client=c)

        assert store.get("http://ollama.test", "qwen3:8b") is None

    @pytest.mark.asyncio
    async def test_an_observation_survives_the_model_unloading(self) -> None:
        """A model drops out of `/api/ps` when its keep-alive expires; its window did not
        change when it did.

        Without this the composer's ring swings between a token fraction and a turn
        fraction while the reader sits still, which reads as a broken ring rather than as
        a model going idle.

        Killed by: src/uclone_x/llm/context_window.py :: for item in cast(list[object], raw):
        Becomes: for item in [*cast(list[object], raw), self._seen.clear()]:
        """
        store = OllamaContextWindows()
        async with _ps_client({"models": [{"model": "qwen3:8b", "context_length": 40_960}]}) as c:
            await store.refresh("http://ollama.test", http_client=c)
        async with _ps_client({"models": []}) as c:
            await store.refresh("http://ollama.test", http_client=c)

        assert store.get("http://ollama.test", "qwen3:8b") == 40_960

    @pytest.mark.asyncio
    async def test_a_daemon_that_is_not_there_leaves_the_store_alone(self) -> None:
        """No window is a fact the surface can state; an exception out of a readout is not.

        Killed by: src/uclone_x/llm/context_window.py :: except Exception as exc:  # never break a readout over a window nobody promised
        Becomes: except ValueError as exc:
        """
        store = OllamaContextWindows()
        store.remember("http://ollama.test", "qwen3:8b", 40_960)

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
            await store.refresh("http://ollama.test", http_client=client)

        assert store.get("http://ollama.test", "qwen3:8b") == 40_960

    @pytest.mark.asyncio
    async def test_a_daemon_too_old_to_report_a_window_reports_none(self) -> None:
        """Ollama gained `context_length` on `/api/ps`; an older one answers without it.

        **The entry with no window is listed first, and that is the whole test.** Asserting
        only that it reports `None` proves nothing: dropping the type check makes
        `remember` compare `None > 0`, the `TypeError` is caught by the guard that keeps a
        missing window from breaking a readout, and the seat still reports `None` -- by
        accident, having also thrown away every model listed after it. So a model that
        *does* report one follows it, and it has to survive.

        Killed by: src/uclone_x/llm/context_window.py :: and isinstance(tokens, int)
        Becomes: and True
        """
        store = OllamaContextWindows()
        async with _ps_client(
            {
                "models": [
                    {"model": "qwen3:8b", "size": 5_000_000},
                    {"model": "llama3.2:1b", "context_length": 32_768},
                ]
            }
        ) as client:
            await store.refresh("http://ollama.test", http_client=client)

        assert store.get("http://ollama.test", "qwen3:8b") is None
        assert store.get("http://ollama.test", "llama3.2:1b") == 32_768

    @pytest.mark.asyncio
    async def test_a_window_of_zero_is_not_remembered(self) -> None:
        """Zero is not a window, and as a denominator it divides to infinity.

        Killed by: src/uclone_x/llm/context_window.py :: if model.strip() and tokens > 0:
        Becomes: if model.strip():
        """
        store = OllamaContextWindows()
        async with _ps_client({"models": [{"model": "qwen3:8b", "context_length": 0}]}) as client:
            await store.refresh("http://ollama.test", http_client=client)

        assert store.get("http://ollama.test", "qwen3:8b") is None

    @pytest.mark.asyncio
    async def test_two_daemons_do_not_share_their_windows(self) -> None:
        """The same model name on another endpoint is another machine's decision.

        Killed by: src/uclone_x/llm/context_window.py :: return (base_url.rstrip("/"), ollama_model_key(model))
        Becomes: return ("", ollama_model_key(model))
        """
        store = OllamaContextWindows()
        store.remember("http://one.test", "qwen3:8b", 40_960)

        assert store.get("http://two.test", "qwen3:8b") is None
        # And a trailing slash is the same endpoint, not a third one.
        assert store.get("http://one.test/", "qwen3:8b") == 40_960


class TestTheCompactionWindow:
    """`compaction_window`, the one resolver the agent's three limit sites share (#1372)."""

    _BASE = "http://localhost:11434"

    def test_a_configured_window_below_the_served_one_is_the_limit(self) -> None:
        """The connector sends `context_limit` as `num_ctx`, so it is what the daemon will
        serve; a larger figure held from an earlier load does not override it.

        Killed by: src/uclone_x/llm/context_window.py :: return configured_tokens  # sent as num_ctx, so it is what the daemon serves
        Becomes: return served
        """
        store = OllamaContextWindows()
        store.remember(self._BASE, "llama3.2:1b", 32_768)

        window = compaction_window(
            "ollama", "llama3.2:1b", base_url=self._BASE, configured=16_384, store=store
        )

        assert window == 16_384

    def test_a_hosted_provider_keeps_the_model_table(self) -> None:
        """Only a server that picks its own window is read from the server."""
        store = OllamaContextWindows()
        assert (
            compaction_window("openai", "gpt-4o", base_url=None, configured=None, store=store)
            == 128_000
        )
        assert (
            compaction_window("openai", "gpt-4o", base_url=None, configured=5_000, store=store)
            == 5_000
        )
