"""What `uclone_x.llm.context_window` may and may not claim about a model's window.

The subject here is not arithmetic. It is the one thing this module exists to refuse: a
window figure that was not measured. A window is the denominator of a ring, and a ring
drawn to the wrong fraction looks exactly like a ring drawn to the right one.
"""

from __future__ import annotations

import httpx
import pytest

from uclone_x.llm.catalog import CatalogEntry, read_catalog
from uclone_x.llm.context_window import (
    PUBLISHED_CONTEXT_WINDOWS,
    ListedContextWindows,
    OllamaContextWindows,
    compaction_window,
    hosted_context_window,
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

    def test_current_claude_models_have_their_published_windows(self) -> None:
        """The four models Anthropic's models overview lists as current on 2026-09-28.

        The table named only retired `claude-3-*` IDs, so every current Claude seat had no
        window. A dated snapshot resolves through its alias; a retired ID a saved seat may
        still name keeps resolving.

        Killed by: src/uclone_x/llm/context_window.py :: "claude-opus-5-5": 1_000_000,
        Becomes: "claude-opus-5-5": 2_000_000,
        Killed by: src/uclone_x/llm/context_window.py :: "claude-haiku-4-5": 200_000,
        Becomes: "claude-haiku-4-x": 200_000,
        Killed by: src/uclone_x/llm/context_window.py :: "claude-3-5-sonnet": 200_000,
        Becomes: "claude-3-5-sonnex": 200_000,
        """
        assert published_context_window("anthropic", "claude-fable-5-1") == 1_000_000
        assert published_context_window("anthropic", "claude-opus-5-5") == 1_000_000
        assert published_context_window("anthropic", "claude-sonnet-5") == 1_000_000
        assert published_context_window("anthropic", "claude-haiku-4-5-20251001") == 200_000
        assert published_context_window("anthropic", "claude-3-5-sonnet-20241022") == 200_000

    def test_legacy_claude_models_still_served_have_their_published_windows(self) -> None:
        """Legacy models Anthropic still serves, from each model's own page on 2026-09-28 (#1917).

        They had no entry, so a seat on any of them showed no window. Opus 4.5 and Sonnet
        4.5 are 200K and are named by dated IDs that resolve through their aliases; the
        others are dateless and 1M. `claude-opus-5` and `claude-fable-5` are prefixes of
        the current `claude-opus-5-5` and `claude-fable-5-1`, and each still resolves.

        Killed by: src/uclone_x/llm/context_window.py :: "claude-opus-5": 1_000_000,
        Becomes: "claude-opus-x": 1_000_000,
        Killed by: src/uclone_x/llm/context_window.py :: "claude-opus-4-5": 200_000,
        Becomes: "claude-opus-4-x": 200_000,
        Killed by: src/uclone_x/llm/context_window.py :: "claude-sonnet-4-6": 1_000_000,
        Becomes: "claude-sonnet-4-x": 1_000_000,
        """
        one_million = {
            "claude-fable-5",
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        }
        for model in one_million:
            assert published_context_window("anthropic", model) == 1_000_000, model
        assert published_context_window("anthropic", "claude-opus-4-5-20251101") == 200_000
        assert published_context_window("anthropic", "claude-sonnet-4-5-20250929") == 200_000
        assert published_context_window("anthropic", "claude-fable-5-1") == 1_000_000
        assert published_context_window("anthropic", "claude-opus-5-5") == 1_000_000

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

        Killed by: src/uclone_x/llm/context_window.py :: "anthropic": {
        Becomes: "ollama": {"llama3.2": 131_072}, "anthropic": {
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

    def test_a_vertex_snapshot_resolves_through_its_alias(self) -> None:
        """Vertex names a Claude snapshot `claude-opus-4-5@20251101`, and `@` was no delimiter (#1920).

        IDs from Anthropic's "Claude on Google Cloud" model table, read 2026-09-28. Each
        had no window, so a seat on Vertex showed none.

        Killed by: src/uclone_x/llm/context_window.py :: _DELIMITERS: frozenset[str] = frozenset({"-", ".", ":", "_", "/", "@"})
        Becomes: _DELIMITERS: frozenset[str] = frozenset({"-", ".", ":", "_", "/", "~"})
        """
        assert published_context_window("anthropic", "claude-opus-4-5@20251101") == 200_000
        assert published_context_window("anthropic", "claude-sonnet-4-5@20250929") == 200_000
        assert published_context_window("anthropic", "claude-haiku-4-5@20251001") == 200_000

    def test_an_at_suffix_matches_its_own_key_and_not_a_longer_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`claude-opus-5-5@...` is Opus 5.5, and `claude-opus-5@...` is Opus 5, not Opus 5.5.

        Both publish 1M today, so the two windows are made to differ here; otherwise a
        match on the wrong key passes. Without `@` as a delimiter, `claude-opus-5-5@date`
        fell through to `claude-opus-5` by its `-`.

        Killed by: src/uclone_x/llm/context_window.py :: _DELIMITERS: frozenset[str] = frozenset({"-", ".", ":", "_", "/", "@"})
        Becomes: _DELIMITERS: frozenset[str] = frozenset({"-", ".", ":", "_", "/", "~"})
        """
        monkeypatch.setitem(PUBLISHED_CONTEXT_WINDOWS["anthropic"], "claude-opus-5-5", 555)
        monkeypatch.setitem(PUBLISHED_CONTEXT_WINDOWS["anthropic"], "claude-opus-5", 5)
        assert published_context_window("anthropic", "claude-opus-5-5@20260101") == 555
        assert published_context_window("anthropic", "claude-opus-5@x") == 5
        assert published_context_window("anthropic", "claude-opus-5@x") != 555

    def test_a_bedrock_id_resolves_with_or_without_a_region_prefix(self) -> None:
        r"""Bedrock prefixes the model with `anthropic.`, and an inference profile adds a geography (#1920).

        IDs from Anthropic's two Bedrock pages and AWS's inference-profile page, read
        2026-09-28. An inference-profile ARN reaches the same ID after the gateway split.

        Killed by: src/uclone_x/llm/context_window.py :: _BEDROCK_PREFIX = re.compile(r"^(?:[a-z][a-z-]*\.)?anthropic\.")
        Becomes: _BEDROCK_PREFIX = re.compile(r"^(?:[a-z][a-z-]*\.)?anthropiq\.")
        """
        assert published_context_window("anthropic", "anthropic.claude-opus-5-5") == 1_000_000
        assert (
            published_context_window("anthropic", "anthropic.claude-opus-4-5-20251101-v1:0")
            == 200_000
        )
        assert (
            published_context_window("anthropic", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
            == 200_000
        )
        assert (
            published_context_window("anthropic", "global.anthropic.claude-opus-4-6-v1")
            == 1_000_000
        )
        arn = (
            "arn:aws:bedrock:us-east-1:111122223333:inference-profile/"
            "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        assert published_context_window("anthropic", arn) == 200_000

    def test_a_bedrock_prefix_is_only_stripped_whole(self) -> None:
        r"""A name that merely contains `anthropic.` past its first segment is not a Bedrock ID.

        Killed by: src/uclone_x/llm/context_window.py :: _BEDROCK_PREFIX = re.compile(r"^(?:[a-z][a-z-]*\.)?anthropic\.")
        Becomes: _BEDROCK_PREFIX = re.compile(r"^(?:[a-z][a-z.]*\.)?anthropic\.")
        """
        assert published_context_window("anthropic", "a.b.anthropic.claude-opus-5-5") is None


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

        Killed by: src/uclone_x/llm/context_window.py :: return sent  # sent as num_ctx, so it is what the daemon serves
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


class TestTheListedWindow:
    """A hosted model's window from its provider's own listing (#1978, catalogue §3.2)."""

    def test_a_gemini_window_comes_from_the_listing_and_not_a_table(self) -> None:
        """Gemini's listing reports `inputTokenLimit` for every model, so no table row is
        kept for it: its rows named only retired `gemini-1.5-*` models, and PR #1806's
        attempt to update them is what §3.6 rejects. The figure here is one no table held,
        so it can only have come from the listing.

        Killed by: src/uclone_x/llm/context_window.py :: from_listing = store.get(provider, model)
        Becomes: from_listing = None
        """
        store = ListedContextWindows()
        store.remember("gemini", [CatalogEntry(id="gemini-2.0-flash", context_window=123_456)])

        assert hosted_context_window("gemini", "gemini-2.0-flash", listed=store) == 123_456
        # The request's own spelling of the same id.
        assert hosted_context_window("gemini", "models/gemini-2.0-flash", listed=store) == 123_456
        assert "gemini" not in PUBLISHED_CONTEXT_WINDOWS
        assert (
            hosted_context_window("gemini", "gemini-2.0-flash", listed=ListedContextWindows())
            is None
        )

    def test_the_listing_wins_over_the_published_table(self) -> None:
        """Where both know a model, the provider's live answer for this key is the one used.

        Killed by: src/uclone_x/llm/context_window.py :: if from_listing is not None:
        Becomes: if from_listing is None:
        """
        store = ListedContextWindows()
        store.remember("anthropic", [CatalogEntry(id="claude-opus-5-5", context_window=777_000)])

        assert hosted_context_window("anthropic", "claude-opus-5-5", listed=store) == 777_000
        # A model the listing did not describe still has its published figure.
        assert hosted_context_window("anthropic", "claude-haiku-4-5", listed=store) == 200_000

    def test_a_listed_id_is_matched_exactly_and_never_by_a_neighbour(self) -> None:
        """A listing names every id it serves, so a prefix neighbour's figure is a guess."""
        store = ListedContextWindows()
        store.remember("gemini", [CatalogEntry(id="gemini-2.5-flash", context_window=1_048_576)])

        assert store.get("gemini", "gemini-2.5-flash-lite") is None
        assert store.get("gemini", "gemini-2.5") is None
        assert store.get("openai", "gemini-2.5-flash") is None

    def test_google_is_the_same_provider_as_gemini(self) -> None:
        """Settings accepts `google` as Gemini's name; the catalogue files it under `gemini`.

        Killed by: src/uclone_x/llm/context_window.py :: return canonical_provider(provider) or provider.strip().lower()
        Becomes: return provider.strip().lower()
        """
        store = ListedContextWindows()
        store.remember("gemini", [CatalogEntry(id="gemini-2.5-pro", context_window=1_048_576)])

        assert store.get("google", "gemini-2.5-pro") == 1_048_576

    def test_an_entry_that_reports_no_window_leaves_the_window_unknown(self) -> None:
        """A missing or zero figure is the provider saying nothing, not a window of zero.

        Killed by: src/uclone_x/llm/context_window.py :: if name and tokens is not None and tokens > 0:
        Becomes: if name and tokens is not None and tokens >= 0:
        """
        store = ListedContextWindows()
        store.remember(
            "anthropic",
            [
                CatalogEntry(id="claude-9-unreleased", context_window=None),
                CatalogEntry(id="claude-9-zero", context_window=0),
            ],
        )

        assert store.get("anthropic", "claude-9-unreleased") is None
        assert store.get("anthropic", "claude-9-zero") is None
        assert hosted_context_window("anthropic", "claude-9-zero", listed=store) is None

    def test_an_unknown_hosted_model_has_no_compaction_window(self) -> None:
        """With no listing figure and no table row, the trigger's window is `None`, and the
        agent falls back to `compaction_threshold_tokens` (`CompactionDriver`) rather than
        to a family figure such as the compactor's old bare `"gemini"` row."""
        empty = ListedContextWindows()

        assert (
            compaction_window(
                "gemini", "gemini-9-unreleased", base_url=None, configured=None, listed=empty
            )
            is None
        )
        assert (
            compaction_window(
                "gemini", "gemini-9-unreleased", base_url=None, configured=4_096, listed=empty
            )
            == 4_096
        )

    @pytest.mark.asyncio
    async def test_reading_the_catalogue_remembers_each_listed_window(self) -> None:
        """Settings reads the listing; the compaction trigger and the room readout use it.

        Killed by: src/uclone_x/llm/catalog.py :: (windows if windows is not None else LISTED_CONTEXT_WINDOWS).remember(provider, listed)
        Becomes: pass
        """
        store = ListedContextWindows()

        async def lister() -> list[CatalogEntry]:
            return [CatalogEntry(id="gemini-2.5-flash", context_window=1_048_576)]

        result = await read_catalog(
            provider="gemini",
            display_provider="Google",
            lister=lister,
            recommend=lambda _provider, _entries: None,
            windows=store,
        )

        assert result.status == "live"
        assert (
            compaction_window(
                "gemini", "gemini-2.5-flash", base_url=None, configured=None, listed=store
            )
            == 1_048_576
        )
