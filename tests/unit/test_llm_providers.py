"""One provider table, one key per provider, and no model id written into a connector.

What these pin, each against the settings file every head reads:

* switching provider never loses a key, and saving one provider's key never touches
  another's -- the single ``llm_api_key`` field used to follow whichever provider was
  saved last;
* a file written before keys were kept per provider migrates: its key belongs to the
  provider it was tagged with, else the provider saved beside it, else to nobody;
* a key is never handed to a provider it was not saved for;
* a key variable in the environment outranks the file, and says which variable it is;
* a connector with no model refuses before the network, in plain words, and one built
  with a model sends it when a request names none.

The session root is a per-test temporary directory (``tests/conftest.py``).
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from uclone_x.agent.models import ProviderFailureKind
from uclone_x.agent.turn_executor import _turn_failure
from uclone_x.errors import LLMCredentialsNotConfiguredError, LLMModelNotConfiguredError
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.factory import (
    create_llm_connector,
    resolve_api_key,
    resolve_deep_model,
)
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.ollama import OLLAMA_ENDPOINT_ENV_VARS, OllamaConnector
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.connectors.saved_choice import (
    api_key_for,
    api_keys,
    delete_api_key,
    read_saved_choice,
    save_api_key,
    save_choice,
    settings_data,
    settings_file,
    update_settings_file,
)
from uclone_x.llm.connectors.vllm import VLLM_ENDPOINT_ENV_VARS
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole
from uclone_x.llm.providers import PROVIDERS, canonical_provider, env_key, env_model, spec_for

_PROVIDER_VARS = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "VLLM_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "GEMINI_MODEL",
    "OLLAMA_MODEL",
    "OLLAMA_INDEPTH_MODEL",
    "OLLAMA_FAST_MODEL",
    "VLLM_MODEL",
    *OLLAMA_ENDPOINT_ENV_VARS,
    *VLLM_ENDPOINT_ENV_VARS,
)

#: Words that would mean a refusal leaked an implementation detail to the person reading it.
_INTERNALS = ("Error", "Exception", "Traceback", "None", "404", "400", "{")


@pytest.fixture(autouse=True)
def _nothing_in_the_environment(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in _PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)


def _write(**values: Any) -> Path:
    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(values), encoding="utf-8")
    return target


def _file() -> dict[str, Any]:
    return json.loads(settings_file().read_text(encoding="utf-8"))


# --- the table --------------------------------------------------------------------------


def test_an_alias_names_the_same_provider_as_its_id() -> None:
    assert canonical_provider(" Google ") == "gemini"
    assert spec_for("google") is PROVIDERS["gemini"]
    assert canonical_provider("acme") is None


def test_the_environment_is_read_with_the_variable_that_carried_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "  AQ.env  ")
    monkeypatch.setenv("OLLAMA_INDEPTH_MODEL", "qwen3:14b")
    assert env_key("gemini") == ("AQ.env", "GOOGLE_API_KEY")
    assert env_model("ollama") == ("qwen3:14b", "OLLAMA_INDEPTH_MODEL")
    assert env_key("ollama") is None


# --- one key per provider -----------------------------------------------------------------


def test_switching_provider_and_back_keeps_the_first_providers_key() -> None:
    """A -> B -> A: the key saved for A is A's key again, and B was never sent it."""
    save_api_key("openai", "sk-openai-0001")
    save_choice(provider="openai", model="gpt-4.1", base_url=None)
    save_choice(provider="ollama", model="qwen3:1.7b", base_url=None)

    ollama = read_saved_choice()
    assert ollama is not None and ollama.api_key is None

    save_choice(provider="openai", model="gpt-4.1", base_url=None)
    back = read_saved_choice()
    assert back is not None and back.api_key == "sk-openai-0001"
    assert _file()["llm_api_keys"] == {"openai": "sk-openai-0001"}


def test_saving_one_providers_key_keeps_the_others() -> None:
    save_api_key("openai", "sk-openai-0001")
    save_api_key("google", "AQ.gemini-0001")
    save_api_key("anthropic", "sk-ant-0001")
    save_api_key("openai", "sk-openai-0002")

    assert _file()["llm_api_keys"] == {
        "openai": "sk-openai-0002",
        "gemini": "AQ.gemini-0001",
        "anthropic": "sk-ant-0001",
    }


def test_removing_one_providers_key_keeps_the_others() -> None:
    save_api_key("openai", "sk-openai-0001")
    save_api_key("gemini", "AQ.gemini-0001")
    delete_api_key("openai")
    assert _file()["llm_api_keys"] == {"gemini": "AQ.gemini-0001"}


def test_an_unknown_provider_or_an_empty_key_is_refused_and_nothing_is_written() -> None:
    with pytest.raises(ValueError, match="There is no provider called 'acme'"):
        save_api_key("acme", "k-0001")
    with pytest.raises(ValueError, match="The key is empty, so nothing was saved."):
        save_api_key("openai", "   ")
    assert not settings_file().exists()


def test_a_key_is_never_returned_for_another_provider() -> None:
    """Killed by: src/uclone_x/llm/connectors/saved_choice.py :: return api_keys(data).get(_provider_id(provider))
    Becomes: return next(iter(api_keys(data).values()), None)
    """
    data = {"llm_provider": "anthropic", "llm_api_keys": {"openai": "sk-openai-0001"}}
    assert api_key_for(data, "openai") == "sk-openai-0001"
    assert api_key_for(data, "anthropic") is None
    assert resolve_api_key("anthropic", data=data) is None


def test_the_factory_refuses_rather_than_send_another_providers_key() -> None:
    _write(llm_provider="anthropic", llm_api_keys={"openai": "sk-openai-0001"})
    with pytest.raises(LLMCredentialsNotConfiguredError):
        create_llm_connector(provider="anthropic")
    built = create_llm_connector(provider="openai", model="gpt-4.1")
    assert built.api_key == "sk-openai-0001"  # type: ignore[attr-defined]


# --- migration of the single legacy key -------------------------------------------------


@pytest.mark.parametrize(
    ("legacy", "owner"),
    [
        ({"llm_api_key_provider": "openai", "llm_provider": "ollama"}, "openai"),
        ({"llm_provider": "google"}, "gemini"),
    ],
    ids=["tagged", "untagged-with-provider"],
)
def test_a_legacy_key_is_read_as_its_owners_and_moved_there_on_the_first_write(
    legacy: dict[str, str], owner: str
) -> None:
    """Killed by: src/uclone_x/llm/connectors/saved_choice.py :: _fold_legacy_key(data, replace=False)
    Becomes: pass
    """
    _write(llm_api_key="k-legacy-0001", **legacy)

    assert api_keys(settings_data()) == {owner: "k-legacy-0001"}

    update_settings_file({"llm_provider": "anthropic"})

    data = _file()
    assert data["llm_api_keys"] == {owner: "k-legacy-0001"}
    assert "llm_api_key" not in data and "llm_api_key_provider" not in data
    assert api_key_for(data, "anthropic") is None


def test_a_legacy_key_with_no_owner_stays_put_and_is_sent_nowhere() -> None:
    """A provider saved later does not claim a key that was saved for none.

    Killed by: src/uclone_x/llm/connectors/saved_choice.py :: data[_LEGACY_KEY_PROVIDER] = ""
    Becomes: pass
    """
    _write(llm_api_key="k-orphan-0001")
    assert api_keys(settings_data()) == {}

    update_settings_file({"llm_provider": "openai"})

    data = _file()
    assert data["llm_api_key"] == "k-orphan-0001"
    assert "llm_api_keys" not in data
    assert api_key_for(data, "openai") is None
    choice = read_saved_choice()
    assert choice is not None and choice.api_key is None


def test_a_per_provider_key_outranks_a_stale_legacy_key_for_the_same_provider() -> None:
    _write(
        llm_provider="openai",
        llm_api_key="sk-old-0001",
        llm_api_keys={"openai": "sk-new-0002"},
    )
    assert api_key_for(settings_data(), "openai") == "sk-new-0002"
    update_settings_file({"llm_model": "gpt-4.1"})
    assert _file()["llm_api_keys"] == {"openai": "sk-new-0002"}


def test_the_settings_file_stays_private_to_its_owner() -> None:
    save_api_key("openai", "sk-openai-0001")
    assert settings_file().stat().st_mode & 0o077 == 0


# --- the environment overrides and says so ----------------------------------------------


def test_an_environment_key_outranks_the_saved_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: src/uclone_x/llm/connectors/factory.py :: if env_key(provider) is not None or data is None:
    Becomes: if data is None:
    """
    _write(llm_provider="openai", llm_api_keys={"openai": "sk-saved-0001"})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-0002")

    built = create_llm_connector(model="gpt-4.1")

    assert isinstance(built, OpenAIConnector)
    assert built.api_key == "sk-env-0002"
    assert env_key("openai") == ("sk-env-0002", "OPENAI_API_KEY")
    assert _file()["llm_api_keys"] == {"openai": "sk-saved-0001"}  # read, never written


# --- the deep model ---------------------------------------------------------------------


def test_the_saved_model_goes_only_to_its_own_provider() -> None:
    data = {"llm_provider": "google", "llm_model": "gemini-2.5-pro"}
    assert resolve_deep_model("gemini", data=data) == "gemini-2.5-pro"
    assert resolve_deep_model("openai", data=data) is None
    assert resolve_deep_model("gemini", "gemini-2.5-flash", data) == "gemini-2.5-flash"


def test_the_model_variable_outranks_the_saved_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: src/uclone_x/llm/connectors/factory.py :: return found[0]
    Becomes: pass
    """
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    data = {"llm_provider": "gemini", "llm_model": "gemini-2.5-pro"}
    assert resolve_deep_model("gemini", data=data) == "gemini-2.5-flash"


def test_the_fast_model_is_read_from_the_settings_file() -> None:
    _write(llm_provider="ollama", llm_model="qwen3:8b", llm_model_fast="qwen3:1.7b")
    choice = read_saved_choice()
    assert choice is not None
    assert (choice.model, choice.model_fast) == ("qwen3:8b", "qwen3:1.7b")


# --- connectors: no model is refused; a configured model is sent ------------------------


def _request(model: str | None = None) -> LLMRequest:
    return LLMRequest(model=model, messages=(ChatMessage(role=MessageRole.USER, content="hi"),))


_HOSTED: tuple[tuple[str, Callable[..., BaseLLMConnector]], ...] = (
    ("OpenAI", OpenAIConnector),
    ("Anthropic", AnthropicConnector),
    ("Google", GeminiConnector),
)


@pytest.mark.parametrize(("display", "connector_cls"), _HOSTED, ids=[d for d, _ in _HOSTED])
@pytest.mark.parametrize("streamed", [False, True], ids=["generate", "stream"])
def test_a_connector_with_no_model_refuses_before_the_network_in_plain_words(
    display: str, connector_cls: Callable[..., BaseLLMConnector], streamed: bool
) -> None:
    """No request goes out, and the sentence names the provider; the head adds where to choose.

    Killed by: src/uclone_x/llm/connectors/base.py :: raise LLMModelNotConfiguredError(provider)
    Becomes: return "default"
    """
    reached: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        reached.append(str(request.url))
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = connector_cls(api_key="k-0001", http_client=client)

    async def drive() -> None:
        if streamed:
            async for _chunk in connector.stream(_request("default")):
                pass
        else:
            await connector.generate(_request())

    with pytest.raises(LLMModelNotConfiguredError) as caught:
        asyncio.run(drive())

    message = str(caught.value)
    assert message == f"No model is chosen for {display}."
    assert not any(word in message for word in _INTERNALS)
    assert reached == []


@pytest.mark.parametrize("streamed", [False, True], ids=["generate", "stream"])
def test_ollama_with_no_model_refuses_like_the_hosted_connectors(
    monkeypatch: pytest.MonkeyPatch, streamed: bool
) -> None:
    """Ollama used to fill in a model id of its own; with none chosen it now refuses.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: return next((found.strip() for found in candidates if found and found.strip()), None)
    Becomes: return next((found.strip() for found in candidates if found and found.strip()), "qwen3:8b")
    """
    for var in PROVIDERS["ollama"].model_env_vars:
        monkeypatch.delenv(var, raising=False)
    reached: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        reached.append(str(request.url))
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = OllamaConnector(base_url="http://127.0.0.1:11434", http_client=client)

    async def drive() -> None:
        if streamed:
            async for _chunk in connector.stream(_request("default")):
                pass
        else:
            await connector.generate(_request())

    with pytest.raises(LLMModelNotConfiguredError) as caught:
        asyncio.run(drive())

    message = str(caught.value)
    assert message == "No model is chosen for Ollama."
    assert not any(word in message for word in _INTERNALS)
    assert reached == []
    # What the agent reads to attribute a turn, and the window probe, report no model
    # rather than a guessed one, and send nothing.
    assert connector._default_model is None  # pyright: ignore[reportPrivateUsage]
    assert asyncio.run(connector.observe_context_window()) is None
    assert reached == []


def test_ollama_sends_the_model_it_was_built_with(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saved `qwen3:8b` keeps working: the factory hands it to the connector."""
    for var in PROVIDERS["ollama"].model_env_vars:
        monkeypatch.delenv(var, raising=False)
    connector = OllamaConnector(base_url="http://127.0.0.1:11434", model="qwen3:8b")
    assert connector._resolve_model(None) == "qwen3:8b"  # pyright: ignore[reportPrivateUsage]
    assert connector._default_model == "qwen3:8b"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(("display", "connector_cls"), _HOSTED, ids=[d for d, _ in _HOSTED])
def test_a_connector_sends_its_own_model_when_the_request_names_none(
    display: str, connector_cls: Callable[..., BaseLLMConnector]
) -> None:
    """Killed by: src/uclone_x/llm/connectors/base.py :: chosen = named_model(requested) or named_model(configured)
    Becomes: chosen = named_model(requested)
    """
    del display
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url) + " " + request.content.decode("utf-8"))
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = connector_cls(api_key="k-0001", model="model-under-test", http_client=client)

    with pytest.raises(Exception):  # noqa: B017 -- the 500 is not what is under test
        asyncio.run(connector.generate(_request()))

    assert sent and all("model-under-test" in line for line in sent)


def test_an_unchosen_model_ends_the_turn_as_model_unavailable_with_the_plain_sentence() -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: no_model = _in_cause_chain(exc, LLMModelNotConfiguredError)
    Becomes: no_model = None
    """
    stop, error, failure = _turn_failure(LLMModelNotConfiguredError("Google"), "not_started", "a1")

    assert stop == "model_unavailable"
    assert error == "No model is chosen for Google."
    assert failure is not None
    assert failure.kind is ProviderFailureKind.MODEL_UNAVAILABLE
    assert (failure.provider, failure.retryable) == ("Google", False)
