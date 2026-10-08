"""The model gateway: connections, model refs, the default models and a connector per ref.

model-gateway.md steps 1-3 (§3.1-3.6, §4 "Unit"). Every connection here is a `mock` row or
an Ollama row whose listing is replaced, so no model server is needed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from uclone_x.core.models import AgentLLMConfig, PersonaDefinition
from uclone_x.errors import ProviderFailureKind
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.catalog import CatalogEntry
from uclone_x.llm.connections import (
    ConnectionError_,
    ConnectionKeyMissingError,
    ModelRef,
    ModelRefError,
    saved_connections,
)
from uclone_x.llm.connectors.saved_choice import (
    add_connection,
    api_key_for,
    read_saved_choice,
    remove_connection,
    save_api_key,
    save_default_models,
    settings_data,
    update_connection,
)
from uclone_x.llm.gateway import (
    DefaultBinding,
    ModelGateway,
    ModelRefUnavailableError,
    RefusingConnector,
    UnsupportedConnectionError,
    connections_in_effect,
    default_models_in_effect,
)
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole

#: Every variable the environment rules (S4) read, cleared so the machine's own do not leak in.
_ENV = (
    "LLM_PROVIDER",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VLLM_API_KEY",
    "VLLM_BASE_URL",
    "OLLAMA_BASE_URL",
    "OLLAMA_INDEPTH_BASE_URL",
    "OLLAMA_FAST_BASE_URL",
    "LOCAL_LLM_BASE_URL",
    "OLLAMA_HOST",
    "GEMINI_MODEL",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "VLLM_MODEL",
    "OLLAMA_MODEL",
    "OLLAMA_INDEPTH_MODEL",
    "OLLAMA_FAST_MODEL",
    "GEMINI_BASE_URL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
)

#: Words a plain refusal never carries (provider-model-catalog G7).
_INTERNALS = ("Error", "Exception", "Traceback", "404", "500", "None", "ModelRef")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _plain(text: str) -> None:
    for word in _INTERNALS:
        assert word not in text, f"{word!r} reached a plain refusal: {text}"


class _Factory:
    """A connector factory that hands out one recording mock per (connection kind, address)."""

    def __init__(self) -> None:
        self.built: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> MockLLMConnector:
        self.built.append(kwargs)
        return MockLLMConnector(default_model=kwargs.get("model") or "mock")


# -- ModelRef (§3.1, decision 1) --


class TestModelRef:
    def test_it_splits_at_the_first_slash(self) -> None:
        ref = ModelRef.parse("gpu-box/meta-llama/Llama-3.3-70B")
        assert (ref.connection_id, ref.model) == ("gpu-box", "meta-llama/Llama-3.3-70B")
        assert str(ref) == "gpu-box/meta-llama/Llama-3.3-70B"

    @pytest.mark.parametrize("bare", ["gemini-3.8-pro", "/qwen3", "ollama/", ""])
    def test_a_bare_id_is_refused_in_plain_words(self, bare: str) -> None:
        """
        Killed by: src/uclone_x/llm/connections.py :: if not sep or not head.strip() or not tail.strip():
        Becomes: if False:
        """
        with pytest.raises(ModelRefError) as refused:
            ModelRef.parse(bare)
        assert "which connection" in str(refused.value)
        _plain(str(refused.value))


class TestPersonaRefusesABareModel:
    def test_a_persona_naming_a_bare_id_does_not_load(self) -> None:
        """
        Killed by: src/uclone_x/core/models.py :: if not (sep and head.strip() and tail.strip()):
        Becomes: if False:
        """
        with pytest.raises(ValueError, match="which connection"):
            PersonaDefinition(
                name="scout",
                role="Scout",
                system_prompt="x",
                llm_config=AgentLLMConfig(model_name="gemini-3.8-pro"),
            )

    def test_a_ref_and_an_auto_picture_model_load(self) -> None:
        """
        Killed by: src/uclone_x/core/models.py :: if label == "image_model" and clean == "auto":
        Becomes: if False:
        """
        persona = PersonaDefinition(
            name="scout",
            role="Scout",
            system_prompt="x",
            llm_config=AgentLLMConfig(
                model_name="ollama/qwen3:14b", fast_model="gemini/flash", image_model="auto"
            ),
        )
        assert persona.llm_config.image_model == "auto"


# -- The file (§3.2) --


class TestConnectionsInTheFile:
    def test_the_first_of_a_kind_takes_the_kind_and_another_its_label(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/connectors/saved_choice.py :: if clean_kind not in taken and not _clean(label)
        Becomes: if True
        """
        path = tmp_path / "settings.json"
        first = add_connection("ollama", path=path)
        second = add_connection(
            "ollama", label="GPU box", base_url="http://10.0.0.5:11434", path=path
        )
        assert (first.id, second.id) == ("ollama", "gpu-box")
        rows = saved_connections(settings_data(path))
        assert [(r.id, r.base_url) for r in rows] == [
            ("ollama", "http://127.0.0.1:11434"),
            ("gpu-box", "http://10.0.0.5:11434"),
        ]

    def test_a_key_saved_for_one_connection_never_touches_another(self, tmp_path: Path) -> None:
        """S3, one key per connection.

        Killed by: src/uclone_x/llm/connectors/saved_choice.py :: rows = [replace(row, key=key) if row.id == conn_id else row for row in rows]
        Becomes: rows = [replace(row, key=key) for row in rows]
        """
        path = _write(
            tmp_path / "settings.json",
            {
                "connections": [
                    {"id": "openai", "kind": "openai", "key": "sk-first"},
                    {"id": "proxy", "kind": "openai", "base_url": "https://proxy.example/v1"},
                ]
            },
        )
        save_api_key("proxy", "sk-second", path=path)
        assert api_key_for(settings_data(path), "openai") == "sk-first"
        assert api_key_for(settings_data(path), "proxy") == "sk-second"

    def test_a_file_with_only_the_old_keys_has_no_connection(self, tmp_path: Path) -> None:
        """No migration (owner ruling 2026-09-28): the old keys are never read.

        Killed by: src/uclone_x/llm/connections.py :: raw = data.get(CONNECTIONS_KEY)
        Becomes: raw = data.get(CONNECTIONS_KEY) or [{"id": "gemini", "kind": data.get("llm_provider")}]
        """
        path = _write(
            tmp_path / "settings.json",
            {
                "llm_provider": "gemini",
                "llm_model": "gemini-3.8-pro",
                "llm_api_keys": {"gemini": "AQ.secret"},
            },
        )
        assert saved_connections(settings_data(path)) == []
        assert read_saved_choice(path) is None

    def test_a_removed_or_changed_row_leaves_the_others(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/connectors/saved_choice.py :: kept = [row for row in rows if row.id != conn_id]
        Becomes: kept = [row for row in rows if row.id == conn_id]
        """
        path = tmp_path / "settings.json"
        add_connection("ollama", path=path)
        add_connection("vllm", base_url="http://10.0.0.5:8000/v1", path=path)
        update_connection("vllm", label="Box", path=path)
        remove_connection("ollama", path=path)
        rows = saved_connections(settings_data(path))
        assert [(r.id, r.label) for r in rows] == [("vllm", "Box")]
        with pytest.raises(ConnectionError_) as refused:
            remove_connection("ollama", path=path)
        _plain(str(refused.value))

    def test_a_default_that_is_not_a_ref_is_refused_before_writing(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/connectors/saved_choice.py :: refuse_bare_model(value, slot=slot, allow_auto=slot == "image")
        Becomes: pass
        """
        path = tmp_path / "settings.json"
        with pytest.raises(ModelRefError):
            save_default_models({"deep": "gemini-3.8-pro"}, path=path)
        assert not path.exists()


class TestEnvironment:
    def test_a_key_variable_supplies_the_key_of_the_connection_of_its_kind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S4 as revised: it overrides the saved key, and never another row's.

        Killed by: src/uclone_x/llm/gateway.py :: changes["key"], changes["key_env_var"] = key
        Becomes: pass
        """
        monkeypatch.setenv("GEMINI_API_KEY", "AQ.from-env")
        data = {
            "connections": [
                {"id": "gemini", "kind": "gemini", "key": "AQ.saved"},
                {"id": "gemini-work", "kind": "gemini", "key": "AQ.work"},
            ]
        }
        rows = {c.id: c for c in connections_in_effect(data)}
        assert (rows["gemini"].key, rows["gemini"].key_env_var) == ("AQ.from-env", "GEMINI_API_KEY")
        assert rows["gemini"].source == "settings"
        assert rows["gemini-work"].key == "AQ.work"

    def test_with_none_saved_it_makes_an_ephemeral_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: base = saved or Connection(id=kind, kind=kind, source="env")
        Becomes: base = saved or Connection(id=kind, kind=kind)
        """
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        rows = connections_in_effect({})
        assert [(c.id, c.kind, c.source) for c in rows] == [("openai", "openai", "env")]

    def test_a_model_variable_is_read_as_kind_slash_id_over_the_deep_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: return replace(saved, deep=f"{kind}/{model}", env_vars={"deep": variable})
        Becomes: return replace(saved, deep=model, env_vars={"deep": variable})
        """
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5")
        data = {"default_models": {"deep": "ollama/qwen3:14b"}}
        defaults = default_models_in_effect(data)
        assert defaults.deep == "openai/gpt-5"
        assert defaults.env_vars == {"deep": "OPENAI_MODEL"}


# -- The gateway (§3.3) --


def _gateway(tmp_path: Path, data: dict[str, Any], **kwargs: Any) -> tuple[ModelGateway, _Factory]:
    factory = _Factory()
    path = _write(tmp_path / "settings.json", data)
    return ModelGateway(path, connector_factory=factory, **kwargs), factory


_TWO = {
    "connections": [{"id": "mock", "kind": "mock"}, {"id": "box", "kind": "mock"}],
    "default_models": {"deep": "mock/deep-model", "fast": "box/fast-model"},
}


class TestResolve:
    def test_an_own_ref_wins_and_an_empty_slot_takes_the_default(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: if own and own.strip():
        Becomes: if False:
        """
        gateway, _ = _gateway(tmp_path, _TWO)
        assert str(gateway.resolve("box/own", "deep")) == "box/own"
        assert str(gateway.resolve(None, "deep")) == "mock/deep-model"
        assert str(gateway.resolve(None, "fast")) == "box/fast-model"

    def test_an_empty_fast_default_means_deep(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/connections.py :: return self.fast or self.deep
        Becomes: return self.fast
        """
        gateway, _ = _gateway(tmp_path, {"default_models": {"deep": "mock/deep-model"}})
        assert str(gateway.resolve(None, "fast")) == "mock/deep-model"

    def test_nothing_saved_resolves_to_nothing(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: chosen = self.defaults().get(slot)
        Becomes: chosen = self.defaults().get(slot) or "mock/remembered"
        """
        gateway, _ = _gateway(tmp_path, {})
        assert gateway.resolve(None, "deep") is None


class TestConnectorFor:
    def test_it_is_kept_per_connection_and_dropped_when_the_row_changes(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: if kept is None or kept[0] != fingerprint:
        Becomes: if kept is None:
        """
        gateway, factory = _gateway(tmp_path, _TWO)
        ref = ModelRef.parse("box/fast-model")
        first = gateway.connector_for(ref)
        assert gateway.connector_for(ref) is first
        assert len(factory.built) == 1
        update_connection("box", base_url="http://127.0.0.1:9", path=gateway.settings_path)
        assert gateway.connector_for(ref) is not first
        assert factory.built[-1]["base_url"] == "http://127.0.0.1:9"

    def test_it_builds_with_the_connection_s_own_key_and_address(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: provider=conn.kind,
        Becomes: provider="mock",
        """
        gateway, factory = _gateway(
            tmp_path,
            {
                "connections": [
                    {"id": "proxy", "kind": "openai", "key": "sk-p", "base_url": "http://h"}
                ]
            },
        )
        gateway.connector_for(ModelRef.parse("proxy/gpt-5"))
        built = factory.built[0]
        assert (built["provider"], built["api_key"], built["base_url"], built["model"]) == (
            "openai",
            "sk-p",
            "http://h",
            "gpt-5",
        )

    def test_the_real_factory_gates_a_paid_connector(self, tmp_path: Path) -> None:
        """llm-token-gateway G8: every paid connector is built through `create_llm_connector`.

        Killed by: src/uclone_x/llm/gateway.py :: from uclone_x.llm.connectors.factory import create_llm_connector
        Becomes: from uclone_x.llm.connectors.factory import _build_connector as create_llm_connector
        """
        path = _write(
            tmp_path / "settings.json",
            {"connections": [{"id": "openai", "kind": "openai", "key": "sk-test"}]},
        )
        connector = ModelGateway(path).connector_for(ModelRef.parse("openai/gpt-5"))
        assert getattr(type(connector), "_usage_gated", False) is True

    def test_a_removed_connection_is_refused_never_replaced(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: raise ModelRefUnavailableError(ref)
        Becomes: return self.default_deep()[0]
        """
        gateway, _ = _gateway(tmp_path, _TWO)
        with pytest.raises(ModelRefUnavailableError) as refused:
            gateway.connector_for(ModelRef.parse("gone/qwen3"))
        assert refused.value.kind is ProviderFailureKind.MODEL_UNAVAILABLE
        assert "gone/qwen3" in str(refused.value)
        _plain(str(refused.value))


class TestBind:
    def test_a_pinned_ref_runs_on_its_own_connection(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: return SeatBinding(llm=llm, llm_config=config, pinned_ref=pinned, deep_ref=deep_ref)
        Becomes: return SeatBinding(llm=llm, llm_config=config, pinned_ref=None, deep_ref=deep_ref)
        """
        gateway, _ = _gateway(tmp_path, _TWO)
        seat = gateway.bind(AgentLLMConfig(model_name="box/own"))
        assert seat.llm is gateway.connector_for(ModelRef.parse("box/own"))
        assert seat.llm_config.model_name == "own"
        assert seat.pinned_ref == "box/own"

    def test_a_follower_takes_the_default_and_its_fast_only_on_the_same_connection(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: if fast_ref is not None and fast_ref.connection_id == deep_conn_id
        Becomes: if fast_ref is not None
        """
        gateway, _ = _gateway(tmp_path, _TWO)
        seat = gateway.bind(AgentLLMConfig())
        assert seat.llm is gateway.connector_for(ModelRef.parse("mock/deep-model"))
        assert seat.llm_config.model_name == "deep-model"
        # The default fast is on `box`, another connection: one seat holds one connector.
        assert seat.llm_config.fast_model is None
        assert seat.pinned_ref is None

    def test_a_missing_connection_gives_a_refusing_connector(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: raise ModelRefUnavailableError(ref)
        Becomes: pass
        """
        gateway, _ = _gateway(tmp_path, _TWO)
        seat = gateway.bind(AgentLLMConfig(model_name="gone/qwen3"))
        assert isinstance(seat.llm, RefusingConnector)
        request = LLMRequest(
            model="qwen3", messages=(ChatMessage(role=MessageRole.USER, content="hi"),)
        )
        with pytest.raises(ModelRefUnavailableError):
            asyncio.run(seat.llm.generate(request))

    def test_a_default_binding_answers_followers_and_a_bare_flag_model(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: if pinned is not None and binding is not None and "/" not in pinned:
        Becomes: if False:
        """
        mine = MockLLMConnector()
        gateway, _ = _gateway(tmp_path, _TWO, default_binding=DefaultBinding(mine, "m"))
        assert gateway.bind(AgentLLMConfig()).llm is mine
        flagged = gateway.bind(AgentLLMConfig(model_name="qwen3:8b"))
        assert (flagged.llm, flagged.llm_config.model_name) == (mine, "qwen3:8b")
        assert gateway.bind(AgentLLMConfig(model_name="box/own")).llm is not mine


class TestModelSet:
    def test_it_is_the_union_and_a_connection_that_cannot_list_shows_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider-model-catalog G3: no remembered list fills the gap.

        Killed by: src/uclone_x/llm/gateway.py :: Listing("unreachable", unreachable)
        Becomes: Listing("connected", None, (CatalogEntry(id="remembered"),))
        """
        import uclone_x.llm.gateway as gateway_module

        async def listing(base_url: str | None = None, *_a: Any, **_k: Any) -> Any:
            return [CatalogEntry(id="qwen3:14b")] if base_url == "http://127.0.0.1:11434" else None

        monkeypatch.setattr(gateway_module, "list_ollama_entries", listing)
        gateway, _ = _gateway(
            tmp_path,
            {
                "connections": [
                    {"id": "ollama", "kind": "ollama", "base_url": "http://127.0.0.1:11434"},
                    {"id": "gpu-box", "kind": "ollama", "base_url": "http://10.0.0.5:11434"},
                ]
            },
        )
        models = asyncio.run(gateway.model_set("chat"))
        ok, down = models.groups
        assert [m.ref for m in ok.models] == ["ollama/qwen3:14b"]
        assert (down.status, down.models) == ("unreachable", ())
        assert down.detail is not None
        _plain(down.detail)


# -- Rooms: a connector per seat, routing on the default fast model (§3.4, §4 "Integration") --


class _Recording(MockLLMConnector):
    """A mock that remembers the model each request named."""

    def __init__(self, name: str) -> None:
        super().__init__(default_model="mock", default_response=f"from {name}")
        self.name = name
        self.models: list[str | None] = []

    async def generate(self, request: LLMRequest) -> Any:
        self.models.append(request.model)
        return await super().generate(request)

    async def stream(self, request: LLMRequest) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.models.append(request.model)
        async for chunk in super().stream(request):
            yield chunk


def _room_app(tmp_path: Path, settings: dict[str, Any]) -> tuple[Any, Any, dict[str, _Recording]]:
    from fastapi.testclient import TestClient

    from uclone_x.ui.app import create_ui_app

    storage = tmp_path / "sessions"
    storage.mkdir(parents=True, exist_ok=True)
    _write(storage / "settings.json", settings)
    app = create_ui_app(static_dir=tmp_path / "static", storage_dir=storage, workspace_dir=tmp_path)
    mgr = app.state.session_manager
    built: dict[str, _Recording] = {}

    def factory(**kwargs: Any) -> _Recording:
        name = str(kwargs["base_url"])
        built.setdefault(name, _Recording(name))
        return built[name]

    mgr.gateway._factory = factory  # pyright: ignore[reportPrivateUsage]
    return app, TestClient(app), built


def _clone(client: Any, name: str, model: str | None) -> str:
    draft = {"name": name, "role": "Helper", "system_prompt": "You help.", "model_name": model}
    res = client.post("/api/clones", json=draft)
    assert res.status_code == 201, res.text
    return str(res.json()["persona"]["id"])


_ROOM_SETTINGS: dict[str, Any] = {
    "connections": [
        {"id": "mock", "kind": "mock", "base_url": "http://a.local"},
        {"id": "box", "kind": "mock", "base_url": "http://b.local"},
    ],
    "default_models": {"deep": "mock/deep-model", "fast": "box/fast-model"},
}


class TestTwoSeatsOnTwoConnections:
    def test_each_seat_reaches_its_own_connection_and_routing_the_default_fast(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: src/uclone_x/agent/clone_builder.py :: seat = app.gateway.bind(llm_config)
        Becomes: seat = app.gateway.bind(AgentLLMConfig())
        """
        from uclone_x.room.models import ParticipantKind

        app, client, built = _room_app(tmp_path, _ROOM_SETTINGS)
        stack = app.state.room_stack
        with client:
            pinned = _clone(client, "alpha", "box/own-model")
            follower = _clone(client, "beta", None)
            room_id = stack.service.create("Two connections").room_id
            stack.service.add_participant(room_id, "user", kind=ParticipantKind.HUMAN)
            stack.service.add_participant(room_id, pinned)
            stack.service.add_participant(room_id, follower)
            state = stack.store.load(room_id)
            seats = {p.id: p for p in state.participants}
            alpha = asyncio.run(stack.resolve_agent(state, seats[pinned]))
            beta = asyncio.run(stack.resolve_agent(state, seats[follower]))
            asyncio.run(alpha.execute_turn("hi"))
            asyncio.run(beta.execute_turn("hi"))
            routing = stack.orchestrator(state)._selectors  # pyright: ignore[reportPrivateUsage]

        assert "own-model" in built["http://b.local"].models
        assert "deep-model" in built["http://a.local"].models
        assert "own-model" not in built["http://a.local"].models
        providers = [s._provider for s in routing if hasattr(s, "_provider")]  # pyright: ignore[reportPrivateUsage]
        assert all(p is built["http://b.local"] for p in providers)

    def test_a_removed_connection_refuses_naming_clone_model_and_the_one_action(
        self, tmp_path: Path
    ) -> None:
        """§3.6: never re-run on the default; say which clone, which model, and offer one action.

        Killed by: src/uclone_x/room/orchestrator.py :: provider_failure = self._with_own_model(speaker.id, result.provider_failure)
        Becomes: provider_failure = result.provider_failure
        """
        from uclone_x.room.models import ParticipantKind

        app, client, built = _room_app(tmp_path, _ROOM_SETTINGS)
        stack = app.state.room_stack
        with client:
            pinned = _clone(client, "alpha", "box/own-model")
            assert client.delete("/api/connections/box").status_code == 200
            room_id = stack.service.create("Gone").room_id
            stack.service.add_participant(room_id, "user", kind=ParticipantKind.HUMAN)
            stack.service.add_participant(room_id, pinned)
            state = stack.store.load(room_id)
            asyncio.run(stack.orchestrator(state).post(room_id, "user", "@alpha hello"))
            landed = stack.store.load(room_id).transcript[-1]

        failure = landed.provider_failure
        assert failure is not None
        assert (failure.kind, failure.retryable) == (ProviderFailureKind.MODEL_UNAVAILABLE, False)
        assert (failure.clone, failure.model_ref, failure.action) == (
            "alpha",
            "box/own-model",
            "use_system_default",
        )
        assert "alpha" in failure.message and "box/own-model" in failure.message
        _plain(failure.message)
        # Nothing was sent to the default connection in its place (G7).
        assert built.get("http://a.local") is None or built["http://a.local"].models == []


# -- The routes (§3.7.1) --


_CONNECTION_KEYS = {
    "id",
    "kind",
    "label",
    "base_url",
    "key_set",
    "key_masked",
    "source",
    "env_var",
    "key_env_var",
    "paid",
    "status",
    "detail",
    "model_count",
}


def _api(tmp_path: Path, settings: dict[str, Any]) -> tuple[Any, Any]:
    from fastapi.testclient import TestClient

    from uclone_x.ui.app import create_ui_app

    storage = tmp_path / "sessions"
    storage.mkdir(parents=True, exist_ok=True)
    _write(storage / "settings.json", settings)
    app = create_ui_app(static_dir=tmp_path / "static", storage_dir=storage, workspace_dir=tmp_path)
    return app, TestClient(app)


class TestConnectionRoutes:
    def test_the_listing_has_the_contract_s_shape_and_never_a_key(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/ui/app.py :: "key_masked": self._mask_key(conn.key) if conn.key else None,
        Becomes: "key_masked": conn.key,
        """
        _, client = _api(
            tmp_path,
            {"connections": [{"id": "openai", "kind": "openai", "key": "sk-secret-0001"}]},
        )
        data = client.get("/api/connections").json()
        (conn,) = data["connections"]
        assert set(conn) == _CONNECTION_KEYS
        assert (conn["key_set"], conn["key_masked"], conn["paid"]) == (True, "sk-...0001", True)
        assert "sk-secret-0001" not in json.dumps(data)
        assert [k["kind"] for k in data["kinds"]] == [
            "gemini",
            "openai",
            "anthropic",
            "ollama",
            "vllm",
            "comfyui",
            "remote_gpu",
        ]
        ollama = next(k for k in data["kinds"] if k["kind"] == "ollama")
        assert (ollama["needs_key"], ollama["needs_base_url"]) == (False, True)

    def test_add_change_and_remove_keep_every_other_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Killed by: src/uclone_x/ui/app.py :: fields = {name: changes[name] for name in ("label", "base_url", "key") if name in changes}
        Becomes: fields = {}
        """
        import uclone_x.llm.gateway as gateway_module

        async def listing(*_a: Any, **_k: Any) -> Any:
            return [CatalogEntry(id="qwen3:14b")]

        monkeypatch.setattr(gateway_module, "list_ollama_entries", listing)
        app, client = _api(
            tmp_path, {"connections": [{"id": "openai", "kind": "openai", "key": "sk-keep"}]}
        )
        added = client.post(
            "/api/connections",
            json={"kind": "ollama", "label": "GPU box", "base_url": "http://10.0.0.5:11434"},
        )
        assert added.status_code == 200, added.text
        assert (added.json()["id"], added.json()["status"], added.json()["model_count"]) == (
            "gpu-box",
            "connected",
            1,
        )
        patched = client.patch("/api/connections/gpu-box", json={"label": "Box"})
        assert patched.json()["label"] == "Box"
        assert client.delete("/api/connections/gpu-box").json() == {"removed": "gpu-box"}
        path = app.state.session_manager.settings_file
        rows = saved_connections(settings_data(path))
        assert [(r.id, r.key) for r in rows] == [("openai", "sk-keep")]

    def test_a_connection_set_by_a_variable_is_refused_naming_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Killed by: src/uclone_x/ui/app.py :: if conn.source == "env":
        Becomes: if False:
        """
        monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        _, client = _api(tmp_path, {})
        (conn,) = client.get("/api/connections").json()["connections"]
        assert (conn["source"], conn["env_var"]) == ("env", "OLLAMA_HOST")
        for res in (
            client.delete("/api/connections/ollama"),
            client.patch("/api/connections/ollama", json={"label": "x"}),
        ):
            assert res.status_code == 400
            assert "OLLAMA_HOST" in res.json()["detail"]
            _plain(res.json()["detail"])

    def test_dependents_names_the_clones_and_defaults_on_a_connection(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/ui/app.py :: ("image_model", cfg.image_model),
        Becomes: ("image_model", None),
        """
        _, client = _api(
            tmp_path,
            {
                "connections": [{"id": "box", "kind": "comfyui", "base_url": "http://127.0.0.1:1"}],
                "default_models": {"image": "box/Illustrious-XL-v0.1"},
            },
        )
        draft = {
            "name": "alpha",
            "role": "Helper",
            "system_prompt": "x",
            "image_model": "box/anillustrious_v4",
        }
        assert client.post("/api/clones", json=draft).status_code == 201
        deps = client.get("/api/connections/box/dependents").json()
        assert deps["defaults"] == ["image"]
        assert [(c["name"], c["slots"]) for c in deps["clones"]] == [("alpha", ["image_model"])]


class TestDefaultModelsRoute:
    def test_a_ref_the_set_does_not_list_is_refused_naming_the_connection(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: src/uclone_x/ui/app.py :: if not any(entry.ref == str(ref) for entry in group.models):
        Becomes: if False:
        """
        _, client = _api(tmp_path, {"connections": [{"id": "mock", "kind": "mock"}]})
        res = client.post("/api/settings", json={"default_models": {"deep": "mock/not-there"}})
        assert res.status_code == 400
        assert "Mock does not list the model not-there" in res.json()["detail"]
        gone = client.post("/api/settings", json={"default_models": {"deep": "gone/model"}})
        assert "no connection called gone" in gone.json()["detail"]

    def test_null_clears_a_slot_and_fast_follows_deep_again(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/connectors/saved_choice.py :: stored.pop(slot, None)
        Becomes: pass
        """
        app, client = _api(
            tmp_path,
            {
                "connections": [{"id": "mock", "kind": "mock"}],
                "default_models": {"deep": "mock/mock-llm", "fast": "mock/mock-gpt-4o"},
            },
        )
        res = client.post("/api/settings", json={"default_models": {"fast": None}})
        assert res.status_code == 200, res.text
        assert res.json()["default_models"]["fast"] is None
        assert str(app.state.session_manager.gateway.resolve(None, "fast")) == "mock/mock-llm"


class TestCloneRoute:
    def test_every_clone_entry_carries_its_three_model_slots(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/agent/persona_store.py :: llm_config["image_model"] = draft.image_model
        Becomes: pass
        """
        _, client = _api(tmp_path, {"connections": [{"id": "box", "kind": "mock"}]})
        draft = {
            "name": "alpha",
            "role": "Helper",
            "system_prompt": "x",
            "model_name": "box/deep",
            "fast_model": "box/fast",
            "image_model": "auto",
        }
        assert client.post("/api/clones", json=draft).status_code == 201
        listed = client.get("/api/clones/alpha").json()
        entry = listed.get("persona", listed)
        assert (entry["model_name"], entry["fast_model"], entry["image_model"]) == (
            "box/deep",
            "box/fast",
            "auto",
        )

    def test_an_unknown_connection_or_a_bare_id_is_refused(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/ui/app.py :: if (conn := session_mgr.gateway.connection(conn_id)) is None:
        Becomes: if (conn := session_mgr.gateway.connection(conn_id)) is None and False:
        """
        _, client = _api(tmp_path, {"connections": [{"id": "box", "kind": "mock"}]})
        base = {"name": "alpha", "role": "Helper", "system_prompt": "x"}
        unknown = client.post("/api/clones", json={**base, "model_name": "gone/qwen3"})
        assert unknown.status_code == 422
        assert "no connection called gone" in unknown.json()["detail"]
        bare = client.post("/api/clones", json={**base, "model_name": "qwen3"})
        assert bare.status_code == 422
        assert "which connection" in json.dumps(bare.json())


class TestOldRoutesAreGone:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("delete", "/api/settings/api-keys/gemini"),
            ("post", "/api/models/catalog"),
            # The ComfyUI address check went with the address (step 5): a ComfyUI is a
            # connection now, checked by `POST /api/connections/{id}/check`.
            ("post", "/api/settings/test"),
        ],
    )
    def test_the_pre_gateway_routes_answer_nothing(
        self, tmp_path: Path, method: str, path: str
    ) -> None:
        """No kill declaration: it asserts an absence, and no one-line change brings a
        deleted route back. `main` before the gateway (2b98a12b) still declares both."""
        _, client = _api(tmp_path, {})
        assert getattr(client, method)(path).status_code in (404, 405)


# -- Settings' per-connection Ollama install and remove read what is installed (#2167 part 3) --

_TWO_OLLAMAS: dict[str, Any] = {
    "connections": [
        {"id": "ollama", "kind": "ollama", "base_url": "http://127.0.0.1:11434"},
        {"id": "gpu-box", "kind": "ollama", "base_url": "http://10.0.0.5:11434"},
    ]
}


class TestInstalledModelsRoute:
    def test_it_lists_the_named_connections_models_embedders_included(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Killed by: src/uclone_x/ui/app.py :: base_url = _ollama_address({"connection_id": connection_id})
        Becomes: base_url = None
        """
        import uclone_x.ui.app as app_module

        asked: list[str | None] = []

        async def entries(base_url: str | None = None, *_a: Any, **_k: Any) -> Any:
            asked.append(base_url)
            return [CatalogEntry(id="qwen3:14b"), CatalogEntry(id="bge-m3", chat_capable=False)]

        monkeypatch.setattr(app_module, "list_ollama_entries", entries)
        _, client = _api(tmp_path, _TWO_OLLAMAS)
        res = client.get("/api/models/installed?connection_id=gpu-box")
        assert res.status_code == 200, res.text
        assert res.json()["models"] == [
            {"id": "qwen3:14b", "chat": True},
            {"id": "bge-m3", "chat": False},
        ]
        assert asked == ["http://10.0.0.5:11434"]

    def test_an_unknown_connection_and_a_silent_server_are_said_plainly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Killed by: src/uclone_x/ui/app.py :: if entries is None:
        Becomes: if False:
        """
        import uclone_x.ui.app as app_module

        async def silent(*_a: Any, **_k: Any) -> Any:
            return None

        monkeypatch.setattr(app_module, "list_ollama_entries", silent)
        _, client = _api(tmp_path, _TWO_OLLAMAS)
        unknown = client.get("/api/models/installed?connection_id=nowhere")
        assert unknown.status_code == 400
        _plain(unknown.json()["detail"])
        down = client.get("/api/models/installed?connection_id=ollama")
        assert (down.status_code, down.json()["code"]) == (502, "unreachable")
        _plain(down.json()["detail"])


# -- A connection of a kind this build does not know (P6: reported, never dropped) --

_FUTURE_ROW: dict[str, Any] = {"id": "future", "kind": "martian-llm", "zone": "olympus"}
_NO_ID_ROW: dict[str, Any] = {"kind": "ollama", "base_url": "http://10.0.0.9:11434"}


class TestUnsupportedKind:
    def test_it_is_listed_saying_it_is_not_supported(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/connections.py :: kind=kind if kind is not None else (written_kind or ""),
        Becomes: kind=kind if kind is not None else None,
        """
        _, client = _api(tmp_path, {"connections": [_FUTURE_ROW]})
        (conn,) = client.get("/api/connections").json()["connections"]
        assert (conn["id"], conn["kind"], conn["status"]) == (
            "future",
            "martian-llm",
            "unsupported",
        )
        assert conn["paid"] is False
        assert "This kind of connection is not supported" in conn["detail"]
        assert "martian-llm" in conn["detail"]
        _plain(conn["detail"])
        (group,) = client.get("/api/models?capability=chat").json()["groups"]
        assert (group["connection_id"], group["status"], group["models"]) == (
            "future",
            "unsupported",
            [],
        )

    def test_a_turn_naming_it_is_refused_in_plain_words(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: raise UnsupportedConnectionError(ref, conn)
        Becomes: pass
        """
        gateway, factory = _gateway(tmp_path, {"connections": [_FUTURE_ROW]})
        seat = gateway.bind(AgentLLMConfig(model_name="future/big-model"))
        assert isinstance(seat.llm, RefusingConnector)
        request = LLMRequest(
            model="big-model", messages=(ChatMessage(role=MessageRole.USER, content="hi"),)
        )
        with pytest.raises(UnsupportedConnectionError) as refused:
            asyncio.run(seat.llm.generate(request))
        assert refused.value.kind is ProviderFailureKind.MODEL_UNAVAILABLE
        assert "future/big-model" in str(refused.value)
        assert "does not support" in str(refused.value)
        _plain(str(refused.value))
        assert factory.built == []  # nothing was built for it, so nothing was sent anywhere

    def test_a_save_elsewhere_keeps_it_and_an_unreadable_row_as_written(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: src/uclone_x/llm/connections.py :: return [*(conn.file_row() for conn in connections), *_read_rows(current)[1]]
        Becomes: return [conn.file_row() for conn in connections]
        """
        path = _write(tmp_path / "settings.json", {"connections": [_FUTURE_ROW, _NO_ID_ROW]})
        add_connection("vllm", base_url="http://10.0.0.5:8000/v1", path=path)
        saved = json.loads(path.read_text(encoding="utf-8"))["connections"]
        assert _FUTURE_ROW in saved
        assert _NO_ID_ROW in saved

    def test_it_can_be_removed_but_not_changed(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/connectors/saved_choice.py :: _refuse_unsupported(rows[index])
        Becomes: pass
        """
        path = _write(tmp_path / "settings.json", {"connections": [_FUTURE_ROW]})
        with pytest.raises(ConnectionError_) as refused:
            update_connection("future", label="Mars", path=path)
        assert "does not support" in str(refused.value)
        _plain(str(refused.value))
        assert json.loads(path.read_text(encoding="utf-8"))["connections"] == [_FUTURE_ROW]
        remove_connection("future", path=path)
        assert saved_connections(settings_data(path)) == []


# -- A supported row keeps the fields this build does not know (#2167 part 4) --

_NEWER_ROW: dict[str, Any] = {
    "id": "gpu-box",
    "kind": "vllm",
    "base_url": "http://10.0.0.5:8000/v1",
    "key": "vk-test-0001",
    "timeout_s": 90,
    "headers": {"x-team": "lab"},
}


class TestUnknownFieldsOnASupportedRow:
    def test_a_change_to_the_row_keeps_them(self, tmp_path: Path) -> None:
        """A newer version's field on a row this build reads survives a write to that row.

        Killed by: src/uclone_x/llm/connections.py :: row: dict[str, Any] = {"id": self.id, "kind": self.kind, **unknown}
        Becomes: row: dict[str, Any] = {"id": self.id, "kind": self.kind}
        """
        path = _write(tmp_path / "settings.json", {"connections": [_NEWER_ROW]})
        update_connection("gpu-box", label="GPU box", path=path)
        (saved,) = json.loads(path.read_text(encoding="utf-8"))["connections"]
        assert saved == {**_NEWER_ROW, "label": "GPU box"}

    def test_a_save_elsewhere_keeps_them(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/llm/connections.py :: raw=dict(row),
        Becomes: raw=dict(row) if kind is None else None,
        """
        path = _write(tmp_path / "settings.json", {"connections": [_NEWER_ROW]})
        add_connection("ollama", base_url="http://127.0.0.1:11434", path=path)
        saved = json.loads(path.read_text(encoding="utf-8"))["connections"]
        assert _NEWER_ROW in saved

    def test_a_cleared_known_field_is_not_brought_back(self, tmp_path: Path) -> None:
        """The known fields come from the connection, so clearing the key clears it.

        Killed by: src/uclone_x/llm/connections.py :: if name not in _KNOWN_ROW_FIELDS
        Becomes: if name not in {"id", "kind"}
        """
        path = _write(tmp_path / "settings.json", {"connections": [_NEWER_ROW]})
        update_connection("gpu-box", key="", path=path)
        (saved,) = json.loads(path.read_text(encoding="utf-8"))["connections"]
        assert "key" not in saved
        assert (saved["timeout_s"], saved["headers"]) == (90, {"x-team": "lab"})


# -- A key stays with its connection (S3): a second row of a kind never borrows one --

_VLLM_PAIR: dict[str, Any] = {
    "connections": [
        {"id": "vllm", "kind": "vllm", "base_url": "http://10.0.0.4:8000/v1", "key": "vk-A-0001"},
        {"id": "gpu-box", "kind": "vllm", "base_url": "http://10.0.0.5:8000/v1"},
    ],
    "default_models": {"deep": "gpu-box/qwen"},
}

_OPENAI_PAIR: dict[str, Any] = {
    "connections": [
        {"id": "openai", "kind": "openai", "key": "sk-A-0001"},
        {"id": "proxy", "kind": "openai", "base_url": "https://proxy.example/v1"},
    ],
    "default_models": {"deep": "proxy/gpt-5"},
}


class TestAKeyStaysWithItsConnection:
    def test_a_keyless_vllm_row_gets_neither_the_other_rows_key_nor_the_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Killed by: src/uclone_x/llm/gateway.py :: api_key=connection_key(conn),
        Becomes: api_key=conn.key,
        """
        monkeypatch.setenv("VLLM_API_KEY", "vk-env-0001")
        path = _write(tmp_path / "settings.json", _VLLM_PAIR)
        connector = ModelGateway(path).connector_for(ModelRef.parse("gpu-box/qwen"))
        assert getattr(connector, "base_url", "").startswith("http://10.0.0.5:8000")
        assert getattr(connector, "api_key", None) not in ("vk-A-0001", "vk-env-0001")
        assert connector._auth_headers() == {}  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]

    def test_a_keyless_openai_row_is_refused_in_plain_words_never_given_another_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Killed by: src/uclone_x/llm/connections.py :: if spec is not None and spec.requires_key:
        Becomes: if False:
        """
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-0001")
        path = _write(tmp_path / "settings.json", _OPENAI_PAIR)
        gateway = ModelGateway(path)
        with pytest.raises(ConnectionKeyMissingError) as refused:
            gateway.connector_for(ModelRef.parse("proxy/gpt-5"))
        assert "proxy has no key" in str(refused.value)
        _plain(str(refused.value))
        seat = gateway.bind(AgentLLMConfig(model_name="proxy/gpt-5"))
        assert isinstance(seat.llm, RefusingConnector)
        # The kind's own row still gets the variable's key (S4), and only it.
        own = gateway.connector_for(ModelRef.parse("openai/gpt-5"))
        assert getattr(own, "api_key", None) == "sk-env-0001"

    def test_a_terminal_command_on_a_keyless_row_borrows_no_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The CLI path (`create_llm_connector` from the saved default), as the gateway's.

        Killed by: src/uclone_x/llm/connectors/factory.py :: if key is None and not kind_row:
        Becomes: if False:
        """
        from uclone_x.llm.connectors.factory import create_llm_connector

        monkeypatch.setenv("VLLM_API_KEY", "vk-env-0001")
        vllm = _write(tmp_path / "vllm.json", _VLLM_PAIR)
        connector = create_llm_connector(saved_choice_file=vllm)
        assert getattr(connector, "base_url", "").startswith("http://10.0.0.5:8000")
        assert getattr(connector, "api_key", None) not in ("vk-A-0001", "vk-env-0001")

        # A key variable would name its kind's own connection before the saved choice is
        # read (the factory's step 3), so the sibling's key here is the saved one.
        proxy = _write(tmp_path / "proxy.json", _OPENAI_PAIR)
        with pytest.raises(ConnectionKeyMissingError):
            create_llm_connector(saved_choice_file=proxy)


class TestADefaultChangeReachesOpenSeats:
    def test_a_cleared_or_moved_default_leaves_no_old_connector_or_fast_id(
        self, tmp_path: Path
    ) -> None:
        """§3.4: a seat that follows the default takes the new one on its next turn.

        Killed by: src/uclone_x/room/resolver.py :: agent.rebind_llm(
        Becomes: agent.hot_reload_llm(
        """
        from uclone_x.room.models import ParticipantKind

        settings = {
            "connections": [
                {"id": "mock", "kind": "mock", "base_url": "http://a.local"},
                {"id": "box", "kind": "mock", "base_url": "http://b.local"},
            ],
            "default_models": {"deep": "mock/mock-gpt-4o", "fast": "mock/mock-llm"},
        }
        app, client, built = _room_app(tmp_path, settings)
        stack = app.state.room_stack
        with client:
            follower = _clone(client, "beta", None)
            room_id = stack.service.create("Moves").room_id
            stack.service.add_participant(room_id, "user", kind=ParticipantKind.HUMAN)
            stack.service.add_participant(room_id, follower)
            state = stack.store.load(room_id)
            seat = next(p for p in state.participants if p.id == follower)
            agent = asyncio.run(stack.resolve_agent(state, seat))
            assert agent.llm is built["http://a.local"]
            assert agent.config.llm_config.fast_model == "mock-llm"

            # Moved to another connection: the fast default stays on the old one, so the
            # seat (one connector) has no fast model -- not the old connection's id.
            moved = client.post(
                "/api/settings", json={"default_models": {"deep": "box/mock-gpt-4o"}}
            )
            assert moved.status_code == 200, moved.text
            assert agent.llm is built["http://b.local"]
            assert agent.config.llm_config.model_name == "mock-gpt-4o"
            assert agent.config.llm_config.fast_model is None

            # Cleared: no connector and no model, never the one it had.
            for slot in ("fast", "deep"):
                cleared = client.post("/api/settings", json={"default_models": {slot: None}})
                assert cleared.status_code == 200, cleared.text
            assert agent.llm is None
            assert agent.config.llm_config.model_name is None
