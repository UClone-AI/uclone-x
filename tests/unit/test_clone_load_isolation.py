"""One clone that cannot be read does not stop the others, and migration keeps no 0.3.1 model (#2220).

Two faults, found while checking the 0.4.0 CHANGELOG:

* A hand-edited `clone.yaml` naming a model without its connection (a bare id, refused by
  the gateway's ref rule) raised out of the clone store's load, so the registry could not
  be built and *every* clone failed to load. The one unreadable clone is now listed as
  unreadable, with a plain reason; a turn that names it is refused in plain words; the
  others load.
* 0.3.1 stored a model as a bare id, and the migration carried it over as written. An id
  with a `/` in it (a vLLM id, `Qwen/Qwen3-8B`) then read as the ref `<connection>/<model>`,
  pointing the clone at a connection `Qwen` that does not exist, and nothing was logged.
  An old `agents/<name>/` home is 0.3.1's, so the migration keeps none of its persona's
  models: the clone follows the default model, and the migration log and the server log
  say plainly to choose its model again. A workspace persona file may have been written
  for today's build, where `ollama/custom-model` is the right spelling, so its import
  drops only what is not a ref and says how it read each ref it kept.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from uclone_x.agent import clone_builder
from uclone_x.agent.clone_builder import AppScope, CloneUnreadableError, build_clone
from uclone_x.agent.clone_store import ensure_clone_store
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.persona_store import BUILTIN_PERSONAS_DIR
from uclone_x.agent.session import SessionStore
from uclone_x.cli import main
from uclone_x.cli.commands import run
from uclone_x.core.agent_home import (
    AGENTS_DIR_ENV_VAR,
    AgentHome,
    clone_handles,
    create_clone,
    default_agents_root,
)
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import PlainRefusalError
from uclone_x.llm import MockLLMConnector
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.app import create_ui_app

GOOD = "handle: good\nrole: Helper\nsystem_prompt: I help.\n"
BARE = (
    "handle: bad\nrole: Helper\nsystem_prompt: I help.\nllm_config:\n  model_name: gemini-3-pro\n"
)

#: What no sentence a person reads may carry: a path, a class or exception name, a clone
#: id, or a field name from the file.
INTERNALS = ("/", "Error", "Traceback", "agt_", "model_name", "llm_config", "clone.yaml")


def _two_clones(root: Path) -> tuple[str, str]:
    good = create_clone("good", GOOD, root=root)
    bad = create_clone("bad", BARE, root=root)
    return good.path.name, bad.path.name


def _assert_plain(sentence: str) -> None:
    for word in INTERNALS:
        assert word not in sentence, (word, sentence)


def _host(tmp_path: Path) -> HostDependencies:
    return HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(default_response="Done."),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )


# --- one unreadable clone ---------------------------------------------------------------


def test_one_clone_with_a_bare_model_does_not_stop_the_others_loading(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/clone_store.py :: except (PersonaLoadError, ValidationError) as exc:
    Becomes: except NotImplementedError as exc:
    """
    root = default_agents_root()
    good_id, bad_id = _two_clones(root)
    before = (root / bad_id / "clone.yaml").read_bytes()

    registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)

    assert registry.get_persona("good") is not None
    assert registry.get_persona(good_id) is not None
    assert registry.get_persona("bad") is None
    reason = registry.unreadable_reason("bad")
    assert reason is not None
    assert registry.unreadable_reason(bad_id) == reason
    assert registry.unreadable_reason("good") is None
    # Nothing is written to or removed from the folder it could not read (P6).
    assert (root / bad_id / "clone.yaml").read_bytes() == before


def test_the_reason_says_what_to_change_in_plain_words(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/clone_store.py :: if persona_model_problem(*_raw_models(fields)) is not None:
    Becomes: if False:
    """
    _two_clones(default_agents_root())

    reason = PersonaRegistry(workspace_root=tmp_path, include_defaults=False).unreadable_reason(
        "bad"
    )

    assert reason is not None
    assert "connection" in reason
    _assert_plain(reason)


def test_a_clone_file_with_another_fault_is_unreadable_too(tmp_path: Path) -> None:
    """Any persona the loader refuses is one clone's fault, not the registry's."""
    root = default_agents_root()
    create_clone("good", GOOD, root=root)
    create_clone(
        "odd", "handle: odd\nrole: R\nsystem_prompt: S\nappend_default_prompt: maybe\n", root=root
    )

    registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)

    assert registry.get_persona("good") is not None
    reason = registry.unreadable_reason("odd")
    assert reason is not None
    _assert_plain(reason)


def test_the_load_fault_is_logged_with_its_detail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The person reads a plain sentence; the log keeps the path and the loader's words."""
    _, bad_id = _two_clones(default_agents_root())

    with caplog.at_level(logging.WARNING, logger="uclone_x.agent.clone_store"):
        PersonaRegistry(workspace_root=tmp_path, include_defaults=False)

    logged = caplog.text
    assert bad_id in logged
    assert "gemini-3-pro" in logged


def test_a_turn_that_names_the_unreadable_clone_is_refused_in_plain_words(
    tmp_path: Path,
) -> None:
    """Without the refusal it would answer as a clone with no persona, the fallback.

    Killed by: src/uclone_x/agent/clone_builder.py :: raise CloneUnreadableError(
    Becomes: logger.debug(
    """
    _, bad_id = _two_clones(default_agents_root())
    app = AppScope(
        host=_host(tmp_path),
        workspace_root=tmp_path,
        persona_registry=PersonaRegistry(workspace_root=tmp_path, include_defaults=False),
    )

    for clone_id in ("bad", bad_id):
        with pytest.raises(CloneUnreadableError) as caught:
            build_clone(app, clone_id=clone_id, session_id="sess")
        assert isinstance(caught.value, PlainRefusalError)
        assert "bad" in str(caught.value)
        _assert_plain(str(caught.value).replace("'bad'", ""))

    built = build_clone(app, clone_id="good", session_id="sess")
    assert built.persona is not None and built.persona.name == "good"


# --- the listing ------------------------------------------------------------------------


@pytest.mark.usefixtures("builtin_personas_absent")
def test_the_clone_list_marks_the_unreadable_clone_and_serves_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/clones.py :: if entry.state is AgentHomeState.UNREADABLE or found is not None:
    Becomes: if entry.state is AgentHomeState.UNREADABLE:
    """
    agents_root = tmp_path / "agents"
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))
    good_id, bad_id = _two_clones(agents_root)
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path / "static",
            storage_dir=tmp_path / "sessions",
            llm=MockLLMConnector(),
        )
    )

    response = client.get("/api/clones")

    assert response.status_code == 200, response.text
    rows = {row["handle"]: row for row in response.json()["clones"]}
    assert rows["good"]["status"] == "dormant"
    assert rows["good"]["id"] == good_id
    assert rows["good"]["system_prompt"] == "I help."
    assert rows["bad"]["status"] == "unreadable"
    assert rows["bad"]["id"] == bad_id
    assert "system_prompt" not in rows["bad"]
    assert "connection" in rows["bad"]["reason"]
    _assert_plain(rows["bad"]["reason"])


# --- the CLI ----------------------------------------------------------------------------


def _cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    llm = MagicMock()
    llm.provider_name = "mock"

    def _connector(**kwargs: Any) -> MagicMock:
        return llm

    def _no_binder(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(run, "create_llm_connector", _connector)
    monkeypatch.setattr(clone_builder, "connector_tool_binder", _no_binder)


def test_the_cli_refuses_the_unreadable_clone_in_plain_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/cli/commands/run.py :: except (UnknownToolsModuleError, CloneUnreadableError) as exc:
    Becomes: except UnknownToolsModuleError as exc:
    """
    _two_clones(default_agents_root())
    _cli_env(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        main.app, ["run", "bad", "--provider", "mock", "--prompt", "hi", "--cwd", str(tmp_path)]
    )

    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    assert "connection" in result.output
    _assert_plain(result.output.replace("\n", " "))


# --- the 0.3.1 migration ----------------------------------------------------------------


def _legacy_clone_with_model(tmp_path: Path, name: str, model: str) -> tuple[Path, str]:
    """A 0.3.1 install: `agents/<name>/` and a workspace persona file naming `model`."""
    root = default_agents_root()
    agent_id = AgentHome.for_legacy_name(name, root).agent_id()
    workspace = tmp_path / "ws"
    folder = workspace / ".uclone" / "personas"
    folder.mkdir(parents=True)
    (folder / f"{name}.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "role": "Helper",
                "system_prompt": f"I am {name}.",
                "llm_config": {"model_name": model, "fast_model": model},
            }
        ),
        encoding="utf-8",
    )
    del agent_id
    return workspace, name


def _migrated(root: Path, handle: str) -> dict[str, Any]:
    (agent_id,) = clone_handles(root)[handle]
    loaded = yaml.safe_load((root / agent_id / "clone.yaml").read_text(encoding="utf-8"))
    return cast(dict[str, Any], loaded)


def _log(root: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(root.glob(".migration-*.log")))


@pytest.mark.parametrize("model", ["Qwen/Qwen3-8B", "gemini-3-pro"])
def test_a_0_3_1_model_is_not_carried_over_and_the_log_says_to_choose_again(
    tmp_path: Path, model: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A 0.3.1 id is never a ref; kept, `Qwen/Qwen3-8B` named a connection called `Qwen`.

    Killed by: src/uclone_x/agent/clone_store.py :: mapping, dropped, _ = _earlier_models(mapping, keep_refs=False)
    Becomes: mapping, dropped, _ = _earlier_models(mapping, keep_refs=True)
    """
    root = default_agents_root()
    workspace, name = _legacy_clone_with_model(tmp_path, "mine", model)

    with caplog.at_level(logging.WARNING, logger="uclone_x.agent.clone_store"):
        ensure_clone_store(workspace, builtin_dir=BUILTIN_PERSONAS_DIR)

    clone = _migrated(root, name)
    # Migrated with the persona's own fields, not the default clone's.
    assert clone["role"] == "Helper" and clone["system_prompt"] == f"I am {name}."
    llm = cast(dict[str, Any], clone.get("llm_config") or {})
    assert "model_name" not in llm and "fast_model" not in llm
    log = _log(root)
    assert model in log
    assert "choose its model again" in log
    assert model in caplog.text and "choose its model again" in caplog.text
    # The clone loads: the registry reads it and serves it on the default model.
    registry = PersonaRegistry(workspace_root=workspace, include_defaults=False)
    assert registry.get_persona(name) is not None
    assert registry.unreadable_reason(name) is None


def test_a_0_3_1_workspace_file_with_a_model_is_imported_without_it(tmp_path: Path) -> None:
    """A workspace persona with no old home is imported too, rather than left out.

    Killed by: src/uclone_x/agent/clone_store.py :: stripped, _, _ = _earlier_models(mapping, keep_refs=True)
    Becomes: stripped = mapping
    """
    root = default_agents_root()
    folder = tmp_path / "ws" / ".uclone" / "personas"
    folder.mkdir(parents=True)
    (folder / "fresh.yaml").write_text(
        "name: fresh\nrole: R\nsystem_prompt: S\nllm_config:\n  model_name: gemini-3-pro\n",
        encoding="utf-8",
    )

    ensure_clone_store(tmp_path / "ws", builtin_dir=None, install=False)

    assert "model_name" not in (_migrated(root, "fresh").get("llm_config") or {})
    assert "choose its model again" in _log(root)
    # Reported once: a second start finds the clone already imported.
    again = ensure_clone_store(tmp_path / "ws", builtin_dir=None, install=False)
    assert not any("choose its model again" in line for line in again.lines), again.lines


def test_a_workspace_ref_is_kept_and_the_log_says_how_it_was_read(tmp_path: Path) -> None:
    """A workspace file may be today's, so a ref is kept -- and not in silence.

    Nothing on disk says which build wrote the file, so `Qwen/Qwen3-8B` from 0.3.1 and
    `ollama/custom-model` written for 0.4.0 look alike; the log names the connection it
    was read as, and says to choose again if that is not one of the person's.

    Killed by: src/uclone_x/agent/clone_store.py :: for label, value in kept:
    Becomes: for label, value in ():
    """
    root = default_agents_root()
    folder = tmp_path / "ws" / ".uclone" / "personas"
    folder.mkdir(parents=True)
    (folder / "fresh.yaml").write_text(
        "name: fresh\nrole: R\nsystem_prompt: S\nllm_config:\n  model_name: Qwen/Qwen3-8B\n",
        encoding="utf-8",
    )

    ensure_clone_store(tmp_path / "ws", builtin_dir=None, install=False)

    assert _migrated(root, "fresh")["llm_config"]["model_name"] == "Qwen/Qwen3-8B"
    log = _log(root)
    assert "on the connection 'Qwen'" in log
    assert "choose its model again" in log
