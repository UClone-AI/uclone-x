"""Every head loads the runtime skill store by default, checked without a shipped store (#1721).

The wiring tests that read the shipped skill packages cannot run in a checkout that has
none, so these build their own: a temporary git checkout holding one approved skill for
the app's default, and a stand-in registry for the CLI heads. The stand-in says only that
each head hands its agents whatever `load_runtime_skill_registry` returned; that function
and the store it reads are covered elsewhere.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from uclone_x.room.models import RoomPolicy
from uclone_x.skills import auditor
from uclone_x.skills.approvals import SkillApprovalLedger, SkillPin
from uclone_x.skills.auditor import (
    RUNTIME_SKILL_STORE_DIRNAME,
    SkillRegistry,
    compute_skill_sha256,
    save_skill,
)
from uclone_x.skills.models import SkillManifest, SkillOrigin, SkillStatus

NAME = "pour_over"


def _checkout_with_one_approved_skill(root: Path) -> Path:
    """A git checkout at `root` whose runtime store holds `NAME`, active and approved."""
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    skill_dir = root / RUNTIME_SKILL_STORE_DIRNAME / NAME
    save_skill(
        skill_dir,
        SkillManifest(
            name=NAME,
            description="Brew one cup by hand.",
            origin=SkillOrigin.HUMAN,
            status=SkillStatus.ACTIVE,
            approved_by="human:alice",
        ),
        "# Pour over\nBloom the grounds for thirty seconds.",
    )
    digest = compute_skill_sha256(skill_dir)
    SkillApprovalLedger().pin(NAME, SkillPin(digest, "human:alice", "2026-09-27T00:00:00Z"))
    return root


def test_the_default_ui_app_loads_approved_skills_from_the_checkout_it_runs_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`create_ui_app` with no registry reads `<git top level>/ucx-agent-skills` at startup.

    Started from a subfolder, so the store is found through the checkout's top level and
    not through the working folder itself.

    Killed by: src/uclone_x/ui/app.py :: else SkillRegistry(skills_dir=runtime_skill_store_dir())
    Becomes: else SkillRegistry()
    """
    checkout = _checkout_with_one_approved_skill(tmp_path / "checkout")
    subfolder = checkout / "work"
    subfolder.mkdir()
    monkeypatch.chdir(subfolder)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from uclone_x.ui.app import create_ui_app

    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        eval_reports_dir=tmp_path / "evals",
        workspace_dir=tmp_path / "workspace",
    )
    with TestClient(app) as client:  # entering runs the lifespan, which loads the store
        names = [skill["name"] for skill in client.get("/api/skills").json()["skills"]]
        registry: SkillRegistry = app.state.session_manager.skill_registry
        loaded = registry.get(NAME)

    assert names == [NAME]
    assert loaded is not None and loaded.manifest.status is SkillStatus.ACTIVE


class _Composed(Exception):
    """Stops a head once it has asked for the scope its agents are built from."""


#: What `load_runtime_skill_registry` returns while these tests run.
_SENTINEL = SkillRegistry()


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The `skills=` each head passes to `local_app_scope`; the head stops right there."""
    seen: list[Any] = []

    async def sentinel_registry(_skills_dir: Path | None = None) -> SkillRegistry:
        return _SENTINEL

    def no_memory(_agent_id: str) -> None:
        return None

    def capture(**kwargs: Any) -> Any:
        seen.append(kwargs.get("skills"))
        raise _Composed()

    from uclone_x.agent import clone_builder
    from uclone_x.cli import agent_memory

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setattr(auditor, "load_runtime_skill_registry", sentinel_registry)
    monkeypatch.setattr(clone_builder, "local_app_scope", capture)
    monkeypatch.setattr(agent_memory, "memory_for_agent_id", no_memory)
    return seen


def test_ucx_run_hands_its_agent_the_runtime_skill_store(
    captured: list[Any], tmp_path: Path
) -> None:
    """Killed by: src/uclone_x/cli/commands/run.py :: skills=await load_runtime_skill_registry(),
    Becomes: skills=None,
    """
    from uclone_x.cli.commands.run import run_agent_repl_async

    with pytest.raises(_Composed):
        asyncio.run(run_agent_repl_async(provider="mock", prompt="hi", workspace_dir=tmp_path))

    assert captured == [_SENTINEL]


def test_ucx_loop_hands_its_agent_the_runtime_skill_store(
    captured: list[Any], tmp_path: Path
) -> None:
    """Killed by: src/uclone_x/cli/commands/loop.py :: skills=await load_runtime_skill_registry(),
    Becomes: skills=None,
    """
    from uclone_x.cli.commands.loop import _run_loop_agent  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(_Composed):
        asyncio.run(_run_loop_agent("hi", 60.0, provider="mock", workspace_dir=tmp_path))

    assert captured == [_SENTINEL]


def test_ucx_a2a_serve_hands_its_agent_the_runtime_skill_store(captured: list[Any]) -> None:
    """Killed by: src/uclone_x/cli/commands/a2a.py :: skills=asyncio.run(load_runtime_skill_registry()),
    Becomes: skills=None,
    """
    from uclone_x.cli.commands.a2a import start_a2a_server

    with pytest.raises(_Composed):
        start_a2a_server(agent_id="default")

    assert captured == [_SENTINEL]


def test_ucx_room_hands_its_seats_the_runtime_skill_store(captured: list[Any]) -> None:
    """Killed by: src/uclone_x/cli/commands/room.py :: skills=asyncio.run(load_runtime_skill_registry()),
    Becomes: skills=None,
    """
    from uclone_x.cli.commands.room import build_orchestrator

    with pytest.raises(_Composed):
        build_orchestrator(store=None, policy=RoomPolicy(), provider="mock")  # type: ignore[arg-type]

    assert captured == [_SENTINEL]


#: Words a notice about the skill folder must not show a user: how it was looked for, and the
#: machine's own paths (#1721).
_INTERNALS = ("git", "rev-parse", "toplevel", "SkillRegistry", "Errno", "Traceback", "None")


def test_the_skills_endpoint_says_when_the_app_started_where_there_is_no_skill_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Started in a checkout with no store, `/api/skills` lists nothing and says the folder is missing.

    So the Settings panel can say why nothing is listed rather than that nothing is
    registered; once the folder exists, it is no longer missing (#1721).

    Killed by: src/uclone_x/skills/auditor.py :: return isinstance(self._store, FileSystemSkillStore) and not self._store.root.is_dir()
    Becomes: return False
    """
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from uclone_x.ui.app import create_ui_app

    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        eval_reports_dir=tmp_path / "evals",
        workspace_dir=tmp_path / "workspace",
    )
    with TestClient(app) as client:
        missing = client.get("/api/skills").json()
        (checkout / RUNTIME_SKILL_STORE_DIRNAME).mkdir()
        present = client.get("/api/skills").json()

    assert missing["skills"] == [] and missing["store_missing"] is True
    assert present["store_missing"] is False


def test_a_cli_head_says_in_plain_words_when_there_is_no_skill_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`load_runtime_skill_registry` over a missing folder says so on stderr, once, in plain words.

    stderr and not stdout, so `ucx acp serve`'s protocol stream is untouched; and nothing
    at all when the folder exists (#1721).

    Killed by: src/uclone_x/skills/auditor.py :: print(NO_SKILL_STORE_NOTICE, file=sys.stderr)
    Becomes: pass
    """
    store = tmp_path / RUNTIME_SKILL_STORE_DIRNAME
    asyncio.run(auditor.load_runtime_skill_registry(store))
    missing = capsys.readouterr()
    store.mkdir()
    asyncio.run(auditor.load_runtime_skill_registry(store))
    present = capsys.readouterr()

    assert missing.out == ""
    lines = missing.err.strip().splitlines()
    assert len(lines) == 1
    assert "No skills are loaded" in lines[0]
    assert RUNTIME_SKILL_STORE_DIRNAME in lines[0] and "ucx skill --help" in lines[0]
    for internal in (*_INTERNALS, str(tmp_path)):
        assert internal not in lines[0], internal
    assert present.out == "" and present.err == ""
