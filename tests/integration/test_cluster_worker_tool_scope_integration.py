"""Many users, one OS account, no sandbox: what a worker's model can reach.

Each case is the model asking, through the real agent and the real tools. What it was offered is
read from the request it received, and what it got back from the next request's tool result.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from tests.integration.cluster_harness import ClusterRig, Reply
from uclone_x.llm.models import LLMRequest, MessageRole, ToolCallRequest
from uclone_x.shells.cluster.turn_executor import WORKER_FILE_TOOLS
from uclone_x.skills.auditor import SkillRegistry

_SECRET = "stack-secret-73124"


def _offered(request: LLMRequest) -> set[str]:
    return {tool.name for tool in request.tools or ()}


def _tool_results(request: LLMRequest) -> str:
    return "\n".join(m.content or "" for m in request.messages if m.role is MessageRole.TOOL)


def _call(name: str, **arguments: Any) -> Reply:
    return Reply(tool_calls=(ToolCallRequest(id=f"c-{name}", name=name, arguments=arguments),))


def _rig(tmp_path: Path, **fields: object) -> tuple[ClusterRig, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return ClusterRig(tmp_path, workspace=workspace, **fields), workspace  # pyright: ignore[reportArgumentType]


async def _ask(rig: ClusterRig, user: str, *calls: Reply) -> str:
    """Run one turn as `user`, making `calls`; return everything the tools answered."""
    rig.script.replies.extend([*calls, Reply(text="Done.")])
    before = rig.script.calls
    await rig.worker().execute_single_turn(rig.command(f"sess-{before}", "go", user_id=user))
    return _tool_results(rig.script.requests[-1])


@pytest.mark.asyncio
async def test_the_worker_offers_exactly_the_file_tools_and_no_way_to_execute(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/turn_executor.py :: for name in WORKER_FILE_TOOLS:
    Becomes: for name in [tool.name for tool in every_tool.list_tools()]:
    """
    rig, _ = _rig(tmp_path)
    await _ask(rig, "alice")
    offered = _offered(rig.script.requests[0])
    # Spelled out rather than read from the constant: adding a tool here is the decision.
    assert offered == {"file_read", "file_write", "file_edit", "file_search", "directory_list"}
    assert WORKER_FILE_TOOLS == (
        "file_read",
        "file_write",
        "file_edit",
        "file_search",
        "directory_list",
    )
    for name in ("bash_run", "run_command", "install_package", "browser", "web_fetch"):
        assert name not in offered


@pytest.mark.asyncio
async def test_a_tool_the_model_names_anyway_does_not_run(tmp_path: Path) -> None:
    rig, workspace = _rig(tmp_path)
    marker = tmp_path / "ran.txt"
    answered = await _ask(rig, "alice", _call("bash_run", command=f"touch {marker}"))
    assert not marker.exists()
    assert "bash_run" in answered  # the refusal names it, and the model reads why
    assert workspace.exists()


@pytest.mark.asyncio
async def test_with_a_skill_store_the_worker_offers_load_skill_and_never_propose_skill(
    tmp_path: Path,
) -> None:
    """A proposal would land in a store every user shares, for a page a worker does not have.

    Killed by: src/uclone_x/shells/cluster/turn_executor.py :: store_root = None
    Becomes: store_root = property(lambda self: self._inner.store_root)
    """
    store = tmp_path / "skills"
    store.mkdir()
    rig, _ = _rig(tmp_path, skills=SkillRegistry(skills_dir=store))
    await _ask(rig, "alice")
    offered = _offered(rig.script.requests[0])
    assert "load_skill" in offered
    assert "propose_skill" not in offered


@pytest.mark.asyncio
async def test_with_no_workspace_a_turn_has_no_file_tool_at_all(tmp_path: Path) -> None:
    rig = ClusterRig(tmp_path)
    await _ask(rig, "alice")
    assert _offered(rig.script.requests[0]) == set()


@pytest.mark.asyncio
async def test_bob_cannot_reach_alices_file_through_any_offered_tool(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/turn_executor.py :: folder = self._workspace_root / "users" / content_digest(user_id)
    Becomes: folder = self._workspace_root
    """
    rig, workspace = _rig(tmp_path)
    await _ask(
        rig, "alice", _call("file_write", path="diary.txt", content=f"alice wrote {_SECRET}")
    )
    alice_files = list((workspace / "users").rglob("diary.txt"))
    assert len(alice_files) == 1
    alice_path = alice_files[0]
    relative = alice_path.relative_to(workspace)

    answered = await _ask(
        rig,
        "bob",
        _call("file_read", path="diary.txt"),
        _call("file_read", path=str(alice_path)),
        _call("file_read", path=f"../{alice_path.parent.name}/diary.txt"),
        _call("file_read", path=f"../../{relative}"),
        _call("file_edit", path=str(alice_path), old_string="alice", new_string="bob"),
        _call("file_write", path=f"../{alice_path.parent.name}/diary.txt", content="bob was here"),
    )
    assert _SECRET not in answered
    assert alice_path.read_text(encoding="utf-8") == f"alice wrote {_SECRET}"

    # And Bob cannot find out she is there: listing refuses to leave his folder.
    found = await _ask(
        rig,
        "bob",
        _call("file_search", pattern="alice wrote"),
        _call("file_search", pattern="alice wrote", path=".."),
        _call("directory_list", path="."),
        _call("directory_list", path=".."),
        _call("directory_list", path=str(workspace)),
    )
    assert _SECRET not in found
    assert alice_path.parent.name not in found


@pytest.mark.asyncio
async def test_a_users_folder_is_open_to_the_workers_account_alone(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/turn_executor.py :: folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    Becomes: folder.mkdir(parents=True, exist_ok=True)
    """
    rig, workspace = _rig(tmp_path)
    await _ask(rig, "alice")
    folder = next((workspace / "users").iterdir())
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "user_id", ["../../etc", "..", "a/b", "/etc/passwd", "alice/../bob", "%2e%2e", "ÁLICE", "~root"]
)
@pytest.mark.asyncio
async def test_no_user_id_names_a_folder_outside_the_workspace(
    tmp_path: Path, user_id: str
) -> None:
    rig, workspace = _rig(tmp_path)
    await _ask(rig, user_id, _call("file_write", path="mine.txt", content="x"))
    written = list(tmp_path.rglob("mine.txt"))
    assert len(written) == 1
    assert written[0].parent.parent == workspace / "users"


@pytest.mark.asyncio
async def test_a_symlink_out_of_the_users_folder_is_refused(tmp_path: Path) -> None:
    rig, workspace = _rig(tmp_path)
    outside = tmp_path / "stack.env"
    outside.write_text(f"UCX_POSTGRES_PASSWORD={_SECRET}\n", encoding="utf-8")
    await _ask(rig, "mallory", _call("file_write", path="mine.txt", content="x"))
    folder = next((workspace / "users").iterdir())
    (folder / "link.txt").symlink_to(outside)
    (folder / "linkdir").symlink_to(tmp_path)

    answered = await _ask(
        rig,
        "mallory",
        _call("file_read", path="link.txt"),
        _call("file_read", path="linkdir/stack.env"),
        _call("file_edit", path="link.txt", old_string="UCX", new_string="XXX"),
        _call("file_write", path="link.txt", content="overwritten"),
        _call("file_search", pattern="UCX_POSTGRES"),
        _call("directory_list", path="linkdir"),
    )
    assert _SECRET not in answered
    assert outside.read_text(encoding="utf-8") == f"UCX_POSTGRES_PASSWORD={_SECRET}\n"


@pytest.mark.asyncio
async def test_a_turn_cannot_reach_the_stack_secrets_or_the_workers_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig, _ = _rig(tmp_path)
    stack_env = tmp_path / ".ucx-cluster" / "stack.env"
    stack_env.parent.mkdir()
    stack_env.write_text(f"UCX_POSTGRES_PASSWORD={_SECRET}\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", _SECRET)
    monkeypatch.chdir(tmp_path)

    answered = await _ask(
        rig,
        "mallory",
        _call("file_read", path=str(stack_env)),
        _call("file_read", path=".ucx-cluster/stack.env"),
        _call("file_read", path="../../../.ucx-cluster/stack.env"),
        _call("file_read", path="../../../../.ucx-cluster/stack.env"),
        _call("file_read", path="/proc/self/environ"),
        _call("file_search", pattern="UCX_POSTGRES", path=str(tmp_path)),
        _call("directory_list", path=str(stack_env.parent)),
    )
    assert _SECRET not in answered
    assert _SECRET not in str(rig.script.requests[-1].messages)
    assert os.environ["GEMINI_API_KEY"] == _SECRET  # it was there to be found
