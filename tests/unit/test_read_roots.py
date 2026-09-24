"""Read-only folders outside the workspace, and the prompt section that names them.

A clone asked about a folder by name could neither list a folder nor reach one outside its
workspace, and its prompt never said where the workspace was, so it invented a relative
path and reported the folder missing. Three things fix that: the read-only file tools
accept an absolute path inside a configured read root, every turn states the workspace
and those roots, and the Settings list reaches both new and live agents. Writing stays
confined to the workspace.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, ToolCallRequest
from uclone_x.room.models import Participant, ParticipantKind
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.sandbox.models import WorkspaceIsolation
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools import (
    DirectoryListTool,
    FileEditTool,
    FileReadTool,
    FileSearchTool,
    FileWriteTool,
    ToolContext,
    ToolRegistry,
    ToolResultStatus,
)
from uclone_x.ui.app import READ_ROOTS_ENV_VAR, AgentSessionManager, create_ui_app


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def papers(tmp_path: Path) -> Path:
    """A read root holding one file, beside (not inside) the workspace."""
    root = tmp_path / "papers"
    (root / "sub").mkdir(parents=True)
    (root / "README.md").write_text("hexworld notes\n", encoding="utf-8")
    return root


def _context(workspace: Path, *read_roots: Path) -> ToolContext:
    return ToolContext(
        agent_id="a",
        session_id="s",
        workspace_root=workspace,
        read_roots=read_roots,
        isolation=WorkspaceIsolation(),
    )


def _output(result: Any) -> dict[str, Any]:
    assert result.status == ToolResultStatus.SUCCESS, result.error
    return cast(dict[str, Any], result.output)


# ======================================================================================
# Tools
# ======================================================================================


@pytest.mark.asyncio
async def test_an_absolute_path_inside_a_read_root_is_read_and_named_absolutely(
    workspace: Path, papers: Path
) -> None:
    """The path comes back absolute: a relative one would resolve against the workspace.

    Killed by: src/uclone_x/tools/base.py :: for root in context.read_roots:
    Becomes: for root in ():
    Killed by: src/uclone_x/tools/base.py :: if root == workspace:
    Becomes: if True:
    """
    target = papers / "README.md"
    out = _output(await FileReadTool().execute({"path": str(target)}, _context(workspace, papers)))

    assert out["content"] == "hexworld notes\n"
    assert out["path"] == str(target.resolve())


@pytest.mark.asyncio
async def test_a_tilde_path_inside_a_read_root_is_expanded(
    tmp_path: Path, workspace: Path, papers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/base.py :: candidate = Path(target_path).expanduser()
    Becomes: candidate = Path(target_path)
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    out = _output(
        await FileReadTool().execute({"path": "~/papers/README.md"}, _context(workspace, papers))
    )
    assert out["content"] == "hexworld notes\n"


@pytest.mark.asyncio
async def test_directory_list_and_file_search_work_inside_a_read_root(
    workspace: Path, papers: Path
) -> None:
    """The two tools a clone needs to find a file it was told about only by folder."""
    ctx = _context(workspace, papers)

    listed = _output(await DirectoryListTool().execute({"path": str(papers)}, ctx))
    assert listed["path"] == str(papers.resolve())
    assert {e["path"] for e in listed["entries"]} == {
        str((papers / "sub").resolve()),
        str((papers / "README.md").resolve()),
    }

    found = _output(await FileSearchTool().execute({"path": str(papers), "query": "hexworld"}, ctx))
    assert [m["path"] for m in found["matches"]] == [str((papers / "README.md").resolve())]


@pytest.mark.asyncio
async def test_workspace_paths_stay_relative_when_read_roots_are_set(
    workspace: Path, papers: Path
) -> None:
    """The workspace is tried first, so a relative path means what it always meant."""
    (workspace / "a.txt").write_text("in workspace", encoding="utf-8")
    out = _output(await FileReadTool().execute({"path": "a.txt"}, _context(workspace, papers)))
    assert out["path"] == "a.txt"
    assert out["content"] == "in workspace"


@pytest.mark.asyncio
async def test_a_relative_escape_into_a_read_root_is_refused(workspace: Path, papers: Path) -> None:
    """Only an absolute path is tried against a read root; `..` from the workspace is not."""
    result = await FileReadTool().execute(
        {"path": "../papers/README.md"}, _context(workspace, papers)
    )
    assert result.status == ToolResultStatus.ERROR
    assert result.error is not None and "Path traversal violation" in result.error


@pytest.mark.asyncio
async def test_a_path_outside_every_root_is_refused_and_the_roots_are_named(
    tmp_path: Path, workspace: Path, papers: Path
) -> None:
    """The refusal names where the model may look, so it can retry with a usable path."""
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    result = await FileReadTool().execute(
        {"path": str(tmp_path / "secret.txt")}, _context(workspace, papers)
    )
    assert result.status == ToolResultStatus.ERROR
    assert result.error is not None
    assert str(papers) in result.error and str(workspace) in result.error


@pytest.mark.asyncio
async def test_a_symlink_out_of_a_read_root_is_refused(
    tmp_path: Path, workspace: Path, papers: Path
) -> None:
    """Containment inside a read root is the same resolved-path check as the workspace's."""
    outside = tmp_path / "outside.txt"
    outside.write_text("no", encoding="utf-8")
    (papers / "link.txt").symlink_to(outside)

    result = await FileReadTool().execute(
        {"path": str(papers / "link.txt")}, _context(workspace, papers)
    )
    assert result.status == ToolResultStatus.ERROR


@pytest.mark.asyncio
async def test_without_read_roots_the_original_refusal_is_kept(
    tmp_path: Path, workspace: Path
) -> None:
    """Killed by: src/uclone_x/tools/base.py :: if not context.read_roots:
    Becomes: if False:
    """
    result = await FileReadTool().execute({"path": str(tmp_path / "x.txt")}, _context(workspace))
    assert result.status == ToolResultStatus.ERROR
    assert result.error is not None
    assert "read-only folder" not in result.error


@pytest.mark.asyncio
async def test_a_read_root_is_never_writable(workspace: Path, papers: Path) -> None:
    """The writing tools keep the workspace-only resolver."""
    result = await FileWriteTool().execute(
        {"path": str(papers / "new.md"), "content": "x"}, _context(workspace, papers)
    )
    assert result.status == ToolResultStatus.ERROR
    assert not (papers / "new.md").exists()


@pytest.mark.asyncio
async def test_a_tilde_path_the_clone_wrote_is_read_back_from_the_workspace(
    tmp_path: Path, workspace: Path, papers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writing tool puts `~/x` at `<workspace>/~/x`; a read of `~/x` must find that file,
    not a same-named one under a read root, or reading back what was written lies.

    Killed by: src/uclone_x/tools/base.py :: if inside is not None and inside.exists():
    Becomes: if False:
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    ctx = _context(workspace, papers)
    written = await FileWriteTool().execute({"path": "~/papers/README.md", "content": "mine"}, ctx)
    assert written.status == ToolResultStatus.SUCCESS, written.error
    assert (workspace / "~" / "papers" / "README.md").is_file()

    out = _output(await FileReadTool().execute({"path": "~/papers/README.md"}, ctx))
    assert out["content"] == "mine"


@pytest.mark.asyncio
async def test_a_read_root_is_never_editable(workspace: Path, papers: Path) -> None:
    """`file_edit` also keeps the workspace-only resolver."""
    result = await FileEditTool().execute(
        {
            "path": str(papers / "README.md"),
            "target_content": "hexworld",
            "replacement_content": "changed",
        },
        _context(workspace, papers),
    )
    assert result.status == ToolResultStatus.ERROR
    assert (papers / "README.md").read_text(encoding="utf-8") == "hexworld notes\n"


@pytest.mark.asyncio
async def test_a_recursive_listing_skips_a_symlink_out_of_the_read_root(
    tmp_path: Path, workspace: Path, papers: Path
) -> None:
    """Killed by: src/uclone_x/tools/builtin/filesystem.py :: safe_d = self.resolve_safe_path(d_path, root)
    Becomes: safe_d = d_path
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("no", encoding="utf-8")
    (papers / "escape").symlink_to(outside, target_is_directory=True)

    listed = _output(
        await DirectoryListTool().execute(
            {"path": str(papers), "recursive": True}, _context(workspace, papers)
        )
    )
    names = {e["name"] for e in listed["entries"]}
    assert "README.md" in names
    assert "escape" not in names and "secret.txt" not in names


# ======================================================================================
# Agent: the [Workspace] prompt section and the tool context
# ======================================================================================


class _RecordingLLM(MockLLMConnector):
    """Keeps every request, so a test can read the system prompt the agent sent."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def _agent(
    workspace: Path,
    llm: MockLLMConnector,
    *,
    allowed_tools: tuple[str, ...] = (),
    tools: ToolRegistry | None = None,
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id="reader",
            name="Reader",
            workspace_dir=workspace,
            allowed_tools=allowed_tools,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=llm,
        tools=tools,
    )


def _system_prompt(llm: _RecordingLLM) -> str:
    return "\n".join(m.content or "" for m in llm.requests[-1].messages if m.role.value == "system")


@pytest.mark.asyncio
async def test_the_turn_states_the_workspace_and_every_read_root(
    workspace: Path, papers: Path
) -> None:
    """Killed by: src/uclone_x/agent/base.py :: lines.extend(f"- {root.resolve()}" for root in read_roots)
    Becomes: lines.extend(())
    """
    llm = _RecordingLLM(default_response="ok")
    agent = _agent(workspace, llm)
    agent.set_read_roots((papers,))

    await agent.execute_turn("what is in my papers folder?")

    prompt = _system_prompt(llm)
    assert "[Workspace]" in prompt
    assert str(workspace.resolve()) in prompt
    assert f"- {papers.resolve()}" in prompt


@pytest.mark.asyncio
async def test_an_agent_holding_no_file_tool_gets_no_workspace_section(workspace: Path) -> None:
    """Killed by: src/uclone_x/agent/base.py :: if allowed and not any(
    Becomes: if False and not any(
    """
    llm = _RecordingLLM(default_response="ok")
    await _agent(workspace, llm, allowed_tools=("web_search",)).execute_turn("hi")
    assert "[Workspace]" not in _system_prompt(llm)


@pytest.mark.asyncio
async def test_read_roots_set_on_a_live_agent_reach_its_next_tool_call(
    workspace: Path, papers: Path
) -> None:
    """What makes a Settings change take effect without restarting the conversation.

    Killed by: src/uclone_x/agent/base.py :: self._config = self._config.model_copy(update={"read_roots": read_roots})
    Becomes: pass
    """
    call = ToolCallRequest(
        id="tc_read", name="file_read", arguments={"path": str(papers / "README.md")}
    )
    llm = _RecordingLLM(tool_calls=[call], default_response="done")
    agent = _agent(workspace, llm, tools=ToolRegistry(tools=[FileReadTool()]))
    agent.set_read_roots((papers,))

    await agent.execute_turn("read it")

    tool_messages = [m for m in llm.requests[-1].messages if m.tool_call_id == "tc_read"]
    assert tool_messages, "the tool result never reached the model"
    assert "hexworld notes" in (tool_messages[0].content or "")


@pytest.mark.asyncio
async def test_a_sub_agent_reads_what_its_parent_reads(workspace: Path, papers: Path) -> None:
    """Killed by: src/uclone_x/agent/base.py :: read_roots=self._config.read_roots,  # a child reads what its parent reads
    Becomes: read_roots=(),
    """
    parent = _agent(workspace, _RecordingLLM(default_response="ok"))
    parent.set_read_roots((papers,))
    child = await parent.spawn_subagent(role="scout", goal="find the README")
    assert child.config.read_roots == (papers,)


# ======================================================================================
# Settings: the SessionManager's list
# ======================================================================================


@pytest.fixture
def no_env_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(READ_ROOTS_ENV_VAR, raising=False)


@pytest.mark.usefixtures("no_env_roots")
def test_a_relative_folder_is_refused_with_the_fix(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/ui/app.py :: if not folder.is_absolute():
    Becomes: if False:
    """
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    with pytest.raises(ValueError, match="full path"):
        mgr.update_settings(read_roots=["notes"])


@pytest.mark.usefixtures("no_env_roots")
def test_a_folder_that_does_not_exist_refuses_the_whole_list(tmp_path: Path, papers: Path) -> None:
    """Refused, not filtered: a dropped entry would leave the user believing it is readable."""
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    with pytest.raises(ValueError, match="not a folder"):
        mgr.update_settings(read_roots=[str(papers), str(tmp_path / "missing")])
    assert mgr.get_settings()["read_roots"] == []


@pytest.mark.usefixtures("no_env_roots")
def test_the_list_is_saved_and_read_back_by_a_new_manager(tmp_path: Path, papers: Path) -> None:
    """Killed by: src/uclone_x/ui/app.py :: self._configured_read_roots = tuple(
    Becomes: self._configured_read_roots = () and tuple(
    """
    storage = tmp_path / "store"
    AgentSessionManager(storage_dir=storage, fallback_to_mock=True).update_settings(
        read_roots=[str(papers), f"  {papers}  ", ""]
    )
    saved = json.loads((storage / "settings.json").read_text(encoding="utf-8"))
    assert saved["read_roots"] == [str(papers)]

    reloaded = AgentSessionManager(storage_dir=storage, fallback_to_mock=True)
    assert reloaded.get_settings()["read_roots"] == [str(papers)]
    assert reloaded.read_roots == (papers.resolve(),)


@pytest.mark.usefixtures("no_env_roots")
def test_a_saved_folder_that_was_deleted_is_reported_and_left_out(
    tmp_path: Path, papers: Path
) -> None:
    """Killed by: src/uclone_x/ui/app.py :: if _read_root_problem(entry, self._storage_dir) is not None:
    Becomes: if False:
    """
    gone = tmp_path / "gone"
    gone.mkdir()
    mgr = AgentSessionManager(storage_dir=tmp_path / "store", fallback_to_mock=True)
    mgr.update_settings(read_roots=[str(papers), str(gone)])
    gone.rmdir()

    assert mgr.read_roots == (papers.resolve(),)
    assert mgr.get_settings()["read_roots_missing"] == [str(gone)]


def test_the_environment_list_is_merged_before_the_settings_list(
    tmp_path: Path, papers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/app.py :: for entry in [*self._env_read_root_entries(), *self._configured_read_roots]:
    Becomes: for entry in [*self._configured_read_roots]:
    """
    env_root = tmp_path / "env_root"
    env_root.mkdir()
    monkeypatch.setenv(READ_ROOTS_ENV_VAR, os.pathsep.join([str(env_root), "relative/ignored"]))
    mgr = AgentSessionManager(storage_dir=tmp_path / "store", fallback_to_mock=True)
    mgr.update_settings(read_roots=[str(papers), str(env_root)])

    assert mgr.read_roots == (env_root.resolve(), papers.resolve())


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_env_roots")
async def test_new_and_live_agents_both_receive_the_list(tmp_path: Path, papers: Path) -> None:
    """Killed by: src/uclone_x/ui/app.py :: config = config.model_copy(update={"read_roots": self.read_roots})
    Becomes: pass
    Killed by: src/uclone_x/ui/app.py :: agent.set_read_roots(effective_roots)
    Becomes: pass
    """
    mgr = AgentSessionManager(storage_dir=tmp_path / "store", fallback_to_mock=True)
    live = await mgr.get_or_create_agent(agent_id="live", session_id="sess_live")
    assert live.config.read_roots == ()

    mgr.update_settings(read_roots=[str(papers)])
    assert live.config.read_roots == (papers.resolve(),)

    fresh = await mgr.get_or_create_agent(agent_id="fresh", session_id="sess_fresh")
    assert fresh.config.read_roots == (papers.resolve(),)


@pytest.mark.usefixtures("no_env_roots")
def test_a_folder_holding_the_settings_file_is_refused(tmp_path: Path) -> None:
    """`settings.json` carries the API key; a root above it would let a clone read it.

    Killed by: src/uclone_x/ui/app.py :: if _holds_folder(folder, storage_dir):
    Becomes: if False:
    """
    mgr = AgentSessionManager(storage_dir=tmp_path / "store", fallback_to_mock=True)
    with pytest.raises(ValueError, match="API key"):
        mgr.update_settings(read_roots=[str(tmp_path)])
    assert mgr.read_roots == ()


@pytest.mark.usefixtures("no_env_roots")
def test_a_folder_holding_the_settings_file_is_refused_under_another_spelling(
    tmp_path: Path,
) -> None:
    """On a case-insensitive volume a path comparison lets `/users/me` past `/Users/me`.

    Killed by: src/uclone_x/ui/app.py :: if os.path.samestat(candidate.stat(), folder_stat):
    Becomes: if candidate == folder:
    """
    (tmp_path / "Home" / ".uclone").mkdir(parents=True)
    alias = tmp_path / "home"
    if not alias.is_dir():
        pytest.skip("case-sensitive file system: the two spellings are two folders")
    mgr = AgentSessionManager(storage_dir=tmp_path / "Home" / ".uclone", fallback_to_mock=True)
    with pytest.raises(ValueError, match="API key"):
        mgr.update_settings(read_roots=[str(alias)])
    assert mgr.read_roots == ()


def test_an_environment_root_holding_the_settings_file_is_ignored_and_reported(
    tmp_path: Path, papers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An environment entry cannot be refused at entry, so it is left out and shown."""
    monkeypatch.setenv(READ_ROOTS_ENV_VAR, os.pathsep.join([str(tmp_path), str(papers)]))
    mgr = AgentSessionManager(storage_dir=tmp_path / "store", fallback_to_mock=True)

    assert mgr.read_roots == (papers.resolve(),)
    settings = mgr.get_settings()
    assert settings["read_roots_env"] == [str(papers)]
    assert len(settings["read_roots_env_ignored"]) == 1
    assert "API key" in settings["read_roots_env_ignored"][0]


@pytest.mark.usefixtures("no_env_roots")
def test_a_saved_folder_that_was_deleted_does_not_block_editing_the_list(
    tmp_path: Path, papers: Path
) -> None:
    """Killed by: src/uclone_x/ui/app.py :: entry in already_saved and not Path(entry).expanduser().exists()
    Becomes: False
    """
    gone = tmp_path / "gone"
    gone.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    mgr = AgentSessionManager(storage_dir=tmp_path / "store", fallback_to_mock=True)
    mgr.update_settings(read_roots=[str(papers), str(gone)])
    gone.rmdir()

    mgr.update_settings(read_roots=[str(papers), str(gone), str(other)])
    assert mgr.get_settings()["read_roots"] == [str(papers), str(gone), str(other)]
    # A folder that was never saved is still refused when it does not exist.
    with pytest.raises(ValueError, match="not a folder"):
        mgr.update_settings(read_roots=[str(tmp_path / "never")])


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_env_roots")
async def test_another_origin_cannot_change_the_list(tmp_path: Path, papers: Path) -> None:
    """A page in another tab could otherwise widen what clones read, then ask one to read it.

    Killed by: src/uclone_x/ui/app.py :: _refuse_cross_origin(request)  # names the folders clones can read, and the workspace
    Becomes: pass
    Killed by: src/uclone_x/ui/app.py :: _refuse_cross_origin(request)  # another tab must not widen read_roots
    Becomes: pass
    """
    mgr = AgentSessionManager(storage_dir=tmp_path / "store", fallback_to_mock=True)
    app = create_ui_app(static_dir=tmp_path, session_manager=mgr, storage_dir=tmp_path / "store")
    evil = {"Origin": "https://evil.example.com"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        post = await client.post("/api/settings", json={"read_roots": [str(papers)]}, headers=evil)
        get = await client.get("/api/settings", headers=evil)
        same = await client.get("/api/settings")

    assert post.status_code == 403
    assert get.status_code == 403
    assert same.status_code == 200
    assert mgr.read_roots == ()


# ======================================================================================
# Rooms: seats are cached, so the resolver re-applies the list on every resolve
# ======================================================================================


@pytest.mark.asyncio
async def test_a_room_seat_follows_the_list_after_it_is_built(tmp_path: Path, papers: Path) -> None:
    """Killed by: src/uclone_x/room/resolver.py :: read_roots = self._read_roots()
    Becomes: read_roots = ()
    Killed by: src/uclone_x/room/resolver.py :: cached.set_read_roots(self._read_roots())
    Becomes: pass
    """
    current: list[tuple[Path, ...]] = [(papers,)]
    host = HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )
    resolver = RoomAgentResolver(host, read_roots=lambda: current[0])
    seat = Participant(
        id="scout",
        kind=ParticipantKind.AGENT,
        display_name="scout",
        session_id="sess_room__r1__scout",
    )

    built = await resolver.resolve(seat)
    assert isinstance(built, BaseAgent)
    assert built.config.read_roots == (papers,)

    current[0] = ()
    again = await resolver.resolve(seat)
    assert again is built
    assert built.config.read_roots == ()
