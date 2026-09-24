"""Tests for the agent a live evaluation suite measures (`uclone_x.evaluation.answerer`).

What these pin is one failure that the report could not show. The first live run composed
the agent with no workspace, so every tool call returned "Tool requires a workspace, but
the host provided none" -- and the suite went on to grade a hundred answers produced from
the model's weights alone, publishing a capability figure for a runtime that had never
touched the repository (#666). A crash would have been better; this looked like a result.

So two things are asserted here, and the second matters more than the first. The agent is
composed with a workspace root. And a run whose tools do not work **refuses to start**,
because the workspace can be lost again by a hundred ordinary edits and there would be no
second chance to notice.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext
from uclone_x.evaluation.answerer import (
    LIVE_EVAL_REQUEST_TIMEOUT_SECONDS,
    PREFLIGHT_PROBE_TOOLS,
    AgentAnswerer,
    EvalToolPreflightError,
    EvalTurnFailedError,
    EvalTurnTimedOutError,
    build_agent_answerer,
    create_disposable_checkout,
    default_workspace_root,
    disposable_workspace,
    preflight_agent_tools,
    preflight_probes,
    workspace_git_state,
)
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.tools.registry import create_default_registry


@pytest.fixture
def stub_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compose the real agent against a stub endpoint.

    The provider is not what is under test here; the workspace wiring is, and it is the
    same wiring whichever connector is attached.
    """

    def _factory(**_kwargs: Any) -> MockLLMConnector:
        return MockLLMConnector(default_response="stubbed")

    monkeypatch.setattr(
        "uclone_x.llm.connectors.factory.create_llm_connector",
        _factory,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A throwaway tree with something in it.

    Not empty, because the preflight now reads a file: the `file_io` problems call
    `file_read`, and a preflight that probed only `file_search` passed with `file_read`
    missing from the registry entirely. Not a git repository, because the preflight also
    refuses a dirty checkout.
    """
    (tmp_path / "README.md").write_text("a tree the benchmark may be pointed at\n")
    return tmp_path


def _workspaceless_agent() -> BaseAgent:
    """The agent exactly as #666 found it: a full tool registry and nowhere to use it."""
    return BaseAgent(
        config=AgentConfig(agent_id="a", name="A", system_prompt="s"),
        llm=MockLLMConnector(default_response="stubbed"),
        tools=create_default_registry(enable_mcp=False),
        context=AgentContext(session_id="sess_a", agent_id="a"),
    )


# --- the workspace --------------------------------------------------------------------


def test_the_composed_agent_can_read_the_tree_it_is_asked_about(
    stub_connector: None, tmp_path: Path
) -> None:
    """74 of the 100 frontier problems read files; without this every one of them fails.

    Asserted by executing a tool rather than by inspecting the config, because the config
    field is not the thing that was missing -- what was missing was the workspace arriving
    at the tool host, and only a tool call can say whether it did.

    Killed by: src/uclone_x/evaluation/answerer.py :: workspace_dir=root
    """
    (tmp_path / "NOTES.md").write_text("the frontier set asks about this tree\n")

    answerer = build_agent_answerer(provider="mock", workspace_root=tmp_path)

    assert isinstance(answerer, AgentAnswerer)
    record = asyncio.run(
        answerer.agent.execute_tool_call("file_search", {"query": "frontier", "path": "."})
    )
    assert record.status == "success", record.error
    assert "NOTES.md" in str(record.output)


def test_the_workspace_root_is_overridable(stub_connector: None, workspace: Path) -> None:
    """A run against a worktree and a run against `main` are different measurements.

    The default is the checkout the command was invoked from, which is right for the common
    case and wrong to be the only case: the caller has to be able to name the tree, or the
    report's record of it is a tautology.

    Killed by: src/uclone_x/evaluation/answerer.py :: root = Path(workspace_root).resolve()
    Becomes: root = default_workspace_root()
    """
    answerer = build_agent_answerer(provider="mock", workspace_root=workspace)
    assert answerer.workspace_root == workspace.resolve()
    assert answerer.workspace_root != default_workspace_root()


def test_the_default_workspace_root_is_the_repository(stub_connector: None) -> None:
    """A run started from `evals/` must reach the same files as one started from the root."""
    root = default_workspace_root()
    assert root.is_absolute()
    assert (root / "pyproject.toml").exists()


def test_the_default_workspace_root_is_disposable_and_isolated_from_operator_tree(
    stub_connector: None,
) -> None:
    """The live eval suite must not execute against the operator's live checkout (#687).

    `bash_run` keeps arbitrary shell execution capability, so executing in the operator's
    working tree could modify or delete uncommitted or committed files. When `workspace_root`
    is omitted, `build_agent_answerer` must execute against an isolated disposable checkout
    created from HEAD, keeping the operator's live checkout completely untouched.

    Killed by: src/uclone_x/evaluation/answerer.py :: elif disposable:
    Becomes: elif False:
    """
    answerer = build_agent_answerer(provider="mock")
    try:
        assert answerer.is_disposable is True
        assert answerer.workspace_root != default_workspace_root()
        assert answerer.workspace_root.is_dir()
        assert (answerer.workspace_root / "pyproject.toml").is_file()
        assert answerer.workspace_git.is_repo is True
        assert answerer.workspace_git.dirty is False

        # Verify isolating effect: mutations inside disposable tree do not bleed into repo
        sentinel_path = answerer.workspace_root / "disposable_test_sentinel.tmp"
        sentinel_path.write_text("disposable content\n")
        assert sentinel_path.exists()
        assert not (default_workspace_root() / "disposable_test_sentinel.tmp").exists()
    finally:
        answerer.close()


def test_disposable_workspace_cleaned_up_on_close(stub_connector: None) -> None:
    """Closing the answerer must remove the disposable worktree and leave no residue (#687).

    Killed by: src/uclone_x/evaluation/answerer.py :: cleanup = self._cleanup
    Becomes: return
    """
    answerer = build_agent_answerer(provider="mock")
    disposable_root = answerer.workspace_root
    assert disposable_root.exists()
    answerer.close()
    assert not disposable_root.exists()


def test_disposable_workspace_cleaned_up_on_preflight_failure(
    stub_connector: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preflight failure during construction must clean up the disposable checkout (#687).

    Killed by: src/uclone_x/evaluation/answerer.py :: if cleanup is not None:
    Becomes: if False:
    """
    created_roots: list[Path] = []

    def _failing_preflight(_agent: BaseAgent, root: Path) -> None:
        created_roots.append(root)
        raise EvalToolPreflightError("simulated preflight failure")

    monkeypatch.setattr("uclone_x.evaluation.answerer.preflight_agent_tools", _failing_preflight)

    with pytest.raises(EvalToolPreflightError, match="simulated preflight failure"):
        build_agent_answerer(provider="mock")

    assert len(created_roots) == 1
    assert not created_roots[0].exists()


def test_create_disposable_checkout_direct() -> None:
    """`create_disposable_checkout` returns a populated disposable tree and idempotent cleanup (#687).

    Killed by: src/uclone_x/evaluation/answerer.py :: return temp_dir, cleanup
    Becomes: return root, cleanup
    """
    path, cleanup = create_disposable_checkout()
    try:
        assert path != default_workspace_root()
        assert path.is_dir()
        assert (path / "pyproject.toml").is_file()
    finally:
        cleanup()
    assert not path.exists()
    # Idempotent cleanup: calling again does not raise
    cleanup()


def test_disposable_workspace_context_manager() -> None:
    """`disposable_workspace` context manager yields a valid checkout and tears it down (#687).

    Killed by: src/uclone_x/evaluation/answerer.py :: yield path
    Becomes: yield default_workspace_root()
    """
    recorded: Path | None = None
    with disposable_workspace() as ws:
        recorded = ws
        assert ws.is_dir()
        assert (ws / "pyproject.toml").is_file()
    assert recorded is not None
    assert not recorded.exists()


def test_explicit_workspace_root_bypasses_disposable(stub_connector: None, workspace: Path) -> None:
    """An explicit workspace root is not disposable and close() does not delete it (#687).

    Killed by: src/uclone_x/evaluation/answerer.py :: is_disposable = False
    Becomes: is_disposable = True
    """
    answerer = build_agent_answerer(provider="mock", workspace_root=workspace)
    assert answerer.is_disposable is False
    assert answerer.workspace_root == workspace.resolve()
    answerer.close()
    assert workspace.exists()


# --- the preflight --------------------------------------------------------------------


def test_the_preflight_refuses_an_agent_whose_tools_cannot_run(workspace: Path) -> None:
    """The #666 state, reproduced: a workspaceless agent must not be allowed to answer.

    This is the guard the issue is actually about. Adding the workspace fixes today's run;
    the preflight is what keeps the next silent-wrong-number one regression away from being
    published instead of noticed.

    Killed by: src/uclone_x/evaluation/answerer.py :: if record.status != "success":
    """
    with pytest.raises(EvalToolPreflightError, match="cannot use its tools"):
        preflight_agent_tools(_workspaceless_agent(), workspace)


def test_the_preflight_refuses_a_workspace_root_that_is_not_there(tmp_path: Path) -> None:
    """Named before any tool runs, because "no such directory" has a better message.

    Killed by: src/uclone_x/evaluation/answerer.py :: f"the workspace root {workspace_root} is not a directory, so no tool call can "
    Becomes: f"the workspace root {workspace_root} is unusable, so no tool call can "
    """
    with pytest.raises(EvalToolPreflightError, match="not a directory"):
        preflight_agent_tools(_workspaceless_agent(), tmp_path / "absent")


def test_building_the_answerer_runs_the_preflight(
    stub_connector: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guard has to be on the path the CLI takes, not merely available to it.

    Killed by: src/uclone_x/evaluation/answerer.py :: preflight_agent_tools(agent, root)
    """
    seen: list[Path] = []

    def _refuse(_agent: BaseAgent, root: Path) -> None:
        seen.append(root)
        raise EvalToolPreflightError("tools are dead")

    monkeypatch.setattr("uclone_x.evaluation.answerer.preflight_agent_tools", _refuse)

    with pytest.raises(EvalToolPreflightError):
        build_agent_answerer(provider="mock", workspace_root=tmp_path)

    assert seen == [tmp_path.resolve()]


def test_the_preflight_probes_the_file_tool_the_problems_actually_call(
    workspace: Path,
) -> None:
    """`file_io` is 74 problems and `code_exec` 45 -- but the axis is not the tool.

    The first version of this preflight probed `file_search` and `bash_run`, and passed with
    `file_read` absent from the registry entirely. The 74 file_io problems reach for
    `file_read`, so a green preflight said nothing about the tool they would use.

    Killed by: src/uclone_x/evaluation/answerer.py :: ("file_read", {"path": str(target), "max_lines": 1}),
    """
    probed = {name for name, _args in preflight_probes(workspace)}
    assert probed == set(PREFLIGHT_PROBE_TOOLS)
    assert "file_read" in probed


def test_the_file_read_probe_names_a_file_that_is_there(workspace: Path) -> None:
    """The probe argument cannot be a constant: only a path found in the tree exists."""
    args = dict(next(a for name, a in preflight_probes(workspace) if name == "file_read"))
    assert (workspace / str(args["path"])).is_file()


def test_the_preflight_refuses_a_workspace_with_nothing_to_read(tmp_path: Path) -> None:
    """A skipped probe is how `file_read` went unexercised; refuse instead of skipping.

    Killed by: src/uclone_x/evaluation/answerer.py :: if target is None:
    """
    with pytest.raises(EvalToolPreflightError, match="no readable file"):
        preflight_agent_tools(_workspaceless_agent(), tmp_path)


def test_the_preflight_refuses_a_dirty_git_checkout(stub_connector: None, tmp_path: Path) -> None:
    """`BashRunTool` validates `cwd` and never the command, and both shells are kept.

    The default root is `git rev-parse --show-toplevel` -- the operator's live checkout,
    uncommitted work and `.git` included -- and a hundred problems, 45 of which run
    commands, get to touch it. Refusing a dirty tree does not make the run safe; it bounds
    what an escape can destroy to work that git can return.

    Killed by: src/uclone_x/evaluation/answerer.py :: if git.is_repo and git.dirty:
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "README.md").write_text("committed\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)

    # Clean: the preflight has no objection, so the refusal below is about the dirt and not
    # about the tree merely being a repository.
    build_agent_answerer(provider="mock", workspace_root=tmp_path)

    (tmp_path / "uncommitted.md").write_text("a day's unsaved work\n")

    with pytest.raises(EvalToolPreflightError, match="uncommitted changes"):
        build_agent_answerer(provider="mock", workspace_root=tmp_path)


def test_the_workspace_git_state_is_recorded_for_the_report(
    stub_connector: None, tmp_path: Path
) -> None:
    """P0-2 wants a dataset SHA on a baseline; the tree under test has one too.

    Two runs against the same path at different commits are two different measurements, and
    `workspace_root` alone cannot tell them apart.

    Killed by: src/uclone_x/evaluation/answerer.py :: self.workspace_git = workspace_git_state(workspace_root)
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "README.md").write_text("committed\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)

    answerer = build_agent_answerer(provider="mock", workspace_root=tmp_path)

    assert answerer.workspace_git.is_repo is True
    assert answerer.workspace_git.dirty is False
    sha = answerer.workspace_git.sha
    assert sha is not None and len(sha) == 40


def test_the_first_dirty_path_survives_when_it_is_an_unstaged_modification(
    tmp_path: Path,
) -> None:
    r"""The refusal message must name a file that exists, or the operator cannot act on it.

    `--porcelain` puts the staged column first, so an unstaged-only modification is
    `' M README.md'`. Stripping leading whitespace off the whole block eats that space on
    the first line alone, and the `[3:]` slice then takes a character of the filename with
    it: the operator is told to commit or stash `'EADME.md'`. Only the first entry, only
    when it is unstaged-only -- hence the ordering below, which is the only arrangement
    that shows it.

    Killed by: src/uclone_x/evaluation/answerer.py :: return out.stdout.rstrip("\n")
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "README.md").write_text("committed\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)

    # Unstaged, never added: the ' M' form. A second, later-sorting entry keeps this one
    # first in the porcelain block without being the entry under test.
    (tmp_path / "README.md").write_text("edited, not staged\n")
    (tmp_path / "zz-untracked.txt").write_text("untracked\n")

    state = workspace_git_state(tmp_path)

    assert state.dirty is True
    assert state.dirty_paths == ("README.md", "zz-untracked.txt")
    # The message the operator reads, not just the tuple behind it.
    with pytest.raises(EvalToolPreflightError, match=r"changes: README\.md,"):
        preflight_agent_tools(_workspaceless_agent(), tmp_path)


def test_a_non_repository_workspace_is_recorded_as_such(
    stub_connector: None, workspace: Path
) -> None:
    """`is_repo=False` is an answer; a missing field would read as "clean"."""
    state = workspace_git_state(workspace)
    assert state.is_repo is False
    assert state.sha is None
    assert state.dirty is None


def test_a_working_agent_passes_the_preflight(stub_connector: None, workspace: Path) -> None:
    """A guard that cannot pass is a guard that will be removed."""
    build_agent_answerer(provider="mock", workspace_root=workspace)


# --- the agent's own tool path --------------------------------------------------------


def test_execute_tool_call_uses_the_agents_own_workspace_resolution(tmp_path: Path) -> None:
    """The preflight must ask along the path the agent uses, or it answers a different question.

    A probe that constructed its own `ToolContext` would report `success` for the agent in
    #666, whose tools were fine and whose workspace never reached them.

    Killed by: src/uclone_x/agent/base.py :: _messages, records = await self._execute_tools([request])
    """
    (tmp_path / "found.md").write_text("sentinel\n")
    agent = BaseAgent(
        config=AgentConfig(agent_id="a", name="A", system_prompt="s", workspace_dir=tmp_path),
        llm=MockLLMConnector(default_response="stubbed"),
        tools=create_default_registry(enable_mcp=False),
        context=AgentContext(session_id="sess_a", agent_id="a"),
    )

    ok = asyncio.run(agent.execute_tool_call("file_search", {"query": "sentinel", "path": "."}))
    assert ok.status == "success", ok.error
    assert "found.md" in str(ok.output)

    dead = asyncio.run(_workspaceless_agent().execute_tool_call("file_search", {"query": "x"}))
    assert dead.status == "error"
    assert "workspace" in str(dead.error)


def test_execute_tool_call_names_a_tool_that_is_not_registered() -> None:
    """An unregistered name is the caller's mistake, not a tool failure to be graded.

    Without the check it surfaces as `IndexError` on an empty result list, which reads like
    the tool ran and returned nothing -- the one reading a preflight must never be given.

    Killed by: src/uclone_x/agent/base.py :: if self._resolve_tool(name) is None:
    Becomes: if False:
    """
    with pytest.raises(KeyError, match="not registered"):
        asyncio.run(_workspaceless_agent().execute_tool_call("no_such_tool"))


def test_the_measured_agent_cannot_edit_the_tree_its_answers_are_checked_against(
    stub_connector: None, tmp_path: Path
) -> None:
    """Giving the agent a workspace also gave it write access, and the first run used it.

    `qwen3:1.7b` renamed a row in `docs/nfr-performance-budgets.md` part-way through the
    100 problems. That is not a capability finding: every later problem about that file is
    then a question about a tree the benchmark itself changed. All 100 frontier problems
    ask the agent to check a claim; none ask it to modify anything.

    Containment, not isolation -- `bash_run` can still write, and some problems ask for a
    command to be run. This asserts the part that is closed.

    Killed by: src/uclone_x/evaluation/answerer.py :: if t.name not in WITHHELD_TOOLS
    """
    target = tmp_path / "ground_truth.md"
    target.write_text("the answer is seven\n")

    answerer = build_agent_answerer(provider="mock", workspace_root=tmp_path)

    assert "file_write" not in answerer.agent.config.allowed_tools
    assert "file_edit" not in answerer.agent.config.allowed_tools
    for withheld in ("file_write", "file_edit"):
        with pytest.raises(KeyError, match="not registered"):
            asyncio.run(
                answerer.agent.execute_tool_call(
                    withheld, {"path": "ground_truth.md", "content": "the answer is eight"}
                )
            )
    assert target.read_text() == "the answer is seven\n"


def test_the_measured_agent_keeps_the_tools_the_problems_need(
    stub_connector: None, workspace: Path
) -> None:
    """Withholding the write tools must not quietly withhold the read ones as well."""
    answerer = build_agent_answerer(provider="mock", workspace_root=workspace)
    allowed = set(answerer.agent.config.allowed_tools)
    assert {"file_read", "file_search", "bash_run"} <= allowed


# --- the turn -------------------------------------------------------------------------


def _answerer_over(
    turn_result: Any, workspace: Path, *, request_timeout_s: float | None = None
) -> AgentAnswerer:
    """An `AgentAnswerer` whose agent returns a prepared `TurnResult`.

    The turn outcome is what is under test, not how the provider produced it, and stubbing
    `execute_turn` is the only way to present `is_completed=False` deterministically.
    """

    class _Agent:
        def reset_session(self) -> None:
            return None

        async def execute_turn(self, _prompt: str) -> Any:
            return turn_result

    answerer = AgentAnswerer.__new__(AgentAnswerer)
    answerer._agent = _Agent()  # type: ignore[attr-defined]
    answerer.workspace_root = workspace
    answerer.workspace_git = workspace_git_state(workspace)
    answerer.request_timeout_s = request_timeout_s
    answerer.last_tool_call_count = None
    answerer.last_text_emitted_tool_calls = None
    return answerer


def test_a_turn_that_did_not_complete_is_raised_not_returned(workspace: Path) -> None:
    """`execute_turn` reports a provider failure in its result, not by raising.

    So `return str(turn.content)` turned each of nine `ReadTimeout`s into an empty answer,
    graded `NO_ANSWER`, scored, and counted as a healthy probe. Raising is what routes it to
    the suite's `except Exception`, where it is recorded `UNREACHABLE` and excluded.

    Killed by: src/uclone_x/evaluation/answerer.py :: if turn.error is not None or not turn.is_completed:
    """
    from uclone_x.agent.models import TurnResult

    failed = TurnResult(
        turn_index=0,
        content="",
        is_completed=False,
        error="ReadTimeout",
        provenance=None,
    )
    with pytest.raises(EvalTurnFailedError, match="did not complete"):
        _answerer_over(failed, workspace)("anything")


def test_a_completed_turn_that_carries_an_error_is_also_raised(workspace: Path) -> None:
    """Either half is enough: content alongside an error is not an answer to grade."""
    from uclone_x.agent.models import TurnResult

    half_failed = TurnResult(
        turn_index=0,
        content="a partial answer",
        is_completed=True,
        error="provider returned 500 after a retry",
        provenance=None,
    )
    with pytest.raises(EvalTurnFailedError):
        _answerer_over(half_failed, workspace)("anything")


def test_a_completed_turn_records_how_many_tools_it_used(workspace: Path) -> None:
    """The report has to be able to say the tools ran without a reader grepping logs.

    Killed by: src/uclone_x/evaluation/answerer.py :: self.last_tool_call_count = len(turn.tool_executions)
    """
    from uclone_x.agent.models import ToolExecutionRecord, TurnResult
    from uclone_x.tools.models import ToolResultStatus

    executed = TurnResult(
        turn_index=0,
        content="checked the tree",
        is_completed=True,
        tool_executions=(
            ToolExecutionRecord(tool_name="file_read", status=ToolResultStatus.SUCCESS),
            ToolExecutionRecord(tool_name="bash_run", status=ToolResultStatus.SUCCESS),
        ),
        provenance=None,
    )
    answerer = _answerer_over(executed, workspace)
    assert answerer("anything") == "checked the tree"
    assert answerer.last_tool_call_count == 2


def test_a_failed_turn_leaves_no_stale_tool_count_behind(workspace: Path) -> None:
    """A count carried over from the previous problem would be attributed to this one."""
    from uclone_x.agent.models import TurnResult

    answerer = _answerer_over(
        TurnResult(turn_index=0, content="", is_completed=False, error="x", provenance=None),
        workspace,
    )
    answerer.last_tool_call_count = 7
    with pytest.raises(EvalTurnFailedError):
        answerer("anything")
    assert answerer.last_tool_call_count is None


def test_a_turn_cut_off_at_the_ceiling_is_raised_as_its_own_kind(workspace: Path) -> None:
    """A truncated turn and a broken one both fail the probe, but not for the same reason.

    `EvalTurnFailedError` already keeps both out of the grade, which is right and is not
    changed here. What #1277 needed is the second half: the suite must be able to say in
    the report *which* it was, and prose in `turn.error` is not a thing a suite can branch
    on. A subclass means every existing `except EvalTurnFailedError` still catches it and
    still excludes it, while a handler that wants the distinction can ask for it.

    Killed by: src/uclone_x/evaluation/answerer.py :: if turn.stop_reason == "provider_timeout":
    Becomes: if False:
    """
    from uclone_x.agent.models import TurnResult

    cut_off = TurnResult(
        turn_index=0,
        content="",
        is_completed=False,
        error="Ollama did not answer within 600s (ReadTimeout)",
        stop_reason="provider_timeout",
        provenance=None,
    )
    with pytest.raises(EvalTurnTimedOutError) as excinfo:
        _answerer_over(cut_off, workspace, request_timeout_s=600.0)("anything")

    # Still the type that keeps a failed probe out of the grade.
    assert isinstance(excinfo.value, EvalTurnFailedError)
    # The ceiling that expired is in the message, because the remedy is that number.
    assert "600s" in str(excinfo.value)


def test_an_ordinary_failed_turn_is_not_reported_as_a_ceiling_expiring(workspace: Path) -> None:
    """The narrowing must be a narrowing: everything else keeps the reason it had.

    If every incomplete turn became `EvalTurnTimedOutError`, the report would name a
    ceiling as the cause of failures that had nothing to do with one, and raising it would
    be the recommended fix for a daemon that was never running.
    """
    from uclone_x.agent.models import TurnResult

    broken = TurnResult(
        turn_index=0,
        content="",
        is_completed=False,
        error="Failed to connect to Ollama: ConnectError",
        stop_reason="not_started",
        provenance=None,
    )
    with pytest.raises(EvalTurnFailedError) as excinfo:
        _answerer_over(broken, workspace, request_timeout_s=600.0)("anything")

    assert not isinstance(excinfo.value, EvalTurnTimedOutError)


def test_an_unknown_ceiling_is_said_to_be_unknown_rather_than_guessed(workspace: Path) -> None:
    """`request_timeout_s=None` means "nothing on record says", not "the default".

    An `AgentAnswerer` composed by hand has no ceiling recorded, and printing the
    connector's default there would be a fabricated number in a report about a truncation
    -- the one number a reader of #1277 would act on.

    Killed by: src/uclone_x/evaluation/answerer.py :: else "the ceiling in force"
    Becomes: else f"{LIVE_EVAL_REQUEST_TIMEOUT_SECONDS:g}s"
    """
    from uclone_x.agent.models import TurnResult

    cut_off = TurnResult(
        turn_index=0,
        content="",
        is_completed=False,
        error="cut off",
        stop_reason="provider_timeout",
        provenance=None,
    )
    with pytest.raises(EvalTurnTimedOutError) as excinfo:
        _answerer_over(cut_off, workspace)("anything")

    message = str(excinfo.value)
    assert "the ceiling in force" in message
    assert f"{LIVE_EVAL_REQUEST_TIMEOUT_SECONDS:g}s" not in message


def test_the_live_answerer_is_built_at_a_ceiling_it_can_report(workspace: Path) -> None:
    """The ceiling is a settable parameter and the answerer remembers the value used.

    #1277's third requirement is that the ceiling a run used is recorded with the run. The
    suite reads it off the answerer, so an answerer that passed a timeout to the connector
    without keeping it would leave the report unable to say what it measured at -- and two
    runs at different ceilings are two different measurements, not a before and an after.

    Killed by: src/uclone_x/evaluation/answerer.py :: request_timeout_s=request_timeout_s,
    Becomes: request_timeout_s=None,
    """
    seen: dict[str, Any] = {}

    def _factory(**kwargs: Any) -> MockLLMConnector:
        seen.update(kwargs)
        return MockLLMConnector(default_response="stubbed")

    import uclone_x.llm.connectors.factory as factory_module

    original = factory_module.create_llm_connector
    factory_module.create_llm_connector = _factory  # type: ignore[assignment]
    try:
        answerer = build_agent_answerer(
            provider="mock", workspace_root=workspace, request_timeout_s=123.0
        )
    finally:
        factory_module.create_llm_connector = original  # type: ignore[assignment]

    assert seen["timeout"] == 123.0
    assert answerer.request_timeout_s == 123.0


def test_the_default_live_ceiling_is_not_the_connectors_interactive_one() -> None:
    """The connector default stays 60s for chat; the live suite names its own (#1277).

    The 60s default is right for an interactive turn, where a caller waiting ten minutes
    for a reply is a worse failure than a short one. A benchmark's upper tiers are the
    opposite case, and #1277 is what happens when one number serves both: the frontier
    ladder's hardest tiers could not be measured at all, because every probe that needed
    longer than a chat reply was recorded as an unreachable host.
    """
    import inspect

    from uclone_x.llm.connectors.ollama import (
        DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        OllamaConnector,
    )

    interactive_default = inspect.signature(OllamaConnector.__init__).parameters["timeout"].default
    assert interactive_default == DEFAULT_OLLAMA_TIMEOUT_SECONDS
    assert interactive_default == 180.0
    assert LIVE_EVAL_REQUEST_TIMEOUT_SECONDS > interactive_default


# --- the allowlist on the direct tool path --------------------------------------------


def test_execute_tool_call_enforces_allowed_tools(tmp_path: Path) -> None:
    """The allowlist is applied at advertise time; a direct call must not walk around it.

    `execute_turn` filters the tool *definitions* shown to the model, and `_execute_tools`
    does a bare registry lookup. A public entry point that skipped the check would let any
    caller reach any registered tool while `config.allowed_tools` said otherwise -- and the
    next caller will read the allowlist as a boundary.

    `PermissionError`, not `KeyError`: "not allowed" and "not there" are different answers.

    Killed by: src/uclone_x/agent/base.py :: if allowed and name not in allowed:
    """
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="a",
            name="A",
            system_prompt="s",
            workspace_dir=tmp_path,
            allowed_tools=("file_read",),
        ),
        llm=MockLLMConnector(default_response="stubbed"),
        tools=create_default_registry(enable_mcp=False),
        context=AgentContext(session_id="sess_a", agent_id="a"),
    )

    (tmp_path / "ground_truth.md").write_text("the answer is seven\n")
    with pytest.raises(PermissionError, match="allowed_tools"):
        asyncio.run(
            agent.execute_tool_call(
                "file_write", {"path": "ground_truth.md", "content": "the answer is eight"}
            )
        )
    assert (tmp_path / "ground_truth.md").read_text() == "the answer is seven\n"

    permitted = asyncio.run(agent.execute_tool_call("file_read", {"path": "ground_truth.md"}))
    assert permitted.status == "success", permitted.error


def test_an_empty_allowlist_still_means_every_registered_tool(tmp_path: Path) -> None:
    """`allowed_tools=()` is "unset", as it is at advertise time; changing that here would
    silently disarm every agent composed without one."""
    agent = BaseAgent(
        config=AgentConfig(agent_id="a", name="A", system_prompt="s", workspace_dir=tmp_path),
        llm=MockLLMConnector(default_response="stubbed"),
        tools=create_default_registry(enable_mcp=False),
        context=AgentContext(session_id="sess_a", agent_id="a"),
    )
    (tmp_path / "x.md").write_text("hello\n")
    record = asyncio.run(agent.execute_tool_call("file_read", {"path": "x.md"}))
    assert record.status == "success", record.error


def test_a_completed_turn_records_the_calls_that_were_emitted_as_text(workspace: Path) -> None:
    """The suite cannot tell the two readings of a zero apart unless this is carried.

    `last_tool_call_count == 0` alone describes both a model that declined to use tools --
    a capability measurement -- and a model whose every call was written into the message
    content and discarded, which is a broken harness whose graded answer is the discarded
    call itself (#694). Drop this line and the run reports the first when it is the second.

    Killed by: src/uclone_x/evaluation/answerer.py :: self.last_text_emitted_tool_calls = tuple(turn.text_emitted_tool_calls)
    """
    from uclone_x.agent.models import TurnResult

    discarded = TurnResult(
        turn_index=0,
        content='{"name": "file_read", "arguments": {"path": "pyproject.toml"}}',
        is_completed=True,
        text_emitted_tool_calls=("file_read",),
        provenance=None,
    )
    answerer = _answerer_over(discarded, workspace)

    answerer("anything")

    assert answerer.last_tool_call_count == 0
    assert answerer.last_text_emitted_tool_calls == ("file_read",)


def test_a_completed_turn_records_the_claims_that_rested_on_nothing(workspace: Path) -> None:
    """The observation #700 compares and #733 can be rebuilt on, carried to the suite.

    `TurnResult.unsupported_claims` names the specifics an answer asserted that appear in
    nothing the turn read. Exposing it here is what keeps the report-side work an `evals/`
    change: without it, a suite that wants to say "this tier's rate came from answers that
    consulted nothing" has to reach into the agent.

    It is an observation and not a verdict -- see `agent/grounding.py` for what the support
    test can and cannot tell -- so a consumer should report it, not score on it alone.

    Killed by: src/uclone_x/evaluation/answerer.py ::
        self.last_unsupported_claims = tuple(turn.unsupported_claims)
    """
    from uclone_x.agent.models import TurnResult

    guessed = TurnResult(
        turn_index=0,
        content="The default is 100.",
        is_completed=True,
        unsupported_claims=("100",),
        provenance=None,
    )
    answerer = _answerer_over(guessed, workspace)

    answerer("what is the default?")

    assert answerer.last_unsupported_claims == ("100",)
