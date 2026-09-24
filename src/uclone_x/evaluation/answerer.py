"""Construction of the agent a live evaluation suite measures.

A live suite refuses to build its own agent. `FrontierLiveSuite` says why: what is
measured has to be the caller's own composition, not a convenience object the benchmark
invented for itself, or the benchmark ends up grading a thing that no user ever runs.

That refusal was written and the other half was left for later. `EvalRunner.run` takes an
`options` mapping and threads it to the suite, but nothing ever passed one, and no test
covers the path — so `frontier_live` was registered, listed by `ucx eval list`, described
accurately, and unreachable. It is not that nobody chose to run it; the command did not
work (#664).

This module is the other half, and it lives here rather than under `evals/` because the
composition belongs next to the caller that owns it, which is the CLI. (It is *not* that
an installed build would otherwise fail to reach it: with no backend installed the CLI
exits before the answerer is ever constructed. The placement follows the separate-backend
design; it does not rescue it.)

What it builds is deliberately plain: one `BaseAgent`, the default tool registry, one
turn per problem, a fresh session each time. Nothing here retries, reformats a prompt, or
post-processes an answer. Every one of those would improve the score and none of them is
something the runtime does for a real caller, so each would move the measurement further
from the thing it claims to describe.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Coroutine, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uclone_x.agent import BaseAgent

#: What the agent is told it is doing. Kept minimal on purpose: a prompt that coached the
#: model toward the answer format the graders accept would raise the score by teaching to
#: the test, and the frontier set's `traps` exist precisely to catch an answer shaped
#: right and wrong underneath.
DEFAULT_SYSTEM_PROMPT = (
    "You are a UClone-X agent answering an evaluation question. Use the tools available "
    "to you to check claims against the repository rather than answering from memory. "
    "If you cannot determine the answer, say so plainly instead of guessing."
)

#: The per-request ceiling a live evaluation run sets for itself, in seconds.
#:
#: Declared here rather than inherited, which is the whole of #1277. The connector default
#: is 60 s and stays 60 s: it is written for interactive chat, where a caller waiting on a
#: prompt is better served by a bounded wait than by a reply that eventually arrives. A
#: capability run is the opposite case. Nobody is waiting, and the problems whose declared
#: horizons are 15 to 30+ are the ones the set exists to measure, so a ceiling tuned for a
#: one-shot question deletes exactly the evidence the run was taken for: on 2026-09-20,
#: seven probes ended `UNREACHABLE` on a read timeout -- three of them at 60.014, 60.014
#: and 60.065 s, which is the signature of a cut-off at the 60 s default, and four only
#: after several round-trips had each spent one, at 84.726, 100.195, 114.159 and
#: 217.525 s -- and tier 9 lost five of its ten problems that way.
#:
#: Those seven are the read-timeout subset of that run's eight `UNREACHABLE` probes; the
#: eighth failed for an unrelated reason. They come from
#: `evals/reports/20260920-185937-frontier_live.json` (`qwen3:8b` via Ollama, 110
#: problems, n=1), which is *not* in the tree -- `evals/reports/*.json` is ignored -- so
#: nothing here re-derives them and no test can pin them. That is how this sentence came
#: to carry a figure, 59.618 s, that appears nowhere in the run it cited (#1291). Before
#: citing these again, re-derive them from the report:
#:
#:     probes = json.load(open(report))["probes"]
#:     sorted(p["duration_s"] for p in probes if "ReadTimeout" in (p["message"] or ""))
#:
#: 600 s is ten times that ceiling, chosen so that a cut-off at this one means the request
#: stalled rather than that it was long. It is not a claim that 600 s is enough -- no run
#: has been taken at it yet -- which is why it is a parameter (`--request-timeout`) and why
#: `frontier_live` records the value it ran at in `metadata.request_timeout_s`. Two runs at
#: different ceilings are two different measurements, and the report has to say which.
#:
#: This is a per-*request* ceiling, not a per-problem or per-run one: a turn may take
#: several model round-trips and each gets its own. It bounds a stall, not the run.
LIVE_EVAL_REQUEST_TIMEOUT_SECONDS: float = 600.0


def run_coroutine(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine from sync code, whether or not a loop is already running.

    The suite calls the answerer synchronously, so a bare `asyncio.run` is correct only
    while nothing above it is async. Under a running loop it raises `RuntimeError`, and
    `FrontierLiveSuite` catches a per-problem exception and records `UNREACHABLE` -- so a
    loop-hosted caller would receive a complete, normal-looking hundred-problem report
    scoring zero, with the actual cause visible nowhere. Nothing hosts one today; a run
    endpoint on the dashboard is the obvious next step, and five suites in this tree
    already carry this bridge.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


class EvalTurnFailedError(RuntimeError):
    """Raised when an agent turn did not complete, so its content is not an answer.

    `execute_turn` does not raise on a provider failure: it returns
    `TurnResult(content="", is_completed=False, error=...)`. An answerer that returned
    `str(turn.content)` regardless therefore handed the suite an empty string, which grades
    `NO_ANSWER` -- a *scored* outcome, inside the `correct_rate` denominator, on a probe
    marked `harness_ok=True`. A run that timed out nine times then reported 100/100 harness
    health and quietly enlarged the denominator it was measured against.

    Raising is what routes the failure to the suite's `except Exception`, where it is
    recorded `UNREACHABLE` and excluded from the rates. The distinction the two suites exist
    to keep -- the agent failing is data, the harness failing is not -- is only real if the
    harness's own failures are made to look like harness failures.
    """


class EvalTurnTimedOutError(EvalTurnFailedError):
    """The turn was cut off at the ceiling the run set, rather than failing (#1277).

    A subclass, because a cut-off turn *is* a turn that did not complete and every
    `except EvalTurnFailedError` must keep catching it. It is a distinct type because it
    is a distinct fact: the host answered, the run's own deadline expired, and the repair
    is a longer ceiling rather than a running daemon.

    It exists so `frontier_live` can report the difference. Both endings are `UNREACHABLE`
    in the grade and both must stay there -- a truncated turn is not the model failing,
    and folding it into `WRONG` would make the score fall as the ceiling tightens. What
    changes is that the report now says which of the two happened instead of leaving it to
    whoever notices that no failed probe ran longer than the ceiling.
    """


class EvalToolPreflightError(RuntimeError):
    """Raised when the composed agent cannot execute a tool in its workspace.

    Named, rather than a bare `RuntimeError`, because the whole point of the check is that
    a caller can distinguish "the tools are dead" from any other construction failure and
    stop before a report is written (#666).
    """


#: The tools the preflight must exercise. `file_search` and `bash_run` cover the two
#: capability axes the frontier set leans on -- 84 of the 110 problems carry `file_io` and
#: 45 carry `code_exec` -- but the axis is not the tool: the `file_io` problems reach for
#: `file_read`, and a preflight that probed only `file_search` passed with `file_read`
#: absent from the registry entirely. A probe per tool the problems actually call, then.
PREFLIGHT_PROBE_TOOLS: frozenset[str] = frozenset({"file_read", "file_search", "bash_run"})

#: Directory names never descended when looking for a file to probe `file_read` with.
#: `.git` first: reading a pack file proves nothing about the tree under measurement.
_PROBE_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
)

#: How many directories the probe-target walk will visit before giving up. A workspace whose
#: first readable file is deeper than this is not one the preflight can vouch for quickly,
#: and saying so is better than walking a large tree at every run start.
_PROBE_WALK_LIMIT = 200


def _find_probe_target(workspace_root: Path) -> Path | None:
    """Return a workspace-relative path to some readable regular file, or `None`.

    `file_read` needs a file that exists, and the preflight may not create one: the whole
    point of the run is that the tree is not modified by the thing measuring it. So the
    target is discovered. Preferred names first, so the common case reads the same file
    every run and the probe is reproducible; a bounded walk after that, so an arbitrary
    `--workspace-root` is still covered.
    """
    for preferred in ("README.md", "pyproject.toml", "AGENTS.md"):
        candidate = workspace_root / preferred
        if candidate.is_file() and not candidate.is_symlink():
            return Path(preferred)

    visited = 0
    for dirpath, dirnames, filenames in os.walk(workspace_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _PROBE_SKIP_DIRS)
        for filename in sorted(filenames):
            candidate = Path(dirpath) / filename
            if candidate.is_file() and not candidate.is_symlink():
                return candidate.relative_to(workspace_root)
        visited += 1
        if visited >= _PROBE_WALK_LIMIT:
            break
    return None


def preflight_probes(workspace_root: Path) -> tuple[tuple[str, dict[str, Any]], ...]:
    """The probe calls to make, one per tool in `PREFLIGHT_PROBE_TOOLS`.

    Each is trivial, read-only and bounded, and each goes through the agent's own tool path
    rather than a reconstruction of it -- a probe that built its own `ToolContext` would
    pass in exactly the state #666 describes, where the tool works and the agent never hands
    it a workspace.

    `file_read`'s argument is the one that cannot be a constant, because the only path
    guaranteed to exist is one found in the tree being probed. If no such file is found the
    run is refused rather than the probe skipped: a skipped probe is how `file_read` went
    unexercised in the first place, and a workspace with nothing to read cannot answer the
    74 problems that read files.
    """
    target = _find_probe_target(workspace_root)
    if target is None:
        raise EvalToolPreflightError(
            f"no readable file was found under the workspace root {workspace_root}, so the "
            "`file_read` probe cannot run and the 74 file_io problems have nothing to read. "
            "Pass a populated tree with --workspace-root."
        )
    probes: list[tuple[str, dict[str, Any]]] = [
        ("file_read", {"path": str(target), "max_lines": 1}),
        (
            "file_search",
            {"query": "e", "path": ".", "glob_pattern": "*.md", "max_results": 1},
        ),
        (
            "bash_run",
            {"action": "run", "command": "pwd", "timeout_seconds": 30},
        ),
    ]
    return tuple(probes)


#: Tools withheld from the measured agent. Giving the agent a workspace (#666) also gave it
#: write access to the tree its own answers are checked against, and the first live run used
#: it: `qwen3:1.7b` renamed a row in `docs/nfr-performance-budgets.md` mid-run. That is not a
#: capability finding, it is the benchmark editing its own ground truth -- every later
#: problem about that file is then asking about a tree the run itself changed.
#:
#: No frontier problem asks the agent to modify anything; all 100 ask it to check a claim.
#: This is containment of the common case and not isolation. The shell is kept, because 45
#: problems deliberately ask for a command to be run. The default registry still carries it
#: under two names -- `bash_run` and `run_command`, both `BashRunTool` -- and both stay in
#: `allowed_tools`, but since #1461 (merged 2026-09-23, `c2aa75af`) a request advertises
#: only `bash_run`: `run_command` is an alias and `drop_shadowed_aliases` keeps it out of
#: any request that carries the canonical tool (#1424). Runs before that merge offered the
#: measured agent **two** shells, runs after it one, so the tool list differs across that
#: date in eval history (#1463). Either name can write, delete, or escape the root with
#: `../`: `BashRunTool` validates its working directory and never the command string. What bounds the damage today is the
#: preflight's refusal to start on a dirty checkout. Real containment is a P3 sandbox or a
#: disposable checkout, and it is not built here.
WITHHELD_TOOLS: frozenset[str] = frozenset({"file_write", "file_edit"})


def default_workspace_root() -> Path:
    """The repository the answers are to be checked against.

    The frontier problems ask about *this* tree, so the default is the checkout the command
    was invoked from, not `Path.cwd()` blindly: a run started from `evals/` has to reach the
    same files as a run started from the root or the same question gets two answers.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        candidate = out.stdout.strip()
        if candidate:
            return Path(candidate).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return Path.cwd().resolve()


@dataclass(frozen=True)
class WorkspaceGitState:
    """What git says about the tree the answers were checked against.

    `workspace_root` alone does not identify a measurement. P0-2 asks a baseline to carry
    the SHA of the dataset it was taken against; the frontier problems' other input is the
    *repository*, and two runs against the same path at different commits are two different
    measurements. `dirty` is the second half: run 1 of #666 modified the tree mid-run and
    the report had no field in which to say so.
    """

    is_repo: bool
    sha: str | None = None
    dirty: bool | None = None
    dirty_paths: tuple[str, ...] = ()


def _git(root: Path, *args: str) -> str | None:
    """Run one git command under `root`, returning its stdout with the trailing newline off.

    Only the trailing newline: `--porcelain` encodes the *staged* column as the first
    character of every line, so an unstaged-only modification starts with a space
    (`' M README.md'`). A `.strip()` here eats that space on the first line alone, and the
    `[3:]` slice that follows then removes a character of the filename instead -- telling
    the operator to stash `'EADME.md'`, a path that does not exist.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return out.stdout.rstrip("\n")


def workspace_git_state(workspace_root: Path) -> WorkspaceGitState:
    """Resolve the commit and cleanliness of the checkout the agent will operate on.

    Status is asked of the whole repository, not of `workspace_root` alone: what matters is
    whether there is uncommitted work that a stray `bash_run` could destroy, and that work
    does not have to sit under the subdirectory the benchmark was pointed at.
    """
    if not workspace_root.is_dir():
        return WorkspaceGitState(is_repo=False)
    if _git(workspace_root, "rev-parse", "--is-inside-work-tree") != "true":
        return WorkspaceGitState(is_repo=False)
    sha = _git(workspace_root, "rev-parse", "HEAD")
    status = _git(workspace_root, "status", "--porcelain")
    if status is None:
        return WorkspaceGitState(is_repo=True, sha=sha)
    paths = tuple(line[3:] for line in status.splitlines() if line.strip())
    return WorkspaceGitState(is_repo=True, sha=sha, dirty=bool(paths), dirty_paths=paths)


def create_disposable_checkout(
    source_root: Path | None = None,
) -> tuple[Path, Callable[[], None]]:
    """Create an isolated, disposable checkout of a repository or workspace.

    Returns a tuple of `(disposable_path, cleanup_callable)`.

    If `source_root` is inside a git repository with a valid HEAD commit, a detached
    git worktree is added at HEAD in a temporary directory. If it is not a git repository
    (or worktree creation fails), it falls back to copying the directory tree.

    `cleanup_callable` safely and idempotently removes the disposable worktree/directory
    and unregisters from `atexit`.
    """
    root = default_workspace_root() if source_root is None else Path(source_root).resolve()

    temp_dir = Path(tempfile.mkdtemp(prefix="ucx-eval-disposable-")).resolve()

    is_git_repo = False
    if root.is_dir() and _git(root, "rev-parse", "--is-inside-work-tree") == "true":
        head_sha = _git(root, "rev-parse", "--verify", "HEAD")
        if head_sha is not None:
            is_git_repo = True

    used_git_worktree = False
    if is_git_repo:
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "worktree", "add", "--detach", str(temp_dir), "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                used_git_worktree = True
        except Exception:
            pass

    if not used_git_worktree and root.is_dir():
        shutil.copytree(
            root,
            temp_dir,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"
            ),
        )

    cleaned = False

    def cleanup() -> None:
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        try:
            atexit.unregister(cleanup)
        except Exception:
            pass
        if used_git_worktree:
            try:
                subprocess.run(
                    ["git", "-C", str(root), "worktree", "remove", "--force", str(temp_dir)],
                    capture_output=True,
                    check=False,
                )
            except Exception:
                pass
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            try:
                subprocess.run(
                    ["git", "-C", str(root), "worktree", "prune"],
                    capture_output=True,
                    check=False,
                )
            except Exception:
                pass
        else:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    atexit.register(cleanup)
    return temp_dir, cleanup


@contextmanager
def disposable_workspace(source_root: Path | None = None) -> Generator[Path, None, None]:
    """Context manager yielding a disposable checkout path and ensuring cleanup."""
    path, cleanup = create_disposable_checkout(source_root)
    try:
        yield path
    finally:
        cleanup()


def preflight_agent_tools(agent: BaseAgent, workspace_root: Path) -> None:
    """Execute the probe tool calls, raising `EvalToolPreflightError` on the first failure.

    A capability run whose tools are dead still produces a number -- a real number, honestly
    graded, measuring a toolless model while looking exactly like a figure for the agent.
    That is worse than a crash, because the report cannot show it. So the run refuses to
    start rather than degrading (#666).
    """
    if not workspace_root.is_dir():
        raise EvalToolPreflightError(
            f"the workspace root {workspace_root} is not a directory, so no tool call can "
            "read the repository the frontier problems ask about. Pass an existing tree "
            "with --workspace-root."
        )

    # Containment, before capability. `BashRunTool` validates `cwd` and never the command
    # string, so `bash_run` -- kept, because 45 problems need it -- can `rm -rf`, `sed -i`,
    # or escape the root with `../` at any point in a hundred problems. The default root is
    # `git rev-parse --show-toplevel`: the operator's live checkout, uncommitted work and
    # `.git` included. Refusing a dirty tree does not make the run safe; it makes the
    # blast radius recoverable, which is the difference between losing a commit and losing
    # a day's unsaved work. A disposable checkout is the real fix and is not built here.
    git = workspace_git_state(workspace_root)
    if git.is_repo and git.dirty:
        shown = ", ".join(git.dirty_paths[:5])
        more = f" (+{len(git.dirty_paths) - 5} more)" if len(git.dirty_paths) > 5 else ""
        raise EvalToolPreflightError(
            f"the workspace root {workspace_root} is a git repository with uncommitted "
            f"changes: {shown}{more}. The measured agent keeps both shells, which validate "
            "only their working directory and not the command, so a run can destroy that "
            "work with no way for the report to say it did. Commit or stash first, or point "
            "the benchmark at a throwaway copy with --workspace-root."
        )

    for tool_name, arguments in preflight_probes(workspace_root):
        try:
            record = run_coroutine(agent.execute_tool_call(tool_name, arguments))
        except Exception as exc:  # noqa: BLE001 - any failure here is the same verdict
            raise EvalToolPreflightError(
                f"the preflight tool call {tool_name!r} raised "
                f"{type(exc).__name__}: {exc}. The agent cannot use its tools, so a run "
                "would measure the model's weights rather than the runtime."
            ) from exc
        if record.status != "success":
            raise EvalToolPreflightError(
                f"the preflight tool call {tool_name!r} failed in {workspace_root}: "
                f"{record.error or record.status}. The agent cannot use its tools, so a "
                "run would measure the model's weights rather than the runtime."
            )


class AgentAnswerer:
    """One agent turn per prompt, carrying the workspace root it was composed with.

    The root travels on the answerer rather than being passed beside it, so that whatever a
    report records is the tree the answers were actually checked against and cannot drift
    from it.
    """

    def __init__(
        self,
        agent: BaseAgent,
        workspace_root: Path,
        *,
        cleanup: Callable[[], None] | None = None,
        is_disposable: bool = False,
        request_timeout_s: float | None = None,
    ) -> None:
        self._agent = agent
        self.workspace_root = workspace_root
        self._cleanup = cleanup
        self._is_disposable = is_disposable
        #: The per-request ceiling this composition set on its connector, or `None` when
        #: the caller built the agent itself and the ceiling is not knowable from here.
        #: Carried on the answerer for the same reason `workspace_root` is: the suite
        #: records what the run actually used rather than what a default says it used, so
        #: the recorded ceiling cannot drift from the one in force (#1277).
        #:
        #: `None` is not a ceiling of zero and not the connector default. It means "not
        #: knowable", and a report that printed 60.0 for it would be asserting something
        #: about a composition it never saw.
        self.request_timeout_s = request_timeout_s
        #: Resolved once, at composition, so what the report records is the state of the
        #: tree when the run started rather than whatever it had become by the end.
        self.workspace_git = workspace_git_state(workspace_root)
        #: Tool executions in the most recent turn, or `None` before the first one and
        #: after a turn that failed. The suite reads it to record what the agent actually
        #: did: a hundred problems answered with zero tool calls is the #666 state, and the
        #: report should be able to say so without a reader grepping the logs.
        self.last_tool_call_count: int | None = None
        #: Model round-trips the last turn consumed. Kept beside the tool count because
        #: the two answer different questions: how much was looked up, and how much
        #: reasoning it took to get there. #700 compares this against each problem's
        #: declared horizon; exposed here so that comparison stays an `evals/` change.
        self.last_steps_taken: int | None = None
        #: Registered tools the most recent turn invoked in the message content instead of
        #: the structured channel, or `None` before the first turn and after a failed one.
        #: `last_tool_call_count == 0` alone cannot tell a model that declined to use tools
        #: from one whose every call was discarded, and those two readings of the same run
        #: have opposite meanings: the first is a capability measurement, the second is a
        #: broken harness (#694).
        self.last_text_emitted_tool_calls: tuple[str, ...] | None = None
        #: Specifics the most recent answer asserted that appear in nothing the turn read,
        #: or `None` before the first turn and after a failed one. Carried here so the
        #: report-side comparison stays an `evals/` change: #700 reads it beside
        #: `last_steps_taken`, and it is the runtime-side fabrication signal left after
        #: trap-based detection was removed from the eval set (#733). An observation, not a
        #: verdict -- `agent/grounding.py` says what the support test can and cannot tell.
        self.last_unsupported_claims: tuple[str, ...] | None = None

    @property
    def is_disposable(self) -> bool:
        """Whether this answerer operates on an isolated, disposable checkout."""
        return self._is_disposable

    def close(self) -> None:
        """Clean up any temporary disposable checkout or resources allocated for this answerer."""
        if self._cleanup is not None:
            cleanup = self._cleanup
            self._cleanup = None
            cleanup()

    def __enter__(self) -> AgentAnswerer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def agent(self) -> BaseAgent:
        """The composed agent under measurement."""
        return self._agent

    def __call__(self, prompt: str) -> str:
        """Answer one prompt in a fresh session.

        Raises `EvalTurnFailedError` when the turn did not complete. `execute_turn` reports
        a provider failure in the returned `TurnResult` rather than by raising, so returning
        `turn.content` unconditionally converts a timeout into an empty answer -- which the
        grader scores `NO_ANSWER`, counts toward `correct_rate`, and marks harness-healthy.
        """
        # A fresh session per problem. Without this the hundredth problem answers with the
        # ninety-nine before it in context, which measures the compaction path rather than
        # the ladder -- and makes the score depend on the order the problems happen to be
        # in.
        self._agent.reset_session()
        self.last_tool_call_count = None
        self.last_steps_taken = None
        self.last_text_emitted_tool_calls = None
        self.last_unsupported_claims = None
        turn = run_coroutine(self._agent.execute_turn(prompt))
        if turn.stop_reason == "provider_timeout":
            # Read off the field rather than matched in `turn.error`, so the two layers
            # agree by type instead of by wording (#1277). Checked before the general
            # case below because it is the same condition narrowed: a cut-off turn is
            # also an incomplete one, and the broader branch would swallow it.
            ceiling = (
                f"{self.request_timeout_s:g}s"
                if self.request_timeout_s is not None
                else "the ceiling in force"
            )
            raise EvalTurnTimedOutError(
                f"the agent turn was cut off at {ceiling}: {turn.error or 'no error reported'}. "
                "The host answered and the run's own deadline expired, so this is neither an "
                "answer nor an unreachable host, and must be graded as neither."
            )
        if turn.error is not None or not turn.is_completed:
            raise EvalTurnFailedError(
                f"the agent turn did not complete (is_completed={turn.is_completed}): "
                f"{turn.error or 'no error reported'}. This is a harness failure, not an "
                "answer, and must not be graded as one."
            )
        self.last_tool_call_count = len(turn.tool_executions)
        self.last_steps_taken = turn.steps_taken
        self.last_text_emitted_tool_calls = tuple(turn.text_emitted_tool_calls)
        self.last_unsupported_claims = tuple(turn.unsupported_claims)
        return str(turn.content)


def build_agent_answerer(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    agent_id: str = "eval_live_answerer",
    workspace_root: Path | str | None = None,
    preflight: bool = True,
    disposable: bool = True,
    request_timeout_s: float = LIVE_EVAL_REQUEST_TIMEOUT_SECONDS,
) -> AgentAnswerer:
    """Return a callable that answers one prompt with one agent turn.

    Raises whatever `create_llm_connector` raises when the provider is absent or
    unreachable. That is deliberate: a capability figure produced from a mock would be
    worse than none, so an endpoint that is not there has to stop the run rather than
    quietly degrade it.

    `workspace_root` is the tree the agent's tools operate on. When omitted (`None`),
    `frontier_live` isolates the run by executing against a disposable checkout created
    from HEAD (#687) rather than the operator's live working tree. If an explicit
    `workspace_root` is provided, it is used directly without creating a disposable copy,
    and the preflight refuses to start if it is a dirty git repository.

    Unless `preflight` is disabled, the composed agent executes `PREFLIGHT_PROBES` before
    it is handed back, and `EvalToolPreflightError` stops the run if any of them fails.

    `request_timeout_s` is the per-request ceiling the connector is built with, and the
    answerer carries it so the suite records the value the run actually used. It is named
    here rather than left to the connector default because the two are answers to
    different questions -- see `LIVE_EVAL_REQUEST_TIMEOUT_SECONDS` (#1277).
    """
    from uclone_x.agent import BaseAgent
    from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
    from uclone_x.llm.connectors.factory import create_llm_connector
    from uclone_x.tools.registry import ToolRegistry, create_default_registry

    cleanup: Callable[[], None] | None = None
    is_disposable = False
    if workspace_root is not None:
        root = Path(workspace_root).resolve()
    elif disposable:
        root, cleanup = create_disposable_checkout()
        is_disposable = True
    else:
        root = default_workspace_root()

    try:
        connector = create_llm_connector(provider=provider, timeout=request_timeout_s)

        # MCP off: `create_default_registry` otherwise loads `./mcp.json`, `~/.uclone/mcp.json`
        # and the rest, so the toolset the benchmark measures would be whatever the operator
        # happens to have configured. A score that moves with the machine it ran on is not
        # comparable to the next one.
        #
        # No `workspace_root=` here, deliberately: `create_default_registry` passes it only to
        # the MCP loader, so with MCP off it is a no-op that reads as though it were what wires
        # the agent to the tree. The wiring is `AgentConfig.workspace_dir` below, and one
        # plausible-looking decoy is how #666 stayed invisible.
        full_registry = create_default_registry(enable_mcp=False)
        kept = [t for t in full_registry.list_tools() if t.name not in WITHHELD_TOOLS]
        read_only_registry = ToolRegistry(tools=kept)

        config = AgentConfig(
            agent_id=agent_id,
            name="EvalLiveAnswerer",
            system_prompt=system_prompt,
            # Every frontier problem is a claim about a tree this agent holds. An answer here
            # that consulted nothing is not an abstention, it is a guess -- #697 measured 28 of
            # 100 arriving that way, one of them saying in its own words that it had inferred
            # the value rather than read it. One more round is cheap against a run measured in
            # minutes; a plausible unevidenced answer is not.
            require_evidence_before_answer=True,
            llm_config=AgentLLMConfig(model_name=model),
            # Withheld in the registry as well as here: the registry stops the call, this stops
            # the model being shown a tool it is not allowed to use and spending a step
            # discovering that.
            allowed_tools=tuple(sorted(t.name for t in kept)),
            # Without this the tool host gets no workspace and every file read and every
            # command fails with "Tool requires a workspace, but the host provided none" --
            # and the run still produces a plausible-looking capability figure (#666).
            workspace_dir=root,
        )
        agent = BaseAgent(
            config=config,
            llm=connector,
            tools=read_only_registry,
            # The root is declared once, on the config. `AgentContext.workspace_root` would
            # resolve first and set the same value, and two places that must agree is one place
            # that can silently disagree.
            context=AgentContext(session_id=f"sess_{agent_id}", agent_id=agent_id),
        )

        if preflight:
            preflight_agent_tools(agent, root)
    except Exception:
        if cleanup is not None:
            cleanup()
        raise

    return AgentAnswerer(
        agent,
        root,
        cleanup=cleanup,
        is_disposable=is_disposable,
        request_timeout_s=request_timeout_s,
    )


__all__ = [
    "LIVE_EVAL_REQUEST_TIMEOUT_SECONDS",
    "WITHHELD_TOOLS",
    "AgentAnswerer",
    "DEFAULT_SYSTEM_PROMPT",
    "EvalToolPreflightError",
    "EvalTurnFailedError",
    "EvalTurnTimedOutError",
    "PREFLIGHT_PROBE_TOOLS",
    "WorkspaceGitState",
    "build_agent_answerer",
    "create_disposable_checkout",
    "default_workspace_root",
    "disposable_workspace",
    "preflight_agent_tools",
    "preflight_probes",
    "run_coroutine",
    "workspace_git_state",
]
