"""Global pytest fixtures and configuration."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

import uclone_x
from tests.support.frontend_build_guard import install_frontend_build_guard
from tests.support.live_optin import (
    LIVE_SKIP_REASON,
    PRE_RELEASE_SKIP_REASON,
    apply_live_skip,
    apply_pre_release_skip,
)
from tests.support.session_leak_guard import watching_session_writes
from uclone_x.agent.session import (
    DEFAULT_SESSION_STORAGE_DIR,
    SESSION_STORAGE_DIR_ENV_VAR,
)
from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR, DEFAULT_AGENTS_ROOT
from uclone_x.llm.connectors.ollama import OLLAMA_ENDPOINT_ENV_VARS
from uclone_x.llm.connectors.vllm import (
    VLLM_ENDPOINT_ENV_VARS,
    VLLM_MODEL_ENV_VAR,
)
from uclone_x.shells.ui_process import DASHBOARD_STATE_DIR_ENV_VAR, UI_BIND_HOST_ENV_VAR
from uclone_x.tools.builtin import mcp_loader

# Collection-time isolation: ensure UCLONE_SESSION_DIR is established immediately
# when conftest is imported, preventing import-time singletons (such as the module-level
# FastAPI app in uclone_x.ui.app) from defaulting to DEFAULT_SESSION_STORAGE_DIR.
_COLLECTION_SESSION_DIR = Path(tempfile.gettempdir()) / "uclone-test-session-collection"
_COLLECTION_SESSION_DIR.mkdir(parents=True, exist_ok=True)
if SESSION_STORAGE_DIR_ENV_VAR not in os.environ:
    os.environ[SESSION_STORAGE_DIR_ENV_VAR] = str(_COLLECTION_SESSION_DIR)

# The same, for agent homes. `default_cross_session_memory` creates the directory and
# writes the `id` file as it builds a store, so a module-level composition reaching it
# during collection would write into `~/.uclone/agents` before any fixture could redirect
# it -- which is how 89 test-named directories appeared there while this guard was absent.
_COLLECTION_AGENTS_DIR = Path(tempfile.gettempdir()) / "uclone-test-agents-collection"
_COLLECTION_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
if AGENTS_DIR_ENV_VAR not in os.environ:
    os.environ[AGENTS_DIR_ENV_VAR] = str(_COLLECTION_AGENTS_DIR)

# ---------------------------------------------------------------------------
# Inherited `GIT_*`: scrubbed for the whole session, at import, before collection.
# ---------------------------------------------------------------------------
#
# `git` exports **`GIT_DIR`** — an absolute path to the pushing checkout's git directory —
# into the environment of every hook it runs, whenever the repository is not the plain
# `.git` beside the current directory. A linked worktree is exactly that case, so a
# hook that runs `./ucx test check` (the `pre-push` hook did until #966), and therefore
# this suite, runs with `GIT_DIR` naming that worktree.
#
# `GIT_DIR` outranks `git -C <dir>`: `-C` only changes the working directory, while
# `GIT_DIR` decides which repository is operated on. A test that builds a scratch
# repository and runs `git -C <tmp_path> add -A` therefore stages `<tmp_path>`'s contents
# **into the pushing worktree's index**, collapsing it to the one file the scratch tree
# holds. Measured against `origin/main` at 553be7f: `git ls-files` went 629 -> 1 from
# `tests/unit/test_evaluation_answerer.py` alone, and the push was then refused by the
# damage rather than by the branch (#709). `git reset` repairs it; nothing says so.
#
# Scrubbed here rather than at each call site, because a per-call-site fix protects only
# the call sites that exist today, and the damage is silent. It is scrubbed at **import**
# rather than only in a fixture because module-level and collection-time `git` calls
# (`REPO_ROOT = git rev-parse ...`, `pytest_configure` below) run before any fixture does;
# the autouse fixture further down is the second line, covering anything that puts a
# `GIT_*` variable back mid-session.
#
# Removing them is what makes a hook-invoked run identical to a hand-invoked one, which is
# the only environment this suite has ever been verified in.
# The distribution-surface fitness check already did this for its own subprocesses; this
# generalises that pattern to the session.
_GIT_ENV_PREFIX = "GIT_"
_SCRUBBED_GIT_ENV: dict[str, str] = {
    name: os.environ.pop(name) for name in list(os.environ) if name.startswith(_GIT_ENV_PREFIX)
}

# `TestClient` addresses the app as `http://127.0.0.1`, not Starlette's `http://testserver`.
# A dashboard bound to loopback refuses any request whose `Host` is not a loopback name --
# the DNS-rebinding defence of #1413 -- and `testserver` is not one. Defaulted here, for
# every client at once, so the suite talks to the app the way a browser on this machine
# does rather than switching the guard off; a test that means another `Host` passes
# `base_url` or a `Host` header, which still wins.
try:
    from starlette.testclient import TestClient as _StarletteTestClient
except ImportError:  # the `http` extra is not installed; no app to address
    pass
else:
    _orig_test_client_init = _StarletteTestClient.__init__

    def _loopback_test_client_init(
        self: _StarletteTestClient,
        app: Any,
        base_url: str = "http://127.0.0.1",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _orig_test_client_init(self, app, base_url, *args, **kwargs)

    _StarletteTestClient.__init__ = _loopback_test_client_init  # type: ignore[method-assign]


# Structural guard for child processes spawned by tests: ensure any subprocess
# created via subprocess.Popen or asyncio.create_subprocess_* inherits/receives
# UCLONE_SESSION_DIR even if a test passed a custom or sanitized env dictionary.
# Both variables are carried, not just the session one: an agent home is created by the
# same kind of incidental composition a session is, and a child handed a sanitized env
# would write into the invoking developer's `~/.uclone/agents`.
_ISOLATED_CHILD_ENV_VARS = (SESSION_STORAGE_DIR_ENV_VAR, AGENTS_DIR_ENV_VAR)


def _carry_isolated_dirs_into_child_env(kwargs: dict[str, Any]) -> None:
    """Add this process's redirected storage directories to an explicit child `env`."""
    env = kwargs.get("env")
    if env is None:
        return
    for name in _ISOLATED_CHILD_ENV_VARS:
        value = os.environ.get(name)
        if value is not None and name not in env:
            env = dict(env)
            env[name] = value
            kwargs["env"] = env


_orig_popen_init = subprocess.Popen.__init__


def _guarded_popen_init(self: subprocess.Popen[Any], *args: Any, **kwargs: Any) -> None:
    _carry_isolated_dirs_into_child_env(kwargs)
    _orig_popen_init(self, *args, **kwargs)


subprocess.Popen.__init__ = _guarded_popen_init  # pyright: ignore[reportAttributeAccessIssue]

_orig_asyncio_exec = asyncio.create_subprocess_exec
_orig_asyncio_shell = asyncio.create_subprocess_shell


async def _guarded_asyncio_exec(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
    _carry_isolated_dirs_into_child_env(kwargs)
    return await _orig_asyncio_exec(*args, **kwargs)


async def _guarded_asyncio_shell(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
    _carry_isolated_dirs_into_child_env(kwargs)
    return await _orig_asyncio_shell(*args, **kwargs)


asyncio.create_subprocess_exec = _guarded_asyncio_exec  # pyright: ignore[reportAttributeAccessIssue]
asyncio.create_subprocess_shell = _guarded_asyncio_shell  # pyright: ignore[reportAttributeAccessIssue]

# Structural guard against the suite rebuilding the committed frontend bundle: no test may
# run npm/npx/pnpm/yarn/bun, because `npm run build` rewrites `src/uclone_x/ui_static`,
# which is checked in and which gate stage 6b verifies (#1067). Installed at import for the
# same reason as the three guards above — collection-time code runs before any fixture.
# Why the refusal is a `BaseException`, and why `node` is not on the list:
# `tests/support/frontend_build_guard.py`.
install_frontend_build_guard()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the Tier 3 opt-in flag (see `tests/support/live_optin.py`) and pre-release scenario flag."""
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run Tier 3 tests against a real LLM endpoint (costs tokens off the local provider)",
    )
    parser.addoption(
        "--pre-release",
        action="store_true",
        default=False,
        help="Run Pre-Release Qualification scenarios (acceptance tests for release qualification)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every `live`-marked test unless `--live` was passed, and `pre_release` tests unless requested."""
    apply_live_skip(
        live_enabled=bool(config.getoption("--live")),
        items=items,
        skip_marker=pytest.mark.skip(reason=LIVE_SKIP_REASON),
    )
    markexpr = getattr(config.option, "markexpr", "")
    pre_release_enabled = bool(config.getoption("--pre-release")) or (
        "pre_release" in markexpr and "not pre_release" not in markexpr
    )
    apply_pre_release_skip(
        pre_release_enabled=pre_release_enabled,
        items=items,
        skip_marker=pytest.mark.skip(reason=PRE_RELEASE_SKIP_REASON),
    )


@pytest.fixture
def sample_fixture() -> str:
    return "uclone_x"


@pytest.fixture(autouse=True)
def _isolate_git_environment(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every test's `git` subprocesses pointed at their own scratch repository.

    The import-time scrub at the top of this file is what actually removes the `GIT_DIR` a
    hook hands down (the `pre-push` hook did until #966); this is the per-test guard that
    keeps it removed, so a test that sets a `GIT_*` variable and forgets to unset it cannot
    leak the same damage into everything scheduled after it.

    A test that is *about* `GIT_*` handling overrides this with its own `monkeypatch`,
    which runs inside the test and therefore wins.
    """
    for name in [name for name in os.environ if name.startswith(_GIT_ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolate_llm_provider(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear LLM configuration so tests do not read the developer's environment.

    Without this, `create_llm_connector()` resolves from ambient variables: a machine with
    `OPENAI_API_KEY` set builds an OpenAI connector and one without it builds something else,
    so a test's behaviour depends on who is running it. Nothing here makes a model call, so
    the variation was invisible rather than absent — the same shape as the session-storage
    isolation below, which exists for the same reason.

    A test that is *about* provider resolution overrides this with its own `monkeypatch`,
    which runs inside the test and therefore wins.

    It **clears** rather than pins. An earlier version set `LLM_PROVIDER=mock` for the whole
    suite, which is stronger than isolation requires: every unmarked test reaching the
    factory then silently received a mock where it previously received something else, and
    nothing asserted the difference. Installing a substitution at the test boundary is
    uncomfortably close to the shape P6 forbids in the code under test. With the variable
    cleared, a test that needs a connector says which one.

    The model and base-URL names are cleared too, because `ui/app.py` writes several of them
    into `os.environ` directly rather than through a fixture, so they leak between tests.
    """
    for name in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_INDEPTH_MODEL",
        "OLLAMA_FAST_MODEL",
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "GEMINI_MODEL",
        "COMFYUI_BASE_URL",
        # The embedding seam's own configuration (#1097). A developer following
        # `docs/semantic-retrieval.md` and exporting these could not otherwise run the
        # suite: the default-resolution tests read exactly these two names.
        "UCLONE_EMBEDDING_MODEL",
        "UCLONE_EMBEDDING_DIMENSIONS",
        *OLLAMA_ENDPOINT_ENV_VARS,
        # vLLM's, for the same reason (#1304). Its endpoint variable now selects a
        # provider in `create_llm_connector`, so a developer who exported it to talk to
        # their own server would otherwise change what every unconfigured-environment
        # test resolves to.
        *VLLM_ENDPOINT_ENV_VARS,
        VLLM_MODEL_ENV_VAR,
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolate_dashboard_records(  # pyright: ignore[reportUnusedFunction]
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Keep every test's dashboard records out of `~/.uclone/ui` (#927).

    `start_ui_server` records the dashboard it launches so `ucx ui stop` can identify it,
    and several tests call it with `uvicorn.run` mocked on the product default port. Left
    at the default, such a test would write — and then delete — a record for port 5180 in
    the developer's home, beside a dashboard they may be running there.
    """
    root = tmp_path_factory.mktemp("dashboard-records")
    monkeypatch.setenv(DASHBOARD_STATE_DIR_ENV_VAR, str(root))
    return root


@pytest.fixture(autouse=True)
def _isolate_ui_bind_host(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear the dashboard's bind address, so `create_ui_app` assumes loopback (#1413).

    `start_ui_server` in development mode exports it for uvicorn's factory; a test that
    runs that path with `uvicorn.run` mocked would otherwise decide, for every app built
    after it, whether the rebinding guard is installed.
    """
    monkeypatch.delenv(UI_BIND_HOST_ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def _isolate_session_storage(  # pyright: ignore[reportUnusedFunction]
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point every test's session storage at a temporary directory and guard real storage.

    **Autouse, because opting in did not work.** `UCLONE_SESSION_DIR` was added in #183
    precisely so a headless run need not write into the invoking user's home.

    Fails a test that leaves a new file in real `DEFAULT_SESSION_STORAGE_DIR` (#453) —
    counting only files **this process wrote** during the test. That storage is shared with
    anything else on the machine, and a person's live dashboard writing a session there
    used to fail whichever test was running (#963). What the attribution covers and does not:
    `tests/support/session_leak_guard.py`.
    """
    root = tmp_path_factory.mktemp("session-storage")
    monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(root))

    with watching_session_writes(DEFAULT_SESSION_STORAGE_DIR.resolve()) as guard:
        yield root

    guard.assert_no_leaks()


@pytest.fixture(autouse=True)
def _isolate_agent_homes(  # pyright: ignore[reportUnusedFunction]
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point every test's agent homes at a temporary root and guard the real one.

    **Autouse, for the reason the session fixture is.** `default_cross_session_memory`
    creates `<root>/<username>/` and writes that agent's `id` as a side effect of building
    a store, so any test that composes an agent -- most of `tests/unit/test_cli_run.py`
    among them -- writes a directory somewhere. Left at the default that somewhere is the
    invoking developer's home, beside whatever agents they actually use.

    The leak guard counts only files **this process wrote**, for the reason given in
    `tests/support/session_leak_guard.py`: the real root is shared with anything else on
    the machine, and a listing comparison cannot tell a test's write from a person's.
    """
    root = tmp_path_factory.mktemp("agent-homes")
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(root))

    with watching_session_writes(DEFAULT_AGENTS_ROOT.resolve(), label="agent home") as homes:
        yield root

    homes.assert_no_leaks()


@pytest.fixture(autouse=True)
def _isolate_workspace_dir(  # pyright: ignore[reportUnusedFunction]
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Give every test an empty workspace instead of the directory pytest was started in.

    `AgentSessionManager` and `RoomResolver`, given no workspace, take
    `UCLONE_WORKSPACE_DIR` and failing that the current directory -- which under pytest is
    the checkout, so the persona registry read the checkout's untracked
    `.uclone/personas/`. A developer's own `artist.yaml` there, declaring `generate_image`
    and `file_read`, failed 8 unit tests with "declares tool(s) that are not registered"
    because the tests' one-tool inventories did not contain them. Autouse for the reason
    the session fixture is: each of those tests built its manager with no workspace
    because it was not about workspaces, and would have kept doing so.

    `monkeypatch` also restores the variable after a test that runs `start_ui_server`
    with a workspace, which writes it into `os.environ` directly.
    """
    root = tmp_path_factory.mktemp("workspace")
    monkeypatch.setenv("UCLONE_WORKSPACE_DIR", str(root))
    return root


@pytest.fixture(autouse=True)
def _isolate_mcp_config_discovery(  # pyright: ignore[reportUnusedFunction]
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Keep the invoking developer's `mcp.json` files out of every test.

    `MCPConfigFileLoader`, given no workspace and no explicit path -- which is how
    `create_default_registry()` builds it for the dashboard and every CLI command -- looks in
    `<cwd>/.uclone/mcp.json`, `<cwd>/mcp.json`, `~/.uclone/mcp.json` and
    `~/.config/uclone/mcp.json`. Under pytest the first two are the checkout and the last two
    are the developer's real home, so a test's tool inventory, and whether it spawns an MCP
    server, depended on the machine. Reading the home config is the product's intended
    behaviour; only the tests are redirected.

    **Scoped to the loader, because the broad forms were measured and break the suite.**
    Setting `HOME` for every test failed 123 of the 124 browser tests: Playwright looked
    for its browser under the temporary home's `Library/Caches/ms-playwright`. `monkeypatch.chdir` into a temporary
    directory failed 79 unit tests, 70 of them in `tests/unit/test_quality_gate.py`.
    So only the loader module's `Path` is replaced: its `home()` is an empty temporary
    directory, and its `cwd()` is another one while the process is still in the directory
    pytest was started from. A test that changes directory itself -- the loader's own
    working-directory discovery tests do -- is answered with that directory, unchanged.
    """
    home = tmp_path_factory.mktemp("mcp-home")
    stand_in_cwd = tmp_path_factory.mktemp("mcp-cwd")
    invocation_dir = request.config.invocation_params.dir.resolve()

    class _IsolatedPath(Path):
        @classmethod
        def home(cls) -> _IsolatedPath:
            return cls(home)

        @classmethod
        def cwd(cls) -> _IsolatedPath:
            here = Path(os.getcwd())
            return cls(stand_in_cwd if here.resolve() == invocation_dir else here)

    monkeypatch.setattr(mcp_loader, "Path", _IsolatedPath)
    return home


@pytest.fixture(autouse=True)
def _isolate_default_persona_registry(  # pyright: ignore[reportUnusedFunction]
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start every test except a browser test without the process-wide persona registry.

    `get_default_persona_registry` caches one registry per process, built with whichever tool
    inventory its first caller supplied, and every built-in persona is validated against that
    inventory. So a test's outcome depended on which test had primed the cache before it in
    the same process. Run serially, an earlier test always had; on parallel workers (#967) the
    order changes, and `test_ui_chat_turn_persists_nested_tool_arguments` failed on a worker
    where it built the registry itself, from its one-tool inventory. Reset rather than
    pre-built: a test that needs the registry now builds it from its own inventory.

    **Not for `e2e` tests.** Their UI servers are module-scoped and run in this process, so the
    registry the reset drops is the one a server that outlives the test is using, and a browser
    test makes no claim about a registry of its own. The PR #977 review measured the cost: at
    1-minute load 6-9, `test_fr13_persona_badge_rendered_when_provided_and_absent_when_none`
    failed in 4 of 4 runs of its file with the reset, and passed 4 of 4 without it, losing its
    reply to the page's load-time history request (#975). The review inferred, without proving,
    that rebuilding the registry mid-test slowed the turn enough to lose that race. At load
    3.6-5.0 the case passed with and without the reset, and its turn took about 215 ms either
    way, so the latency explanation is not established; the exemption rests on the A/B.
    """
    # `FixtureRequest.node` is an unannotated property in pytest; in a function-scoped
    # fixture it is the test item.
    node = cast(pytest.Item, request.node)  # pyright: ignore[reportUnknownMemberType]
    if node.get_closest_marker("e2e") is not None:
        return

    from uclone_x.agent import persona_registry

    monkeypatch.setattr(persona_registry, "_default_persona_registry", None)


@pytest.fixture
def builtin_personas_absent(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load no built-in personas, for a test whose subject is not personas.

    The shipped personas declare tools (`nhis_advisor`: `web_search`, `web_fetch`,
    `file_read`), and the default persona registry refuses a persona whose tools are not in
    the inventory it was built with. A test that registers one mock tool therefore cannot
    build that registry. Such tests used to pass because an earlier test in the same process
    had already built the shared registry from a fuller inventory, and the cache is not
    rebuilt for a second one — so the built-in personas were never checked against their
    tools at all. With `_isolate_default_persona_registry` resetting that cache per unit test
    (#967) the dependency is visible, and this fixture states it instead of inheriting it.
    """
    from uclone_x.agent import persona_registry

    empty = tmp_path_factory.mktemp("no-builtin-personas")
    monkeypatch.setattr(persona_registry, "BUILTIN_PERSONAS_DIR", empty)


def finish_after_tools(llm: object) -> None:
    """Make a `return_value` LLM stub answer once its tools have run.

    The agent takes agent steps until the model stops asking for tools (P4, amended
    2026-09-05). A stub built with `AsyncMock(return_value=<response with tool_calls>)`
    describes a model that asks for the same tool forever: it runs to the step ceiling
    instead of completing. This keeps the canned response for the first invocation and
    returns a terminal answer afterwards, which is what a real model does.

    Call it after the stub is constructed and before the agent runs.
    """
    from unittest.mock import AsyncMock

    from uclone_x.core.provenance import Provenance
    from uclone_x.llm.models import FinishReason, ModelResponse, TokenUsage

    generate = getattr(llm, "generate", None)
    canned = getattr(generate, "return_value", None)
    # Only a stub that asks for tools can spin. One that already answers is left exactly
    # as the test wrote it — including stubs that raise, which must keep raising.
    #
    # `isinstance` rather than a truthiness check: `AsyncMock.return_value` is
    # auto-created, so a stub configured with `side_effect` still yields a MagicMock here
    # whose `.tool_calls` is itself a truthy MagicMock. Guarding on that wrapped four
    # raising stubs and turned their failures into answers.
    if not isinstance(canned, ModelResponse) or not canned.tool_calls:
        return

    terminal = ModelResponse(
        finish_reason=FinishReason.STOP,
        content="Done.",
        tool_calls=(),
        usage=TokenUsage(provider="mock", input_tokens=1, output_tokens=1),
        provenance=Provenance.primary("mock"),
    )
    calls = {"n": 0}

    async def _generate(_request: object) -> ModelResponse:
        calls["n"] += 1
        return canned if calls["n"] == 1 else terminal

    llm.generate = AsyncMock(side_effect=_generate)  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# Invocation guard: fail once, clearly, instead of N times, cryptically.
# ---------------------------------------------------------------------------
#
# This suite depends on *how* it is invoked, and nothing used to say so. Two traps,
# both reproduced rather than imagined:
#
# 1. **A bare `pytest` inside a git worktree imports the PRIMARY workspace's `src`.**
#    The shared `.venv` carries `_editable_impl_uclone_x.pth` pointing at the primary
#    checkout, so without `PYTHONPATH` the worktree's own code is never loaded — the run
#    silently tests whatever branch the primary workspace happens to be on. Measured from
#    a worktree: `python -c "import uclone_x"` resolved to
#    `/Users/…/uclone-x/src/uclone_x/__init__.py`, not the worktree's.
#
# 2. **A relative interpreter path poisons `sys.executable`.** Invoking
#    `../../.venv/bin/python -m pytest` leaves that literal string in `sys.executable`,
#    and the sandbox tests hand it to the workspace path validator, which correctly
#    refuses a path containing `../..`:
#      PathTraversalError: Path '…/.worktrees/x/../../.venv/bin/python' … escapes
#      workspace root '…/pytest-of-…/workspace'
#    Six sandbox tests fail with a traversal error that names neither pytest nor the
#    invocation, and the cause is three layers from the symptom.
#
# `./ucx test check` sets both up correctly, which is why the trap only bites someone
# reaching for `pytest` directly. Set UCLONE_SKIP_INVOCATION_GUARD=1 to bypass
# deliberately — testing an installed wheel, for instance.
_INVOCATION_GUARD_OPT_OUT = "UCLONE_SKIP_INVOCATION_GUARD"


def _is_git_worktree(root: Path) -> bool:
    """True when `root` is a linked worktree rather than the primary checkout."""
    try:
        git_dir = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return False
    return git_dir != common


def pytest_configure(config: pytest.Config) -> None:
    """Verify the suite is running against the tree it was collected from."""
    if os.environ.get(_INVOCATION_GUARD_OPT_OUT):
        return

    problems: list[str] = []
    rootdir = Path(str(config.rootpath)).resolve()

    # Trap 2 first: it is the cheaper check and its message is unmistakable.
    # Normalisation, NOT symlink resolution: a venv's `bin/python` is legitimately a
    # symlink to the interpreter uv manages, so comparing against `resolve()` would flag
    # every correct invocation, `./ucx` included. What breaks the sandbox tests is a `..`
    # segment surviving into `sys.executable`, which `normpath` collapses and `resolve`
    # conflates with the symlink hop.
    executable = sys.executable
    if executable:
        normalised = os.path.normpath(executable)
        if not os.path.isabs(executable) or normalised != executable:
            problems.append(
                f"sys.executable is not a normalised absolute path: {executable!r}\n"
                f"    Normalised, it is {normalised}.\n"
                "    Tests that hand the interpreter path to the sandbox path validator\n"
                "    fail with PathTraversalError, naming neither pytest nor this\n"
                "    invocation. Invoke the interpreter by a path with no '..' segments,\n"
                "    or use `./ucx test check`."
            )

    # Trap 1: only meaningful inside a linked worktree, which is exactly where it bites.
    if _is_git_worktree(rootdir):
        imported = Path(uclone_x.__file__).resolve()
        if not imported.is_relative_to(rootdir):
            problems.append(
                f"`uclone_x` was imported from OUTSIDE this worktree:\n"
                f"    imported: {imported}\n"
                f"    worktree: {rootdir}\n"
                "    The shared .venv's editable .pth points at the primary checkout, so a\n"
                "    bare `pytest` here tests whatever branch that workspace is on — not\n"
                "    the code you are editing, and a green run proves nothing about it.\n"
                f"    Use `./ucx test check`, or set PYTHONPATH={rootdir}/src."
            )

    if problems:
        joined = "\n\n".join(f"  {i}. {p}" for i, p in enumerate(problems, 1))
        raise pytest.UsageError(
            "This suite was invoked in a way that makes its result meaningless:\n\n"
            f"{joined}\n\n"
            f"Set {_INVOCATION_GUARD_OPT_OUT}=1 to bypass this check deliberately."
        )
