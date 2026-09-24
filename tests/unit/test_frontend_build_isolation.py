# pyright: reportPrivateUsage=false
"""What stops the suite from running the real frontend build, and how it fails when it does.

Four unit tests call the real `uclone_x.ui.server.start_ui_server`, which calls
`_ensure_frontend_built`, which runs a real `npm run build` in `frontend/` whenever the
committed bundle is older than the source. That rewrote `src/uclone_x/ui_static` — tracked
files that gate stage 6b verifies — and deleted its `assets/` directory for about 1.5 s
mid-build, failing whichever parallel worker happened to read the bundle in that window
(#1067).

Those four tests now take the `frontend_build_suppressed` fixture. The tests here pin the
second half of the answer: the session-wide guard in
`tests/support/frontend_build_guard.py` that makes a *missing* suppression a loud failure
of the offending test rather than a silently rebuilt bundle — so the protection covers the
class of tests that reach the launcher, not only the four that exist today.

The guard has two interception points, and the tests below are in that order: the
`subprocess.run` route `_ensure_frontend_built` takes (#1067), then the
`subprocess.Popen` route the same launcher's `dev=True` branch takes to co-spawn Vite
(#1076), which `call` and `check_call` also reach. The last test is the one the second
route needed: `test_ui_server.py` stubs the module attribute `subprocess.Popen` for that
same spawn, and the guard binds the class's `__init__`, so it asserts that neither of
those disarms the other while both are in place.

Nothing here asserts that the four call sites carry the fixture. That claim needs no
assertion of its own: without it, the guard fails those tests outright, and an assertion
that merely re-read the patched attribute would pass in a tree where the bundle happened
to be fresh and so would prove nothing (R28).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import httpx
import pytest

from tests.support.frontend_build_guard import (
    FRONTEND_TOOLCHAIN_PROGRAMS,
    RealFrontendBuildInTestError,
    ToolchainInsideLiveTreeError,
    program_name,
    run_outside_the_live_tree,
)
from tests.support.vite_diagnosis import (
    FOREIGN_APP_BODY,
    FOREIGN_APP_TITLE,
    answering_as,
    one_line,
)
from uclone_x.ui import server


def _always_rebuild(frontend_dir: Path, static_dir: Path) -> bool:
    """Stand in for `_should_rebuild_frontend`, whose answer depends on file mtimes."""
    return True


def test_the_real_build_helper_is_stopped_before_it_can_run_npm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper still reaches for `npm`, and the guard is what stops it getting there.

    Pointed at a scratch repository rather than this one, so the case is the same whether
    or not the real bundle happens to be fresh — and so that a regression cannot rebuild
    anything real while being measured.

    Killed by: tests/support/frontend_build_guard.py :: if _lift_depth == 0 and program_name(command) in FRONTEND_TOOLCHAIN_PROGRAMS:
    Becomes: if False:
    """
    repo = tmp_path / "repo"
    frontend = repo / "frontend"
    frontend.mkdir(parents=True)
    (frontend / "package.json").write_text('{"scripts": {"build": "exit 3"}}\n')
    monkeypatch.setattr(server, "__file__", str(repo / "src" / "uclone_x" / "ui" / "server.py"))
    monkeypatch.setattr(server, "_should_rebuild_frontend", _always_rebuild)

    with pytest.raises(RealFrontendBuildInTestError) as caught:
        server._ensure_frontend_built()

    assert "ui_static" in str(caught.value)


def test_the_refusal_is_not_swallowed_by_the_except_exception_it_guards() -> None:
    """An `except Exception:` around a build would absorb a guard that raised an ordinary
    exception: npm blocked, nothing reported, the test still green — the silence this guard
    exists to break. The `try` below is that shape, reproduced.

    `_ensure_frontend_built` was that shape until #1075 narrowed its handler to the two
    exceptions a failed `subprocess.run` raises. The reproduction stays, and is the point:
    the guard's class must survive a handler being widened back, wherever that happens, and
    a test that read the current width out of production code would stop asserting the
    property the moment the width was right.

    Killed by: tests/support/frontend_build_guard.py :: class RealFrontendBuildInTestError(BaseException):
    Becomes: class RealFrontendBuildInTestError(Exception):
    """
    with pytest.raises(RealFrontendBuildInTestError):
        try:
            subprocess.run(["npm", "run", "build"], check=True, capture_output=True)
        except Exception as exc:
            pytest.fail(f"an `except Exception:` absorbed the refusal: {exc!r}")


def test_a_toolchain_named_by_path_is_refused_and_every_other_program_still_runs(
    tmp_path: Path,
) -> None:
    """Refusal is decided by the program's basename, and it stops there.

    `npm run build` is not the only spelling: `quality_gate.py` names `npm` bare, while a
    launcher resolving it through `shutil.which` would pass an absolute path. The path used
    here does not exist, so under the mutation the call fails to find a program instead of
    running one.

    The second half is the guard's cost: every other subprocess the suite makes — `git`,
    `ps`, the dashboard children in `test_cli_ui_stop.py` — must still run untouched.

    Killed by: tests/support/frontend_build_guard.py :: return PurePath(first).name
    Becomes: return first
    """
    npm_by_path = tmp_path / "node" / "bin" / "npm"

    with pytest.raises(RealFrontendBuildInTestError):
        subprocess.run([str(npm_by_path), "run", "build"], capture_output=True)

    completed = subprocess.run(
        [sys.executable, "-c", "print('not a build')"], capture_output=True, text=True
    )

    assert completed.stdout.strip() == "not a build"


def _refusal_raised_by(invoke: Callable[[list[str]], object], argv: list[str]) -> str:
    """The refusal message `invoke(argv)` raises, so two routes' messages can be compared."""
    with pytest.raises(RealFrontendBuildInTestError) as caught:
        invoke(argv)
    return str(caught.value)


def test_popen_call_and_check_call_refuse_the_toolchain_exactly_as_run_does(
    tmp_path: Path,
) -> None:
    """The `npm run dev` half of the hazard: `server.py:245-246` spawns Vite through `Popen`.

    `call` builds a `Popen` and `check_call` calls `call`, so one interception on the
    class's `__init__` covers all three — asserted here rather than assumed, because that
    is a fact about the stdlib and the stdlib has restructured these functions before.
    The messages are compared to `run`'s rather than merely being of the right type: the
    refusal is the only place a reader learns what the guard does and does not inspect, so
    a second route that refuses with a thinner explanation is a defect.

    `npm` is spelled as a path that does not exist, as the `run` case above spells it and
    for the same reason: under the mutation the call fails to find a program instead of
    starting a real Vite dev server that nothing would then shut down.

    The last section is the guard's cost on this route — every other child the suite
    spawns through `Popen`, including the dashboards in `test_cli_ui_stop.py`, must still
    run untouched.

    Killed by: tests/support/frontend_build_guard.py :: subprocess.Popen.__init__ = guarded_init
    Becomes: subprocess.Popen.__init__ = current_init
    """
    argv = [str(tmp_path / "node" / "bin" / "npm"), "run", "dev", "--", "--port", "5173"]

    through_run = _refusal_raised_by(lambda command: subprocess.run(command), argv)

    assert _refusal_raised_by(subprocess.Popen, argv) == through_run
    assert _refusal_raised_by(subprocess.call, argv) == through_run
    assert _refusal_raised_by(subprocess.check_call, argv) == through_run

    spawned = subprocess.Popen(
        [sys.executable, "-c", "print('not a build')"], stdout=subprocess.PIPE, text=True
    )
    stdout, _ = spawned.communicate()

    assert stdout.strip() == "not a build"


def test_the_popen_refusal_is_not_swallowed_by_the_except_exception_around_the_vite_spawn(
    tmp_path: Path,
) -> None:
    """`start_ui_server` wraps its Vite `Popen` in a second `except Exception:` (#1076).

    `_ensure_frontend_built` has one around its `subprocess.run`, pinned above; this is the
    other one. Measured at this branch's head: the `try:` opens at `server.py:240` and
    wraps the `Popen` at 245-250, and its handler is the `except Exception as e:` at
    **261-264** that prints `Could not auto-start Vite dev server` and returns. It is *not*
    the `except Exception:` at 257-258 — that one is inside `_cleanup_vite`, runs at
    interpreter exit, and never sees the construction. An ordinary exception raised on this
    route would be absorbed at 261-264, the dev server would silently not start, and the
    test would stay green. The `try` below is that production shape, reproduced.

    Killed by: tests/support/frontend_build_guard.py :: class RealFrontendBuildInTestError(BaseException):
    Becomes: class RealFrontendBuildInTestError(Exception):
    """
    argv = [str(tmp_path / "node" / "bin" / "npm"), "run", "dev"]

    with pytest.raises(RealFrontendBuildInTestError):
        try:
            subprocess.Popen(argv)
        except Exception as exc:
            pytest.fail(f"an `except Exception:` absorbed the refusal: {exc!r}")


@pytest.mark.usefixtures("frontend_build_suppressed")
def test_the_vite_popen_stub_and_the_guard_do_not_disarm_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The composition this issue is actually about, demonstrated in one place.

    `test_ui_server.py::test_start_ui_server_dev_mode` replaces the **module attribute**
    `subprocess.Popen` with a `MagicMock` to catch the co-spawned Vite server; this guard
    wraps the **class's** `__init__`. Those are different attributes, and the point of
    choosing the class one is that neither disarms the other. That the four
    `start_ui_server` tests still pass is not evidence of it — a guard disarmed by the stub
    passes them too — so both halves are asserted here directly, with the stub bound.

    The launcher is the real one, so the argv asserted on is the argv `server.py:245-246`
    actually builds rather than a copy of it that could drift.

    The launcher arguments are unchanged from the test this one mirrors, because mirroring
    them is what it exists to pin. What did change is the transport underneath
    `_diagnose_vite`, which the stubbed `Popen` leaves the launcher reaching for: it was a
    real ten-second poll of the developer's own port 5173 (#1082), which cost ten seconds
    here and let that port pick the branch. `answering_as` fixes the response to a page
    that is *not* this project's, so this test pins the foreign-app branch while
    `test_ui_server.py::test_start_ui_server_dev_mode` pins the ours branch — the two tests
    overlap in the launcher they call, but no longer in the outcome they assert.

    Killed by: tests/support/frontend_build_guard.py :: subprocess.Popen.__init__ = guarded_init
    Becomes: subprocess.Popen.__init__ = current_init
    """
    from uclone_x.shells import ui_process

    real_popen = subprocess.Popen
    stub = MagicMock()
    monkeypatch.setattr("uvicorn.run", MagicMock())
    # The dashboard record reads this process's identity through `ps`, which the stub would
    # otherwise answer with a `MagicMock`.
    identity = ui_process.process_identity(os.getpid())
    monkeypatch.setattr(ui_process, "process_identity", MagicMock(return_value=identity))
    monkeypatch.setattr("subprocess.Popen", stub)
    monkeypatch.setattr(httpx, "get", answering_as(FOREIGN_APP_BODY))

    server.start_ui_server(port=5180, dev=True, host="127.0.0.1", vite_port=5173)

    # Which branch the launcher took, asserted rather than left to the machine: the page
    # served carries no `VITE_IDENTITY_MARKER`, so HMR must be reported unavailable and the
    # foreign application named.
    printed = one_line(capsys.readouterr().out)
    assert "React HMR is not available" in printed, printed
    assert FOREIGN_APP_TITLE in printed, printed
    assert "Instant React HMR Active" not in printed, printed

    # The stub is not disarmed: it received the Vite argv, so nothing real was spawned.
    # `start_ui_server` makes other `Popen` calls on this path, so the Vite one is selected
    # by its program rather than by being the only one.
    vite_calls = [
        call
        for call in stub.call_args_list
        if call.args and program_name(call.args[0]) in FRONTEND_TOOLCHAIN_PROGRAMS
    ]
    assert len(vite_calls) == 1
    argv = cast("list[str]", vite_calls[0].args[0])
    assert argv[:3] == ["npm", "run", "dev"]

    # The guard is not disarmed: the same argv through the real class still refuses, here,
    # while the stub is still the module attribute every other caller would reach.
    assert subprocess.Popen is stub
    with pytest.raises(RealFrontendBuildInTestError):
        escaped = real_popen(
            argv, cwd=str(tmp_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Only reached if the guard is gone, in which case a real Vite is now running.
        escaped.kill()
        escaped.wait(timeout=10)
        pytest.fail("the stub disarmed the guard: a real `npm run dev` was spawned")
    # A mutation run that disables the interception therefore does start one real `npm`,
    # in a `tmp_path` with no `package.json` so it exits at once, and kills it above. That
    # is the cost of asserting the guard rather than re-reading the attribute it patched.


# ---------------------------------------------------------------------------
# The one sanctioned way past the guard, and what it checks instead (#1077)
# ---------------------------------------------------------------------------


def test_the_lift_refuses_a_cwd_inside_the_live_tree() -> None:
    """The lift enforces the invariant the program-name refusal stands in for.

    The refusal is not really about `npm`. It is about a test writing into the live
    `frontend/`, where a mutant decides gate stage 6b and where a parallel worker reads the
    file (#967, #1067). `run_outside_the_live_tree` therefore checks the thing that actually
    matters — the working directory — and refuses inside the tree even though the caller has
    asked for the exemption. Without this the exemption is a way to turn the guard off.

    Killed by: tests/support/frontend_build_guard.py :: if resolved_cwd == resolved_live or resolved_live in resolved_cwd.parents:
    Becomes: if False:
    """
    with pytest.raises(ToolchainInsideLiveTreeError) as refused:
        run_outside_the_live_tree(
            ["npx", "vitest", "run"],
            cwd=Path.cwd() / "frontend",
            live_root=Path.cwd(),
        )
    assert "inside the live worktree" in str(refused.value)


def test_the_lift_runs_the_toolchain_outside_the_tree_and_re_arms_afterwards(
    tmp_path: Path,
) -> None:
    """Inside the lift the toolchain runs; outside it, one line later, it is refused again.

    Both halves are needed. Without the first, an exemption that never actually lifted would
    pass and the lethality ratchet would have no way to run vitest at all. Without the second,
    a lift that never restored would leave the guard off for the rest of the process — the
    failure that would matter, because it is silent and the suite goes green under it.

    `_ORIGINAL_RUN` alone is not enough to explain why the counter exists: the real
    `subprocess.run` builds a `subprocess.Popen`, and the wrapper installed on that class is
    what would refuse. Suspending the shared decision is the only thing that reaches both
    interception points.

    Killed by: tests/support/frontend_build_guard.py :: _lift_depth += 1
    Becomes: _lift_depth += 0
    """
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    done = run_outside_the_live_tree(
        ["npx", "--version"],
        cwd=outside,
        live_root=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr

    with pytest.raises(RealFrontendBuildInTestError):
        subprocess.run(["npm", "run", "build"], cwd=str(outside), check=False)
