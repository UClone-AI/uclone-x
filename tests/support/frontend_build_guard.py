"""Refuse a Node toolchain invocation from inside the test suite (#1067).

`uclone_x.ui.server._ensure_frontend_built` runs a real `npm run build` in `frontend/`
whenever `_should_rebuild_frontend` says the committed bundle is older than the source —
which is the ordinary state of a working tree someone is editing. Four unit tests call the
real `start_ui_server`, which calls that helper, so the suite could shell out to a real
production build. Two harms followed, and the second is the one that matters:

1.  **Flakiness.** Vite's `emptyOutDir` deletes `src/uclone_x/ui_static/assets` for roughly
    1.5 s mid-build. On parallel workers (#967) another worker reading the bundle inside
    that window fails on a bundle nobody changed — this is how #1067 surfaced.
2.  **The suite rewriting tracked files.** Gate stage 6b builds `frontend/` into a
    temporary directory and compares it byte for byte with the committed
    `src/uclone_x/ui_static`. A test run that rebuilds the committed bundle decides 6b on
    what the tests just wrote rather than on what the branch contains. No test may be able
    to do that.

The rule is that no test may run the real build. This module enforces one part of that
rule, and the boundary is stated here exactly so nobody has to re-derive it.

## The mechanism, which is the only thing you should rely on

`install_frontend_build_guard` rebinds exactly two names:

1.  **`subprocess.run`** — the module-level function.
2.  **`subprocess.Popen.__init__`** — the constructor on the **class**, wrapped around
    whatever was bound there at install time.

At either of them the guard resolves a single **program name** from the call's first
positional argument (or its `args=` keyword) — `argv[0]` for a sequence, the first
whitespace-delimited token for a `str`/`bytes` command line, then that value's
`PurePath(...).name` — and raises when the result is in `FRONTEND_TOOLCHAIN_PROGRAMS`.
Nothing else is inspected: not the rest of the argv, not `shell=`, not `cwd=`, not the
environment.

Those two sentences are the whole contract. Work out your own case from them rather than
trusting a list of spellings, because a list of what a tripwire catches is a claim about
every command anyone will ever write and cannot be verified by enumeration.

**Why the second point is the class's `__init__` and not the name `subprocess.Popen`.**
`subprocess.call` builds a `Popen` and `subprocess.check_call` calls `call`, so both reach
the constructor — read in `subprocess.py` on CPython 3.12.13, the interpreter this suite
runs on, and unchanged on 3.14.6. The stdlib has restructured these before, so that is
also asserted at runtime by `tests/unit/test_frontend_build_isolation.py` rather than left
resting on a reading. Binding the *class attribute* instead of the
*module attribute* is what lets the guard compose with a test that replaces
`subprocess.Popen` itself — see below. `subprocess.check_output` calls `run`, so it is
covered by the first point.

## Composing with a test that stubs `subprocess.Popen`

`tests/unit/test_ui_server.py` does `monkeypatch.setattr("subprocess.Popen", mock_popen)`
to catch the Vite dev server `uclone_x.ui.server` co-spawns at `server.py:245-246`. That
rebinds the module attribute; this guard rebinds the class's `__init__`. They are different
attributes, so **neither disarms the other**:

*   the stub still receives the `npm run dev` argv, and nothing real is spawned — a
    `MagicMock` is not the `Popen` class and never runs its `__init__`;
*   the guard is still installed on the class for every other caller, and is still there
    unchanged when `monkeypatch` undoes the stub.

The consequence to be honest about: **while that stub is bound, `call` and `check_call`
reach the stub rather than this guard**, because they look `Popen` up by module-global
name. That is the stub's job for the duration of that test, not a hole in the guard.
`run` and `check_output` are *not* in that set, because `_guarded_run` decides before it
delegates: a toolchain program is refused there whether or not `Popen` is stubbed, and
`run` reaches a bound stub only for a command the guard permits. Measured at this file's
head with the module attribute stubbed and a toolchain argv — `Popen`, `call`,
`check_call`: reached the stub; `run`, `check_output`: refused, stub never called.

`install_frontend_build_guard` wraps **whatever `subprocess.Popen.__init__` is bound to at
the moment it is called**, not at import. `tests/conftest.py` binds its own
`UCLONE_SESSION_DIR`-propagating `__init__` there a few lines earlier, and capturing the
pristine constructor at import would silently drop that propagation for every subprocess in
the suite. `tests/unit/test_session_storage_isolation.py` pins the chain from the other end.
Installing twice is a no-op rather than a second layer — while `subprocess.Popen` is the
real class, which it is at the one place install is called. Installing while a stub is
bound there is not a no-op, and what happens depends on the stub's shape: a `MagicMock`
refuses the assignment with `AttributeError: Attempting to set unsupported magic method
'__init__'`, while a plain-class stub accepts it silently and comes away carrying the
wrapper on its own `__init__`. So do not read the absence of an error as evidence that the
install landed on the real class — against the second shape it did not, and said nothing.
What holds for both shapes, and is the half worth relying on, is that the real class is
left alone: `subprocess.Popen.__init__` is the same object afterwards, so that install
adds no wrapper there. Both shapes measured, not reasoned about.

**A consequence, measured rather than designed:** `asyncio.create_subprocess_exec` and
`asyncio.create_subprocess_shell` are refused for a toolchain program too, because
asyncio's unix transport constructs a `subprocess.Popen`. Nothing here targets them and
nothing should rely on the coupling; a test that legitimately needs an asyncio toolchain
spawn will meet this refusal and should read this note rather than re-derive it.

## The one sanctioned way past it (#1077)

`run_outside_the_live_tree` suspends the refusal for the duration of a single call, and is
the only thing in the repository that does. It exists because the frontend lethality ratchet
runs `npx vitest run`, which is a test runner and not a build, and because the alternative —
reaching the same subprocess through one of the gaps listed below — would be a hole nobody
declared. In exchange it enforces the invariant the program-name proxy stands in for: the
`cwd` must resolve **outside** the live worktree, so the command cannot write a tracked
frontend source or the committed bundle. What it costs is that the lift is process-wide while
that one subprocess runs; the suite's workers run tests one at a time, so nothing else in the
process is in a position to use the window.

## Known gaps — illustrative, not exhaustive

**No completeness is claimed here.** This guard is a tripwire for the accidental case #1067
is about, not a sandbox; anyone determined to run a build can, and these are examples of how,
not the boundary of how.

*   **Either guarded function, whenever the resolved program name is not the toolchain
    program.** These reach a real build through the guard itself:
    -   a shell string whose first token is something else —
        `run("cd frontend && npm run build", shell=True)` resolves to `cd`, and
        `run("NODE_ENV=production npm run build", shell=True)` to `NODE_ENV=production`;
    -   a list argv whose `argv[0]` is an interpreter or shell —
        `run(["sh", "-c", "npm run build"])` resolves to `sh`.
*   **`os.system`, and a direct `node_modules/.bin/vite`** — a bare interpreter or binary
    path is not a package-manager name.
*   **A child interpreter.** A `python -c` subprocess re-imports an unguarded `subprocess`,
    so nothing installed here reaches it. `tests/unit/test_cli_ui_stop.py` is that case and
    suppresses the build by hand; its comment says so.

## Why a tripwire is the right size for #1067

The defect as filed is that unit tests calling the real `start_ui_server` execute a real
`npm run build`. Those call sites reach it through `subprocess.run(["npm", "run", "build"],
...)` at `server.py:89` — a plain list argv whose `argv[0]` is `npm`, which the mechanism
above refuses. The same launcher's `dev=True` branch reaches `npm run dev` through
`subprocess.Popen` at `server.py:245-246` (#1076), which the second interception point
refuses the same way. So the guard turns the filed defect, and the accidental repeat of it,
into a loud failure. It does not make the suite incapable of building, and must not be read
as evidence that the gaps above are handled; they are tracked separately.

A test that calls the real `start_ui_server` takes the `frontend_build_suppressed` fixture;
a test *about* a build command asserts on a captured argv instead of running it, as
`tests/unit/test_frontend_bundle_freshness.py` already does.

**`BaseException`, deliberately.** `_ensure_frontend_built` wraps its `subprocess.run` in
`except Exception:` and prints a warning, and `start_ui_server` wraps its Vite `Popen` in a
second `except Exception:` that does the same. A guard raising an ordinary exception would
therefore be swallowed by the very function it guards, at both call sites: the build would
be blocked and the test would stay green, which is precisely the silence this guard exists
to break. pytest records a `BaseException` that is neither `Exit` nor `KeyboardInterrupt` as
a failure of the test that raised it, and of that test alone.

Only package-manager entry points are named. **`node` is deliberately absent**: Playwright's
Python driver spawns its own `node` binary on every browser test, and that is not a
frontend build.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path, PurePath
from typing import Any, Final, cast

#: Entry points that can drive a frontend build. Matched on the program's **basename**, so
#: an absolute path to one is refused exactly as the bare name is.
FRONTEND_TOOLCHAIN_PROGRAMS: Final[frozenset[str]] = frozenset(
    {"npm", "npx", "pnpm", "yarn", "bun"}
)

_REFUSAL: Final[str] = (
    "A test invoked the Node toolchain: {command!r}. No test in this suite may run "
    "npm/npx/pnpm/yarn/bun — `npm run build` rewrites the committed "
    "`src/uclone_x/ui_static` bundle that gate stage 6b verifies, so the suite would be "
    "grading its own output (#1067). A test that calls the real `start_ui_server` takes "
    "the `frontend_build_suppressed` fixture; a test about a build command asserts on a "
    "captured argv instead of running it. "
    "How this guard works, so you do not over-trust it: it replaces `subprocess.run` and "
    "wraps `subprocess.Popen.__init__` (which is what `call` and `check_call` build on), "
    "and refuses when the resolved program name — argv[0] for a sequence, the first token "
    "for a command string — is npm/npx/pnpm/yarn/bun. Nothing else is inspected, so it is "
    "a tripwire and not a sandbox: `os.system`, a child interpreter, and calls resolving "
    "to another name (`sh -c ...`, `cd ... && npm ...`, `node_modules/.bin/vite`) all "
    "reach a real build unguarded, and while a test has the name `subprocess.Popen` "
    "stubbed, `Popen`/`call`/`check_call` reach that stub rather than this guard. That "
    "list is not exhaustive. "
    "See `tests/support/frontend_build_guard.py`."
)


class RealFrontendBuildInTestError(BaseException):
    """A test reached a real Node toolchain invocation.

    Derived from `BaseException` so that the `except Exception:` inside
    `uclone_x.ui.server._ensure_frontend_built` cannot absorb it — see this module's
    docstring.
    """


def _decoded(value: object) -> str | None:
    """`os.fsdecode` for the spellings `subprocess` accepts, and `None` for anything else."""
    if isinstance(value, (str, bytes, os.PathLike)):
        return os.fsdecode(cast("str | bytes | os.PathLike[str]", value))
    return None


def program_name(command: object) -> str:
    """The basename of the executable a `subprocess` call names, however it was spelled.

    Accepts the three shapes `subprocess.run` accepts: an argv sequence, a `str`/`bytes`
    command line (`shell=True`), and a path-like. Anything it cannot read yields `""`,
    which matches nothing and therefore never refuses a call the guard did not understand.
    """
    whole = _decoded(command)
    if whole is not None:
        words = whole.split()
        first = words[0] if words else ""
    elif isinstance(command, Iterable):
        items = list(cast("Iterable[object]", command))
        first = (_decoded(items[0]) or "") if items else ""
    else:
        first = ""
    return PurePath(first).name


def _commanded(args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
    """The command a `subprocess` call names, however the caller spelled the argument.

    `run`, `call`, `check_call` and `Popen` all take it first positionally or as `args=`.
    """
    return args[0] if args else kwargs.get("args")


#: Depth of the sanctioned lift below. A counter rather than a flag so that a nested lift
#: cannot re-arm the guard on the way out of the inner one.
_lift_depth: int = 0


def _refuse_frontend_toolchain(command: object) -> None:
    """Raise if `command` names a frontend toolchain program; return otherwise.

    The single decision both interception points share, so that neither can drift from the
    other and so that the refusal a `Popen` raises is the one a `run` raises.

    `run_outside_the_live_tree` is the one thing that can suspend this, and it is a named
    exemption rather than a discovered hole — see its docstring for what it costs.
    """
    if _lift_depth == 0 and program_name(command) in FRONTEND_TOOLCHAIN_PROGRAMS:
        raise RealFrontendBuildInTestError(_REFUSAL.format(command=command))


class ToolchainInsideLiveTreeError(BaseException):
    """A sanctioned toolchain run was aimed at the live worktree. Refused.

    `BaseException` for the same reason `RealFrontendBuildInTestError` is: the callers this
    guard protects wrap their subprocess work in `except Exception:`.
    """


def run_outside_the_live_tree(
    command: list[str],
    *,
    cwd: Path,
    live_root: Path,
    **kwargs: Any,
) -> Any:
    """Run a toolchain command in a tree that is **not** the live worktree (#1077).

    The guard above refuses npm/npx/pnpm/yarn/bun by program name, which is a proxy for the
    thing actually forbidden: a test rewriting the committed `src/uclone_x/ui_static` bundle
    that gate stage 6b grades (#1067). The frontend lethality ratchet needs `npx vitest run`,
    which is not a build and writes no bundle — but it is `npx`, so the proxy refuses it.

    Rather than reach a real build through one of the gaps the module docstring lists — `sh
    -c`, a bare `node`, `node_modules/.bin/...` — this is a **named exemption**, and it is
    narrower than the guard it lifts: it enforces the invariant the proxy stands in for,
    instead of the proxy. `cwd` must resolve outside `live_root`, so whatever the command
    writes, it cannot be the committed bundle or a tracked frontend source. A caller that
    points it at the live tree gets `ToolchainInsideLiveTreeError` and no subprocess at all.

    The lift is process-wide for the duration of the call and the suite's workers run their
    tests one at a time, so nothing else in this process can slip a build through the window.
    That is the honest cost, and it is written here rather than left to be discovered.
    """
    global _lift_depth
    resolved_cwd = Path(cwd).resolve()
    resolved_live = Path(live_root).resolve()
    if resolved_cwd == resolved_live or resolved_live in resolved_cwd.parents:
        raise ToolchainInsideLiveTreeError(
            f"refused to run {command!r} in {resolved_cwd}, which is inside the live worktree "
            f"{resolved_live}. The exemption in `run_outside_the_live_tree` exists so that a "
            "mutation harness can run vitest in a private copy of the tree; aimed at the live "
            "tree it would be the very thing `frontend_build_guard` refuses (#1067)."
        )
    _lift_depth += 1
    try:
        return _ORIGINAL_RUN(command, cwd=resolved_cwd, **kwargs)
    finally:
        _lift_depth -= 1


_ORIGINAL_RUN: Final[Callable[..., Any]] = subprocess.run

#: Set on an installed `Popen.__init__` wrapper so a second `install_…` call is a no-op
#: rather than a second layer wrapping the first.
_INSTALLED_MARKER: Final[str] = "_uclone_x_frontend_build_guard"


def _guarded_run(*args: Any, **kwargs: Any) -> Any:
    _refuse_frontend_toolchain(_commanded(args, kwargs))
    return _ORIGINAL_RUN(*args, **kwargs)


def _guarding_popen_init(inner: Callable[..., None]) -> Callable[..., None]:
    """Wrap `inner` — whatever is bound to `Popen.__init__` — with the same refusal.

    `inner` is a parameter rather than a module-level capture on purpose: `tests/conftest.py`
    binds its own `UCLONE_SESSION_DIR`-propagating `__init__` there *after* importing this
    module and *before* calling `install_frontend_build_guard`, so a capture taken at import
    would hold the pristine constructor and drop that propagation when it called through.
    """

    def _guarded_popen_init(self: subprocess.Popen[Any], *args: Any, **kwargs: Any) -> None:
        _refuse_frontend_toolchain(_commanded(args, kwargs))
        inner(self, *args, **kwargs)

    setattr(_guarded_popen_init, _INSTALLED_MARKER, True)
    return _guarded_popen_init


def install_frontend_build_guard() -> None:
    """Point `subprocess.run` and `subprocess.Popen.__init__` at the guard for this process.

    Installed at `tests/conftest.py` **import** time, beside the `subprocess.Popen` and
    `asyncio.create_subprocess_*` guards that are installed the same way and for the same
    reason: module-level and collection-time code runs before any fixture does.

    `Popen` is guarded on the **class's** `__init__` rather than on the module attribute
    `subprocess.Popen`, so that a test which stubs that attribute — as
    `tests/unit/test_ui_server.py` does for the co-spawned Vite server — neither disarms
    this guard nor is disarmed by it. The module docstring says what that costs.
    """
    subprocess.run = _guarded_run  # pyright: ignore[reportAttributeAccessIssue]
    current_init = cast("Callable[..., None]", subprocess.Popen.__init__)
    if not getattr(current_init, _INSTALLED_MARKER, False):
        guarded_init = _guarding_popen_init(current_init)
        subprocess.Popen.__init__ = guarded_init  # pyright: ignore[reportAttributeAccessIssue]
