"""Shared harness for the browser-observable system tests.

One component per concern (R12): starting the UI server, finding a free port, and waiting
for readiness each exist here once. Before this module every E2E file carried its own copy,
three of them under the same fixture name and already diverging in what they passed to
`create_ui_app`, so a fix to any of them reached one file out of four.

Readiness is a polled condition rather than a fixed sleep (R7): a fixed sleep fails on a
loaded machine and passes falsely on a fast one, and the four copies it replaced each slept
half a second whether or not the server was up.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import ExitStack, closing
from pathlib import Path
from time import monotonic
from typing import Any

import pytest
import uvicorn
from playwright.async_api import Locator, Page

from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.ui.app import create_ui_app

# The static bundle is resolved from this file rather than the working directory, so the
# suite behaves the same from any worktree or invocation directory (R10).
STATIC_DIR: Path = Path(__file__).resolve().parents[2] / "src" / "uclone_x" / "ui_static"

_READY_TIMEOUT_SECONDS = 10.0
_READY_POLL_SECONDS = 0.02
_SHUTDOWN_TIMEOUT_SECONDS = 2.0

UIServerFactory = Callable[..., str]


# The dock is named once, and the dock it replaced is named as retired.
#
# The three tests that open it used to resolve it as
# `"[data-testid='artifacts-dock'], [data-testid='internals-dock']"`, a compatibility
# alternation kept while #845 replaced `InternalsDock` with `ArtifactsDock`. An alternation
# cannot say which dock it found, so what it actually asserted was "at least one dock-shaped
# element exists".
#
# That is a real blind spot, but it is a *latent* one: it did not cause the false green, and
# the tree at each commit says so.
#
#   * `833a09e` resolved the dock with a **single** selector, `internals-dock` — the retired
#     one. The bundle mounted both docks, so that stale selector matched, and the suite
#     reported 3 passed while the surface was doubled. The alternation did not exist yet.
#   * `105d0ef` (#867) introduced the alternation, two commits after that green. Against the
#     same doubled bundle it raised `strict mode violation: ... resolved to 2 elements`, and
#     kept raising it through `866c801`..`4121db6`. Where it met the doubled surface it
#     errored; it never passed there.
#
# So the false green was a stale single selector still aimed at a retired dock that was still
# mounted. What the alternation contributed was to turn that same defect into a Playwright
# error nobody had written as an assertion. Its own blind spot stayed unexercised — it would
# be equally green if the two docks swapped back, matching the retired half and reporting
# nothing — which is exactly why it is worth closing before it is exercised.
#
# `InternalsDock.tsx` was deleted outright by #1055, along with the alias that re-exported
# `ArtifactsDock` under its name — so the retired selector below should now find nothing in
# any bundle. It stays as a regression guard rather than a live tree-shaking concern: "there
# is exactly one dock, and it is the one #845 shipped" is stated here instead of inherited.
_DOCK_SELECTOR: str = "[data-testid='artifacts-dock']"
_RETIRED_DOCK_SELECTOR: str = "[data-testid='internals-dock']"


async def dock_locator(page: Page) -> Locator:
    """Return the dock #845 shipped, refusing a page that still mounts the retired one.

    Await this **after** whatever makes the dock appear. The retired-dock count is taken
    eagerly, at the moment this coroutine is awaited, and the dock mounts on toggle: awaited
    beforehand, the count runs against a closed dock, reads zero of both docks, and passes
    without having looked at the state it exists to reject.
    """
    retired = await page.locator(_RETIRED_DOCK_SELECTOR).count()
    assert retired == 0, (
        f"{retired} element(s) still carry {_RETIRED_DOCK_SELECTOR}; the dock #845 retired "
        "is mounted alongside the one that replaced it"
    )
    return page.locator(_DOCK_SELECTOR)


#: Two animation frames. Layout the browser has been asked to recompute is readable after
#: them; a fixed sleep would be a guess that fails on a loaded machine and passes falsely on
#: a fast one (R7).
TWO_FRAMES = "() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))"

#: What a surface holds past its own edges: its scroll overflow, and every control whose box is
#: not inside the surface's. A control inside a region that scrolls on its own (a wide table in an
#: `overflow-x-auto` wrapper) is that region's business and is not reported. Called with no
#: argument it measures the dock's surface; given a selector, that element instead.
SURFACE_OVERFLOW = """(selector) => {
    const which = selector || "[data-testid='dock-surface']";
    const surface = document.querySelector(which);
    if (!surface) return [`${which}: not rendered`];
    const s = surface.getBoundingClientRect();
    const problems = [];
    if (surface.scrollWidth > surface.clientWidth + 1) {
        problems.push(`the surface holds ${surface.scrollWidth}px in ${surface.clientWidth}px`);
    }
    const scrollsOnItsOwn = (el) => {
        for (let n = el.parentElement; n && n !== surface; n = n.parentElement) {
            if (['auto', 'scroll'].includes(getComputedStyle(n).overflowX)) return true;
        }
        return false;
    };
    for (const el of surface.querySelectorAll('button, select, input')) {
        const b = el.getBoundingClientRect();
        if (b.width < 1 || b.height < 1 || scrollsOnItsOwn(el)) continue;
        if (b.left < s.left - 1 || b.right > s.left + surface.clientWidth + 1) {
            const name = (el.getAttribute('aria-label') || el.textContent
                || el.getAttribute('placeholder') || el.tagName).trim().slice(0, 30);
            problems.push(`${el.tagName.toLowerCase()} "${name}" runs x ${b.left.toFixed(0)}-`
                + `${b.right.toFixed(0)} in a surface x ${s.left.toFixed(0)}-`
                + `${(s.left + surface.clientWidth).toFixed(0)}`);
        }
    }
    return problems;
}"""


async def set_the_rail(page: Page, rail: str) -> None:
    """Leave the rail `"open"` or `"closed"`, whichever it started as, once the page has loaded.

    The rail's first state follows the window since #1062: closed below 600px, open from 600px.
    A test parametrised over the rail's state therefore cannot assume it starts open and close
    it; it names the state it wants. Clicking only when the rail is in the other state keeps a
    test that wants the default from toggling it at all.

    It returns once the rail has actually entered or left the tree, not on the click: the rail
    is mounted and unmounted by a state change, so a measurement taken on the click alone can
    read the layout that still had it. It replaces `close_the_rail`, whose one click opened the
    rail at the widths where it now starts closed.
    """
    assert rail in ("open", "closed"), rail
    toggle = page.locator("[data-testid='toggle-sidebar']")
    await toggle.wait_for(state="visible")
    is_open = await page.locator("[data-testid='chat-sidebar']").count() > 0
    if (rail == "open") == is_open:
        return
    await toggle.click()
    await page.locator("[data-testid='chat-sidebar']").wait_for(
        state="visible" if rail == "open" else "detached"
    )


async def turn_on_developer_mode(page: Page) -> None:
    """Switch developer mode on the way a user does: Settings, the switch, close.

    Developer mode is off by default (owner ruling 2026-09-22), and the dock's developer
    instruments -- Knowledge Graph, DAG, EventBus, Ontology -- are in a drawer that exists
    only while it is on. A test about one of them starts here. It returns once the switch
    reads on and the dialog is gone, so the next click lands on the page and not on the
    modal's backdrop. ACP and Evals are not dock surfaces any more (#1358): they are Settings'
    Diagnostics section, and `open_diagnostics` is the way to them.
    """
    await _switch_developer_mode_on_in_settings(page)
    await page.locator("[role='dialog'] button[title='Close']").click()
    await page.locator("[role='dialog']").wait_for(state="detached")


async def open_diagnostics(page: Page) -> Locator:
    """Open Settings with developer mode on and return its Diagnostics section (#1358).

    The ACP report and the Evals scorecard live there, below the developer-mode switch, and
    only while that switch is on. The dialog is left open: the section is inside it.
    """
    await _switch_developer_mode_on_in_settings(page)
    section = page.locator("[data-testid='settings-diagnostics']")
    await section.wait_for(state="visible")
    return section


async def _switch_developer_mode_on_in_settings(page: Page) -> None:
    """Open Settings and leave its developer-mode switch on, the way a user does."""
    await page.locator("[data-testid='open-settings']").click()
    toggle = page.locator("[data-testid='developer-mode-switch']")
    await toggle.wait_for(state="visible")
    if await toggle.get_attribute("aria-checked") != "true":
        await toggle.click()
    # Waited for rather than read once: the switch re-renders after the click's state update.
    await page.wait_for_selector(
        "[data-testid='developer-mode-switch'][aria-checked='true']", timeout=5000
    )


def _free_port() -> int:
    """Return a port the OS has just confirmed is free.

    Ephemeral rather than fixed, so parallel worktrees on one machine do not collide (R10).
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


def _await_accepting(port: int, *, timeout_seconds: float = _READY_TIMEOUT_SECONDS) -> None:
    """Block until `port` accepts a connection, or fail loudly at the deadline.

    Failing loudly matters more than it looks: a server that never came up would otherwise
    surface as an unrelated Playwright navigation error several frames away from the cause
    (P6).
    """
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
            probe.settimeout(_READY_POLL_SECONDS)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
    raise RuntimeError(
        f"UI test server did not accept connections on port {port} within {timeout_seconds:.1f}s"
    )


# `running_ui` and `mock_llm` are public because three modules outside this one use them:
# a fixture that is one module's input (R12) still starts its server through the single
# implementation here rather than growing a fourth copy. They were named with a leading
# underscore while `conftest.py` was their only caller, and the first module to import one
# silenced the resulting strict-mode error with a file-wide `reportPrivateUsage=false` --
# which disabled the check for everything else in that file too. The name is what was wrong.


@contextlib.contextmanager
def running_ui(
    *,
    storage_dir: Path | None,
    llm: MockLLMConnector | None,
    budget_tracker: TokenBudgetManager | None = None,
    workspace_dir: Path | None = None,
    eval_reports_dir: Path | None = None,
    configure: Callable[[Any], None] | None = None,
) -> Generator[str]:
    """Run the UI app on an ephemeral port for the duration of the context.

    `workspace_dir` defaults to the app's own default, the process's cwd; a test that writes
    into the workspace (a persona file, #892) passes its own.

    So does `eval_reports_dir`, and that default read the developer's own `evals/reports/`,
    which is not committed. The Evaluations surface therefore rendered whatever that machine
    happened to have run, and on a machine with none of it -- CI, a fresh clone, a worktree --
    it rendered no suites at all. A surface that overflowed its dock on one probe name did so
    for months without a single red: `test_no_dock_surface_holds_a_control_past_its_edge`
    walked Evaluations and found nothing to measure. Pass a directory the test wrote.

    Each server gets its own `EventBus`. Left to default, `create_ui_app` hands every app
    the process-wide `get_ui_event_bus()`, which is right for the one server a real process
    runs and wrong for this one, which starts a server per test, each on its own thread and
    event loop. The bus's `asyncio.PriorityQueue` binds to the first loop that waits on it,
    so from the second server on every dispatcher task drained what was queued and then
    died with `RuntimeError: <PriorityQueue ...> is bound to a different event loop` --
    logged in every E2E run (#1415). Delivery survived only because each publish starts a
    fresh dispatcher. Nothing a test asserts about one server should depend on the servers
    before it.
    """
    app = create_ui_app(
        static_dir=STATIC_DIR,
        bus=EventBus(),
        storage_dir=storage_dir,
        llm=llm,
        budget_tracker=budget_tracker,
        workspace_dir=workspace_dir,
        eval_reports_dir=eval_reports_dir,
    )
    # A hook rather than another keyword per setting: what a test needs to arrange is
    # sometimes a field on a Core object rather than an argument `create_ui_app` takes,
    # and the alternative is a test that builds and serves its own app beside this one.
    if configure is not None:
        configure(app)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _await_accepting(port)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)


class _E2EMockLLMConnector(MockLLMConnector):
    """Mock connector that returns a valid silence decision for unscripted selector requests."""

    async def generate(self, request: Any) -> Any:
        from uclone_x.llm.models import MessageRole

        if (
            not self._responses
            and getattr(request, "messages", None)
            and any(
                "allocate the floor in a multi-agent chat room" in (getattr(m, "content", "") or "")
                for m in request.messages
                if getattr(m, "role", None) is MessageRole.SYSTEM
            )
        ):
            self._responses.append(
                '{"speaker_id": null, "confidence": 1.0, "reasoning": "mock selector default"}'
            )
        return await super().generate(request)


def mock_llm(*, streaming_chunk_delay: float = 0.0) -> MockLLMConnector:
    return _E2EMockLLMConnector(
        default_model="mock-gpt-4o",
        default_response="Mock response from UClone-X BaseAgent.",
        streaming_chunk_delay=streaming_chunk_delay,
    )


#: The longest probe name the seeded report carries: 43 characters, no break opportunity.
LONGEST_SEEDED_PROBE_NAME = "carry_over::evidence_reached_the_second_turn"

#: One evaluation report, so the Evaluations surface has suites to draw on every machine.
#:
#: The probe names are the shape this project's own reports carry -- a namespace, two colons
#: and an underscored sentence -- and the longest is an unbreakable 43-character token. That
#: is what a dock 519px wide has to fit into a suite card 240px wide, and what nothing checked
#: while this directory was the developer's own (see `running_ui`).
_SEEDED_EVAL_REPORT: dict[str, object] = {
    "timestamp": "2026-09-20T12:00:00+00:00",
    "suite": "carry_over",
    "model": None,
    "provider": None,
    "summary": {
        "total_probes": 3,
        "passed_probes": 2,
        "failed_probes": 1,
        "pass_rate": 2 / 3,
        "duration_s": 1.5,
        "p50_latency_s": 0.4,
        "p95_latency_s": None,
        "worst_latency_s": 0.9,
    },
    "probes": [
        {
            "name": LONGEST_SEEDED_PROBE_NAME,
            "passed": True,
            "duration_s": 0.4,
            "latency_s": 0.4,
            "message": "[CORRECT] the second turn cited the first turn's evidence",
        },
        {
            "name": "carry_over::the_second_turn_did_not_refetch",
            "passed": True,
            "duration_s": 0.3,
            "latency_s": 0.3,
            "message": "[CORRECT] no tool call repeated the first turn's fetch",
        },
        {
            "name": "carry_over::stop_reasons_are_declared",
            "passed": False,
            "duration_s": 0.9,
            "latency_s": 0.9,
            "message": "[WRONG] a turn ended with no stop reason",
        },
    ],
}


def seed_eval_reports(directory: Path) -> Path:
    """Write `_SEEDED_EVAL_REPORT` into `directory` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "20260920-120000-carry_over.json").write_text(
        json.dumps(_SEEDED_EVAL_REPORT), encoding="utf-8"
    )
    return directory


@pytest.fixture(scope="module")
def ui_test_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A running UI server with isolated session storage and a scripted model.

    The model is substituted at the connector because no test served by this fixture makes a
    claim about the model itself (R1); session storage and the evaluation reports are
    per-module temporary state so nothing is read that the suite did not create (R10).
    """
    with running_ui(
        storage_dir=tmp_path_factory.mktemp("ui_sessions"),
        llm=mock_llm(),
        eval_reports_dir=seed_eval_reports(tmp_path_factory.mktemp("ui_eval_reports")),
    ) as url:
        yield url


@pytest.fixture
def fresh_ui_server(tmp_path: Path) -> Iterator[str]:
    """`ui_test_server` for one test alone: its own server over its own session storage.

    For a test that clears or seeds the default conversation, which on the module's shared
    server would decide what every later test in the module starts from.
    """
    with running_ui(storage_dir=tmp_path, llm=mock_llm()) as url:
        yield url


@pytest.fixture(scope="module")
def streaming_ui_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """`ui_test_server`, with the scripted model pausing between the words it streams.

    With no pause every delta of a reply reaches the browser inside one frame, so a live
    bubble that *replaces* its text on each delta and one that *appends* render the same
    last state -- and a browser test cannot tell a working stream from a broken one. The
    pause is what makes the intermediate states exist to be observed. A separate server
    rather than a slower `ui_test_server`, so no other test pays for it.
    """
    with running_ui(
        storage_dir=tmp_path_factory.mktemp("ui_sessions_streaming"),
        llm=mock_llm(streaming_chunk_delay=0.15),
    ) as url:
        yield url


@pytest.fixture
def scripted_reply_ui_server(tmp_path: Path) -> Iterator[UIServerFactory]:
    """Start a UI server whose model answers every turn with the caller's own text.

    For a layout case, which needs a *particular* reply on screen -- a 5,000-character
    unbroken token, a table wider than its column -- rather than any reply at all.

    The playground suite this serves the port of got that content by intercepting
    `/api/chat/stream` (retired with that suite, #1208) with `page.route`, which the room
    suites deliberately do not do: a
    test that answers the head's request itself renders a payload the test wrote, and the
    blank screen `test_room_chat_e2e` was opened for was a real response the fixtures'
    shape disagreed with. Scripting the *model* keeps the whole path real -- route, service,
    transcript, render -- and still decides the words.

    `model` is the name the connector reports having served the turn, which the surface prints
    as attribution. It is the test's input for the same reason the reply is: a picker's natural
    width is its longest option's, and a served-by line's is the model's own name, so a case
    about a narrow column needs a name as long as a locally pulled model's rather than the
    short default.

    `streaming_chunk_delay` is the pause between the reply's streamed words, for a case about
    what the page does *while* a reply is arriving (#1380). At the default of none the whole
    reply is delivered before a browser can act on it, and a case that means to scroll
    mid-stream would be scrolling a finished transcript.

    A per-test server rather than a module-scoped one: the reply is the test's input, so two
    tests wanting different text cannot share a connector.
    """

    with ExitStack() as stack:

        def _start(
            reply: str, *, model: str = "mock-gpt-4o", streaming_chunk_delay: float = 0.0
        ) -> str:
            llm = _E2EMockLLMConnector(
                default_model=model,
                default_response=reply,
                streaming_chunk_delay=streaming_chunk_delay,
            )
            return stack.enter_context(running_ui(storage_dir=tmp_path, llm=llm))

        yield _start
