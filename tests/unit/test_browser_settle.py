"""When a page counts as settled: its load event, then the network quiet (design §3.2)."""

from __future__ import annotations

import asyncio

from uclone_x.browser.cdp import CdpEvent
from uclone_x.browser.service import settle


def _event(method: str, request: str | None = None) -> CdpEvent:
    return CdpEvent(method, {"requestId": request} if request else {}, "S1")


async def test_it_waits_for_a_request_in_flight_to_finish() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    events.put_nowait(_event("Page.loadEventFired"))
    events.put_nowait(_event("Network.requestWillBeSent", "r1"))
    waiting = asyncio.create_task(settle(events, expect_load=True, idle_cap=5, quiet=0.05))

    await asyncio.sleep(0.3)
    assert not waiting.done()

    events.put_nowait(_event("Network.loadingFinished", "r1"))
    await asyncio.wait_for(waiting, 1)


async def test_it_waits_for_the_load_event_before_the_network() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    waiting = asyncio.create_task(settle(events, expect_load=True, quiet=0.01))

    await asyncio.sleep(0.2)
    assert not waiting.done()

    events.put_nowait(_event("Page.loadEventFired"))
    await asyncio.wait_for(waiting, 1)


async def test_a_page_that_never_goes_quiet_is_observed_at_the_cap() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    events.put_nowait(_event("Network.requestWillBeSent", "poll"))

    await asyncio.wait_for(settle(events, expect_load=False, idle_cap=0.1, quiet=0.01), 1)


async def test_a_page_that_never_loads_is_observed_after_the_load_timeout() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()

    await asyncio.wait_for(
        settle(events, expect_load=True, load_timeout=0.05, idle_cap=0.1, quiet=0.01), 1
    )


# -- after an action (step 2): what the page did while it settled -------------------------


def _with(method: str, **params: object) -> CdpEvent:
    return CdpEvent(method, dict(params), "S1")


async def test_a_dialog_ends_the_wait_at_once_even_while_the_action_is_blocked() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    blocked: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    events.put_nowait(_with("Page.javascriptDialogOpening", type="alert", message="hi"))

    seen = await asyncio.wait_for(settle(events, expect_load=False, busy=blocked, quiet=5), 1)

    assert seen.dialog == {"type": "alert", "message": "hi"}
    assert not blocked.done()


async def test_the_quiet_period_starts_only_once_the_action_returned() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    busy: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    waiting = asyncio.create_task(settle(events, expect_load=False, busy=busy, quiet=0.05))

    await asyncio.sleep(0.3)
    assert not waiting.done()

    busy.set_result(None)
    await asyncio.wait_for(waiting, 1)


async def test_a_main_frame_navigation_waits_for_its_load_event() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    events.put_nowait(_with("Page.frameStartedLoading", frameId="child"))
    waiting = asyncio.create_task(settle(events, expect_load=False, main_frame="MAIN", quiet=0.05))
    await asyncio.sleep(0.02)
    events.put_nowait(_with("Page.frameStartedLoading", frameId="MAIN"))

    await asyncio.sleep(0.3)
    assert not waiting.done()

    events.put_nowait(_with("Page.loadEventFired"))
    seen = await asyncio.wait_for(waiting, 1)
    assert seen.navigated


async def test_a_page_restored_from_the_back_forward_cache_counts_as_loaded() -> None:
    # Going back to a cached page fires no load event, only the frame's stop-loading.
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    events.put_nowait(_with("Page.frameStartedLoading", frameId="MAIN"))
    events.put_nowait(_with("Page.frameStoppedLoading", frameId="child"))
    waiting = asyncio.create_task(settle(events, expect_load=False, main_frame="MAIN", quiet=0.05))

    await asyncio.sleep(0.3)
    assert not waiting.done()

    events.put_nowait(_with("Page.frameStoppedLoading", frameId="MAIN"))
    seen = await asyncio.wait_for(waiting, 1)
    assert seen.navigated


async def test_a_frame_inside_the_page_loading_is_not_a_navigation() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    events.put_nowait(_with("Page.frameStartedLoading", frameId="child"))

    seen = await asyncio.wait_for(
        settle(events, expect_load=False, main_frame="MAIN", quiet=0.05), 1
    )

    assert not seen.navigated


async def test_a_download_is_waited_for_and_its_file_reported() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    events.put_nowait(_with("Browser.downloadWillBegin", guid="g1", suggestedFilename="a.csv"))
    waiting = asyncio.create_task(settle(events, expect_load=False, quiet=0.05))

    await asyncio.sleep(0.3)
    assert not waiting.done()

    events.put_nowait(
        _with("Browser.downloadProgress", guid="g1", state="completed", filePath="/d/a.csv")
    )
    seen = await asyncio.wait_for(waiting, 1)
    assert seen.downloads == {"g1": {"name": "a.csv", "state": "completed", "path": "/d/a.csv"}}


async def test_a_download_that_outlasts_the_cap_is_reported_as_in_progress() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    events.put_nowait(_with("Browser.downloadWillBegin", guid="g1", suggestedFilename="big.iso"))

    seen = await asyncio.wait_for(
        settle(events, expect_load=False, quiet=0.01, idle_cap=0.05, download_wait=0.1), 2
    )

    assert seen.downloads["g1"]["state"] == "in progress"


async def test_a_page_opening_a_window_is_reported() -> None:
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    events.put_nowait(_with("Page.windowOpen", url="http://127.0.0.1/b.html"))

    seen = await asyncio.wait_for(settle(events, expect_load=False, quiet=0.01), 1)

    assert seen.windows == 1


async def test_quiet_period_restarts_on_page_load_event() -> None:
    """When a page takes longer than quiet to load, the quiet period must start from the
    load event, not from the wait start (#2117).

    Killed by: src/uclone_x/browser/service.py :: quiet_since = time_fn()  # loadEventFired
    Becomes: pass
    """
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    simulated_time = 100.0

    def clock() -> float:
        return simulated_time

    waiting = asyncio.create_task(settle(events, expect_load=True, quiet=0.05, clock=clock))
    await asyncio.sleep(0)

    # 1.0s passes before Page.loadEventFired arrives (> quiet of 0.05s)
    simulated_time = 101.0
    events.put_nowait(_event("Page.loadEventFired"))
    await asyncio.sleep(0.01)

    # Quiet period must restart from loadEventFired, so waiting is not done yet
    assert not waiting.done()

    # Once simulated time advances past quiet, settle completes
    simulated_time = 101.1
    await asyncio.wait_for(waiting, 1)


async def test_quiet_period_restarts_on_frame_stopped_loading() -> None:
    """When a frame stops loading after wait start, the quiet period must start from the
    frame-stop event, not from the wait start (#2117).

    Killed by: src/uclone_x/browser/service.py :: quiet_since = time_fn()  # frameStoppedLoading
    Becomes: pass
    """
    events: asyncio.Queue[CdpEvent] = asyncio.Queue()
    simulated_time = 100.0

    def clock() -> float:
        return simulated_time

    waiting = asyncio.create_task(
        settle(events, expect_load=False, main_frame="MAIN", quiet=0.05, clock=clock)
    )
    await asyncio.sleep(0)

    # 1.0s passes before Page.frameStoppedLoading arrives (> quiet of 0.05s)
    simulated_time = 101.0
    events.put_nowait(_with("Page.frameStoppedLoading", frameId="child"))
    await asyncio.sleep(0.01)

    # Quiet period must restart from frameStoppedLoading, so waiting is not done yet
    assert not waiting.done()

    # Once simulated time advances past quiet, settle completes
    simulated_time = 101.1
    await asyncio.wait_for(waiting, 1)
