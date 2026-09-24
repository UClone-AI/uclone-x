"""Unit tests for LoopScheduler, runner, and concurrency protection (FR-Loop, P4, P6)."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.loop import (
    LoopJob,
    LoopScheduler,
    LoopStatus,
    execute_loop_tick,
)
from uclone_x.agent.models import TurnResult


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock(spec=BaseAgent)
    agent.execute_turn = AsyncMock(
        return_value=TurnResult(
            turn_index=1,
            content="Task completed successfully",
            stop_reason="model_stopped",
            provenance=None,
            error=None,
        )
    )
    agent.reset_session = MagicMock()
    return agent


@pytest.mark.asyncio
async def test_execute_loop_tick_success(mock_agent: MagicMock) -> None:
    job = LoopJob(
        job_id="test-job-1",
        interval_seconds=1.0,
        prompt="Test prompt",
        max_runs=5,
    )

    result = await execute_loop_tick(mock_agent, job, tick_index=1)

    assert result.success is True
    assert result.skipped is False
    assert result.content == "Task completed successfully"
    assert job.runs_count == 1
    assert job.consecutive_failures == 0
    assert job.status == LoopStatus.ACTIVE
    mock_agent.execute_turn.assert_awaited_once_with("Test prompt")


@pytest.mark.asyncio
async def test_execute_loop_tick_concurrency_lock_skips(mock_agent: MagicMock) -> None:
    """uclone2 PID lock test: Overlapping tick is skipped if lock is held."""
    job = LoopJob(
        job_id="test-job-lock",
        interval_seconds=1.0,
        prompt="Slow prompt",
    )

    # Acquire lock manually to simulate an active long-running turn
    await job.lock.acquire()
    try:
        result = await execute_loop_tick(mock_agent, job, tick_index=2)
        assert result.skipped is True
        assert result.success is False
        assert "concurrency lock" in result.content
        assert job.runs_count == 0  # Not counted as a finished run
    finally:
        job.lock.release()


@pytest.mark.asyncio
async def test_execute_loop_tick_watchdog_timeout(mock_agent: MagicMock) -> None:
    """Watchdog timeout test: Long-running turn is cancelled and flagged."""

    async def _hang(_prompt: str) -> TurnResult:
        await asyncio.sleep(5.0)
        return TurnResult(
            turn_index=1, content="never", stop_reason="model_stopped", provenance=None
        )

    mock_agent.execute_turn = AsyncMock(side_effect=_hang)

    job = LoopJob(
        job_id="test-job-timeout",
        interval_seconds=1.0,
        prompt="Hanging prompt",
        timeout_seconds=0.05,  # Very short watchdog
    )

    result = await execute_loop_tick(mock_agent, job, tick_index=1)

    assert result.success is False
    assert result.stop_reason == "timeout"
    assert "timed out" in (result.error or "")
    assert job.consecutive_failures == 1


@pytest.mark.asyncio
async def test_execute_loop_tick_until_pattern_exit(mock_agent: MagicMock) -> None:
    """Exit condition test: Loop terminates when until_pattern matches."""
    mock_agent.execute_turn = AsyncMock(
        return_value=TurnResult(
            turn_index=1,
            content="Server status: ALL_SYSTEMS_GO and green",
            stop_reason="model_stopped",
            provenance=None,
        )
    )

    job = LoopJob(
        job_id="test-job-until",
        interval_seconds=1.0,
        prompt="Check health",
        until_pattern="ALL_SYSTEMS_GO",
    )

    await execute_loop_tick(mock_agent, job, tick_index=1)
    assert job.status == LoopStatus.COMPLETED


@pytest.mark.asyncio
async def test_execute_loop_tick_max_runs_exit(mock_agent: MagicMock) -> None:
    job = LoopJob(
        job_id="test-job-max",
        interval_seconds=1.0,
        prompt="Limited run",
        max_runs=2,
    )

    await execute_loop_tick(mock_agent, job, tick_index=1)
    assert job.status == LoopStatus.ACTIVE

    await execute_loop_tick(mock_agent, job, tick_index=2)
    assert job.status == LoopStatus.COMPLETED


@pytest.mark.asyncio
async def test_execute_loop_tick_consecutive_failures_abort(mock_agent: MagicMock) -> None:
    mock_agent.execute_turn = AsyncMock(
        return_value=TurnResult(
            turn_index=1,
            content="",
            error="Connection refused",
            stop_reason="not_started",
            provenance=None,
        )
    )

    job = LoopJob(
        job_id="test-job-failures",
        interval_seconds=1.0,
        prompt="Failing run",
        max_consecutive_failures=2,
    )

    await execute_loop_tick(mock_agent, job, tick_index=1)
    assert job.status == LoopStatus.ACTIVE
    assert job.consecutive_failures == 1

    await execute_loop_tick(mock_agent, job, tick_index=2)
    assert job.status == LoopStatus.FAILED
    assert job.consecutive_failures == 2


@pytest.mark.asyncio
async def test_loop_scheduler_lifecycle(mock_agent: MagicMock) -> None:
    completed_ticks: list[tuple[str, int]] = []

    def _callback(job: LoopJob, res: Any) -> None:
        completed_ticks.append((job.job_id, res.tick_index))

    scheduler = LoopScheduler(agent=mock_agent, on_tick_completed=_callback)

    job = scheduler.add_job(
        interval_seconds=0.05,
        prompt="Tick prompt",
        max_runs=2,
        run_immediately=True,
    )

    assert scheduler.get_job(job.job_id) is job
    assert len(scheduler.list_jobs()) == 1

    await scheduler.wait_job(job.job_id)
    assert job.status == LoopStatus.COMPLETED
    assert job.runs_count == 2
    assert len(completed_ticks) == 2

    scheduler.cancel_all()
