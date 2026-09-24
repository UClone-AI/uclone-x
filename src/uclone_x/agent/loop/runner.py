"""Runner executing individual loop ticks with watchdog timeout and concurrency lock (FR-Loop, P4, P6)."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.loop.models import LoopJob, LoopStatus, LoopTickResult

logger = logging.getLogger(__name__)


async def execute_loop_tick(
    agent: BaseAgent,
    job: LoopJob,
    tick_index: int,
) -> LoopTickResult:
    """Execute a single tick of a recurring loop job.

    Key guarantees:
        1. Concurrency Guard (uclone2 PID lock pattern): If the job is already
           running, the tick is skipped immediately without overlapping.
        2. Watchdog Timeout: Execution is capped by `job.timeout_seconds` using
           `asyncio.wait_for`.
        3. Budget & Exit Conditions: Automatically updates run counts, consecutive
           failures, and checks `until_pattern` or `max_runs` for completion.
    """
    now = datetime.now(UTC)

    # 1. Concurrency check (skip tick if prior execution is still running)
    if job.lock.locked():
        logger.warning(
            "Loop job %s tick %d skipped: previous execution is still running.",
            job.job_id,
            tick_index,
        )
        return LoopTickResult(
            tick_index=tick_index,
            started_at=now,
            finished_at=now,
            duration_seconds=0.0,
            success=False,
            skipped=True,
            content="Skipped: Previous tick still in progress (concurrency lock).",
        )

    async with job.lock:
        job.is_running = True
        started_at = datetime.now(UTC)
        error_msg: str | None = None
        content = ""
        success = False
        stop_reason: str | None = None

        try:
            # Clean context if requested (uclone2 headless poller mode)
            if job.clean_context:
                agent.reset_session()

            # Execute with watchdog timeout
            turn_result = await asyncio.wait_for(
                agent.execute_turn(job.prompt),
                timeout=job.timeout_seconds,
            )

            if turn_result.error is not None:
                error_msg = turn_result.error
                stop_reason = turn_result.stop_reason or "error"
            else:
                success = True
                content = turn_result.content or ""
                stop_reason = turn_result.stop_reason or "completed"

        except TimeoutError:
            error_msg = f"Turn execution timed out after {job.timeout_seconds:.1f}s (watchdog)."
            stop_reason = "timeout"
            logger.error("Loop job %s tick %d timed out.", job.job_id, tick_index)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            stop_reason = "exception"
            logger.exception(
                "Loop job %s tick %d failed with unhandled error.", job.job_id, tick_index
            )
        finally:
            finished_at = datetime.now(UTC)
            job.is_running = False

        duration = (finished_at - started_at).total_seconds()
        job.runs_count += 1
        job.last_run_at = finished_at

        # Consecutive failure accounting (uclone2 max_attempts pattern)
        if success:
            job.consecutive_failures = 0
        else:
            job.consecutive_failures += 1

        # Check exit conditions
        if job.until_pattern and content and re.search(job.until_pattern, content):
            job.status = LoopStatus.COMPLETED
            logger.info("Loop job %s met exit pattern '%s'.", job.job_id, job.until_pattern)
        elif job.max_runs is not None and job.runs_count >= job.max_runs:
            job.status = LoopStatus.COMPLETED
            logger.info("Loop job %s reached max_runs cap (%d).", job.job_id, job.max_runs)
        elif job.consecutive_failures >= job.max_consecutive_failures:
            job.status = LoopStatus.FAILED
            logger.error(
                "Loop job %s failed: reached max consecutive failures (%d).",
                job.job_id,
                job.max_consecutive_failures,
            )

        tick_result = LoopTickResult(
            tick_index=tick_index,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            success=success,
            content=content,
            error=error_msg,
            skipped=False,
            stop_reason=stop_reason,
        )

        job.last_result = tick_result
        job.history.append(tick_result)
        return tick_result
