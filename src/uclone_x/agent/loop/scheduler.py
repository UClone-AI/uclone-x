"""Scheduler managing lifecycle of concurrent recurring loop jobs (FR-Loop, P4, P6)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.loop.models import LoopJob, LoopStatus, LoopTickResult
from uclone_x.agent.loop.runner import execute_loop_tick

logger = logging.getLogger(__name__)


class LoopScheduler:
    """Asyncio-based scheduler for managing recurring loop jobs on a BaseAgent."""

    def __init__(
        self,
        agent: BaseAgent,
        on_tick_completed: Callable[[LoopJob, LoopTickResult], None] | None = None,
    ) -> None:
        self.agent = agent
        self.on_tick_completed = on_tick_completed
        self._jobs: dict[str, LoopJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def add_job(
        self,
        interval_seconds: float,
        prompt: str,
        job_id: str | None = None,
        max_runs: int | None = None,
        timeout_seconds: float = 600.0,
        clean_context: bool = False,
        until_pattern: str | None = None,
        max_consecutive_failures: int = 3,
        run_immediately: bool = False,
    ) -> LoopJob:
        """Register a new recurring loop job and spawn its background task."""
        effective_id = job_id or f"loop-{uuid.uuid4().hex[:6]}"
        if effective_id in self._jobs and self._jobs[effective_id].status == LoopStatus.ACTIVE:
            raise ValueError(f"Loop job '{effective_id}' is already active.")

        job = LoopJob(
            job_id=effective_id,
            interval_seconds=interval_seconds,
            prompt=prompt,
            max_runs=max_runs,
            timeout_seconds=timeout_seconds,
            clean_context=clean_context,
            until_pattern=until_pattern,
            max_consecutive_failures=max_consecutive_failures,
            next_run_at=(
                datetime.now(UTC)
                if run_immediately
                else datetime.now(UTC) + timedelta(seconds=interval_seconds)
            ),
        )

        self._jobs[effective_id] = job
        task = asyncio.create_task(
            self._run_job_loop(job, run_immediately=run_immediately),
            name=f"ucx-loop-{effective_id}",
        )
        self._tasks[effective_id] = task
        return job

    def get_job(self, job_id: str) -> LoopJob | None:
        """Retrieve a loop job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[LoopJob]:
        """List all tracked loop jobs."""
        return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel an active loop job."""
        job = self._jobs.get(job_id)
        if not job or job.status != LoopStatus.ACTIVE:
            return False

        job.status = LoopStatus.CANCELLED
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        return True

    def cancel_all(self) -> int:
        """Cancel all currently active loop jobs."""
        cancelled_count = 0
        for job_id in list(self._jobs.keys()):
            if self.cancel_job(job_id):
                cancelled_count += 1
        return cancelled_count

    async def wait_job(self, job_id: str) -> LoopJob:
        """Wait for a specific loop job to complete or terminate."""
        task = self._tasks.get(job_id)
        if task:
            try:
                await task
            except asyncio.CancelledError:
                pass
        job = self._jobs[job_id]
        return job

    async def _run_job_loop(self, job: LoopJob, run_immediately: bool = False) -> None:
        """Background coroutine executing the recurring loop for a job."""
        tick_index = 0
        try:
            if not run_immediately:
                await asyncio.sleep(job.interval_seconds)

            while job.status == LoopStatus.ACTIVE:
                tick_index += 1
                result = await execute_loop_tick(self.agent, job, tick_index)

                if self.on_tick_completed:
                    try:
                        self.on_tick_completed(job, result)
                    except Exception as cb_exc:
                        logger.warning("Error in on_tick_completed callback: %s", cb_exc)

                # Check if job completed or failed during the tick
                if job.status != LoopStatus.ACTIVE:
                    break

                job.next_run_at = datetime.now(UTC) + timedelta(seconds=job.interval_seconds)
                await asyncio.sleep(job.interval_seconds)

        except asyncio.CancelledError:
            job.status = LoopStatus.CANCELLED
            logger.info("Loop job %s background task cancelled.", job.job_id)
        except Exception as exc:
            job.status = LoopStatus.FAILED
            logger.exception("Unexpected error in loop job %s: %s", job.job_id, exc)
