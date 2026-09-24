"""Models and data structures for recurring loop execution (FR-Loop, P1, P4, P6)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class LoopStatus(StrEnum):
    """Lifecycle status of a recurring loop job."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class LoopTickResult:
    """Result of a single execution tick of a loop job."""

    tick_index: int
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    success: bool
    content: str = ""
    error: str | None = None
    skipped: bool = False
    stop_reason: str | None = None


@dataclass
class LoopJob:
    """Configuration and live state for a recurring loop job."""

    job_id: str
    interval_seconds: float
    prompt: str
    max_runs: int | None = None
    timeout_seconds: float = 600.0  # Default 10 min watchdog
    clean_context: bool = False
    until_pattern: str | None = None
    max_consecutive_failures: int = 3

    # Live runtime states
    runs_count: int = 0
    consecutive_failures: int = 0
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    is_running: bool = False
    status: LoopStatus = LoopStatus.ACTIVE
    last_result: LoopTickResult | None = None
    history: list[LoopTickResult] = field(default_factory=list[LoopTickResult])

    # Concurrency lock to prevent overlapping runs (uclone2 pattern)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    @property
    def lock(self) -> asyncio.Lock:
        """Concurrency lock ensuring at most one execution tick runs at a time."""
        return self._lock
