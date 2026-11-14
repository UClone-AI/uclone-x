"""Whether the running turn was admitted by the usage limit at its start, and what it spent (#2431).

Owner ruling 2026-10-30: "한도는 턴을 시작할때 확인하고, 일단 턴이 시작하면 한도를 넘더라도
끝내야 한다" ("check the limit when a turn starts; once it has started, finish it even past
the limit"). `enter_admitted_turn` is that one check; the gate (`gate.py`) reads `ADMITTED`
and books, but does not refuse, a call made inside an admitted turn.

The owner's amendment the same day: "폭주 상한 방지 한도는 두자" ("keep a ceiling that stops
a runaway"). An admitted turn also carries its own spend (`TurnSpend`): the gate adds every
paid call it books, and once the spend passes the ceiling, the person's `per_10_minutes`
limit, the gate stops the turn before its next call (`check_turn_ceiling`): at a step
boundary, never midway through a call. A
turn continued from that stop (`continuing=True`) is not checked again at its start, and
spends from zero, so continuing grants one more ceiling's worth. With no 10-minute limit
set there is no ceiling.

Kept apart from `gate.py`, which imports connectors and the compactor, so the agent's turn
loop can import it without an import cycle.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from uclone_x.errors import TurnCeilingReachedError

__all__ = [
    "ADMITTED",
    "TurnSpend",
    "book_turn_spend",
    "check_turn_ceiling",
    "enter_admitted_turn",
    "leave_admitted_turn",
    "turn_admission",
    "turn_ceiling_passed",
]

#: True while a turn that was admitted at its start is running. A context variable, so
#: turns running at once, and the tasks a turn starts, each see their own.
ADMITTED: ContextVar[bool] = ContextVar("uclone_x_usage_admitted_turn", default=False)


@dataclass
class TurnSpend:
    """The paid tokens one admitted turn has spent, and the ceiling it stops at."""

    ceiling: int | None
    used: int = 0


_SPEND: ContextVar[TurnSpend | None] = ContextVar("uclone_x_usage_turn_spend", default=None)

#: What `turn_admission` returns and `leave_admitted_turn` puts back.
Admission = tuple[bool, TurnSpend | None]


def turn_admission() -> Admission:
    """The admission in force now, to put back with `leave_admitted_turn` after a turn."""
    return ADMITTED.get(), _SPEND.get()


def enter_admitted_turn(connector: object, *, continuing: bool = False) -> Admission:
    """Check the usage limit once, at a turn's start, and mark the rest of the turn admitted.

    Raises `UsageLimitReachedError` when ``connector`` is paid (wrapped by the gate) and a
    limit is reached; the turn then stops as `usage_limit` before any call. A turn started
    inside an admitted one (a helper's) is not checked again and spends against the outer
    turn. A turn ``continuing`` one its ceiling stopped is the same turn: not checked, and
    given a fresh ceiling. Returns the admission to put back with `leave_admitted_turn`.
    """
    before = turn_admission()
    if ADMITTED.get():
        return before
    admit: object = getattr(connector, "usage_admit", None)
    if callable(admit) and not continuing:
        admit()  # the one check a turn gets
    ceiling: object = getattr(connector, "usage_turn_ceiling", None)
    found: object = ceiling() if callable(ceiling) else None
    _SPEND.set(TurnSpend(ceiling=found if isinstance(found, int) else None))
    ADMITTED.set(True)
    return before


def leave_admitted_turn(before: Admission) -> None:
    """Put back the admission `enter_admitted_turn` or `turn_admission` returned."""
    admitted, spend = before
    ADMITTED.set(admitted)
    _SPEND.set(spend)


def book_turn_spend(tokens: int) -> None:
    """Add a booked paid call's tokens to the admitted turn's spend, when one is running."""
    spend = _SPEND.get()
    if spend is not None and ADMITTED.get():
        spend.used += tokens


def turn_ceiling_passed() -> TurnSpend | None:
    """The running turn's spend when it has passed its ceiling, else `None`."""
    spend = _SPEND.get()
    if spend is None or spend.ceiling is None or not ADMITTED.get():
        return None
    return spend if spend.used > spend.ceiling else None


def check_turn_ceiling() -> None:
    """Stop the admitted turn before its next call once its spend passed its ceiling.

    Raises:
        TurnCeilingReachedError: The running turn's own spend is past its ceiling.
    """
    spend = turn_ceiling_passed()
    if spend is not None and spend.ceiling is not None:
        raise TurnCeilingReachedError(spend.used, spend.ceiling)
