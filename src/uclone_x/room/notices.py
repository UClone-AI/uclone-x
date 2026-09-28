"""Notices the application writes into a conversation, as codes a head words (#1641).

A `/loop` reply is stored in the transcript, so a sentence written here would stay in the
language that was active when it was written, and switching languages later would leave old
notes in one language and new ones in the other (multilingual-ui.md §3.3). So a note carries
a `code` and its `params`, and the head writes the sentence from its catalog every time it
draws the row: `frontend/src/i18n/locales/<language>/notices.json`, key for key.

`content` is still filled, in English, from `NOTICE_FALLBACK`. It is what an export, the CLI
and a head that predates a code show, and it is never what a current head shows for a code it
knows. A code added here and not to the head's catalog fails
`tests/unit/test_notice_contract.py`, not a user's screen.
"""

from __future__ import annotations

from typing import Final, Literal

__all__ = [
    "NOTICE_FALLBACK",
    "NoticeCode",
    "NoticeParams",
    "interval_text",
    "notice_content",
]

NoticeCode = Literal[
    "loop.help",
    "loop.active",
    "loop.none_active",
    "loop.stopped",
    "loop.nothing_to_stop",
    "loop.registered",
    "loop.missing_prompt",
    "loop.interval_too_short",
    "loop.no_interval",
]
"""Every notice the Core writes. Closed: a head renders exactly these."""

NoticeParams = dict[str, str | int | float]
"""A notice's values, by the placeholder names its sentence uses. Never a sentence."""

NOTICE_FALLBACK: Final[dict[NoticeCode, str]] = {
    "loop.help": (
        "ℹ️ **Repeating tasks with `/loop`:**\n\n"
        "• `/loop <interval> <prompt>`: run a prompt on a schedule "
        "(for example `/loop 30s check the status` or `/loop every 10 minutes summarize`)\n"
        "• `/loop list`: show this conversation's repeating task\n"
        "• `/loop stop`: stop this conversation's repeating task "
        "(the stop button at the top stops it too)"
    ),
    "loop.active": ('🔄 **Repeating task:** `{job_id}` (every {interval})\nPrompt: "{prompt}"'),
    "loop.none_active": "ℹ️ No repeating task is running in this conversation.",
    "loop.stopped": "🛑 The repeating task was stopped.",
    "loop.nothing_to_stop": "ℹ️ There is no repeating task to stop.",
    "loop.registered": (
        "🔄 **Repeating task started** (every {interval}, ID: `{job_id}`):\n"
        '"{prompt}"\n\n'
        "*To stop it, type `/loop stop` or press the stop button at the top of the "
        "conversation.*"
    ),
    "loop.missing_prompt": (
        "⚠️ Add what to run after the interval, for example `/loop 5m check the build`."
    ),
    "loop.interval_too_short": (
        "⚠️ A repeating task can run at most once every {interval}. Choose a longer interval."
    ),
    "loop.no_interval": (
        "⚠️ That `/loop` command has no interval. Write it as `/loop 5m <prompt>` "
        "or `/loop every 10 minutes <prompt>`."
    ),
}


def interval_text(seconds: float) -> str:
    """An interval in English words, the way the head's catalog writes it.

    Seconds under a minute, or when the interval is not a whole number of minutes; minutes
    otherwise. The head does the same from the raw `interval_seconds` param.
    """
    if seconds < 60 or seconds % 60:
        count: float = seconds
        unit = "second"
    else:
        count = seconds // 60
        unit = "minute"
    shown = int(count) if float(count).is_integer() else count
    return f"{shown} {unit}" if shown == 1 else f"{shown} {unit}s"


def notice_content(code: NoticeCode, params: NoticeParams) -> str:
    """The English fallback for `code`, filled from `params`.

    An `interval_seconds` param is written as the `{interval}` its sentence uses.
    """
    values = {name: str(value) for name, value in params.items()}
    raw_interval = params.get("interval_seconds")
    if isinstance(raw_interval, int | float):
        values["interval"] = interval_text(float(raw_interval))
    return NOTICE_FALLBACK[code].format_map(values)
