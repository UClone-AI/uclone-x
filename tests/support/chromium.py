"""Where Playwright's Chromium is, for the browser e2e tests that launch it over R2.

Asking `sync_playwright()` in the test process starts Playwright's driver only to read one
path. Its shutdown then leaves a `TargetClosedError` future unread, and asyncio prints
"Future exception was never retrieved" after the test. The path is read once, in a child
process whose output is captured.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

_ASK = (
    "from playwright.sync_api import sync_playwright\n"
    "with sync_playwright() as p:\n"
    "    print(p.chromium.executable_path)\n"
)


@functools.cache
def chromium_executable() -> Path:
    """The Chromium Playwright installed. Raises if Playwright cannot say where it is."""
    done = subprocess.run(
        [sys.executable, "-c", _ASK], capture_output=True, text=True, timeout=120, check=False
    )
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or not lines:
        raise RuntimeError(f"Playwright did not name its Chromium: {done.stderr.strip()[-300:]}")
    return Path(lines[-1])
