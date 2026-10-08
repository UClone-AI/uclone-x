"""A clone's browser: U0's installed Chrome, controlled over CDP (design `browser-agent.md`).

The names below load on first access, so importing this package, or `browser.turn` (which
every clone composes), loads no `websockets`: that comes with the `http` extra, and a base or
`[cli]` install must still start (#2159, `tests/fitness/test_distribution_install.py`).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uclone_x.browser.cdp import CdpClosedError, CdpConnection, CdpError, CdpEvent
    from uclone_x.browser.chrome import R2Link, find_chrome, launch_args
    from uclone_x.browser.link import BrowserLink
    from uclone_x.browser.service import BrowserService, TabKey

_LAZY_NAMES: dict[str, str] = {
    "CdpClosedError": "uclone_x.browser.cdp",
    "CdpConnection": "uclone_x.browser.cdp",
    "CdpError": "uclone_x.browser.cdp",
    "CdpEvent": "uclone_x.browser.cdp",
    "R2Link": "uclone_x.browser.chrome",
    "find_chrome": "uclone_x.browser.chrome",
    "launch_args": "uclone_x.browser.chrome",
    "BrowserLink": "uclone_x.browser.link",
    "BrowserService": "uclone_x.browser.service",
    "TabKey": "uclone_x.browser.service",
}


def __getattr__(name: str) -> Any:
    """Load a name on first access, so importing this package opens no socket library."""
    module = _LAZY_NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


__all__ = [
    "BrowserLink",
    "BrowserService",
    "CdpClosedError",
    "CdpConnection",
    "CdpError",
    "CdpEvent",
    "R2Link",
    "TabKey",
    "find_chrome",
    "launch_args",
]
