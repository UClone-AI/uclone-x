"""The `browser` name on an install without the browser's extra (#2164).

The browser drives Chrome over a websocket, and `websockets` comes with the `http` extra
only. A base or `[cli]` install therefore cannot build `BrowserTool`; until #2164 the
default registry imported it anyway, so `ucx acp`, `ucx a2a` and `ucx loop` died on
`ModuleNotFoundError` before the first turn.

The name stays registered, as this stand-in, for two reasons. A persona that lists
`browser` (Scout does) is refused at load when the name is unknown, so leaving it out would
turn a missing extra into a persona that cannot load. And a clone asked to browse should be
able to tell the person why it cannot, in words, instead of the name simply not existing.
The stand-in takes any arguments and always refuses with `BROWSER_UNAVAILABLE`.

This module imports nothing from the `http` extra, so it loads on every install.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict

from uclone_x.errors import PlainRefusalError
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResult

#: What a call to the browser says on an install without it, in words for the person.
BROWSER_UNAVAILABLE: Final = (
    "The browser is not part of this install, so no page was opened. "
    "To use it, install the http extra: pip install 'uclone-x[http]'"
)


class _AnyArguments(BaseModel):
    """Accept whatever the model sends: the answer is the same refusal for every call."""

    model_config = ConfigDict(extra="ignore")


class BrowserUnavailableTool(BaseTool[_AnyArguments]):
    """Stands in for `browser` where its extra is missing, and says so when called."""

    name: str = "browser"
    writes_files: ClassVar[bool] = False
    description: str = "Real Chrome browser. Not available on this install; a call says why."

    async def run(self, params: _AnyArguments, context: ToolContext) -> dict[str, Any] | ToolResult:
        """Refuse in plain words, naming what to install."""
        raise PlainRefusalError(BROWSER_UNAVAILABLE, reason_code="browser_not_installed")
