"""The default registry on an install without the browser's extra (#2164).

`websockets` comes with the `http` extra only. These tests make it unimportable in this
process, the way a `[cli]` install has it, and build the registry `ucx acp`, `ucx a2a` and
`ucx loop` build. The fitness lane runs the same path from the built wheel
(`tests/fitness/test_distribution_install.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.browser.unavailable import BROWSER_UNAVAILABLE, BrowserUnavailableTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import create_default_registry

#: The modules that bring `websockets` in when the browser tool is imported. Dropped from
#: `sys.modules` for one test so the import runs again; `monkeypatch` puts them back.
_BROWSER_MODULES = (
    "uclone_x.browser.tool",
    "uclone_x.browser.extension",
    "uclone_x.browser.cdp",
)


@pytest.fixture
def without_websockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `websockets` unimportable, as on a base or `[cli]` install."""
    for name in [n for n in sys.modules if n == "websockets" or n.startswith("websockets.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "websockets", None)
    for name in _BROWSER_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.usefixtures("without_websockets")
async def test_without_websockets_the_registry_builds_and_the_browser_says_what_to_install() -> (
    None
):
    """The registry builds, and a call to `browser` refuses in plain words.

    Killed by: src/uclone_x/tools/registry.py ::
        return BrowserUnavailableTool()
    Becomes: raise
    """
    registry = create_default_registry(enable_mcp=False)

    tool = registry.get("browser")
    assert isinstance(tool, BrowserUnavailableTool)
    result = await tool.execute(
        {"action": "open", "url": "https://example.com"},
        ToolContext(agent_id="scout", session_id="s-1"),
    )
    assert result.success is False
    assert result.error == BROWSER_UNAVAILABLE
    assert "uclone-x[http]" in (result.error or "")
    for internal in ("Traceback", "ModuleNotFoundError", "websockets", "Error:"):
        assert internal not in (result.error or ""), result.error


@pytest.mark.usefixtures("without_websockets")
def test_without_websockets_a_persona_listing_the_browser_still_loads(tmp_path: Path) -> None:
    """Scout lists `browser`; an unknown name would refuse the persona at load.

    Killed by: src/uclone_x/browser/unavailable.py :: name: str = "browser"
    Becomes: name: str = "browser_unavailable"
    """
    registry = create_default_registry(enable_mcp=False)
    personas = PersonaRegistry(
        workspace_root=tmp_path, tool_names=[tool.name for tool in registry.list_tools()]
    )

    scout = personas.get_persona("scout")
    assert scout is not None
    assert "browser" in scout.allowed_tools


def test_with_websockets_the_registry_has_the_real_browser() -> None:
    """The stand-in is for a missing extra only, never in place of a working browser.

    Killed by: src/uclone_x/tools/registry.py :: return BrowserTool()
    Becomes: return __import__("uclone_x.browser.unavailable", fromlist=["x"]).BrowserUnavailableTool()
    """
    from uclone_x.browser.tool import BrowserTool

    tool = create_default_registry(enable_mcp=False).get("browser")
    assert isinstance(tool, BrowserTool)


def test_an_import_failure_other_than_websockets_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the extra's own module is answered with the stand-in; anything else propagates.

    Killed by: src/uclone_x/tools/registry.py ::
        if (exc.name or "").partition(".")[0] != "websockets":
    Becomes: if False:
    """
    monkeypatch.setitem(sys.modules, "uclone_x.browser.tool", None)

    with pytest.raises(ModuleNotFoundError):
        create_default_registry(enable_mcp=False)
