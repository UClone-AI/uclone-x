"""The suite must not load the invoking developer's `mcp.json` files.

`MCPConfigFileLoader` with no workspace looks in `<cwd>/.uclone/mcp.json`, `<cwd>/mcp.json`,
`~/.uclone/mcp.json` and `~/.config/uclone/mcp.json`. The dashboard and every CLI command
build it that way, so under pytest the checkout and the developer's real home decided what
a test's tool inventory held -- and whether it spawned a server. Reading the home config is
intended product behaviour; `tests/conftest.py::_isolate_mcp_config_discovery` redirects it
for tests.

The claim is about that conftest, and the places it protects are ones this test must not
write to: the real home and the real checkout. So it runs a **child pytest** with the
repo-wide conftest as a plugin, started in a fake checkout and with `HOME` pointing at a
fake home, both holding a planted `mcp.json` in every place the loader looks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Discovery stops at the first file it finds, so the child names what it found: that
# identifies which of the four planted locations leaked first.
_CHILD_TEST_MODULE = """
from uclone_x.tools.builtin.mcp_loader import MCPConfigFileLoader


def test_no_planted_mcp_config_is_loaded():
    loader = MCPConfigFileLoader()
    found = loader.discover_config_file()
    assert found is None, f"LEAKED: {found}"
    assert loader.load_configs() == []
"""

_PLANTED = (
    ("checkout", Path(".uclone/mcp.json")),
    ("checkout", Path("mcp.json")),
    ("home", Path(".uclone/mcp.json")),
    ("home", Path(".config/uclone/mcp.json")),
)


def test_a_planted_home_or_checkout_mcp_config_does_not_reach_a_test(tmp_path: Path) -> None:
    """None of the four planted configs is discovered by a test run under the conftest.

    The child reports the file it discovered, so a failure says which lookup leaked. It is
    started in the fake checkout because the loader's working-directory lookup is only
    redirected while the process is in the directory pytest was started from.

    Killed by: tests/conftest.py :: monkeypatch.setattr(mcp_loader, "Path", _IsolatedPath)
    Becomes: _ = (mcp_loader, "Path", _IsolatedPath)
    """
    roots = {"checkout": tmp_path / "checkout", "home": tmp_path / "home"}
    for kind, rel in _PLANTED:
        target = roots[kind] / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"mcpServers": {"planted": {"command": "/usr/bin/false"}}}))

    child_dir = tmp_path / "child"
    child_dir.mkdir()
    (child_dir / "test_planted_mcp.py").write_text(_CHILD_TEST_MODULE)

    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["HOME"] = str(roots["home"])
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # No ini file is found from the child's rootdir, so it inherits no `addopts`.
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "tests.conftest"]
    argv.append(str(child_dir / "test_planted_mcp.py"))
    result = subprocess.run(
        argv, capture_output=True, text=True, env=env, cwd=str(roots["checkout"]), check=False
    )

    # Count the tests, never the exit code: a run that collects nothing exits non-zero too.
    assert "1 passed" in result.stdout, (
        f"a planted mcp.json reached the child test, or it did not run:\n"
        f"{result.stdout}\n{result.stderr}"
    )
