"""The single-agent runtime as a library: what a caller of the base install relies on (#2011).

`examples/single_agent.py` is the published way to embed one agent without the app, so it
is run here rather than only read, in a child process that imports the tree under test.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from importlib.resources import files
from pathlib import Path

from uclone_x import agent
from uclone_x.agent.session import SessionStore
from uclone_x.cli.environment_provenance import repo_subprocess_env

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "single_agent.py"

#: Import roots of the provider SDKs the connectors deliberately do not use.
PROVIDER_SDK_ROOTS = frozenset({"anthropic", "openai", "google.genai"})

#: Packages that belong to the app around the agent, not to the agent.
APP_PACKAGES = frozenset({"room", "ui", "cli", "shells", "artifacts", "link"})


def test_example_runs_one_turn_without_loading_the_app() -> None:
    """The example completes a turn, and the process never imports a head or the room.

    Killed by: examples/single_agent.py :: print(result.content)
    Becomes: print(result.content.upper())
    """
    probe = (
        "import runpy, sys\n"
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')\n"
        "print(sorted({m.split('.')[1] for m in sys.modules if m.startswith('uclone_x.')}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=repo_subprocess_env(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    reply, loaded = result.stdout.splitlines()[-2:]
    assert reply == "Hello from a single agent."
    assert APP_PACKAGES.isdisjoint(ast.literal_eval(loaded)), loaded


def test_composition_root_is_exported_from_the_agent_package() -> None:
    """`from uclone_x.agent import compose_agent, HostDependencies` works and is public.

    Killed by: src/uclone_x/agent/__init__.py :: "compose_agent",
    Becomes:
    """
    for name in ("compose_agent", "HostDependencies", "MissingCapabilityError"):
        assert name in agent.__all__
        assert getattr(agent, name).__module__ == "uclone_x.agent.composition"


def test_session_store_accepts_a_path_string(tmp_path: Path) -> None:
    """A `str` storage dir is taken as a path; it raised `AttributeError` on `.resolve()`.

    Killed by: src/uclone_x/agent/session.py :: Path(storage_dir).resolve()
    Becomes: storage_dir.resolve()
    """
    store = SessionStore(storage_dir=str(tmp_path / "sessions"))
    assert store.storage_dir == (tmp_path / "sessions").resolve()
    assert store.storage_dir.is_dir()


def test_package_declares_itself_typed() -> None:
    """PEP 561 marker, so a downstream type checker reads the annotations.

    Deleting `src/uclone_x/py.typed` fails this; a declaration cannot express a deleted file.
    """
    assert files("uclone_x").joinpath("py.typed").is_file()


def test_no_source_module_imports_a_provider_sdk() -> None:
    """Every connector speaks HTTP through `httpx`, which is why no extra installs an SDK.

    If this fails, a module now needs a provider SDK: declare it in an extra and say so in
    the public README, rather than let the base install fail at import.
    """
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "uclone_x").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            else:
                continue
            for name in names:
                if any(name == root or name.startswith(root + ".") for root in PROVIDER_SDK_ROOTS):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {name}")
    assert offenders == []


def test_llm_extra_is_declared_and_installs_nothing() -> None:
    """`uclone-x[llm]` keeps resolving for existing install commands, and pulls no SDK."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]
    assert extras["llm"] == []
