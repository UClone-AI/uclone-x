"""Unit tests verifying kernel dependency footprint and optional framework extras (C9).

docs/core-shell-architecture.md §1 goal 3:
"The kernel's dependency footprint is Pydantic and the standard library.
Frameworks used by a shell (FastAPI, Typer, Rich, uvicorn) and by an adapter
(tree-sitter, OTel SDK, provider SDKs) are optional extras."

Principle 6 (Fail-Fast & Zero Silent Fallbacks):
When an optional framework is missing and a shell or adapter component requires it,
a named MissingDependencyError is raised explaining which extra to install.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from uclone_x.cli.environment_provenance import repo_subprocess_env
from uclone_x.errors import MissingDependencyError, UCloneXError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_in_isolated_python(script: str) -> subprocess.CompletedProcess[str]:
    """Execute python script in a clean, isolated subprocess with repo PYTHONPATH.

    The docstring promised the repo PYTHONPATH and the code passed no `env` at all, so the
    child imported `uclone_x` from whatever the shared venv's editable install named — a
    different worktree, or the bare repository root's stale untracked tree. That is why
    these twelve tests failed under a direct `pytest` and passed under `./ucx test check`,
    which sets the path: the same commit produced irreconcilable results for three people
    on one day (#679, #718). They now depend on the tree under test rather than on the
    machine's install state.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=repo_subprocess_env(REPO_ROOT),
    )


def test_missing_dependency_error_attributes_and_message() -> None:
    """MissingDependencyError carries extra, package, feature, and install instructions.

    Killed by: src/uclone_x/errors.py :: class MissingDependencyError(UCloneXError, ImportError):
    """
    err = MissingDependencyError(extra="cli", package="typer", feature="CLI shell")
    assert isinstance(err, UCloneXError)
    assert isinstance(err, ImportError)
    assert err.extra == "cli"
    assert err.package == "typer"
    assert err.feature == "CLI shell"
    assert "pip install 'uclone-x[cli]'" in str(err)
    assert "Optional dependency 'typer' is required for CLI shell" in str(err)

    # Without feature string
    err_no_feature = MissingDependencyError(extra="http", package="fastapi")
    assert "pip install 'uclone-x[http]'" in str(err_no_feature)
    assert "Optional dependency 'fastapi' is required." in str(err_no_feature)


def test_kernel_core_packages_importable_without_optional_frameworks() -> None:
    """Kernel packages (core, engine, agent, errors, sandbox) import without shell frameworks.

    Killed by: pyproject.toml :: "pydantic>=2.7.0",
    """
    script = (
        "import sys\n"
        "sys.modules['fastapi'] = None\n"
        "sys.modules['uvicorn'] = None\n"
        "sys.modules['typer'] = None\n"
        "sys.modules['rich'] = None\n"
        "sys.modules['tree_sitter'] = None\n"
        "sys.modules['tree_sitter_languages'] = None\n"
        "sys.modules['tree_sitter_language_pack'] = None\n"
        "sys.modules['opentelemetry'] = None\n"
        "import uclone_x.core\n"
        "import uclone_x.engine\n"
        "import uclone_x.agent\n"
        "import uclone_x.errors\n"
        "import uclone_x.sandbox\n"
        "import uclone_x.llm\n"
        "print('KERNEL_IMPORT_OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Kernel import failed without extras:\n{result.stderr}"
    assert "KERNEL_IMPORT_OK" in result.stdout


def test_cli_shell_raises_named_error_when_typer_missing() -> None:
    """Importing uclone_x.cli when typer is missing raises MissingDependencyError.

    Killed by: src/uclone_x/cli/__init__.py :: pkg = "typer" if "typer" in str(exc) else "rich"
    """
    script = (
        "import sys\n"
        "sys.modules['typer'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "try:\n"
        "    import uclone_x.cli\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'cli'\n"
        "    assert exc.package == 'typer'\n"
        "    assert \"pip install 'uclone-x[cli]'\" in str(exc)\n"
        "    print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_cli_shell_raises_named_error_when_rich_missing() -> None:
    """Importing uclone_x.cli when rich is missing raises MissingDependencyError.

    Killed by: src/uclone_x/cli/__init__.py :: pkg = "typer" if "typer" in str(exc) else "rich"
    """
    script = (
        "import sys\n"
        "sys.modules['rich'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "try:\n"
        "    import uclone_x.cli\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'cli'\n"
        "    assert exc.package == 'rich'\n"
        "    assert \"pip install 'uclone-x[cli]'\" in str(exc)\n"
        "    print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_cli_main_raises_named_error_when_typer_missing() -> None:
    """Importing uclone_x.cli.main when typer is missing raises MissingDependencyError.

    Killed by: src/uclone_x/cli/main.py :: pkg = "typer" if "typer" in str(exc) else "rich"
    """
    script = (
        "import sys\n"
        "sys.modules['typer'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "try:\n"
        "    import uclone_x.cli.main\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'cli'\n"
        "    assert exc.package == 'typer'\n"
        "    print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_ui_shell_raises_named_error_when_fastapi_missing() -> None:
    """Importing uclone_x.ui when fastapi is missing raises MissingDependencyError.

    Killed by: src/uclone_x/ui/__init__.py :: pkg = "fastapi" if "fastapi" in str(exc) else "uvicorn"
    """
    script = (
        "import sys\n"
        "sys.modules['fastapi'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "try:\n"
        "    import uclone_x.ui\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'http'\n"
        "    assert exc.package == 'fastapi'\n"
        "    assert \"pip install 'uclone-x[http]'\" in str(exc)\n"
        "    print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_ui_shell_raises_named_error_when_uvicorn_missing() -> None:
    """Importing uclone_x.ui when uvicorn is missing raises MissingDependencyError.

    Killed by: src/uclone_x/ui/__init__.py :: pkg = "fastapi" if "fastapi" in str(exc) else "uvicorn"
    """
    script = (
        "import sys\n"
        "sys.modules['uvicorn'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "try:\n"
        "    import uclone_x.ui\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'http'\n"
        "    assert exc.package == 'uvicorn'\n"
        "    assert \"pip install 'uclone-x[http]'\" in str(exc)\n"
        "    print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_ui_app_raises_named_error_when_fastapi_missing() -> None:
    """Importing uclone_x.ui.app when fastapi is missing raises MissingDependencyError.

    Killed by: src/uclone_x/ui/app.py :: feature="developer UI FastAPI application",
    """
    script = (
        "import sys\n"
        "sys.modules['fastapi'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "try:\n"
        "    import uclone_x.ui.app\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'http'\n"
        "    assert exc.package == 'fastapi'\n"
        "    print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_ui_server_raises_named_error_when_uvicorn_missing() -> None:
    """Importing uclone_x.ui.server when uvicorn is missing raises MissingDependencyError.

    Killed by: src/uclone_x/ui/server.py :: feature="UI server runner",
    """
    script = (
        "import sys\n"
        "sys.modules['uvicorn'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "try:\n"
        "    import uclone_x.ui.server\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'http'\n"
        "    assert exc.package == 'uvicorn'\n"
        "    print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_a2a_server_shell_raises_named_error_when_fastapi_missing() -> None:
    """Importing shells.a2a_server when fastapi is missing raises MissingDependencyError.

    Killed by: src/uclone_x/shells/a2a_server.py :: pkg = "fastapi" if "fastapi" in str(exc) else "uvicorn"
    """
    script = (
        "import sys\n"
        "sys.modules['fastapi'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "try:\n"
        "    import uclone_x.shells.a2a_server\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'http'\n"
        "    assert exc.package == 'fastapi'\n"
        "    assert \"pip install 'uclone-x[http]'\" in str(exc)\n"
        "    print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_a2a_server_shell_raises_named_error_when_uvicorn_missing() -> None:
    """Importing shells.a2a_server when uvicorn is missing raises MissingDependencyError.

    Killed by: src/uclone_x/shells/a2a_server.py :: pkg = "fastapi" if "fastapi" in str(exc) else "uvicorn"
    """
    script = (
        "import sys\n"
        "sys.modules['uvicorn'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "try:\n"
        "    import uclone_x.shells.a2a_server\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'http'\n"
        "    assert exc.package == 'uvicorn'\n"
        "    assert \"pip install 'uclone-x[http]'\" in str(exc)\n"
        "    print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_code_intel_tree_sitter_raises_named_error_when_missing() -> None:
    """Calling require_tree_sitter when tree-sitter is missing raises MissingDependencyError.

    Killed by: src/uclone_x/code_intel/ast_parser.py :: package="tree-sitter",
    Becomes: package="not-tree-sitter",
    """
    script = (
        "import sys\n"
        "sys.modules['tree_sitter'] = None\n"
        "sys.modules['tree_sitter_languages'] = None\n"
        "sys.modules['tree_sitter_language_pack'] = None\n"
        "from uclone_x.code_intel.ast_parser import ASTParser\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "parser = ASTParser(use_tree_sitter=False)\n"
        "try:\n"
        "    parser.require_tree_sitter('python')\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'code-intel'\n"
        "    assert exc.package in {'tree-sitter', 'tree-sitter-language-pack'}\n"
        "    assert \"pip install 'uclone-x[code-intel]'\" in str(exc)\n"
        "try:\n"
        "    ASTParser(strict_tree_sitter=True)\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError:\n"
        "    pass\n"
        "print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_telemetry_otel_sdk_raises_named_error_when_missing() -> None:
    """Invoking OpenTelemetry SDK adapter components without telemetry extra raises.

    Killed by: src/uclone_x/telemetry/otel_sdk.py :: package="opentelemetry-api",
    """
    script = (
        "import sys\n"
        "sys.modules['opentelemetry'] = None\n"
        "sys.modules['opentelemetry.trace'] = None\n"
        "sys.modules['opentelemetry.exporter'] = None\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "from uclone_x.telemetry.otel_sdk import (\n"
        "    require_opentelemetry_api,\n"
        "    get_opentelemetry_tracer,\n"
        "    create_opentelemetry_sdk_exporter,\n"
        ")\n"
        "from uclone_x.telemetry.exporter import create_telemetry_exporter\n"
        "try:\n"
        "    require_opentelemetry_api()\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'telemetry'\n"
        "    assert exc.package == 'opentelemetry-api'\n"
        "    assert \"pip install 'uclone-x[telemetry]'\" in str(exc)\n"
        "try:\n"
        "    get_opentelemetry_tracer()\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'telemetry'\n"
        "try:\n"
        "    create_opentelemetry_sdk_exporter()\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'telemetry'\n"
        "try:\n"
        "    create_telemetry_exporter(use_otel_sdk=True)\n"
        "    sys.exit(1)\n"
        "except MissingDependencyError as exc:\n"
        "    assert exc.extra == 'telemetry'\n"
        "print('OK')\n"
    )
    result = _run_in_isolated_python(script)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "OK" in result.stdout


def test_pyproject_dependencies_conform_to_kernel_core_specification() -> None:
    """pyproject.toml declares narrow kernel dependencies and proper optional extras (C9).

    Killed by: pyproject.toml :: [project.optional-dependencies]
    """
    pyproject_path = REPO_ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    project = data["project"]
    dependencies = project["dependencies"]

    # Base dependencies must contain ONLY kernel necessities
    base_pkgs = [dep.split(">=")[0].split("[")[0].strip() for dep in dependencies]
    assert "pydantic" in base_pkgs
    assert "pydantic-settings" in base_pkgs
    assert "pyyaml" in base_pkgs
    assert "httpx" in base_pkgs

    # Shell and adapter frameworks must NOT be base dependencies
    assert "typer" not in base_pkgs
    assert "rich" not in base_pkgs
    assert "fastapi" not in base_pkgs
    assert "uvicorn" not in base_pkgs
    assert "tree-sitter" not in base_pkgs
    assert "opentelemetry-api" not in base_pkgs

    # Optional dependency groups
    extras = project["optional-dependencies"]
    assert "cli" in extras
    assert any("typer" in dep for dep in extras["cli"])
    assert any("rich" in dep for dep in extras["cli"])

    assert "http" in extras
    assert any("fastapi" in dep for dep in extras["http"])
    assert any("uvicorn" in dep for dep in extras["http"])

    assert "code-intel" in extras
    assert any("tree-sitter" in dep for dep in extras["code-intel"])

    assert "telemetry" in extras
    assert any("opentelemetry" in dep for dep in extras["telemetry"])

    assert "all" in extras
    all_extra = extras["all"]
    assert len(all_extra) == 1
    assert "cli" in all_extra[0]
    assert "http" in all_extra[0]
    assert "telemetry" in all_extra[0]


def test_console_script_reports_the_missing_extra_without_a_traceback() -> None:
    """`ucx` from a base install must print the instruction, not raise it.

    `pip install uclone-x` installs the console script and not the `cli` extra,
    so this is the state of every user who follows the shortest install command
    that works. The import error is correct and stays; what it must not do is
    reach the terminal as a traceback whose final line is the sentence the user
    needed.

    Killed by: src/uclone_x/shells/entry.py :: except MissingDependencyError
    """
    script = (
        "import sys\n"
        "sys.modules['rich'] = None\n"
        "from uclone_x.shells.entry import main\n"
        "try:\n"
        "    main()\n"
        "except SystemExit as exc:\n"
        "    print(f'EXIT={exc.code}')\n"
    )
    result = _run_in_isolated_python(script)

    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
    assert "EXIT=1" in result.stdout
    assert "pip install 'uclone-x[cli]'" in result.stderr
    assert "Traceback (most recent call last)" not in result.stderr


def _run_a_command_that_raises(raised: str) -> subprocess.CompletedProcess[str]:
    """Run `ucx <command>` through the launcher, where the command body raises `raised`.

    A subprocess rather than `pytest.raises`, because what is under test is what reaches
    the terminal: a traceback is printed by `sys.excepthook` as the interpreter unwinds,
    which an in-process call never gets as far as.
    """
    script = (
        "import sys\n"
        "from uclone_x.cli.main import app\n"
        "from uclone_x.errors import MissingDependencyError\n"
        "@app.command('needs-an-extra')\n"
        "def needs_an_extra() -> None:\n"
        f"    raise {raised}\n"
        "from uclone_x.shells.entry import main\n"
        "sys.argv = ['ucx', 'needs-an-extra']\n"
        "main()\n"
    )
    return _run_in_isolated_python(script)


def test_a_command_whose_deferred_import_needs_an_extra_names_it_without_a_traceback() -> None:
    """A `MissingDependencyError` raised *while a command runs* is reported like one at import.

    #926. The launcher caught the error only around importing the CLI, so a command that
    defers an optional import into its body -- the pattern #656 and #881 recommend -- sent
    the same instruction to the terminal as the last line of a Rich traceback, exit 1.

    The mutation keeps the import-time catch and removes only the execution-time one: the
    clause matches while `app` is unbound, which is during the import and never after it.

    Killed by: src/uclone_x/shells/entry.py :: except MissingDependencyError as exc:
    Becomes: except (MissingDependencyError if "app" not in locals() else ()) as exc:
    """
    result = _run_a_command_that_raises(
        "MissingDependencyError(extra='http', package='fastapi', feature='a deferred import')"
    )

    assert result.returncode == 1, f"exit {result.returncode}:\n{result.stderr}"
    assert "ucx: Optional dependency 'fastapi' is required for a deferred import" in result.stderr
    assert "pip install 'uclone-x[http]'" in result.stderr
    assert "Traceback (most recent call last)" not in result.stderr, result.stderr


@pytest.mark.parametrize(
    ("raised", "type_name"),
    [
        pytest.param("RuntimeError('a defect, not a missing extra')", "RuntimeError", id="other"),
        pytest.param(
            "ImportError(\"cannot import name 'helper' from 'a_module_with_a_typo'\")",
            "ImportError",
            id="plain-import-error",
        ),
    ],
)
def test_the_launcher_reports_no_other_exception_as_a_missing_extra(
    raised: str, type_name: str
) -> None:
    """Only the named error is converted; anything else keeps its traceback (P6).

    Converting a broader class would print a developer's defect as a one-line message with
    the evidence discarded.

    Two broadenings, and each row pins one. `Exception` is the obvious one. `ImportError` is
    the nearer one: `MissingDependencyError` subclasses it, so widening the clause to its
    base reads as a tidy-up, and a command body with a genuine import defect -- a typo in a
    name, a circular import -- would then print as one line with no traceback. A
    `RuntimeError` row alone lets that mutation through (#950's review, #954).

    Killed by: src/uclone_x/shells/entry.py :: except MissingDependencyError as exc:
    Becomes: except Exception as exc:
    Killed by: src/uclone_x/shells/entry.py :: except MissingDependencyError as exc:
    Becomes: except ImportError as exc:
    """
    result = _run_a_command_that_raises(raised)

    assert result.returncode == 1, f"exit {result.returncode}:\n{result.stderr}"
    assert "Traceback (most recent call last)" in result.stderr, result.stderr
    assert type_name in result.stderr, result.stderr
    assert "ucx: " not in result.stderr, result.stderr
    assert "uclone-x[" not in result.stderr, result.stderr


def test_console_script_entry_point_is_the_launcher() -> None:
    """`[project.scripts]` must point at the launcher, not past it.

    Pointing `ucx` back at `uclone_x.cli.main:app` -- or moving the launcher
    into `uclone_x.cli`, where importing it trips the very guard it catches, or
    up into the kernel, where it may not import the shell at all --
    restores the traceback without changing a line inside either module, and
    nothing else in the suite would notice: the entry point is only exercised
    by an installed distribution.
    """
    manifest = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")

    assert 'ucx = "uclone_x.shells.entry:main"' in manifest


def test_code_intel_names_a_distribution_that_has_a_cp313_wheel() -> None:
    """`[code-intel]`, and through it `[all]`, must resolve on every supported interpreter.

    `tree-sitter-languages` publishes no wheel above cp312, so on Python 3.13 the extra
    failed to resolve and installed *nothing* -- the beginner got a resolver traceback
    rather than a diagnosis (#1100). `requires-python` claims `>=3.11`, so an extra that
    cannot be installed on 3.13 makes that claim false for `[all]`.

    Killed by: pyproject.toml :: "tree-sitter-language-pack>=1.20.0",
    Becomes: "tree-sitter-languages>=1.10.0",
    """
    pyproject_path = REPO_ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    code_intel = data["project"]["optional-dependencies"]["code-intel"]
    assert any(dep.startswith("tree-sitter-language-pack") for dep in code_intel), code_intel
    assert not any(dep.startswith("tree-sitter-languages") for dep in code_intel), (
        "tree-sitter-languages has no cp313 wheel; it must not be what [code-intel] installs"
    )
