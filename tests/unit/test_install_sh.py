"""install.sh on a machine that has only what a normal Mac ships.

Each case runs the real script under a PATH made of stubs: fake ``python3``,
``uv``, ``curl``, ``uname``, ``sysctl`` and ``xcode-select``, plus links to the
handful of real tools the script needs. Nothing touches the network or the host
interpreter; every stub appends what it was asked to do to a log.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.pre_release

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"
# /bin/bash on macOS is 3.2, which is what a beginner's Mac runs the script with.
BASH = "/bin/bash" if Path("/bin/bash").exists() else (shutil.which("bash") or "bash")
REAL_TOOLS = ("sed", "dirname", "grep", "cat", "mkdir", "sh", "chmod", "cp", "ln", "readlink")

# A uv that records its calls and, for `uv venv`, makes a venv whose ucx has `start`.
# Like the real one, `uv venv --python 3.12` probes the first python3 on PATH
# unless told to use only its own Pythons (checked against uv 0.11 in review).
UV_STUB = """#!/bin/sh
echo "uv $*" >> "$STUB_LOG"
if [ "$1" = "venv" ]; then
    if [ "${UV_PYTHON_PREFERENCE:-}" != "only-managed" ]; then
        p="$(command -v python3 2>/dev/null)" && "$p" -c probe >/dev/null
    fi
    for last; do :; done
    mkdir -p "$last/bin"
    printf '#!/bin/sh\\nexit 0\\n' > "$last/bin/python"
    cat > "$last/bin/ucx" <<'EOU'
#!/bin/sh
echo "ucx $*" >> "$STUB_LOG"
if [ "$1 $2" = "install --help" ]; then exit "${STUB_UCX_HAS_INSTALL:-0}"; fi
if [ "$1" = "install" ]; then exit "${STUB_UCX_INSTALL_EXIT:-0}"; fi
if [ "$*" = "start" ]; then read -r l || true; echo "ucx start stdin=$l" >> "$STUB_LOG"; fi
exit 0
EOU
    chmod +x "$last/bin/python" "$last/bin/ucx"
fi
exit 0
"""

# The astral installer, as curl would return it: puts uv in ~/.local/bin.
CURL_STUB = """#!/bin/sh
echo "curl $*" >> "$STUB_LOG"
cat <<'EOS'
mkdir -p "$HOME/.local/bin"
cp "$STUB_UV_SOURCE" "$HOME/.local/bin/uv"
chmod +x "$HOME/.local/bin/uv"
echo "astral-installer no-modify-path=$UV_NO_MODIFY_PATH" >> "$STUB_LOG"
EOS
"""


def _write(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _python_stub(version: str) -> str:
    return f'#!/bin/sh\necho "ran $0" >> "$STUB_LOG"\necho {version}\n'


class Machine:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.bin = tmp_path / "bin"
        self.home = tmp_path / "home"
        self.log = tmp_path / "log"
        self.bin.mkdir()
        self.home.mkdir()
        self.log.write_text("")
        for tool in REAL_TOOLS:
            real = shutil.which(tool)
            assert real, tool
            (self.bin / tool).symlink_to(real)
        self.uv_source = tmp_path / "uv-source"
        _write(self.uv_source, UV_STUB)
        _write(self.bin / "curl", CURL_STUB)
        self.os("Darwin", clt=True)
        self.ram_bytes(16 * 1024**3)

    def os(self, name: str, *, clt: bool) -> None:
        _write(self.bin / "uname", f"#!/bin/sh\necho {name}\n")
        _write(self.bin / "xcode-select", f"#!/bin/sh\nexit {0 if clt else 2}\n")

    def ram_bytes(self, n: int) -> None:
        _write(self.bin / "sysctl", f"#!/bin/sh\necho {n}\n")

    def python(self, name: str, version: str, where: Path | None = None) -> Path:
        d = where or self.bin
        d.mkdir(exist_ok=True)
        _write(d / name, _python_stub(version))
        return d / name

    def uv_on_path(self) -> None:
        _write(self.bin / "uv", UV_STUB)

    def run(
        self,
        *args: str,
        script: Path = INSTALL_SH,
        extra_path: list[Path] | None = None,
        env_extra: dict[str, str] | None = None,
        image: str | None = "--no-image",
        tty: str | None = None,
        standalone: bool = False,
        piped: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        # Every case but the image-consent one answers the image question in advance, so
        # that the branch under test is never the one deciding a 1.0 GB install. Passing
        # ``image=None`` leaves the question open and hands the decision to the script.
        path = ":".join(str(p) for p in [*(extra_path or []), self.bin])
        env = {
            "PATH": path,
            "HOME": str(self.home),
            "STUB_LOG": str(self.log),
            "STUB_UV_SOURCE": str(self.uv_source),
            "UCX_INSTALL_CLT_SHIM_DIR": str(self.root / "shims"),
            # The terminal the script asks on. Absent by default, so no case waits on
            # the developer's own terminal; ``tty`` is the answer the user types.
            "UCX_INSTALL_TTY": str(self.root / "no-tty"),
            **(env_extra or {}),
        }
        if tty is not None:
            (self.root / "tty").write_text(tty + "\n")
            env["UCX_INSTALL_TTY"] = str(self.root / "tty")
        venv = [] if standalone else ["--venv", str(self.root / "venv")]
        if standalone:
            script = self.root / "lone" / "install.sh"
            script.parent.mkdir(exist_ok=True)
            shutil.copy(INSTALL_SH, script)
        flags = [*([image] if image else []), *venv, *args]
        # `piped` is the README's `curl ... | bash` itself: the script arrives on stdin,
        # so anything in it that reads stdin eats the script's own remaining lines.
        argv = [BASH, "-s", "--", *flags] if piped else [BASH, str(script), *flags]
        return subprocess.run(
            argv,
            env=env,
            cwd=self.root,
            input=INSTALL_SH.read_text() if piped else None,
            # Otherwise stdin is never a terminal either, and a new session has no /dev/tty.
            stdin=None if piped else subprocess.DEVNULL,
            start_new_session=True,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines()


@pytest.fixture
def machine(tmp_path: Path) -> Machine:
    return Machine(tmp_path)


def test_a_mac_with_only_python_39_refuses_without_consent(machine: Machine) -> None:
    machine.python("python3", "3.9")
    result = machine.run()
    assert result.returncode == 1
    assert "--yes" in result.stdout
    assert not any(c.startswith("curl") for c in machine.calls())
    assert not (machine.root / "venv").exists()


def test_yes_installs_uv_then_a_private_python(machine: Machine) -> None:
    machine.python("python3", "3.9")
    result = machine.run("--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = machine.calls()
    assert "curl -LsSf https://astral.sh/uv/install.sh" in calls
    assert "astral-installer no-modify-path=1" in calls
    assert f"uv venv --seed --python 3.12 {machine.root / 'venv'}" in calls
    assert any(c.startswith("uv pip install") and f"{REPO_ROOT}[cli,http]" in c for c in calls)


def test_a_rerun_reuses_the_environment_without_asking_again(machine: Machine) -> None:
    machine.python("python3", "3.9")
    assert machine.run("--yes").returncode == 0
    for args in ((), ("--dry-run",)):
        machine.log.write_text("")
        result = machine.run(*args)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Reusing the existing environment." in result.stdout
        assert not any(c.startswith(("curl", "uv venv")) for c in machine.calls())
        assert any(c.startswith("uv pip install") for c in machine.calls())


def test_an_existing_uv_still_asks_before_fetching_python(machine: Machine) -> None:
    machine.python("python3", "3.9")
    machine.uv_on_path()
    refused = machine.run()
    assert refused.returncode == 1
    assert not any(c.startswith("uv venv") for c in machine.calls())
    result = machine.run("--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"uv venv --python 3.12 {machine.root / 'venv'}" in machine.calls()
    assert not any(c.startswith("curl") for c in machine.calls())


def test_uv_off_path_gives_the_venv_pip_for_the_dashboard(machine: Machine) -> None:
    machine.python("python3", "3.9")
    result = machine.run("--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"uv venv --seed --python 3.12 {machine.root / 'venv'}" in machine.calls()


def test_a_failed_uv_download_says_so(machine: Machine) -> None:
    machine.python("python3", "3.9")
    _write(machine.bin / "curl", "#!/bin/sh\nexit 6\n")
    result = machine.run("--yes")
    assert result.returncode == 1
    assert "The uv installer failed" in result.stderr


def test_uv_in_local_bin_is_found_off_path(machine: Machine) -> None:
    local_bin = machine.home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    _write(local_bin / "uv", UV_STUB)
    result = machine.run("--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(c.startswith("curl") for c in machine.calls())


def test_a_usable_python_is_used_as_is(machine: Machine) -> None:
    py = machine.python("python3.12", "3.12")
    machine.uv_on_path()
    result = machine.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"uv venv --python {py} {machine.root / 'venv'}" in machine.calls()


def test_the_command_line_tools_shim_is_never_run(machine: Machine) -> None:
    shim = machine.python("python3", "3.12", where=machine.root / "shims")
    machine.os("Darwin", clt=False)
    machine.uv_on_path()
    result = machine.run("--yes", extra_path=[machine.root / "shims"])
    assert result.returncode == 0, result.stdout + result.stderr
    # Neither the script nor uv's interpreter search may run it.
    assert f"ran {shim}" not in machine.calls()
    assert f"uv venv --python 3.12 {machine.root / 'venv'}" in machine.calls()


def test_with_the_command_line_tools_the_system_python_is_eligible(machine: Machine) -> None:
    shim = machine.python("python3", "3.12", where=machine.root / "shims")
    machine.uv_on_path()
    result = machine.run(extra_path=[machine.root / "shims"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"uv venv --python {shim} {machine.root / 'venv'}" in machine.calls()


def test_dry_run_fetches_no_python_when_uv_is_present(machine: Machine) -> None:
    machine.python("python3", "3.9")
    machine.uv_on_path()
    result = machine.run("--dry-run", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "would fetch Python 3.12" in result.stdout
    assert not any(c.startswith("uv ") for c in machine.calls())


def test_dry_run_installs_nothing_when_uv_is_missing(machine: Machine) -> None:
    machine.python("python3", "3.9")
    result = machine.run("--dry-run", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "would install uv" in result.stdout
    assert machine.calls() == ["ran " + str(machine.bin / "python3")]


@pytest.mark.parametrize(
    ("gb", "expected"),
    [
        (8, "qwen3:1.7b"),
        (16, "qwen3:8b; images at 512px"),
        (24, "images at the default 768px"),
        (64, "both loaded at once"),
    ],
)
def test_the_summary_names_what_fits_in_memory(machine: Machine, gb: int, expected: str) -> None:
    machine.python("python3.12", "3.12")
    machine.uv_on_path()
    machine.ram_bytes(gb * 1024**3)
    result = machine.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"memory        {gb} GB" in result.stdout
    assert expected in result.stdout


# --- the models -------------------------------------------------------------
# The core install leaves a dashboard that cannot answer or draw: the weights are
# the other ~8 GB. These cases pin who asks for them, who does not, and what the
# summary says when they only half-arrive.


def _ready(machine: Machine) -> None:
    machine.python("python3.12", "3.12")
    machine.uv_on_path()


def test_without_a_terminal_the_models_are_offered_but_not_fetched(machine: Machine) -> None:
    _ready(machine)
    result = machine.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Re-run with --with-models" in result.stdout
    assert "models        skipped" in result.stdout
    assert "ucx install --yes" not in machine.calls()
    # ...and the closing hint tells them the one command that finishes the job.
    assert f"{machine.root / 'venv'}/bin/ucx install" in result.stdout


def test_with_models_fetches_them(machine: Machine) -> None:
    _ready(machine)
    result = machine.run("--with-models")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ucx install --yes --no-image" in machine.calls()
    assert "models        installed" in result.stdout
    assert f"{machine.root / 'venv'}/bin/ucx install" not in result.stdout


def test_yes_accepts_the_models_too(machine: Machine) -> None:
    _ready(machine)
    result = machine.run("--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ucx install --yes --no-image" in machine.calls()


def test_yes_consents_to_the_image_engine_as_well_as_the_models(machine: Machine) -> None:
    """One flag, one meaning: `--yes` answers the image question the way it answers theirs.

    The documented one-liner pipes the script into bash, so `[ -t 0 ]` is false. Deciding
    the image question on that alone sent a run that had said yes to everything down the
    "not a terminal, so not asking" branch: no engine, and -- because the checkpoint half
    is gated on the engine -- no checkpoint either, under a summary reading `models
    installed`. A regression here restores that silent, undocumented no.
    """
    _ready(machine)
    result = machine.run("--yes", image=None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(
        c.startswith("uv pip install") and f"{REPO_ROOT}[media]" in c for c in machine.calls()
    )
    assert "image engine  installed" in result.stdout
    # The engine went in, so the 6.9 GB checkpoint has something to load it: the models
    # step must ask for both halves rather than narrowing itself to the LLM.
    assert "ucx install --yes" in machine.calls()
    assert "ucx install --yes --no-image" not in machine.calls()


def test_no_models_does_not_even_probe(machine: Machine) -> None:
    _ready(machine)
    result = machine.run("--no-models")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(c.startswith("ucx install") for c in machine.calls())
    assert "models        skipped" in result.stdout


def test_the_checkpoint_is_asked_for_only_with_the_image_engine(machine: Machine) -> None:
    _ready(machine)
    result = machine.run("--with-image", "--with-models")
    assert result.returncode == 0, result.stdout + result.stderr
    # A 6.9 GB checkpoint with no engine to load it is a wasted download, so the
    # image half is requested exactly when the engine went in.
    assert "ucx install --yes" in machine.calls()
    assert "ucx install --yes --no-image" not in machine.calls()


def test_a_build_without_ucx_install_says_so_instead_of_claiming_success(
    machine: Machine,
) -> None:
    _ready(machine)
    result = machine.run("--with-models", env_extra={"STUB_UCX_HAS_INSTALL": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "This build has no" in result.stdout
    assert "models        not in this build" in result.stdout
    assert "ucx install --yes --no-image" not in machine.calls()


def test_a_half_finished_download_is_reported_not_swallowed(machine: Machine) -> None:
    _ready(machine)
    result = machine.run("--with-models", env_extra={"STUB_UCX_INSTALL_EXIT": "1"})
    # The core is installed and the dashboard starts, so the run still succeeds --
    # but the summary must not read "installed".
    assert result.returncode == 0, result.stdout + result.stderr
    assert "models        incomplete" in result.stdout
    assert f"{machine.root / 'venv'}/bin/ucx install" in result.stdout


def test_dry_run_describes_the_models_without_fetching_them(machine: Machine) -> None:
    _ready(machine)
    result = machine.run("--dry-run", "--with-models")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "models        would install" in result.stdout
    assert not any(c.startswith("ucx") for c in machine.calls())


def test_run_outside_a_checkout_installs_the_published_package(machine: Machine) -> None:
    py = machine.python("python3.12", "3.12")
    machine.uv_on_path()
    result = machine.run("--with-image", image=None, standalone=True)
    assert result.returncode == 0, result.stdout + result.stderr
    venv = machine.home / ".uclone-x" / "venv"
    calls = machine.calls()
    assert f"uv venv --python {py} {venv}" in calls
    assert any(c.endswith("uclone-x[cli,http]") for c in calls)
    # The published package has no image extra; installing it would be a silent no-op.
    assert not any(c.endswith("uclone-x[media]") for c in calls)
    assert "image engine  not in the published package" in result.stdout


# --- the command and the first start ------------------------------------------
# The README promises: paste one line, and UClone-X opens. These cases pin the two
# steps after the install that make that true -- a `ucx` the user can type, and the
# offer to start it -- on the piped path, where stdin is not the terminal.


def test_the_published_install_links_ucx_and_says_how_to_reach_it(machine: Machine) -> None:
    _ready(machine)
    result = machine.run(standalone=True)
    assert result.returncode == 0, result.stdout + result.stderr
    link = machine.home / ".local" / "bin" / "ucx"
    assert link.is_symlink()
    assert os.readlink(link) == str(machine.home / ".uclone-x" / "venv" / "bin" / "ucx")
    # ~/.local/bin is not on this PATH, so the short name would not resolve.
    assert f"    {link} start" in result.stdout
    assert 'export PATH="$HOME/.local/bin:$PATH"' in result.stdout


def test_with_local_bin_on_path_the_short_name_is_given(machine: Machine) -> None:
    _ready(machine)
    local_bin = machine.home / ".local" / "bin"
    result = machine.run(standalone=True, extra_path=[local_bin])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "    ucx start" in result.stdout
    assert "export PATH" not in result.stdout


def test_someone_elses_ucx_on_path_is_left_alone(machine: Machine) -> None:
    _ready(machine)
    local_bin = machine.home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    _write(local_bin / "ucx", "#!/bin/sh\n# uv tool install's\n")
    result = machine.run(standalone=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (local_bin / "ucx").read_text() == "#!/bin/sh\n# uv tool install's\n"
    assert "left it alone" in result.stdout
    assert f"{machine.home / '.uclone-x' / 'venv'}/bin/ucx start" in result.stdout


def test_a_checkout_gets_no_link(machine: Machine) -> None:
    _ready(machine)
    result = machine.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (machine.home / ".local" / "bin" / "ucx").exists()


def test_a_piped_run_asks_on_the_terminal_and_starts(machine: Machine) -> None:
    """The one-liner, answered yes: Python fetched, models fetched, dashboard started.

    stdin is not a terminal here, exactly as under `curl ... | bash`. Before the
    questions moved to /dev/tty every one of them was skipped on that path, and a
    stock Mac with Python 3.9 ended at "Nothing was installed".
    """
    machine.python("python3", "3.9")
    machine.uv_on_path()
    result = machine.run(tty="y")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = machine.calls()
    assert f"uv venv --python 3.12 {machine.root / 'venv'}" in calls
    assert "ucx install --yes --no-image" in calls
    # Started last, with the terminal -- not curl's pipe -- as its stdin.
    assert calls[-1] == "ucx start stdin=y"


def test_the_readme_one_liner_end_to_end(machine: Machine) -> None:
    """`curl ... | bash`, answered yes: the script comes in on stdin and runs to the end.

    Every other case hands bash a file. This one hands it the script on stdin, so a
    later command that read stdin -- a bare `read`, a prompt, a child inheriting it --
    would swallow the rest of the script and the start would never be reached.
    """
    _ready(machine)
    result = machine.run(tty="y", piped=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = machine.calls()
    assert any(c.endswith("uclone-x[cli,http]") for c in calls)
    assert "ucx install --yes --no-image" in calls
    assert (machine.home / ".local" / "bin" / "ucx").is_symlink()
    assert calls[-1] == "ucx start stdin=y"


def test_a_relative_venv_is_linked_absolutely(machine: Machine) -> None:
    _ready(machine)
    result = machine.run("--venv", "rel-venv", standalone=True)
    assert result.returncode == 0, result.stdout + result.stderr
    link = machine.home / ".local" / "bin" / "ucx"
    assert os.path.isabs(os.readlink(link))
    assert link.resolve().exists()


def test_enter_starts_it_and_skips_the_big_download(machine: Machine) -> None:
    _ready(machine)
    result = machine.run(tty="")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ucx install --yes --no-image" not in machine.calls()
    assert machine.calls()[-1] == "ucx start stdin="


def test_answering_no_leaves_it_stopped(machine: Machine) -> None:
    _ready(machine)
    result = machine.run(tty="n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(c.startswith("ucx start stdin") for c in machine.calls())
    assert f"{machine.root / 'venv'}/bin/ucx start" in result.stdout


def test_no_start_does_not_ask(machine: Machine) -> None:
    _ready(machine)
    result = machine.run("--no-start", tty="y")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(c.startswith("ucx start stdin") for c in machine.calls())
    assert "Start UClone-X now?" not in result.stdout


def test_without_a_terminal_it_never_starts(machine: Machine) -> None:
    """The real /dev/tty, in a session that has none: CI, cron, `--yes` in a script.

    A started server never returns, so a scripted install that reached it would hang.
    """
    _ready(machine)
    result = machine.run("--yes", env_extra={"UCX_INSTALL_TTY": "/dev/tty"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(c.startswith("ucx start stdin") for c in machine.calls())
    assert "Start UClone-X now?" not in result.stdout


def test_dry_run_neither_links_nor_starts(machine: Machine) -> None:
    _ready(machine)
    result = machine.run("--dry-run", tty="y", standalone=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (machine.home / ".local" / "bin" / "ucx").exists()
    assert not any(c.startswith("ucx") for c in machine.calls())


def test_install_sh_parses_and_is_executable() -> None:
    assert subprocess.run([BASH, "-n", str(INSTALL_SH)], check=False).returncode == 0
    assert os.access(INSTALL_SH, os.X_OK)
