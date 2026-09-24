"""Bootstrap helper for zero-barrier local AI setup with pre-flight system inspection."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Final, NamedTuple, cast

import httpx
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm

console = Console()

DEFAULT_OLLAMA_URL = "http://localhost:11434"
RECOMMENDED_8B_MODEL = "qwen3:8b"
LIGHTWEIGHT_MODEL = "qwen3:1.7b"
# qwen3:8b measured 5.5 GB resident under Ollama (2026-09-19, Apple silicon). On an
# 8 GB Mac that leaves ~2.5 GB for macOS and a browser, so 8b is recommended from 16 GB.
MIN_RAM_FOR_8B_GB = 16.0
MIN_DISK_SPACE_GB = 10.0

#: Where the official `ollama.com/install.sh` leaves the binary on macOS when it cannot
#: create its `/usr/local/bin` symlink. Measured on the target Mac mini (macOS 26.5, arm64,
#: 2026-09-21): `/usr/local/bin` is root:wheel and not user-writable, so the installer falls
#: back to `sudo ln`, which under a non-interactive ssh session has no tty and fails. The app
#: bundle itself lands fine — `/Applications` is group-writable for admin — so after a
#: "failed" install the binary is present and simply not on `PATH`. Looking here is what
#: turns that into a working install rather than a false negative.
OLLAMA_APP_BINARIES: Final[tuple[str, ...]] = (
    "/Applications/Ollama.app/Contents/Resources/ollama",
    "~/Applications/Ollama.app/Contents/Resources/ollama",
)

#: The single-file SDXL base checkpoint the image engine can actually load, and the only
#: weights this project downloads. Verified reachable 2026-09-21: HTTP 200 with no token.
SDXL_CHECKPOINT_URL: Final[str] = (
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/"
    "resolve/main/sd_xl_base_1.0.safetensors"
)
SDXL_CHECKPOINT_FILENAME: Final[str] = "sd_xl_base_1.0.safetensors"
#: Where a checkpoint goes when `UCX_IMAGE_CHECKPOINT` names nowhere else. Same directory
#: `DEFAULT_CHECKPOINTS` already searches, so a download lands where the prober looks.
DEFAULT_CHECKPOINT_DIR: Final[str] = "~/ai_models/checkpoints"
#: Quoted in the consent question so the size is on screen *before* the transfer starts
#: (installation-flow.md §4 rule 2). Measured 2026-09-21 as the server's `content-length`.
#: It is never used to decide whether a download succeeded — that comparison is against the
#: length this run's response actually declared, so a stale constant cannot pass a short file.
SDXL_CHECKPOINT_PUBLISHED_BYTES: Final[int] = 6_938_078_334
#: 1 MiB. Large enough that the write loop is not the bottleneck on a 7 GB file, small enough
#: that a cancelled run loses at most a megabyte of a resumable transfer.
DOWNLOAD_CHUNK_BYTES: Final[int] = 1024 * 1024
#: No total timeout: a 7 GB body legitimately takes many minutes. The connect/read timeouts
#: are what must stay finite, so a dead network fails instead of hanging forever.
DOWNLOAD_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(30.0, read=120.0, write=120.0, pool=30.0)


class SystemSpecs(NamedTuple):
    total_ram_gb: float
    free_disk_gb: float
    recommended_model: str


def get_system_specs() -> SystemSpecs:
    """Inspect system RAM and root disk free space to determine suitable model tier."""
    total_ram_bytes: int = 0
    os_type = platform.system()

    if os_type == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            total_ram_bytes = int(out.strip())
        except (subprocess.CalledProcessError, ValueError):
            total_ram_bytes = 8 * (1024**3)
    elif os_type == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        total_ram_bytes = int(parts[1]) * 1024
                        break
        except Exception:
            total_ram_bytes = 8 * (1024**3)
    elif os_type == "Windows":
        try:
            import ctypes

            kernel32: Any = getattr(ctypes, "windll", None)
            if kernel32 is not None:
                c_ulonglong = ctypes.c_ulonglong

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", c_ulonglong),
                        ("ullAvailPhys", c_ulonglong),
                        ("ullTotalPageFile", c_ulonglong),
                        ("ullAvailPageFile", c_ulonglong),
                        ("ullTotalVirtual", c_ulonglong),
                        ("ullAvailVirtual", c_ulonglong),
                        ("ullAvailExtendedVirtual", c_ulonglong),
                    ]

                stat = MemoryStatus()
                stat.dwLength = ctypes.sizeof(MemoryStatus)
                kernel32.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                total_ram_bytes = int(stat.ullTotalPhys)
            else:
                total_ram_bytes = 8 * (1024**3)
        except Exception:
            total_ram_bytes = 8 * (1024**3)
    else:
        total_ram_bytes = 8 * (1024**3)

    total_ram_gb = round(total_ram_bytes / (1024**3), 1)

    # Disk space check on root mount
    try:
        usage = shutil.disk_usage(os.path.abspath(os.sep))
        free_disk_gb = round(usage.free / (1024**3), 1)
    except Exception:
        free_disk_gb = 50.0

    # Model recommendation based on specs
    # Compared rounded: Linux reports MemTotal below the nominal size, so a 16 GB
    # machine reads ~15.6 and would otherwise drop a tier.
    if round(total_ram_gb) >= MIN_RAM_FOR_8B_GB:
        recommended_model = RECOMMENDED_8B_MODEL
    else:
        recommended_model = LIGHTWEIGHT_MODEL

    return SystemSpecs(
        total_ram_gb=total_ram_gb,
        free_disk_gb=free_disk_gb,
        recommended_model=recommended_model,
    )


def is_ollama_reachable(url: str = DEFAULT_OLLAMA_URL, timeout: float = 2.0) -> bool:
    """Check if Ollama server endpoint is reachable and responding."""
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/tags", headers={"User-Agent": "UCX-Bootstrap"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_installed_ollama_models(url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    """Retrieve list of model names already pulled in Ollama."""
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/tags", headers={"User-Agent": "UCX-Bootstrap"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        pass
    return []


def stdin_is_interactive() -> bool:
    """Whether a prompt written now could actually be answered.

    `rich.prompt.Confirm.ask` reads `sys.stdin`. With stdin at `/dev/null` — every
    `ssh host 'ucx …'`, every CI step, every `</dev/null` in an install test — the read
    returns immediately at EOF and `Confirm.ask` raises `EOFError` out of the command.
    Asking is therefore not a neutral act: it is a crash on exactly the machines the
    automated install path runs on, which is why every prompt in this module goes through
    `ask_consent` and every `ask_consent` starts here.

    A closed or replaced stdin raises rather than answering, and that is still "cannot be
    asked", so the exception is the same answer as `False`.
    """
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def ask_consent(
    question: str,
    *,
    interactive: bool = True,
    assume_yes: bool = False,
    default: bool = True,
) -> bool:
    """Get a yes/no for something that costs the user time, disk or money.

    Three outcomes, and only one of them prompts:

    * `assume_yes` — the caller passed `--yes`, which is consent given in advance.
    * no answerable stdin, or `interactive=False` — **decline**, and say what was skipped
      and how to allow it. Not a hang, not an `EOFError`, and not a silent yes: a default
      that spends 7 GB of someone's disk because nobody was there to say no is the
      substituted-default failure P6 forbids.
    * otherwise — ask.

    `KeyboardInterrupt` and `EOFError` are caught even on the prompting path, because a
    terminal can lose its input between the `isatty` check and the read.
    """
    if assume_yes:
        console.print(f"[dim]✔ {escape(question)} — yes (--yes)[/dim]")
        return True
    if not interactive or not stdin_is_interactive():
        console.print(
            f"[yellow]⏭ Skipped:[/yellow] {escape(question)} "
            "[dim](no terminal to answer on; pass --yes to accept in advance)[/dim]"
        )
        return False
    try:
        return Confirm.ask(question, default=default)
    except (EOFError, KeyboardInterrupt):
        console.print("[yellow]⏭ No answer received — skipping.[/yellow]")
        return False


def ollama_binary() -> str | None:
    """The `ollama` executable to run, or None when this machine has none.

    `PATH` first. Then the macOS app bundle, because the official installer's symlink step
    is the one part of it that needs a password: it writes `/usr/local/bin/ollama`, falls
    back to `sudo ln` when that directory is not writable, and has no tty to ask on under
    ssh. The app is installed correctly in that case and only the symlink is missing, so
    treating "not on PATH" as "not installed" throws away a working install.
    """
    found = shutil.which("ollama")
    if found is not None:
        return found
    for candidate in OLLAMA_APP_BINARIES:
        path = os.path.expanduser(candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def ollama_version(binary: str | None = None) -> str | None:
    """Run the binary and return what it says, or None if it will not run.

    The installer's exit code is not the question — on the measured Mac mini it is nonzero
    for a `sudo ln` that failed while the app installed perfectly well, and it could equally
    be zero for an app that cannot execute. The only honest verification is executing the
    thing and reading its answer (P6).
    """
    resolved = binary if binary is not None else ollama_binary()
    if resolved is None:
        return None
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return (completed.stdout or completed.stderr).strip() or None


def is_ollama_installed() -> bool:
    """Whether an `ollama` executable exists, on PATH or in the macOS app bundle."""
    return ollama_binary() is not None


def ollama_child_env() -> dict[str, str]:
    """The environment handed to every `ollama` subprocess.

    A copy of this process's, unmodified. It exists as a named function so that the
    preservation is a property with a test rather than an accident of not passing `env=`:
    an install test points `OLLAMA_MODELS` at a throwaway directory, and a child that was
    given a scrubbed environment would write the weights into the user's real
    `~/.ollama` instead — several gigabytes, in the one place the test was avoiding.
    """
    return dict(os.environ)


def start_ollama_daemon(endpoint: str | None = None) -> bool:
    """Attempt to start ollama background service, and wait for the endpoint it serves.

    `endpoint` defaults to `ollama_endpoint()`, resolved here rather than in the signature
    so a variable exported after import still counts. The readiness loop polled a
    hard-coded `localhost:11434` regardless of what the caller had configured, which fails
    in both directions: a daemon that came up exactly where `OLLAMA_HOST` said reads as a
    failed start, and on a machine already running something on the default port, a start
    that never succeeded reads as ready because a *different* daemon answered.
    """
    target = endpoint if endpoint is not None else ollama_endpoint()
    binary = ollama_binary()
    if binary is None:
        return False
    try:
        subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=ollama_child_env(),
        )
        # Wait up to 6 seconds for the endpoint to become available
        for _ in range(12):
            time.sleep(0.5)
            if is_ollama_reachable(target):
                return True
    except Exception:
        return False
    return False


def pull_ollama_model(model: str) -> bool:
    """Pull specified model using 'ollama pull'.

    Runs the *resolved* binary. A bare `"ollama"` is only correct when the symlink step of
    the installer succeeded, and on the machine this path exists for it does not.
    """
    binary = ollama_binary()
    if binary is None:
        console.print("[bold red]✖ No ollama executable to pull with.[/bold red]")
        return False
    console.print(f"[bold cyan]⬇️ Downloading model '{model}' via Ollama...[/bold cyan]")
    try:
        res = subprocess.run(
            [binary, "pull", model],
            check=False,
            stdin=subprocess.DEVNULL,
            env=ollama_child_env(),
        )
        return res.returncode == 0
    except Exception as e:
        console.print(f"[bold red]✖ Failed to pull model {model}: {e}[/bold red]")
        return False


def ollama_installer_env() -> dict[str, str]:
    """The environment `ollama.com/install.sh` is run under on the automated path.

    Two deliberate differences from this process's environment:

    * `OLLAMA_NO_START=1` — the installer ends with `open -a Ollama --args hidden`, a GUI
      launch. Over ssh there is no session to launch into, and the app is not what we want
      running anyway: `start_ollama_daemon` runs `ollama serve` itself.
    * `SUDO_ASKPASS` removed — the installer's `/usr/local/bin` symlink falls back to
      `sudo`. With no askpass helper and no tty, `sudo` refuses immediately instead of
      waiting on a password that will never arrive. The failed symlink is *expected* here,
      not an error: `ollama_binary` finds the app bundle afterwards.

    `OLLAMA_MODELS` and everything else are carried through unchanged.
    """
    env = dict(os.environ)
    env["OLLAMA_NO_START"] = "1"
    env.pop("SUDO_ASKPASS", None)
    return env


def install_ollama_platform() -> bool:
    """Install Ollama, then verify by running it — never by trusting the exit code.

    On macOS without Homebrew this used to return False without trying anything, which is
    the stock Mac. It now takes the same official installer Linux does, under an
    environment that neither prompts for a password nor opens a GUI, and then asks the
    binary for its version. That last step is the whole point: the installer exits nonzero
    on the measured Mac mini (its `sudo ln` fails) while leaving a perfectly usable app
    behind, so exit code and reality disagree in both directions and only one of them can
    be measured.
    """
    os_type = platform.system()
    if os_type == "Darwin" and shutil.which("brew"):
        console.print("[bold cyan]📦 Installing Ollama via Homebrew...[/bold cyan]")
        subprocess.run(["brew", "install", "ollama"], check=False, stdin=subprocess.DEVNULL)
    elif os_type in ("Darwin", "Linux"):
        console.print("[bold cyan]📦 Installing Ollama via official install script...[/bold cyan]")
        try:
            subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
                check=False,
                stdin=subprocess.DEVNULL,
                env=ollama_installer_env(),
                timeout=900,
            )
        except subprocess.TimeoutExpired:
            console.print(
                "[bold red]✖ The Ollama installer did not finish in 15 minutes.[/bold red]"
            )
            return False
        except Exception as exc:
            console.print(f"[bold red]✖ Could not run the Ollama installer: {exc}[/bold red]")
            return False
    else:
        console.print(
            f"[bold red]✖ No automated Ollama install for {escape(os_type)}. "
            "Download it from https://ollama.com and run this again.[/bold red]"
        )
        return False

    binary = ollama_binary()
    version = ollama_version(binary)
    if binary is None or version is None:
        console.print(
            "[bold red]✖ Ollama is still not runnable after the install.[/bold red] "
            "Looked on PATH and at " + escape(", ".join(OLLAMA_APP_BINARIES)) + "."
        )
        return False
    console.print(f"[bold green]✔ Ollama ready:[/bold green] {escape(binary)} ({escape(version)})")
    return True


class LlmSetup(NamedTuple):
    """What the local LLM half of the install ended up being, and why."""

    ready: bool
    model: str
    endpoint: str
    #: Empty when ready; a short hyphenated token otherwise, for the summary line.
    reason: str = ""

    def summary_line(self) -> str:
        """The one line a shell script greps for."""
        if self.ready:
            return f"setup llm: ready model={self.model} endpoint={self.endpoint}"
        return f"setup llm: unavailable reason={self.reason or 'unknown'}"


def ollama_endpoint() -> str:
    """The Ollama address this machine is configured for, resolved the one documented way.

    This used to read `OLLAMA_BASE_URL` alone — a third answer to a question the connector
    layer already answered, and the only one of the three that ignored `OLLAMA_HOST`. That
    omission is not cosmetic: `OLLAMA_HOST` is the variable Ollama's own tooling sets, and
    the one `scripts/verify_online_install.sh` uses to point an install at an isolated
    daemon on a free port. Resolving the install against `localhost:11434` while every turn
    afterwards resolved `OLLAMA_HOST` produced both halves of the same lie — `ollama-not-
    running` about a daemon that was running, and on a machine with a daemon on the default
    port, `ready` about a daemon the run would never speak to.

    Imported inside the function because this module is on the `ucx` startup path and the
    connector package is not; `image_checkpoint_target` defers its import for the same
    reason.
    """
    from uclone_x.llm.connectors.ollama import resolve_ollama_base_url

    return resolve_ollama_base_url()


def setup_local_llm(interactive: bool = True, assume_yes: bool = False) -> LlmSetup:
    """Ensure local Ollama and default profile are ready with pre-flight hardware inspection.

    This is `ensure_local_profile` with its reason kept instead of thrown away, so that a
    caller printing a machine-readable verdict can say *which* step failed rather than
    reporting a bare false.
    """
    specs = get_system_specs()
    endpoint = ollama_endpoint()

    # If Ollama is already running and has the recommended model, proceed immediately
    if is_ollama_reachable(endpoint):
        models = get_installed_ollama_models(endpoint)
        # If recommended or any valid qwen/gemma model exists
        for m in models:
            if specs.recommended_model in m:
                return LlmSetup(True, m, endpoint)
        if models:
            # Use already pulled model if available
            return LlmSetup(True, models[0], endpoint)

    # Pre-flight specs presentation
    console.print("\n[bold cyan]🖥️  UClone-X Local AI Pre-Flight Inspection[/bold cyan]")
    console.print(f"  • System Memory: [bold]{specs.total_ram_gb} GB[/bold]")
    console.print(f"  • Free Disk Space: [bold]{specs.free_disk_gb} GB[/bold]")
    console.print(
        f"  • Recommended Default Model: [bold green]{specs.recommended_model}[/bold green]"
    )

    if specs.free_disk_gb < MIN_DISK_SPACE_GB:
        console.print(
            f"[bold yellow]⚠️ Warning: Low disk space ({specs.free_disk_gb} GB available). "
            f"At least {MIN_DISK_SPACE_GB} GB is recommended.[/bold yellow]"
        )

    if not ask_consent(
        f"Set up local private AI with [bold green]{specs.recommended_model}[/bold green]?",
        interactive=interactive,
        assume_yes=assume_yes,
        default=True,
    ):
        console.print(
            "[yellow]Local AI setup skipped. You can configure cloud API keys in Settings.[/yellow]"
        )
        return LlmSetup(False, specs.recommended_model, endpoint, "declined")

    # 1. Install Ollama if missing
    if not is_ollama_installed() and not is_ollama_reachable(endpoint):
        console.print(
            "[bold yellow]Ollama binary not found on PATH. Attempting install...[/bold yellow]"
        )
        if not install_ollama_platform():
            console.print(
                "[bold red]Could not automatically install Ollama. "
                "Please download from https://ollama.com and run again.[/bold red]"
            )
            return LlmSetup(False, specs.recommended_model, endpoint, "ollama-install-failed")

    # 2. Start daemon if not reachable
    if not is_ollama_reachable(endpoint):
        console.print("[cyan]Starting Ollama local service...[/cyan]")
        # The endpoint this run resolved, not the daemon's default: the probe above and the
        # wait below must be asking about the same address or the verdict is about a
        # different process than the one that was started.
        if not start_ollama_daemon(endpoint):
            console.print(
                "[bold red]Could not start Ollama service. Please run 'ollama serve' manually.[/bold red]"
            )
            return LlmSetup(False, specs.recommended_model, endpoint, "ollama-not-running")

    # 3. Ensure recommended model is pulled
    models = get_installed_ollama_models(endpoint)
    model_already_present = any(specs.recommended_model in m for m in models)
    if not model_already_present:
        success = pull_ollama_model(specs.recommended_model)
        if not success:
            return LlmSetup(False, specs.recommended_model, endpoint, "model-pull-failed")

    return LlmSetup(True, specs.recommended_model, endpoint)


def ensure_local_profile(interactive: bool = True, assume_yes: bool = False) -> tuple[bool, str]:
    """Ensure local Ollama and default profile are ready.

    Returns:
        (success: bool, selected_model: str)
    """
    result = setup_local_llm(interactive=interactive, assume_yes=assume_yes)
    return result.ready, result.model


class ImageEngineReport(NamedTuple):
    """What each image engine offers right now, inspected rather than assumed (#1095)."""

    remote_url: str | None
    remote_alive: bool
    comfy_url: str
    comfy_alive: bool
    dependency_problems: tuple[str, ...]
    checkpoint: str | None

    @property
    def ready(self) -> bool:
        """Whether any engine can actually generate an image.

        A configured `UCX_IMAGE_REMOTE_URL` is an address, not a worker. It counts only
        once `/health` has answered, the same probe `RemoteCudaImageEngine.is_available`
        makes before the dispatcher will use it; reporting ready from the variable alone
        promised an engine that then refused to generate (P6).
        """
        return self.remote_ready or self.comfy_alive or self.in_process_ready

    @property
    def remote_ready(self) -> bool:
        """A remote worker that is both configured and answering."""
        return bool(self.remote_url) and self.remote_alive

    @property
    def dependencies_ok(self) -> bool:
        """Whether every package the in-process engine imports is present and new enough."""
        return not self.dependency_problems

    @property
    def in_process_ready(self) -> bool:
        """The daemon-free baseline: every import is satisfied and a checkpoint file exists."""
        return self.dependencies_ok and self.checkpoint is not None

    @property
    def engine(self) -> str:
        """The engine that would be selected, matching `ImagePipelineDispatcher`'s order."""
        if self.remote_ready:
            return "remote-cuda"
        if self.comfy_alive:
            return "comfyui-local"
        if self.in_process_ready:
            return "diffusers-sdxl"
        return "none"


def probe_image_engines() -> ImageEngineReport:
    """Inspect all three image engines without installing, starting or downloading anything.

    Every field is read from this machine as it is: two HTTP probes -- the remote worker's
    `/health` and a ComfyUI daemon somebody else started -- an import, and a file on disk.
    Nothing here can report ready for an engine that would then fail to produce an image
    (P6), which is why the remote worker is probed rather than inferred from its variable.
    """
    import asyncio

    from uclone_x.tools.builtin.image import (
        REMOTE_URL_ENV,
        ComfyUIImageEngine,
        LocalDiffusersImageEngine,
        RemoteCudaImageEngine,
        in_process_dependency_problems,
    )

    remote_url = os.getenv(REMOTE_URL_ENV) or os.getenv("UCX_MEDIA_REMOTE_URL")
    try:
        # Built inside the `try` so that a client which rejects the configured address
        # reads as "not there", the same as a refused connection. No address, no request.
        remote_alive = (
            asyncio.run(RemoteCudaImageEngine(base_url=remote_url).is_available())
            if remote_url
            else False
        )
    except Exception:
        remote_alive = False
    comfy = ComfyUIImageEngine()
    try:
        comfy_alive = asyncio.run(comfy.is_available())
    except Exception:
        comfy_alive = False

    return ImageEngineReport(
        remote_url=remote_url or None,
        remote_alive=remote_alive,
        comfy_url=comfy.base_url,
        comfy_alive=comfy_alive,
        dependency_problems=tuple(
            problem.describe() for problem in in_process_dependency_problems()
        ),
        checkpoint=LocalDiffusersImageEngine().resolve_checkpoint(),
    )


#: What to type when this environment has neither installer (#971 acceptance criterion 2).
#:
#: The criterion is two halves, and only the first was met here: no subprocess is launched
#: (the only one available is the one already known to fail), but the refusal named nothing
#: to run next. A user who answered "yes" to the install offer was told what their machine
#: lacks and left with no command -- the same P6 dead end as the original silence, one step
#: further along. `core/environment_install.py` already names both remedies on its own
#: no-installer path; this keeps the `ucx start` path saying the same thing.
NO_INSTALLER_REMEDY: Final[str] = (
    "Install uv (https://docs.astral.sh/uv/), "
    "or seed pip with `{python} -m ensurepip --upgrade`, "
    "then run `ucx start` again."
)


def diffusers_install_command() -> list[str] | None:
    """The command that installs the in-process image packages into this process (#971).

    Which installer, in which order, is not decided here: this delegates to
    `core.environment_install.installer_command`, which is the one place that order and
    its reason live (#1278). Until then this path preferred pip while the shared path
    preferred uv, so a pip-seeded venv with uv on `PATH` installed the image packages
    with a different resolver and a different cache from every other in-process install.

    What *is* decided here is the argument list. All of the requirements are named, with
    their floors: `diffusers` pulls in neither `torch` nor `transformers`, and an install
    that stops at `diffusers` leaves the engine unable to generate (#1095).
    """
    from uclone_x.core.environment_install import installer_command
    from uclone_x.tools.builtin.image import IN_PROCESS_REQUIREMENTS

    packages = [requirement for _, requirement, _ in IN_PROCESS_REQUIREMENTS]
    return installer_command(*packages)


def install_diffusers() -> bool:
    """Install the in-process image packages and report honestly whether it worked.

    Exit 0 is not the question — the question is whether every requirement is now importable
    at its floor in *this* interpreter, so the answer is re-measured rather than inferred
    from a status code.

    The re-measurement is `in_process_dependency_problems`, and it is deliberately not
    `environment_install.still_missing`, which is presence-only: an environment holding
    `diffusers 0.28` satisfies `still_missing` and violates the `>=0.31.0` floor that
    `IN_PROCESS_REQUIREMENTS` declares. Consolidating this path onto the shared installer
    moved the *order* (#1278) and had to leave the floor check standing here.
    """
    import importlib

    from uclone_x.tools.builtin.image import in_process_dependency_problems

    command = diffusers_install_command()
    if command is None:
        console.print(
            "[bold red]✖ Cannot install the image packages: this environment has neither pip "
            "nor uv.[/bold red] " + NO_INSTALLER_REMEDY.format(python=escape(sys.executable))
        )
        return False

    console.print(
        f"[bold cyan]📦 Installing in-process image dependencies:[/bold cyan] {command[0]}"
    )
    try:
        returncode = subprocess.run(command, check=False).returncode
    except Exception as exc:
        console.print(f"[bold red]✖ Installing the image packages failed: {exc}[/bold red]")
        return False

    importlib.invalidate_caches()
    problems = [problem.describe() for problem in in_process_dependency_problems()]
    if returncode == 0 and not problems:
        return True
    console.print(
        "[bold red]✖ The image packages are still not usable[/bold red] "
        f"(exit code {returncode}): {'; '.join(problems) if problems else 'unknown reason'}. "
        f"Run manually: {' '.join(command)}"
    )
    return False


def image_checkpoint_target() -> Path:
    """Where a downloaded checkpoint goes on this machine.

    The same question `LocalDiffusersImageEngine.checkpoint_resolution` asks, answered with
    the same expansion helper, so a file written here is a file the prober then finds.
    `UCX_IMAGE_CHECKPOINT` wins when set — including when it names a directory that does not
    exist yet, which the download creates, because an install test pointing the variable at
    a throwaway directory is the case this is for.
    """
    from uclone_x.tools.builtin.image import IMAGE_CHECKPOINT_ENV, expand_checkpoint_path

    configured = os.getenv(IMAGE_CHECKPOINT_ENV)
    if configured:
        return Path(expand_checkpoint_path(configured))
    return Path(os.path.expanduser(DEFAULT_CHECKPOINT_DIR)) / SDXL_CHECKPOINT_FILENAME


class DownloadOutcome(NamedTuple):
    """What a checkpoint download did, with a short reason when it did not finish."""

    ok: bool
    path: str | None
    reason: str = ""


def _describe_gb(num_bytes: int) -> str:
    """Bytes as the GB figure a person reads before agreeing to spend their disk on it."""
    return f"{num_bytes / 1_000_000_000:.2f} GB"


#: A safetensors header is the JSON that describes the tensors; SDXL base's is a few hundred
#: kilobytes. 100 MB is far above anything legitimate and far below a size worth reading into
#: memory, which is the whole point of the bound: the length comes from the *file's own*
#: first eight bytes, so an HTML page or a run of zeros can claim 2**63 and a check that
#: trusted it would try to allocate that before deciding the file was junk.
MAX_SAFETENSORS_HEADER_BYTES: Final[int] = 100 * 1024 * 1024


def _install_part(part: Path, destination: Path) -> None:
    """Move a verified `.part` onto the name the prober searches.

    Both callers reach here only after `_safetensors_complete` said yes, and keeping the
    rename in one place is what lets that be checked by reading rather than by trusting two
    call sites to stay in step. `replace` and not `rename`: the destination may exist when a
    previous run left an unusable file there, and `rename` refuses that on Windows.
    """
    part.replace(destination)


def _safetensors_complete(path: Path) -> tuple[bool, str]:
    """Whether `path` is a whole safetensors file, and a short reason when it is not.

    The format describes its own length, which is what makes completeness checkable without
    a constant: eight bytes of little-endian `uint64` header length `n`, then `n` bytes of
    UTF-8 JSON, then the tensor data. Every entry in that JSON carries
    `data_offsets: [start, end]` relative to the start of the data block, so a complete file
    is exactly `8 + n + max(end)` bytes — a figure derived from the file rather than
    advertised about it.

    This is the check the declared-length comparison cannot make, in three cases it does not
    reach. A response with no `content-length` is not compared to anything at all. A
    `content-length` is supplied by the same response as the body, so a captive portal or a
    corporate proxy answering `200` with an HTML error page passes it truthfully. And a file
    already sitting at the destination — an interrupted `curl -O` — has no response behind
    it to compare with. In all three the file itself is the only source that cannot be
    forged by whatever produced it.

    Never raises. Both callers are deciding whether an install may claim to be ready, and a
    traceback out of either says strictly less to the person reading the log than
    `truncated` does.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                return False, "not-safetensors"
            header_length = int.from_bytes(prefix, "little")
            if header_length <= 0 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
                return False, "not-safetensors"
            if 8 + header_length > size:
                return False, "truncated"
            raw_header = handle.read(header_length)
    except OSError:
        return False, "unreadable"
    if len(raw_header) != header_length:
        return False, "truncated"

    try:
        parsed: object = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False, "not-safetensors"
    if not isinstance(parsed, dict):
        return False, "not-safetensors"
    # Everything below comes from a file this function exists to distrust, so it is read as
    # `object` and narrowed one step at a time rather than taken at the shape it claims.
    header = cast("dict[str, object]", parsed)

    data_end = 0
    for name, entry in header.items():
        # `__metadata__` is the one key that is not a tensor: it holds free-form strings and
        # has no `data_offsets`, so reading it as an entry would reject every file that
        # carries one — which the real SDXL checkpoint does.
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            return False, "not-safetensors"
        offsets = cast("dict[str, object]", entry).get("data_offsets")
        if not isinstance(offsets, list):
            return False, "not-safetensors"
        pair = cast("list[object]", offsets)
        # Both lengths are checked: the second counts only the values that survived being
        # narrowed to a non-negative int, so a pair of strings and a pair of negatives are
        # rejected here rather than reaching `max` as something that is not a byte count.
        bounds = [value for value in pair if isinstance(value, int) and value >= 0]
        if len(pair) != 2 or len(bounds) != 2:
            return False, "not-safetensors"
        data_end = max(data_end, bounds[1])
    if data_end == 0:
        # Well-formed JSON naming no tensors. A checkpoint with nothing in it loads and
        # generates nothing, so it is not a file this install may report as ready.
        return False, "no-tensors"

    expected = 8 + header_length + data_end
    if size < expected:
        return False, "truncated"
    if size > expected:
        # Longer than its own header says it is: a whole body appended to a partial one, or
        # a second download concatenated onto the first.
        return False, "trailing-bytes"
    return True, ""


def _finish_or_discard_complete_part(part: Path, destination: Path) -> DownloadOutcome:
    """Answer an HTTP 416 by looking at the `.part` the resume offset came from.

    416 says the offset is at or past the end of the file, and a `.part` at exactly full
    length is the ordinary way to get there: the bytes all arrived and the process died
    before `part.replace`. There are only two honest answers, and both of them end the
    loop that returning `http-416` every run could not.
    """
    complete, reason = _safetensors_complete(part) if part.exists() else (False, "no-part")
    if complete:
        _install_part(part, destination)
        console.print(
            "[bold green]✔ The partial download was already complete[/bold green] — "
            f"installed as [cyan]{escape(str(destination))}[/cyan]."
        )
        return DownloadOutcome(True, str(destination))
    try:
        part.unlink(missing_ok=True)
    except OSError:
        # Removing it is a courtesy; failing to is not worse than the 416 that got us here,
        # and the reason below still tells the reader the run has to start over.
        pass
    console.print(
        f"[bold red]✖ The server refused to resume from where the partial file ends "
        f"({escape(reason)}).[/bold red] The partial file has been discarded; run this "
        "again to download the checkpoint from the beginning."
    )
    return DownloadOutcome(False, None, "partial-discarded")


def download_image_checkpoint(
    target: Path | None = None,
    *,
    url: str = SDXL_CHECKPOINT_URL,
    interactive: bool = True,
    assume_yes: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> DownloadOutcome:
    """Fetch the SDXL base checkpoint, resumably, and refuse to claim a partial file.

    The half of the install that did not exist. `install.sh` installs uv, Python and the
    package; `ensure_local_image_pipeline` installed `diffusers`; nothing anywhere fetched
    weights, so a beginner who ran every documented step still could not generate an image.

    Four properties, each of which is a way this can lie and does not:

    * **Consent first.** The size is on screen before a byte moves, and no terminal means
      no, never yes (installation-flow.md §4 rule 2).
    * **`.part` until finished.** The bytes land in `<name>.part` and are renamed only after
      the size check passes. An interrupted run therefore never leaves a file at the path
      the prober searches — which would read as a complete checkpoint and fail at load time,
      hours later, as a corrupt-safetensors traceback.
    * **Resume.** An existing `.part` is continued with a `Range` request rather than
      restarted; on a 7 GB file over a home connection that is the difference between a
      recoverable interruption and an unusable command. A server that ignores `Range` and
      answers `200` is detected and the partial file is discarded, because appending a
      whole body to a partial one produces a plausible-sized corrupt file.
    * **The finished size is checked against what the server said**, not against a constant
      in this file, and a mismatch keeps the `.part` and fails loudly. A truncated transfer
      that renames itself is the precise failure the `.part` discipline exists to prevent,
      so it must not be reachable by a short read either.
    * **And then against the file itself.** `_safetensors_complete` decides the last word,
      because the declared length is absent from a chunked response and truthful about an
      HTML error page, and because a file already at the destination has no response behind
      it at all. Both the early return and the rename go through it.
    """
    destination = target if target is not None else image_checkpoint_target()
    if destination.exists():
        # Existence was the whole readiness test here, so a 0-byte file, a half-finished
        # `curl -O`, or a saved error page at this path made the install skip the download
        # and report ready — and the failure surfaced hours later as a corrupt-safetensors
        # traceback from a generation the user had by then attributed to something else.
        complete, reason = _safetensors_complete(destination)
        if complete:
            return DownloadOutcome(True, str(destination))
        console.print(
            f"[yellow]The file already at {escape(str(destination))} is not a usable "
            f"checkpoint ({reason}); fetching it again.[/yellow]"
        )

    part = destination.with_name(destination.name + ".part")
    resume_from = part.stat().st_size if part.exists() else 0

    prompt = (
        f"Download the SDXL base image checkpoint (~{_describe_gb(SDXL_CHECKPOINT_PUBLISHED_BYTES)}) "
        f"to {escape(str(destination))}?"
    )
    if resume_from:
        prompt = (
            f"Resume the SDXL base image checkpoint download "
            f"({_describe_gb(resume_from)} of "
            f"~{_describe_gb(SDXL_CHECKPOINT_PUBLISHED_BYTES)} already fetched) "
            f"to {escape(str(destination))}?"
        )
    if not ask_consent(prompt, interactive=interactive, assume_yes=assume_yes, default=False):
        return DownloadOutcome(False, None, "declined")

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        console.print(
            f"[bold red]✖ Cannot create {escape(str(destination.parent))}: {exc}[/bold red]"
        )
        return DownloadOutcome(False, None, "destination-unwritable")

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    expected_total: int | None = None
    try:
        with httpx.Client(
            transport=transport, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT
        ) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 416:
                    # "Range Not Satisfiable": the offset we asked to resume from is at or
                    # past the end of the file. The way that happens here is an interruption
                    # between the last chunk write and the rename, which leaves a `.part`
                    # at full length — and the generic branch below would then answer
                    # `http-416` on this run and on every run after it, because nothing it
                    # does changes the offset that caused it. Deciding it against the file
                    # instead either finishes the install or clears the way for a fresh one.
                    return _finish_or_discard_complete_part(part, destination)
                if response.status_code >= 400:
                    console.print(
                        f"[bold red]✖ The checkpoint server answered HTTP "
                        f"{response.status_code}.[/bold red] Tried {escape(url)}"
                    )
                    return DownloadOutcome(False, None, f"http-{response.status_code}")

                declared = response.headers.get("content-length")
                body_bytes = int(declared) if declared is not None and declared.isdigit() else None

                if resume_from and response.status_code != 206:
                    # The server ignored the Range header and is sending the whole file.
                    console.print(
                        "[yellow]The server did not honour the resume request; "
                        "starting the download again from the beginning.[/yellow]"
                    )
                    resume_from = 0
                expected_total = None if body_bytes is None else resume_from + body_bytes

                mode = "ab" if resume_from else "wb"
                written = resume_from
                milestone = 0
                console.print(
                    f"[bold cyan]⬇️ Downloading the SDXL base checkpoint[/bold cyan] → "
                    f"[cyan]{escape(str(part))}[/cyan]"
                )
                with part.open(mode) as handle:
                    for chunk in response.iter_bytes(DOWNLOAD_CHUNK_BYTES):
                        handle.write(chunk)
                        written += len(chunk)
                        milestone = _print_progress(written, expected_total, milestone)
    except httpx.HTTPError as exc:
        console.print(
            f"[bold red]✖ Could not reach the checkpoint server:[/bold red] {escape(str(exc))}. "
            "Check the network and run this again — the partial file is kept and resumed."
        )
        return DownloadOutcome(False, None, "network-error")
    except OSError as exc:
        console.print(f"[bold red]✖ Writing the checkpoint failed:[/bold red] {escape(str(exc))}")
        return DownloadOutcome(False, None, "write-error")

    actual = part.stat().st_size
    if expected_total is not None and actual != expected_total:
        console.print(
            f"[bold red]✖ The download is the wrong size:[/bold red] got {actual} bytes, the "
            f"server declared {expected_total}. Keeping [cyan]{escape(str(part))}[/cyan] so the "
            "next run can resume; nothing was installed."
        )
        return DownloadOutcome(False, None, "size-mismatch")

    # The declared-length check above stays: it is cheaper, and it names a short transfer as
    # a short transfer. This one is what catches the cases it cannot see — no
    # `content-length` at all, or a truthful one attached to a body that is not a
    # checkpoint.
    complete, reason = _safetensors_complete(part)
    if not complete:
        console.print(
            f"[bold red]✖ What arrived is not a complete checkpoint ({escape(reason)}).[/bold red] "
            f"Keeping [cyan]{escape(str(part))}[/cyan] so the next run can try to resume it; if "
            "that run reports the same thing, delete that file and run this again on a network "
            "that is not intercepting the download."
        )
        return DownloadOutcome(False, None, reason)

    _install_part(part, destination)
    console.print(
        f"[bold green]✔ Checkpoint ready:[/bold green] [cyan]{escape(str(destination))}[/cyan] "
        f"({_describe_gb(actual)})"
    )
    return DownloadOutcome(True, str(destination))


def _print_progress(done: int, total: int | None, last_milestone: int) -> int:
    """Print a percentage line every 5%, and return the milestone reached.

    Plain lines rather than a `rich` progress bar: the run this matters for is a
    non-interactive ssh install whose output is a log file, where a bar redraws into
    thousands of control sequences and a reader learns nothing.
    """
    if total is None or total <= 0:
        return last_milestone
    milestone = min(100, int(done * 100 / total) // 5 * 5)
    if milestone > last_milestone:
        console.print(f"[dim]  … {milestone}% ({_describe_gb(done)} / {_describe_gb(total)})[/dim]")
        return milestone
    return last_milestone


class ImageSetup(NamedTuple):
    """What the image half of the install ended up being, and why."""

    ready: bool
    engine: str
    checkpoint: str | None = None
    reason: str = ""

    def summary_line(self) -> str:
        """The one line a shell script greps for.

        `checkpoint=-` for the two engines that have no local file — a remote CUDA worker
        and someone else's ComfyUI daemon both hold their own weights. The key is still
        printed so the line's shape does not change with the engine.
        """
        if self.ready:
            return f"setup image: ready engine={self.engine} checkpoint={self.checkpoint or '-'}"
        return f"setup image: unavailable reason={self.reason or 'unknown'}"


def setup_local_image(
    interactive: bool = True,
    assume_yes: bool = False,
    *,
    allow_download: bool = True,
    transport: httpx.BaseTransport | None = None,
) -> ImageSetup:
    """Report which image engine is ready, and install or download the missing halves (#1095).

    The beginner path is the in-process one, and it has two requirements that fail
    independently: the `diffusers`/`torch`/`transformers` packages, and a single-file SDXL
    checkpoint. Until now only the first was ever offered — with no checkpoint on disk this
    printed "place one in ~/ai_models/checkpoints/" and returned, so the documented install
    ended with a user who could not generate an image. Both are now offered, each behind its
    own confirmation with its own size.

    A ComfyUI daemon is *detected* and used when it is there, and never installed or started
    from here — on a machine that runs one for another project, seizing its port or its
    lifetime is not ours to do.
    """
    report = probe_image_engines()

    if report.remote_ready:
        console.print(
            f"[bold green]✔ Remote CUDA GPU image worker answering at[/bold green] "
            f"[cyan]{report.remote_url}[/cyan]"
        )
        return ImageSetup(True, "remote-cuda")

    if report.remote_url:
        console.print(
            f"[bold yellow]⚠️ Remote CUDA GPU image worker at[/bold yellow] "
            f"[cyan]{report.remote_url}[/cyan] did not answer; falling back to the local "
            "engines."
        )

    if report.comfy_alive:
        console.print(
            f"[bold green]✔ Local ComfyUI detected at[/bold green] [cyan]{report.comfy_url}[/cyan]"
            " — using it for image generation."
        )
        return ImageSetup(True, "comfyui-local")

    if report.in_process_ready:
        console.print(
            "[bold green]✔ In-process image engine ready[/bold green] "
            f"([cyan]{Path(report.checkpoint or '').name}[/cyan], no daemon required)."
        )
        return ImageSetup(True, "diffusers-sdxl", report.checkpoint)

    console.print("[bold cyan]🎨 Local image generation[/bold cyan]")
    checkpoint = report.checkpoint
    if checkpoint is None:
        if not allow_download:
            console.print(
                "[bold yellow]⚠️ No image checkpoint found.[/bold yellow] Place a single-file "
                "SDXL checkpoint in [cyan]~/ai_models/checkpoints/[/cyan] or set "
                "[cyan]UCX_IMAGE_CHECKPOINT[/cyan]. Run `ucx media status` for details."
            )
            return ImageSetup(False, "none", None, "no-checkpoint")
        outcome = download_image_checkpoint(
            interactive=interactive, assume_yes=assume_yes, transport=transport
        )
        if not outcome.ok or outcome.path is None:
            console.print(
                "[dim]No image checkpoint. `ucx media status` explains what the engine "
                "is looking for.[/dim]"
            )
            return ImageSetup(False, "none", None, outcome.reason or "no-checkpoint")
        checkpoint = outcome.path
    else:
        console.print(f"  • Checkpoint found: [cyan]{escape(checkpoint)}[/cyan]")

    if report.dependencies_ok:
        console.print("[bold green]✔ In-process image engine ready.[/bold green]")
        return ImageSetup(True, "diffusers-sdxl", checkpoint)

    for problem in report.dependency_problems:
        console.print(f"  • Missing: {problem} [dim](no model download needed)[/dim]")
    if not ask_consent(
        "Install the in-process image packages now?",
        interactive=interactive,
        assume_yes=assume_yes,
        default=True,
    ):
        console.print(
            "[dim]Local image generation skipped. Enable it later with `ucx start`.[/dim]"
        )
        return ImageSetup(False, "none", checkpoint, "declined")

    if not install_diffusers():
        return ImageSetup(False, "none", checkpoint, "package-install-failed")
    console.print("[bold green]✔ In-process image engine ready.[/bold green]")
    return ImageSetup(True, "diffusers-sdxl", checkpoint)


def ensure_local_image_pipeline(
    interactive: bool = True, assume_yes: bool = False
) -> tuple[bool, str]:
    """Report which image engine is ready, installing the missing halves.

    Returns:
        (ready: bool, engine: str)
    """
    result = setup_local_image(interactive=interactive, assume_yes=assume_yes)
    return result.ready, result.engine


class LocalSetupResult(NamedTuple):
    """The verdict of a whole `ucx install` run: one entry per requested part."""

    llm: LlmSetup | None = None
    image: ImageSetup | None = None

    @property
    def ok(self) -> bool:
        """True only when every part that was *asked for* is actually ready.

        A part that was not requested cannot make the run fail, and a part that was
        requested cannot be excused by the other one succeeding — which is what makes the
        exit code usable as the install test's assertion.
        """
        return all(part.ready for part in (self.llm, self.image) if part is not None)

    def summary_lines(self) -> list[str]:
        """The machine-readable last lines, in a fixed order, for the parts that ran.

        A part that was switched off with `--no-llm` / `--no-image` prints no line at all
        rather than a third status word: a grep for `setup llm: ready` then correctly
        finds nothing, instead of matching a line that says something was skipped.
        """
        return [part.summary_line() for part in (self.llm, self.image) if part is not None]


def run_local_setup(
    *,
    llm: bool = True,
    image: bool = True,
    interactive: bool = True,
    assume_yes: bool = False,
    allow_download: bool = True,
    transport: httpx.BaseTransport | None = None,
) -> LocalSetupResult:
    """Run the local-model setup steps and return what each one achieved.

    The single code path behind both `ucx install` and `ucx start`. `start` had its own
    copy of this sequence inline; two copies of "prepare the local models" is how one of
    them silently stops matching the other.
    """
    return LocalSetupResult(
        llm=(setup_local_llm(interactive=interactive, assume_yes=assume_yes) if llm else None),
        image=(
            setup_local_image(
                interactive=interactive,
                assume_yes=assume_yes,
                allow_download=allow_download,
                transport=transport,
            )
            if image
            else None
        ),
    )
