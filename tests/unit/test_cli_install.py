"""Unit tests for `ucx install`: the non-interactive local-model install path.

Three things are pinned here, and they are the three that were missing when this was
measured on a stock Mac mini at `origin/main` e9c4fb1d:

1. **The checkpoint download.** `install.sh` installed uv, Python and the package;
   `ensure_local_image_pipeline` installed `diffusers`. Nothing fetched weights, so a
   beginner who ran every documented step could not generate an image. The tests below
   cover resume, a short body, the rename-on-success, and refusal without consent.
2. **A usable path with no terminal.** Both bootstrap steps called `rich.prompt.Confirm`
   unconditionally, which raises `EOFError` under `</dev/null`. An automated install test
   could not drive them at all.
3. **The machine-readable verdict.** A shell script needs a line it can grep and an exit
   code it can trust, and both must come from the probers rather than from an installer's
   exit code (P6, installation-flow.md §5).

No test here touches the network: the HTTP layer is a real `httpx.Client` over an
`httpx.MockTransport`, so the `Range`/`206`/`content-length` semantics under test are
httpx's own rather than a hand-rolled double's idea of them.
"""

# `_safetensors_complete` is imported directly: its failure modes are the point of several
# tests here, and enumerating them through an HTTP round trip each time would say less.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, Any, cast
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from uclone_x.cli import main
from uclone_x.cli.commands import bootstrap
from uclone_x.cli.commands.bootstrap import _safetensors_complete

runner = CliRunner()


def _whole_safetensors(payload: bytes) -> bytes:
    """A real, complete safetensors file with `payload` as its single tensor.

    Eight bytes of little-endian header length, that many bytes of UTF-8 JSON, then the
    tensor data — the layout `_safetensors_complete` reads a file's true length out of.
    Built here rather than asserted about a fixture blob so that every byte a test feeds
    the downloader is a byte a real server could send, and so a truncation under test is
    produced by slicing a whole file rather than by describing one.
    """
    header = json.dumps(
        {"weight": {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}}
    ).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + payload


BODY = _whole_safetensors(b"".join(bytes([i % 251]) for i in range(4096)))

#: What a captive portal or an intercepting proxy sends with a `200` and an honest
#: `content-length`: every length check passes and the bytes are not a checkpoint.
ERROR_PAGE = b"<!DOCTYPE html><html><body><h1>Sign in to continue</h1></body></html>"


class _Stream(httpx.SyncByteStream):
    """A response body of exactly these bytes, whatever the headers claim."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __iter__(self) -> Iterator[bytes]:
        yield self._payload


def _serving(
    payload: bytes,
    *,
    status: int = 200,
    declared_length: int | None = None,
    honour_range: bool = True,
    seen: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """A transport that serves `payload`, optionally lying about its length.

    `declared_length` is what `content-length` says; leaving it None makes the header
    truthful. The pair is what lets a truncated transfer be simulated exactly as a real
    one arrives — a full set of headers followed by a body that stops early.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = payload
        code = status
        header_map: dict[str, str] = {}
        range_header = request.headers.get("range")
        if range_header is not None and honour_range:
            start = int(range_header.removeprefix("bytes=").rstrip("-"))
            body = payload[start:]
            code = 206
            header_map["content-range"] = f"bytes {start}-{len(payload) - 1}/{len(payload)}"
        header_map["content-length"] = str(
            declared_length if declared_length is not None else len(body)
        )
        return httpx.Response(code, headers=header_map, stream=_Stream(body))

    return httpx.MockTransport(handler)


def _unreachable() -> httpx.MockTransport:
    """A transport that refuses, the way a machine with no network does."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network is unreachable", request=request)

    return httpx.MockTransport(handler)


# Annotated stand-ins rather than lambdas: strict mode cannot infer a lambda's parameter
# types, and these stand in for functions whose signatures are part of what is under test.


def _nothing_on_path(name: str, mode: int = os.X_OK, path: str | None = None) -> str | None:
    """`shutil.which` on the bare-PATH machine this work is for: it finds nothing."""
    return None


def _ollama_unreachable(url: str = "", timeout: float = 2.0) -> bool:
    return False


class _ReachabilityProbe:
    """`is_ollama_reachable`, remembering every address it was asked about.

    A stub that ignores its argument and answers True cannot tell a daemon at the
    configured endpoint from one at some other address — which is how a readiness loop that
    polled a hard-coded `localhost:11434` sat under a passing test. The recorded list is
    what makes the polled address an assertion rather than an assumption.
    """

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.urls: list[str] = []

    def __call__(self, url: str = "", timeout: float = 2.0) -> bool:
        self.urls.append(url)
        return self.answer


def _version_ok(binary: str | None = None) -> str | None:
    return "ollama version 0.5.0"


def _version_missing(binary: str | None = None) -> str | None:
    return None


def _run_ok(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess([], 0)


def _no_sleep(seconds: float) -> None:
    return None


# --- the download ------------------------------------------------------------------


def test_a_finished_download_is_renamed_off_the_part_file(tmp_path: Path) -> None:
    """The bytes land in `.part` and only a verified file gets the real name.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: part.replace(destination)
    Becomes: destination.write_bytes(part.read_bytes())

    A direct write to `destination` leaves the `.part` file behind, which the assertion
    below catches; more importantly, an interrupted run then leaves a file at the path the
    prober would load.
    """
    target = tmp_path / "models" / "sd_xl_base_1.0.safetensors"

    outcome = bootstrap.download_image_checkpoint(target, assume_yes=True, transport=_serving(BODY))

    assert outcome.ok is True
    assert outcome.path == str(target)
    assert target.read_bytes() == BODY
    assert not target.with_name(target.name + ".part").exists()


def test_a_short_body_fails_loudly_and_never_becomes_the_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transfer that stops early is a failure, not a 3 GB 'checkpoint'.

    This is the failure the `.part` discipline exists for, so it must also be unreachable
    by a body that simply ends before its declared length: safetensors that is short by a
    megabyte does not announce itself until a load hours later.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if expected_total is not None and actual != expected_total:
    Becomes: if False:

    The truncated file is then renamed and reported ready.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    transport = _serving(BODY[:100], declared_length=len(BODY))

    outcome = bootstrap.download_image_checkpoint(target, assume_yes=True, transport=transport)

    assert outcome.ok is False
    assert outcome.reason == "size-mismatch"
    assert not target.exists()
    part = target.with_name(target.name + ".part")
    assert part.read_bytes() == BODY[:100], "the partial bytes are kept so a resume can use them"
    assert "wrong size" in " ".join(capsys.readouterr().out.split())


def test_an_existing_part_file_is_resumed_rather_than_refetched(tmp_path: Path) -> None:
    """A `Range` request continues the partial file; the finished file is still whole.

    On a 7 GB body over a home connection this is the difference between an interruption
    that costs a minute and one that costs the whole download.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    Becomes: headers = {}

    The server then sends the whole body, which is appended to the partial one and
    produces a file of the wrong size (caught here by the byte comparison, not merely by
    the header assertion).
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    part = target.with_name(target.name + ".part")
    part.write_bytes(BODY[:1000])
    seen: list[httpx.Request] = []

    outcome = bootstrap.download_image_checkpoint(
        target, assume_yes=True, transport=_serving(BODY, seen=seen)
    )

    assert outcome.ok is True
    assert [r.headers.get("range") for r in seen] == ["bytes=1000-"]
    assert target.read_bytes() == BODY
    assert not part.exists()


def test_a_server_that_ignores_the_range_request_restarts_instead_of_appending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `200` answer to a `Range` request carries the whole file, not the remainder.

    Appending it to the partial bytes yields a file that is too long but plausible, and
    on a real 7 GB checkpoint nothing downstream would notice until a load failed. The
    partial file is discarded instead.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if resume_from and response.status_code != 206:
    Becomes: if False:

    `resume_from` is then left alone, the whole body is appended to the partial bytes, and
    the result is a plausible-sized corrupt file.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    target.with_name(target.name + ".part").write_bytes(BODY[:1000])

    outcome = bootstrap.download_image_checkpoint(
        target, assume_yes=True, transport=_serving(BODY, honour_range=False)
    )

    assert outcome.ok is True
    assert target.read_bytes() == BODY, "the stale partial bytes must not be prepended"
    assert "did not honour the resume request" in " ".join(capsys.readouterr().out.split())


def test_no_bytes_are_requested_without_consent(tmp_path: Path) -> None:
    """Without a yes, nothing is fetched and nothing is written — not even a `.part`.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if not ask_consent(prompt, interactive=interactive, assume_yes=assume_yes, default=False):
    Becomes: if False:

    The consent gate is skipped and the run falls straight through to the request.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    seen: list[httpx.Request] = []

    with patch.object(bootstrap, "stdin_is_interactive", return_value=True):
        with patch.object(bootstrap.Confirm, "ask", return_value=False):
            outcome = bootstrap.download_image_checkpoint(
                target, transport=_serving(BODY, seen=seen)
            )

    assert (outcome.ok, outcome.reason) == (False, "declined")
    assert seen == []
    assert list(tmp_path.iterdir()) == []


def test_the_size_is_on_screen_before_the_question_is_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--yes` still prints what it agreed to, with the GB figure (installation-flow §4.2).

    A confirmation the user never sees is not consent, and `--yes` is consent given in
    advance rather than permission to say nothing.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"

    bootstrap.download_image_checkpoint(target, assume_yes=True, transport=_serving(BODY))

    out = capsys.readouterr().out
    assert "6.94 GB" in " ".join(out.split())
    # `console.print` wraps at the terminal width, and a path has no spaces, so a long
    # one is folded mid-token: at 80 columns the basename itself came out as
    # `sd_x l_base_1.0.safetensors`. That is fine for a sentence a person reads; it is
    # exactly why the summary lines below use builtin `print`. Compare with the folds
    # taken out, so the check does not depend on how wide the terminal is.
    assert target.name in "".join(out.split())


def test_an_unreachable_network_is_a_sentence_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No network is the most likely failure of all, and it must not raise."""
    target = tmp_path / "sd_xl_base_1.0.safetensors"

    outcome = bootstrap.download_image_checkpoint(target, assume_yes=True, transport=_unreachable())

    assert (outcome.ok, outcome.reason) == (False, "network-error")
    assert not target.exists()
    assert "Could not reach the checkpoint server" in " ".join(capsys.readouterr().out.split())


def test_an_http_error_names_the_status_it_got(tmp_path: Path) -> None:
    """A 404 from a moved URL is reported as a 404, not as 'no checkpoint'."""
    target = tmp_path / "sd_xl_base_1.0.safetensors"

    outcome = bootstrap.download_image_checkpoint(
        target, assume_yes=True, transport=_serving(b"nope", status=404)
    )

    assert (outcome.ok, outcome.reason) == (False, "http-404")
    assert not target.exists()


def test_a_whole_checkpoint_already_on_disk_is_not_downloaded_again(tmp_path: Path) -> None:
    """Re-running the installer is free; it does not re-fetch 7 GB.

    The file written here is a complete safetensors, which is what makes the skip legal.
    Until this change the test wrote twelve bytes of `b"already here"` and asserted they
    were accepted — the early return tested for existence and nothing else, so the test
    pinned open exactly the hole that let an interrupted `curl -O` be reported ready.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: complete, reason = _safetensors_complete(destination)
    Becomes: complete, reason = (False, "forced")

    A machine that already has the checkpoint then downloads it again, every run. The
    anchor is the `destination` call and not the `if` below it: the 416 path has an
    identical `if complete:`, and an anchor matching both is not a mutation the ratchet
    can replay.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    target.write_bytes(BODY)
    seen: list[httpx.Request] = []

    outcome = bootstrap.download_image_checkpoint(
        target, assume_yes=True, transport=_serving(BODY, seen=seen)
    )

    assert (outcome.ok, outcome.path) == (True, str(target))
    assert seen == []
    assert target.read_bytes() == BODY


def test_a_truncated_file_already_at_the_destination_is_not_trusted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A half-finished `curl -O` at the checkpoint path is re-fetched, not reported ready.

    This is the shape the early return could not see: existence was the whole readiness
    test, so a 0-byte file or an interrupted manual download made `ucx install` skip the
    download and print `setup image: ready`, and the file failed at load time hours later.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: complete, reason = _safetensors_complete(destination)
    Becomes: complete, reason = (True, "")

    The junk file is then returned as the installed checkpoint.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    target.write_bytes(BODY[: len(BODY) // 2])

    outcome = bootstrap.download_image_checkpoint(target, assume_yes=True, transport=_serving(BODY))

    assert outcome.ok is True
    assert target.read_bytes() == BODY, "the good bytes replaced the junk ones"
    assert "not a usable checkpoint (truncated)" in " ".join(capsys.readouterr().out.split())


def test_an_empty_file_already_at_the_destination_is_not_trusted(tmp_path: Path) -> None:
    """The 0-byte case specifically: `curl -O` creates the file before it writes to it."""
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    target.touch()
    seen: list[httpx.Request] = []

    outcome = bootstrap.download_image_checkpoint(
        target, assume_yes=True, transport=_serving(BODY, seen=seen)
    )

    assert outcome.ok is True
    assert target.read_bytes() == BODY
    assert seen != [], "the empty file must not have satisfied the install"


def test_the_download_target_follows_the_environment_variable_the_prober_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`UCX_IMAGE_CHECKPOINT` decides where the file goes, and `~` means the same here
    as it does to the engine that will load it.

    Writing to the default directory while the engine reads the variable would leave 7 GB
    on disk and the engine still reporting no checkpoint.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("UCX_IMAGE_CHECKPOINT", "~/elsewhere/mine.safetensors")

    assert bootstrap.image_checkpoint_target() == tmp_path / "elsewhere" / "mine.safetensors"

    monkeypatch.delenv("UCX_IMAGE_CHECKPOINT")
    default = bootstrap.image_checkpoint_target()
    assert default == tmp_path / "ai_models" / "checkpoints" / "sd_xl_base_1.0.safetensors"


def test_the_default_target_is_a_path_the_image_prober_searches() -> None:
    """The download and the search must name the same file.

    A download that lands one directory away from `DEFAULT_CHECKPOINTS` succeeds and
    changes nothing, which is the most expensive way for these two to disagree.
    """
    from uclone_x.tools.builtin.image import DEFAULT_CHECKPOINTS, expand_checkpoint_path

    expanded = {expand_checkpoint_path(candidate) for candidate in DEFAULT_CHECKPOINTS}
    assert str(bootstrap.image_checkpoint_target()) in expanded


def test_the_two_checkpoints_that_already_worked_are_still_searched() -> None:
    """SDXL base was appended to `DEFAULT_CHECKPOINTS`, not substituted for them.

    Replacing the list would un-find the checkpoint on every machine that already has one
    — an upgrade that breaks a working install.
    """
    from uclone_x.tools.builtin.image import DEFAULT_CHECKPOINTS

    assert "~/ai_models/checkpoints/anillustrious_v4.safetensors" in DEFAULT_CHECKPOINTS
    assert "~/ai_models/checkpoints/Illustrious-XL-v0.1.safetensors" in DEFAULT_CHECKPOINTS


# --- consent with no terminal ------------------------------------------------------


def test_no_terminal_means_no_prompt_and_no_eoferror(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The `</dev/null` case: decline, explain, and never reach `Confirm.ask`.

    `Confirm.ask` on a closed stdin raises `EOFError` out of the command, which is how an
    automated install of this project failed before there was any way to drive it.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if not interactive or not stdin_is_interactive():
    Becomes: if not interactive:

    The prompt is then reached with no stdin.
    """
    with patch.object(bootstrap, "stdin_is_interactive", return_value=False):
        with patch.object(bootstrap.Confirm, "ask", side_effect=EOFError) as ask:
            answer = bootstrap.ask_consent("Do the thing?", interactive=True)

    assert answer is False
    ask.assert_not_called()
    out = " ".join(capsys.readouterr().out.split())
    assert "Skipped" in out and "--yes" in out


def test_yes_is_consent_given_in_advance_and_asks_nothing() -> None:
    with patch.object(bootstrap.Confirm, "ask", side_effect=AssertionError("asked anyway")):
        assert bootstrap.ask_consent("Do the thing?", assume_yes=True) is True


def test_a_terminal_that_disappears_mid_prompt_declines_rather_than_raising() -> None:
    """`isatty` was true a moment ago; the read can still hit EOF. Not a crash."""
    with patch.object(bootstrap, "stdin_is_interactive", return_value=True):
        with patch.object(bootstrap.Confirm, "ask", side_effect=EOFError):
            assert bootstrap.ask_consent("Do the thing?") is False


def test_a_non_interactive_llm_setup_declines_without_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing is installed on a machine that could not be asked, and the reason says so.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: return LlmSetup(False, specs.recommended_model, endpoint, "declined")
    Becomes: return LlmSetup(True, specs.recommended_model, endpoint, "declined")

    That is the declined branch of the `ask_consent` call in `setup_local_llm`; mutated, a
    machine whose owner was never asked reports a ready model.
    """
    monkeypatch.setattr(
        bootstrap, "get_system_specs", lambda: bootstrap.SystemSpecs(16.0, 50.0, "qwen3:8b")
    )
    monkeypatch.setattr(bootstrap, "is_ollama_reachable", _ollama_unreachable)
    monkeypatch.setattr(bootstrap, "stdin_is_interactive", lambda: False)

    with patch.object(bootstrap, "install_ollama_platform") as install:
        with patch.object(bootstrap, "pull_ollama_model") as pull:
            result = bootstrap.setup_local_llm(interactive=True)

    assert (result.ready, result.reason) == (False, "declined")
    install.assert_not_called()
    pull.assert_not_called()


# --- Ollama on a stock Mac ----------------------------------------------------------


def test_the_official_installer_runs_without_a_gui_launch_or_a_password_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`OLLAMA_NO_START=1`, no `SUDO_ASKPASS`, stdin closed — and `OLLAMA_MODELS` kept.

    Measured on the target Mac mini: `/usr/local/bin` is not user-writable, so the
    installer's symlink falls back to `sudo`, and `open -a Ollama` is a GUI launch that
    over ssh is both impossible and wrong. `OLLAMA_MODELS` is in the same environment and
    must survive: an install test points it at a throwaway directory, and losing it writes
    gigabytes into the user's real `~/.ollama`.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: env["OLLAMA_NO_START"] = "1"
    Becomes:

    With the line deleted the installer ends with `open -a Ollama`, a GUI launch.
    """
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("shutil.which", _nothing_on_path)
    monkeypatch.setenv("OLLAMA_MODELS", "/tmp/throwaway-models")
    monkeypatch.setenv("SUDO_ASKPASS", "/usr/bin/false")
    calls: list[dict[str, Any]] = []

    def fake_run(command: object, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess([], 1)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    monkeypatch.setattr(bootstrap, "ollama_binary", lambda: "/Applications/Ollama.app/x/ollama")
    monkeypatch.setattr(bootstrap, "ollama_version", _version_ok)

    assert bootstrap.install_ollama_platform() is True

    env = calls[0]["env"]
    assert env["OLLAMA_NO_START"] == "1"
    assert env["OLLAMA_MODELS"] == "/tmp/throwaway-models"
    assert "SUDO_ASKPASS" not in env
    assert calls[0]["stdin"] is subprocess.DEVNULL


def test_a_nonzero_installer_exit_is_not_the_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installer fails its symlink and exits nonzero while the app installs fine.

    The only honest check is running the binary, so a nonzero exit with a working binary
    must read as success — and a zero exit with no runnable binary must read as failure.
    The second half is the one that matters: it is the false success.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if binary is None or version is None:
    Becomes: if False:

    A machine with no ollama then reports a successful install.
    """
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("shutil.which", _nothing_on_path)
    monkeypatch.setattr(bootstrap.subprocess, "run", _run_ok)
    monkeypatch.setattr(bootstrap, "ollama_binary", lambda: None)
    monkeypatch.setattr(bootstrap, "ollama_version", _version_missing)

    assert bootstrap.install_ollama_platform() is False


def test_the_binary_is_found_in_the_app_bundle_when_the_symlink_never_happened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not on PATH is not the same as not installed, on the machine this is for.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: for candidate in OLLAMA_APP_BINARIES:
    Becomes: for candidate in ():

    With the bundle loop in `ollama_binary` emptied, the install reports failure while a
    working ollama sits in /Applications.
    """
    bundled = tmp_path / "Ollama.app" / "Contents" / "Resources" / "ollama"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("#!/bin/sh\n")
    bundled.chmod(0o755)
    monkeypatch.setattr("shutil.which", _nothing_on_path)
    monkeypatch.setattr(bootstrap, "OLLAMA_APP_BINARIES", (str(bundled),))

    assert bootstrap.ollama_binary() == str(bundled)
    assert bootstrap.is_ollama_installed() is True


def test_a_non_executable_bundle_path_is_not_offered_as_a_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover file without the executable bit would fail at exec time instead."""
    stub = tmp_path / "ollama"
    stub.write_text("")
    stub.chmod(0o644)
    monkeypatch.setattr("shutil.which", _nothing_on_path)
    monkeypatch.setattr(bootstrap, "OLLAMA_APP_BINARIES", (str(stub),))

    assert bootstrap.ollama_binary() is None


def test_pull_and_serve_run_the_resolved_binary_not_the_bare_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`"ollama"` only works once the symlink exists, which is exactly what failed.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: [binary, "pull", model]
    Becomes: ["ollama", "pull", model]

    The pull then fails with FileNotFoundError on the machine where the fallback was
    needed.

    The serve half also pins *where* the wait looks: the probe records its argument, so
    `start_ollama_daemon` polling anything other than the endpoint this environment
    configures fails here. It previously called `is_ollama_reachable()` with no argument at
    all, against a stub that ignored its own parameter — a defect and a test that could not
    see it, fitted to each other.
    """
    resolved = "/Applications/Ollama.app/Contents/Resources/ollama"
    monkeypatch.setattr(bootstrap, "ollama_binary", lambda: resolved)
    monkeypatch.setenv("OLLAMA_MODELS", "/tmp/throwaway-models")
    ran: list[Sequence[str]] = []
    envs: list[dict[str, str]] = []

    def fake_run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        ran.append(command)
        envs.append(kwargs["env"])
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    assert bootstrap.pull_ollama_model("qwen3:8b") is True
    assert ran == [[resolved, "pull", "qwen3:8b"]]
    assert envs[0]["OLLAMA_MODELS"] == "/tmp/throwaway-models"

    popen_calls: list[Sequence[str]] = []
    popen_envs: list[dict[str, str]] = []

    class _Popen:
        def __init__(self, command: Sequence[str], **kwargs: Any) -> None:
            popen_calls.append(command)
            popen_envs.append(kwargs["env"])

    monkeypatch.setattr(bootstrap.subprocess, "Popen", _Popen)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:41234")
    probe = _ReachabilityProbe()
    monkeypatch.setattr(bootstrap, "is_ollama_reachable", probe)
    monkeypatch.setattr(bootstrap.time, "sleep", _no_sleep)

    assert bootstrap.start_ollama_daemon() is True
    assert popen_calls == [[resolved, "serve"]]
    assert popen_envs[0]["OLLAMA_MODELS"] == "/tmp/throwaway-models"
    assert probe.urls == ["http://127.0.0.1:41234"]

    assert bootstrap.start_ollama_daemon("http://10.0.0.7:11434") is True
    assert probe.urls[-1] == "http://10.0.0.7:11434", "an explicit endpoint wins over the default"


# --- where the daemon is ------------------------------------------------------------


def test_a_schemeless_ollama_host_is_an_address_this_install_can_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`OLLAMA_HOST=127.0.0.1:41234` is Ollama's own documented form, and it must work.

    Two failures, both measured. `ollama_endpoint` read `OLLAMA_BASE_URL` alone, so the
    install looked at `localhost:11434` while `scripts/verify_online_install.sh` had
    isolated the daemon on a free port — `setup llm: unavailable reason=ollama-not-running`
    about a daemon that was running. And on a machine that does serve 11434, the install
    reported ready against a daemon no later turn would speak to, because the connector
    layer *does* read `OLLAMA_HOST` — and then crashed with `UnsupportedProtocol: Request
    URL is missing an 'http://' or 'https://' protocol`, because nothing supplied the
    scheme this variable never carries.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: cleaned = f"http://{cleaned}"
    Becomes: cleaned = cleaned

    The endpoint then reaches httpx as `127.0.0.1:41234`, which it refuses to request.
    """
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:41234")

    assert bootstrap.ollama_endpoint() == "http://127.0.0.1:41234"


def test_the_install_and_the_connectors_resolve_the_endpoint_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One resolution order, not a shorter one for the installer.

    `ollama_endpoint` was a third implementation of a question `resolve_ollama_base_url`
    already answered, and the only one that ignored `OLLAMA_HOST`. An install that resolves
    a different daemon than every turn afterwards reports on a process it does not use.
    """
    from uclone_x.llm.connectors.ollama import resolve_ollama_base_url

    assert bootstrap.ollama_endpoint() == resolve_ollama_base_url() == "http://localhost:11434"

    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:41234")
    assert bootstrap.ollama_endpoint() == resolve_ollama_base_url()

    # Higher in the documented precedence list than OLLAMA_HOST, and still the same answer
    # on both sides of the install boundary.
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.7:11434/v1")
    assert bootstrap.ollama_endpoint() == resolve_ollama_base_url() == "http://10.0.0.7:11434"


def test_the_daemon_start_is_asked_about_the_endpoint_the_setup_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`setup_local_llm` hands its endpoint to the start rather than letting it re-guess.

    The probe and the wait have to be asking about the same address, or the verdict is
    about a different process than the one that was started — ready because some unrelated
    daemon answers 11434, or unavailable because the one that was asked for is not there.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if not start_ollama_daemon(endpoint):
    Becomes: if not start_ollama_daemon():

    The wait then falls back to the default address on a machine configured for another.
    """
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:41234")
    monkeypatch.setattr(
        bootstrap, "get_system_specs", lambda: bootstrap.SystemSpecs(16.0, 50.0, "qwen3:8b")
    )
    monkeypatch.setattr(bootstrap, "is_ollama_reachable", _ReachabilityProbe(answer=False))
    monkeypatch.setattr(bootstrap, "is_ollama_installed", lambda: True)
    asked: list[str | None] = []

    def fake_start(endpoint: str | None = None) -> bool:
        asked.append(endpoint)
        return False

    monkeypatch.setattr(bootstrap, "start_ollama_daemon", fake_start)

    result = bootstrap.setup_local_llm(assume_yes=True)

    assert asked == ["http://127.0.0.1:41234"]
    assert (result.ready, result.reason) == (False, "ollama-not-running")
    assert result.endpoint == "http://127.0.0.1:41234"


# --- the summary lines and the exit code --------------------------------------------


def test_the_ready_summary_lines_have_the_documented_shape() -> None:
    assert (
        bootstrap.LlmSetup(True, "qwen3:8b", "http://localhost:11434").summary_line()
        == "setup llm: ready model=qwen3:8b endpoint=http://localhost:11434"
    )
    assert (
        bootstrap.ImageSetup(True, "diffusers-sdxl", "/m/sd_xl_base_1.0.safetensors").summary_line()
        == "setup image: ready engine=diffusers-sdxl checkpoint=/m/sd_xl_base_1.0.safetensors"
    )


def test_the_unavailable_summary_lines_carry_the_reason() -> None:
    assert (
        bootstrap.LlmSetup(False, "qwen3:8b", "http://x", "declined").summary_line()
        == "setup llm: unavailable reason=declined"
    )
    assert (
        bootstrap.ImageSetup(False, "none", None, "size-mismatch").summary_line()
        == "setup image: unavailable reason=size-mismatch"
    )


def test_an_engine_with_no_local_file_still_prints_the_checkpoint_key() -> None:
    """A remote worker holds its own weights; the line's shape must not change."""
    assert (
        bootstrap.ImageSetup(True, "remote-cuda").summary_line()
        == "setup image: ready engine=remote-cuda checkpoint=-"
    )


def test_a_part_that_was_not_requested_prints_no_line_at_all() -> None:
    """`--no-image` must not emit a line a grep could match either way.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: return [part.summary_line() for part in (self.llm, self.image) if part is not None]
    Becomes: return [part.summary_line() if part is not None else "setup: skipped" for part in (self.llm, self.image)]

    Trading `LocalSetupResult.summary_lines`'s `if part is not None` filter for a third
    status word is exactly the line a grep would then match either way.
    """
    result = bootstrap.LocalSetupResult(llm=bootstrap.LlmSetup(True, "qwen3:8b", "http://x"))
    assert result.summary_lines() == ["setup llm: ready model=qwen3:8b endpoint=http://x"]


def test_one_failed_part_fails_the_whole_run() -> None:
    """A ready LLM does not excuse a missing image engine, and vice versa.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: all(part.ready for part in (self.llm, self.image) if part is not None)
    Becomes: any(part.ready for part in (self.llm, self.image) if part is not None)

    `LocalSetupResult.ok` over `any` lets the install test pass with half an install.
    """
    ready_llm = bootstrap.LlmSetup(True, "qwen3:8b", "http://x")
    broken_image = bootstrap.ImageSetup(False, "none", None, "no-checkpoint")
    assert bootstrap.LocalSetupResult(ready_llm, broken_image).ok is False
    assert bootstrap.LocalSetupResult(llm=ready_llm).ok is True


def test_a_part_that_was_skipped_cannot_make_the_run_fail() -> None:
    assert bootstrap.LocalSetupResult().ok is True


# --- the CLI surface ----------------------------------------------------------------


def _fake_setup(captured: dict[str, Any]) -> Any:
    """A `run_local_setup` stand-in that records the keyword arguments it was called with."""

    def run(**kwargs: Any) -> bootstrap.LocalSetupResult:
        captured.update(kwargs)
        return bootstrap.LocalSetupResult(
            llm=bootstrap.LlmSetup(True, "qwen3:8b", "http://localhost:11434")
            if kwargs.get("llm")
            else None,
            image=bootstrap.ImageSetup(True, "diffusers-sdxl", "/m/sd_xl_base_1.0.safetensors")
            if kwargs.get("image")
            else None,
        )

    return run


def test_ucx_install_prints_both_lines_last_and_exits_zero() -> None:
    """The contract a shell script reads: two lines, at the end, exit 0.

    Printed with `print` rather than `console.print` on purpose — rich folds a long
    checkpoint path at the terminal width, and a grep cannot read a wrapped line.
    """
    captured: dict[str, Any] = {}
    with patch.object(bootstrap, "run_local_setup", _fake_setup(captured)):
        result = runner.invoke(main.app, ["install", "--yes"])

    assert result.exit_code == 0
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert lines[-2:] == [
        "setup llm: ready model=qwen3:8b endpoint=http://localhost:11434",
        "setup image: ready engine=diffusers-sdxl checkpoint=/m/sd_xl_base_1.0.safetensors",
    ]


def test_ucx_install_passes_yes_through_to_the_consent_checks() -> None:
    captured: dict[str, Any] = {}
    with patch.object(bootstrap, "run_local_setup", _fake_setup(captured)):
        runner.invoke(main.app, ["install", "--yes"])
    assert captured["assume_yes"] is True

    captured.clear()
    with patch.object(bootstrap, "run_local_setup", _fake_setup(captured)):
        runner.invoke(main.app, ["install"])
    assert captured["assume_yes"] is False


def test_ucx_install_no_image_runs_and_reports_only_the_llm() -> None:
    captured: dict[str, Any] = {}
    with patch.object(bootstrap, "run_local_setup", _fake_setup(captured)):
        result = runner.invoke(main.app, ["install", "--yes", "--no-image"])

    assert captured["image"] is False
    assert "setup image" not in result.output
    assert result.exit_code == 0


def test_ucx_install_exits_one_when_a_requested_part_is_not_ready() -> None:
    """The exit code is the install test's assertion, so a half install must be nonzero.

    Killed by: src/uclone_x/cli/main.py :: raise typer.Exit(0 if result.ok else 1)
    Becomes: raise typer.Exit(0)
    """
    unavailable = bootstrap.LocalSetupResult(
        llm=bootstrap.LlmSetup(True, "qwen3:8b", "http://localhost:11434"),
        image=bootstrap.ImageSetup(False, "none", None, "declined"),
    )
    with patch.object(bootstrap, "run_local_setup", return_value=unavailable):
        result = runner.invoke(main.app, ["install", "--yes"])

    assert result.exit_code == 1
    assert result.output.splitlines()[-1] == "setup image: unavailable reason=declined"


def test_ucx_start_uses_the_same_setup_path_as_ucx_install() -> None:
    """`start` delegates rather than keeping its own copy of the sequence.

    Two copies of "prepare the local models" is how one of them silently stops matching
    the other; `start` had the whole sequence inline.

    Killed by: src/uclone_x/cli/main.py :: result = run_local_setup(llm=True, image=True, interactive=True, assume_yes=False)
    Becomes: result = run_local_setup(llm=True, image=False, interactive=True, assume_yes=False)

    A re-inlined `ensure_local_profile` / `ensure_local_image_pipeline` pair is the drift
    this test exists to catch, and its smallest executable form is `start` asking for a
    different set of parts than `install` does — which this assertion reads straight off
    the recorded keyword arguments.
    """
    captured: dict[str, Any] = {}
    with patch.object(bootstrap, "run_local_setup", _fake_setup(captured)):
        with patch("uclone_x.ui.server.start_ui_server") as serve:
            runner.invoke(main.app, ["start", "--no-open", "--port", "5999"])

    assert captured == {"llm": True, "image": True, "interactive": True, "assume_yes": False}
    serve.assert_called_once()


def test_ucx_start_exports_the_endpoint_the_setup_actually_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`OLLAMA_BASE_URL` comes from the result, not from a second hard-coded literal.

    Killed by: src/uclone_x/cli/main.py :: os.environ.setdefault("OLLAMA_BASE_URL", result.llm.endpoint)
    Becomes: os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

    A second hard-coded literal then contradicts the endpoint the prober reached and the
    summary line reported.
    """
    for key in ("LLM_PROVIDER", "OLLAMA_MODEL", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    elsewhere = bootstrap.LocalSetupResult(
        llm=bootstrap.LlmSetup(True, "qwen3:8b", "http://10.0.0.7:11434"),
        image=bootstrap.ImageSetup(False, "none", None, "declined"),
    )

    with patch.object(bootstrap, "run_local_setup", return_value=elsewhere):
        with patch("uclone_x.ui.server.start_ui_server"):
            runner.invoke(main.app, ["start", "--no-open", "--port", "5999"])

    assert os.environ["OLLAMA_BASE_URL"] == "http://10.0.0.7:11434"
    assert os.environ["OLLAMA_MODEL"] == "qwen3:8b"


def test_ucx_install_exists_in_an_installed_build_unlike_ucx_setup() -> None:
    """`install` is the end user's command, so it must not be checkout-only.

    `ucx setup` is a different command with a different job — it writes this repository's
    git hooks — and `register_developer_commands` deliberately withholds it from an
    installed build. An install path registered the same way would not exist on the
    machine it is for.
    """
    names = {
        command.name or (command.callback.__name__ if command.callback else "")
        for command in main.app.registered_commands
    }
    assert "install" in names
    assert main.install_models.__module__ == "uclone_x.cli.main"


# --- the whole image path, end to end over a fake server ----------------------------


def test_the_image_half_downloads_the_checkpoint_and_then_reports_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap this work exists to close, exercised through `setup_local_image`.

    Before this change the same inputs — no checkpoint, packages present — printed
    "place a checkpoint in ~/ai_models/checkpoints/" and returned `(False, "none")`.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: outcome = download_image_checkpoint(
    Becomes: outcome = DownloadOutcome(False, None, "no-checkpoint") or download_image_checkpoint(

    Short-circuiting the `download_image_checkpoint` call in `setup_local_image` restores
    the original dead end, `ImageSetup(False, "none", None, "no-checkpoint")`.
    """
    target = tmp_path / "ckpt" / "sd_xl_base_1.0.safetensors"
    monkeypatch.setenv("UCX_IMAGE_CHECKPOINT", str(target))
    report = bootstrap.ImageEngineReport(
        remote_url=None,
        remote_alive=False,
        comfy_url="http://127.0.0.1:8188",
        comfy_alive=False,
        dependency_problems=(),
        checkpoint=None,
    )

    with patch.object(bootstrap, "probe_image_engines", return_value=report):
        result = bootstrap.setup_local_image(assume_yes=True, transport=_serving(BODY))

    assert (result.ready, result.engine, result.checkpoint) == (
        True,
        "diffusers-sdxl",
        str(target),
    )
    assert target.read_bytes() == BODY


def test_a_failed_checkpoint_download_is_reported_with_its_own_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The download's reason reaches the summary line rather than being flattened.

    'no-checkpoint' and 'network-error' send the reader to different places, and
    collapsing them is the P6 failure `CheckpointResolution` already split apart for the
    prober (#1120).
    """
    monkeypatch.setenv("UCX_IMAGE_CHECKPOINT", str(tmp_path / "sd_xl_base_1.0.safetensors"))
    report = bootstrap.ImageEngineReport(
        remote_url=None,
        remote_alive=False,
        comfy_url="http://127.0.0.1:8188",
        comfy_alive=False,
        dependency_problems=(),
        checkpoint=None,
    )

    with patch.object(bootstrap, "probe_image_engines", return_value=report):
        result = bootstrap.setup_local_image(assume_yes=True, transport=_unreachable())

    assert result.summary_line() == "setup image: unavailable reason=network-error"


def test_an_unwritable_destination_is_named_before_any_bytes_are_fetched(
    tmp_path: Path,
) -> None:
    """A read-only checkpoint directory is the reason, not a network failure.

    Fetching 7 GB and only then discovering the write fails is the expensive ordering,
    and 'network-error' would send the reader to the wrong place entirely.
    """
    blocker = tmp_path / "blocked"
    blocker.write_text("this is a file, so it cannot also be a directory")
    seen: list[httpx.Request] = []

    outcome = bootstrap.download_image_checkpoint(
        blocker / "sub" / "sd_xl_base_1.0.safetensors",
        assume_yes=True,
        transport=_serving(BODY, seen=seen),
    )

    assert (outcome.ok, outcome.reason) == (False, "destination-unwritable")
    assert seen == [], "the directory is checked before the download starts"


def test_a_disk_that_fills_mid_download_is_reported_as_a_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENOSPC on a 7 GB file is likely enough that it must not surface as a traceback."""
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    real_open = Path.open

    def failing_open(self: Path, *args: Any, **kwargs: Any) -> IO[Any]:
        if self.name.endswith(".part"):
            raise OSError(28, "No space left on device")
        return cast(IO[Any], real_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", failing_open)

    outcome = bootstrap.download_image_checkpoint(target, assume_yes=True, transport=_serving(BODY))

    assert (outcome.ok, outcome.reason) == (False, "write-error")


def _chunked(payload: bytes) -> httpx.MockTransport:
    """A `200` with no `content-length` at all, the way a chunked response arrives."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={}, stream=_Stream(payload))

    return httpx.MockTransport(handler)


def test_a_server_that_declares_no_length_still_completes(tmp_path: Path) -> None:
    """A chunked response carrying the whole file is accepted; refusing it would be wrong.

    The percentages are simply not printed, because there is no total to divide by.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"

    outcome = bootstrap.download_image_checkpoint(target, assume_yes=True, transport=_chunked(BODY))

    assert outcome.ok is True
    assert target.read_bytes() == BODY


def test_a_short_body_with_no_declared_length_is_still_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A connection that closes early on a chunked response must not become a checkpoint.

    This test used to assert the opposite half of the same call: with no `content-length`
    the size comparison is skipped entirely, so it asserted that *any* body — this one, an
    error page, a single byte — was renamed into place and reported ready. There is
    something to verify against after all, and it is the file: safetensors states its own
    length in its first eight bytes and its header's `data_offsets`.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if not complete:
    Becomes: if False:

    The short body is then renamed to `sd_xl_base_1.0.safetensors` and reported ready.
    The anchor is the guard rather than the `_safetensors_complete(part)` call above it:
    the 416 path opens with that same call, so a substring naming it names two lines and
    the ratchet cannot replay it.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"

    outcome = bootstrap.download_image_checkpoint(
        target, assume_yes=True, transport=_chunked(BODY[:300])
    )

    assert (outcome.ok, outcome.reason) == (False, "truncated")
    assert not target.exists()
    part = target.with_name(target.name + ".part")
    assert part.read_bytes() == BODY[:300], "the partial bytes are kept so a resume can use them"
    assert "not a complete checkpoint (truncated)" in " ".join(capsys.readouterr().out.split())


def test_an_error_page_served_with_a_truthful_length_is_not_a_checkpoint(
    tmp_path: Path,
) -> None:
    """A captive portal answers `200` with HTML and a correct `content-length`.

    Every length check in this function compares the body against a number that arrived
    with the body, so an intercepting proxy passes all of them truthfully. Only the file's
    own structure separates a checkpoint from a sign-in page.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if not complete:
    Becomes: if False:

    The HTML page is then installed as the SDXL checkpoint.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"

    outcome = bootstrap.download_image_checkpoint(
        target, assume_yes=True, transport=_serving(ERROR_PAGE)
    )

    assert (outcome.ok, outcome.reason) == (False, "not-safetensors")
    assert not target.exists()


def test_a_complete_part_file_is_installed_when_the_server_refuses_the_resume(
    tmp_path: Path,
) -> None:
    """HTTP 416 on a full-length `.part` is a finished download, not a dead end.

    An interruption between the last chunk write and the rename leaves the `.part` at full
    length; the next run asks to resume from the end of the file and the server answers
    416. Reported as `http-416` that repeats forever, because nothing about running it
    again changes the offset — and the bytes the user already paid for are right there.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if response.status_code == 416:
    Becomes: if False:

    The generic `>= 400` branch then returns `http-416` and keeps the part, on this run and
    on every run after it.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    part = target.with_name(target.name + ".part")
    part.write_bytes(BODY)

    outcome = bootstrap.download_image_checkpoint(
        target, assume_yes=True, transport=_serving(BODY, status=416, honour_range=False)
    )

    assert (outcome.ok, outcome.path) == (True, str(target))
    assert target.read_bytes() == BODY
    assert not part.exists()


def test_an_unusable_part_file_is_discarded_when_the_server_refuses_the_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """416 over a `.part` that is not a whole file leaves nothing behind to wedge on.

    Keeping it would ask the next run to resume from the same offset and be refused the
    same way. Discarding it is what makes "run this again" true rather than advice that
    cannot work.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: part.unlink(missing_ok=True)
    Becomes: pass

    The unusable partial file then survives, and every later run is refused at the same
    offset.
    """
    target = tmp_path / "sd_xl_base_1.0.safetensors"
    part = target.with_name(target.name + ".part")
    part.write_bytes(BODY + b"stray trailing bytes")

    outcome = bootstrap.download_image_checkpoint(
        target, assume_yes=True, transport=_serving(BODY, status=416, honour_range=False)
    )

    assert (outcome.ok, outcome.reason) == (False, "partial-discarded")
    assert not part.exists()
    assert not target.exists()
    out = " ".join(capsys.readouterr().out.split())
    assert "discarded" in out and "run this again" in out


def _framed(header: bytes, data: bytes = b"", *, declared: int | None = None) -> bytes:
    """A safetensors frame whose declared header length can be made to lie."""
    length = len(header) if declared is None else declared
    return length.to_bytes(8, "little") + header + data


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"", "not-safetensors"),
        (b"\x01\x02\x03", "not-safetensors"),
        ((2**63).to_bytes(8, "little"), "not-safetensors"),
        ((0).to_bytes(8, "little"), "not-safetensors"),
        (b"<!DOCTYPE html><html>a sign-in page</html>", "not-safetensors"),
        (_framed(b"[1,2,3]"), "not-safetensors"),
        (_framed(b'{"weight":1}'), "not-safetensors"),
        (_framed(b'{"weight":{"data_offsets":[0]}}'), "not-safetensors"),
        (_framed(b'{"weight":{"data_offsets":["0","4"]}}'), "not-safetensors"),
        (_framed(b'{"weight":{"data_offsets":[0,4]}}', b"\x00" * 12), "trailing-bytes"),
        (_framed(b'{"weight":', declared=500), "truncated"),
        (_framed(b'{"weight":{"data_offsets":[0,64]}}', b"\x00" * 32), "truncated"),
        (_framed(b'{"__metadata__":{"format":"pt"}}'), "no-tensors"),
    ],
)
def test_the_completeness_check_names_what_is_wrong_and_never_raises(
    tmp_path: Path, payload: bytes, reason: str
) -> None:
    """Each way a file can fail to be a checkpoint answers with its own short reason.

    The reason reaches `setup image: unavailable reason=…`, which is the only thing a
    non-interactive install leaves the reader, so `truncated` and `not-safetensors` must
    not collapse into one another: one says finish the download, the other says the bytes
    never came from Hugging Face. The absurd header length is in this list because it is
    the one input that could turn a check into an out-of-memory kill — it is rejected on
    the declared figure, before any read.
    """
    path = tmp_path / "candidate.safetensors"
    path.write_bytes(payload)

    assert _safetensors_complete(path) == (False, reason)


def test_the_completeness_check_accepts_a_file_that_carries_metadata(tmp_path: Path) -> None:
    """`__metadata__` is not a tensor, and the real SDXL checkpoint has one.

    Reading it as an entry — it has no `data_offsets` — would reject every file Hugging
    Face actually serves, which is a refusal that looks exactly like a corrupt download.
    """
    payload = b"\x00" * 64
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "U8", "shape": [64], "data_offsets": [0, 64]},
        }
    ).encode("utf-8")
    path = tmp_path / "candidate.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)

    assert _safetensors_complete(path) == (True, "")


def test_an_unreadable_file_is_a_reason_rather_than_an_oserror(tmp_path: Path) -> None:
    """A path that cannot be opened is still an answer: the install must not crash."""
    assert _safetensors_complete(tmp_path / "nothing-here.safetensors") == (
        False,
        "unreadable",
    )


def test_the_dead_end_message_is_still_reachable_when_downloading_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`allow_download=False` keeps the old behaviour for callers that only want a probe."""
    monkeypatch.setenv("UCX_IMAGE_CHECKPOINT", str(tmp_path / "sd_xl_base_1.0.safetensors"))
    report = bootstrap.ImageEngineReport(
        remote_url=None,
        remote_alive=False,
        comfy_url="http://127.0.0.1:8188",
        comfy_alive=False,
        dependency_problems=(),
        checkpoint=None,
    )

    with patch.object(bootstrap, "probe_image_engines", return_value=report):
        result = bootstrap.setup_local_image(assume_yes=True, allow_download=False)

    assert result.summary_line() == "setup image: unavailable reason=no-checkpoint"


# --- Ollama on a connection that drops ------------------------------------------------


class _FlakyInstaller:
    """The official installer, leaving a runnable binary only from run `works_from` on.

    `works_from=None` never leaves one. Every run is counted, so "retried" and "gave up"
    are assertions about how many times the installer ran, not about a message.
    """

    def __init__(self, works_from: int | None) -> None:
        self.works_from = works_from
        self.runs = 0

    def run(self, *args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        self.runs += 1
        return subprocess.CompletedProcess([], 1)

    def binary(self) -> str | None:
        if self.works_from is not None and self.runs >= self.works_from:
            return "/usr/local/bin/ollama"
        return None


def _flaky_linux(monkeypatch: pytest.MonkeyPatch, installer: _FlakyInstaller) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("shutil.which", _nothing_on_path)
    monkeypatch.setattr(bootstrap.subprocess, "run", installer.run)
    monkeypatch.setattr(bootstrap, "ollama_binary", installer.binary)
    monkeypatch.setattr(bootstrap, "ollama_version", _version_ok)
    monkeypatch.setattr(bootstrap, "OLLAMA_INSTALL_RETRY_DELAY_S", 0.0)


def test_a_dropped_download_is_tried_again_until_ollama_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two cut-off downloads, then a good one: the install succeeds on the third run.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if found is not None or attempt == OLLAMA_INSTALL_ATTEMPTS:
    Becomes: if True:

    Mutated, the first dropped download is the verdict again, as before the fix.
    """
    installer = _FlakyInstaller(works_from=3)
    _flaky_linux(monkeypatch, installer)

    assert bootstrap.install_ollama_platform() is True

    assert installer.runs == 3
    out = " ".join(capsys.readouterr().out.split())
    assert "The Ollama download did not finish" in out
    assert "Trying again (2 of 3)" in out
    assert "Trying again (3 of 3)" in out
    # Plain language: the person reads why and what happens next, not the command line.
    for internal in ("curl", "install.sh", "Traceback", "Exception", "exit", "returncode"):
        assert internal not in out.split("✔ Ollama ready")[0], internal


def test_ollama_that_never_runs_is_tried_three_times_in_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three attempts in total, then the honest failure -- never a fourth.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: OLLAMA_INSTALL_ATTEMPTS: Final = 3
    Becomes: OLLAMA_INSTALL_ATTEMPTS: Final = 4
    """
    installer = _FlakyInstaller(works_from=None)
    _flaky_linux(monkeypatch, installer)

    assert bootstrap.install_ollama_platform() is False
    assert installer.runs == 3


def test_an_installer_that_ran_out_of_time_is_not_run_again(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """After 15 minutes the person has waited enough; a retry would double it.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: except subprocess.TimeoutExpired:
    Becomes: except ():

    Mutated, the timeout falls to the generic branch and reports "could not run" instead.
    """
    installer = _FlakyInstaller(works_from=None)
    _flaky_linux(monkeypatch, installer)

    def timing_out(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        installer.runs += 1
        raise subprocess.TimeoutExpired("installer", 900)

    monkeypatch.setattr(bootstrap.subprocess, "run", timing_out)

    assert bootstrap.install_ollama_platform() is False
    assert installer.runs == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "did not finish in 15 minutes" in out
    assert "Trying again" not in out


# --- the model setup chose becomes the saved default ----------------------------------


def _ready_llm_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    def ready(**kwargs: object) -> bootstrap.LlmSetup:
        return bootstrap.LlmSetup(True, "qwen3:1.7b", "http://localhost:11434")

    monkeypatch.setattr(bootstrap, "setup_local_llm", ready)


def test_setup_saves_the_model_it_made_ready(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ucx install --yes` then `ucx run` works: the install saves what it set up.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: remember_local_llm(llm_result)
    Becomes: pass
    """
    from uclone_x.llm.connectors.saved_choice import read_saved_choice

    _ready_llm_setup(monkeypatch)

    bootstrap.run_local_setup(image=False, interactive=False, assume_yes=True)

    saved = read_saved_choice()
    assert saved is not None
    assert (saved.provider, saved.model, saved.base_url) == (
        "ollama",
        "qwen3:1.7b",
        "http://localhost:11434",
    )
    out = " ".join(capsys.readouterr().out.split())
    assert "Saved qwen3:1.7b as the default model for `ucx run`, rooms and the dashboard." in out


def _save_settings(**values: object) -> None:
    from uclone_x.llm.connectors.saved_choice import settings_file

    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(values))


def _setup_says(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> str:
    """Run setup with a ready `qwen3:1.7b` at localhost; what it printed, on single spaces."""
    _ready_llm_setup(monkeypatch)
    bootstrap.run_local_setup(image=False, interactive=False, assume_yes=True)
    out = " ".join(capsys.readouterr().out.split())
    for internal in ("Traceback", "Error", "llm_provider", "settings.json", "None"):
        assert internal not in out, internal
    return out


def test_setup_keeps_another_provider_already_saved_and_says_how_to_switch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Killed by: src/uclone_x/cli/commands/bootstrap.py :: if saved.provider != "ollama" or saved.model not in (None, result.model):
    Becomes: if False:
    """
    from uclone_x.llm.connectors.saved_choice import read_saved_choice

    _save_settings(llm_provider="openai", llm_model="gpt-4o-mini")

    out = _setup_says(monkeypatch, capsys)

    saved = read_saved_choice()
    assert saved is not None and saved.model == "gpt-4o-mini"
    assert (
        "gpt-4o-mini (openai) is already saved as the default model, so it stays the default "
        "rather than qwen3:1.7b, which setup just prepared. To use qwen3:1.7b instead, run "
        "`ucx llm use qwen3:1.7b` or choose it in the dashboard's Settings."
    ) in out


def test_setup_names_a_different_saved_ollama_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An earlier install saved `qwen3:8b`; this one prepared `qwen3:1.7b`.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if saved.provider != "ollama" or saved.model not in (None, result.model):
    Becomes: if saved.provider != "ollama":
    """
    _save_settings(llm_provider="ollama", llm_model="qwen3:8b")

    out = _setup_says(monkeypatch, capsys)

    assert "qwen3:8b is already saved as the default model" in out
    assert "`ucx llm use qwen3:1.7b`" in out


def test_setup_names_a_different_saved_address(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Killed by: src/uclone_x/cli/commands/bootstrap.py :: elif not _same_address(saved.base_url, result.endpoint):
    Becomes: elif False:
    """
    _save_settings(
        llm_provider="ollama", llm_model="qwen3:1.7b", llm_base_url="http://gpu-box:11434"
    )

    out = _setup_says(monkeypatch, capsys)

    assert (
        "The saved default asks for qwen3:1.7b at http://gpu-box:11434, but setup prepared it "
        "at http://localhost:11434. To use this one, run "
        "`ucx llm use qwen3:1.7b --base-url http://localhost:11434`."
    ) in out


def test_setup_says_it_kept_the_same_model_saved_by_an_earlier_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not "chosen in Settings": an earlier install is as likely to have saved it.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: console.print(f"[dim]Kept {model}, the model already saved as the default.[/dim]")
    Becomes: pass
    """
    _save_settings(
        llm_provider="ollama", llm_model="qwen3:1.7b", llm_base_url="http://localhost:11434/"
    )

    out = _setup_says(monkeypatch, capsys)

    assert "Kept qwen3:1.7b, the model already saved as the default." in out
    assert "Settings" not in out


def test_a_read_only_session_directory_still_says_kept(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the model already saved there is nothing to write, so nothing is opened for writing.

    A session directory left read-only (or owned by root after a `sudo` install) made the
    lock file impossible to create, and a re-install that changed nothing reported a
    failure instead of "Kept".

    Two things each keep this passing, so no single mutation fails it: nothing is locked
    when there is nothing to write (declared on the test below), and a lock that cannot
    be opened falls back to an unlocked write (declared in `test_llm_saved_choice.py`).
    """
    from uclone_x.llm.connectors.saved_choice import lock_file, settings_file

    if os.geteuid() == 0:
        pytest.skip("root writes into a read-only directory")
    _save_settings(
        llm_provider="ollama", llm_model="qwen3:1.7b", llm_base_url="http://localhost:11434/"
    )
    directory = settings_file().parent
    directory.chmod(0o500)
    try:
        out = _setup_says(monkeypatch, capsys)
    finally:
        directory.chmod(0o700)

    assert "Kept qwen3:1.7b, the model already saved as the default." in out
    assert not lock_file(settings_file()).exists()


def test_a_re_install_that_changes_nothing_does_not_touch_the_lock(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Killed by: src/uclone_x/llm/connectors/saved_choice.py :: if not pending:
    Becomes: if False:
    """
    from uclone_x.llm.connectors.saved_choice import lock_file, settings_file

    _save_settings(
        llm_provider="ollama", llm_model="qwen3:1.7b", llm_base_url="http://localhost:11434/"
    )

    out = _setup_says(monkeypatch, capsys)

    assert "Kept qwen3:1.7b, the model already saved as the default." in out
    assert not lock_file(settings_file()).exists()


def test_a_settings_file_that_cannot_be_read_is_reported_plainly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The setup still finishes, and the message says what to do, not what raised.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: except (OSError, ValueError):
    Becomes: except OSError:
    """
    from uclone_x.llm.connectors.saved_choice import settings_file

    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{half written")
    _ready_llm_setup(monkeypatch)

    result = bootstrap.run_local_setup(image=False, interactive=False, assume_yes=True)

    assert result.llm is not None and result.llm.ready
    assert target.read_text() == "{half written"
    out = " ".join(capsys.readouterr().out.split())
    assert "Could not save this model as the default" in out
    assert "--provider ollama --model qwen3:1.7b" in out
    for internal in ("ValueError", "OSError", "Traceback", "JSON", "could not be read, so"):
        assert internal not in out, internal
