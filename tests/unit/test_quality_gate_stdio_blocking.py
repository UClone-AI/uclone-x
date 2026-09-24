"""The gate's stages must not inherit, or leave behind, a non-blocking stdout (#993).

`O_NONBLOCK` is shared by every process holding the same pipe. `git push` over SSH ran the
gate while `ssh` held the pipe non-blocking, and pytest's terminal writer raised
`BlockingIOError` as soon as the pipe filled; a `node` stage that is killed leaves the flag
set for whatever the gate writes next. See `restore_blocking_stdio` in
`src/uclone_x/cli/quality_gate.py`.

Each test runs a small driver in a child interpreter whose stdout is a pipe this test does
not read until the driver has had time to fill it, which is the slow `| tail` or `| grep`
of the report. The driver imports `quality_gate` from the same source tree as this test, so
a mutation of that file is what the driver runs.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from uclone_x.cli import quality_gate

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="O_NONBLOCK on an inherited pipe is a POSIX behaviour"
)

#: Larger than a default pipe buffer on macOS and Linux (64 KiB), so the write must wait.
_PAYLOAD_BYTES = 2 * 1024 * 1024

#: How long the pipe stays unread after the driver says it is about to write. A write into
#: a non-blocking full pipe fails within microseconds, so this only has to outlast that.
_STALL_SECONDS = 0.5

_TIMEOUT_SECONDS = 60.0

_MARKER = "about to write"


def _source_root() -> Path:
    """The `src` directory `quality_gate` was imported from (the worktree's, under `./ucx`)."""
    return Path(quality_gate.__file__).resolve().parents[2]


def _child_env() -> dict[str, str]:
    """The environment for a child interpreter that imports this tree's `quality_gate`."""
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(_source_root()), os.environ.get("PYTHONPATH", "")]),
    }


def _run_behind_a_stalled_reader(driver: str, *, stalled_fd: int = 1) -> tuple[int, int, str]:
    """Run `driver` with one stream on a pipe nobody reads for a while; return (rc, bytes, text).

    stdout and stderr are two distinct pipes. The driver writes `_MARKER` on the other stream
    just before its large write to `stalled_fd`. This waits for that line, leaves `stalled_fd`
    unread for `_STALL_SECONDS`, then drains both. `bytes` counts what arrived on `stalled_fd`;
    `text` is the other stream, then whatever the stalled one carried besides its `x` payload
    (a traceback, when the write failed).
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(driver)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_env(),
    )
    signal = proc.stderr if stalled_fd == 1 else proc.stdout
    assert signal is not None
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    early = b""
    while _MARKER.encode() not in early:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.kill()
            proc.communicate()
            pytest.fail(f"the driver never reached its write; output so far: {early!r}")
        ready, _, _ = select.select([signal], [], [], remaining)
        if ready:
            chunk = os.read(signal.fileno(), 4096)
            if not chunk:
                break
            early += chunk
    time.sleep(_STALL_SECONDS)
    try:
        stdout, stderr = proc.communicate(timeout=_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        pytest.fail("the driver did not finish after its output was drained")
    stalled, other = (stdout, early + stderr) if stalled_fd == 1 else (stderr, early + stdout)
    text = (other + stalled.lstrip(b"x")).decode(errors="replace")
    return proc.returncode, len(stalled), text


def test_a_stage_does_not_inherit_a_pipe_another_process_made_non_blocking() -> None:
    """A stage run after a sibling made the shared stdout non-blocking writes a full pipe safely.

    The driver stands in for the gate under `git push`: it makes its own stdout non-blocking,
    as the `ssh` transport sharing that pipe did, then runs a stage that writes more than the
    pipe holds, as pytest's report does. Without the reset the stage dies with the
    `BlockingIOError` from #993.

    Killed by: src/uclone_x/cli/quality_gate.py :: restore_blocking_stdio()  # before
    Becomes: None  # before
    """
    returncode, written, stderr = _run_behind_a_stalled_reader(
        f"""
        import os, sys
        from uclone_x.cli import quality_gate

        os.set_blocking(1, False)
        stage = (
            "import sys\\n"
            "sys.stderr.write({_MARKER!r} + '\\\\n'); sys.stderr.flush()\\n"
            "sys.stdout.buffer.write(b'x' * {_PAYLOAD_BYTES}); sys.stdout.buffer.flush()\\n"
        )
        sys.exit(quality_gate.run_stage([sys.executable, "-c", stage]).returncode)
        """
    )

    assert "BlockingIOError" not in stderr
    assert (returncode, written) == (0, _PAYLOAD_BYTES), stderr


def test_a_stage_does_not_inherit_a_non_blocking_stderr_on_a_pipe_of_its_own() -> None:
    """A stderr made non-blocking on its own pipe is restored before the stage writes to it.

    Under `2>&1` fd 1 and fd 2 share one pipe, so restoring fd 1 restores both and nothing
    would notice fd 2 being dropped. A caller that captures the two streams separately, or an
    `ssh` holding only the inherited stderr, depends on fd 2 itself being restored (#999). The
    driver checks that its stdout stayed blocking, which proves the two are distinct pipes.

    Killed by: src/uclone_x/cli/quality_gate.py :: _STD_STREAM_FDS: Final[tuple[int, ...]] = (1, 2)
    Becomes: _STD_STREAM_FDS: Final[tuple[int, ...]] = (1,)
    """
    returncode, written, text = _run_behind_a_stalled_reader(
        f"""
        import os, sys
        from uclone_x.cli import quality_gate

        os.set_blocking(2, False)
        assert os.get_blocking(1), "stdout and stderr share one pipe"
        stage = (
            "import sys\\n"
            "sys.stdout.write({_MARKER!r} + '\\\\n'); sys.stdout.flush()\\n"
            "sys.stderr.buffer.write(b'x' * {_PAYLOAD_BYTES}); sys.stderr.buffer.flush()\\n"
        )
        sys.exit(quality_gate.run_stage([sys.executable, "-c", stage]).returncode)
        """,
        stalled_fd=2,
    )

    assert "BlockingIOError" not in text
    assert (returncode, written) == (0, _PAYLOAD_BYTES), text


def test_the_gate_writes_safely_after_a_stage_left_stdout_non_blocking() -> None:
    """After a stage exits leaving the shared stdout non-blocking, the gate's own write waits.

    The stage sets the flag and exits without clearing it, which is what a killed `node`
    process does. The gate then writes more than the pipe holds, as its report lines and the
    next stage do. The restore is reported on stderr, not done silently.

    Killed by: src/uclone_x/cli/quality_gate.py :: left_non_blocking = restore_blocking_stdio()
    Becomes: left_non_blocking: list[int] = []
    """
    returncode, written, stderr = _run_behind_a_stalled_reader(
        f"""
        import os, sys
        from uclone_x.cli import quality_gate

        quality_gate.run_stage([sys.executable, "-c", "import os; os.set_blocking(1, False)"])
        sys.stderr.write({_MARKER!r} + "\\n"); sys.stderr.flush()
        sys.stdout.buffer.write(b"x" * {_PAYLOAD_BYTES}); sys.stdout.buffer.flush()
        """
    )

    assert "BlockingIOError" not in stderr
    assert (returncode, written) == (0, _PAYLOAD_BYTES), stderr
    assert "stdout was non-blocking after" in stderr


def test_restore_blocking_stdio_reports_only_the_descriptors_it_changed() -> None:
    """Returns the descriptors that were non-blocking, and leaves blocking ones alone."""
    read_a, write_a = os.pipe()
    read_b, write_b = os.pipe()
    try:
        os.set_blocking(write_a, False)

        assert quality_gate.restore_blocking_stdio((write_a, write_b)) == [write_a]
        assert os.get_blocking(write_a) and os.get_blocking(write_b)
    finally:
        for fd in (read_a, write_a, read_b, write_b):
            os.close(fd)


def test_restore_blocking_stdio_skips_a_closed_stream_and_restores_the_other() -> None:
    """A closed stderr has no writer to protect: it is skipped, and stdout is still restored.

    Run in a child interpreter that closes its own fd 2 after its imports and has no other
    thread, so nothing can open a descriptor and take the number between the close and the
    check. In this process a pytest thread could, and the check would then see an open
    descriptor instead of a closed one (#999).

    Killed by: src/uclone_x/cli/quality_gate.py :: if exc.errno == errno.EBADF:
    Becomes: if exc.errno == -1:
    """
    driver = """
        import os, sys, threading
        from uclone_x.cli import quality_gate

        assert threading.active_count() == 1, threading.enumerate()
        os.set_blocking(1, False)
        os.close(2)
        try:
            outcome = repr((quality_gate.restore_blocking_stdio(), os.get_blocking(1)))
        except OSError as exc:
            outcome = f"raised {exc!r}"
        sys.stdout.write(outcome + "\\n")
        """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(driver)],
        capture_output=True,
        env=_child_env(),
        timeout=_TIMEOUT_SECONDS,
    )

    assert (result.returncode, result.stdout.decode()) == (0, "([1], True)\n"), result.stderr
