"""Console-script entry point for `ucx`.

The `cli` extra is optional, so `pip install uclone-x` installs this script
next to a shell whose framework is absent. Importing `uclone_x.cli` then raises
`MissingDependencyError`, which is correct — P6 requires the failure to be
named at composition time rather than fixed up silently — but a process launched
from a shell prompt is a different boundary from an import inside a program.
Raising through the console script prints twenty lines of traceback whose last
line is the one-line instruction the user needed.

So the exception is caught here, at the outermost frame, and reported as a
message. Nothing is substituted and nothing continues: the exit status is
non-zero and the shell never starts.

The same holds while a command *runs*, not only while the CLI is imported
(#926). A command that defers an optional import into its body — so that
`ucx --help` does not need that extra (#656, #881) — raises the same error
from inside `app()`. Typer does not intercept it: `Typer.__call__` re-raises
every exception and leaves the Rich traceback to `sys.excepthook`, so catching
it around `app()` is what stops that traceback, for every command at once.
Only `MissingDependencyError` and `ExtensionError` are caught; any other
exception keeps its traceback, because reporting a defect as a one-line message
discards the evidence. An `ExtensionError` is an installed extension that could
not load, not a defect in `ucx`: its message is written for a person (#2283),
and the cause it chains -- the extension's own exception and file -- goes to
`ucx.log` -- logged by the registry where it failed, once (#2290, #2292).

Placement is load-bearing twice over, and both constraints were found by a
test rather than by reasoning:

* Not inside `uclone_x.cli`. The guard that raises is
  `uclone_x/cli/__init__.py`, which runs when *any* module under that package
  is imported — so a launcher there would raise while being imported, before
  its own `try` block existed, producing the exact traceback it exists to
  prevent.
* Not at the top of `uclone_x` either. A bare `*.py` beside `errors.py` counts
  as kernel under the layering rule, and the kernel may not import a shell.
  Starting the shell is a shell's job, so it belongs here, beside
  `a2a_server.py`.
"""

from __future__ import annotations

import logging
import sys

from uclone_x.core.logging_setup import log_to_file_not_terminal, reason_is_in_the_log
from uclone_x.errors import MissingDependencyError
from uclone_x.extensions.contract import ExtensionError

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the `ucx` CLI, or report the extra that has to be installed first."""
    try:
        from uclone_x.cli.main import app

        app()
    except MissingDependencyError as exc:
        print(f"ucx: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except ExtensionError as exc:
        # Its words are written for a person (#2283); the cause it chains -- the
        # extension's own exception, its file -- is not. Where the registry logged that
        # cause as the extension failed (with where its entry point points; that record
        # serves the web head too) it is not logged again (#2292). Any other -- a
        # conflict, or an error from a part the registry does not wrap -- has its
        # traceback here, the one record of it, since the printed line says the reason
        # is in the log (#2290, #2294 review).
        with log_to_file_not_terminal():
            logger.error(
                "ucx: an extension stopped the command: %s",
                exc,
                exc_info=None if exc.cause_logged else exc,
            )
        print(f"ucx: {exc} {reason_is_in_the_log()}", file=sys.stderr)
        raise SystemExit(1) from exc
