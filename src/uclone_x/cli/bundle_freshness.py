"""Whether the committed UI bundle is what the current frontend source builds (#878).

`frontend/vite.config.ts` builds into `src/uclone_x/ui_static`, and that directory is
**committed**: it ships in the package, `ucx ui` serves it, and the browser suite's fixture
(`tests/e2e/conftest.py`) serves exactly it. So the E2E suite tests the *artifact*, and
nothing checked that the artifact still agreed with the source. Rebuilding every commit that
touched `frontend/` or `ui_static` and has `fc7b6ec`'s frontend dependencies (21 of them)
found four where it did not — `ba4c3e9`, `4007b98`, `c3ff0a8` and `f2acb5d`: a source edit
committed without a rebuild, so the served UI was older than the tree that carried it.

Design — the smallest check that makes that divergence unmergeable:

* **Build into a temporary directory and compare it with the committed one, byte for
  byte, file by file.** `vite build --outDir <tmp>` from `frontend/`. The committed
  directory is never written: a gate that rebuilt it would rewrite tracked files on every
  run and settle any divergence by overwriting the evidence of it.
* **Nothing is ignored.** A comparison with an ignore list is only as good as the proof
  that what it ignores is noise, so the determinism was measured before choosing: two
  builds of one tree were byte-identical, a build from a copy of `frontend/` at a
  different absolute path was byte-identical to both, and each matched the committed
  bundle at `fc7b6ec`. Vite hashes asset names from content and embeds no timestamp or
  absolute path here. Not measured: a build on Linux, which the published repository's CI
  performs; if the output ever depends on the platform, that CI goes red on its first run,
  and that is where the finding belongs.
* **Two inputs from outside the tree change the output, and both are pinned** — on the
  build here, in the rebuild command the failure prints, and on `-fe`'s build. Unpinned,
  either one turned a fresh tree `stale`, and the printed fix committed a bundle every
  other machine rejects (reviews of PR #956):
  * **`NODE_ENV=production`.** `development` in the shell or in an untracked
    `frontend/.env.local` makes vite emit React's development build (main chunk
    922 KB → 1.55 MB). A process-environment value wins over `.env*` files.
  * **`BROWSERSLIST=defaults`.** `frontend/postcss.config.js` runs autoprefixer, whose
    browser targets come from browserslist. Browserslist reads a `BROWSERSLIST` variable, or
    a `BROWSERSLIST_CONFIG` file, or a `.browserslistrc` / `browserslist` file /
    `package.json` key in `frontend/` **or any directory above it**. So a config in a home
    directory changed the CSS prefixes and the hashed names. The variable is read before
    every one of those files, and `defaults` is what browserslist uses when it finds no
    config, so the pin gives the committed bytes. `BROWSERSLIST_ENV` only picks a section
    of a config file, which the pin never reads. The cost is that a `browserslist` key added
    to `frontend/package.json` is overridden too: to change targets, change this pin.
    Measured with browserslist 4.28.8 and autoprefixer 10.5.4.
* **Symlinks and other non-regular entries are differences.** Vite never emits one, and a
  link to a file with the right content, or a linked directory `rglob` does not descend
  into, would otherwise compare equal or invisible.
* **Only the bundler runs.** `npm run build` is `tsc && vite build`; `tsc` has `noEmit`,
  so it changes nothing emitted and is left to `-fe`. Measured at `fc7b6ec`: about 1.5 s.
* **It cannot pass by not running** (P6). No `npm`, no `node_modules`, a build that fails,
  a build that exits 0 having emitted nothing into the temporary directory, and a build that
  changed the committed directory are each a failure with its own message, never a pass.
  The last two are the forms a bundler ignoring `--outDir` would take: it would leave the
  temporary directory empty, or fill both, and the second compares `fresh` by construction.
  A committed bundle with no `frontend/package.json` beside it is a failure too — there is
  no source it could be checked against.

What it judges is the **working tree**, like every other gate stage: a bundle rebuilt but
not staged passes here and still reaches the commit stale. Staging the rebuild is the
ui-authoring skill's step, and `git status` shows it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from rich.markup import escape

#: The committed bundle, relative to the repository root. `frontend/vite.config.ts`'s
#: `build.outDir` names the same directory. A string rather than a `Path`, and joined to a
#: root inside the call: a relative `Path` constant resolves against whatever the cwd is (#436).
BUNDLE_DIR: Final = "src/uclone_x/ui_static"

#: What `check_bundle_freshness` found. `absent` is "does not apply" and passes with a note;
#: every other non-`fresh` answer is a failure, and each prints a different cause.
BundleStatus = Literal[
    "fresh",
    "stale",
    "absent",
    "source-missing",
    "npm-missing",
    "deps-missing",
    "build-failed",
    "bundle-written",
]

#: The gate's exit code for any failed answer here. Named, not a bare `1`, so a reader
#: tracing the gate's exit code lands on this stage.
_BUNDLE_FAILURE_EXIT: Final = 1

#: How many names of each kind of difference to print before summarising the rest.
_NAMES_SHOWN: Final = 10

#: Pinned for the build this stage runs, for the rebuild it tells a person to run, and for the
#: gate's own `-fe` build (stage 7), which writes the committed directory. They must agree.
#: An unpinned `npm run build` under `NODE_ENV=development`, or under a browserslist config
#: from a parent directory, commits a bundle that turns the machine that built it green and
#: every other one red. Why each value, and why the value pins: the module docstring.
BUILD_ENVIRONMENT: Final = {"NODE_ENV": "production", "BROWSERSLIST": "defaults"}

#: The rebuild the failure prints. It spells out every pin above, so it cannot disagree with
#: them.
_REBUILD: Final = (
    "(cd frontend && "
    + " ".join(f"{name}={value}" for name, value in BUILD_ENVIRONMENT.items())
    + " npm run build) && git add -A src/uclone_x/ui_static"
)


@dataclass(frozen=True)
class BundleDifference:
    """How a freshly built bundle differs from the committed one, as relative POSIX paths."""

    #: Built from the current source, absent from the committed bundle.
    not_committed: tuple[str, ...]
    #: In the committed bundle, not produced by the current source.
    not_built: tuple[str, ...]
    #: In both, with different bytes.
    changed: tuple[str, ...]

    @property
    def identical(self) -> bool:
        return not (self.not_committed or self.not_built or self.changed)


def _digests(tree: Path) -> dict[str, str]:
    """Every entry under `tree` but real directories, by relative POSIX path, to a digest.

    A regular file digests to the SHA-256 of its content. A symlink — to a file, to a
    directory, or dangling — digests to its target text, and anything else to its kind, so
    neither can equal a built file: vite emits neither. A `tree` that does not exist yields
    nothing: `rglob` on a missing directory is empty.
    """
    digests: dict[str, str] = {}
    for path in tree.rglob("*"):
        name = path.relative_to(tree).as_posix()
        if path.is_symlink():
            digests[name] = f"symlink -> {os.readlink(path)}"
        elif path.is_file():
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif not path.is_dir():
            digests[name] = "not a regular file"
    return digests


def compare_bundles(built: Path, committed: Path) -> BundleDifference:
    """Compare two bundle directories by the full content of every file, recursively.

    A missing `committed` directory compares as empty, so every built file is reported as
    not committed rather than the comparison raising.
    """
    built_files = _digests(built)
    committed_files = _digests(committed)
    return BundleDifference(
        not_committed=tuple(sorted(built_files.keys() - committed_files.keys())),
        not_built=tuple(sorted(committed_files.keys() - built_files.keys())),
        changed=tuple(
            sorted(
                name
                for name in built_files.keys() & committed_files.keys()
                if built_files[name] != committed_files[name]
            )
        ),
    )


def _named(label: str, names: tuple[str, ...]) -> list[str]:
    if not names:
        return []
    lines = [f"[red]  {label}:[/red]"]
    lines += [f"[red]    {escape(name)}[/red]" for name in names[:_NAMES_SHOWN]]
    if len(names) > _NAMES_SHOWN:
        lines.append(f"[red]    … and {len(names) - _NAMES_SHOWN} more[/red]")
    return lines


def check_bundle_freshness(root: Path | None = None) -> tuple[BundleStatus, int, list[str]]:
    """Build the frontend into a temporary directory and compare it with the committed bundle.

    Args:
        root: Repository root. Defaults to the process cwd, matching the gate's other stages.

    Returns:
        The status, the exit code the gate should record (0 for `fresh` and `absent`), and
        the lines to print.
    """
    base = Path.cwd() if root is None else root
    frontend = base / "frontend"
    committed = base / BUNDLE_DIR
    if not (frontend / "package.json").is_file():
        if committed.exists():
            return (
                "source-missing",
                _BUNDLE_FAILURE_EXIT,
                [
                    f"[bold red]✖ {BUNDLE_DIR} exists but frontend/package.json does not, so "
                    "there is no source to check the bundle against.[/bold red]",
                ],
            )
        return (
            "absent",
            0,
            [f"[yellow]! No frontend/package.json in {base}; bundle not checked.[/yellow]"],
        )

    unverifiable = (
        f"[red]  {BUNDLE_DIR} cannot be compared with its source, so it is reported as a "
        "failure and not as a pass.[/red]"
    )
    if shutil.which("npm") is None:
        return (
            "npm-missing",
            _BUNDLE_FAILURE_EXIT,
            ["[bold red]✖ `npm` was not found on PATH.[/bold red]", unverifiable],
        )
    if not (frontend / "node_modules").is_dir():
        return (
            "deps-missing",
            _BUNDLE_FAILURE_EXIT,
            [
                "[bold red]✖ frontend/node_modules is missing, so the bundle cannot be "
                "rebuilt.[/bold red]",
                unverifiable,
                "[red]  Fix: the same as for the vitest suite (stage 6) — link the primary "
                "workspace's node_modules, or run `npm ci --prefix frontend`.[/red]",
            ],
        )

    with tempfile.TemporaryDirectory(prefix="ucx-bundle-freshness-") as scratch:
        built = Path(scratch) / "ui_static"
        command = [
            "npm",
            "exec",
            "--no",
            "--",
            "vite",
            "build",
            "--outDir",
            str(built),
            "--emptyOutDir",
            "--logLevel",
            "error",
        ]
        # Read before the build so a build that writes the committed directory is caught:
        # comparing afterwards alone would pass it by construction.
        committed_before = _digests(committed)
        try:
            build = subprocess.run(
                command,
                cwd=frontend,
                capture_output=True,
                text=True,
                env={**os.environ, **BUILD_ENVIRONMENT},
            )
        except OSError as exc:
            return (
                "npm-missing",
                _BUNDLE_FAILURE_EXIT,
                [
                    f"[bold red]✖ `npm` could not be run: {type(exc).__name__}: "
                    f"{escape(str(exc))}[/bold red]",
                    unverifiable,
                ],
            )
        if build.returncode != 0:
            output = (build.stderr.strip() or build.stdout.strip())[-2000:]
            return (
                "build-failed",
                build.returncode,
                [
                    f"[bold red]✖ The frontend build failed (exit {build.returncode}).[/bold red]",
                    unverifiable,
                    # Escaped: build output is full of `[plugin:name]` prefixes, which
                    # Rich would otherwise parse as markup and can raise on.
                    f"[dim]{escape(output)}[/dim]",
                ],
            )
        if _digests(committed) != committed_before:
            return (
                "bundle-written",
                _BUNDLE_FAILURE_EXIT,
                [
                    f"[bold red]✖ The frontend build changed {BUNDLE_DIR}, which it was told "
                    "not to write.[/bold red]",
                    unverifiable,
                    "[red]  Restore it from git before reading any comparison: "
                    "git checkout -- src/uclone_x/ui_static[/red]",
                ],
            )
        if not _digests(built):
            return (
                "build-failed",
                _BUNDLE_FAILURE_EXIT,
                [
                    f"[bold red]✖ The frontend build exited 0 but emitted nothing into {built}."
                    "[/bold red]",
                    unverifiable,
                ],
            )
        difference = compare_bundles(built, committed)

    if difference.identical:
        return "fresh", 0, [f"[green]✔ {BUNDLE_DIR} is what frontend/ builds.[/green]"]
    return (
        "stale",
        _BUNDLE_FAILURE_EXIT,
        [
            f"[bold red]✖ {BUNDLE_DIR} is stale: it is not what frontend/ currently builds."
            "[/bold red]",
            "[red]  It ships in the package and the browser suite serves it, so both would be "
            "running a UI the source no longer describes (#878).[/red]",
            *_named("Changed", difference.changed),
            *_named("Built from the source but not committed", difference.not_committed),
            *_named(f"In {BUNDLE_DIR} but not built from the source", difference.not_built),
            f"[red]  Fix: {_REBUILD} — then commit it.[/red]",
        ],
    )
