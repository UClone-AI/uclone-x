"""What the committed UI bundle was built from, as a digest of the inputs (#1075).

`ucx ui` serves the committed `src/uclone_x/ui_static`, and the launcher has to decide
whether that bundle is still what `frontend/` builds before it serves it. That is a
**content** question. It used to be answered with a **timestamp** — the newest mtime under
`frontend/src` against `ui_static/index.html` — and a timestamp is wrong in both
directions at once:

* `git checkout`, `git stash pop`, a rebase and a fresh clone all stamp every file with
  the checkout time, so a **pristine tree at a known SHA read as stale** and rebuilt to
  byte-identical output. Because `frontend/vite.config.ts` builds into the committed
  directory with `emptyOutDir: true`, every one of those false positives **rewrote tracked
  files**.
* The mtime walk covered `frontend/src` and nothing else, so `package-lock.json`,
  `vite.config.ts`, `tailwind.config.js`, `postcss.config.js`, `tsconfig*.json` and
  `frontend/index.html` — all of which change the bundle's bytes — left a genuinely stale
  bundle looking fresh.
* It covered `*.test.tsx` and `*.spec.ts`, which the bundler never emits, so editing a test
  triggered a production rebuild (#1067).

## The shape, and why it is this one

The repository already answers this question correctly: **gate stage 6b**
(`uclone_x.cli.bundle_freshness`) builds `frontend/` into a temporary directory and compares
it with the committed bundle byte for byte. That is the ground truth, and it costs a real
`vite build`. This module is the same question made cheap enough for a launcher: hash the
**inputs** rather than rebuild the **output**, and record the answer beside the bundle so a
checkout can be recognised as matching without running anything.

**The record cannot live inside `src/uclone_x/ui_static`, and that is not a naming
preference.** Two independent things forbid it. `emptyOutDir: true` means the next build
deletes it. And stage 6b compares the two directories with **nothing ignored** by design,
so a file in the committed bundle that `vite build` does not emit is reported as
`not_built` and turns the gate red. So the record sits beside the bundle, not in it:
`src/uclone_x/ui_build_inputs.sha256`, one hex digest and a newline.

## What is hashed

`CONFIG_INPUTS` is an explicit list rather than "every file in `frontend/`" so that an
untracked `.env.local` in someone's working copy cannot change the digest the repository
records. A fitness check in the development repository compares the list with the files
git actually tracks at that level, so a config file added later is a failing test naming
itself rather than a silent gap.

A file that does not exist contributes the marker `absent`, so **deleting** an input
changes the digest as surely as editing one does. Symlinks and other non-regular entries
contribute their kind, as they do in stage 6b's `_digests`, so a link cannot be mistaken
for the file it points at.

## What is deliberately *not* hashed

**`NODE_ENV` and `BROWSERSLIST`.** Both change the emitted bytes — stage 6b pins them for
exactly that reason — but they are process environment, not files. Folding them in would
make the recorded digest depend on the machine that wrote it, so a developer with
`NODE_ENV=development` exported would rebuild on every single launch and **never
converge**, because the rebuild cannot change the environment it ran under. That is the
same non-convergence the old `st_size < 10000` heuristic risked, and it is the one failure
mode worth refusing outright. Environment pinning is stage 6b's job and stays there.

**Test and spec files.** `*.test.*`, `*.spec.*`, `__tests__/` and `__snapshots__/` are
excluded: `vite build` reads none of them (`frontend/vite.config.ts` confines them to the
`test.include` glob), so they cannot change the bundle. `src/test/setup.ts` is *not*
excluded — it is vitest-only in fact, but the exclusion rule here is the one a reader can
check by name, and widening it to a directory called `test` would start guessing.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Final

#: The digest record, relative to the directory that holds the bundle. Beside
#: `ui_static`, never inside it — the module docstring gives the two reasons.
RECORD_NAME: Final = "ui_build_inputs.sha256"

#: Files directly under `frontend/` that change what `vite build` emits. Explicit, so an
#: untracked file in a working copy cannot move the digest; checked against `git ls-files`
#: by a fitness test in the development repository, so a new one cannot be forgotten.
CONFIG_INPUTS: Final = (
    "index.html",
    "package.json",
    "package-lock.json",
    "postcss.config.js",
    "tailwind.config.js",
    "tsconfig.json",
    "tsconfig.node.json",
    "vite.config.ts",
)

#: Directories under `frontend/` walked in full, minus the exclusions below. `public/` is
#: copied into the output verbatim by vite, so it belongs here even when it does not exist.
SOURCE_TREES: Final = ("src", "public")

#: Directory names under `SOURCE_TREES` whose contents the bundler never reads.
EXCLUDED_DIRECTORIES: Final = ("__tests__", "__snapshots__")

#: Filename infixes the bundler never reads. Matched on the name, not the path, so
#: `Foo.test.tsx` and `rooms.spec.ts` are excluded wherever they sit.
EXCLUDED_INFIXES: Final = (".test.", ".spec.")

#: What an input that is not there contributes. A missing file must be distinguishable
#: from an empty one, or deleting `tailwind.config.js` would leave the digest unmoved.
_ABSENT: Final = "absent"


def is_bundler_input(relative_path: Path) -> bool:
    """Whether a path under a `SOURCE_TREES` directory is something `vite build` reads."""
    if any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts):
        return False
    return not any(infix in relative_path.name for infix in EXCLUDED_INFIXES)


def _entry_digest(path: Path) -> str:
    """The content of one input, as stage 6b's `_digests` characterises a bundle entry."""
    if path.is_symlink():
        return f"symlink -> {os.readlink(path)}"
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.exists():
        return "not a regular file"
    return _ABSENT


def build_inputs(frontend_dir: Path) -> dict[str, str]:
    """Every build input under `frontend_dir`, by relative POSIX path, to its digest.

    Deterministic and free of any timestamp: the same tree yields the same mapping on a
    fresh clone, after a rebase, and on another machine.
    """
    inputs: dict[str, str] = {name: _entry_digest(frontend_dir / name) for name in CONFIG_INPUTS}
    for tree in SOURCE_TREES:
        root = frontend_dir / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(frontend_dir)
            if not is_bundler_input(relative):
                continue
            if path.is_dir() and not path.is_symlink():
                continue
            inputs[relative.as_posix()] = _entry_digest(path)
    return inputs


def build_input_digest(frontend_dir: Path) -> str:
    """One hex digest over every build input under `frontend_dir`.

    The path is hashed with the content, so moving a component to a new name changes the
    digest even though the bytes of the tree as a whole did not.
    """
    inputs = build_inputs(frontend_dir)
    running = hashlib.sha256()
    for name in sorted(inputs):
        running.update(f"{name}\0{inputs[name]}\n".encode())
    return running.hexdigest()


def recorded_digest(bundle_parent: Path) -> str | None:
    """The digest recorded beside the bundle, or `None` when there is no readable record.

    `None` is "no answer", never "matches": a missing or unreadable record means the
    launcher cannot tell, and telling it to rebuild is the only answer that cannot serve
    the wrong code.
    """
    record = bundle_parent / RECORD_NAME
    try:
        text = record.read_text(encoding="utf-8")
    except OSError:
        return None
    digest = text.strip()
    return digest or None


def write_record(frontend_dir: Path, bundle_parent: Path) -> str:
    """Record the current inputs' digest beside the bundle, and return it.

    Written by the launcher after a build it ran itself, so that the record always travels
    with the bundle it describes. It is a tracked file, and it is written only in the case
    where `vite build` has just rewritten the tracked bundle anyway.
    """
    digest = build_input_digest(frontend_dir)
    (bundle_parent / RECORD_NAME).write_text(f"{digest}\n", encoding="utf-8")
    return digest
