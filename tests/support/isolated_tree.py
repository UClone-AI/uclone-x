"""A private copy of a working tree, for mutation checks that must not write into the live one."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def isolated_copy_of_tree(source: Path, destination: Path) -> Path:
    """Clone `source` into `destination` as it stands on disk, and return `destination`.

    `swarm.core.mutation.run_isolated_mutation` writes each mutant into the target file and
    restores it in a `finally`. Against the live tree that is safe only while nothing else reads
    the tree during the window. That held while the suite ran in one process, and stopped
    holding when the gate began running it on several workers (#967): at 16 workers,
    `test_every_declaration_still_matches_its_target` on another worker read the mutant of
    `evals/suites/ontology.py` and failed, and any worker importing or reading a target inside
    the window would do the same.

    So mutations go into this copy instead. It is a `--shared` clone — objects borrowed from
    `source`, not copied — checked out at `source`'s `HEAD`, because the harness reads pristine
    content with `git show HEAD:<path>` and so needs a repository. Every uncommitted change is
    then laid over it: modified and untracked files copied, deleted ones removed. The gate runs
    over a dirty tree, and the declarations it checks are usually part of the uncommitted work.

    **Against `HEAD`, not against the index.** This read `git ls-files --modified`, which
    compares the working tree to the *index*, so a change that had been staged was invisible
    here and the copy was `HEAD`. The gate's own recipe stages before it runs -- `git add -A`, so
    that the check which reads every git-tracked file from disk does not trip over a rebuilt
    asset -- which made that the ordinary case rather than a corner: every declaration the
    lethality ratchet scored was then applied to the pre-change source and
    run against the pre-change tests. A new test simply failed to resolve, but an edited
    assertion on an existing test scored `KILLED` against the old file -- a green verdict about
    code nobody ran. `--no-renames` is set so that a staged rename arrives as the delete and the
    add it is, rather than as one path that would leave the old file in the copy.
    """
    head = _git("-C", str(source), "rev-parse", "HEAD").strip()
    _git("clone", "--quiet", "--shared", "--no-checkout", str(source), str(destination))
    _git("-C", str(destination), "checkout", "--quiet", "--detach", head)
    changed = _git("-C", str(source), "diff", "--name-only", "--no-renames", "-z", "HEAD")
    untracked = _git("-C", str(source), "ls-files", "-z", "--others", "--exclude-standard")
    for relative in sorted({entry for entry in (changed + untracked).split("\0") if entry}):
        on_disk = source / relative
        copied = destination / relative
        if on_disk.is_file():
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(on_disk, copied)
        elif not on_disk.exists():
            copied.unlink(missing_ok=True)
    return destination
