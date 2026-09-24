# Contributing to UClone-X

Thank you for considering a contribution. Please read how this repository is
maintained first, because it is unusual and it affects what happens to your
pull request.

## How this repository is maintained

UClone-X is developed in a private repository and published here as periodic
snapshots. This has two consequences worth stating plainly:

1. **Your pull request will not be merged with the green button.** A maintainer
   applies the change in the development repository, with your authorship
   preserved (`Co-authored-by`, or you as the commit author), and it appears
   here in the next snapshot. The pull request is then closed with a link to
   the commit that carries it.
2. **Snapshots overwrite the published tree.** Do not build long-lived branches
   on top of a published commit expecting a linear history; rebase onto the
   latest snapshot instead.

If that model does not work for your change, open an issue first and say so —
for a large contribution we can arrange something better.

## Before you open a pull request

Run the gate. It is the same gate the maintainers run, and it is the only
automated check:

```bash
uv sync --all-extras
npm ci --prefix frontend
./ucx test check
```

That runs `ruff format --check`, `ruff check`, `pyright` in strict mode,
`pytest` with branch coverage, and the frontend's `vitest` suite. All five must
pass — and the last one must actually run: without `frontend/node_modules` the
gate fails instead of skipping it:

* **Zero** ruff findings, formatting included.
* **Zero** pyright errors under `typeCheckingMode = "strict"`. New code is
  fully annotated; `Any` needs a reason in a comment.
* Branch coverage at or above **70%**, and new behaviour comes with tests that
  fail without it.

CI runs this gate on your pull request on Linux, on **3.11, 3.12 and 3.13**
alike: every leg syncs all extras and runs `./ucx test check --fast` —
everything above except the browser suite, vitest included.

The 3.13 leg used to be a reduced one, running neither Pyright nor the
per-package coverage floors, because `code_intel` depended on
`tree-sitter-languages`, which publishes nothing for 3.13: the extra could not
be installed there, so a type check would have failed on a missing package
rather than on anything you wrote. That extra now depends on
`tree-sitter-language-pack`, which ships cp313 wheels, so the reduction has no
cause left and 3.13 is checked like the others.

It also installs the built wheel into a clean environment, to check the package
works for someone who only ran `pip install uclone-x`. No test that needs the
network or a real model endpoint runs anywhere, so a green run says nothing
about the rendered UI.

Run the gate locally anyway. CI tells you *that* something broke, several
minutes later, on a machine you cannot inspect; the local run tells you what
broke while the change is still in your head.

## What makes a change easy to accept

* **It respects the core principles.** [P0–P9](docs/principles/core-principles.md)
  are normative, not aspirational. A change that contradicts one needs to argue
  with the principle in an issue first, not work around it in code.
* **It fails honestly.** A degraded path reports that it is degraded. Silent
  fallbacks that answer in the name of a component that never ran are the
  single most common rejection reason.
* **It is typed at the boundary.** Protocols and immutable Pydantic models
  across subsystem boundaries; concrete classes stay inside their subsystem.
* **Its tests would fail without it.** A test that passes against the
  unmodified code is not evidence.
* **It says why in the commit message.** Describe the failure the change fixes
  and how you verified the fix, not just what you edited.

## Reporting bugs

Run `ucx report --open`: it fills the bug-report form with your versions and the
failures it recorded, and you review the text before submitting. By hand works too
— open an issue with the `ucx` version (`ucx version`), your Python version, the
provider and model, and the smallest reproduction you can manage. Include what
you expected and what happened.

## Reporting security issues

Do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
