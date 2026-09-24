# AGENTS.md — guidance for AI coding assistants

If you are an AI assistant making changes in this repository, read this first.

## 1. The core principles are normative

[docs/principles/core-principles.md](docs/principles/core-principles.md) defines
P0–P9. They constrain every design decision in this codebase. Do not modify,
soften or work around a principle: if your change appears to require that, stop
and raise it in an issue instead.

## 2. Verify with the gate, not with your judgment

```bash
./ucx test check
```

Ruff format and lint, pyright in strict mode, pytest with branch coverage
≥ 70%, and the frontend vitest suite. All five pass, or the change is not done.
The vitest suite needs `npm ci --prefix frontend` once; without it the gate
fails rather than skipping the suite.

CI repeats this gate on a pull request — without the browser suite — across
three Python versions and a from-scratch install of the built package. Do not use it as your test runner:
a pull request opened to find out whether the gate passes spends five minutes
of someone else's wall clock to answer a question your own machine answers in
ninety seconds.

## 3. Verify effects, never exit status

A command's effect is confirmed by observing the effect. An exit code of `0`
from a pipeline is the last stage's status, and a retry loop can exit `0`
having done nothing. Read the output and check the thing you claim changed.

## 4. Failure is reported, never simulated

If a component cannot do its job, it says so. Never substitute a stub, a mock
or a cached value for a component that failed and then report success — a
fallback that answers in the name of something that never ran is worse than an
error, because it cannot be debugged. This is P6, and it is the most common
reason a change is rejected here.

## 5. Types at the boundary, tests that would fail

* Subsystem boundaries are Protocols and immutable Pydantic v2 models.
  Concrete classes stay inside their subsystem.
* Full annotations; `Any` carries a comment saying why.
* A regression test is evidence only if you have watched it fail without the
  fix. Break the fix, confirm that specific test fails, restore it.

## 6. Scope

Change what the task requires. Do not reformat untouched code, rename
unrelated symbols, or upgrade dependencies as a side effect. New dependencies
are declared in `pyproject.toml` in the same change that imports them.

## 7. Say what you did

Commit messages describe the failure being fixed and how the fix was verified.
"Updated files" is not a commit message.
