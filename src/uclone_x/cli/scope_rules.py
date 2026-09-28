"""The declarative rules a diff-scoped selection follows: `tests/scope-rules.toml`.

The import graph says which tests import a changed module. It cannot say that a change to
`pyproject.toml` reaches every test, that a stylesheet reaches the browser suite, or that a
new file of any kind can fail the export classification. Those facts are rules, and they
live in one file so that they can be read, reviewed and changed without reading code.

Loading is strict. An unknown table or key, a wrong type or a missing section raises
`ScopeRulesError`: a misspelt `prefixs` that was quietly ignored would stop a rule from
applying, and every selection after it would be narrower than its author meant (P6).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

# Relative to the repository root. A string, not a `Path`: resolved against a `repo` argument
# every time, never against the process cwd.
RULES_PATH: Final[str] = "tests/scope-rules.toml"

_PATH_RULE_KEYS: Final[frozenset[str]] = frozenset({"files", "names", "prefixes", "suffixes"})
_SUPPORTED_VERSION: Final[int] = 1


class ScopeRulesError(ValueError):
    """The rules file is missing, unparseable, or not in the shape this loader reads."""


@dataclass(frozen=True)
class PathRule:
    """A set of repository-relative paths, named four ways."""

    files: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()

    def matches(self, path: str) -> bool:
        """Whether `path` is one this rule names."""
        return (
            path in self.files
            or path.rsplit("/", 1)[-1] in self.names
            or path.startswith(self.prefixes)
            or path.endswith(self.suffixes)
        )

    def entries(self) -> tuple[str, ...]:
        """Every file and prefix the rule names: what must exist in the tree for it to apply."""
        return (*self.files, *self.prefixes)


@dataclass(frozen=True)
class ScopeRules:
    """`tests/scope-rules.toml`, parsed. The file's comments say what each rule is for."""

    whole_suite: PathRule
    browser_suite: PathRule
    frontend: PathRule
    fitness: PathRule
    fitness_tests: str
    always: tuple[str, ...]
    browser_tests: str
    max_age_hours: int
    max_merges: int

    def is_browser_test(self, path: str) -> bool:
        """Whether `path` is a test file of the browser suite."""
        name = path.rsplit("/", 1)[-1]
        return path.startswith(self.browser_tests) and name.startswith("test_")

    def reaches_browser_suite(self, path: str) -> bool:
        """Whether a change to `path` calls for the whole browser suite.

        A browser test file does not: it runs itself. Everything a whole-suite rule names
        does, because every test — browser tests included — depends on it.
        """
        if self.is_browser_test(path):
            return False
        return self.whole_suite.matches(path) or self.browser_suite.matches(path)


def _table(document: dict[str, object], key: str, allowed: frozenset[str]) -> dict[str, object]:
    raw = document.get(key)
    if not isinstance(raw, dict):
        raise ScopeRulesError(f"[{key}] is missing or is not a table")
    table = cast(dict[str, object], raw)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ScopeRulesError(f"[{key}] has unknown key(s): {', '.join(unknown)}")
    return table


def _strings(table: dict[str, object], section: str, key: str) -> tuple[str, ...]:
    raw = table.get(key, [])
    if not isinstance(raw, list):
        raise ScopeRulesError(f"[{section}].{key} must be a list of strings")
    items = cast(list[object], raw)
    if not all(isinstance(item, str) and item for item in items):
        raise ScopeRulesError(f"[{section}].{key} must be a list of non-empty strings")
    return tuple(cast(list[str], items))


def _string(table: dict[str, object], section: str, key: str) -> str:
    raw = table.get(key)
    if not isinstance(raw, str) or not raw:
        raise ScopeRulesError(f"[{section}].{key} must be a non-empty string")
    return raw


def _positive_int(table: dict[str, object], section: str, key: str) -> int:
    raw = table.get(key)
    # `bool` is an `int` subclass; `true` is not a count.
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ScopeRulesError(f"[{section}].{key} must be a positive integer")
    return raw


def _path_rule(
    document: dict[str, object], section: str, extra: frozenset[str] = frozenset()
) -> tuple[PathRule, dict[str, object]]:
    table = _table(document, section, _PATH_RULE_KEYS | extra)
    rule = PathRule(
        files=_strings(table, section, "files"),
        names=_strings(table, section, "names"),
        prefixes=_strings(table, section, "prefixes"),
        suffixes=_strings(table, section, "suffixes"),
    )
    if not (rule.files or rule.names or rule.prefixes or rule.suffixes):
        raise ScopeRulesError(f"[{section}] names no path")
    return rule, table


def parse_scope_rules(text: str) -> ScopeRules:
    """Parse the rules file's text. Raises `ScopeRulesError` on anything it does not expect."""
    try:
        document = cast(dict[str, object], tomllib.loads(text))
    except tomllib.TOMLDecodeError as exc:
        raise ScopeRulesError(f"not valid TOML: {exc}") from exc
    sections = {
        "version",
        "whole_suite",
        "browser_suite",
        "frontend",
        "fitness",
        "always",
        "browser_tests",
        "browser_freshness",
    }
    unknown = sorted(set(document) - sections)
    if unknown:
        raise ScopeRulesError(f"unknown top-level key(s): {', '.join(unknown)}")
    if document.get("version") != _SUPPORTED_VERSION:
        raise ScopeRulesError(
            f"version must be {_SUPPORTED_VERSION}, found {document.get('version')!r}"
        )

    whole_suite, _ = _path_rule(document, "whole_suite")
    browser_suite, _ = _path_rule(document, "browser_suite")
    frontend, _ = _path_rule(document, "frontend")
    fitness, fitness_table = _path_rule(document, "fitness", frozenset({"tests"}))
    always = _table(document, "always", frozenset({"tests"}))
    browser_tests = _table(document, "browser_tests", frozenset({"dir"}))
    freshness = _table(document, "browser_freshness", frozenset({"max_age_hours", "max_merges"}))
    return ScopeRules(
        whole_suite=whole_suite,
        browser_suite=browser_suite,
        frontend=frontend,
        fitness=fitness,
        fitness_tests=_string(fitness_table, "fitness", "tests"),
        always=_strings(always, "always", "tests"),
        browser_tests=_string(browser_tests, "browser_tests", "dir"),
        max_age_hours=_positive_int(freshness, "browser_freshness", "max_age_hours"),
        max_merges=_positive_int(freshness, "browser_freshness", "max_merges"),
    )


def load_scope_rules(repo: Path) -> ScopeRules:
    """Read and parse `repo`'s rules file. Raises `ScopeRulesError` when it cannot."""
    path = repo / RULES_PATH
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScopeRulesError(f"cannot read {path}: {exc}") from exc
    try:
        return parse_scope_rules(text)
    except ScopeRulesError as exc:
        raise ScopeRulesError(f"{path}: {exc}") from exc


def missing_entries(rules: ScopeRules, repo: Path) -> list[str]:
    """Every path the rules name that is absent from `repo`: a rule that can never match.

    A renamed directory leaves its rule behind, and a rule for a path that no longer exists
    fails open — the change it was written to catch now lands under the new name, unmatched.
    """
    named = [
        *rules.whole_suite.entries(),
        *rules.browser_suite.entries(),
        *rules.frontend.entries(),
        *rules.fitness.entries(),
        rules.fitness_tests,
        *rules.always,
        rules.browser_tests,
    ]
    return sorted({entry for entry in named if not (repo / entry).exists()})
