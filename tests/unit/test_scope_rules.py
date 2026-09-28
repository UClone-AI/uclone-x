"""Unit tests for the declarative scope rules (uclone_x.cli.scope_rules)."""

import re
from pathlib import Path

import pytest

from uclone_x.cli.scope_rules import (
    RULES_PATH,
    PathRule,
    ScopeRulesError,
    load_scope_rules,
    missing_entries,
    parse_scope_rules,
)

_ROOT = Path(__file__).resolve().parents[2]
_TEXT = (_ROOT / RULES_PATH).read_text(encoding="utf-8")
# The `[always]` list as the file spells it, one entry per line: the only `tests = [`.
_ALWAYS_MATCH = re.search(r"^tests = \[[^\]]*\]", _TEXT, re.MULTILINE)
assert _ALWAYS_MATCH is not None
_ALWAYS_TESTS = _ALWAYS_MATCH.group(0)


def test_the_repository_rules_parse() -> None:
    rules = load_scope_rules(_ROOT)
    assert rules.max_age_hours == 24
    assert rules.max_merges == 20
    assert "tests/unit/test_oss_export.py" in rules.always


def test_a_rule_for_an_absent_path_is_reported(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/cli/scope_rules.py :: if not (repo / entry).exists()
    Becomes: if False
    """
    rules = load_scope_rules(_ROOT)
    assert "tests/support/" in missing_entries(rules, tmp_path)


@pytest.mark.parametrize(
    ("path", "matches"),
    [
        ("ucx", True),  # files
        ("tests/e2e/conftest.py", True),  # names
        ("tests/support/fake.py", True),  # prefixes
        ("README.md", False),
        ("src/ucx", False),
    ],
)
def test_a_path_rule_matches_by_file_name_and_prefix(path: str, matches: bool) -> None:
    rule = PathRule(files=("ucx",), names=("conftest.py",), prefixes=("tests/support/",))
    assert rule.matches(path) is matches


def test_a_path_rule_matches_by_suffix() -> None:
    assert PathRule(suffixes=(".md",)).matches("docs/a.md")
    assert not PathRule(suffixes=(".md",)).matches("docs/a.mdx")


@pytest.mark.parametrize(
    ("path", "reaches"),
    [
        ("src/uclone_x/ui/server.py", True),
        ("frontend/src/App.tsx", True),
        ("tests/e2e/conftest.py", True),
        ("tests/e2e/helpers.py", True),
        ("pyproject.toml", True),  # a whole-suite path reaches the browser suite too
        ("tests/e2e/test_room_chat_e2e.py", False),  # a browser test runs itself only
        ("src/uclone_x/agent/loop.py", False),
        ("docs/guides/x.md", False),
    ],
)
def test_which_changes_call_for_the_whole_browser_suite(path: str, reaches: bool) -> None:
    """Killed by: src/uclone_x/cli/scope_rules.py :: if self.is_browser_test(path):
    Becomes: if False:
    """
    assert load_scope_rules(_ROOT).reaches_browser_suite(path) is reaches


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('prefixes = ["tests/support/"]', 'prefixs = ["tests/support/"]', "unknown key"),
        ("version = 1", "version = 2", "version must be 1"),
        ("max_merges = 20", "max_merges = 0", "positive integer"),
        ("max_merges = 20", "max_merges = true", "positive integer"),
        ("max_age_hours = 24", 'max_age_hours = "24"', "positive integer"),
        ('dir = "tests/e2e/"', 'dir = ""', "non-empty string"),
        (_ALWAYS_TESTS, 'tests = "tests/unit/test_oss_export.py"', "list of strings"),
        ("\n[browser_freshness]\n", "\n[browser_freshnes]\n", "unknown top-level"),
        ("version = 1", "version = 1\nversion = 1", "not valid TOML"),
    ],
)
def test_a_malformed_rules_file_is_refused(old: str, new: str, message: str) -> None:
    """Killed by: src/uclone_x/cli/scope_rules.py :: unknown = sorted(set(table) - allowed)
    Becomes: unknown = []
    """
    assert old in _TEXT
    with pytest.raises(ScopeRulesError, match=message):
        parse_scope_rules(_TEXT.replace(old, new, 1))


def test_a_rule_naming_no_path_is_refused() -> None:
    text = _TEXT.replace('names = ["conftest.py"]\n', "").replace(
        'files = ["pyproject.toml", "uv.lock", "ucx", "tests/scope-rules.toml"]\n', ""
    )
    text = text.replace('prefixes = ["tests/support/"]\n', "")
    with pytest.raises(ScopeRulesError, match=r"\[whole_suite\] names no path"):
        parse_scope_rules(text)


def test_a_missing_rules_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ScopeRulesError, match="cannot read"):
        load_scope_rules(tmp_path)
