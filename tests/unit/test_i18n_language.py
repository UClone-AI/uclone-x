"""The language choice and how `"system"` resolves."""

from __future__ import annotations

import pytest

from uclone_x.i18n import environment_language_hints, is_ui_language, resolve_language


def test_an_explicit_choice_ignores_every_hint() -> None:
    assert resolve_language("en", ["ko-KR"]) == "en"
    assert resolve_language("ko", ["en-US"]) == "ko"


@pytest.mark.parametrize(
    ("hints", "expected"),
    [
        (["ko-KR", "en-US"], "ko"),
        (["ko_KR.UTF-8"], "ko"),
        (["KO"], "ko"),
        (["en-GB", "ko"], "en"),
        # A language this build cannot write is skipped, not treated as English.
        (["ja-JP", "ko-KR"], "ko"),
        (["ja-JP"], "en"),
        ([], "en"),
    ],
)
def test_system_follows_the_first_supported_hint(hints: list[str], expected: str) -> None:
    # Killed by: src/uclone_x/i18n/language.py ::         if found is not None:
    # Becomes:         if True:
    assert resolve_language("system", hints) == expected


def test_the_cli_reads_the_locale_environment_in_posix_precedence() -> None:
    env = {"LANG": "en_US.UTF-8", "LC_ALL": "ko_KR.UTF-8", "LC_MESSAGES": "C"}

    assert environment_language_hints(env) == ["ko_KR.UTF-8", "en_US.UTF-8"]
    assert resolve_language("system", environment_language_hints(env)) == "ko"


@pytest.mark.parametrize("value", ["system", "en", "ko"])
def test_the_three_choices_are_accepted(value: str) -> None:
    assert is_ui_language(value)


@pytest.mark.parametrize("value", ["fr", "", "KO", None, 1])
def test_anything_else_is_refused(value: object) -> None:
    assert not is_ui_language(value)
