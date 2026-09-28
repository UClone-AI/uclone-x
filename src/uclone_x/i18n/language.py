"""Which language UClone-X writes in: the saved choice, and how `"system"` resolves.

The choice is the Core's, not a head's, because more than one reader needs it: the desktop
head, the CLI, and the notices the Core writes into a conversation. Stored per browser, the
CLI would be Korean while the dashboard is English on the same machine.

`"system"` is kept as the choice rather than resolved once and stored, so a user who changes
their OS language is not left with a stale first-run guess. The head resolves it from
`navigator.languages` with the same rule as `resolve_language` (`frontend/src/i18n/language.ts`);
the CLI resolves it from the locale environment.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Final, Literal, TypeGuard, get_args

#: A language UClone-X can write in.
Language = Literal["en", "ko"]
#: What the user can choose: a language, or whatever the OS or browser prefers.
UiLanguage = Literal["system", "en", "ko"]

UI_LANGUAGES: Final[tuple[UiLanguage, ...]] = get_args(UiLanguage)
DEFAULT_UI_LANGUAGE: Final[UiLanguage] = "system"
#: What `"system"` falls back to when no hint names a language this build can write.
FALLBACK_LANGUAGE: Final[Language] = "en"

#: The locale variables a POSIX program consults for message language, in precedence order.
LOCALE_ENV_VARS: Final[tuple[str, ...]] = ("LC_ALL", "LC_MESSAGES", "LANG")


def is_ui_language(value: object) -> TypeGuard[UiLanguage]:
    """Whether `value` is a choice the settings file may hold."""
    return isinstance(value, str) and value in UI_LANGUAGES


def _language_of(hint: str) -> Language | None:
    """The language a locale tag such as `ko-KR`, `ko_KR.UTF-8` or `en` names, if supported."""
    primary = hint.strip().lower().replace("_", "-").split(".")[0].split("-")[0]
    if primary == "ko":
        return "ko"
    if primary == "en":
        return "en"
    return None


def resolve_language(choice: UiLanguage, hints: Iterable[str]) -> Language:
    """The language to write in: the choice itself, or the first supported hint for `"system"`.

    Hints are in preference order, like `navigator.languages`. A hint naming a language this
    build cannot write is skipped, so `["ja", "ko"]` resolves to Korean rather than English.
    """
    if choice != "system":
        return choice
    for hint in hints:
        found = _language_of(hint)
        if found is not None:
            return found
    return FALLBACK_LANGUAGE


def environment_language_hints(environ: Mapping[str, str] | None = None) -> list[str]:
    """The locale hints a CLI process has, in precedence order, `C`/`POSIX` left out."""
    env = os.environ if environ is None else environ
    hints: list[str] = []
    for name in LOCALE_ENV_VARS:
        value = env.get(name, "").strip()
        if value and value not in ("C", "POSIX"):
            hints.append(value)
    return hints
