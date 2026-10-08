"""Core's refusals in the language a terminal head writes in (#1862).

A turn that refuses a step carries a code (`StepRefusalCode`) beside Core's English
sentence. The CLI and the ACP server say the refusal from this module instead of printing
the sentence as Core wrote it, so a person whose language is Korean reads it in Korean.

English lives once, in Core (`STEP_REFUSAL_TEXT`). Every other language is a JSON catalog,
`locales/<language>/refusals.json`, keyed by code, in the `{name}` form the head's catalogs
use. A code a catalog lacks fails `tests/unit/test_refusal_i18n.py`, not a person's screen.

The language is the saved `ui_language` choice, `"system"` resolving from the locale
environment, by the same rule the desktop head uses (`resolve_language`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from importlib import resources
from typing import cast

from uclone_x.core.tool_results import STEP_REFUSAL_TEXT, StepRefusalCode
from uclone_x.i18n.language import (
    DEFAULT_UI_LANGUAGE,
    Language,
    environment_language_hints,
    is_ui_language,
    resolve_language,
)

__all__ = ["head_language", "refusal_catalog", "refusal_text"]


def head_language(
    settings: Mapping[str, object], environ: Mapping[str, str] | None = None
) -> Language:
    """The language a terminal head writes in: the saved choice, or the locale's for `"system"`.

    `settings` is the settings file's contents; a missing or unknown `ui_language` counts as
    the default, `"system"`, as it does for the desktop head.
    """
    choice = settings.get("ui_language")
    return resolve_language(
        choice if is_ui_language(choice) else DEFAULT_UI_LANGUAGE,
        environment_language_hints(environ),
    )


@cache
def refusal_catalog(language: Language) -> Mapping[StepRefusalCode, str]:
    """Each step refusal's sentence in `language`.

    English is Core's own sentences. Any other language is read from its JSON catalog.
    """
    if language == "en":
        return STEP_REFUSAL_TEXT
    source = resources.files("uclone_x.i18n").joinpath("locales").joinpath(language)
    source = source.joinpath("refusals.json")
    return cast("dict[StepRefusalCode, str]", json.loads(source.read_text(encoding="utf-8")))


def refusal_text(code: StepRefusalCode, language: Language) -> str:
    """The refusal `code` names, as a person reading `language` is told it."""
    return refusal_catalog(language)[code]
