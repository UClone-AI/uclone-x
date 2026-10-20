"""Core's tool-call refusals, as plain sentences in each language a head writes in (#2196).

A refusal of a tool call reaches the person's screen through the turn's tool record, and
the model reads it as the call's result. It says what happened and what to do, and never
the agent's id, its permission list or an exception's text (#2196 item 1).

English lives here, once (`TOOL_REFUSAL_TEXT`). Every other language is a JSON catalog,
`locales/<language>/tool_refusals.json`, keyed by code, as the step refusals' catalogs are
(`uclone_x.i18n.refusals`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from importlib import resources
from typing import Final, Literal, cast

from uclone_x.i18n.language import Language

__all__ = ["TOOL_REFUSAL_TEXT", "ToolRefusalCode", "tool_refusal_catalog", "tool_refusal_text"]

ToolRefusalCode = Literal["tool.out_of_range"]
"""Why a tool call was refused, as a key each language's catalog has."""

TOOL_REFUSAL_TEXT: Final[Mapping[ToolRefusalCode, str]] = {
    "tool.out_of_range": (
        "This assistant is not set up to use that tool, so it was not run. "
        "Use only the tools listed in this request."
    ),
}


@cache
def tool_refusal_catalog(language: Language) -> Mapping[ToolRefusalCode, str]:
    """Each tool refusal's sentence in `language`: Core's own for English, else its catalog."""
    if language == "en":
        return TOOL_REFUSAL_TEXT
    source = resources.files("uclone_x.i18n").joinpath("locales").joinpath(language)
    source = source.joinpath("tool_refusals.json")
    return cast("dict[ToolRefusalCode, str]", json.loads(source.read_text(encoding="utf-8")))


def tool_refusal_text(code: ToolRefusalCode, language: Language) -> str:
    """The refusal `code` names, in `language`."""
    return tool_refusal_catalog(language)[code]
