"""The Core's notice codes and the head's `notices` catalog name the same set (multilingual-ui.md §4).

A note is stored with a `code` and worded by the head. A code the Core writes and the head's
catalog lacks would show the English fallback on a Korean screen, which is the mixed screen
the catalog exists to prevent; so the gap fails here, not on screen.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest

from uclone_x.room.notices import NOTICE_FALLBACK, NoticeCode, NoticeParams, notice_content

LOCALES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "locales"
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _codes(language: str) -> dict[str, Any]:
    catalog = json.loads((LOCALES / language / "notices.json").read_text(encoding="utf-8"))
    codes: dict[str, Any] = catalog["codes"]
    assert isinstance(codes, dict)
    return codes


def test_every_notice_code_is_in_the_english_catalog_and_nothing_else_is() -> None:
    """Equal, not just contained: a catalog key no code names is a notice nobody can send.

    Killed by: src/uclone_x/room/notices.py :: "loop.nothing_to_stop",
    Becomes: "loop.nothing_to_stop", "loop.renamed",
    """
    assert set(get_args(NoticeCode)) == set(_codes("en"))


@pytest.mark.parametrize("language", ["ko"])
def test_every_other_language_words_the_same_codes(language: str) -> None:
    """`ko: Messages` refuses a missing key; this also refuses an extra one, on the Core's side."""
    assert set(_codes(language)) == set(_codes("en"))


def test_the_stored_english_fallback_is_the_english_catalog_sentence() -> None:
    """One English wording: what an export shows is what an English screen shows.

    Killed by: src/uclone_x/room/notices.py :: "loop.stopped": "🛑 The repeating task was stopped.",
    Becomes: "loop.stopped": "🛑 Repeating task stopped.",
    """
    assert dict(NOTICE_FALLBACK) == _codes("en")


def test_the_fallback_fills_every_placeholder_its_sentence_uses() -> None:
    """Each code's fallback, filled with the params the Core sends, leaves no `{marker}`."""
    params: dict[str, NoticeParams] = {
        "loop.active": {"job_id": "j", "interval_seconds": 300.0, "prompt": "p"},
        "loop.registered": {"job_id": "j", "interval_seconds": 30.0, "prompt": "p"},
        "loop.interval_too_short": {"interval_seconds": 1.0},
    }
    for code in get_args(NoticeCode):
        text = notice_content(code, params.get(code, {}))
        assert not PLACEHOLDER.search(text), (code, text)
