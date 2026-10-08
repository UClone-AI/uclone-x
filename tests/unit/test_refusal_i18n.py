"""Core's step refusals, said in the person's language by the terminal heads (#1862)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from importlib import resources
from typing import get_args

import pytest

from uclone_x.agent.loop import LoopTickResult
from uclone_x.agent.models import TurnResult
from uclone_x.cli.commands import run
from uclone_x.core.provenance import Provenance
from uclone_x.core.tool_results import STEP_REFUSAL_TEXT, StepRefusalCode
from uclone_x.i18n.language import LOCALE_ENV_VARS, Language
from uclone_x.i18n.refusals import head_language, refusal_catalog, refusal_text
from uclone_x.llm.connectors.saved_choice import settings_file

CODES: tuple[StepRefusalCode, ...] = get_args(StepRefusalCode)
LANGUAGES: tuple[Language, ...] = get_args(Language)
_HANGUL = re.compile(r"[가-힣]")
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _korean_catalog_text() -> str:
    source = resources.files("uclone_x.i18n").joinpath("locales").joinpath("ko")
    return source.joinpath("refusals.json").read_text(encoding="utf-8")


def _refused(code: StepRefusalCode) -> TurnResult:
    return TurnResult(
        turn_index=1,
        content="",
        error=STEP_REFUSAL_TEXT[code],
        error_code=code,
        stop_reason="step_results_over_window",
        provenance=Provenance.primary("fake"),
    )


@pytest.fixture
def korean_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LOCALE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANG", "ko_KR.UTF-8")


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_language_says_every_step_refusal_with_the_same_placeholders(
    language: Language,
) -> None:
    """A code a catalog lacks, or a sentence that drops a value English uses, fails here."""
    catalog = refusal_catalog(language)

    assert set(catalog) == set(CODES)
    for code in CODES:
        assert catalog[code].strip(), code
        assert set(_PLACEHOLDER.findall(catalog[code])) == set(
            _PLACEHOLDER.findall(STEP_REFUSAL_TEXT[code])
        ), code


@pytest.mark.parametrize("code", CODES)
def test_a_korean_locale_is_told_the_refusal_in_korean_without_internals(
    code: StepRefusalCode, korean_locale: None
) -> None:
    """With no saved choice, `"system"` follows the locale, and Korean is Korean text only:
    no code, identifier, handle, path or English sentence, and polite 합니다체 throughout.

    Killed by: src/uclone_x/i18n/refusals.py :: return refusal_catalog(language)[code]
    Becomes: return refusal_catalog("en")[code]
    """
    text = run.turn_error_text(_refused(code))

    assert text is not None
    assert text == json.loads(_korean_catalog_text())[code]
    assert _HANGUL.search(text), text
    assert text != STEP_REFUSAL_TEXT[code]
    assert not re.search(r"[A-Za-z]+[_.][A-Za-z]|tr_|[/\\{}]|Error|Exception", text), text
    for sentence in filter(None, (s.strip() for s in text.split("."))):
        assert sentence.endswith(("니다", "십시오")), sentence


def test_a_saved_english_choice_outranks_a_korean_locale(korean_locale: None) -> None:
    settings = settings_file()
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"ui_language": "en"}), encoding="utf-8")

    assert run.turn_error_text(_refused("step.no_room")) == STEP_REFUSAL_TEXT["step.no_room"]


def test_an_unknown_saved_choice_counts_as_system() -> None:
    assert head_language({"ui_language": "fr"}, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert head_language({}, {}) == "en"


def test_an_uncoded_error_is_core_sentence_unchanged(korean_locale: None) -> None:
    result = TurnResult(
        turn_index=1,
        content="",
        error="Plain sentence.",
        stop_reason="provider_outage",
        provenance=Provenance.primary("fake"),
    )

    assert run.turn_error_text(result) == "Plain sentence."


def test_a_loop_tick_says_its_refused_step_in_the_locale_language(korean_locale: None) -> None:
    now = datetime.now(UTC)
    tick = LoopTickResult(
        tick_index=1,
        started_at=now,
        finished_at=now,
        duration_seconds=0.0,
        success=False,
        error=STEP_REFUSAL_TEXT["step.no_room"],
        turn=_refused("step.no_room"),
    )

    assert run.tick_error_text(tick) == refusal_text("step.no_room", "ko")
