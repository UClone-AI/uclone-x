"""Tests for technical terminology preservation and hardware search grounding (#829).

Enforces P0 (Non-Expert Operability) and P6 (Fail-Fast & Zero Silent Fallbacks):
- Standard technical acronyms (TRS, TRRS, TS, ADC, DAC) must never be translated into invented phonetic terms.
- System prompt and search tools must guide agents toward technical specifications and pinouts.
"""

from __future__ import annotations

import pytest

from uclone_x.agent.models import DEFAULT_SYSTEM_PROMPT
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.terminology import (
    FORBIDDEN_PHONETIC_TRANSLATIONS,
    STANDARD_TECHNICAL_ACRONYMS,
    TECHNICAL_GROUNDING_INSTRUCTION,
    evaluate_hardware_search_grounding,
    explain_technical_acronym,
    is_standard_audio_connector,
    validate_terminology_preservation,
)
from uclone_x.tools.builtin.web import WebSearchParams, WebSearchTool


def test_the_clone_persona_opts_in_to_the_terminology_clause_verbatim() -> None:
    """The clause is opt-in (#1424): not in the default, stated in the persona that wants it.

    Clone states it in its own `system_prompt`. It is held equal to the constant so the
    two copies cannot drift apart -- a reworded constant fails here until the persona is
    reworded with it.

    Killed by: src/uclone_x/agent/terminology.py :: Never invent translated terms for standard technical acronyms
    Becomes: Always invent translated terms for standard technical acronyms
    """
    clone = PersonaRegistry().get_persona("clone")
    assert clone is not None
    assert f"- {TECHNICAL_GROUNDING_INSTRUCTION}\n" in clone.system_prompt
    assert "Never invent translated terms for standard technical acronyms" in clone.system_prompt
    assert "prioritize technical specifications, pinouts" in clone.system_prompt
    assert "Never invent translated terms" not in DEFAULT_SYSTEM_PROMPT


def test_standard_technical_acronym_definitions() -> None:
    """Canonical audio and hardware acronym expansions must match industry standards.

    Killed by: src/uclone_x/agent/terminology.py :: "TRS": "Tip-Ring-Sleeve"
    Becomes: "TRS": "Tri-Ring-Sleeve"
    """
    assert STANDARD_TECHNICAL_ACRONYMS["TRS"] == "Tip-Ring-Sleeve"
    assert STANDARD_TECHNICAL_ACRONYMS["TRRS"] == "Tip-Ring-Ring-Sleeve"
    assert STANDARD_TECHNICAL_ACRONYMS["TS"] == "Tip-Sleeve"
    assert STANDARD_TECHNICAL_ACRONYMS["ADC"] == "Analog-to-Digital Converter"
    assert STANDARD_TECHNICAL_ACRONYMS["DAC"] == "Digital-to-Analog Converter"

    assert explain_technical_acronym("TRS") == "Tip-Ring-Sleeve"
    assert explain_technical_acronym("trrs") == "Tip-Ring-Ring-Sleeve"
    assert explain_technical_acronym("UNKNOWN") is None


def test_audio_connector_recognition() -> None:
    """Audio connector acronyms must be recognized correctly."""
    assert is_standard_audio_connector("TRS") is True
    assert is_standard_audio_connector("trrs") is True
    assert is_standard_audio_connector("ts") is True
    assert is_standard_audio_connector("USB-C") is False
    assert is_standard_audio_connector("Bluetooth") is False


@pytest.mark.parametrize(
    ("bad_text", "expected_acronym"),
    [
        ("마이크는 TRS (트리플리트) 케이블을 사용합니다.", "TRS"),
        ("오디오 어댑터는 TS (트리플) 규격입니다.", "TS"),
        ("TRS (트리플 리트) 및 TS (트리플) 단자", "TRS"),
    ],
)
def test_forbidden_phonetic_translation_detection(bad_text: str, expected_acronym: str) -> None:
    """Invented phonetic translations (e.g. 트리플리트) must be detected as violations."""
    assert expected_acronym in FORBIDDEN_PHONETIC_TRANSLATIONS
    violations = validate_terminology_preservation(bad_text)
    assert len(violations) >= 1
    assert any(expected_acronym in v for v in violations)


def test_valid_technical_terminology_passes_validation() -> None:
    """Valid technical explanations must produce zero violations."""
    clean_text = (
        "This microphone requires a 3.5mm TRS (Tip-Ring-Sleeve) to TRRS (Tip-Ring-Ring-Sleeve) adapter "
        "with an integrated ADC and preamplifier."
    )
    violations = validate_terminology_preservation(clean_text)
    assert violations == []


def test_hardware_search_grounding_evaluator() -> None:
    """Search queries with commercial shopping terms must include technical specs."""
    grounded, _ = evaluate_hardware_search_grounding("best cheap microphone adapter amazon")
    assert grounded is False

    grounded, _ = evaluate_hardware_search_grounding(
        "TRS to TRRS microphone adapter pinout and ADC preamp requirements"
    )
    assert grounded is True

    grounded, _ = evaluate_hardware_search_grounding(
        "buy USB-C audio interface with phantom power and ADC specifications"
    )
    assert grounded is True


def test_web_search_tool_description_mentions_technical_specifications() -> None:
    """Web search tool description and parameters must guide agents toward technical specs."""
    tool = WebSearchTool()
    assert "technical specifications" in tool.description.lower()

    fields = WebSearchParams.model_fields
    assert "pinouts" in str(fields["query"].description).lower()
