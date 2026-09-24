"""Canonical technical terminology, acronym definitions, and hardware grounding rules (#829).

This module enforces P0 (Non-Expert Operability) and P6 (Explicit State & Zero Silent Fallbacks):
1. Standard technical acronyms (e.g. TRS, TRRS, ADC, DAC) must never be translated into invented
   phonetic terms (e.g. translating TRS to '트리플리트').
2. Hardware and audio connector inquiries must ground search queries and evaluations in technical
   specifications (pinouts, ADC/preamp requirements) rather than generic shopping link titles.

`TECHNICAL_GROUNDING_INSTRUCTION` is opt-in (#1424). It is not part of the default prompt
every persona is given, because most personas never touch hardware and the default is
paid for on every request. A persona that wants it states it in its own `system_prompt`,
as `personas/clone.yaml` does: no extra persona key, nothing for the editor to drop on a
save, and the text sits in the persona file where its author can read it.
"""

from __future__ import annotations

import re
from typing import Final

STANDARD_TECHNICAL_ACRONYMS: Final[dict[str, str]] = {
    "TRS": "Tip-Ring-Sleeve",
    "TRRS": "Tip-Ring-Ring-Sleeve",
    "TS": "Tip-Sleeve",
    "ADC": "Analog-to-Digital Converter",
    "DAC": "Digital-to-Analog Converter",
    "USB-C": "Universal Serial Bus Type-C",
    "PCIe": "Peripheral Component Interconnect Express",
}

FORBIDDEN_PHONETIC_TRANSLATIONS: Final[dict[str, tuple[str, ...]]] = {
    "TRS": ("트리플리트", "트리플 리트", "트리플-리트"),
    "TS": ("트리플",),
}

TECHNICAL_GROUNDING_INSTRUCTION: Final[str] = (
    "Never invent translated terms for standard technical acronyms (e.g. TRS, TRRS, TS, ADC, DAC, USB-C, PCIe). "
    "Retain the standard technical term and explain its true English expansion when necessary. "
    "When evaluating hardware or technical queries, prioritize technical specifications, pinouts, "
    "standards, and electrical compatibility (such as whether an ADC or preamp is required) over generic shopping listings."
)


def validate_terminology_preservation(text: str) -> list[str]:
    """Inspect text for forbidden phonetic translations of standard technical acronyms."""
    violations: list[str] = []
    for acronym, forbidden_patterns in FORBIDDEN_PHONETIC_TRANSLATIONS.items():
        for pattern in forbidden_patterns:
            if pattern in text:
                violations.append(
                    f"Forbidden phonetic translation for acronym '{acronym}': '{pattern}'"
                )
    return violations


def explain_technical_acronym(acronym: str) -> str | None:
    """Return the canonical English expansion of a technical acronym, or None if unknown."""
    return STANDARD_TECHNICAL_ACRONYMS.get(acronym.upper())


def is_standard_audio_connector(name: str) -> bool:
    """Return True if the term is a canonical audio connector acronym (TS, TRS, TRRS)."""
    return name.upper() in {"TS", "TRS", "TRRS"}


def evaluate_hardware_search_grounding(query: str) -> tuple[bool, str]:
    """Check if a hardware search query includes technical specifications or pinouts."""
    q_lower = query.lower()
    shopping_keywords = {"buy", "cheap", "best price", "amazon", "ebay", "shopping"}
    technical_keywords = {
        "pinout",
        "adc",
        "preamp",
        "pre-amp",
        "trs",
        "trrs",
        "ts",
        "specification",
        "impedance",
        "phantom power",
        "wiring",
        "schematic",
        "dac",
    }

    has_shopping = any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in shopping_keywords)
    has_technical = any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in technical_keywords)

    if has_shopping and not has_technical:
        return (
            False,
            "Query contains commercial shopping keywords without technical specifications or pinouts.",
        )
    return (True, "Query contains technical grounding criteria or does not skew purely commercial.")


__all__ = [
    "FORBIDDEN_PHONETIC_TRANSLATIONS",
    "STANDARD_TECHNICAL_ACRONYMS",
    "TECHNICAL_GROUNDING_INSTRUCTION",
    "evaluate_hardware_search_grounding",
    "explain_technical_acronym",
    "is_standard_audio_connector",
    "validate_terminology_preservation",
]
