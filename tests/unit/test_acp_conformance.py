"""The ACP conformance registry, and its agreement with the document that governs it.

`docs/acp-protocol-spec.md` §3.3 names the failure these tests exist to prevent:

> A hardcoded capability block is the mechanism by which a conformance claim becomes false
> without anyone editing a document.

So the tests below read that document and compare it against
`uclone_x.acp.conformance`. A method added to one and not the other fails here, which is the
only thing that keeps `initialize`'s future capability response honest.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from uclone_x.acp import (
    ACP_SDK_VERSION,
    ACP_TRANSPORT,
    AcpMethod,
    AcpMethodStatus,
    AcpSide,
    acp_mcp_descriptor_registry,
    acp_method_registry,
    conformance_summary,
    implemented,
    shell_presence,
)

SPEC_PATH = Path(__file__).resolve().parents[2] / "docs" / "acp-protocol-spec.md"

_BACKTICKED = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


def _table_after(heading: str) -> list[str]:
    """Return the rows of the first markdown table following `heading`."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    start = text.index(heading)
    rows: list[str] = []
    seen_header = False
    for line in text[start:].splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            seen_header = True
            rows.append(stripped)
        elif seen_header and stripped == "":
            break
    # Drop the header row and the alignment row.
    return rows[2:]


def _documented_method_names(heading: str) -> set[str]:
    """Every method named in the first column of the table under `heading`."""
    names: set[str] = set()
    for row in _table_after(heading):
        first_cell = row.split("|")[1]
        names.update(_BACKTICKED.findall(first_cell))
    return names


def _registry_names(side: AcpSide) -> set[str]:
    return {m.name for m in acp_method_registry() if m.side is side}


def test_the_specification_file_is_where_the_tests_think_it_is() -> None:
    """A parse that silently matched nothing would make every comparison below vacuous."""
    assert SPEC_PATH.is_file(), SPEC_PATH
    assert _table_after("## 2. Agent-side conformance table")
    assert _table_after("## 4. Client-side surface")


def test_agent_side_registry_and_document_name_the_same_methods() -> None:
    assert _registry_names(AcpSide.AGENT) == _documented_method_names(
        "## 2. Agent-side conformance table"
    )


def test_client_side_registry_and_document_name_the_same_methods() -> None:
    assert _registry_names(AcpSide.CLIENT) == _documented_method_names("## 4. Client-side surface")


def test_the_pinned_sdk_version_matches_the_document() -> None:
    """§1's version line is load-bearing and nothing else compares it yet.

    The acceptance criterion in §7 is a `pyproject.toml` pin checked against this string, and
    it cannot be written before ACP is a dependency (#649). This is the half that can be
    asserted today: document against constant.
    """
    text = SPEC_PATH.read_text(encoding="utf-8")
    documented = re.search(r"agent-client-protocol==([0-9]+\.[0-9]+\.[0-9]+)", text)
    assert documented is not None, "the specification no longer names an SDK version"
    assert documented.group(1) == ACP_SDK_VERSION


def test_no_method_claims_to_be_implemented_while_the_document_says_no() -> None:
    """The registry may not run ahead of the document.

    Every row of §2 currently reads `**No**` in its Implemented column. When one becomes
    Yes it must do so in the same change as the code and its test, which is what this
    assertion forces.
    """
    documented_yes: set[str] = set()
    for row in _table_after("## 2. Agent-side conformance table"):
        cells = [c.strip() for c in row.split("|")]
        # | name | counterpart | Specified | Implemented | Note |
        if len(cells) < 6:
            continue
        if cells[4].replace("*", "").lower() == "yes":
            documented_yes.update(_BACKTICKED.findall(cells[1]))

    registry_yes = {m.name for m in implemented() if m.side is AcpSide.AGENT}
    assert registry_yes == documented_yes


def test_a_refusal_must_carry_a_reason() -> None:
    """§0 requires a reason for a refusal, so the model refuses to express one without."""
    with pytest.raises(ValueError, match="no reason given"):
        AcpMethod(
            name="authenticate",
            side=AcpSide.AGENT,
            status=AcpMethodStatus.OUT_OF_SCOPE,
            note="   ",
        )


def test_an_unbuilt_method_needs_no_reason() -> None:
    """Scheduled work is not a refusal; requiring prose for it would only produce filler."""
    method = AcpMethod(name="prompt", side=AcpSide.AGENT)
    assert method.status is AcpMethodStatus.NOT_IMPLEMENTED
    assert method.note == ""


def test_authenticate_is_out_of_scope_and_says_why() -> None:
    method = next(m for m in acp_method_registry() if m.name == "authenticate")
    assert method.status is AcpMethodStatus.OUT_OF_SCOPE
    assert "local" in method.note


def test_the_acp_mcp_descriptor_is_not_implementable_and_says_why() -> None:
    """§5: it must be refused with a stated reason rather than silently dropped."""
    descriptor = next(d for d in acp_mcp_descriptor_registry() if d.name == "AcpMcpServer")
    assert descriptor.status is AcpMethodStatus.NOT_IMPLEMENTABLE
    assert "serverId" in descriptor.note


def test_absence_is_reported_as_absence_rather_than_as_an_empty_success() -> None:
    """P6: "no shell installed" must not read the same as "a shell with nothing to show"."""
    presence = shell_presence()
    assert presence.serving is False
    assert presence.shell_module_present is False
    assert presence.sdk_installed is False
    assert "#649" in presence.reason
    assert presence.transport == ACP_TRANSPORT


def test_the_summary_carries_what_the_surface_renders() -> None:
    summary = conformance_summary()
    assert summary.serving is False
    assert summary.transport == ACP_TRANSPORT
    assert summary.sdk_version_specified == ACP_SDK_VERSION
    assert len(summary.methods) == len(acp_method_registry())

    agent_implemented = len([m for m in implemented() if m.side is AcpSide.AGENT])
    client_implemented = len([m for m in implemented() if m.side is AcpSide.CLIENT])
    assert summary.counts["agent"].implemented == agent_implemented
    assert summary.counts["client"].implemented == client_implemented

    for side in ("agent", "client"):
        counts = summary.counts[side]
        assert counts.total == (
            counts.implemented
            + counts.not_implemented
            + counts.not_implementable
            + counts.out_of_scope
        )


def test_the_mcp_loader_warning_names_the_actual_hazard() -> None:
    """The env-expansion hazard in §5 is the reason descriptors bypass the ordinary loader."""
    warning = conformance_summary().mcp_loader_warning
    assert "parse_config_dict" in warning
    assert "${VAR}" in warning
