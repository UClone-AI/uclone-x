"""The ACP conformance registry: what it may claim, and how it reports absence.

Its agreement with the ACP specification is checked in the lab repository, where that
document lives; the public summary is `docs/public/protocols.md`.
"""

from __future__ import annotations

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
