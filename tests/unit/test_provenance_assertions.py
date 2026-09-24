"""Tests for the shared provenance assertion helper (#377).

Each test here corresponds to a mistake found in the approved revision of
the hybrid quality-assurance design note, so the helper cannot regress
into any of them.
"""

from __future__ import annotations

import pytest

from tests.support.provenance import assert_provenance
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    MissingProvenanceError,
    Provenance,
    ServiceRef,
)
from uclone_x.llm.models import FinishReason, ModelResponse, TokenUsage


def _response(prov: Provenance | None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content="ok",
        usage=TokenUsage(provider="ollama", model="qwen2.5-coder:7b"),
        model_name="qwen2.5-coder:7b",
        provenance=prov,
    )


def _primary(provider: str = "ollama", model: str | None = "qwen2.5-coder:7b") -> Provenance:
    ref = ServiceRef(provider=provider, model=model)
    return Provenance(path=ExecutionPath.PRIMARY, requested=ref, served_by=ref)


def test_matching_provenance_passes_and_is_returned() -> None:
    prov = _primary()

    returned = assert_provenance(_response(prov), provider="ollama", model="qwen2.5-coder:7b")

    assert returned is prov
    assert returned.degraded is False


def test_absent_provenance_raises_rather_than_passing() -> None:
    """`provenance` is `Provenance | None`; a test must never read through the None."""
    with pytest.raises(MissingProvenanceError, match="carries no provenance"):
        assert_provenance(_response(None), provider="ollama")


def test_wrong_provider_fails() -> None:
    with pytest.raises(AssertionError, match="expected served_by.provider 'gemini'"):
        assert_provenance(_response(_primary()), provider="gemini")


def test_wrong_model_fails() -> None:
    """`served_by` is a ServiceRef, so the model lives at `served_by.model`."""
    with pytest.raises(AssertionError, match="expected served_by.model 'llama3'"):
        assert_provenance(_response(_primary()), provider="ollama", model="llama3")


def test_model_check_is_skipped_when_no_model_is_expected() -> None:
    assert_provenance(_response(_primary(model=None)), provider="ollama")


def test_silent_model_substitution_fails_by_default() -> None:
    """A provider that answers with a different model is degraded (P6), even on primary."""
    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="gemini", model="gemini-1.5-pro"),
        served_by=ServiceRef(provider="gemini", model="gemini-1.5-pro-002"),
    )

    with pytest.raises(AssertionError, match="degraded result"):
        assert_provenance(_response(prov), provider="gemini", model="gemini-1.5-pro-002")


def test_degradation_can_be_allowed_when_it_is_the_behaviour_under_test() -> None:
    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="gemini", model="gemini-1.5-pro"),
        served_by=ServiceRef(provider="gemini", model="gemini-1.5-pro-002"),
    )

    returned = assert_provenance(
        _response(prov),
        provider="gemini",
        model="gemini-1.5-pro-002",
        allow_degraded=True,
    )

    assert returned.degraded is True


def test_unexpected_execution_path_fails() -> None:
    """A failover that the test did not ask for must not read as a primary result."""
    prov = Provenance(
        path=ExecutionPath.FAILOVER,
        requested=ServiceRef(provider="openai", model="gpt-4o"),
        served_by=ServiceRef(provider="anthropic", model="claude-sonnet-4"),
        attempts=(AttemptRecord(provider="openai", model="gpt-4o", error_class="TimeoutError"),),
    )

    with pytest.raises(AssertionError, match="expected provenance.path 'primary'"):
        assert_provenance(
            _response(prov), provider="anthropic", model="claude-sonnet-4", allow_degraded=True
        )


def test_expected_failover_path_passes() -> None:
    prov = Provenance(
        path=ExecutionPath.FAILOVER,
        requested=ServiceRef(provider="openai", model="gpt-4o"),
        served_by=ServiceRef(provider="anthropic", model="claude-sonnet-4"),
        attempts=(AttemptRecord(provider="openai", model="gpt-4o", error_class="TimeoutError"),),
    )

    returned = assert_provenance(
        _response(prov),
        provider="anthropic",
        model="claude-sonnet-4",
        path=ExecutionPath.FAILOVER,
        allow_degraded=True,
    )

    assert returned.attempts[0].error_class == "TimeoutError"


def test_provenance_has_no_latency_field() -> None:
    """Pins the design-document error: `provenance.latency_ms` does not exist.

    If a future change adds it, this test fails and the assertion helper plus
    §6.1/§9 of the design document should be updated together.
    """
    assert not hasattr(_primary(), "latency_ms")
    assert "latency_ms" not in Provenance.model_fields
