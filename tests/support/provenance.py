"""Provenance assertions for Tier 3 (and any other) tests.

This exists because the same four mistakes were made in a design document that had been
approved, and would otherwise have been copied into every live test written against it:

1. `Provenance` has no `latency_ms`. Its fields are `path`, `requested`, `served_by`,
   `attempts`, plus a **computed** `degraded` (`uclone_x/core/provenance.py`).
2. `served_by` is a `ServiceRef(provider, model)`, so comparing it to a model name never
   holds — the comparison silently fails rather than raising.
3. `ModelResponse.provenance` is `Provenance | None`, so unguarded attribute access does
   not survive `pyright --strict`. `require_provenance` is the repository's existing
   answer and is used here instead of an `assert ... is not None`.
4. `degraded` is derived from `served_by != requested`, which includes a *model* alias on
   a `primary` path. It is therefore not a synonym for "a failover happened", and a test
   that wants to allow a provider's own alias repair has to say so.

One helper, so the correction lives in one place.
"""

from __future__ import annotations

from uclone_x.core.provenance import ExecutionPath, Provenance, require_provenance
from uclone_x.llm.models import ModelResponse

__all__ = ["assert_provenance"]


def assert_provenance(
    response: ModelResponse,
    *,
    provider: str,
    model: str | None = None,
    path: ExecutionPath = ExecutionPath.PRIMARY,
    allow_degraded: bool = False,
    envelope: str = "ModelResponse",
) -> Provenance:
    """Assert that `response` is attributable to `provider` (and `model`), and return it.

    Args:
        response: The envelope under test.
        provider: The provider expected to have served the response.
        model: The model expected to have served it. `None` skips the model check, which
            is the right call only for providers that do not name one.
        path: Expected execution path. `PRIMARY` means nothing was retried or failed
            over; `attempts` is then required to be empty by `Provenance` itself.
        allow_degraded: Permit `served_by != requested`. Left `False` by default so a
            silent model substitution fails the test (P6) rather than passing quietly.
        envelope: Name used in the `MissingProvenanceError` message.

    Raises:
        MissingProvenanceError: If `response.provenance` is absent — a result with no
            attribution is not a result a test may accept.
        AssertionError: If attribution does not match.
    """
    prov = require_provenance(response.provenance, envelope)

    assert prov.path is path, f"expected provenance.path {path.value!r}, got {prov.path.value!r}"
    assert prov.served_by.provider == provider, (
        f"expected served_by.provider {provider!r}, got {prov.served_by.provider!r}"
    )
    if model is not None:
        assert prov.served_by.model == model, (
            f"expected served_by.model {model!r}, got {prov.served_by.model!r}"
        )
    if not allow_degraded:
        assert prov.degraded is False, (
            "provenance reports a degraded result: requested "
            f"{prov.requested.provider}/{prov.requested.model} but was served by "
            f"{prov.served_by.provider}/{prov.served_by.model}. Pass allow_degraded=True "
            "only if the substitution is the behaviour under test."
        )
    return prov
