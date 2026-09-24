"""In-band execution provenance, required by Principle 6.

`docs/principles/details/p6-fail-fast-observability.md` permits a declared retry or
provider failover only when it is attributable **in band**: a telemetry span informs a
human reading a trace later, but it never enters the calling agent's reasoning context,
and "an agent that cannot tell a primary result from a failover result cannot reason
about the reliability of its own conclusions".

Four properties of that rule are enforced here rather than left to convention:

* `path` has no default. A caller states which path produced the value.
* `degraded` is computed from `requested` and `served_by`, so a producer cannot report a
  substituted result as undegraded.
* `attempts` is empty if and only if `path` is `PRIMARY`, which the principle states as
  an "if and only if".
* A `PRIMARY` result is served by the *provider* that was requested. Its **model** may
  differ, and then `degraded` is `True` on a `primary` path — that is the provider-side
  alias case, which the principle wants visible rather than forbidden.

**Caller Obligations for Non-Primary Provenance (P6 Checks 4 & 5):**
When constructing and returning or publishing a non-primary provenance (e.g. `path=ExecutionPath.FAILOVER`
or `ExecutionPath.RETRY`):
1. **P6 Check 5 (Telemetry correlation)**: The caller MUST emit an OpenTelemetry-compatible `failover.event`
   span (`tracer.start_span("failover.event", attributes={...})`) and thread its `span_id` into
   the corresponding `AttemptRecord.span_id`.
2. **P6 Check 4 (Decision plane announcement)**: When publishing results across the event bus, the caller
   MUST publish an `EventType.PROVIDER_FAILOVER` (or `RETRY`) notice onto `EventBus` strictly ordered
   *before* the final result/reply event, ensuring `(priority, sequence)` is lower than the result event.

The validators enforce what the principle states, and no more. Where a rule is narrower
than a reader might expect, its docstring says which sentence of P6 it comes from and
what it deliberately does not catch, so the next tightening starts from the principle
rather than from the previous tightening (issue #149).

Absence is handled by `require_provenance`: the field is typed `Provenance | None` on
result envelopes so that a non-conformant value — one deserialized from a peer, or built
by a component that has not been updated — is *representable* and can be rejected. It is
never defaulted, because a missing marker read as "nothing went wrong" is the
default-masquerading-as-a-real-answer that Principle 6 exists to forbid.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from uclone_x.errors import MissingProvenanceError

__all__ = [
    "AttemptRecord",
    "ExecutionPath",
    "Provenance",
    "ServiceRef",
    "require_provenance",
]


class ExecutionPath(StrEnum):
    """Which path produced a value (Principle 6, `provenance.path`).

    Using `path=FAILOVER` or `path=RETRY` obligates the producer to:
    (a) Emit a `failover.event` telemetry span and capture its `span_id` into `AttemptRecord.span_id` (Check 5).
    (b) Publish an `EventType.PROVIDER_FAILOVER` (or `RETRY`) event onto the bus strictly ordered before the result (Check 4).
    """

    PRIMARY = "primary"
    RETRY = "retry"
    FAILOVER = "failover"


class ServiceRef(BaseModel):
    """A provider and, where one applies, the model within it."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    provider: str
    model: str | None = None


class AttemptRecord(BaseModel):
    """One failed attempt preceding the value that was finally returned.

    Principle 6 Check 5 requires `span_id` to correlate 1:1 with an emitted `failover.event`
    telemetry span when `path == 'failover'`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    provider: str
    model: str | None = None
    error_class: str
    status_code: int | None = None
    span_id: str | None = None


class Provenance(BaseModel):
    """Attribution travelling with a value across a component boundary.

    When `path != ExecutionPath.PRIMARY`, callers must satisfy P6 falsifiable checks:
    - Check 4: Publish matching `PROVIDER_FAILOVER` event before the result event on `EventBus`.
    - Check 5: Emit a `failover.event` span in telemetry and thread `span_id` into `attempts[*].span_id`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: ExecutionPath
    requested: ServiceRef
    served_by: ServiceRef
    attempts: tuple[AttemptRecord, ...] = Field(default_factory=tuple)

    @computed_field
    @property
    def degraded(self) -> bool:
        """True if and only if `served_by` differs from `requested`."""
        return self.served_by != self.requested

    @model_validator(mode="before")
    @classmethod
    def _strip_computed_degraded(cls, data: object) -> object:
        """Strip peer-supplied or serialized 'degraded' field so it is always recomputed (P6, issue #28)."""
        if isinstance(data, Mapping):
            mapping = cast(Mapping[str, object], data)
            cleaned: dict[str, object] = dict(mapping)
            cleaned.pop("degraded", None)
            if "attempts" in cleaned and isinstance(cleaned["attempts"], list):
                raw_attempts = cast(list[object], cleaned["attempts"])
                cleaned["attempts"] = tuple(raw_attempts)
            return cleaned
        return data

    @model_validator(mode="after")
    def _attempts_match_path(self) -> Self:
        """Enforce Principle 6: `attempts` is empty if and only if `path` is primary."""
        if self.path is ExecutionPath.PRIMARY and self.attempts:
            raise ValueError("provenance.attempts must be empty when path is 'primary'")
        if self.path is not ExecutionPath.PRIMARY and not self.attempts:
            raise ValueError(
                f"provenance.attempts must be non-empty when path is '{self.path.value}'"
            )
        return self

    @model_validator(mode="after")
    def _primary_path_names_the_requested_provider(self) -> Self:
        """A `primary` result is served by the provider that was requested (#136, #149).

        **Where this comes from.** P6 defines the recovery vocabulary by who served the
        request: a re-attempt "hits the same provider (retry) or a different one
        (provider failover)". So if `served_by.provider` is not `requested.provider`,
        the requested provider did not serve this value, and exactly one of two things
        happened. Either something was attempted and failed — in which case the path is
        `retry`/`failover` and `_attempts_match_path` above requires the attempts to be
        recorded — or nothing was attempted, which is the #136 echo: a substitution P6's
        Classification Procedure forbids at question 1, before attribution is reached.
        Neither is `primary`. This validator is a restatement of P6's own vocabulary, not
        an addition to it.

        **What it deliberately does not catch, and why (#149).** It compares `provider`
        only. It says nothing about `model`, so

            requested=gemini/gemini-1.5-pro, served_by=gemini/gemini-1.5-pro-002

        is legal on a `primary` path and reports `degraded=True`. That is correct and it
        is the case P6 cares most about — "a model swap changes tool-calling behaviour,
        output formatting, and determinism, so an agent mid-plan may continue under
        assumptions that no longer hold". One call was made, the requested provider
        answered, nothing failed, so falsifiable check 4 rightly does not apply: there is
        no failover to announce on the bus. An earlier version of this rule compared the
        whole `ServiceRef` and rejected that value, which forbade the alias case and made
        the connectors' own repair unrepresentable (#149). P6 nowhere says that `primary`
        implies not-degraded; that was an interpretation hardened into a constraint.

        **Residual, stated rather than hidden.** A same-provider *substitution* — asked
        for `gpt-4o`, silently served `gpt-4o-mini`, nothing recorded — is structurally
        identical to alias resolution, and no rule over `(requested, served_by, path,
        attempts)` can separate them. Telling them apart would need vocabulary P6 does
        not define (a declared alias map, or a producer-supplied discriminant), and
        inventing it here would repeat the over-reach #149 reports. The behaviour stays
        forbidden by the Classification Procedure.

        The available control is stronger than reading connectors by hand, though, and
        is worth stating because the weaker one is the obvious thing to reach for:
        `path is PRIMARY and degraded` is **machine-detectable**. It cannot say which of
        the two a given result is, but it enumerates every one of them for adjudication —
        a report a consumer, a test, or a lint can produce, rather than a review someone
        has to remember to do. That is the follow-on control for this gap.
        """
        if (
            self.path is ExecutionPath.PRIMARY
            and self.served_by.provider != self.requested.provider
        ):
            raise ValueError(
                "provenance.path 'primary' cannot be served by a different provider: "
                f"served_by provider {self.served_by.provider!r} differs from requested "
                f"provider {self.requested.provider!r}. Either return no value, or use "
                "'failover' -- which obliges you to emit a `failover.event` span, thread "
                "its span_id into each AttemptRecord (P6 check 5), and publish a "
                "PROVIDER_FAILOVER notice before the result (P6 check 4). Recording the "
                "attempt alone is not compliance."
            )
        return self

    @classmethod
    def primary(
        cls,
        provider: str,
        model: str | None = None,
        served_model: str | None = None,
    ) -> Provenance:
        """Build the unremarkable case: the requested service answered, first time.

        `model` is what was asked of `provider`. `served_model` is what `provider` says
        it actually ran, and is passed when a connector can read it back from the
        response; omit it when there is nothing to read back, and `served_by` is then
        identical to `requested`.

        Passing both is how a provider-side alias stays visible: `served_model` equal to
        `model` yields `degraded=False`, and a resolved alias such as `gemini-1.5-pro`
        answered by `gemini-1.5-pro-002` yields `degraded=True` while `path` stays
        `primary`, because one call was made and nothing failed. Collapsing the two onto
        the served name — which is what reading `model_name` off the response and passing
        it as `model` did in all four connectors — erases the alias and reports
        `degraded=False` for a model the caller never asked for (#149).
        """
        requested = ServiceRef(provider=provider, model=model)
        served = (
            requested if served_model is None else ServiceRef(provider=provider, model=served_model)
        )
        return cls(path=ExecutionPath.PRIMARY, requested=requested, served_by=served)


def require_provenance(provenance: Provenance | None, envelope: str) -> Provenance:
    """Return `provenance`, or raise if it is absent.

    Principle 6's sixth falsifiable check: "A consumer handed a result envelope with
    `provenance` omitted raises rather than proceeding." Consumers call this instead of
    reading the field directly, so the check has one implementation.
    """
    if provenance is None:
        raise MissingProvenanceError(
            f"{envelope} carries no provenance; refusing to treat it as a primary result"
        )
    return provenance
