"""JSON wire-serialisation boundary for A2A envelopes and bus events.

`a2a/models.py` declares its envelopes `strict=True, extra="forbid", frozen=True`, so
does `AgentEvent` (`engine/event_bus.py`), and `Provenance` (P6) additionally exposes
`degraded` as a `computed_field`. Those choices make the round trip
`model_dump(mode="json")` -> `model_validate(...)` asymmetric, in two ways that are easy
to get wrong at a call site:

1. **A derived field is not an input field.** `model_dump` emits `degraded`, but
   `degraded` is not something a sender gets to state: it is computed from `requested`
   and `served_by` so that "a substituted result cannot be reported as clean"
   (`docs/llm-agnostic-interface.md`). Egress therefore *excludes* it, and the wire
   carries the authoritative fields only. On ingress `Provenance`'s own `mode="before"`
   validator (`core/provenance.py`, issue #28) **discards** any `degraded` it is handed
   and the receiver recomputes it — a peer's assertion is corrected, not raised on. Both
   halves are needed: the exclusion keeps UClone-X from putting a derived claim on the
   wire for a foreign peer to trust, and the strip keeps a foreign peer from getting one
   past us. Verified by `test_wire_payload_omits_derived_provenance_field` and
   `test_wire_ignores_peer_asserted_degraded`.
2. **Strict validation must be given JSON, not a `dict`.** In strict mode Pydantic
   accepts a JSON string for an enum and a JSON array for a `tuple` *only while it is
   parsing JSON*. Handing the same payload to `model_validate` as an already-decoded
   `dict` puts it in Python mode, where `"completed"` is not a `TaskStatus`, `"failover"`
   is not an `ExecutionPath`, and `[]` is not a `tuple[AttemptRecord, ...]`. Ingress
   therefore goes through `model_validate_json`. The `mode="before"` strip in (1) does
   not defeat this: JSON parsing mode survives it, verified by parsing a failover
   provenance whose `path` arrives as the string `"failover"`.

Asymmetry (2) is not repaired by loosening a model — that would mean giving up
`strict=True`. The fix belongs here, at the boundary, so there is one implementation of
it rather than one per endpoint.

Ingress does **not** default or repair the *authoritative* fields. A payload that omits
`provenance` parses to `None` and is rejected by the caller (`require_provenance` /
`MissingProvenanceError`) — P6's zero-silent-fallback rule.

`AgentEvent` is serialised here too, because an event that carries a result carries P6
provenance in a typed envelope field (issue #51, Option B) and it must survive the same
boundary under the same derived-field policy. The other transport a bus event takes —
P2's in-process fastpath — moves the frozen `AgentEvent` by reference and so carries the
`Provenance` object itself with no serialisation at all.
"""

from __future__ import annotations

from typing import Any, Final

from uclone_x.a2a.models import TaskMessage, TaskResult
from uclone_x.engine.event_bus import AgentEvent

__all__ = [
    "agent_event_from_wire_json",
    "agent_event_to_wire",
    "agent_event_to_wire_json",
    "task_message_to_wire",
    "task_result_from_wire_json",
    "task_result_to_wire",
    "task_result_to_wire_json",
]

# Derived (`computed_field`) members that must not appear on the wire, keyed by the
# field that owns them. See point (1) in the module docstring.
_TASK_RESULT_DERIVED: Final[dict[str, set[str]]] = {"provenance": {"degraded"}}

# `AgentEvent.provenance` owns the same derived member, and for the same reason.
_AGENT_EVENT_DERIVED: Final[dict[str, set[str]]] = {"provenance": {"degraded"}}


def task_message_to_wire(message: TaskMessage) -> dict[str, Any]:
    """Serialise a dispatch envelope to JSON-mode primitives for an HTTP body."""
    return message.model_dump(mode="json")


def task_result_to_wire(result: TaskResult) -> dict[str, Any]:
    """Serialise a result envelope to JSON-mode primitives, without derived fields."""
    return result.model_dump(mode="json", exclude=_TASK_RESULT_DERIVED)


def task_result_to_wire_json(result: TaskResult) -> str:
    """Serialise a result envelope to a JSON document, without derived fields."""
    return result.model_dump_json(exclude=_TASK_RESULT_DERIVED)


def task_result_from_wire_json(payload: str | bytes) -> TaskResult:
    """Parse a result envelope from a JSON document under strict validation.

    Raises `pydantic.ValidationError` on a non-conformant payload, which transports map
    to `InvalidAgentResponseError`. A payload asserting the derived
    `provenance.degraded` is not rejected here: `Provenance` strips it and recomputes
    the value, so the assertion cannot survive either way (see the module docstring).
    """
    return TaskResult.model_validate_json(payload)


def agent_event_to_wire(event: AgentEvent) -> dict[str, Any]:
    """Serialise a bus event to JSON-mode primitives, without derived fields."""
    return event.model_dump(mode="json", exclude=_AGENT_EVENT_DERIVED)


def agent_event_to_wire_json(event: AgentEvent) -> str:
    """Serialise a bus event to a JSON document, without derived fields."""
    return event.model_dump_json(exclude=_AGENT_EVENT_DERIVED)


def agent_event_from_wire_json(payload: str | bytes) -> AgentEvent:
    """Parse a bus event from a JSON document under strict validation.

    Ingress for the event envelope, including its typed `provenance` (issue #51). Like
    `task_result_from_wire_json` this must be given JSON rather than a decoded `dict`:
    `AgentEvent` is `strict=True`, so `type`, `source`, `priority` and
    `provenance.path` only accept their JSON forms while Pydantic is parsing JSON.
    """
    return AgentEvent.model_validate_json(payload)
