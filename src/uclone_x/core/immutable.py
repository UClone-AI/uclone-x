"""Immutable mapping fields for frozen models.

`frozen=True` prevents attribute rebinding; it does nothing about the contents of a
`dict` field, so `event.payload["k"] = v` succeeds on a frozen model. That matters more
here than in an ordinary codebase: P2 requires co-located agents to exchange events by
reference with no serialisation, and `docs/a2a-protocol-spec.md` section 10.3 carries the
obligation forward as a requirement on the fastpath — "where objects are shared by
reference they MUST be immutable (frozen models) so zero-copy cannot become shared
mutable state".

These aliases validate an incoming mapping and then hand back a read-only view of a
private copy, so a producer cannot retain a writable reference to what it published and
a consumer cannot mutate what it received. Each pairs the validator with a serializer
that unwraps the view back into a plain `dict`, because `mappingproxy` is not JSON
serialisable and an envelope that cannot be serialised cannot cross the A2A boundary or
reach a telemetry exporter.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, TypeVar, cast

from pydantic import AfterValidator, JsonValue, PlainSerializer

__all__ = [
    "ImmutableIntMapping",
    "ImmutableJsonMapping",
    "ImmutableMapping",
    "ImmutableStrMapping",
    "freeze_mapping",
    "unwrap_immutable",
]

_V = TypeVar("_V")


def _deep_freeze(val: object) -> object:
    """Recursively freeze mappings into MappingProxyType and sequences into tuples."""
    if isinstance(val, Mapping):
        mapping = cast(Mapping[str, object], val)
        return MappingProxyType({str(k): _deep_freeze(v) for k, v in mapping.items()})
    elif isinstance(val, (list, tuple)):
        seq = cast(list[object] | tuple[object, ...], val)
        return tuple(_deep_freeze(item) for item in seq)
    return val


def freeze_mapping(value: Mapping[str, _V]) -> Mapping[str, _V]:
    """Return a deep read-only view over a private copy of `value`."""
    return cast(Mapping[str, _V], _deep_freeze(value))


def _deep_unwrap(val: object) -> object:
    """Recursively unwrap MappingProxyType into dict and tuples into lists for serialization."""
    if isinstance(val, Mapping):
        mapping = cast(Mapping[str, object], val)
        return {str(k): _deep_unwrap(v) for k, v in mapping.items()}
    elif isinstance(val, (list, tuple)):
        seq = cast(list[object] | tuple[object, ...], val)
        return [_deep_unwrap(item) for item in seq]
    return val


def unwrap_immutable(value: object) -> object:
    """Recursively unwrap MappingProxyType into dict and tuples into lists for serialization."""
    return _deep_unwrap(value)


def _unwrap_json(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], _deep_unwrap(value))


def _unwrap_str(value: Mapping[str, str]) -> dict[str, str]:
    return dict(value)


def _unwrap_int(value: Mapping[str, int]) -> dict[str, int]:
    return dict(value)


ImmutableMapping = Annotated[
    Mapping[str, JsonValue],
    AfterValidator(freeze_mapping),
    PlainSerializer(_unwrap_json, return_type=dict[str, JsonValue]),
]
"""A JSON-shaped mapping that cannot be mutated after validation."""

ImmutableStrMapping = Annotated[
    Mapping[str, str],
    AfterValidator(freeze_mapping),
    PlainSerializer(_unwrap_str, return_type=dict[str, str]),
]
"""A string-to-string mapping that cannot be mutated after validation."""

ImmutableJsonMapping = Annotated[
    Mapping[str, JsonValue],
    AfterValidator(freeze_mapping),
    PlainSerializer(_unwrap_json, return_type=dict[str, JsonValue]),
]
"""An arbitrary JSON-document mapping (e.g. a JSON Schema) that cannot be mutated."""

ImmutableIntMapping = Annotated[
    Mapping[str, int],
    AfterValidator(freeze_mapping),
    PlainSerializer(_unwrap_int, return_type=dict[str, int]),
]
"""A name-to-count mapping (e.g. tokens per provider) that cannot be mutated."""
