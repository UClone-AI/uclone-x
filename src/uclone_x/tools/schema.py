"""The tool parameter schema a model is sent, without the bytes that restate (#1542).

`BaseTool.parameters_schema` is Pydantic's raw `model_json_schema()`, and tests, argument
validation and anything that inspects a tool read it as it is. What a model is sent goes
through `advertised_parameters_schema` first, once, where the agent builds its
`ToolDefinition`s — so all four connectors, the request's token estimate and the context
snapshot see the same bytes.

The pass removes only what the rest of the schema already says:

* every `title` annotation — Pydantic derives it from the field or class name, which the
  model already reads as the property key or the tool name;
* the top-level `description`, the params model's docstring, which restates the tool's
  own description;
* `anyOf: [X, {"type": "null"}]` with `default: null` becomes `X` (with more branches,
  only the `null` one goes): the field is optional (not in `required`) and leaving it out
  means unset. Validation is Pydantic's, not the
  schema's, so an explicit `null` a model sends is still accepted;
* a trailing `(default: ...)` in a description whose node carries a `default` key.

Everything with a validation meaning survives: `type`, `properties`, `required`, `enum`,
`const`, `minimum`/`maximum`, `additionalProperties`, `items`, `$defs` and `$ref`. A
property or `$defs` entry that happens to be *named* `title` or `description` is a
parameter, not an annotation, and is kept. Values under `default`, `enum`, `const` and
`examples` are data and are never walked into.

Long list defaults (`file_search.ignore_patterns`) are kept: a model that passes the list
replaces the defaults, so it needs to see them, and the issue leaves dropping them open.

The output is deterministic — key order follows the input — because the tools layer is
part of the cacheable prefix.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["advertised_parameters_schema"]

# Keywords whose value is a map of name -> subschema.
_SCHEMA_MAPS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)
# Keywords whose value is a subschema, or a list of subschemas.
_SCHEMA_VALUES = frozenset(
    {
        "items",
        "prefixItems",
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "anyOf",
        "oneOf",
        "allOf",
    }
)
_NULL_BRANCH: dict[str, Any] = {"type": "null"}
_DEFAULT_SUFFIX = re.compile(r"\s*\(default[^)]*\)\s*$", re.IGNORECASE)


def advertised_parameters_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return `schema` as it is sent to a model: the same constraints, fewer bytes.

    The input is not modified.
    """
    compacted = _compact_node(schema)
    compacted.pop("description", None)
    return compacted


def _compact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _compact_node(value)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(value, list):
        return [_compact_value(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def _compact_node(node: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {
                name: _compact_value(sub)
                for name, sub in value.items()  # pyright: ignore[reportUnknownVariableType]
            }
        elif key in _SCHEMA_VALUES:
            out[key] = _compact_value(value)
        else:
            out[key] = value

    description = out.get("description")
    if "default" in out and isinstance(description, str):
        stripped = _DEFAULT_SUFFIX.sub("", description)
        if stripped:
            out["description"] = stripped

    return _collapse_optional_null(out)


def _collapse_optional_null(node: dict[str, Any]) -> dict[str, Any]:
    """Drop the `null` branch of an `anyOf` whose default is `null`.

    One branch left becomes the node itself, when none of its keys would overwrite the
    node's own; several stay an `anyOf`.
    """
    branches = node.get("anyOf")
    if not (
        isinstance(branches, list)
        and _NULL_BRANCH in branches
        and "default" in node
        and node["default"] is None
    ):
        return node
    others: list[Any] = [b for b in branches if b != _NULL_BRANCH]  # pyright: ignore[reportUnknownVariableType]
    if not others:
        return node
    rest = {k: v for k, v in node.items() if k != "default"}
    if len(others) > 1:
        rest["anyOf"] = others
        return rest
    del rest["anyOf"]
    other = others[0]
    if not isinstance(other, dict) or set(other) & set(rest):  # pyright: ignore[reportUnknownArgumentType]
        return node
    return {**other, **rest}  # pyright: ignore[reportUnknownArgumentType]
