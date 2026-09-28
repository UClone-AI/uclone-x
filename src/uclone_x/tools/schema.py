"""The tool parameter schema a model is sent, without the bytes that restate (#1542).

`BaseTool.parameters_schema` is Pydantic's raw `model_json_schema()`, and tests, argument
validation and anything that inspects a tool read it as it is. What a model is sent goes
through `advertised_parameters_schema` first, once, where the agent builds its
`ToolDefinition`s — so all four connectors, the request's token estimate and the context
snapshot see the same bytes.

The pass removes only what the rest of the schema already says:

* every `title` annotation — Pydantic derives it from the field or class name, which the
  model already reads as the property key or the tool name;
* the top-level `description` of a schema Pydantic derived from a tool's params model
  (`params_model=True`): it is the model's docstring, which restates the tool's own
  description. An MCP server's or a hand-written schema keeps it, because nothing says
  the tool description carries what it says (#1543);
* `anyOf: [X, {"type": "null"}]` with `default: null` becomes `X` (with more branches,
  only the `null` one goes): the field is optional (not in `required`) and leaving it out
  means unset. Validation is Pydantic's, not the
  schema's, so an explicit `null` a model sends is still accepted;
* a trailing `(default: x)` in a description whose node's `default` is that same `x`.
  Any other parenthetical stays: "(defaults to the workspace root when omitted)" or
  "(default: 3, max 10)" says something the `default` key does not (#1543).

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

import json
import re
from typing import Any

from uclone_x.tools.base import BaseTool
from uclone_x.tools.protocols import ToolProtocol

__all__ = ["advertised_parameters_schema", "advertised_tool_parameters"]

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
_DEFAULT_SUFFIX = re.compile(r"\s*\(default:\s*([^()]*?)\s*\)\s*$", re.IGNORECASE)


def advertised_tool_parameters(tool: ToolProtocol) -> dict[str, Any]:
    """`tool`'s parameter schema as the agent sends it: `advertised_parameters_schema`,
    told whether the schema is the one Pydantic derives from the tool's params model.

    Only a `BaseTool` that keeps `BaseTool.parameters_schema` has one; an MCP tool, a
    `LocalTool` or a subclass that overrides the property hands in its own schema.
    """
    # The class attribute is the `BaseTool.parameters_schema` property object itself only
    # when the class inherits it unchanged; an override, an MCP tool or a `LocalTool`
    # has another one.
    inherited = getattr(type(tool), "parameters_schema", None)
    from_params_model = inherited is BaseTool.parameters_schema
    return advertised_parameters_schema(tool.parameters_schema, params_model=from_params_model)


def advertised_parameters_schema(
    schema: dict[str, Any], *, params_model: bool = False
) -> dict[str, Any]:
    """Return `schema` as it is sent to a model: the same constraints, fewer bytes.

    `params_model` says the schema is a params model's `model_json_schema()`, whose
    top-level `description` is the docstring and is dropped. The input is not modified.
    """
    compacted = _compact_node(schema)
    if params_model:
        compacted.pop("description", None)
    return compacted


def _states_default(text: str, default: Any) -> bool:
    """Whether `text`, from a `(default: text)` suffix, is `default` written out.

    Quotes around the value are ignored; it matches as JSON (`true`, `10`) or as Python
    writes it (`True`, `None`, `fast`).
    """
    text = text.strip("`'\"")
    return text in {json.dumps(default, ensure_ascii=False), str(default)}


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
        match = _DEFAULT_SUFFIX.search(description)
        if match and match.start() > 0 and _states_default(match.group(1), out["default"]):
            out["description"] = description[: match.start()]

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
