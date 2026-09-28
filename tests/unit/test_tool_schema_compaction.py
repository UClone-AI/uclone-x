"""The tool schema a model is sent, without the bytes that restate (#1542).

`advertised_parameters_schema` must remove only restated information: titles, the
params model's docstring, `anyOf`-null with a null default, and a `(default: ...)` suffix
the `default` key already carries. Every constraint a provider or a validator reads stays.
"""

from __future__ import annotations

import copy
import json
from enum import StrEnum
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from tests.conftest import finish_after_tools
from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.provenance import Provenance
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.models import FinishReason, LLMRequest, ModelResponse, TokenUsage
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.memory.tools import (
    QueryMemoryFactsTool,
    RecordMemoryFactTool,
    RetractMemoryFactTool,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.filesystem import FileReadParams, FileReadTool
from uclone_x.tools.builtin.skill_loader import LoadSkillTool
from uclone_x.tools.client import MCPTool
from uclone_x.tools.protocols import ToolProtocol
from uclone_x.tools.registry import LocalTool, ToolRegistry, create_default_registry
from uclone_x.tools.schema import advertised_parameters_schema, advertised_tool_parameters

# --- a params model exercising every rule ------------------------------------------


class Colour(StrEnum):
    RED = "red"
    BLUE = "blue"


class Chapter(BaseModel):
    """A nested model: it lands in `$defs` and is reached by `$ref`."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="The chapter's title.")
    number: int = Field(ge=1, description="Its position.")


class SampleParams(BaseModel):
    """Parameters for the sample tool."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="A parameter that happens to be called title.")
    colour: Colour = Field(description="Which colour.")
    limit: int = Field(default=10, ge=1, le=100, description="How many (default: 10)")
    note: str | None = Field(default=None, description="An optional note.")
    maybe: int | None = Field(description="Required, and null is a valid answer.")
    fallback: int | None = Field(default=5, description="Nullable with a real default.")
    chapter: Chapter | None = Field(default=None, description="An optional chapter.")
    when: str | int | None = Field(default=None, description="A label or a number.")
    chapters: list[Chapter] = Field(default_factory=list[Chapter])


def _raw() -> dict[str, Any]:
    return SampleParams.model_json_schema()


def _schema_nodes(node: Any) -> list[dict[str, Any]]:
    """Every subschema in `node`, never a `properties` map or a `default` value."""
    found: list[dict[str, Any]] = []
    if isinstance(node, list):
        for item in node:  # pyright: ignore[reportUnknownVariableType]
            found += _schema_nodes(item)
        return found
    if not isinstance(node, dict):
        return found
    found.append(node)  # pyright: ignore[reportUnknownArgumentType]
    for key, value in node.items():  # pyright: ignore[reportUnknownVariableType]
        if key in ("properties", "$defs") and isinstance(value, dict):
            for sub in value.values():  # pyright: ignore[reportUnknownVariableType]
                found += _schema_nodes(sub)
        elif key in ("items", "anyOf", "oneOf", "allOf", "additionalProperties", "not"):
            found += _schema_nodes(value)
    return found


def _is_null_union(node: dict[str, Any]) -> bool:
    branches = node.get("anyOf")
    return isinstance(branches, list) and {"type": "null"} in branches


# --- each rule -----------------------------------------------------------------------


def test_titles_are_dropped_at_every_level() -> None:
    """Every `title` annotation goes, in properties, `$defs` and at the top.

    Killed by: src/uclone_x/tools/schema.py :: if key == "title" and isinstance(value, str):
    Becomes: if key == "titleX" and isinstance(value, str):
    """
    raw = _raw()
    assert any("title" in n for n in _schema_nodes(raw)), "the fixture must carry titles"
    out = advertised_parameters_schema(raw)
    assert [n for n in _schema_nodes(out) if "title" in n] == []


def test_a_parameter_named_title_is_kept() -> None:
    """A property named `title` is a parameter, not an annotation (the issue's script drops it).

    Killed by: src/uclone_x/tools/schema.py :: for name, sub in value.items()
    Becomes: for name, sub in value.items() if name != "title"
    """
    out = advertised_parameters_schema(_raw())
    assert out["properties"]["title"] == {
        "description": "A parameter that happens to be called title.",
        "type": "string",
    }
    assert "title" in out["required"]
    assert out["$defs"]["Chapter"]["properties"]["title"]["type"] == "string"
    assert out["$defs"]["Chapter"]["required"] == ["title", "number"]


def test_top_level_description_is_dropped_but_property_descriptions_are_kept() -> None:
    """The params docstring restates the tool description; field descriptions do not.

    Killed by: src/uclone_x/tools/schema.py :: compacted.pop("description", None)
    Becomes: compacted.get("description", None)
    """
    raw = _raw()
    assert raw["description"] == "Parameters for the sample tool."
    out = advertised_parameters_schema(raw, params_model=True)
    assert "description" not in out
    assert out["properties"]["colour"]["description"] == "Which colour."
    assert out["$defs"]["Chapter"]["description"].startswith("A nested model")


def test_optional_null_with_null_default_collapses_to_the_type() -> None:
    """`anyOf: [X, null]` with `default: null` is sent as `X`.

    Killed by: src/uclone_x/tools/schema.py :: and node["default"] is None
    Becomes: and node["default"] == "never"
    """
    raw = _raw()
    assert (
        _is_null_union(raw["properties"]["note"]) and raw["properties"]["note"]["default"] is None
    )
    out = advertised_parameters_schema(raw)
    assert out["properties"]["note"] == {"type": "string", "description": "An optional note."}
    assert out["properties"]["chapter"] == {
        "$ref": "#/$defs/Chapter",
        "description": "An optional chapter.",
    }


def test_a_wider_optional_union_loses_only_its_null_branch() -> None:
    """`anyOf: [A, B, null]` with `default: null` keeps `A` and `B`.

    Killed by: src/uclone_x/tools/schema.py :: rest["anyOf"] = others
    Becomes: rest["anyOf"] = branches
    """
    out = advertised_parameters_schema(_raw())
    assert out["properties"]["when"] == {
        "anyOf": [{"type": "string"}, {"type": "integer"}],
        "description": "A label or a number.",
    }


def test_null_is_kept_where_it_is_the_meaning() -> None:
    """A required nullable field, and a nullable one with a real default, keep `anyOf`.

    Killed by: src/uclone_x/tools/schema.py :: and node["default"] is None
    Becomes: and True
    """
    raw = _raw()
    out = advertised_parameters_schema(raw)
    assert _is_null_union(out["properties"]["maybe"])
    assert "maybe" in out["required"]
    assert _is_null_union(out["properties"]["fallback"])
    assert out["properties"]["fallback"]["default"] == 5


def test_default_suffix_is_stripped_only_when_a_default_key_carries_it() -> None:
    """A trailing `(default: ...)` goes only where the `default` key says it.

    Killed by: src/uclone_x/tools/schema.py :: out["description"] = description[: match.start()]
    Becomes: out["description"] = description
    """
    out = advertised_parameters_schema(_raw())
    assert out["properties"]["limit"] == {
        "default": 10,
        "description": "How many",
        "maximum": 100,
        "minimum": 1,
        "type": "integer",
    }
    no_default = {
        "type": "object",
        "properties": {"x": {"type": "integer", "description": "Size (default: 10)"}},
    }
    assert (
        advertised_parameters_schema(no_default)["properties"]["x"]["description"]
        == "Size (default: 10)"
    )


def _described(description: str, default: Any, type_: Any = "integer") -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"x": {"type": type_, "default": default, "description": description}},
    }


@pytest.mark.parametrize(
    ("description", "default", "type_"),
    [
        ("Where to look (defaults to the workspace root when omitted)", None, ["string", "null"]),
        ("How many (default: 3, max 10)", 3, "integer"),
        ("How many bytes (default: 45KB)", 46080, "integer"),
        ("How many (default: 5)", 3, "integer"),
        ("Which mode (default: fast)", "slow", "string"),
        ("(default: 10)", 10, "integer"),
    ],
)
def test_a_default_parenthetical_that_says_more_than_the_default_is_kept(
    description: str, default: Any, type_: Any
) -> None:
    """Only `(default: x)` with `x` equal to the node's `default` restates it (#1543).

    "(defaults to the workspace root when omitted)" is the only place a null default's
    meaning is written down; "(default: 3, max 10)" carries a limit the key does not.

    A description that is nothing but "(default: 10)" is kept whole, never emptied.

    Killed by: src/uclone_x/tools/schema.py :: if match and match.start() > 0 and _states_default(match.group(1), out["default"]):
    Becomes: if match and match.start() > 0:

    Killed by: src/uclone_x/tools/schema.py :: if match and match.start() > 0 and
    Becomes: if match and
    """
    out = advertised_parameters_schema(_described(description, default, type_))
    assert out["properties"]["x"]["description"] == description


@pytest.mark.parametrize(
    ("description", "default", "type_"),
    [
        ("How many (default: 10)", 10, "integer"),
        ("Which mode (default: 'fast')", "fast", "string"),
        ("Which mode (Default: fast)", "fast", "string"),
        ("Recurse (default: true)", True, "boolean"),
        ("Recurse (default: False)", False, "boolean"),
        ("Scale (default: 1.5)", 1.5, "number"),
    ],
)
def test_a_default_parenthetical_that_restates_the_default_is_stripped(
    description: str, default: Any, type_: Any
) -> None:
    """Quoted or not, JSON or Python spelling: the value the `default` key carries.

    Killed by: src/uclone_x/tools/schema.py :: return text in {json.dumps(default, ensure_ascii=False), str(default)}
    Becomes: return text in {json.dumps(default, ensure_ascii=False)}
    """
    out = advertised_parameters_schema(_described(description, default, type_))
    assert out["properties"]["x"]["description"] == description.rsplit(" (", 1)[0]
    assert out["properties"]["x"]["default"] == default


def test_top_level_description_is_kept_unless_the_schema_is_a_params_model() -> None:
    """An MCP server's or a hand-written schema may say what its tool description does not.

    Killed by: src/uclone_x/tools/schema.py :: if params_model:
    Becomes: if True:
    """
    schema = {
        "type": "object",
        "description": "Coordinates are WGS84; all times are UTC.",
        "properties": {"lat": {"type": "number"}},
    }
    assert advertised_parameters_schema(schema)["description"] == schema["description"]
    assert "description" not in advertised_parameters_schema(schema, params_model=True)


class _HandWrittenSchemaTool(FileReadTool):
    """A `BaseTool` that overrides `parameters_schema`: its schema is not the params model's."""

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "description": "Hand-written.", "properties": {}}


def test_advertised_tool_parameters_drops_the_description_only_for_a_params_model() -> None:
    """The tool, not the schema, says where the schema came from.

    Killed by: src/uclone_x/tools/schema.py :: from_params_model = inherited is BaseTool.parameters_schema
    Becomes: from_params_model = True
    """
    schema: dict[str, Any] = {
        "type": "object",
        "description": "Server-side note.",
        "properties": {},
    }
    mcp = MCPTool(name="m", description="d", parameters_schema=schema, client=MagicMock())
    local = LocalTool(name="l", description="d", parameters_schema=schema)
    assert advertised_tool_parameters(mcp)["description"] == "Server-side note."
    assert advertised_tool_parameters(local)["description"] == "Server-side note."
    assert advertised_tool_parameters(_HandWrittenSchemaTool())["description"] == "Hand-written."
    assert "description" not in advertised_tool_parameters(FileReadTool())


def test_constraints_survive_unchanged() -> None:
    out = advertised_parameters_schema(_raw())
    assert out["type"] == "object"
    assert out["additionalProperties"] is False
    assert out["required"] == ["title", "colour", "maybe"]
    assert out["properties"]["colour"] == {"$ref": "#/$defs/Colour", "description": "Which colour."}
    assert out["$defs"]["Colour"] == {"enum": ["red", "blue"], "type": "string"}
    assert out["$defs"]["Chapter"]["additionalProperties"] is False
    assert out["$defs"]["Chapter"]["properties"]["number"]["minimum"] == 1
    assert out["properties"]["chapters"] == {
        "items": {"$ref": "#/$defs/Chapter"},
        "type": "array",
    }


def test_default_values_are_data_and_are_not_rewritten() -> None:
    """A `default` value is data: its keys are never read as schema keywords.

    Killed by: src/uclone_x/tools/schema.py :: "allOf",
    Becomes: "allOf", "default",
    """
    schema = {
        "type": "object",
        "properties": {
            "cfg": {
                "type": "object",
                "default": {"title": "kept", "anyOf": [1, None]},
                "title": "Cfg",
            },
            "tags": {"type": "array", "items": {"type": "string"}, "default": ["a"] * 12},
        },
    }
    out = advertised_parameters_schema(schema)
    assert out["properties"]["cfg"] == {
        "type": "object",
        "default": {"title": "kept", "anyOf": [1, None]},
    }
    assert out["properties"]["tags"]["default"] == ["a"] * 12


def test_input_is_not_modified_and_output_is_deterministic() -> None:
    """The raw schema has other readers, so the pass builds a copy.

    Killed by: src/uclone_x/tools/schema.py :: out: dict[str, Any] = {}
    Becomes: out: dict[str, Any] = node
    """
    raw = _raw()
    before = copy.deepcopy(raw)
    first = json.dumps(advertised_parameters_schema(raw))
    assert raw == before
    assert json.dumps(advertised_parameters_schema(copy.deepcopy(before))) == first


# --- every registered tool -----------------------------------------------------------


def _registered_tools() -> list[ToolProtocol]:
    tools: list[ToolProtocol] = list(create_default_registry(enable_mcp=False).list_tools())
    tools += [
        c(MagicMock()) for c in (RecordMemoryFactTool, RetractMemoryFactTool, QueryMemoryFactsTool)
    ]
    tools.append(LoadSkillTool(MagicMock()))
    return tools


def test_no_registered_tool_advertises_a_title_or_an_optional_null() -> None:
    """The acceptance check of #1542, over every registered tool.

    Killed by: src/uclone_x/tools/schema.py :: and node["default"] is None
    Becomes: and node["default"] == "never"
    """
    tools = _registered_tools()
    assert len(tools) > 20
    for tool in tools:
        out = advertised_parameters_schema(tool.parameters_schema)
        for node in _schema_nodes(out):
            assert "title" not in node, (tool.name, node)
            if _is_null_union(node):
                assert node.get("default", "unset") is not None, (tool.name, node)


def test_a_collapsed_field_still_accepts_an_explicit_null() -> None:
    """The schema says 'omit to leave unset'; validation still takes a model's `null`.

    Killed by: src/uclone_x/tools/schema.py :: and node["default"] is None
    Becomes: and node["default"] == "never"
    """
    collapsed: list[tuple[str, str]] = []
    for tool in _registered_tools():
        if not isinstance(tool, BaseTool):
            continue
        model_cls = tool._resolve_params_type()  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownVariableType]
        raw = tool.parameters_schema["properties"]
        out = advertised_parameters_schema(tool.parameters_schema)["properties"]
        for name, prop in raw.items():
            if _is_null_union(prop) and not _is_null_union(out[name]):
                collapsed.append((tool.name, name))
                field = model_cls.model_fields[name]  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                assert TypeAdapter(field.annotation).validate_python(None) is None  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert ("file_read", "end_line") in collapsed
    parsed = FileReadParams.model_validate({"path": "a.txt", "start_line": None, "end_line": None})
    assert parsed.start_line is None and parsed.end_line is None


# --- the request the connectors send -------------------------------------------------


def _connector_parameters(request: LLMRequest, index: int = 0) -> dict[str, dict[str, Any]]:
    openai = OpenAIConnector(api_key="k")._build_payload(request)  # pyright: ignore[reportPrivateUsage]
    ollama = OllamaConnector()._build_payload(request)  # pyright: ignore[reportPrivateUsage]
    anthropic = AnthropicConnector(api_key="k")._build_payload(request)  # pyright: ignore[reportPrivateUsage]
    gemini = GeminiConnector(api_key="k")._build_payload(request)  # pyright: ignore[reportPrivateUsage]
    return {
        "openai": openai["tools"][index]["function"]["parameters"],
        "ollama": ollama["tools"][index]["function"]["parameters"],
        "anthropic": anthropic["tools"][index]["input_schema"],
        "gemini": gemini["tools"][0]["functionDeclarations"][index]["parametersJsonSchema"],
    }


@pytest.mark.asyncio
async def test_every_connector_sends_the_advertised_schema() -> None:
    """The agent builds its tool definitions from the advertised schema; all four connectors send it.

    A tool whose schema is not a params model's (an MCP tool) keeps its top-level
    `description` in every connector (#1543).

    Killed by: src/uclone_x/agent/tool_invoker.py :: parameters=advertised_tool_parameters(t),
    Becomes: parameters=advertised_parameters_schema(t.parameters_schema, params_model=True),
    """
    llm = MagicMock(spec=LLMProviderProtocol)
    llm.generate = AsyncMock(
        return_value=ModelResponse(
            content="ok",
            usage=TokenUsage(provider="mock", model="m", input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
            model_name="m",
            provenance=Provenance.primary("mock", "m"),
        )
    )
    finish_after_tools(llm)
    tool = FileReadTool()
    config = AgentConfig(
        agent_id="schema-1542",
        name="Schema",
        role="Tester",
        system_prompt="Test.",
        llm_config=AgentLLMConfig(model_name="m"),
    )
    mcp_schema = {
        "type": "object",
        "description": "Server-side note.",
        "properties": {"q": {"type": "string", "title": "Q"}},
    }
    mcp = MCPTool(
        name="zz_mcp_lookup",
        description="Look up.",
        parameters_schema=mcp_schema,
        client=MagicMock(),
    )
    agent = BaseAgent(config=config, llm=llm, tools=ToolRegistry([tool, mcp]))
    await agent.execute_turn("read something")

    request: LLMRequest = llm.generate.call_args[0][0]
    expected = advertised_tool_parameters(tool)
    sent = _connector_parameters(request)
    for provider, params in sent.items():
        assert params == expected, provider
        assert "title" not in params and "description" not in params, provider
        assert params["properties"]["end_line"]["type"] == "integer", provider
        assert params["required"] == ["path"], provider
    for provider, params in _connector_parameters(request, index=1).items():
        assert params == {
            "type": "object",
            "description": "Server-side note.",
            "properties": {"q": {"type": "string"}},
        }, provider
