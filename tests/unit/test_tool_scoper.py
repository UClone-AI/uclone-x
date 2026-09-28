"""The eval-only lexical scoper, and the canonical order of the agent's tools layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ToolDefinition
from uclone_x.tools.registry import ToolRegistry, create_default_registry
from uclone_x.tools.tool_scoper import LexicalToolScoper, ToolScopingResult


def _tools(*names: str) -> tuple[ToolDefinition, ...]:
    return tuple(
        ToolDefinition(name=name, description=f"{name} description", parameters={})
        for name in names
    )


@pytest.mark.asyncio
async def test_lexical_scoper_keeps_the_matching_tool() -> None:
    scoper = LexicalToolScoper(top_k=2)
    tools = (
        ToolDefinition(name="math_add", description="add numbers", parameters={}),
        ToolDefinition(name="math_sub", description="subtract numbers", parameters={}),
        ToolDefinition(name="system_analyze", description="analyze system logs", parameters={}),
    )

    result = await scoper.scope_tools("Can you analyze the logs?", tools)

    assert [tool.name for tool in result.selected] == ["math_add", "system_analyze"]
    assert result.scores[0] == ("system_analyze", 1.0)
    assert result.method == "lexical overlap"


@pytest.mark.asyncio
async def test_a_registry_within_budget_is_advertised_whole() -> None:
    scoper = LexicalToolScoper(top_k=5)
    tools = _tools("tool_0", "tool_1", "tool_2")

    result = await scoper.scope_tools("do something", tools)

    assert len(result.selected) == 3
    assert result.withheld == ()
    assert result.notice() == ""


@pytest.mark.asyncio
async def test_a_query_no_tool_scores_on_withholds_nothing() -> None:
    """An all-zero ranking is registry order, not a selection, so nothing is withheld.

    Killed by: src/uclone_x/tools/tool_scoper.py :: if all(score == 0 for score, _, _ in scored):
    Becomes: if False:
    """
    scoper = LexicalToolScoper(top_k=2)
    tools = _tools("alpha", "beta", "gamma", "delta")

    result = await scoper.scope_tools("zzzz", tools)

    assert result.selected == tools
    assert result.withheld == ()
    assert result.reason == "no tool scored above zero"


@pytest.mark.asyncio
async def test_withheld_tools_are_named_in_the_notice() -> None:
    """Scoping in silence is indistinguishable from a registry that never held the tool.

    Killed by: src/uclone_x/tools/tool_scoper.py :: f"{len(self.withheld)} were withheld: " + ", ".join(named),
    Becomes: "",
    """
    scoper = LexicalToolScoper(top_k=1)
    tools = (
        ToolDefinition(name="analyze", description="analyze logs", parameters={}),
        ToolDefinition(name="write_file", description="write a file", parameters={}),
        ToolDefinition(name="read_file", description="read a file", parameters={}),
    )

    result = await scoper.scope_tools("analyze this", tools)
    notice = result.notice()

    assert result.selected[0].name == "analyze"
    assert set(result.withheld) == {"write_file", "read_file"}
    assert "write_file" in notice
    assert "read_file" in notice
    assert "2 were withheld" in notice


def test_the_notice_bounds_how_many_names_it_lists() -> None:
    """The count is exact; the names are bounded so the notice is not the new bloat.

    Killed by: src/uclone_x/tools/tool_scoper.py :: named = self.withheld[:MAX_NAMED_WITHHELD]
    Becomes: named = self.withheld
    """
    result = ToolScopingResult(
        selected=_tools("kept"),
        withheld=tuple(f"withheld_{i}" for i in range(25)),
        method="lexical overlap",
    )

    notice = result.notice()

    assert "25 were withheld" in notice
    assert "withheld_19" in notice
    assert "withheld_20" not in notice
    assert "and 5 more" in notice


@pytest.mark.asyncio
async def test_equal_scores_break_on_the_name_not_the_input_order() -> None:
    """Which of two equal scores is kept does not depend on the order tools arrive in.

    Killed by: src/uclone_x/tools/tool_scoper.py :: scored.sort(key=lambda entry: (-entry[0], entry[2].name))
    Becomes: scored.sort(key=lambda entry: (-entry[0], entry[1]))
    """
    scoper = LexicalToolScoper(top_k=2)
    tools = (
        ToolDefinition(name="zeta", description="analyze logs", parameters={}),
        ToolDefinition(name="beta", description="analyze logs", parameters={}),
        ToolDefinition(name="alpha", description="analyze logs", parameters={}),
        ToolDefinition(name="omega", description="unrelated", parameters={}),
    )

    results = [
        await scoper.scope_tools("analyze the logs", order)
        for order in (tools, tuple(reversed(tools)))
    ]

    for result in results:
        assert [tool.name for tool in result.selected] == ["alpha", "beta"]
        assert set(result.withheld) == {"zeta", "omega"}


@pytest.mark.asyncio
async def test_the_selected_tools_come_back_in_name_order_not_score_order() -> None:
    """The score picks which tools; the tools layer is in name order (design §5.1).

    Killed by: src/uclone_x/tools/tool_scoper.py :: selected=_by_name(selected),
    Becomes: selected=selected,
    """
    scoper = LexicalToolScoper(top_k=2)
    tools = (
        ToolDefinition(name="alpha", description="unrelated", parameters={}),
        ToolDefinition(name="zeta_fetch", description="unrelated", parameters={}),
        ToolDefinition(name="beta", description="analyze logs", parameters={}),
    )

    # `zeta_fetch` is named in the query and outranks `beta`, which only matches a word.
    result = await scoper.scope_tools("zeta_fetch and analyze", tools)

    assert result.scores[0][0] == "zeta_fetch"
    assert [tool.name for tool in result.selected] == ["beta", "zeta_fetch"]


def test_the_advertised_tools_do_not_depend_on_registration_order(tmp_path: Path) -> None:
    """The tools layer is the same bytes however the registry was filled (design §5.1).

    A registry keeps insertion order, so an agent built from the same tools registered in a
    different order advertised them in a different order, which moves every schema in the
    request prefix and misses the provider's prefix cache.

    Killed by: src/uclone_x/agent/tool_invoker.py :: for t in sorted(self.available_tools(), key=lambda tool: tool.name)
    Becomes: for t in self.available_tools()
    """
    builtins = create_default_registry(enable_mcp=False).list_tools()

    def advertised(order: list[object]) -> list[ToolDefinition]:
        registry = ToolRegistry()
        for tool in order:
            registry.register(tool)  # type: ignore[arg-type]
        agent = BaseAgent(
            config=AgentConfig(agent_id="a", name="A", system_prompt="s", workspace_dir=tmp_path),
            llm=MockLLMConnector(default_response="stubbed"),
            tools=registry,
            context=AgentContext(session_id="sess_a", agent_id="a"),
        )
        return agent.advertised_tool_definitions()

    forward = advertised(list(builtins))
    backward = advertised(list(reversed(builtins)))

    assert len(forward) > 1
    assert forward == backward
    assert [d.name for d in forward] == sorted(d.name for d in forward)
