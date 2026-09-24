"""Tool scoping: what is advertised, what is withheld, and whether the turn says so."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ToolDefinition
from uclone_x.tools.registry import create_default_registry
from uclone_x.tools.tool_scoper import LexicalToolScoper, ToolScopingResult


def _tools(*names: str) -> tuple[ToolDefinition, ...]:
    return tuple(
        ToolDefinition(name=name, description=f"{name} description", parameters={})
        for name in names
    )


@pytest.mark.asyncio
async def test_lexical_scoper_ranks_the_matching_tool_first() -> None:
    scoper = LexicalToolScoper(top_k=2)
    tools = (
        ToolDefinition(name="math_add", description="add numbers", parameters={}),
        ToolDefinition(name="math_sub", description="subtract numbers", parameters={}),
        ToolDefinition(name="system_analyze", description="analyze system logs", parameters={}),
    )

    result = await scoper.scope_tools("Can you analyze the logs?", tools)

    assert len(result.selected) == 2
    assert result.selected[0].name == "system_analyze"
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
async def test_equal_scores_keep_registry_order() -> None:
    """Ties break on the registry index, so scoping is deterministic across runs.

    Killed by: src/uclone_x/tools/tool_scoper.py :: scored.sort(key=lambda entry: (-entry[0], entry[1]))
    Becomes: scored.sort(key=lambda entry: (-entry[0], entry[2].name), reverse=True)
    """
    scoper = LexicalToolScoper(top_k=2)
    tools = (
        ToolDefinition(name="zeta", description="analyze logs", parameters={}),
        ToolDefinition(name="alpha", description="analyze logs", parameters={}),
        ToolDefinition(name="omega", description="unrelated", parameters={}),
    )

    result = await scoper.scope_tools("analyze the logs", tools)

    assert [tool.name for tool in result.selected] == ["zeta", "alpha"]


def test_a_withheld_tool_still_executes_when_the_model_calls_it(tmp_path: Path) -> None:
    """The notice makes a promise about withheld tools, and the promise has to hold.

    An earlier wording promised that naming a withheld tool would get it "advertised on the
    next turn". Nothing implements that: scoping runs against the *user's* next input, so a
    model that asks for a tool is answered by whatever the user types next. The notice now
    says the true thing instead — a withheld tool is withheld from the *advertisement*, not
    from the registry, and a call for it runs — which is only worth saying if it is so.

    Killed by: src/uclone_x/tools/tool_scoper.py :: "A withheld tool is registered and still executes: emit a call for it by name "
    Becomes: "A withheld tool is gone for this turn. "
    """
    scoper = LexicalToolScoper(top_k=1)
    agent = BaseAgent(
        config=AgentConfig(agent_id="a", name="A", system_prompt="s", workspace_dir=tmp_path),
        llm=MockLLMConnector(default_response="stubbed"),
        tools=create_default_registry(enable_mcp=False),
        context=AgentContext(session_id="sess_a", agent_id="a"),
        tool_scoper=scoper,
    )
    definitions = tuple(
        ToolDefinition(name=tool.name, description=tool.description, parameters={})
        for tool in agent.tools.list_tools()  # pyright: ignore[reportOptionalMemberAccess]
    )

    result = asyncio.run(scoper.scope_tools("use file_read on the notes", definitions))
    assert "file_write" in result.withheld, result.withheld
    assert "still executes" in result.notice()

    record = asyncio.run(
        agent.execute_tool_call("file_write", {"path": "note.md", "content": "seven\n"})
    )

    assert record.status == "success", record.error
    assert (tmp_path / "note.md").read_text() == "seven\n"


class _MockScoperEmbedder:
    @property
    def model_name(self) -> str:
        return "mock-scoper-embed"

    @property
    def dimensions(self) -> int:
        return 4

    def __init__(self, mapping: dict[str, tuple[float, ...]] | None = None) -> None:
        self.mapping = mapping or {}
        self.should_fail = False

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if self.should_fail:
            raise RuntimeError("Embedder connection lost")
        result: list[tuple[float, ...]] = []
        for t in texts:
            if t in self.mapping:
                result.append(self.mapping[t])
            else:
                result.append((0.01, 0.01, 0.01, 0.01))
        return tuple(result)


@pytest.mark.asyncio
async def test_semantic_tool_scoper_matches_by_similarity() -> None:
    from uclone_x.tools.tool_scoper import SemanticToolScoper

    embedder = _MockScoperEmbedder(
        mapping={
            "generate_image: generate an image": (1.0, 0.0, 0.0, 0.0),
            "file_write: write files": (0.0, 1.0, 0.0, 0.0),
            "draw a photo of a beach": (0.95, 0.05, 0.0, 0.0),
        }
    )
    scoper = SemanticToolScoper(embedder, threshold=0.5, top_k=1)
    tools = (
        ToolDefinition(name="generate_image", description="generate an image", parameters={}),
        ToolDefinition(name="file_write", description="write files", parameters={}),
    )

    result = await scoper.scope_tools("draw a photo of a beach", tools)
    assert len(result.selected) == 1
    assert result.selected[0].name == "generate_image"
    assert result.withheld == ("file_write",)
    assert result.method == "semantic similarity"


@pytest.mark.asyncio
async def test_semantic_tool_scoper_withholds_all_tools_for_unrelated_chat() -> None:
    from uclone_x.tools.tool_scoper import SemanticToolScoper

    embedder = _MockScoperEmbedder(
        mapping={
            "generate_image: generate an image": (1.0, 0.0, 0.0, 0.0),
            "file_write: write files": (0.0, 1.0, 0.0, 0.0),
            "hello how are you": (0.0, 0.0, 0.0, 1.0),
        }
    )
    scoper = SemanticToolScoper(embedder, threshold=0.5, top_k=2)
    tools = (
        ToolDefinition(name="generate_image", description="generate an image", parameters={}),
        ToolDefinition(name="file_write", description="write files", parameters={}),
    )

    result = await scoper.scope_tools("hello how are you", tools)
    assert len(result.selected) == 0
    assert set(result.withheld) == {"generate_image", "file_write"}
    assert "0 of 2 registered tools are advertised" in result.notice()


@pytest.mark.asyncio
async def test_semantic_tool_scoper_always_include_and_skill_matching() -> None:
    from uclone_x.tools.tool_scoper import SemanticToolScoper

    embedder = _MockScoperEmbedder(
        mapping={
            "generate_image: generate an image": (1.0, 0.0, 0.0, 0.0),
            "record_memory_fact: record memory": (0.0, 1.0, 0.0, 0.0),
            "photo_skill: advanced photo editing": (0.9, 0.1, 0.0, 0.0),
            "draw something": (0.95, 0.05, 0.0, 0.0),
        }
    )
    scoper = SemanticToolScoper(
        embedder,
        threshold=0.5,
        always_include=("record_memory_fact",),
        skills_provider=lambda: [("photo_skill", "advanced photo editing")],
    )
    tools = (
        ToolDefinition(name="generate_image", description="generate an image", parameters={}),
        ToolDefinition(name="record_memory_fact", description="record memory", parameters={}),
    )

    result = await scoper.scope_tools("draw something", tools)
    selected_names = {t.name for t in result.selected}
    assert "generate_image" in selected_names
    assert "record_memory_fact" in selected_names
    assert "photo_skill" in result.matched_skills
    assert "photo_skill" in result.notice()


@pytest.mark.asyncio
async def test_semantic_tool_scoper_fallback_on_embedder_failure() -> None:
    from uclone_x.tools.tool_scoper import SemanticToolScoper

    embedder = _MockScoperEmbedder()
    embedder.should_fail = True

    scoper = SemanticToolScoper(embedder, top_k=1)
    tools = (
        ToolDefinition(name="generate_image", description="generate an image", parameters={}),
        ToolDefinition(name="file_write", description="write files", parameters={}),
    )

    result = await scoper.scope_tools("generate_image now", tools)
    # Falls back to lexical overlap, which matches 'generate_image'
    assert len(result.selected) == 1
    assert result.selected[0].name == "generate_image"
    assert result.method == "lexical overlap"
