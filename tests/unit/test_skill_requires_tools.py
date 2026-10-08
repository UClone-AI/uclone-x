"""A skill offers itself only to a clone that can use every tool it requires (#1826).

Owner ruling 2026-09-27 (Decision B of the 2026-09-26 runtime skills review):
`SKILL.md` may declare `requires_tools`; the `[Available Approved Skills]` catalog and
`load_skill` offer the skill only when the clone's declared tool scope holds every one; a
skill with none stays available to all. The same rule is what `/api/skills` reports as
`hidden_from`, so the Settings panel can say which clones cannot use a skill and why.

Also the reviewer follow-up of the same ruling: the frontmatter ends only at a line that is
exactly `---`, so a value holding `---` no longer cuts the header in half.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.llm.models import MessageRole
from uclone_x.skills.auditor import (
    Skill,
    SkillRegistry,
    compute_skill_sha256,
    manifest_from_dict,
    parse_skill_markdown,
    serialize_skill_markdown,
)
from uclone_x.skills.models import (
    AuditVerdict,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
    missing_required_tools,
    skill_hidden_from,
)
from uclone_x.tools.builtin.media_registry import ModelProfile, PromptFamily
from uclone_x.tools.builtin.skill_loader import LoadSkillParams, LoadSkillTool
from uclone_x.tools.models import ToolContext

_FRONTMATTER = """---
name: painter
description: Paint a picture
origin: human
status: active
requires_tools:
  - generate_image
  - set_avatar
---
Call generate_image, then set_avatar.
"""


def _skill(
    name: str, *, requires: tuple[str, ...] = (), family_sections: bool = False
) -> tuple[Skill, SkillAuditReport]:
    manifest = SkillManifest(
        name=name,
        description=f"{name} steps",
        origin=SkillOrigin.HUMAN,
        status=SkillStatus.ACTIVE,
        content_sha256=f"sha_{name}",
        requires_tools=requires,
        family_sections=family_sections,
    )
    report = SkillAuditReport(
        skill_name=name,
        is_safe=True,
        recommendation=AuditVerdict.APPROVE,
        content_sha256=f"sha_{name}",
        auditor_version="0.1.0",
    )
    return Skill(manifest=manifest, instructions_markdown=f"{name} body"), report


def _registry(*skills: tuple[Skill, SkillAuditReport]) -> SkillRegistry:
    registry = SkillRegistry()
    for skill, report in skills:
        registry.register(skill, report)
    return registry


def _load(tool: LoadSkillTool, name: str) -> str:
    context = ToolContext(agent_id="a", session_id="s", workspace_root=Path("/tmp"))
    return asyncio.run(tool.run(LoadSkillParams(skill_name=name), context))


def _frontmatter_data(requires: object) -> dict[str, Any]:
    return {"name": "painter", "origin": "human", "requires_tools": requires}


# --- The manifest field ---------------------------------------------------------------


def test_requires_tools_is_read_from_the_frontmatter_and_written_back() -> None:
    """Killed by: src/uclone_x/skills/auditor.py :: requires_tools=tuple(requires_list),
    Becomes: requires_tools=(),
    """
    data, body = parse_skill_markdown(_FRONTMATTER)
    manifest = manifest_from_dict(data)

    assert manifest.requires_tools == ("generate_image", "set_avatar")
    again, _ = parse_skill_markdown(serialize_skill_markdown(manifest, body))
    assert again["requires_tools"] == ["generate_image", "set_avatar"]


def test_requires_tools_given_as_one_string_is_refused_not_split_into_letters() -> None:
    """Killed by: src/uclone_x/skills/auditor.py :: if not isinstance(requires_val, list | tuple):
    Becomes: if False:
    """
    with pytest.raises(ValueError, match="'requires_tools' must be a list of tool names"):
        manifest_from_dict(_frontmatter_data("generate_image"))


def test_a_requires_tools_entry_that_is_not_text_is_refused() -> None:
    """Killed by: src/uclone_x/skills/auditor.py :: if not isinstance(item, str):
    Becomes: if False:
    """
    with pytest.raises(ValueError, match="'requires_tools' entry 3 is not a tool name"):
        manifest_from_dict(_frontmatter_data([3]))


def test_a_requires_tools_entry_that_is_not_a_tool_name_is_refused() -> None:
    """Killed by: src/uclone_x/skills/models.py :: if not _TOOL_NAME.fullmatch(tool):
    Becomes: if False:
    """
    with pytest.raises(ValueError, match="'generate image' is not a tool name"):
        manifest_from_dict(_frontmatter_data(["generate image"]))


def test_a_tool_required_twice_is_refused() -> None:
    """Killed by: src/uclone_x/skills/models.py :: if len(set(self.requires_tools)) != len(self.requires_tools):
    Becomes: if False:
    """
    with pytest.raises(ValueError, match="more than once"):
        manifest_from_dict(_frontmatter_data(["generate_image", "generate_image"]))


def test_missing_tools_are_named_in_declared_order_and_an_empty_scope_misses_none() -> None:
    """An empty scope is no restriction, the same reading `ToolInvoker.in_tool_range` gives it.

    Killed by: src/uclone_x/skills/models.py :: if not scope:
    Becomes: if False:
    """
    manifest = _skill("painter", requires=("generate_image", "set_avatar"))[0].manifest

    assert missing_required_tools(manifest, ("file_read",)) == ("generate_image", "set_avatar")
    assert missing_required_tools(manifest, ("set_avatar", "file_read")) == ("generate_image",)
    assert missing_required_tools(manifest, ()) == ()


# --- The frontmatter boundary ---------------------------------------------------------


def test_values_holding_the_delimiter_do_not_end_the_frontmatter() -> None:
    """The header ends at a line that is exactly `---`, not at the first `---` anywhere.

    Killed by: src/uclone_x/skills/auditor.py :: (?P<yaml>.*?)^---
    Becomes: (?P<yaml>.*?)---
    """
    text = (
        "---\nname: a---b\ndescription: ends in ---\nauthor: x---\norigin: human\n"
        "tags:\n  - y---\n---\nBody --- text.\n"
    )

    data, body = parse_skill_markdown(text)

    assert data["name"] == "a---b"
    assert data["description"] == "ends in ---"
    assert data["author"] == "x---"
    assert data["tags"] == ["y---"]
    assert body == "Body --- text.\n"


def test_the_digest_leaves_out_its_own_line_after_a_value_holding_the_delimiter(
    tmp_path: Path,
) -> None:
    """The digest rule reads the same boundary as the parser, so the two cannot disagree.

    Killed by: src/uclone_x/skills/auditor.py :: end = match.end("yaml")
    Becomes: end = data.find(b"---", 3)
    """
    skill_md = tmp_path / "SKILL.md"
    header = "---\nname: a---b\norigin: human\ncontent_sha256: {}\n---\nBody.\n"
    skill_md.write_text(header.format("0" * 64), encoding="utf-8")
    first = compute_skill_sha256(tmp_path)
    skill_md.write_text(header.format("f" * 64), encoding="utf-8")

    assert compute_skill_sha256(tmp_path) == first


# --- The catalog and load_skill -------------------------------------------------------


def _agent(allowed_tools: tuple[str, ...], registry: SkillRegistry) -> BaseAgent:
    config = AgentConfig(
        agent_id="scoped",
        name="Scoped",
        system_prompt="Base.",
        llm_config=AgentLLMConfig(model_name="mock"),
        allowed_tools=allowed_tools,
    )
    return BaseAgent(config=config, skills=registry)


def _catalog(agent: BaseAgent) -> str:
    messages = agent._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]
    assert messages[0].role == MessageRole.SYSTEM
    return messages[0].content or ""


def test_the_catalog_leaves_out_a_skill_whose_tools_the_clone_cannot_use() -> None:
    """Killed by: src/uclone_x/agent/prompt_assembler.py :: and not missing_required_tools(s.manifest, scope)
    Becomes: and True
    """
    registry = _registry(_skill("painter", requires=("generate_image",)), _skill("notes"))

    scoped = _catalog(_agent(("file_read", "load_skill"), registry))
    assert "- notes:" in scoped
    assert "painter" not in scoped

    assert "- painter:" in _catalog(_agent(("generate_image", "load_skill"), registry))
    assert "- painter:" in _catalog(_agent((), registry))


def test_load_skill_refuses_a_skill_whose_tools_the_agent_cannot_use() -> None:
    """The catalog hides it; this closes the path of a name the model recalls or guesses.

    Killed by: src/uclone_x/agent/base.py :: tool_scope=lambda: self._config.allowed_tools,
    Becomes: tool_scope=None,
    """
    registry = _registry(_skill("painter", requires=("generate_image", "set_avatar")))
    agent = _agent(("set_avatar", "load_skill"), registry)
    tool = agent._tool_invoker.local_tools["load_skill"]  # pyright: ignore[reportPrivateUsage]
    assert isinstance(tool, LoadSkillTool)

    with pytest.raises(ValueError, match=r"needs the tool\(s\) generate_image, which this agent"):
        _load(tool, "painter")
    assert agent._loaded_skills == set()  # pyright: ignore[reportPrivateUsage]


def test_load_skill_returns_the_skill_when_every_tool_is_in_scope() -> None:
    """Killed by: src/uclone_x/tools/builtin/skill_loader.py :: if missing:
    Becomes: if True:
    """
    registry = _registry(_skill("painter", requires=("generate_image",)))
    tool = LoadSkillTool(registry, tool_scope=lambda: ("generate_image",))

    assert _load(tool, "painter") == "painter body"


def test_an_image_model_override_skill_is_refused_when_its_tools_are_out_of_scope() -> None:
    """The override path returns a different skill; it is held to the same rule.

    Killed by: src/uclone_x/tools/builtin/skill_loader.py :: self._require_scope(override)
    Becomes: pass
    """
    registry = _registry(
        _skill("media-character", requires=("generate_image",), family_sections=True),
        _skill("pony-rules", requires=("generate_image", "character_sheet")),
    )
    profile = ModelProfile(
        model_id="pony", display_name="pony", family=PromptFamily.DANBOORU, skill_name="pony-rules"
    )
    tool = LoadSkillTool(
        registry, profile_provider=lambda: profile, tool_scope=lambda: ("generate_image",)
    )

    with pytest.raises(ValueError, match="'pony-rules' needs the tool\\(s\\) character_sheet"):
        _load(tool, "media-character")


# --- What the Settings panel is told --------------------------------------------------


def test_hidden_from_names_each_clone_and_the_tools_it_lacks() -> None:
    """Killed by: src/uclone_x/skills/models.py :: if missing:
    Becomes: if True:
    """
    scopes = {"scout": ("web_search",), "artist": ("generate_image",), "clone": ()}

    assert skill_hidden_from(("generate_image",), scopes) == [
        {"persona": "scout", "missing_tools": ["generate_image"]}
    ]
    assert skill_hidden_from((), scopes) == []


def test_api_skills_reports_which_shipped_clones_cannot_use_a_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skill needing `generate_image` is hidden from `scout`, which lacks it, not `artist`.

    The skills are registered by hand over an empty store (the working directory is
    `tmp_path`, outside any checkout), so the test reads no skill package from disk; the
    clones are the shipped personas.

    Killed by: src/uclone_x/ui/app.py :: entry["hidden_from"] = skill_hidden_from(entry.get("requires_tools", []), scopes)
    Becomes: entry["hidden_from"] = []
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.chdir(tmp_path)
    from uclone_x.ui.app import create_ui_app

    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        eval_reports_dir=tmp_path / "evals",
        workspace_dir=tmp_path / "workspace",
    )
    with TestClient(app) as client:
        registry = app.state.session_manager.skill_registry
        for skill, report in (_skill("painter", requires=("generate_image",)), _skill("notes")):
            registry.register(skill, report)
        skills = {s["name"]: s for s in client.get("/api/skills").json()["skills"]}

    hidden = {h["persona"]: h["missing_tools"] for h in skills["painter"]["hidden_from"]}
    assert hidden["scout"] == ["generate_image"]
    assert "artist" not in hidden
    assert "clone" not in hidden
    assert skills["notes"]["hidden_from"] == []


def test_hidden_from_follows_the_scope_a_seated_clone_runs_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`hidden_from` reads the scope the seated agent resolves, not the persona's list alone.

    Seats get no operator list today, so the two agree; this sets one (as an A2A node's
    config does) on the one config every head seats a clone with, and `artist`, whose own
    list holds `generate_image`, must now be reported as lacking it (#1865).

    Killed by: src/uclone_x/ui/app.py :: scopes = {p.name: seat_tool_scope(p) for p in registry.list_personas()}
    Becomes: scopes = {p.name: p.granted_tools for p in registry.list_personas()}
    """
    import uclone_x.agent.bootstrap as bootstrap
    from uclone_x.agent.models import PersonaDefinition

    seat_config = bootstrap.agent_config_for_persona

    def with_operator_list(persona: PersonaDefinition, **kwargs: Any) -> AgentConfig:
        config = seat_config(persona, **kwargs)
        if persona.name != "artist":
            return config
        return config.model_copy(update={"allowed_tools": ("web_search",)})

    monkeypatch.setattr(bootstrap, "agent_config_for_persona", with_operator_list)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.chdir(tmp_path)
    from uclone_x.ui.app import create_ui_app

    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        eval_reports_dir=tmp_path / "evals",
        workspace_dir=tmp_path / "workspace",
    )
    with TestClient(app) as client:
        skill, report = _skill("painter", requires=("generate_image",))
        app.state.session_manager.skill_registry.register(skill, report)
        skills = {s["name"]: s for s in client.get("/api/skills").json()["skills"]}

    hidden = {h["persona"]: h["missing_tools"] for h in skills["painter"]["hidden_from"]}
    assert hidden["artist"] == ["generate_image"]


def test_the_seat_scope_is_the_scope_the_seated_agent_resolves() -> None:
    """One rule, two readers: the agent's own resolution and what Settings is told."""
    from uclone_x.agent.bootstrap import agent_config_for_persona, seat_tool_scope
    from uclone_x.agent.models import PersonaDefinition

    for tools in ((), ("generate_image",), ("web_search", "file_read")):
        persona = PersonaDefinition(name="p1865", role="r", system_prompt="s", allowed_tools=tools)
        agent = BaseAgent(
            config=agent_config_for_persona(persona, llm_config=AgentLLMConfig(model_name="mock"))
        )
        agent.define_persona(persona)
        assert agent.config.allowed_tools == seat_tool_scope(persona), tools
