"""Unit tests for CharacterSheetTool (P0/P8/P9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from uclone_x.sandbox.models import WorkspaceIsolation
from uclone_x.tools import ToolContext, ToolResultStatus
from uclone_x.tools.builtin.character import (
    CharacterSheetError,
    CharacterSheetTool,
    sanitize_character_id,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def tool_context(workspace: Path) -> ToolContext:
    return ToolContext(
        agent_id="test_artist",
        session_id="test_session",
        workspace_root=workspace,
        isolation=WorkspaceIsolation(),
    )


@pytest.mark.asyncio
async def test_character_sheet_save_and_get(tool_context: ToolContext, workspace: Path) -> None:
    tool = CharacterSheetTool()

    # 1. Save new character
    save_res = await tool.execute(
        params={
            "action": "save",
            "character_id": "elena",
            "name": "Elena",
            "gender": "female",
            "danbooru_tags": "silver hair, twin braids, amber eyes, knight armor, white cape",
            "negative_tags": "short hair, helmet",
            "base_seed": 424242,
            "default_style": "anime",
        },
        context=tool_context,
    )
    assert save_res.status == ToolResultStatus.SUCCESS
    assert isinstance(save_res.output, dict)
    save_output = cast(dict[str, Any], save_res.output)
    assert save_output["status"] == "success"
    assert save_output["character_id"] == "elena"

    # Verify file written to workspace
    char_file = workspace / "characters" / "elena.yaml"
    assert char_file.exists()
    with open(char_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["name"] == "Elena"
    assert data["gender"] == "female"
    assert data["base_seed"] == 424242

    # 2. Get existing character
    get_res = await tool.execute(
        params={
            "action": "get",
            "character_id": "elena",
        },
        context=tool_context,
    )
    assert get_res.status == ToolResultStatus.SUCCESS
    assert isinstance(get_res.output, dict)
    get_output = cast(dict[str, Any], get_res.output)
    char_data = cast(dict[str, Any], get_output["character"])
    assert char_data["name"] == "Elena"
    assert (
        char_data["danbooru_tags"]
        == "silver hair, twin braids, amber eyes, knight armor, white cape"
    )


@pytest.mark.asyncio
async def test_character_sheet_get_not_found(tool_context: ToolContext) -> None:
    tool = CharacterSheetTool()
    get_res = await tool.execute(
        params={
            "action": "get",
            "character_id": "nonexistent_char",
        },
        context=tool_context,
    )
    assert get_res.status == ToolResultStatus.SUCCESS
    assert isinstance(get_res.output, dict)
    get_output = cast(dict[str, Any], get_res.output)
    assert get_output["status"] == "not_found"


@pytest.mark.asyncio
async def test_character_sheet_list(tool_context: ToolContext) -> None:
    tool = CharacterSheetTool()

    # Save two characters
    await tool.execute(
        params={
            "action": "save",
            "character_id": "elena",
            "name": "Elena",
            "gender": "female",
            "danbooru_tags": "silver hair, twin braids",
        },
        context=tool_context,
    )
    await tool.execute(
        params={
            "action": "save",
            "character_id": "kaito",
            "name": "Kaito",
            "gender": "male",
            "danbooru_tags": "black hair, blue eyes",
        },
        context=tool_context,
    )

    list_res = await tool.execute(
        params={"action": "list"},
        context=tool_context,
    )
    assert list_res.status == ToolResultStatus.SUCCESS
    assert isinstance(list_res.output, dict)
    list_output = cast(dict[str, Any], list_res.output)
    assert list_output["count"] == 2
    raw_chars = cast(list[dict[str, Any]], list_output["characters"])
    c_ids = [c["character_id"] for c in raw_chars]
    assert "elena" in c_ids
    assert "kaito" in c_ids


@pytest.mark.asyncio
async def test_character_sheet_compose_multi_character(tool_context: ToolContext) -> None:
    tool = CharacterSheetTool()

    # Save female character
    await tool.execute(
        params={
            "action": "save",
            "character_id": "elena",
            "gender": "female",
            "danbooru_tags": "silver hair, twin braids, amber eyes, knight armor",
            "negative_tags": "helmet",
            "base_seed": 100,
        },
        context=tool_context,
    )

    # Save male character
    await tool.execute(
        params={
            "action": "save",
            "character_id": "kaito",
            "gender": "male",
            "danbooru_tags": "black hair, messy hair, blue eyes, dark coat",
            "negative_tags": "glasses",
            "base_seed": 200,
        },
        context=tool_context,
    )

    # Compose both
    compose_res = await tool.execute(
        params={
            "action": "compose",
            "character_ids": ["elena", "kaito"],
            "scene_context": "standing together in ancient ruins, moonlight",
        },
        context=tool_context,
    )
    assert compose_res.status == ToolResultStatus.SUCCESS
    assert isinstance(compose_res.output, dict)
    compose_output = cast(dict[str, Any], compose_res.output)
    prompt = str(compose_output["composed_danbooru_prompt"])
    assert "1girl" in prompt
    assert "1boy" in prompt
    assert "silver hair" in prompt
    assert "black hair" in prompt
    assert "ancient ruins" in prompt
    assert "masterpiece, newest, high quality" in prompt

    # Verify negative prompt includes base plus character-specific negatives
    neg = str(compose_output["composed_negative_prompt"])
    assert "helmet" in neg
    assert "glasses" in neg
    assert "bad anatomy" in neg

    assert compose_output["recommended_aspect_ratio"] == "16:9"
    assert compose_output["seeds"] == [100, 200]


@pytest.mark.asyncio
async def test_character_sheet_compose_single_character(tool_context: ToolContext) -> None:
    tool = CharacterSheetTool()
    await tool.execute(
        params={
            "action": "save",
            "character_id": "elena",
            "gender": "female",
            "danbooru_tags": "silver hair, amber eyes",
        },
        context=tool_context,
    )

    compose_res = await tool.execute(
        params={
            "action": "compose",
            "character_ids": ["elena"],
        },
        context=tool_context,
    )
    assert compose_res.status == ToolResultStatus.SUCCESS
    assert isinstance(compose_res.output, dict)
    compose_output = cast(dict[str, Any], compose_res.output)
    prompt = str(compose_output["composed_danbooru_prompt"])
    assert "1girl, solo" in prompt
    assert compose_output["recommended_aspect_ratio"] == "3:4"


def test_sanitize_character_id() -> None:
    assert sanitize_character_id("Elena_1") == "elena_1"
    assert sanitize_character_id("kaito-v2") == "kaito-v2"
    with pytest.raises(CharacterSheetError):
        sanitize_character_id("")
    with pytest.raises(CharacterSheetError):
        sanitize_character_id("../escaped")
    with pytest.raises(CharacterSheetError):
        sanitize_character_id("foo/bar")
