"""`character_sheet` while a story is open: the story's codex characters (#1556, #2205).

The core's `character_sheet` keeps character sheets in the workspace's `characters/`
folder. While a story is open, a story's characters are its codex entries, and what they
look like is each entry's `visual` block; this is that tool with the story's behaviour.
The story extension registers it in place of the core's (`story/ucx_extension.py`), so a
clone sees one `character_sheet` either way. With no story open it is the core's tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from uclone_x.errors import PlainRefusalError
from uclone_x.story.schemas import CharacterEntry
from uclone_x.tools.builtin.character import (
    CharacterSheetError,
    CharacterSheetParams,
    CharacterSheetTool,
    sanitize_character_id,
)
from uclone_x.tools.builtin.character_prompt import compose_character_prompt
from uclone_x.tools.models import ToolContext

__all__ = ["StoryCharacterSheetTool"]


class StoryCharacterSheetTool(CharacterSheetTool):
    """`character_sheet`, reading and proposing to the open story's codex characters."""

    description = (
        "Manage persistent character visual DNA (Danbooru tags, prose descriptions, base seeds) "
        "and compose multi-character scene prompts to ensure character consistency and prevent attribute bleeding. "
        "Characters are stored in 'characters/<id>.yaml' in the workspace. While a story "
        "is open, characters are the story's codex characters and their 'visual' block, and "
        "'save' proposes the change to the character's 'visual' block for a person to apply."
    )

    async def run(
        self,
        params: CharacterSheetParams,
        context: ToolContext,
    ) -> dict[str, Any]:
        """The story's characters while one is open; otherwise the workspace sheets."""
        if context.story_id is not None:
            workspace = context.require_workspace()
            return self._run_in_story(params, context, workspace / "characters")
        return await super().run(params, context)

    def _run_in_story(
        self, params: CharacterSheetParams, context: ToolContext, legacy_dir: Path
    ) -> dict[str, Any]:
        """The same actions over the open story's codex characters.

        A story's characters are its codex entries, and what they look like is each entry's
        `visual` block. Sheets in the workspace's `characters/` folder belong to no story:
        they are shown read-only, with how to bring one into the story. `save` writes no
        sheet: it proposes the change to the entry's `visual` block, which a person applies
        with story_codex 'apply' (#1557).
        """
        from uclone_x.story.work import StoryWork  # an adapter: loaded only with a story

        if params.action == "save":
            return _propose_visual(params, context)
        codex = StoryWork.open_in(context).codex()
        story_chars = {
            item.entry.id: item.entry
            for item in codex.items
            if isinstance(item.entry, CharacterEntry)
        }
        unreadable = [
            {"file": u.file, "reason": u.reason}
            for u in codex.unreadable
            if u.file.startswith("codex/characters/")
        ]

        if params.action == "list":
            result: dict[str, Any] = {
                "status": "success",
                "action": "list",
                "source": "story codex",
                "characters": [
                    {
                        k: v
                        for k, v in _sheet_of(entry).items()
                        if k in ("character_id", "name", "gender", "danbooru_tags", "base_seed")
                    }
                    for entry in story_chars.values()
                ],
                "count": len(story_chars),
            }
            legacy = (
                sorted(p.stem for p in legacy_dir.glob("*.yaml")) if legacy_dir.is_dir() else []
            )
            if legacy:
                result["workspace_sheets_read_only"] = legacy
                result["migration_hint"] = _MIGRATION_HINT
            if unreadable:
                result["unreadable_files"] = unreadable
            return result

        if params.action == "get":
            if not params.character_id or not params.character_id.strip():
                raise PlainRefusalError("Say which character to get, in 'character_id'.")
            char_id = params.character_id.strip()
            entry = story_chars.get(char_id)
            if entry is not None:
                return {
                    "status": "success",
                    "action": "get",
                    "source": "story codex",
                    "character_id": char_id,
                    "character": _sheet_of(entry),
                    **({} if entry.visual else {"note": _NO_VISUAL}),
                }
            legacy_sheet, legacy_problem = self._legacy_sheet(legacy_dir, char_id)
            if legacy_sheet is not None:
                return {
                    "status": "not_in_story",
                    "action": "get",
                    "character_id": char_id,
                    "workspace_sheet_read_only": legacy_sheet,
                    "migration_hint": _MIGRATION_HINT,
                }
            result = {
                "status": "not_found",
                "action": "get",
                "character_id": char_id,
                "message": f"The story's codex has no character '{char_id}'.",
            }
            if legacy_problem is not None:
                result["workspace_sheet_unreadable"] = legacy_problem
            if unreadable:
                result["unreadable_files"] = unreadable
            return result

        # compose
        if not params.character_ids:
            raise PlainRefusalError("Say which characters to compose, in 'character_ids'.")
        missing = [cid for cid in params.character_ids if cid not in story_chars]
        if missing:
            in_workspace = [
                cid for cid in missing if self._legacy_sheet(legacy_dir, cid) != (None, None)
            ]
            message = (
                f"The story's codex has no character {', '.join(repr(m) for m in missing)}, "
                "so no prompt was composed."
            )
            if in_workspace:
                message += " " + _MIGRATION_HINT
            raise PlainRefusalError(message)
        composed = compose_character_prompt(
            [_sheet_of(story_chars[cid]) for cid in params.character_ids], params.scene_context
        )
        notes = _compose_notes([story_chars[cid] for cid in params.character_ids])
        if notes:
            composed["notes"] = notes
        return composed

    @staticmethod
    def _legacy_sheet(
        legacy_dir: Path, char_id: str
    ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        """A workspace sheet named `char_id`, read only, or why it could not be read.

        `(None, None)` when there is no such sheet -- including for an id no sheet can have.
        A sheet that is there and does not read is reported by its file and a plain reason,
        never taken for a missing one (P6).
        """
        try:
            clean_id = sanitize_character_id(char_id)
        except CharacterSheetError:
            return None, None  # not a name a workspace sheet can have, so there is none
        sheet = legacy_dir / f"{clean_id}.yaml"
        if not sheet.is_file():
            return None, None
        where = f"characters/{clean_id}.yaml"
        try:
            raw: object = yaml.safe_load(sheet.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            return None, {
                "file": where,
                "reason": "It could not be read as YAML text, so it was not shown.",
            }
        if not isinstance(raw, dict):
            return None, {
                "file": where,
                "reason": "It is not a set of named fields, so it was not shown.",
            }
        return {str(k): v for k, v in cast(dict[object, object], raw).items()}, None


_MIGRATION_HINT = (
    "Sheets in the workspace's characters/ folder belong to no story and are read-only while "
    "one is open. To use one here, copy its tags into the 'visual' block of a codex entry in "
    "the story: codex/characters/<id>.yaml."
)
_NO_VISUAL = (
    "This character has no 'visual' block in the story's codex yet, so it has no tags to draw from."
)


def _sheet_of(entry: CharacterEntry) -> dict[str, Any]:
    """A codex character as the sheet `get` returns, from its `visual` block.

    A field the codex does not give is `None`, not a default: the sheet says what the
    story says and nothing more (#1576).
    """
    visual = entry.visual
    return {
        "character_id": entry.id,
        "name": entry.name,
        "gender": visual.gender if visual else None,
        "danbooru_tags": ", ".join(visual.tags) if visual else "",
        "prose_description": (visual.prose if visual else None) or "",
        "negative_tags": ", ".join(visual.negative_tags) if visual else "",
        "base_seed": visual.base_seed if visual else None,
        "default_style": visual.default_style if visual else None,
    }


def _compose_notes(entries: list[CharacterEntry]) -> list[str]:
    """What a composed prompt lacks because the codex does not say it, one line each."""
    notes: list[str] = []
    for entry in entries:
        if entry.visual is None:
            notes.append(
                f"'{entry.id}' has no 'visual' block in the story's codex, so no tags of its "
                "own went into the prompt."
            )
        elif entry.visual.gender is None:
            notes.append(
                f"'{entry.id}' has no gender in its 'visual' block, so the prompt counts it "
                "as a person rather than a girl or a boy."
            )
    return notes


def _split_tags(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [t.strip() for t in text.split(",") if t.strip()]


def _propose_visual(params: CharacterSheetParams, context: ToolContext) -> dict[str, Any]:
    """`save` while a story is open: a proposal to change a codex character's looks."""
    from uclone_x.story.tools import propose_visual  # an adapter: loaded only with a story

    if not params.character_id or not params.character_id.strip():
        raise PlainRefusalError("Say which character to change, in 'character_id'.")
    visual: dict[str, Any] = {
        "tags": _split_tags(params.danbooru_tags),
        "prose": params.prose_description,
        "negative_tags": _split_tags(params.negative_tags),
        "base_seed": params.base_seed,
        "gender": params.gender,
        "default_style": params.default_style,
    }
    result = propose_visual(
        context,
        params.character_id.strip(),
        {k: v for k, v in visual.items() if v is not None},
    )
    if params.name is not None:
        result.setdefault("notes", []).append(
            "The name is not part of how a character looks, so it was not proposed. A "
            "character's name is the 'name' of its codex entry."
        )
    return {"status": "proposed", "action": "save", **result}
