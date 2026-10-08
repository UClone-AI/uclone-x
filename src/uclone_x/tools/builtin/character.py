"""Tool for managing character sheets, visual DNA, and multi-character prompt composition (P0/P8/P9)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import PathTraversalError, PlainRefusalError, UCloneXError
from uclone_x.tools.base import BaseTool, replace_file
from uclone_x.tools.builtin.character_prompt import compose_character_prompt
from uclone_x.tools.models import ToolContext


class CharacterSheetError(UCloneXError):
    """Raised when character sheet management encounters validation or storage errors."""


class CharacterSheetParams(BaseModel):
    """Parameters for character sheet management and multi-character prompt composition."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["save", "get", "list", "compose"] = Field(
        ...,
        description=(
            "Action to perform: 'save' (register/update character visual DNA), "
            "'get' (retrieve character), 'list' (list all characters), "
            "'compose' (merge multiple characters into a multi-character scene prompt)."
        ),
    )
    character_id: str | None = Field(
        default=None,
        description="Unique character identifier (e.g. 'elena', 'kaito'). Required for 'save' and 'get'.",
    )
    name: str | None = Field(
        default=None,
        description="Display name of the character (e.g. 'Elena', 'Kaito').",
    )
    gender: Literal["female", "male", "other"] | None = Field(
        default=None,
        description="Character gender for subject root tag derivation ('1girl', '1boy').",
    )
    danbooru_tags: str | None = Field(
        default=None,
        description=(
            "Comma-separated Danbooru tags defining visual traits "
            "(e.g. 'silver hair, long twin braids, amber eyes, ornate silver knight armor, white cape')."
        ),
    )
    prose_description: str | None = Field(
        default=None,
        description="Rich natural language descriptive prose for FLUX or natural prose models.",
    )
    negative_tags: str | None = Field(
        default=None,
        description="Character-specific negative prompt tags to exclude unwanted attributes or errors.",
    )
    base_seed: int | None = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Optional base seed for visual consistency across scenes.",
    )
    default_style: Literal["photorealistic", "anime", "artistic", "diagram"] | None = Field(
        default=None,
        description="Default style preset for image generation. A new workspace sheet "
        "without one gets 'anime'.",
    )
    character_ids: list[str] | None = Field(
        default=None,
        description="List of character IDs to compose together. Required for 'compose'.",
    )
    scene_context: str | None = Field(
        default=None,
        description=(
            "Optional scene context or background description for 'compose' "
            "(e.g. 'standing together in ancient library, glowing particles, moonlight')."
        ),
    )


def sanitize_character_id(character_id: str) -> str:
    """Validate and sanitize character identifier to prevent path traversal."""
    cleaned = character_id.strip().lower()
    if not cleaned:
        raise CharacterSheetError("character_id cannot be empty.")
    if not re.match(r"^[a-z0-9_-]+$", cleaned):
        raise CharacterSheetError(
            f"Invalid character_id '{character_id}'. Use only alphanumeric characters, underscores, and hyphens."
        )
    return cleaned


class CharacterSheetTool(BaseTool[CharacterSheetParams]):
    """Agent tool to manage persistent character sheets and compose multi-character prompts."""

    name = "character_sheet"
    writes_files: ClassVar[bool] = True
    description = (
        "Manage persistent character visual DNA (Danbooru tags, prose descriptions, base seeds) "
        "and compose multi-character scene prompts to ensure character consistency and prevent attribute bleeding. "
        "Characters are stored in 'characters/<id>.yaml' in the workspace."
    )
    params_type = CharacterSheetParams

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=CharacterSheetParams,
        )

    def _characters_dir(self, workspace_root: Path) -> Path:
        """The workspace `characters/` folder, created if it is missing.

        A link named `characters` is left as it is, even one that leads nowhere: creating
        the folder through it failed with a raw error naming the link's full path, before
        `save` could check where the link leads (#1589 follow-up b). Reading through a link
        that leads nowhere finds no sheets; `save` checks the link and creates the folder
        it leads to (`_sheet_to_write`).
        """
        chars_dir = workspace_root / "characters"
        if not chars_dir.is_symlink():
            try:
                chars_dir.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                raise PlainRefusalError(
                    "'characters' in the workspace is a file, not a folder, so no character "
                    "sheet can be read or saved there."
                ) from None
        return chars_dir

    def _sheet_to_write(self, workspace: Path, clean_id: str) -> Path:
        """Where `save` writes `characters/<clean_id>.yaml`, checked as the file tools check.

        The name is fixed, but `characters/` or the sheet can be a link. The path is resolved
        with `resolve_write_path`, so a link leading out of the workspace is refused, and so
        is one leading into the story library: a story's characters are its codex, written
        only by the story tools (#1589). The sheet is then written as a new file
        (`replace_file`), so a name hardlinked to another file leaves that file unchanged.
        """
        rel = f"characters/{clean_id}.yaml"
        try:
            sheet = self.resolve_write_path(rel, workspace)
        except PathTraversalError:
            raise PlainRefusalError(
                f"'{rel}' leads outside the workspace through a link, so the character "
                "sheet was not saved."
            ) from None
        # `characters` can be a link to a folder in the workspace that does not exist yet.
        try:
            sheet.parent.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, NotADirectoryError):
            raise PlainRefusalError(
                "'characters' leads to a file, not a folder, so the character sheet was not saved."
            ) from None
        return sheet

    def _load_character(self, chars_dir: Path, char_id: str) -> dict[str, Any] | None:
        """Load character YAML file if it exists."""
        clean_id = sanitize_character_id(char_id)
        char_file = chars_dir / f"{clean_id}.yaml"
        if not char_file.exists():
            return None
        try:
            with open(char_file, encoding="utf-8") as f:
                raw_obj: object = yaml.safe_load(f)
                if isinstance(raw_obj, dict):
                    raw_dict = cast(dict[object, object], raw_obj)
                    return {str(k): v for k, v in raw_dict.items()}
                return None
        except Exception as exc:
            raise CharacterSheetError(
                f"Failed to read character file '{char_file}': {exc}"
            ) from exc

    def produced_artifacts(self, output: Any) -> tuple[str, ...]:
        """The sheet a `save` wrote, the one action that names a `file_path` (#2085)."""
        path: object = (
            cast(dict[str, object], output).get("file_path") if isinstance(output, dict) else None
        )
        return (path,) if isinstance(path, str) and path else ()

    async def run(
        self,
        params: CharacterSheetParams,
        context: ToolContext,
    ) -> dict[str, Any]:
        """Execute character sheet action."""
        workspace = context.require_workspace()
        chars_dir = self._characters_dir(workspace)

        match params.action:
            case "save":
                if not params.character_id:
                    raise CharacterSheetError("character_id is required for 'save' action.")
                clean_id = sanitize_character_id(params.character_id)
                char_file = self._sheet_to_write(workspace, clean_id)

                existing = self._load_character(chars_dir, clean_id) or {}
                updated: dict[str, Any] = {
                    "character_id": clean_id,
                    "name": params.name or existing.get("name") or clean_id.capitalize(),
                    "gender": params.gender or existing.get("gender") or "other",
                    "danbooru_tags": params.danbooru_tags or existing.get("danbooru_tags") or "",
                    "prose_description": params.prose_description
                    or existing.get("prose_description")
                    or "",
                    "negative_tags": params.negative_tags or existing.get("negative_tags") or "",
                    "base_seed": (
                        params.base_seed
                        if params.base_seed is not None
                        else existing.get("base_seed")
                    ),
                    "default_style": params.default_style
                    or existing.get("default_style")
                    or "anime",
                }

                sheet_text = yaml.dump(updated, sort_keys=False, allow_unicode=True)
                replace_file(char_file, sheet_text.encode("utf-8"))

                rel_path = str(char_file.relative_to(workspace.resolve()))
                return {
                    "status": "success",
                    "action": "save",
                    "character_id": clean_id,
                    "file_path": rel_path,
                    "character": updated,
                }

            case "get":
                if not params.character_id:
                    raise CharacterSheetError("character_id is required for 'get' action.")
                clean_id = sanitize_character_id(params.character_id)
                data = self._load_character(chars_dir, clean_id)
                if data is None:
                    return {
                        "status": "not_found",
                        "action": "get",
                        "character_id": clean_id,
                        "message": f"Character '{clean_id}' was not found in characters/ directory.",
                    }
                return {
                    "status": "success",
                    "action": "get",
                    "character_id": clean_id,
                    "character": data,
                }

            case "list":
                found: list[dict[str, Any]] = []
                for p in sorted(chars_dir.glob("*.yaml")):
                    try:
                        with open(p, encoding="utf-8") as f:
                            raw_obj: object = yaml.safe_load(f)
                            if isinstance(raw_obj, dict):
                                raw_dict = cast(dict[object, object], raw_obj)
                                char_dict: dict[str, Any] = {str(k): v for k, v in raw_dict.items()}
                                found.append(
                                    {
                                        "character_id": str(
                                            char_dict.get("character_id") or p.stem
                                        ),
                                        "name": str(char_dict.get("name") or p.stem),
                                        "gender": str(char_dict.get("gender") or "other"),
                                        "danbooru_tags": str(char_dict.get("danbooru_tags") or ""),
                                        "base_seed": char_dict.get("base_seed"),
                                    }
                                )
                    except Exception:
                        continue
                return {
                    "status": "success",
                    "action": "list",
                    "characters": found,
                    "count": len(found),
                }

            case "compose":
                if not params.character_ids:
                    raise CharacterSheetError("character_ids is required for 'compose' action.")
                if len(params.character_ids) < 1:
                    raise CharacterSheetError("character_ids must contain at least 1 character ID.")

                loaded_chars: list[dict[str, Any]] = []
                missing: list[str] = []
                for cid in params.character_ids:
                    char_data = self._load_character(chars_dir, cid)
                    if char_data is None:
                        missing.append(cid)
                    else:
                        loaded_chars.append(char_data)

                if missing:
                    raise CharacterSheetError(
                        f"Cannot compose scene: character(s) {missing} not found in characters/."
                    )

                return compose_character_prompt(loaded_chars, params.scene_context)

        raise CharacterSheetError(f"Unsupported action '{params.action}'.")
