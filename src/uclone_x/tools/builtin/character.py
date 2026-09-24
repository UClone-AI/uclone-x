"""Tool for managing character sheets, visual DNA, and multi-character prompt composition (P0/P8/P9)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import UCloneXError
from uclone_x.tools.base import BaseTool
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
        default="anime",
        description="Default style preset for image generation.",
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
        """Resolve and ensure characters directory within workspace."""
        chars_dir = workspace_root / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)
        return chars_dir

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
                char_file = chars_dir / f"{clean_id}.yaml"

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

                with open(char_file, "w", encoding="utf-8") as f:
                    yaml.dump(updated, f, sort_keys=False, allow_unicode=True)

                rel_path = str(char_file.relative_to(workspace))
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

                # Derive Root Subject Tag
                females = sum(1 for c in loaded_chars if c.get("gender") == "female")
                males = sum(1 for c in loaded_chars if c.get("gender") == "male")
                others = len(loaded_chars) - females - males

                root_tags: list[str] = []
                if females > 0 and males == 0 and others == 0:
                    root_tags.append("1girl" if females == 1 else f"{females}girls")
                elif males > 0 and females == 0 and others == 0:
                    root_tags.append("1boy" if males == 1 else f"{males}boys")
                elif females > 0 and males > 0:
                    f_tag = "1girl" if females == 1 else f"{females}girls"
                    m_tag = "1boy" if males == 1 else f"{males}boys"
                    root_tags.extend([f_tag, m_tag])
                else:
                    root_tags.append(f"{len(loaded_chars)}people")

                if len(loaded_chars) == 1:
                    root_tags.append("solo")

                # Compose Danbooru tags per character with attribute segregation
                char_danbooru_blocks: list[str] = []
                for c in loaded_chars:
                    ctags = c.get("danbooru_tags", "").strip().rstrip(",")
                    if ctags:
                        char_danbooru_blocks.append(f"{ctags}")

                # Combine prompt
                prompt_parts: list[str] = [", ".join(root_tags)]
                if char_danbooru_blocks:
                    prompt_parts.append(", ".join(char_danbooru_blocks))

                if params.scene_context:
                    prompt_parts.append(params.scene_context.strip())

                # Quality tags
                prompt_parts.append("masterpiece, newest, high quality, cinematic lighting")
                composed_danbooru_prompt = ", ".join(p for p in prompt_parts if p)

                # Collect negative tags
                neg_set: set[str] = set()
                for c in loaded_chars:
                    cneg = c.get("negative_tags", "")
                    if cneg:
                        for t in cneg.split(","):
                            cleaned_t = t.strip()
                            if cleaned_t:
                                neg_set.add(cleaned_t)

                base_neg = "worst quality, bad anatomy, deformed, bad hands, animal, blurry, text, watermark"
                for t in base_neg.split(","):
                    neg_set.add(t.strip())

                composed_negative = ", ".join(sorted(neg_set))

                # Recommendation for aspect ratio
                rec_aspect = "16:9" if len(loaded_chars) >= 2 else "3:4"

                return {
                    "status": "success",
                    "action": "compose",
                    "characters": loaded_chars,
                    "composed_danbooru_prompt": composed_danbooru_prompt,
                    "composed_negative_prompt": composed_negative,
                    "recommended_aspect_ratio": rec_aspect,
                    "seeds": [
                        c.get("base_seed") for c in loaded_chars if c.get("base_seed") is not None
                    ],
                    "guidance": (
                        "Multi-character prompt composed. To avoid attribute bleeding in Danbooru/SDXL models, "
                        "ensure distinct character features and avoid conflicting color keywords. "
                        f"Recommended aspect ratio: {rec_aspect}."
                    ),
                }

        raise CharacterSheetError(f"Unsupported action '{params.action}'.")
