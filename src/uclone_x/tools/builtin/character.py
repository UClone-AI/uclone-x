"""Tool for managing character sheets, visual DNA, and multi-character prompt composition (P0/P8/P9)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import PathTraversalError, PlainRefusalError, UCloneXError
from uclone_x.story.schemas import CharacterEntry
from uclone_x.tools.base import BaseTool, replace_file
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
        "Characters are stored in 'characters/<id>.yaml' in the workspace. While a story "
        "is open, characters are the story's codex characters and their 'visual' block, and "
        "'save' proposes the change to the character's 'visual' block for a person to apply."
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

    async def run(
        self,
        params: CharacterSheetParams,
        context: ToolContext,
    ) -> dict[str, Any]:
        """Execute character sheet action."""
        workspace = context.require_workspace()
        if context.story_id is not None:
            return self._run_in_story(params, context, workspace / "characters")
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

                return _compose(loaded_chars, params.scene_context)

        raise CharacterSheetError(f"Unsupported action '{params.action}'.")

    # -- while a story is open (#1556) ------------------------------------------------

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
        composed = _compose(
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


def _compose(loaded_chars: list[dict[str, Any]], scene_context: str | None) -> dict[str, Any]:
    """One prompt for the characters in `loaded_chars`, each a sheet as `get` returns it."""
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

    if scene_context:
        prompt_parts.append(scene_context.strip())

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
        "seeds": [c.get("base_seed") for c in loaded_chars if c.get("base_seed") is not None],
        "guidance": (
            "Multi-character prompt composed. To avoid attribute bleeding in Danbooru/SDXL models, "
            "ensure distinct character features and avoid conflicting color keywords. "
            f"Recommended aspect ratio: {rec_aspect}."
        ),
    }
