# pyright: reportPrivateUsage=false
"""Scene-by-scene illustration planning and prompt composition for the Novelist (`writer`).

Mirrors the Drawer (`artist`) clone's structured visual pipeline (`ShowSelfTool` in
`uclone_x.agent.self_scene` and `compose_character_prompt` in
`uclone_x.tools.builtin.character_prompt`):

1. **Temporal Visual Snapshot**: resolves each scene character's `CharacterEntry` at the
   scene's story-time position (`timeline.entry_snapshot(..., through_scene=True)`),
   folding initial `visual.tags` with all `visual.progressions` (`add_tags` / `remove_tags`)
   up through that scene and deriving a deterministic `base_seed` when none is set.
2. **Structured Visual Beat Extraction (`SCENE_VISUAL_MARK`)**: extracts a single key-frame
   visual beat (`characters`, `action`, `expression`, `location`, `lighting`, `camera`) in
   concise English visual tags from the written scene manuscript and temporal KG state.
3. **Engine-Aware Prompt Composition**: composes a 75-token-budget Danbooru prompt via
   `compose_character_prompt` (for SDXL/Animagine/Pony/Illustrious engines) or a natural
   prose prompt (for Gemini/Flux prose engines), stripping any non-ASCII tokens on Danbooru
   models so CLIP never receives Korean/CJK characters.
4. **Direct `generate_image` Execution**: invokes the registered `generate_image` tool
   directly on the current `ToolContext` and returns the workspace `relative_url`
   (`/api/artifacts/content?path=...`) to bind to `Scene.illustration` and `StartState`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from uclone_x.tools.builtin.character_prompt import compose_character_prompt
from uclone_x.tools.builtin.media_registry import ImageModelSource, PromptFamily
from uclone_x.tools.models import ToolContext
from uclone_x_story.context import CodexIndex
from uclone_x_story.kg_plan import candidate_names
from uclone_x_story.schemas import CharacterEntry, Outline, PlaceEntry, Scene, status_of
from uclone_x_story.start_state import StoryModel, _ask_json, _text, _texts
from uclone_x_story.timeline import Placement, entry_snapshot, place_scenes

if TYPE_CHECKING:
    from uclone_x.tools.protocols import ToolProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_NEGATIVE_TAGS",
    "SCENE_VISUAL_MARK",
    "SceneVisualPlan",
    "can_illustrate",
    "character_seed",
    "character_sheet_at_scene",
    "compose_scene_image_request",
    "extract_scene_visual",
    "illustrate_scene",
    "resolve_image_tool",
    "strip_non_ascii",
    "style_for_genre",
]

#: First words of the prompt that extracts a structured English visual beat for a written scene.
SCENE_VISUAL_MARK = "Plan the key-frame illustration for scene"

DEFAULT_NEGATIVE_TAGS: tuple[str, ...] = (
    "lowres",
    "bad anatomy",
    "bad hands",
    "text",
    "error",
    "missing fingers",
    "extra digit",
    "fewer digits",
    "cropped",
    "worst quality",
    "low quality",
    "jpeg artifacts",
    "signature",
    "watermark",
)

_GENRE_STYLE_MAP: tuple[tuple[str, str], ...] = (
    ("실사", "photorealistic"),
    ("포토리얼", "photorealistic"),
    ("photorealistic", "photorealistic"),
    ("live action", "photorealistic"),
    ("수채화", "artistic"),
    ("유화", "artistic"),
    ("동화", "artistic"),
    ("회화", "artistic"),
    ("painterly", "artistic"),
    ("watercolor", "artistic"),
    ("oil painting", "artistic"),
    ("artistic", "artistic"),
)


@dataclass(frozen=True)
class SceneVisualPlan:
    """Structured English visual beat extracted for a single story scene."""

    characters: tuple[str, ...] = ()
    action: str | None = None
    expression: str | None = None
    location: str | None = None
    lighting: str | None = None
    camera: str | None = None
    outfit: str | None = None


def strip_non_ascii(value: str | None) -> str | None:
    """Drop non-ASCII characters (such as Hangul/CJK) from a Danbooru tag fragment so SDXL CLIP
    receives only English tags."""
    if not value:
        return None
    ascii_only = "".join(ch for ch in value if ord(ch) < 128)
    tokens = [" ".join(part.split()) for part in ascii_only.split(",")]
    cleaned = ", ".join(t for t in tokens if t and any(c.isalnum() for c in t))
    return cleaned if cleaned else None


def character_seed(story_id: str, entry_id: str) -> int:
    """Deterministic 32-bit `base_seed` derived from `(story_id, entry_id)` when a character's
    `Visual.base_seed` was not explicitly set."""
    digest = hashlib.sha256(f"{story_id}:{entry_id}".encode()).digest()[:4]
    return int.from_bytes(digest, "big")


def style_for_genre(genre: str, style_notes: str = "") -> str:
    """Choose the default `generate_image` style preset from the story's genre and style notes."""
    combined = f"{genre} {style_notes}".casefold()
    for keyword, style in _GENRE_STYLE_MAP:
        if keyword in combined:
            return style
    return "anime"


def character_sheet_at_scene(
    entry: CharacterEntry,
    placements: Mapping[str, Placement],
    scene_id: str,
    *,
    story_id: str,
    default_style: str = "anime",
) -> dict[str, Any]:
    """Build a `compose_character_prompt`-compatible character sheet dict for `entry` at
    `scene_id`, folding `entry.visual.tags` and `entry.visual.progressions` in story-time order."""
    snap = entry_snapshot(entry, placements, scene_id, through_scene=True)
    visual = entry.visual
    raw_tags = snap.visual_tags if snap.visual_tags else (list(visual.tags) if visual else [])
    tags = [cleaned for raw in raw_tags if (cleaned := strip_non_ascii(raw))]
    neg_tags = [
        cleaned
        for raw in (visual.negative_tags if visual else ())
        if (cleaned := strip_non_ascii(raw))
    ]
    gender_raw = (
        visual.gender
        if visual and visual.gender
        else str(snap.state.get("gender") or "").strip().lower()
    )
    gender = gender_raw if gender_raw in ("female", "male", "other") else None
    seed = (
        visual.base_seed
        if visual is not None and visual.base_seed is not None
        else character_seed(story_id, entry.id)
    )
    style = (
        visual.default_style
        if visual is not None and visual.default_style is not None
        else default_style
    )
    prose = (visual.prose if visual and visual.prose else "") or entry.profile
    return {
        "id": entry.id,
        "character_id": entry.id,
        "name": entry.name,
        "display_name": entry.name,
        "gender": gender,
        "danbooru_tags": ", ".join(tags),
        "core_tags": tags,
        "negative_tags": ", ".join(neg_tags),
        "base_seed": seed,
        "default_style": style,
        "prose": prose,
    }


def resolve_image_tool(
    explicit_tool: ToolProtocol | None,
    context: ToolContext,
) -> ToolProtocol | None:
    """Return `explicit_tool` when injected, or look up `generate_image` on `context.agent_delegate`."""
    if explicit_tool is not None:
        return explicit_tool
    delegate = context.agent_delegate
    if delegate is None:
        return None
    for attr in ("_tools", "tools"):
        registry = getattr(delegate, attr, None)
        getter = getattr(registry, "get", None)
        if callable(getter):
            found = getter("generate_image")
            if found is not None and (
                isinstance(found, ImageModelSource)
                or getattr(found, "_dispatcher", None) is not None
            ):
                return cast("ToolProtocol", found)
    return None


async def can_illustrate(
    image_tool: ToolProtocol | None,
    context: ToolContext,
) -> bool:
    """Whether `image_tool` is present and backed by a ready image engine (or is a test double)."""
    if image_tool is None:
        return False
    dispatcher = getattr(image_tool, "_dispatcher", None)
    if dispatcher is None:
        return True
    try:
        choice = dispatcher.engine_choice(context.image_model)
        if getattr(choice, "refusal", None) is not None:
            return False
        if await dispatcher._draws_with_gemini_now(choice):
            return True
        return bool(await dispatcher._any_local_ready(choice))
    except Exception:  # noqa: BLE001
        logger.debug("scene_illustration: image engine availability check failed", exc_info=True)
        return False


async def _resolve_prompt_family(
    image_tool: ToolProtocol | None,
    context: ToolContext,
) -> PromptFamily:
    """Determine whether the active image engine expects Danbooru tags or natural prose."""
    if isinstance(image_tool, ImageModelSource):
        try:
            return image_tool.active_profile(context.image_model).family
        except Exception:  # noqa: BLE001
            pass
    dispatcher = getattr(image_tool, "_dispatcher", None)
    if dispatcher is not None:
        try:
            choice = dispatcher.engine_choice(context.image_model)
            if await dispatcher._draws_with_gemini_now(choice):
                return PromptFamily.NATURAL_PROSE
            profile = dispatcher.get_active_profile(context.image_model)
            if getattr(profile, "family", None) == PromptFamily.NATURAL_PROSE:
                return PromptFamily.NATURAL_PROSE
            return PromptFamily.DANBOORU
        except Exception:  # noqa: BLE001
            pass
    if context.image_model == "gemini":
        return PromptFamily.NATURAL_PROSE
    return PromptFamily.DANBOORU


def _resolve_scene_characters(
    plan_names: Sequence[str],
    scene: Scene,
    codex: CodexIndex,
    cast_lines: Sequence[Mapping[str, Any]],
    placements: Mapping[str, Placement],
) -> list[CharacterEntry]:
    """Resolve up to 2 focal living characters in `scene` from `plan_names` and `scene.characters`."""
    char_by_id: dict[str, CharacterEntry] = {}
    name_to_id: dict[str, str] = {}
    for item in codex.items:
        if item.kind == "characters" and isinstance(item.entry, CharacterEntry):
            char_by_id[item.entry.id] = item.entry
            name_to_id[item.entry.id.casefold()] = item.entry.id
            name_to_id[item.entry.name.casefold()] = item.entry.id
            for alias in item.entry.aliases:
                if alias.strip():
                    name_to_id[alias.strip().casefold()] = item.entry.id

    for line in cast_lines:
        if line.get("kind") != "characters":
            continue
        cid = _text(line.get("id"))
        cname = _text(line.get("name"))
        if not cid or not cname:
            continue
        if cid not in char_by_id:
            char_by_id[cid] = CharacterEntry(
                id=cid,
                name=cname,
                aliases=_texts(line.get("aliases")),
                profile=_text(line.get("profile")),
                state={"status": status_of(line.get("status"))},
            )
        name_to_id.setdefault(cid.casefold(), cid)
        name_to_id.setdefault(cname.casefold(), cid)
        for alias in _texts(line.get("aliases")):
            name_to_id.setdefault(alias.casefold(), cid)

    ordered_ids: list[str] = []
    for raw_name in (*plan_names, *scene.characters):
        for cand in candidate_names(raw_name):
            matched_id = name_to_id.get(cand.casefold())
            if matched_id and matched_id not in ordered_ids:
                ordered_ids.append(matched_id)
                break

    if not ordered_ids:
        for cid in char_by_id:
            ordered_ids.append(cid)
            if len(ordered_ids) >= 2:
                break

    living: list[CharacterEntry] = []
    for cid in ordered_ids:
        entry = char_by_id.get(cid)
        if entry is None:
            continue
        if scene.id in placements:
            snap = entry_snapshot(entry, placements, scene.id, through_scene=False)
            if status_of(snap.state.get("status")) == "dead":
                continue
        living.append(entry)
        if len(living) >= 2:
            break
    return living


def _scene_place_descriptions(scene: Scene, codex: CodexIndex) -> list[str]:
    places: list[str] = []
    for item in codex.items:
        if item.kind == "places" and isinstance(item.entry, PlaceEntry):
            if item.entry.id in scene.places or not scene.places:
                desc = (
                    f"{item.entry.name}: {item.entry.profile}"
                    if item.entry.profile
                    else item.entry.name
                )
                places.append(desc)
    return places[:2]


async def extract_scene_visual(
    model: StoryModel,
    *,
    scene: Scene,
    text: str,
    characters: Sequence[CharacterEntry],
    places: Sequence[str],
    genre: str,
) -> SceneVisualPlan:
    """Ask `model` for a structured English visual key-frame beat (`SceneVisualPlan`) for `scene`."""
    char_lines = [
        f"- {c.name} (id={c.id}, gender={c.visual.gender if c.visual and c.visual.gender else c.state.get('gender', 'unknown')}): "
        f"{(c.visual.prose if c.visual and c.visual.prose else '') or c.profile}"
        + (f" [tags: {', '.join(c.visual.tags)}]" if c.visual and c.visual.tags else "")
        for c in characters
    ]
    place_block = "\n".join(f"- {p}" for p in places) if places else f"- {scene.title}"
    prompt = (
        f"{SCENE_VISUAL_MARK} [{scene.id}] '{scene.title}'.\n"
        f"Genre: {genre}\n"
        f"Scene summary: {scene.summary}\n"
        f"Characters in scene:\n"
        + ("\n".join(char_lines) if char_lines else "- None specified")
        + f"\nSetting / places:\n{place_block}\n\n"
        f"Scene prose excerpt:\n{text[:1800]}\n\n"
        "Select ONE vivid visual key-frame moment from this scene for an illustration.\n"
        "Write ALL visual descriptor fields (`action`, `expression`, `location`, `lighting`, "
        "`camera`, `outfit`) in concise ENGLISH Danbooru-style visual tags (ASCII only, no "
        "Korean/CJK words in tag fields). Use `characters` to list the 1 or 2 focal character "
        "names or ids visible in the frame.\n"
        'Reply as JSON: {"characters": [1 or 2 character names], '
        '"action": "2-4 English visual tags for pose and action, e.g. holding lantern, walking on wet cobblestones", '
        '"expression": "1-2 English tags for facial expression, e.g. focused expression, parted lips", '
        '"location": "2-4 English setting/background tags, e.g. lighthouse interior, spiral staircase, rain outside window", '
        '"lighting": "1-3 English lighting/atmosphere tags, e.g. warm lantern light, dramatic shadows, night", '
        '"camera": "1-2 English framing tags, e.g. medium shot, cinematic angle", '
        '"outfit": "optional English outfit tags if the scene specifies attire"}'
    )
    found = await _ask_json(model, prompt, temperature=0.3, max_tokens=600)
    if not found:
        return SceneVisualPlan(
            characters=tuple(c.id for c in characters[:2]),
            action="dramatic pose, story scene",
            expression="focused expression",
            location="detailed background, atmospheric setting",
            lighting="cinematic lighting",
            camera="medium shot",
        )
    return SceneVisualPlan(
        characters=tuple(_texts(found.get("characters"))[:2])
        or tuple(c.id for c in characters[:2]),
        action=strip_non_ascii(_text(found.get("action"))) or "dramatic pose",
        expression=strip_non_ascii(_text(found.get("expression"))) or "focused expression",
        location=strip_non_ascii(_text(found.get("location"))) or "detailed background",
        lighting=strip_non_ascii(_text(found.get("lighting"))) or "cinematic lighting",
        camera=strip_non_ascii(_text(found.get("camera"))) or "medium shot",
        outfit=strip_non_ascii(_text(found.get("outfit"))),
    )


def compose_scene_image_request(
    plan: SceneVisualPlan,
    sheets: Sequence[Mapping[str, Any]],
    *,
    scene: Scene,
    genre: str,
    style_notes: str = "",
    prompt_family: PromptFamily = PromptFamily.DANBOORU,
) -> dict[str, Any]:
    """Deterministically compose the `generate_image` tool request dict from `plan` and the
    time-aware character `sheets`."""
    style = (
        str(sheets[0].get("default_style"))
        if sheets and sheets[0].get("default_style")
        else style_for_genre(genre, style_notes)
    )
    seed: int | None = (
        int(sheets[0]["base_seed"])
        if sheets and isinstance(sheets[0].get("base_seed"), int)
        else None
    )

    if prompt_family == PromptFamily.NATURAL_PROSE:
        char_descriptions: list[str] = []
        for sheet in sheets[:2]:
            name = str(
                sheet.get("display_name") or sheet.get("name") or sheet.get("character_id") or ""
            )
            tags = str(sheet.get("danbooru_tags") or "").strip() or ", ".join(
                cast("Sequence[str]", sheet.get("core_tags") or ())
            )
            prose = str(sheet.get("prose") or "").strip()
            desc = tags or prose or name
            if name and desc and name not in desc:
                char_descriptions.append(f"{name} ({desc})")
            elif desc:
                char_descriptions.append(desc)
        subject_clause = (
            f"Featuring {'; '.join(char_descriptions)}."
            if char_descriptions
            else "Atmospheric story scene."
        )
        parts = [
            f"Scene illustration for '{scene.title}'.",
            subject_clause,
            f"Action: {plan.action}." if plan.action else "",
            f"Expression: {plan.expression}." if plan.expression else "",
            f"Outfit: {plan.outfit}." if plan.outfit else "",
            f"Setting: {plan.location}." if plan.location else "",
            f"Lighting: {plan.lighting}." if plan.lighting else "",
            f"Framing: {plan.camera}." if plan.camera else "",
        ]
        prompt_text = " ".join(p for p in parts if p)
        prose_req: dict[str, Any] = {
            "prompt": prompt_text,
            "aspect_ratio": "16:9",
            "style": style,
        }
        if seed is not None:
            prose_req["seed_override"] = seed
        return prose_req

    loaded_chars: list[dict[str, Any]] = []
    for raw_sheet in sheets[:2]:
        sheet_copy = dict(raw_sheet)
        if not sheet_copy.get("danbooru_tags") and sheet_copy.get("core_tags"):
            sheet_copy["danbooru_tags"] = ", ".join(cast("Sequence[str]", sheet_copy["core_tags"]))
        if isinstance(sheet_copy.get("negative_tags"), list):
            sheet_copy["negative_tags"] = ", ".join(
                cast("Sequence[str]", sheet_copy["negative_tags"])
            )
        loaded_chars.append(sheet_copy)

    scene_parts = [
        p
        for p in (
            plan.outfit,
            plan.action,
            plan.expression,
            plan.location,
            plan.lighting,
            plan.camera,
        )
        if p
    ]
    scene_context = ", ".join(scene_parts)
    if loaded_chars:
        composed = compose_character_prompt(loaded_chars, scene_context)
        prompt_str = str(composed["composed_danbooru_prompt"])
        neg_str = str(composed.get("composed_negative_prompt") or ", ".join(DEFAULT_NEGATIVE_TAGS))
    else:
        prompt_str = ", ".join(
            p
            for p in (
                "no humans, scenery",
                scene_context,
                "masterpiece, newest, high quality, cinematic lighting",
            )
            if p
        )
        neg_str = ", ".join(DEFAULT_NEGATIVE_TAGS)

    request: dict[str, Any] = {
        "prompt": prompt_str,
        "negative_prompt": neg_str,
        "aspect_ratio": "16:9",
        "style": style,
    }
    if seed is not None:
        request["seed_override"] = seed
    return request


async def illustrate_scene(
    *,
    model: StoryModel,
    image_tool: ToolProtocol | None,
    context: ToolContext,
    story_id: str,
    outline: Outline,
    codex: CodexIndex,
    cast_lines: Sequence[Mapping[str, Any]],
    scene_id: str,
    text: str,
    genre: str,
    style_notes: str = "",
) -> str | None:
    """Generate a key-frame illustration for `scene_id` when `image_tool` is available and
    ready. Returns the workspace `relative_url` (`/api/artifacts/content?path=...`), or `None`
    when no image engine is ready or generation fails."""
    resolved_tool = resolve_image_tool(image_tool, context)
    if not await can_illustrate(resolved_tool, context) or resolved_tool is None:
        return None
    located = outline.find(scene_id)
    if located is None:
        return None
    _, scene = located
    placements = place_scenes(outline)
    initial_chars = _resolve_scene_characters((), scene, codex, cast_lines, placements)
    places = _scene_place_descriptions(scene, codex)
    plan = await extract_scene_visual(
        model,
        scene=scene,
        text=text,
        characters=initial_chars,
        places=places,
        genre=genre,
    )
    focal_chars = _resolve_scene_characters(plan.characters, scene, codex, cast_lines, placements)
    default_style = style_for_genre(genre, style_notes)
    sheets = [
        character_sheet_at_scene(
            c,
            placements,
            scene.id,
            story_id=story_id,
            default_style=default_style,
        )
        for c in focal_chars
    ]
    family = await _resolve_prompt_family(resolved_tool, context)
    request = compose_scene_image_request(
        plan,
        sheets,
        scene=scene,
        genre=genre,
        style_notes=style_notes,
        prompt_family=family,
    )
    try:
        result = await resolved_tool.execute(request, context)
    except Exception:  # noqa: BLE001
        logger.warning(
            "scene_illustration: image tool execution raised for %s", scene_id, exc_info=True
        )
        return None
    if not result.success or not isinstance(result.output, Mapping):
        return None
    output = cast("Mapping[str, Any]", result.output)
    url = output.get("relative_url") or output.get("path")
    return str(url).strip() if isinstance(url, str) and url.strip() else None
