"""Extensible Image Model Registry and Prompt Family resolver for UClone-X (P0/P8/P9)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from uclone_x.skills.protocols import SkillRegistryProtocol

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default_models.yaml"
USER_CONFIG_PATH = Path.home() / ".uclone" / "media" / "models.yaml"

#: The negative a profile applies to every prompt of every domain (design §3.7). Only
#: domain-neutral quality terms belong here: anatomy terms (`bad anatomy`, `bad hands`)
#: are character rules and live in the `media-character` skill, because merged into an
#: architecture or engineering prompt they fight the domain skill behind its back.
DOMAIN_NEUTRAL_NEGATIVE = "worst quality, low quality, blurry, watermark, text"


class PromptFamily(StrEnum):
    """Prompt grammar families recognized by UClone-X image tools."""

    DANBOORU = "danbooru"  # Illustrious, Pony, Anime SDXL (comma-separated Danbooru tags)
    NATURAL_PROSE = "prose"  # FLUX.1, FLUX.2 Klein, SD 3.5 (high-density descriptive English)
    HYBRID = "hybrid"  # Standard SDXL Base, Photorealistic checkpoints
    GENERIC = "generic"  # Universal safe fallback for unknown zero-day checkpoints


class ModelProfile(BaseModel):
    """Declarative specification for an image generation model checkpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str = Field(description="Unique identifier for the profile (e.g. 'anillustrious_v4')")
    display_name: str = Field(description="Human-readable model name")
    filename: str = Field(default="", description="Target safetensors filename")
    family: PromptFamily = Field(description="Prompt grammar family guiding agent generation")
    skill_name: str | None = Field(
        default=None,
        description=(
            "Override: a P9 runtime skill (ucx-agent-skills/<name>) that replaces family x "
            "domain routing for this model. When set, `load_skill` of any family-sections "
            "domain skill returns this skill whole. Last resort: a model that differs in one "
            "domain gets a `## model: <model_id>` section there, and one that differs by a "
            "few lines gets `prompt_notes` (design §3.4)."
        ),
    )
    prompt_notes: str = Field(
        default="",
        description="A few lines appended to whatever `load_skill` returns for an image "
        "domain skill, e.g. a Pony checkpoint's score_9 quality tags (design §3.4).",
    )
    engine_type: Literal["comfyui", "diffusers", "mlx", "auto"] = Field(default="auto")
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    steps: int = Field(default=20, ge=1, le=150)
    cfg: float = Field(default=7.0, ge=1.0, le=30.0)
    sampler: str = Field(default="euler")
    scheduler: str = Field(default="normal")
    default_negative: str = Field(default="")
    suppress_negative: bool = Field(
        default=False,
        description="Whether this model suppresses negative prompts (e.g. FLUX/flow-matching).",
    )

    @field_validator("family", mode="before")
    @classmethod
    def _validate_family(cls, v: Any) -> Any:
        if isinstance(v, str):
            return PromptFamily(v)
        return v


@runtime_checkable
class ImageModelSource(Protocol):
    """What an agent needs from the image tool to route prompt skills (design §3.1-3.2).

    Declared here, in the kernel, so the agent can reach the registered `generate_image`
    tool without importing the adapter that implements it.
    """

    def active_profile(self) -> ModelProfile:
        """The profile of the checkpoint a generation would use now."""
        ...

    def bind_skill_registry(self, registry: SkillRegistryProtocol) -> None:
        """Give the tool the skill store its description lists image domains from."""
        ...


#: The quality tags the `danbooru` sections of the `media-*` skills end a prompt with. They
#: are added at call time, and only to a prompt that states no quality of its own
#: (Decision A: a skill fills in only what the person left open).
DANBOORU_QUALITY_TAGS = "masterpiece, best quality, newest, absurdres"

#: Tags that already state a quality. A prompt carrying any one of them, or a Pony-style
#: `score_*` tag, gets no quality tags added.
QUALITY_TAG_NAMES = frozenset(
    {
        "masterpiece",
        "best quality",
        "high quality",
        "amazing quality",
        "very aesthetic",
        "newest",
        "absurdres",
        "highres",
    }
)

#: What the tool result says when a negative prompt the model gave cannot be used.
NEGATIVE_NOT_USED = "The negative prompt was not used: this image model ignores negative prompts."


@dataclass(frozen=True)
class PromptFill:
    """The prompt and negative prompt an engine receives, and what was changed to get them.

    `changes` holds one plain sentence per change, for the tool result (P6: nothing is
    added or left out without saying so).
    """

    prompt: str
    negative_prompt: str
    changes: tuple[str, ...] = ()


def _tag_name(raw: str) -> str:
    """A comma-separated tag with its weight syntax removed: `(masterpiece:1.2)` -> `masterpiece`."""
    return re.sub(r":[\d.]+$", "", raw.strip(" ()[]{}\t\n")).strip().lower()


def has_quality_tag(prompt: str) -> bool:
    """Whether ``prompt`` already carries a quality tag of its own.

    Tags are separated by commas or newlines, and `_` reads as a space (`best_quality`).
    """
    for line in prompt.splitlines():
        for raw in line.split(","):
            name = _tag_name(raw).replace("_", " ")
            if name in QUALITY_TAG_NAMES or name.startswith("score "):
                return True
    return False


def names_term(prompt: str, term: str) -> bool:
    """Whether ``term`` appears in ``prompt`` as a whole word or phrase, ignoring case."""
    pattern = r"(?<![\w-])" + re.escape(term.lower()) + r"(?![\w-])"
    return re.search(pattern, prompt.lower()) is not None


def fill_prompt_defaults(prompt: str, negative_prompt: str, profile: ModelProfile) -> PromptFill:
    """Fill in only what the prompt left open, by the active model family's rules.

    Deterministic, with no model call. The rules (the owner's ruling on #1723):

    1. The prompt's own words are never removed or reordered, and it is never translated.
    2. `danbooru`: `DANBOORU_QUALITY_TAGS` are appended only when the prompt has no
       quality tag (`has_quality_tag`). Other families get nothing added to the prompt.
    3. A negative prompt the model gave is passed on unchanged.
    4. With none given, the profile's `default_negative` is used, less any term the prompt
       itself names (a diagram asking for text labels keeps its text).
    5. A model that ignores negative prompts (`prose`, `suppress_negative`) gets none, and
       a negative the model gave is reported as unused rather than dropped silently.
    """
    changes: list[str] = []
    effective_prompt = prompt.strip()
    if profile.family == PromptFamily.DANBOORU and not has_quality_tag(effective_prompt):
        effective_prompt = ", ".join(t for t in (effective_prompt, DANBOORU_QUALITY_TAGS) if t)
        changes.append(f"Added quality tags the prompt did not have: {DANBOORU_QUALITY_TAGS}.")

    given_negative = negative_prompt.strip()
    if profile.suppress_negative or profile.family == PromptFamily.NATURAL_PROSE:
        if given_negative:
            changes.append(NEGATIVE_NOT_USED)
        return PromptFill(effective_prompt, "", tuple(changes))
    if given_negative:
        return PromptFill(effective_prompt, given_negative, tuple(changes))

    defaults = [t.strip() for t in profile.default_negative.split(",") if t.strip()]
    kept = [t for t in defaults if not names_term(prompt, t)]
    effective_negative = ", ".join(kept)
    if effective_negative:
        changes.append(f"No negative prompt was given, so this one was used: {effective_negative}.")
    return PromptFill(effective_prompt, effective_negative, tuple(changes))


class ModelRegistry:
    """Cascading registry resolving model profiles from sidecars, user config, and defaults."""

    def __init__(
        self,
        default_config_path: Path | None = None,
        user_config_path: Path | None = None,
    ) -> None:
        self._default_config_path = default_config_path or DEFAULT_CONFIG_PATH
        self._user_config_path = user_config_path or USER_CONFIG_PATH
        self._profiles: dict[str, ModelProfile] = {}
        self._fallback_profile: ModelProfile = ModelProfile(
            model_id="generic_fallback",
            display_name="Universal Image Generation Model",
            filename="",
            family=PromptFamily.GENERIC,
            engine_type="auto",
            width=1024,
            height=1024,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            default_negative=DOMAIN_NEUTRAL_NEGATIVE,
        )
        self._load_registry()

    def _load_yaml(self, path: Path) -> dict[str, Any] | None:
        """Safely load and parse YAML configuration file."""
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                raw_obj: object = yaml.safe_load(f)
                if isinstance(raw_obj, dict):
                    raw_dict = cast(dict[object, object], raw_obj)
                    return {str(k): v for k, v in raw_dict.items()}
        except Exception as exc:
            logger.warning("Failed to parse YAML from %s: %s", path, exc)
        return None

    def _load_registry(self) -> None:
        """Load default baseline and user override configurations."""
        # 1. Load default shipped config
        default_data = self._load_yaml(self._default_config_path)
        if default_data:
            fallback_dict = default_data.get("default_fallback")
            if fallback_dict and isinstance(fallback_dict, dict):
                try:
                    self._fallback_profile = ModelProfile.model_validate(fallback_dict)
                except Exception as exc:
                    logger.warning(
                        "Invalid default_fallback in %s: %s", self._default_config_path, exc
                    )
            models_list: object = default_data.get("models")
            if isinstance(models_list, list):
                models_seq = cast(list[object], models_list)
                for m in models_seq:
                    if isinstance(m, dict):
                        m_obj_dict = cast(dict[object, object], m)
                        m_dict: dict[str, Any] = {str(k): v for k, v in m_obj_dict.items()}
                        try:
                            p = ModelProfile.model_validate(m_dict)
                            self._profiles[p.model_id] = p
                            if p.filename:
                                self._profiles[p.filename] = p
                        except Exception as exc:
                            logger.warning(
                                "Failed to register model profile %s: %s", str(m_dict), exc
                            )

        # 2. Layer user config overrides if available
        user_data = self._load_yaml(self._user_config_path)
        if user_data:
            user_models_list: object = user_data.get("models")
            if isinstance(user_models_list, list):
                user_seq = cast(list[object], user_models_list)
                for m in user_seq:
                    if isinstance(m, dict):
                        m_user_obj = cast(dict[object, object], m)
                        m_user_dict: dict[str, Any] = {str(k): v for k, v in m_user_obj.items()}
                        try:
                            p = ModelProfile.model_validate(m_user_dict)
                            self._profiles[p.model_id] = p
                            if p.filename:
                                self._profiles[p.filename] = p
                        except Exception as exc:
                            logger.warning(
                                "Failed to register user model profile %s: %s",
                                str(m_user_dict),
                                exc,
                            )

    def register(self, profile: ModelProfile) -> None:
        """Register or override a model profile in memory."""
        self._profiles[profile.model_id] = profile
        if profile.filename:
            self._profiles[profile.filename] = profile

    def _probe_sidecar(self, checkpoint_path: str | Path) -> ModelProfile | None:
        """Check for a co-located <name>.meta.json sidecar file."""
        p = Path(checkpoint_path).expanduser()
        sidecar_candidates = [
            p.with_suffix(".meta.json"),
            p.with_name(f"{p.stem}.json"),
        ]
        for candidate in sidecar_candidates:
            if candidate.exists():
                try:
                    with open(candidate, encoding="utf-8") as f:
                        raw_json: object = json.load(f)
                    if isinstance(raw_json, dict):
                        json_dict = cast(dict[object, object], raw_json)
                        sidecar_dict: dict[str, Any] = {str(k): v for k, v in json_dict.items()}
                        sidecar_dict.setdefault("model_id", p.stem)
                        sidecar_dict.setdefault("display_name", p.stem)
                        sidecar_dict.setdefault("filename", p.name)
                        return ModelProfile.model_validate(sidecar_dict)
                except Exception as exc:
                    logger.warning("Failed to load sidecar metadata %s: %s", candidate, exc)
        return None

    def _heuristic_detect(self, filename: str) -> ModelProfile | None:
        """Heuristically assign a family when not registered."""
        name_lower = filename.lower()
        if any(k in name_lower for k in ("pony", "illust", "anime", "waifu")):
            return ModelProfile(
                model_id=f"auto_{filename}",
                display_name=f"Auto-detected Anime ({filename})",
                filename=filename,
                family=PromptFamily.DANBOORU,
                width=832,
                height=1216,
                steps=30,
                cfg=5.5,
                sampler="dpmpp_2m",
                scheduler="karras",
                default_negative=DOMAIN_NEUTRAL_NEGATIVE,
            )
        if any(k in name_lower for k in ("flux", "klein", "sd3")):
            return ModelProfile(
                model_id=f"auto_{filename}",
                display_name=f"Auto-detected FLUX ({filename})",
                filename=filename,
                family=PromptFamily.NATURAL_PROSE,
                width=1024,
                height=1024,
                steps=4 if "schnell" in name_lower or "klein" in name_lower else 20,
                cfg=1.0 if "schnell" in name_lower or "klein" in name_lower else 3.5,
                default_negative="",
            )
        return None

    def resolve(self, checkpoint_identifier: str | None = None) -> ModelProfile:
        """Resolve model profile via cascading priority order."""
        if not checkpoint_identifier:
            return self._fallback_profile

        clean_id = Path(checkpoint_identifier).name

        # Priority 1: Sidecar JSON if identifier is an existing path
        if os.path.exists(checkpoint_identifier):
            sidecar = self._probe_sidecar(checkpoint_identifier)
            if sidecar:
                return sidecar

        # Priority 2 & 3: Look up in user/default registry by exact ID or filename
        if checkpoint_identifier in self._profiles:
            return self._profiles[checkpoint_identifier]
        if clean_id in self._profiles:
            return self._profiles[clean_id]

        # Priority 4: Heuristic filename pattern matching
        heuristic = self._heuristic_detect(clean_id)
        if heuristic:
            return heuristic

        # Priority 5: Universal generic fallback
        return self._fallback_profile
