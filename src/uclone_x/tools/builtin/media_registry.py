"""Extensible Image Model Registry and Prompt Family resolver for UClone-X (P0/P8/P9)."""

from __future__ import annotations

import json
import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default_models.yaml"
USER_CONFIG_PATH = Path.home() / ".uclone" / "media" / "models.yaml"


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
    skill_name: str = Field(description="Associated P9 skill directory name under skills/")
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


def optimize_prompts(
    prompt: str,
    negative_prompt: str,
    profile: ModelProfile,
) -> tuple[str, str]:
    """Optimize positive and negative prompts according to the model profile and grammar family."""
    cleaned_neg = negative_prompt.strip()

    # 1. Negative prompt handling
    if profile.suppress_negative or profile.family == PromptFamily.NATURAL_PROSE:
        # Flow-matching and prose models (e.g. FLUX) degrade with negative prompts
        effective_negative = ""
    elif not cleaned_neg:
        effective_negative = profile.default_negative
    elif profile.family == PromptFamily.DANBOORU and profile.default_negative:
        # Merge user-supplied negative tags with default safety tags, avoiding duplicates
        user_tags = [t.strip() for t in cleaned_neg.split(",") if t.strip()]
        default_tags = [t.strip() for t in profile.default_negative.split(",") if t.strip()]
        user_tag_set = {t.lower() for t in user_tags}
        merged_tags = list(user_tags)
        for dt in default_tags:
            if dt.lower() not in user_tag_set:
                merged_tags.append(dt)
        effective_negative = ", ".join(merged_tags)
    else:
        effective_negative = cleaned_neg

    # 2. Positive prompt handling
    effective_prompt = prompt.strip()
    return effective_prompt, effective_negative


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
            skill_name="media-prompt-generic",
            engine_type="auto",
            width=1024,
            height=1024,
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            default_negative="worst quality, blurry, deformed, bad anatomy, bad hands, text",
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
                skill_name="media-prompt-danbooru",
                width=832,
                height=1216,
                steps=30,
                cfg=5.5,
                sampler="dpmpp_2m",
                scheduler="karras",
                default_negative="worst quality, bad anatomy, deformed, bad hands, animal, blurry, text",
            )
        if any(k in name_lower for k in ("flux", "klein", "sd3")):
            return ModelProfile(
                model_id=f"auto_{filename}",
                display_name=f"Auto-detected FLUX ({filename})",
                filename=filename,
                family=PromptFamily.NATURAL_PROSE,
                skill_name="media-prompt-flux",
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
