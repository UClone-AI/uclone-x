"""Agent lifecycle and tool execution hook subsystem."""

from uclone_x.agent.hooks.artifact_validation import (
    MD_IMAGE_RE,
    ArtifactValidationHook,
    extract_artifact_rel_path,
    is_artifact_missing,
    sanitize_hallucinated_artifacts,
)
from uclone_x.agent.hooks.models import (
    FailurePolicy,
    HookAction,
    HookContext,
    HookDecision,
    HookEvent,
)
from uclone_x.agent.hooks.protocols import BaseHook
from uclone_x.agent.hooks.runner import HookRunner
from uclone_x.agent.hooks.script_runner import ScriptHook

__all__ = [
    "ArtifactValidationHook",
    "BaseHook",
    "FailurePolicy",
    "HookAction",
    "HookContext",
    "HookDecision",
    "HookEvent",
    "HookRunner",
    "MD_IMAGE_RE",
    "ScriptHook",
    "extract_artifact_rel_path",
    "is_artifact_missing",
    "sanitize_hallucinated_artifacts",
]
