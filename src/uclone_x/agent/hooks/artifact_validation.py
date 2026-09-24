"""Validation and sanitization hook for hallucinated or missing artifact links."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from uclone_x.agent.hooks.models import HookAction, HookContext, HookDecision
from uclone_x.agent.hooks.protocols import BaseHook

logger = logging.getLogger(__name__)

# Matches markdown images: ![alt](url)
MD_IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")

# Matches markdown links: [title](url)
MD_LINK_RE = re.compile(r"(?<!!)\[(.*?)\]\((.*?)\)")

# Matches artifact content endpoint: /api/artifacts/content
ARTIFACT_ENDPOINT = "/api/artifacts/content"


def extract_artifact_rel_path(raw_url: str) -> str | None:
    """Extract relative workspace path if raw_url targets artifacts storage."""
    clean_url = raw_url.strip()
    if clean_url.startswith(ARTIFACT_ENDPOINT):
        parsed = urlparse(clean_url)
        params = parse_qs(parsed.query)
        path_list = params.get("path")
        if path_list and path_list[0]:
            return unquote(path_list[0]).strip()
        return None

    unquoted = unquote(clean_url).strip()
    if unquoted.startswith("artifacts/"):
        return unquoted

    return None


def is_artifact_missing(rel_path: str, workspace_root: Path | None) -> bool:
    """Check if an artifact path is missing or invalid on disk."""
    if not rel_path:
        return False
    if ".." in rel_path:
        return True
    ws = Path.cwd() if workspace_root is None else workspace_root
    target = (ws / rel_path).resolve()
    try:
        target.relative_to(ws.resolve())
    except ValueError:
        return True

    return not target.is_file()


def sanitize_hallucinated_artifacts(content: str, workspace_root: Path | None) -> tuple[str, int]:
    """Replace nonexistent artifact image references with a reader-facing notice banner.

    Returns:
        tuple of (sanitized_content, replacement_count)
    """
    replacements = 0

    def _replace_image(match: re.Match[str]) -> str:
        nonlocal replacements
        raw_url = match.group(2)
        rel_path = extract_artifact_rel_path(raw_url)
        if rel_path and (
            rel_path.startswith("artifacts/images/") or rel_path.startswith("artifacts/")
        ):
            if is_artifact_missing(rel_path, workspace_root):
                replacements += 1
                filename = Path(rel_path).name or rel_path
                return f"> ⚠️ *[이미지 생성 도구가 실행되지 않아 이미지가 표시되지 않습니다: {filename}]*"
        return match.group(0)

    def _replace_link(match: re.Match[str]) -> str:
        nonlocal replacements
        raw_url = match.group(2)
        rel_path = extract_artifact_rel_path(raw_url)
        if rel_path and (
            rel_path.startswith("artifacts/images/") or rel_path.startswith("artifacts/")
        ):
            if is_artifact_missing(rel_path, workspace_root):
                replacements += 1
                filename = Path(rel_path).name or rel_path
                return f"*[생성되지 않은 이미지 링크: {filename}]*"
        return match.group(0)

    # First replace markdown images
    sanitized = MD_IMAGE_RE.sub(_replace_image, content)
    # Then replace markdown links pointing to missing artifacts
    sanitized = MD_LINK_RE.sub(_replace_link, sanitized)

    return sanitized, replacements


class ArtifactValidationHook(BaseHook):
    """Post-turn hook that guards against nonexistent/hallucinated artifact links in agent output."""

    def __init__(
        self,
        workspace_root: Path | Callable[[], Path | None] | None = None,
        name: str = "artifact_validation_hook",
    ) -> None:
        super().__init__(name=name)
        self._workspace_root_spec = workspace_root

    @property
    def workspace_root(self) -> Path | None:
        if callable(self._workspace_root_spec):
            return self._workspace_root_spec()
        return self._workspace_root_spec

    async def on_post_turn(self, context: HookContext) -> HookDecision:
        content = context.payload.get("content")
        if not isinstance(content, str) or not content:
            return HookDecision(action=HookAction.ALLOW)

        ws = self.workspace_root
        if ws is None and "workspace_root" in context.payload:
            payload_ws = context.payload["workspace_root"]
            if isinstance(payload_ws, str) and payload_ws:
                ws = Path(payload_ws)

        sanitized, count = sanitize_hallucinated_artifacts(content, ws)
        if count > 0:
            logger.info(
                "ArtifactValidationHook sanitized %d hallucinated artifact link(s) for agent %s",
                count,
                context.agent_id,
            )
            return HookDecision(
                action=HookAction.MODIFY,
                reason=f"Sanitized {count} nonexistent artifact link(s)",
                modified_payload={"content": sanitized},
            )

        return HookDecision(action=HookAction.ALLOW)
