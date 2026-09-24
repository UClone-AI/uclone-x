"""Tests for ArtifactValidationHook and artifact path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.agent.hooks.artifact_validation import (
    ArtifactValidationHook,
    extract_artifact_rel_path,
    is_artifact_missing,
    sanitize_hallucinated_artifacts,
)
from uclone_x.agent.hooks.models import HookAction, HookContext, HookEvent


def test_extract_artifact_rel_path() -> None:
    # Query parameter format
    assert (
        extract_artifact_rel_path("/api/artifacts/content?path=artifacts/images/img_abc123.png")
        == "artifacts/images/img_abc123.png"
    )
    # Encoded query parameter format
    assert (
        extract_artifact_rel_path("/api/artifacts/content?path=artifacts%2Fimages%2Fimg_abc123.png")
        == "artifacts/images/img_abc123.png"
    )
    # Direct relative path
    assert (
        extract_artifact_rel_path("artifacts/images/img_abc123.png")
        == "artifacts/images/img_abc123.png"
    )
    # External URL or non-artifact
    assert extract_artifact_rel_path("https://example.com/image.png") is None
    assert extract_artifact_rel_path("/static/logo.png") is None


def test_is_artifact_missing(tmp_path: Path) -> None:
    img_dir = tmp_path / "artifacts" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    real_file = img_dir / "img_real.png"
    real_file.write_bytes(b"PNGDATA")

    # Real file exists
    assert not is_artifact_missing("artifacts/images/img_real.png", tmp_path)

    # Missing file
    assert is_artifact_missing("artifacts/images/img_fake.png", tmp_path)

    # Traversal attempt
    assert is_artifact_missing("../outside.png", tmp_path)
    assert is_artifact_missing("artifacts/images/../../secret.txt", tmp_path)


def test_sanitize_hallucinated_artifacts(tmp_path: Path) -> None:
    img_dir = tmp_path / "artifacts" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    real_file = img_dir / "img_real.png"
    real_file.write_bytes(b"PNGDATA")

    content = (
        "Here are two images:\n\n"
        "![Real](/api/artifacts/content?path=artifacts/images/img_real.png)\n"
        "![Fake](/api/artifacts/content?path=artifacts/images/img_fake.png)\n"
        "Also see [Fake Link](/api/artifacts/content?path=artifacts/images/img_fake2.png)."
    )

    sanitized, count = sanitize_hallucinated_artifacts(content, tmp_path)
    assert count == 2

    # Real image is intact
    assert "![Real](/api/artifacts/content?path=artifacts/images/img_real.png)" in sanitized

    # Fake image is replaced with banner
    assert "img_fake.png" in sanitized
    assert (
        "> ⚠️ *[이미지 생성 도구가 실행되지 않아 이미지가 표시되지 않습니다: img_fake.png]*"
        in sanitized
    )

    # Fake link is replaced
    assert "*[생성되지 않은 이미지 링크: img_fake2.png]*" in sanitized


@pytest.mark.asyncio
async def test_artifact_validation_hook_on_post_turn(tmp_path: Path) -> None:
    hook = ArtifactValidationHook(workspace_root=tmp_path)

    content_with_hallucination = (
        "Here is the generated idol photo:\n"
        "![Idol](/api/artifacts/content?path=artifacts/images/img_hallucinated.png)"
    )

    ctx = HookContext(
        agent_id="test_agent",
        event_type=HookEvent.POST_TURN,
        payload={"content": content_with_hallucination},
    )

    decision = await hook.on_post_turn(ctx)
    assert decision.action == HookAction.MODIFY
    assert decision.modified_payload is not None
    modified_content = decision.modified_payload["content"]
    assert "img_hallucinated.png" in modified_content
    assert "> ⚠️ *[" in modified_content


@pytest.mark.asyncio
async def test_artifact_validation_hook_allows_existing_files(tmp_path: Path) -> None:
    hook = ArtifactValidationHook(workspace_root=tmp_path)
    img_dir = tmp_path / "artifacts" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "img_valid.png").write_bytes(b"DATA")

    content_valid = "![Valid](/api/artifacts/content?path=artifacts/images/img_valid.png)"
    ctx = HookContext(
        agent_id="test_agent",
        event_type=HookEvent.POST_TURN,
        payload={"content": content_valid},
    )

    decision = await hook.on_post_turn(ctx)
    assert decision.action == HookAction.ALLOW
