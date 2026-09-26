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


def test_a_link_with_the_workspace_directory_for_a_host_is_re_rooted(tmp_path: Path) -> None:
    """The fresh-HOME Artist reply linked `https://<workspace dir>/work/api/artifacts/...` (#1618).

    The UI serves the rooted path on whatever host it was opened from, so the prefix is
    dropped, for an inline image and for a plain link alike. A prefixed link to a file that
    does not exist is still treated as a missing image, not passed through.

    Killed by: src/uclone_x/agent/hooks/artifact_validation.py :: return _rooted(match, "!")
    Becomes: return match.group(0)
    Killed by: src/uclone_x/agent/hooks/artifact_validation.py :: return _rooted(match, "")
    Becomes: return match.group(0)
    Killed by: src/uclone_x/agent/hooks/artifact_validation.py :: clean_url = rooted_artifact_url(raw_url) or raw_url.strip()
    Becomes: clean_url = raw_url.strip()
    """
    img_dir = tmp_path / "artifacts" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "img_1.png").write_bytes(b"PNG")
    host = "https://ucx-fresh-test2-pypi022/work"
    content = (
        f"![Night]({host}/api/artifacts/content?path=artifacts/images/img_1.png)\n"
        f"[Open it](ucx-fresh-test2-pypi022/work/api/artifacts/content?path=artifacts/images/img_1.png)\n"
        f"![Gone]({host}/api/artifacts/content?path=artifacts/images/img_gone.png)"
    )

    sanitized, count = sanitize_hallucinated_artifacts(content, tmp_path)

    assert count == 3
    assert "ucx-fresh-test2-pypi022" not in sanitized
    assert "![Night](/api/artifacts/content?path=artifacts/images/img_1.png)" in sanitized
    assert "[Open it](/api/artifacts/content?path=artifacts/images/img_1.png)" in sanitized
    assert "img_gone.png]*" in sanitized


def test_a_rooted_link_and_an_unrelated_url_are_left_as_written(tmp_path: Path) -> None:
    """Only a prefixed link to our own endpoint is rewritten; nothing else is touched."""
    img_dir = tmp_path / "artifacts" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "img_1.png").write_bytes(b"PNG")
    content = (
        "![Night](/api/artifacts/content?path=artifacts/images/img_1.png)\n"
        "[Docs](https://example.com/api/artifacts/list)"
    )

    assert sanitize_hallucinated_artifacts(content, tmp_path) == (content, 0)
