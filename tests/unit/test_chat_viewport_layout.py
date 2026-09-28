"""Unit tests for the modern chat layout's static assets (#339)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from uclone_x.ui.app import create_ui_app


@pytest.mark.asyncio
async def test_ui_static_assets_exist_and_served(tmp_path: Path) -> None:
    """Verify that static bundle files generated for the modern layout are properly served."""
    static_dir = Path(__file__).resolve().parents[2] / "src" / "uclone_x" / "ui_static"
    assert (static_dir / "index.html").exists()

    app = create_ui_app(static_dir=static_dir)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "<!doctype html>" in res.text.lower() or "<html" in res.text.lower()
