# pyright: reportPrivateUsage=false
"""One clone's picture, and what happens when it has none (#1300).

Pictures only: nothing here draws a face from a clone's name. A clone either has an image
file installed beside its definition or it gets the single default the head ships, so the
same clone looks the same on every head rather than depending on a generator's version.

The route is therefore allowed to refuse, and most clones will make it refuse. What these
tests hold it to is that the refusal names the remedy, and that the path it reads is built
from the persona the registry loaded rather than from the name in the URL.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.tools.registry import ToolRegistry, create_default_registry
from uclone_x.ui.app import create_ui_app

PERSONAS_SUBDIR = Path(".uclone") / "personas"

#: A 1x1 transparent PNG. Small enough to inline, and a real one: the route hands the bytes
#: back with a media type, and a test that fed it arbitrary bytes would not notice if it
#: ever started re-encoding them.
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _tools(workspace: Path) -> ToolRegistry:
    """The tools the shipped clones declare.

    Not an empty registry: the persona loader validates a clone's declared tools against
    whatever inventory it is handed, so an empty one makes every built-in clone fail to load
    before this route is reached at all.
    """
    return create_default_registry(workspace_root=workspace, enable_mcp=False)


def _client(workspace: Path) -> TestClient:
    app = create_ui_app(
        static_dir=workspace / "static",
        storage_dir=workspace / "sessions",
        llm=MockLLMConnector(default_response="ok"),
        tools=_tools(workspace),
        workspace_dir=workspace,
    )
    return TestClient(app)


def _install(workspace: Path, name: str) -> Path:
    """Write one persona into the workspace and return the file it landed in."""
    directory = workspace / PERSONAS_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    definition = directory / f"{name}.yaml"
    definition.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "role": "Site Surveyor",
                "description": "Maps a site before anyone digs.",
                "system_prompt": "You survey.",
                "allowed_tools": [],
            }
        ),
        encoding="utf-8",
    )
    return definition


def test_a_clone_with_a_picture_beside_it_serves_that_picture(tmp_path: Path) -> None:
    """The bytes on disk, under the type the extension names.

    The bytes, not merely a 200 with the right header: an answer that found the file and
    then sent nothing is the empty container this project forbids, and it would otherwise
    reach the browser as a broken image with no sentence saying why.

    Killed by: src/uclone_x/ui/app.py :: content=path.read_bytes()
    Becomes: content=b""
    """
    definition = _install(tmp_path, "surveyor")
    definition.with_suffix(".png").write_bytes(ONE_PIXEL_PNG)

    response = _client(tmp_path).get("/api/personas/surveyor/avatar")

    assert response.status_code == 200
    assert response.content == ONE_PIXEL_PNG
    assert response.headers["content-type"] == "image/png"


def test_a_clone_with_no_picture_says_what_would_put_one_there(tmp_path: Path) -> None:
    """A refusal, not an empty body: the reader is told where the file goes and what it may be.

    Most clones reach this, and the head draws its default when it does. The sentence is for
    the reader who installed a picture and is looking for why it is not on screen.

    Killed by: src/uclone_x/ui/app.py :: if candidate.is_file():
    Becomes: if True:
    """
    _install(tmp_path, "surveyor")

    response = _client(tmp_path).get("/api/personas/surveyor/avatar")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "surveyor" in detail
    assert ".png" in detail and ".webp" in detail


def test_one_format_is_chosen_when_a_clone_has_more_than_one(tmp_path: Path) -> None:
    """Two files, one answer, and the same answer on every machine.

    A directory listing is not ordered, so "whichever comes first" would differ between
    installs. `_AVATAR_FORMATS` decides, and PNG is ahead of JPEG in it.

    Killed by: src/uclone_x/ui/app.py :: for suffix, mime in _AVATAR_FORMATS:
    Becomes: for suffix, mime in reversed(_AVATAR_FORMATS):
    """
    definition = _install(tmp_path, "surveyor")
    definition.with_suffix(".png").write_bytes(ONE_PIXEL_PNG)
    definition.with_suffix(".jpg").write_bytes(b"not a png")

    response = _client(tmp_path).get("/api/personas/surveyor/avatar")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_a_name_no_clone_carries_reaches_no_file(tmp_path: Path) -> None:
    """The path comes from the loaded persona, never from the request.

    `source_of` answers `None` for a name nothing installed, and this guard is the whole of
    what stops the route there. Without it the name from the URL is what a path gets composed
    from, which is how a segment like `..` starts naming files outside the personas
    directory. Removing the guard does not produce a wrong picture here, it produces a fault,
    and a fault is still not the refusal a reader is owed.

    Killed by: src/uclone_x/ui/app.py :: if source is None:
    Becomes: if source is None and False:
    """
    _install(tmp_path, "surveyor")
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(ONE_PIXEL_PNG)

    response = _client(tmp_path).get(f"/api/personas/{outside.stem}/avatar")

    assert response.status_code == 404
    assert response.content != ONE_PIXEL_PNG


@pytest.mark.parametrize("name", ["clone", "artist", "guardian", "pioneer", "scout", "writer"])
def test_shipped_builtin_clones_serve_default_avatar(tmp_path: Path, name: str) -> None:
    """Each shipped built-in clone carries a cute pastel avatar served as PNG."""
    response = _client(tmp_path).get(f"/api/personas/{name}/avatar")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0
