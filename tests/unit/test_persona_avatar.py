# pyright: reportPrivateUsage=false
"""One clone's picture, what happens when it has none (#1300), and how it is changed.

Pictures only: nothing here draws a face from a clone's name. A clone either has an image
file installed beside its definition or it gets the single default the head ships, so the
same clone looks the same on every head rather than depending on a generator's version.

The route is therefore allowed to refuse, and most clones will make it refuse. What these
tests hold it to is that the refusal names the remedy, and that the path it reads is built
from the persona the registry loaded rather than from the name in the URL.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from uclone_x.agent.persona_avatar import MAX_AVATAR_BYTES
from uclone_x.agent.persona_registry import BUILTIN_PERSONAS_DIR
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

    Killed by: src/uclone_x/ui/app.py :: content=found.path.read_bytes(),
    Becomes: content=b"",
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

    Killed by: src/uclone_x/agent/persona_avatar.py :: if candidate.is_file():
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
    installs. `AVATAR_FORMATS` decides, and PNG is ahead of JPEG in it.

    Killed by: src/uclone_x/agent/persona_avatar.py :: for suffix, mime in AVATAR_FORMATS:
    Becomes: for suffix, mime in reversed(AVATAR_FORMATS):
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

    Killed by: src/uclone_x/agent/persona_avatar.py :: if source is None:
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


def test_the_picture_is_sent_as_what_it_is_and_never_from_a_stale_cache(tmp_path: Path) -> None:
    """`nosniff` so a browser never reads the bytes as anything but the image type sent.

    Killed by: src/uclone_x/ui/app.py :: headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    Becomes: headers={},
    """
    definition = _install(tmp_path, "surveyor")
    definition.with_suffix(".png").write_bytes(ONE_PIXEL_PNG)

    response = _client(tmp_path).get("/api/personas/surveyor/avatar")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-cache"


def _payload(client: TestClient, name: str) -> dict[str, object]:
    personas = client.get("/api/personas").json()["personas"]
    return next(p for p in personas if p["name"] == name)


def test_the_persona_payload_says_where_its_picture_is_shown_from(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/ui/app.py :: "avatar_url": avatar_url(persona.name, PersonaAvatarStore(registry).find(persona.name)),
    Becomes: "avatar_url": None,
    """
    _install(tmp_path, "surveyor")
    client = _client(tmp_path)

    assert _payload(client, "surveyor")["avatar_url"] is None
    shipped = _payload(client, "artist")["avatar_url"]
    assert isinstance(shipped, str) and shipped.startswith("/api/personas/artist/avatar?v=")
    assert client.get(shipped).content == (BUILTIN_PERSONAS_DIR / "artist.png").read_bytes()


def test_put_with_a_workspace_path_sets_the_picture(tmp_path: Path) -> None:
    """The route a head calls when the user clicks Use as avatar under a drawn image.

    Killed by: src/uclone_x/ui/app.py :: store.set_from_path(name, _avatar_source(await _avatar_json(request)))
    Becomes: pass
    """
    _install(tmp_path, "surveyor")
    drawn = tmp_path / "artifacts" / "images" / "img_1.png"
    drawn.parent.mkdir(parents=True)
    drawn.write_bytes(ONE_PIXEL_PNG)
    client = _client(tmp_path)

    response = client.put(
        "/api/personas/surveyor/avatar", json={"source_path": "artifacts/images/img_1.png"}
    )

    assert response.status_code == 200, response.text
    url = response.json()["persona"]["avatar_url"]
    assert isinstance(url, str) and "?v=" in url
    assert client.get(url).content == ONE_PIXEL_PNG


def test_put_with_the_image_itself_sets_the_picture(tmp_path: Path) -> None:
    """An upload: the body is the picture.

    Killed by: src/uclone_x/ui/app.py :: store.set(name, await _avatar_upload(request))
    Becomes: pass
    """
    _install(tmp_path, "surveyor")
    client = _client(tmp_path)

    response = client.put(
        "/api/personas/surveyor/avatar",
        content=ONE_PIXEL_PNG,
        headers={"content-type": "image/png"},
    )

    assert response.status_code == 200, response.text
    assert client.get("/api/personas/surveyor/avatar").content == ONE_PIXEL_PNG


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        (b"<svg xmlns='http://www.w3.org/2000/svg'></svg>", {"content-type": "image/svg+xml"}),
        (b"<html>not a picture</html>", {"content-type": "image/png"}),
    ],
    ids=["svg", "html-as-png"],
)
def test_put_refuses_what_is_not_a_picture_in_plain_words(
    tmp_path: Path, body: bytes, headers: dict[str, str]
) -> None:
    _install(tmp_path, "surveyor")

    response = _client(tmp_path).put("/api/personas/surveyor/avatar", content=body, headers=headers)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "PNG, JPEG, WebP or GIF" in detail
    for internal in ("Error", "Traceback", "sniff", "magic", str(tmp_path)):
        assert internal not in detail


def test_put_refuses_a_source_path_outside_the_workspace(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/ui/app.py :: return PathValidator().resolve_safe_path(Path(raw), session_mgr.workspace_dir)
    Becomes: return Path(raw)
    """
    workspace = tmp_path / "work"
    workspace.mkdir()
    _install(workspace, "surveyor")
    outside = tmp_path / "outside.png"
    outside.write_bytes(ONE_PIXEL_PNG)
    client = _client(workspace)

    response = client.put("/api/personas/surveyor/avatar", json={"source_path": str(outside)})

    assert response.status_code == 422
    assert "outside the workspace" in response.json()["detail"]
    assert client.get("/api/personas/surveyor/avatar").status_code == 404


def test_a_chunked_upload_is_read_no_further_than_the_cap(tmp_path: Path) -> None:
    """A chunked body has no `content-length`, so the cap is counted as it arrives (#1772).

    Driven over raw ASGI rather than `TestClient`, which reads a streamed body in full
    before the app sees any of it and so cannot tell a capped read from a full one.

    Killed by: src/uclone_x/ui/app.py :: if len(body) > MAX_AVATAR_BYTES:
    Becomes: if False:
    """
    _install(tmp_path, "surveyor")
    app = _client(tmp_path).app
    chunk = b"\x00" * (1024 * 1024)
    pulled: list[int] = []
    answer: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if len(pulled) >= 40:
            return {"type": "http.disconnect"}
        pulled.append(len(chunk))
        return {"type": "http.request", "body": chunk, "more_body": len(pulled) < 40}

    async def send(message: dict[str, Any]) -> None:
        answer.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "PUT",
        "scheme": "http",
        "path": "/api/personas/surveyor/avatar",
        "raw_path": b"/api/personas/surveyor/avatar",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1"), (b"content-type", b"image/png")],
        "client": ("testclient", 50000),
        "server": ("127.0.0.1", 80),
    }
    asyncio.run(app(scope, receive, send))  # pyright: ignore[reportArgumentType]

    status = next(m["status"] for m in answer if m["type"] == "http.response.start")
    body = json.loads(
        b"".join(m.get("body", b"") for m in answer if m["type"] == "http.response.body")
    )
    assert status == 422
    assert body["code"] == "too_large"
    assert "10 MB" in body["detail"]
    assert sum(pulled) <= MAX_AVATAR_BYTES + len(chunk)


def test_a_body_that_is_not_json_is_refused_in_plain_words(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/ui/app.py :: raise _avatar_no_source() from exc
    Becomes: raise
    """
    _install(tmp_path, "surveyor")

    response = _client(tmp_path).put(
        "/api/personas/surveyor/avatar",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "no_source"
    detail = response.json()["detail"]
    assert "source_path" in detail
    for internal in ("Error", "Traceback", "JSONDecode", "Expecting"):
        assert internal not in detail


def test_each_refusal_says_which_reason_it_is(tmp_path: Path) -> None:
    """The head words its own message from `code`, so each reason has its own (#1780).

    Killed by: src/uclone_x/ui/app.py :: return JSONResponse({"detail": str(exc), "code": exc.reason_code}, status_code=status)
    Becomes: return JSONResponse({"detail": str(exc), "code": None}, status_code=status)
    """
    workspace = tmp_path / "work"
    workspace.mkdir()
    _install(workspace, "surveyor")
    outside = tmp_path / "outside.png"
    outside.write_bytes(ONE_PIXEL_PNG)
    client = _client(workspace)
    url = "/api/personas/surveyor/avatar"

    def code(response: Any) -> object:
        return response.json()["code"]

    assert code(client.put(url, json={"source_path": str(outside)})) == "outside_workspace"
    assert code(client.put(url, json={"source_path": "missing.png"})) == "no_file"
    assert code(client.put(url, json={})) == "no_source"
    assert (
        code(client.put(url, content=b"<html/>", headers={"content-type": "image/png"}))
        == "not_an_image"
    )
    assert (
        code(
            client.put(
                "/api/personas/nobody/avatar",
                content=ONE_PIXEL_PNG,
                headers={"content-type": "image/png"},
            )
        )
        == "no_clone"
    )


def test_put_for_a_name_no_clone_carries_is_404(tmp_path: Path) -> None:
    response = _client(tmp_path).put(
        "/api/personas/nobody/avatar", content=ONE_PIXEL_PNG, headers={"content-type": "image/png"}
    )

    assert response.status_code == 404
    assert "nobody" in response.json()["detail"]
    assert not (tmp_path / PERSONAS_SUBDIR / "nobody.png").exists()


def test_delete_goes_back_to_the_shipped_picture(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/ui/app.py :: PersonaAvatarStore(registry).reset(name)
    Becomes: None
    """
    client = _client(tmp_path)
    client.put(
        "/api/personas/artist/avatar",
        content=ONE_PIXEL_PNG + b"\x00",
        headers={"content-type": "image/png"},
    )
    assert client.get("/api/personas/artist/avatar").content == ONE_PIXEL_PNG + b"\x00"

    response = client.delete("/api/personas/artist/avatar")

    assert response.status_code == 200, response.text
    shipped = (BUILTIN_PERSONAS_DIR / "artist.png").read_bytes()
    assert client.get(response.json()["persona"]["avatar_url"]).content == shipped


def test_a_shipped_picture_is_not_reported_as_chosen(tmp_path: Path) -> None:
    """The head offers Reset only for a chosen picture; a shipped one has nothing to reset (#1780).

    Killed by: src/uclone_x/ui/app.py :: "avatar_chosen": PersonaAvatarStore(registry).chosen(persona.name) is not None,
    Becomes: "avatar_chosen": True,
    """
    client = _client(tmp_path)

    def artist() -> dict[str, object]:
        personas: list[dict[str, object]] = client.get("/api/personas").json()["personas"]
        return next(p for p in personas if p["name"] == "artist")

    assert artist()["avatar_url"] is not None
    assert artist()["avatar_chosen"] is False

    changed = client.put(
        "/api/personas/artist/avatar", content=ONE_PIXEL_PNG, headers={"content-type": "image/png"}
    )

    assert changed.json()["persona"]["avatar_chosen"] is True
    assert artist()["avatar_chosen"] is True


@pytest.mark.parametrize("method", ["put", "delete"])
def test_another_site_cannot_change_a_clones_picture(tmp_path: Path, method: str) -> None:
    """The app answers any origin, so an open tab elsewhere could otherwise post here.

    Killed by: src/uclone_x/ui/app.py :: _refuse_cross_origin(request)  # before anything is read or written
    Becomes: pass
    """
    client = _client(tmp_path)
    client.put(
        "/api/personas/artist/avatar", content=ONE_PIXEL_PNG, headers={"content-type": "image/png"}
    )
    hostile = {"origin": "https://example.invalid"}

    if method == "put":
        response = client.put(
            "/api/personas/artist/avatar",
            content=ONE_PIXEL_PNG + b"\x00",
            headers={**hostile, "content-type": "image/png"},
        )
    else:
        response = client.delete("/api/personas/artist/avatar", headers=hostile)

    assert response.status_code == 403
    assert client.get("/api/personas/artist/avatar").content == ONE_PIXEL_PNG


def test_a_change_answers_with_the_picture_that_undoes_it(tmp_path: Path) -> None:
    """The head's Undo puts back what `previous_path` names, or resets when it is null.

    Killed by: src/uclone_x/ui/app.py :: registry, name, undo_with=store.previous(name) if had_chosen else None
    Becomes: registry, name
    """
    _install(tmp_path, "surveyor")
    client = _client(tmp_path)
    first = ONE_PIXEL_PNG
    second = ONE_PIXEL_PNG + b"\x00"

    initial = client.put(
        "/api/personas/surveyor/avatar", content=first, headers={"content-type": "image/png"}
    )
    assert initial.json()["previous_path"] is None

    replaced = client.put(
        "/api/personas/surveyor/avatar", content=second, headers={"content-type": "image/png"}
    )
    previous = replaced.json()["previous_path"]
    assert previous == ".uclone/personas/surveyor.prev.png"

    undone = client.put("/api/personas/surveyor/avatar", json={"source_path": previous})

    assert undone.status_code == 200, undone.text
    assert client.get("/api/personas/surveyor/avatar").content == first


def test_a_reset_answers_with_the_picture_it_put_aside(tmp_path: Path) -> None:
    """Undoing a reset puts the kept copy back.

    Killed by: src/uclone_x/ui/app.py :: return _avatar_answer(registry, name, undo_with=kept)
    Becomes: return _avatar_answer(registry, name)
    """
    _install(tmp_path, "surveyor")
    client = _client(tmp_path)
    client.put(
        "/api/personas/surveyor/avatar",
        content=ONE_PIXEL_PNG,
        headers={"content-type": "image/png"},
    )

    reset = client.delete("/api/personas/surveyor/avatar")
    kept = reset.json()["previous_path"]
    assert reset.json()["persona"]["avatar_url"] is None
    assert kept == ".uclone/personas/surveyor.prev.png"

    client.put("/api/personas/surveyor/avatar", json={"source_path": kept})

    assert client.get("/api/personas/surveyor/avatar").content == ONE_PIXEL_PNG
