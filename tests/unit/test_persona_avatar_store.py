"""`PersonaAvatarStore`: where a clone's picture is found, and how it is changed.

What these pin:

* **Lookup order**: a picture chosen in the workspace wins; otherwise the one beside the
  loaded definition; otherwise the one shipped beside the built-in of that name. The last
  is what keeps an edited built-in's picture: editing moves its definition into the
  workspace, and the picture used to be looked for only beside the definition.
* **Only pictures**: the bytes decide, not the name. SVG and an HTML page named `.png` are
  refused, and so is anything over 10 MiB.
* **One picture per clone**: setting a picture in another format removes the old one, so
  lookup order cannot bring it back; the replaced one is kept as `<name>.prev.<ext>`, one
  deep.
* **Reset** puts the chosen picture aside and the shipped one shows again.
* **Version** changes when the picture does, so the head's `?v=` address changes.
* **The write guard**: general writing tools refuse any path in the personas folder,
  however it is spelled, while writing elsewhere under `.uclone/` is unchanged.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from uclone_x.agent.persona_avatar import (
    MAX_AVATAR_BYTES,
    AvatarPersonaNotFound,
    AvatarRefused,
    PersonaAvatarStore,
    avatar_url,
    sniff_image_format,
)
from uclone_x.agent.persona_registry import BUILTIN_PERSONAS_DIR, PersonaRegistry
from uclone_x.agent.persona_store import PersonaLoadError
from uclone_x.tools.builtin.filesystem import FileWriteTool
from uclone_x.tools.builtin.image import GenerateImageTool, ImagePipelineDispatcher
from uclone_x.tools.models import NoIsolation, ToolContext

PERSONAS = Path(".uclone") / "personas"

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 16
GIF = b"GIF89a\x01\x00\x01\x00" + b"\x00" * 16
WEBP = b"RIFF\x1a\x00\x00\x00WEBPVP8 " + b"\x00" * 16


def _install(workspace: Path, name: str) -> Path:
    folder = workspace / PERSONAS
    folder.mkdir(parents=True, exist_ok=True)
    definition = folder / f"{name}.yaml"
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


def _store(workspace: Path) -> PersonaAvatarStore:
    return PersonaAvatarStore(PersonaRegistry(workspace_root=workspace))


class TestLookup:
    def test_an_edited_builtin_keeps_its_shipped_picture(self, tmp_path: Path) -> None:
        """D1: the override moves the definition to the workspace; the picture stays shipped.

        Killed by: src/uclone_x/agent/persona_avatar.py :: folders.append((BUILTIN_PERSONAS_DIR, name))
        Becomes: folders.append((source.parent, name))
        """
        _install(tmp_path, "writer")

        found = _store(tmp_path).find("writer")

        assert found is not None
        assert found.path == BUILTIN_PERSONAS_DIR / "writer.png"
        assert found.mime == "image/png"

    def test_a_picture_chosen_in_the_workspace_wins_over_the_shipped_one(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        store.set("artist", JPEG)

        found = store.find("artist")

        assert found is not None
        assert found.path == tmp_path / PERSONAS / "artist.jpg"
        assert found.path.read_bytes() == JPEG

    def test_a_name_no_clone_carries_has_no_picture(self, tmp_path: Path) -> None:
        stray = tmp_path / PERSONAS / "nobody.png"
        stray.parent.mkdir(parents=True)
        stray.write_bytes(PNG)

        assert _store(tmp_path).find("nobody") is None

    def test_a_name_that_climbs_out_of_the_folder_writes_nothing(self, tmp_path: Path) -> None:
        """A path is built from a name only once the name is a loaded clone (#1772).

        Killed by: src/uclone_x/agent/persona_avatar.py :: if self._registry.source_of(name) is None:  # only a loaded clone's name makes a path
        Becomes: if False:
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)

        with pytest.raises(AvatarPersonaNotFound) as refused:
            store.set("../escape", PNG)

        assert refused.value.reason_code == "no_clone"
        assert not (tmp_path / ".uclone" / "escape.png").exists()

    def test_a_kept_picture_cannot_be_taken_for_another_clone(self, tmp_path: Path) -> None:
        """`writer.prev.png` is writer's kept picture, never a clone named `writer.prev` (#1772).

        The loader refuses a name with a dot, so no clone can be named after a kept copy.

        Killed by: src/uclone_x/agent/persona_store.py :: refuse_an_unusable_username(persona_name)
        Becomes: pass
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        store.set("surveyor", PNG)
        store.set("surveyor", JPEG)
        assert store.previous("surveyor") == tmp_path / PERSONAS / "surveyor.prev.png"

        _install(tmp_path, "surveyor.prev")
        with pytest.raises(PersonaLoadError, match="surveyor.prev"):
            PersonaRegistry(workspace_root=tmp_path).get_persona("surveyor.prev")

    def test_the_url_carries_the_version_and_is_null_without_a_picture(
        self, tmp_path: Path
    ) -> None:
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)

        assert avatar_url("surveyor", store.find("surveyor")) is None
        record = store.set("surveyor", PNG)
        assert avatar_url("surveyor", record) == f"/api/personas/surveyor/avatar?v={record.version}"


class TestSet:
    @pytest.mark.parametrize(
        ("data", "mime"),
        [(PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif"), (WEBP, "image/webp")],
    )
    def test_each_accepted_format_is_recognised_by_its_bytes(self, data: bytes, mime: str) -> None:
        assert sniff_image_format(data) == mime

    @pytest.mark.parametrize(
        "data",
        [
            b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
            b"<!doctype html><html><body>not a picture</body></html>",
            b"",
        ],
        ids=["svg", "html-named-png", "empty"],
    )
    def test_what_is_not_a_picture_is_refused_whatever_it_is_called(
        self, tmp_path: Path, data: bytes
    ) -> None:
        """Decided by content: an HTML page renamed `face.png` is still refused.

        Killed by: src/uclone_x/agent/persona_avatar.py :: mime = sniff_image_format(data)
        Becomes: mime = "image/png"
        """
        _install(tmp_path, "surveyor")
        source = tmp_path / "face.png"
        source.write_bytes(data)
        store = _store(tmp_path)

        with pytest.raises(AvatarRefused) as caught:
            store.set_from_path("surveyor", source)

        assert "PNG, JPEG, WebP or GIF" in str(caught.value)
        assert store.find("surveyor") is None

    def test_a_picture_over_ten_mib_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/persona_avatar.py :: if len(data) > MAX_AVATAR_BYTES:
        Becomes: if False:
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)

        with pytest.raises(AvatarRefused) as caught:
            store.set("surveyor", PNG + b"\x00" * MAX_AVATAR_BYTES)

        assert "10 MB" in str(caught.value)
        assert store.find("surveyor") is None

    def test_a_new_format_removes_the_old_file_so_it_cannot_come_back(self, tmp_path: Path) -> None:
        """PNG is looked for before JPEG, so a PNG left behind would win over the new JPEG.

        Killed by: src/uclone_x/agent/persona_avatar.py :: if other != target:
        Becomes: if False:
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        store.set("surveyor", PNG)

        store.set("surveyor", JPEG)

        found = store.find("surveyor")
        assert found is not None
        assert (found.mime, found.path.read_bytes()) == ("image/jpeg", JPEG)
        assert not (tmp_path / PERSONAS / "surveyor.png").exists()

    def test_the_replaced_picture_is_kept_one_deep(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/persona_avatar.py :: if current:
        Becomes: if False:
        Killed by: src/uclone_x/agent/persona_avatar.py :: if older != kept:
        Becomes: if False:
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        store.set("surveyor", PNG)
        store.set("surveyor", JPEG)
        assert store.previous("surveyor") == tmp_path / PERSONAS / "surveyor.prev.png"

        store.set("surveyor", GIF)

        kept = sorted(p.name for p in (tmp_path / PERSONAS).glob("surveyor.prev.*"))
        assert kept == ["surveyor.prev.jpg"]
        assert (tmp_path / PERSONAS / "surveyor.prev.jpg").read_bytes() == JPEG

    def test_the_version_changes_when_the_picture_does(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/persona_avatar.py :: version=f"{stat.st_mtime_ns}-{stat.st_size}"
        Becomes: version="v"
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)

        first = store.set("surveyor", PNG).version
        second = store.set("surveyor", PNG + b"\x00").version

        assert first != second

    def test_a_name_no_clone_carries_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(AvatarPersonaNotFound):
            _store(tmp_path).set("nobody", PNG)
        assert not (tmp_path / PERSONAS / "nobody.png").exists()


class TestReset:
    def test_reset_puts_the_chosen_picture_aside_and_the_shipped_one_shows(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/agent/persona_avatar.py :: kept = self._keep_previous(name, current[0])
        Becomes: kept = None
        """
        store = _store(tmp_path)
        store.set("artist", JPEG)

        kept = store.reset("artist")

        assert kept == tmp_path / PERSONAS / "artist.prev.jpg"
        assert kept is not None and kept.read_bytes() == JPEG
        found = store.find("artist")
        assert found is not None
        assert found.path == BUILTIN_PERSONAS_DIR / "artist.png"

    def test_reset_with_nothing_chosen_changes_nothing(self, tmp_path: Path) -> None:
        assert _store(tmp_path).reset("artist") is None


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(
        agent_id="artist",
        session_id="sess_1",
        workspace_root=workspace,
        isolation=NoIsolation(),
    )


class TestTheWriteGuard:
    @pytest.mark.parametrize(
        "path",
        [
            ".uclone/personas/writer.yaml",
            ".UClone/Personas/writer.png",
            "./.uclone/x/../personas/a.png",
        ],
        ids=["plain", "other-case", "detour"],
    )
    async def test_file_write_into_the_personas_folder_is_refused(
        self, tmp_path: Path, path: str
    ) -> None:
        """Checked even before the folder exists, so nothing can be planted there early.

        Killed by: src/uclone_x/tools/base.py :: if in_personas_dir(resolved, workspace_root):
        Becomes: if False:
        Killed by: src/uclone_x/tools/base.py :: if tuple(part.casefold() for part in parts[:2]) == _PERSONAS_PARTS:
        Becomes: if tuple(parts[:2]) == _PERSONAS_PARTS:
        """
        result = await FileWriteTool().execute(
            {"path": path, "content": "name: writer\n", "overwrite": True}, _ctx(tmp_path)
        )

        assert result.success is False
        assert result.error is not None
        assert "where clones' definitions and pictures are kept" in result.error
        assert "set_avatar" in result.error
        assert not (tmp_path / PERSONAS).exists()

    async def test_a_personas_folder_that_is_a_link_is_refused_by_its_target(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/tools/base.py :: if folder.exists() and folder.samefile(personas):
        Becomes: if False:
        """
        real = tmp_path / "kept"
        real.mkdir()
        (tmp_path / ".uclone").mkdir()
        (tmp_path / PERSONAS).symlink_to(real, target_is_directory=True)

        result = await FileWriteTool().execute(
            {"path": "kept/writer.yaml", "content": "name: writer\n"}, _ctx(tmp_path)
        )

        assert result.success is False
        assert not (real / "writer.yaml").exists()

    async def test_generate_image_into_the_personas_folder_is_refused(self, tmp_path: Path) -> None:
        """How Artist could have replaced another clone's face: an output path there."""
        dispatcher = AsyncMock(spec=ImagePipelineDispatcher)

        result = await GenerateImageTool(dispatcher=dispatcher).execute(
            {"prompt": "a friendly face", "output_path": ".uclone/personas/writer.png"},
            _ctx(tmp_path),
        )

        assert result.success is False
        assert result.error is not None
        assert "where clones' definitions and pictures are kept" in result.error
        dispatcher.dispatch.assert_not_called()

    async def test_the_rest_of_the_workspace_is_written_as_before(self, tmp_path: Path) -> None:
        for path in (".uclone/notes.txt", "personas/draft.yaml", ".uclone/personas-old/a.txt"):
            result = await FileWriteTool().execute(
                {"path": path, "content": "kept\n"}, _ctx(tmp_path)
            )
            assert result.success, (path, result.error)
