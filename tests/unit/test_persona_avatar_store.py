"""`PersonaAvatarStore`: where a clone's picture is found, and how it is changed.

What these pin:

* **Lookup order**: a picture chosen in the clone's directory wins; otherwise the one
  shipped beside the built-in of that handle. The last is what keeps an edited built-in's
  picture: a workspace edit of a built-in is imported as a clone of its own (2026-09-27,
  clone-data-scopes §3.8 step 2), and it still shows the shipped face.
* **Only pictures**: the bytes decide, not the name. SVG and an HTML page named `.png` are
  refused, and so is anything over 10 MiB.
* **One picture per clone**: setting a picture in another format removes the old one, so
  lookup order cannot bring it back; the replaced one is kept as `avatar.prev.<ext>` in the
  clone's directory, one deep.
* **Reset** puts the chosen picture aside and the shipped one shows again.
* **Version** changes when the picture does, so the head's `?v=` address changes.
* **Change order** is the store's (#1809): every change gets the next id, counted on disk so
  a new store (a restart) goes on from it, and an undo naming a change that is no longer
  the latest is refused with nothing written -- including after A, B, C, B, where the
  picture is B again but the change that made it the first time is not the latest.
* **The write guard**: general writing tools refuse any path in the personas folder,
  however it is spelled, while writing elsewhere under `.uclone/` is unchanged.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from uclone_x.agent import persona_avatar
from uclone_x.agent.persona_avatar import (
    MAX_AVATAR_BYTES,
    AvatarPersonaNotFound,
    AvatarRefused,
    AvatarStaleChange,
    PersonaAvatarStore,
    avatar_url,
    sniff_image_format,
)
from uclone_x.agent.persona_registry import BUILTIN_PERSONAS_DIR, PersonaRegistry
from uclone_x.agent.persona_store import PersonaLoadError, persona_from_mapping
from uclone_x.core.agent_home import AgentHome, default_agents_root
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


def _home(name: str) -> Path:
    """The clone directory of `name`, where its chosen and replaced pictures are kept."""
    return AgentHome.for_handle(name).path


class TestLookup:
    def test_an_edited_builtin_keeps_its_shipped_picture(self, tmp_path: Path) -> None:
        """D1: the workspace edit is imported as its own clone; the picture stays shipped.

        Killed by: src/uclone_x/agent/persona_avatar.py :: template = record.template or (name if self._registry.has_builtin(name) else None)
        Becomes: template = record.template
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
        assert found.path == _home("artist") / "avatar.jpg"
        assert found.path.read_bytes() == JPEG

    def test_a_name_no_clone_carries_has_no_picture(self, tmp_path: Path) -> None:
        stray = tmp_path / PERSONAS / "nobody.png"
        stray.parent.mkdir(parents=True)
        stray.write_bytes(PNG)

        assert _store(tmp_path).find("nobody") is None

    def test_a_name_that_climbs_out_of_the_folder_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path is built only inside a clone's own directory, never from a name (#1772).

        Since 2026-09-27 the folder is the one the clone record names, so a name no clone
        carries has no folder at all.

        Killed by: src/uclone_x/agent/persona_avatar.py :: return record.path if record is not None else None
        Becomes: return record.path if record is not None else Path(name)
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        (tmp_path / "here").mkdir()
        monkeypatch.chdir(tmp_path / "here")

        with pytest.raises(AvatarPersonaNotFound) as refused:
            store.set("../escape", PNG)

        assert refused.value.reason_code == "no_clone"
        assert not (tmp_path / ".uclone" / "escape.png").exists()
        assert not (tmp_path / "escape").exists()
        assert not list(default_agents_root().glob("avatar.*"))

    def test_a_kept_picture_cannot_be_taken_for_another_clone(self, tmp_path: Path) -> None:
        """`writer.prev.png` is writer's kept picture, never a clone named `writer.prev` (#1772).

        The loader refuses a name with a dot, so no clone can be named after a kept copy.
        Since 2026-09-27 a workspace file is imported rather than loaded in place, and a
        refused one is reported, not raised: the registry simply has no such clone.

        Killed by: src/uclone_x/agent/persona_store.py :: refuse_an_unusable_username(persona_name)
        Becomes: pass
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        store.set("surveyor", PNG)
        store.set("surveyor", JPEG)
        assert store.previous("surveyor") == _home("surveyor") / "avatar.prev.png"

        definition = _install(tmp_path, "surveyor.prev")
        with pytest.raises(PersonaLoadError, match="surveyor.prev"):
            persona_from_mapping(yaml.safe_load(definition.read_text()), definition)
        assert PersonaRegistry(workspace_root=tmp_path).get_persona("surveyor.prev") is None

    def test_the_url_carries_the_version_and_is_null_without_a_picture(
        self, tmp_path: Path
    ) -> None:
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)

        assert avatar_url("surveyor", store.find("surveyor")) is None
        store.set("surveyor", PNG)
        record = store.find("surveyor")
        assert record is not None
        assert avatar_url("surveyor", record) == f"/api/clones/surveyor/avatar?v={record.version}"


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
        assert not (_home("surveyor") / "avatar.png").exists()

    def test_the_replaced_picture_is_kept_one_deep(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/persona_avatar.py :: replaced = self._keep_previous(name, current[0]) if current else None
        Becomes: replaced = None
        Killed by: src/uclone_x/agent/persona_avatar.py :: if older != kept:
        Becomes: if False:
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        store.set("surveyor", PNG)
        store.set("surveyor", JPEG)
        assert store.previous("surveyor") == _home("surveyor") / "avatar.prev.png"

        store.set("surveyor", GIF)

        kept = sorted(p.name for p in _home("surveyor").glob("avatar.prev.*"))
        assert kept == ["avatar.prev.jpg"]
        assert (_home("surveyor") / "avatar.prev.jpg").read_bytes() == JPEG

    def test_the_version_changes_when_the_picture_does(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/persona_avatar.py :: version=f"{stat.st_mtime_ns}-{stat.st_size}"
        Becomes: version="v"
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)

        store.set("surveyor", PNG)
        first = store.find("surveyor")
        store.set("surveyor", PNG + b"\x00")
        second = store.find("surveyor")
        assert first is not None and second is not None

        assert first.version != second.version

    def test_a_name_no_clone_carries_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(AvatarPersonaNotFound):
            _store(tmp_path).set("nobody", PNG)
        assert not (tmp_path / PERSONAS / "nobody.png").exists()


def _files(folder: Path) -> dict[str, bytes]:
    """Every file in the personas folder, by name, for "nothing on disk changed"."""
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}


class TestChangeOrder:
    def test_each_change_gets_the_next_id(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/persona_avatar.py :: change_id = latest + 1
        Becomes: change_id = 1
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)

        ids = [
            store.set("surveyor", PNG).change_id,
            store.set("surveyor", JPEG).change_id,
            store.reset("surveyor").change_id,
            store.reset("surveyor").change_id,  # nothing to put aside, still a change
        ]

        assert ids == [1, 2, 3, 4]
        assert store.latest_change("surveyor") == 4

    def test_the_count_survives_a_new_store(self, tmp_path: Path) -> None:
        """A restart builds a new store; the ids go on rather than starting again.

        Killed by: src/uclone_x/agent/persona_avatar.py :: replace_file(_change_file(folder),
        Becomes: (_change_file(folder),
        """
        _install(tmp_path, "surveyor")
        _store(tmp_path).set("surveyor", PNG)
        _store(tmp_path).set("surveyor", JPEG)

        after_restart = _store(tmp_path)

        assert after_restart.latest_change("surveyor") == 2
        assert after_restart.set("surveyor", GIF).change_id == 3

    def test_a_change_that_fails_part_way_still_takes_its_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The id is counted before the picture is written, so a write that fails (or a crash)
        after `.prev` moved cannot leave an older Undo matching a picture it no longer describes.

        Killed by: src/uclone_x/agent/persona_avatar.py :: replace_file(_change_file(folder),
        Becomes: (_change_file(folder),
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        store.set("surveyor", PNG)
        held = store.set("surveyor", JPEG)  # the Undo a tab is holding
        real_replace = persona_avatar.replace_file

        def picture_write_fails(path: Path, data: bytes) -> None:
            if not path.name.startswith("."):
                raise OSError("disk full")
            real_replace(path, data)

        monkeypatch.setattr(persona_avatar, "replace_file", picture_write_fails)
        with pytest.raises(OSError):
            store.set("surveyor", GIF)
        monkeypatch.undo()

        assert store.latest_change("surveyor") == held.change_id + 1
        with pytest.raises(AvatarStaleChange):
            store.set("surveyor", PNG, undo_of=held.change_id)

    def test_an_undo_after_a_b_c_b_is_refused_and_writes_nothing(self, tmp_path: Path) -> None:
        """The picture is B again, but the change that first made it B is not the latest.

        Called on `set` directly, not `set_from_path`, so it is the check under the lock
        that refuses.

        Killed by: src/uclone_x/agent/persona_avatar.py :: if undo_of is not None and (undo_of < 1 or undo_of != latest):
        Becomes: if False:
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        store.set("surveyor", PNG)  # A, change 1
        first_b = store.set("surveyor", JPEG)  # B, change 2
        store.set("surveyor", GIF)  # C, change 3
        store.set("surveyor", JPEG)  # B again, change 4
        folder = tmp_path / PERSONAS
        before = _files(folder)

        with pytest.raises(AvatarStaleChange) as refused:
            store.set("surveyor", PNG, undo_of=first_b.change_id)
        with pytest.raises(AvatarStaleChange):
            store.reset("surveyor", undo_of=first_b.change_id)

        assert refused.value.reason_code == "stale_change"
        assert _files(folder) == before
        assert store.latest_change("surveyor") == 4

    def test_an_undo_of_the_latest_change_is_made(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/persona_avatar.py :: or undo_of != latest):
        Becomes: or undo_of == latest):
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        store.set("surveyor", PNG)
        replaced = store.set("surveyor", JPEG)
        assert replaced.previous is not None

        undone = store.set_from_path("surveyor", replaced.previous, undo_of=replaced.change_id)

        assert undone.change_id == 3
        found = store.find("surveyor")
        assert found is not None and found.path.read_bytes() == PNG

    def test_a_stale_undo_whose_kept_copy_is_gone_says_stale_not_missing(
        self, tmp_path: Path
    ) -> None:
        """A PNG, then a JPEG, then a GIF: the kept `.prev.png` is replaced by `.prev.jpg`.

        The Undo of the JPEG change still names `.prev.png`; it must be refused for what it
        is -- a later change -- not as a missing file.

        Killed by: src/uclone_x/agent/persona_avatar.py :: if undo_of is not None and undo_of != self.latest_change(name):
        Becomes: if False:
        """
        _install(tmp_path, "surveyor")
        store = _store(tmp_path)
        store.set("surveyor", PNG)
        to_jpeg = store.set("surveyor", JPEG)
        store.set("surveyor", GIF)
        assert to_jpeg.previous is not None and not to_jpeg.previous.exists()

        with pytest.raises(AvatarStaleChange):
            store.set_from_path("surveyor", to_jpeg.previous, undo_of=to_jpeg.change_id)


class TestReset:
    def test_reset_puts_the_chosen_picture_aside_and_the_shipped_one_shows(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/agent/persona_avatar.py :: kept = self._keep_previous(name, current[0])
        Becomes: kept = None
        """
        store = _store(tmp_path)
        store.set("artist", JPEG)

        kept = store.reset("artist").previous

        assert kept == _home("artist") / "avatar.prev.jpg"
        assert kept is not None and kept.read_bytes() == JPEG
        found = store.find("artist")
        assert found is not None
        assert found.path == BUILTIN_PERSONAS_DIR / "artist.png"

    def test_reset_with_nothing_chosen_changes_nothing(self, tmp_path: Path) -> None:
        assert _store(tmp_path).reset("artist").previous is None


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
