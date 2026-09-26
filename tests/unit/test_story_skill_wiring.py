"""Skill-supplied story data, and the genre a draw takes from the open story (#1572).

A skill package carries story data at `<skill>/resources/story/{muse,structures}/<id>.yaml`.
The agent puts its active skills' folders on every tool call (`ToolContext.skill_dirs`), and
`muse_spark` and `story_outline` read them on every call, after the bundled data, so a skill
adds a genre or a structure, or replaces one by id, and the result names where it came from.

What these pin:

* **The agent hands its active skills' folders to the tools**, and a skill's table is drawn.
* **A skill adds and replaces by id**, in skill folder-name order, and the result says whose it is.
* **A damaged skill file is refused naming the skill**, in plain words.
* **`story_outline init` builds from a structure**, bundled or a skill's.
* **A draw with no genre uses the open story's and says so**; with neither, it is refused.
"""

from __future__ import annotations

import asyncio
import gc
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ToolCallRequest
from uclone_x.skills.auditor import Skill, SkillRegistry
from uclone_x.skills.models import (
    AuditVerdict,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.story.muse import MuseSparkTool
from uclone_x.story.skill_data import MAX_DATA_FILE_BYTES, data_roots
from uclone_x.story.tool import StoryLibraryTool
from uclone_x.story.tools import StoryOutlineTool
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import NoIsolation, ToolContext, ToolResult
from uclone_x.tools.registry import ToolRegistry

ROOM = "room_a"

_WESTERN = (
    "genre: western\ntitle: Western\nslots:\n"
    "  - slot: character\n    entries: [a drifter, a sheriff]\n"
    "  - slot: place\n    entries: [a ghost town, a river crossing]\n"
)
_HEIST = """\
id: heist
title: Heist
description: Plan, crew, job, twist.
acts:
  - id: setup
    title: The Plan
    beats:
      - id: the-mark
        title: The Mark
        purpose: Show what is to be taken and why it cannot be.
      - id: the-crew
        title: The Crew
        purpose: Gather the people who can take it.
  - id: job
    title: The Job
    beats:
      - id: the-turn
        title: The Turn
        purpose: The plan breaks, and the real plan shows.
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _skill(root: Path, name: str, files: dict[str, str]) -> Path:
    """A skill package folder whose `resources/story/` holds `files`."""
    folder = root / name
    _write(folder / "SKILL.md", f"---\nname: {name}\n---\n")
    for relative, text in files.items():
        _write(folder / "resources" / "story" / relative, text)
    return folder


def _ctx(
    workspace: Path,
    *,
    skills: tuple[Path, ...] = (),
    room: str | None = ROOM,
    story: str | None = None,
) -> ToolContext:
    return ToolContext(
        agent_id="writer",
        session_id="s1",
        workspace_root=workspace,
        skill_dirs=skills,
        room_id=room,
        story_id=story,
        isolation=NoIsolation(),
    )


def _run(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> ToolResult:
    return asyncio.run(tool.execute(args, ctx))


def _ok(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> dict[str, Any]:
    result = _run(tool, ctx, **args)
    assert result.success, result.error
    assert isinstance(result.output, dict)
    return result.output


def _story(workspace: Path, **extra: Any) -> str:
    out = _ok(StoryLibraryTool(), _ctx(workspace), action="create", title="The Salt Road", **extra)
    story_id = out["open_story_id"]
    assert isinstance(story_id, str)
    return story_id


# --- a skill's muse tables ----------------------------------------------------------------


def test_a_skill_adds_a_genre_that_is_listed_and_drawn_with_its_source(tmp_path: Path) -> None:
    """A table under the skill's `resources/story/muse/` is a genre like a bundled one.

    Killed by: src/uclone_x/story/muse.py :: data_roots(context.skill_dirs, self._data_roots)
    Becomes: data_roots((), self._data_roots)
    """
    skill = _skill(tmp_path / "skills", "weird-west", {"muse/western.yaml": _WESTERN})
    ctx = _ctx(tmp_path, skills=(skill,))
    tool = MuseSparkTool()

    genres = _ok(tool, ctx, action="genres")["genres"]
    assert {"genre": "western", "title": "Western", "source": "skill 'weird-west'"} in genres
    drawn = _ok(tool, ctx, action="draw", genre="Western", seed=3)
    assert (drawn["genre"], drawn["source"]) == ("western", "skill 'weird-west'")
    assert set(drawn["cards"][0]) == {"character", "place"}


def test_a_skill_replaces_a_bundled_table_by_its_genre(tmp_path: Path) -> None:
    """Same id replaces: the skill's `fantasy` is drawn, and the result says whose it is.

    Killed by: src/uclone_x/story/skill_data.py :: return (*base, *found)
    Becomes: return (*found, *base)
    """
    fantasy = _WESTERN.replace("genre: western", "genre: fantasy").replace("Western", "Mine")
    skill = _skill(tmp_path / "skills", "my-fantasy", {"muse/fantasy.yaml": fantasy})

    drawn = _ok(MuseSparkTool(), _ctx(tmp_path, skills=(skill,)), action="draw", genre="fantasy")
    assert drawn["source"] == "skill 'my-fantasy'"
    assert set(drawn["cards"][0]) == {"character", "place"}


def test_two_skills_with_one_genre_resolve_by_skill_name_not_by_order(tmp_path: Path) -> None:
    """Which skill wins does not depend on the order the registry listed them in.

    Killed by: src/uclone_x/story/skill_data.py :: key=lambda root: root.parents[len(SKILL_DATA_PARTS) - 1].name,
    Becomes: key=lambda root: 0,
    """
    first = _skill(tmp_path / "skills", "aaa-west", {"muse/western.yaml": _WESTERN})
    last = _skill(tmp_path / "skills", "zzz-west", {"muse/western.yaml": _WESTERN})
    tool = MuseSparkTool()

    for order in ((first, last), (last, first)):
        drawn = _ok(tool, _ctx(tmp_path, skills=order), action="draw", genre="western")
        assert drawn["source"] == "skill 'zzz-west'"


def test_a_skill_without_story_data_adds_nothing_and_is_not_an_error(tmp_path: Path) -> None:
    """Most skills are not about stories; their folders are passed all the same.

    Killed by: src/uclone_x/story/skill_data.py :: if _is_folder(d.joinpath(*SKILL_DATA_PARTS), d.joinpath(*SKILL_DATA_PARTS))
    Becomes: if True
    """
    unrelated = tmp_path / "skills" / "sql-helper"
    _write(unrelated / "SKILL.md", "---\nname: sql-helper\n---\n")

    assert data_roots((unrelated,)) == data_roots()
    drawn = _ok(MuseSparkTool(), _ctx(tmp_path, skills=(unrelated,)), action="draw", genre="horror")
    assert drawn["source"] == "bundled"


def test_a_damaged_skill_table_is_refused_naming_the_skill_in_plain_words(tmp_path: Path) -> None:
    """The person fixing it needs the skill and the file in it, not a path or a class name.

    Killed by: src/uclone_x/story/skill_data.py :: where = f"{where} of the skill '{skill}'"
    Becomes: where = f"{where}"
    """
    skill = _skill(
        tmp_path / "skills",
        "weird-west",
        {"muse/western.yaml": _WESTERN.replace("[a drifter, a sheriff]", "[a drifter]")},
    )
    result = _run(MuseSparkTool(), _ctx(tmp_path, skills=(skill,)), action="draw", genre="horror")

    assert not result.success
    assert result.error == (
        "The idea tables could not be loaded, so nothing was drawn: the field "
        "'slots[0].entries' in muse/western.yaml of the skill 'weird-west' is missing or "
        "holds a value that does not fit."
    )


_REFUSED = "The idea tables could not be loaded, so nothing was drawn: "


def _refusal(tmp_path: Path, skill: Path) -> str | None:
    result = _run(MuseSparkTool(), _ctx(tmp_path, skills=(skill,)), action="draw", genre="horror")
    assert not result.success
    return result.error


def test_a_story_folder_linked_out_is_refused_even_when_its_files_link_back(
    tmp_path: Path,
) -> None:
    """`resources/story` leads out; its file links back to a dot-file the hash never counts.

    The file resolves inside the skill, so only the check on the folder refuses it.

    Killed by: src/uclone_x/story/skill_data.py :: _require_inside(home, root, root)
    Becomes: None
    """
    skill = _skill(tmp_path / "skills", "weird-west", {})
    hidden = _write(skill / ".hidden.yaml", _WESTERN)
    outside = tmp_path / "outside"
    (outside / "muse").mkdir(parents=True)
    (outside / "muse" / "western.yaml").symlink_to(hidden)
    (skill / "resources").mkdir()
    (skill / "resources" / "story").symlink_to(outside, target_is_directory=True)

    assert _refusal(tmp_path, skill) == (
        f"{_REFUSED}resources/story of the skill 'weird-west' leads out of the skill's "
        "folder through a link, which is not allowed."
    )


def test_a_muse_folder_linked_out_is_refused_even_when_its_files_link_back(
    tmp_path: Path,
) -> None:
    """The same out-and-back, one folder down: `muse` itself leads out.

    Killed by: src/uclone_x/story/skill_data.py :: _require_inside(home, root, directory)
    Becomes: None
    """
    skill = _skill(tmp_path / "skills", "weird-west", {"structures/.keep": ""})
    hidden = _write(skill / ".hidden.yaml", _WESTERN)
    outside = tmp_path / "outside" / "muse"
    outside.mkdir(parents=True)
    (outside / "western.yaml").symlink_to(hidden)
    (skill / "resources" / "story" / "muse").symlink_to(outside, target_is_directory=True)

    assert _refusal(tmp_path, skill) == (
        f"{_REFUSED}muse of the skill 'weird-west' leads out of the skill's "
        "folder through a link, which is not allowed."
    )


def test_a_skills_data_file_linked_from_outside_it_is_refused(tmp_path: Path) -> None:
    """A link to one file is caught too: the check is on where the file resolves, not its name.

    Killed by: src/uclone_x/story/skill_data.py :: PathValidator().resolve_safe_path(path.absolute(), home)
    Becomes: PathValidator().resolve_safe_path(path.parent.absolute(), home)
    """
    outside = _write(tmp_path / "outside" / "western.yaml", _WESTERN)
    skill = _skill(tmp_path / "skills", "weird-west", {})
    link = skill / "resources" / "story" / "muse" / "western.yaml"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    assert _refusal(tmp_path, skill) == (
        f"{_REFUSED}muse/western.yaml of the skill 'weird-west' leads out of the skill's "
        "folder through a link, which is not allowed."
    )


def test_a_link_that_stays_inside_the_skill_is_read(tmp_path: Path) -> None:
    """A link that resolves inside the skill's folder is not refused."""
    skill = _skill(tmp_path / "skills", "weird-west", {})
    real = _write(skill / "tables" / "western.yaml", _WESTERN)
    link = skill / "resources" / "story" / "muse" / "western.yaml"
    link.parent.mkdir(parents=True)
    link.symlink_to(real)

    drawn = _ok(MuseSparkTool(), _ctx(tmp_path, skills=(skill,)), action="draw", genre="western")
    assert drawn["source"] == "skill 'weird-west'"


def test_an_oversized_skill_table_is_refused_before_it_is_read(tmp_path: Path) -> None:
    """One file a skill carries cannot fill the prompt.

    Killed by: src/uclone_x/story/skill_data.py :: if len(data) > MAX_DATA_FILE_BYTES:
    Becomes: if False:
    """
    padding = "# " + "x" * MAX_DATA_FILE_BYTES + "\n"
    skill = _skill(tmp_path / "skills", "weird-west", {"muse/western.yaml": _WESTERN + padding})

    assert _refusal(tmp_path, skill) == (
        f"{_REFUSED}muse/western.yaml of the skill 'weird-west' is larger than 256 KB."
    )


def test_a_skill_table_that_is_a_fifo_is_refused_without_hanging(tmp_path: Path) -> None:
    """Only a regular file is read: a FIFO would block a plain open until a writer came.

    The call runs in a thread with a deadline, so a regression fails instead of hanging.

    Killed by: src/uclone_x/story/skill_data.py :: if not stat.S_ISREG(os.fstat(fd).st_mode):
    Becomes: if False:
    """
    skill = _skill(tmp_path / "skills", "weird-west", {})
    fifo = skill / "resources" / "story" / "muse" / "western.yaml"
    fifo.parent.mkdir(parents=True)
    os.mkfifo(fifo)
    errors: list[str | None] = []
    call = threading.Thread(target=lambda: errors.append(_refusal(tmp_path, skill)), daemon=True)
    call.start()
    try:
        call.join(timeout=10)
        assert not call.is_alive(), "reading a FIFO blocked"
    finally:
        if call.is_alive():
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            call.join(timeout=10)

    assert errors == [
        f"{_REFUSED}muse/western.yaml of the skill 'weird-west' is not a regular file."
    ]


def test_a_folder_named_like_a_table_is_refused_and_leaves_no_descriptor_open(
    tmp_path: Path,
) -> None:
    """A folder opens like a file; its descriptor is closed after the refusal, every call.

    Killed by: src/uclone_x/story/skill_data.py :: os.close(fd)
    Becomes: pass

    The file is opened by walking down from the skill's folder (#1604); the folder
    descriptors that walk holds are closed too.

    Killed by: src/uclone_x/story/skill_data.py :: os.close(folder_fd)
    Becomes: pass

    Killed by: src/uclone_x/story/skill_data.py :: os.close(above)
    Becomes: pass
    """
    if not Path("/dev/fd").is_dir():
        pytest.skip("counts descriptors through /dev/fd")
    skill = _skill(tmp_path / "skills", "weird-west", {})
    (skill / "resources" / "story" / "muse" / "western.yaml").mkdir(parents=True)

    assert _refusal(tmp_path, skill) == (
        f"{_REFUSED}muse/western.yaml of the skill 'weird-west' is not a regular file."
    )
    gc.collect()  # a file an earlier test dropped must not close mid-count
    before = len(os.listdir("/dev/fd"))
    for _ in range(20):
        _refusal(tmp_path, skill)
    assert len(os.listdir("/dev/fd")) == before


@pytest.mark.parametrize(
    ("relative", "call", "refused"),
    [
        ("muse/western.yaml", {"action": "draw", "genre": "horror"}, _REFUSED),
        (
            "structures/loop.yaml",
            {"action": "structures"},
            "The story structures could not be loaded, so the outline was not changed: ",
        ),
    ],
)
def test_a_data_file_that_links_to_itself_is_refused_in_plain_words(
    tmp_path: Path, relative: str, call: dict[str, str], refused: str
) -> None:
    """Python 3.11 and 3.12 raise `RuntimeError` resolving a loop; it must not reach the result.

    On 3.13 `resolve()` does not raise and the open refuses it with the same words, so the
    declaration below is killed only where `resolve()` raises.

    Killed by: src/uclone_x/story/skill_data.py :: except (RuntimeError, OSError) as exc:  # a link that loops
    Becomes: except ZeroDivisionError as exc:  # a link that loops
    """
    skill = _skill(tmp_path / "skills", "weird-west", {})
    loop = skill / "resources" / "story" / relative
    loop.parent.mkdir(parents=True)
    loop.symlink_to(loop.name)
    tool: BaseTool[Any] = MuseSparkTool() if relative.startswith("muse") else StoryOutlineTool()

    result = _run(tool, _ctx(tmp_path, skills=(skill,), story=_story(tmp_path)), **call)

    assert not result.success
    error = result.error or ""
    assert error == f"{refused}{relative} of the skill 'weird-west' could not be read."
    assert str(tmp_path) not in error
    assert "Error" not in error


@pytest.mark.parametrize("as_on_314", [False, True], ids=["this-python", "as-on-3.14"])
def test_a_story_folder_that_cannot_be_searched_is_refused_in_plain_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, as_on_314: bool
) -> None:
    """A folder under one the process may not search is refused, not taken as missing.

    Python 3.11 to 3.13's `is_dir()` raised `PermissionError` there; 3.14's answers False
    for every error, which would let the skill silently supply nothing. The loader stats
    the folder itself now (#1604). `as-on-3.14` gives `Path.is_dir` its 3.14 behaviour, so
    the 3.14 case runs on every version.

    Killed by: src/uclone_x/story/skill_data.py :: raise _unreadable(root, path, exc) from exc  # stat refused
    Becomes: return False  # stat refused

    Killed by: src/uclone_x/story/skill_data.py :: mode = os.stat(path).st_mode
    Becomes: mode = stat.S_IFDIR if path.is_dir() else 0
    """
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("root searches any folder, and the test needs one it cannot")
    if as_on_314:
        is_dir = Path.is_dir

        def is_dir_as_on_314(self: Path) -> bool:
            try:
                return is_dir(self)
            except OSError:
                return False

        monkeypatch.setattr(Path, "is_dir", is_dir_as_on_314)
    skill = _skill(tmp_path / "skills", "weird-west", {"muse/western.yaml": _WESTERN})
    (skill / "resources").chmod(0o600)
    try:
        error = _refusal(tmp_path, skill)
    finally:
        (skill / "resources").chmod(0o755)

    assert error == f"{_REFUSED}resources/story of the skill 'weird-west' could not be read."


@pytest.mark.parametrize(
    ("folder", "call", "refused"),
    [
        ("muse", {"action": "draw", "genre": "horror"}, _REFUSED),
        (
            "structures",
            {"action": "structures"},
            "The story structures could not be loaded, so the outline was not changed: ",
        ),
    ],
)
def test_a_data_folder_that_cannot_be_listed_is_refused_not_skipped(
    tmp_path: Path, folder: str, call: dict[str, str], refused: str
) -> None:
    """`glob` answered nothing for a mode-000 folder, so the skill silently supplied nothing.

    Killed by: src/uclone_x/story/skill_data.py :: except OSError as exc:  # listing refused
    Becomes: except ZeroDivisionError as exc:  # listing refused
    """
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("root lists any folder, and the test needs one it cannot")
    table = "muse/western.yaml" if folder == "muse" else "structures/heist.yaml"
    skill = _skill(
        tmp_path / "skills", "weird-west", {table: _WESTERN if folder == "muse" else _HEIST}
    )
    locked = skill / "resources" / "story" / folder
    tool: BaseTool[Any] = MuseSparkTool() if folder == "muse" else StoryOutlineTool()
    ctx = _ctx(tmp_path, skills=(skill,), story=_story(tmp_path))
    locked.chmod(0o000)
    try:
        result = _run(tool, ctx, **call)
    finally:
        locked.chmod(0o755)

    assert not result.success
    assert result.error == f"{refused}{folder} of the skill 'weird-west' could not be read."


def test_a_data_folder_that_is_a_link_that_loops_supplies_nothing(tmp_path: Path) -> None:
    """A `muse` folder that is a looping link is not a folder, so the skill adds no tables.

    Killed by: src/uclone_x/story/skill_data.py :: if not _is_folder(directory, root):
    Becomes: if False:
    """
    skill = _skill(tmp_path / "skills", "weird-west", {"structures/heist.yaml": _HEIST})
    loop = skill / "resources" / "story" / "muse"
    loop.symlink_to(loop.name)
    ctx = _ctx(tmp_path, skills=(skill,), story=_story(tmp_path))

    drawn = _ok(MuseSparkTool(), ctx, action="draw", genre="horror")
    genres = _ok(MuseSparkTool(), ctx, action="genres")["genres"]
    structures = _ok(StoryOutlineTool(), ctx, action="structures")["structures"]

    # The looping `muse` supplies no table, and the rest of the skill still counts (#1604).
    assert drawn["source"] == "bundled"
    assert [g["genre"] for g in genres if g["source"] != "bundled"] == []
    assert [s["id"] for s in structures if s["source"] == "skill 'weird-west'"] == ["heist"]


def test_a_data_folder_swapped_for_a_link_after_the_check_is_not_followed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is opened folder by folder from the skill, so a swap after the check fails.

    `O_NOFOLLOW` covers only a path's last part. A process that could write in the skill
    folder could swap `muse` for a link out between the containment check and the open,
    and the old open by path followed it (#1604). The swap is made inside that window here.

    Killed by: src/uclone_x/story/skill_data.py :: walk_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    Becomes: walk_flags = os.O_RDONLY | os.O_DIRECTORY

    Killed by: src/uclone_x/story/skill_data.py :: fd = os.open(read_from, flags) if beneath is None else _open_beneath(beneath, read_from, flags)
    Becomes: fd = os.open(read_from, flags)
    """
    from uclone_x.story import skill_data

    outside = _write(
        tmp_path / "outside" / "western.yaml", _WESTERN.replace("a drifter", "from outside")
    )
    skill = _skill(tmp_path / "skills", "weird-west", {"muse/western.yaml": _WESTERN})
    muse = skill / "resources" / "story" / "muse"
    checked = skill_data._require_inside  # pyright: ignore[reportPrivateUsage]

    def check_then_swap(home: Path, root: Path, path: Path) -> Path:
        resolved = checked(home, root, path)
        if path.name == "western.yaml":
            muse.rename(muse.with_name("muse-before"))
            muse.symlink_to(outside.parent, target_is_directory=True)
        return resolved

    monkeypatch.setattr(skill_data, "_require_inside", check_then_swap)

    result = _run(MuseSparkTool(), _ctx(tmp_path, skills=(skill,)), action="draw", genre="western")

    assert not result.success
    assert result.error == (
        f"{_REFUSED}muse/western.yaml of the skill 'weird-west' could not be read."
    )


def test_a_skill_table_that_repeats_itself_by_alias_is_refused(tmp_path: Path) -> None:
    """An alias is how a few lines of YAML expand to millions of entries; none is needed here.

    Killed by: src/uclone_x/story/skill_data.py :: if _uses_alias(text):
    Becomes: if False:
    """
    aliased = (
        "genre: western\ntitle: Western\nslots:\n"
        "  - slot: character\n    entries: &people [a drifter, a sheriff]\n"
        "  - slot: rival\n    entries: *people\n"
    )
    skill = _skill(tmp_path / "skills", "weird-west", {"muse/western.yaml": aliased})

    assert _refusal(tmp_path, skill) == (
        f"{_REFUSED}muse/western.yaml of the skill 'weird-west' repeats part of itself with "
        "a YAML alias ('*'), which is not allowed."
    )


# --- the agent hands the tools its skills -------------------------------------------------


def _registry_with(skill_dir: Path, *, status: SkillStatus = SkillStatus.ACTIVE) -> SkillRegistry:
    name = skill_dir.name
    manifest = SkillManifest(
        name=name,
        description="Western idea tables.",
        origin=SkillOrigin.HUMAN,
        status=status,
        content_sha256=f"sha_{name}",
    )
    registry = SkillRegistry()
    registry.register(
        Skill(manifest=manifest, instructions_markdown="", directory=skill_dir),
        SkillAuditReport(
            skill_name=name,
            is_safe=True,
            recommendation=AuditVerdict.APPROVE,
            content_sha256=f"sha_{name}",
        ),
    )
    return registry


def test_the_agent_puts_its_active_skills_folders_on_every_tool_call(tmp_path: Path) -> None:
    """End to end: a draw the model makes through the agent reaches the skill's table.

    Killed by: src/uclone_x/agent/base.py :: skill_dirs=self.active_skill_dirs(),
    Becomes: skill_dirs=(),
    """
    skill = _skill(tmp_path / "skills", "weird-west", {"muse/western.yaml": _WESTERN})
    tools = ToolRegistry()
    tools.register(MuseSparkTool())
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="writer", name="Writer", llm_config=AgentLLMConfig(model_name="mock")
        ),
        llm=MockLLMConnector(
            responses=["", "done"],
            tool_calls=[
                ToolCallRequest(
                    id="m1",
                    name="muse_spark",
                    arguments={"action": "draw", "genre": "western", "seed": 1},
                )
            ],
        ),
        tools=tools,
        skills=_registry_with(skill),
    )

    result = asyncio.run(agent.execute_turn("give me an idea"))
    [execution] = result.tool_executions
    assert execution.status == "success", execution
    assert isinstance(execution.output, dict)
    assert execution.output["source"] == "skill 'weird-west'"


def test_a_skill_that_is_not_active_supplies_nothing(tmp_path: Path) -> None:
    """Only an approved skill's data is read; a pending package is quarantined (P9).

    Killed by: src/uclone_x/agent/base.py :: if skill.manifest.status != SkillStatus.ACTIVE:
    Becomes: if False:
    """
    active = _skill(tmp_path / "skills", "weird-west", {"muse/western.yaml": _WESTERN})
    pending = _skill(tmp_path / "skills", "pending-west", {"muse/western.yaml": _WESTERN})
    registry = _registry_with(active)
    pending_manifest = SkillManifest(
        name="pending-west",
        description="Not approved.",
        origin=SkillOrigin.SYNTHESIZED,
        status=SkillStatus.PENDING,
        content_sha256="sha_p",
    )
    # `register` admits on the report; the manifest's own status is what quarantines it.
    registry.register(
        Skill(manifest=pending_manifest, instructions_markdown="", directory=pending),
        SkillAuditReport(
            skill_name="pending-west",
            is_safe=True,
            recommendation=AuditVerdict.APPROVE,
            content_sha256="sha_p",
        ),
    )
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="writer", name="Writer", llm_config=AgentLLMConfig(model_name="mock")
        ),
        skills=registry,
    )

    assert agent.active_skill_dirs() == (active,)


# --- story_outline builds from a structure ------------------------------------------------


def test_init_from_a_bundled_structure_lays_out_acts_as_chapters_and_beats_as_scenes(
    tmp_path: Path,
) -> None:
    """Each scene is a beat to be rewritten: its title, and what it is for as the summary.

    Killed by: src/uclone_x/story/tools.py :: "summary": beat.purpose,
    Becomes: "summary": "",
    """
    story = _story(tmp_path)
    ctx = _ctx(tmp_path, story=story)
    outline = StoryOutlineTool()

    created = _ok(outline, ctx, action="init", structure="Kishotenketsu")
    assert (created["structure"], created["source"]) == ("kishotenketsu", "bundled")
    assert (created["chapters"], created["scenes"]) == (4, 4)

    chapters = _ok(outline, ctx, action="get")["chapters"]
    assert [(c["id"], c["title"], c["act"]) for c in chapters] == [
        ("ch01", "Introduction", "ki"),
        ("ch02", "Development", "sho"),
        ("ch03", "Twist", "ten"),
        ("ch04", "Reconciliation", "ketsu"),
    ]
    assert chapters[2]["scenes"] == [
        {
            "id": "ch03.s01",
            "title": "Twist",
            "summary": "An unexpected turn or new element that recasts what came before.",
            "written": False,
        }
    ]


def test_init_from_a_skills_structure_and_the_list_names_it(tmp_path: Path) -> None:
    """A structure a skill adds is listed with its source and builds an outline.

    Killed by: src/uclone_x/story/tools.py :: return load_structure_templates_sourced(data_roots(context.skill_dirs))
    Becomes: return load_structure_templates_sourced(data_roots(()))
    """
    skill = _skill(tmp_path / "skills", "heists", {"structures/heist.yaml": _HEIST})
    story = _story(tmp_path)
    ctx = _ctx(tmp_path, skills=(skill,), story=story)
    outline = StoryOutlineTool()

    listed = _ok(outline, ctx, action="structures")["structures"]
    assert {
        "id": "heist",
        "title": "Heist",
        "description": "Plan, crew, job, twist.",
        "beats": 3,
        "source": "skill 'heists'",
    } in listed
    assert {s["id"] for s in listed} >= {"three-act", "save-the-cat"}

    created = _ok(outline, ctx, action="init", structure="heist")
    assert (created["chapters"], created["scenes"], created["source"]) == (
        2,
        3,
        "skill 'heists'",
    )
    chapters = _ok(outline, ctx, action="get")["chapters"]
    assert [s["title"] for s in chapters[0]["scenes"]] == ["The Mark", "The Crew"]


def test_an_unknown_structure_is_refused_naming_the_structures_there_are(tmp_path: Path) -> None:
    """P6: no default structure stands in, and nothing is written.

    Killed by: src/uclone_x/story/tools.py :: found = templates.get(_structure_id(requested))
    Becomes: found = templates.get(_structure_id(requested)) or next(iter(templates.values()))
    """
    story = _story(tmp_path)
    ctx = _ctx(tmp_path, story=story)

    result = _run(StoryOutlineTool(), ctx, action="init", structure="five-act")
    assert result.error == (
        "There is no structure 'five-act', so the outline was not created. "
        "Structures: heros-journey, kishotenketsu, save-the-cat, three-act."
    )
    assert not (tmp_path / "stories" / story / "outline.yaml").exists()


def test_init_with_both_titles_and_a_structure_is_refused(tmp_path: Path) -> None:
    """Two answers to one question: neither is picked silently.

    Killed by: src/uclone_x/story/tools.py :: if params.chapter_titles:
    Becomes: if False:
    """
    ctx = _ctx(tmp_path, story=_story(tmp_path))
    result = _run(
        StoryOutlineTool(), ctx, action="init", structure="three-act", chapter_titles=["One"]
    )
    assert result.error == (
        "Give either 'chapter_titles' or a 'structure', not both, so the outline was not created."
    )


# --- the genre a draw takes from the open story -------------------------------------------


def test_a_draw_without_a_genre_uses_the_open_storys_and_says_so(tmp_path: Path) -> None:
    """P6: the default is stated in the result, not applied silently.

    Killed by: src/uclone_x/story/muse.py :: result["note"] = note
    Becomes: pass
    """
    story = _story(tmp_path, genre="Science Fiction")
    drawn = _ok(MuseSparkTool(), _ctx(tmp_path, story=story), action="draw", seed=9)

    assert drawn["genre"] == "science-fiction"
    assert drawn["note"] == (
        "No genre was given, so the open story's genre 'Science Fiction' was used."
    )


def test_a_named_genre_carries_no_note(tmp_path: Path) -> None:
    """The note is for a genre the model did not choose, not a line on every draw.

    Killed by: src/uclone_x/story/muse.py :: if note is not None:
    Becomes: if True:
    """
    story = _story(tmp_path, genre="horror")
    drawn = _ok(
        MuseSparkTool(), _ctx(tmp_path, story=story), action="draw", genre="mystery", seed=9
    )
    assert drawn["genre"] == "mystery"
    assert "note" not in drawn


@pytest.mark.parametrize(
    ("with_story", "expected"),
    [
        (
            False,
            "Name a genre to draw from: none was given, and no story is open to take one "
            "from. Genres: fantasy, horror, mystery, romance, science-fiction.",
        ),
        (
            True,
            "Name a genre to draw from: none was given, and the open story 'The Salt Road' "
            "has no genre. Genres: fantasy, horror, mystery, romance, science-fiction.",
        ),
    ],
)
def test_a_draw_with_no_genre_anywhere_is_refused_in_plain_words(
    tmp_path: Path, with_story: bool, expected: str
) -> None:
    """Nothing is drawn from a table nobody chose.

    Killed by: src/uclone_x/story/muse.py :: if record.genre is None or not record.genre.strip():
    Becomes: if False:
    """
    story = _story(tmp_path) if with_story else None
    result = _run(MuseSparkTool(), _ctx(tmp_path, story=story), action="draw", seed=1)
    assert not result.success
    assert result.error == expected


def test_a_storys_genre_with_no_table_is_refused_saying_it_came_from_the_story(
    tmp_path: Path,
) -> None:
    """The model did not name the genre, so the refusal says whose it was.

    Killed by: src/uclone_x/story/muse.py :: whose = f"the open story's genre '{genre}'"
    Becomes: whose = f"the genre '{genre}'"
    """
    story = _story(tmp_path, genre="western")
    result = _run(MuseSparkTool(), _ctx(tmp_path, story=story), action="draw", seed=1)
    assert result.error == (
        "There is no table for the open story's genre 'western', so nothing was drawn. "
        "Genres: fantasy, horror, mystery, romance, science-fiction."
    )
