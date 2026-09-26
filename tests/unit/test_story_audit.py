"""Story time, the scene audit, and codex proposals a person applies (#1557).

What these pin, in order of what it would cost to get wrong:

* **The model cannot apply its own proposal.** `story_codex 'apply'` runs only once a person
  answers the runtime's approval request with yes: no hook, no argument and no chat message
  stands in for that, and without a way to ask, the call is refused.
* **A flashback is not a contradiction.** A scene is checked against the codex as it stands
  at that point of *story* time, so a character who dies later is alive in a flashback and
  dead in a later scene.
* **A fact is checked only with a quote the scene contains.** The model proposes the facts;
  a fact with no quote, or a quote the scene does not have, is rejected by name.
* **A scene's context applies the changes before it** in story-time order.
* **`character_sheet` over a story says what it did not show**, and invents no looks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.hooks.models import HookAction, HookContext, HookDecision
from uclone_x.agent.hooks.permission import HumanApprovalHook, PermissionMode
from uclone_x.agent.hooks.protocols import BaseHook
from uclone_x.agent.models import AgentConfig
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.llm.models import ToolCallRequest
from uclone_x.room.a2a_handlers import (
    _DeferredApprovalRunner,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.story.quotes import quote_found
from uclone_x.story.schemas import Outline
from uclone_x.story.timeline import assumptions, place_scenes, time_key
from uclone_x.story.tool import StoryLibraryTool
from uclone_x.story.tools import (
    StoryAuditTool,
    StoryCodexTool,
    StoryContextTool,
    StoryManuscriptTool,
    StoryOutlineTool,
)
from uclone_x.tools.base import BaseTool, tool_writes_files
from uclone_x.tools.builtin.character import CharacterSheetTool
from uclone_x.tools.models import NoIsolation, ToolContext, ToolResult
from uclone_x.tools.registry import ToolRegistry

ROOM = "room_a"


def _ctx(workspace: Path, *, story: str | None = None, approved: bool = False) -> ToolContext:
    return ToolContext(
        agent_id="writer",
        session_id=f"sess_room__{ROOM}__writer",
        workspace_root=workspace,
        room_id=ROOM,
        story_id=story,
        isolation=NoIsolation(),
        approved_by_person=approved,
    )


async def _call(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> ToolResult:
    return await tool.execute(args, ctx)


async def _ok(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> dict[str, Any]:
    result = await _call(tool, ctx, **args)
    assert result.success, result.error
    assert isinstance(result.output, dict)
    return result.output


async def _refused(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> str:
    result = await _call(tool, ctx, **args)
    assert not result.success, result.output
    assert result.error is not None
    return result.error


def _story_dir(workspace: Path, story_id: str) -> Path:
    return workspace / "stories" / story_id


def _codex(workspace: Path, story_id: str, kind: str, entry: dict[str, Any]) -> Path:
    path = _story_dir(workspace, story_id) / "codex" / kind / f"{entry['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(entry, allow_unicode=True), encoding="utf-8")
    return path


def _set_story_yaml(workspace: Path, story_id: str, **fields: Any) -> None:
    """Edits `story.yaml` by hand, as a person would."""
    path = _story_dir(workspace, story_id) / "story.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.update(fields)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


#: Reading order, and when each scene happens. s03 has no time: it continues s02.
_SCENES = (
    ("The gate", "day 3"),
    ("Vane falls", "day 10"),
    ("The funeral", None),
    ("Years before", "day 1"),
)

_TEXT = {
    "ch01.s01": "Mara waited at the gate while Lord Vane counted the toll.",
    "ch01.s02": "The arrow found Lord Vane's throat, and he died on the bridge.",
    "ch01.s03": "At the funeral, Lord Vane laughed at the mourners.",
    "ch01.s04": "Young Lord Vane laughed at the gate, alive and proud.",
}


async def _timed_story(workspace: Path) -> tuple[str, ToolContext]:
    """A story whose Vane dies in ch01.s02 (day 10), with a later scene and a flashback."""
    out = await _ok(StoryLibraryTool(), _ctx(workspace), action="create", title="The Salt Road")
    story_id = out["open_story_id"]
    ctx = _ctx(workspace, story=story_id)
    outline = StoryOutlineTool()
    await _ok(outline, ctx, action="init", chapter_titles=["The Crossing"])
    for title, when in _SCENES:
        extra = {"story_time": when} if when is not None else {}
        await _ok(outline, ctx, action="set_scene", chapter_id="ch01", title=title, **extra)
    for scene_id, text in _TEXT.items():
        await _ok(StoryManuscriptTool(), ctx, action="write", scene_id=scene_id, text=text)
    _codex(
        workspace,
        story_id,
        "characters",
        {
            "id": "vane",
            "name": "Lord Vane",
            "aliases": ["Vane"],
            "state": {"status": "alive"},
            "progressions": [{"at": "ch01.s02", "set": {"status": "dead"}}],
        },
    )
    return story_id, ctx


def _alive(quote: str = "Lord Vane laughed") -> dict[str, str]:
    return {"subject": "Lord Vane", "predicate": "status", "object": "alive", "quote": quote}


# --------------------------------------------------------------------------------------
# Story time
# --------------------------------------------------------------------------------------


class TestStoryTime:
    def test_a_time_reads_its_sign_and_its_dotted_parts(self) -> None:
        """#1584 item 8: `-5` is below zero; a dot separates parts, and is not a decimal point.

        Dotted values of any depth compare part by part, so the order stays total when
        depths mix (#1597 review B3), and a date written with dots is in date order.

        Killed by: src/uclone_x/story/timeline.py :: return [_number(part, signed and not index) for index, part in enumerate(parts)]
        Becomes: return [_number(parts[0], signed)]
        Killed by: src/uclone_x/story/timeline.py :: return [_number(part, signed and not index) for index, part in enumerate(parts)]
        Becomes: return [_number(part, False) for index, part in enumerate(parts)]
        Killed by: src/uclone_x/story/timeline.py :: return [_number(part, signed and not index) for index, part in enumerate(parts)]
        Becomes: return [_number(part, signed) for index, part in enumerate(parts)]
        Killed by: src/uclone_x/story/timeline.py :: if signed and match.start() > 0 and not text[match.start() - 1].isspace():
        Becomes: if False:
        """
        times = ["1.1", "1.2", "1.2.1", "1.5", "1.10", "2"]
        keys = [time_key(t) for t in times]
        assert all(earlier < later for earlier, later in zip(keys, keys[1:], strict=False))
        assert time_key("-5") < time_key("3")
        assert time_key("day -2") < time_key("day 1")
        # The sign is the first part's only: the parts after a dot still count up.
        assert time_key("-1") < time_key("-1.2") < time_key("-1.5")
        assert time_key(-1) == time_key("-1")
        # A dash inside a date is a separator, not a minus sign.
        assert time_key("2024-05-01") < time_key("2024-05-10") < time_key("2024-06-01")
        assert time_key("2024.9.30") < time_key("2024.10.01")

    def test_a_superscript_is_text_not_a_number(self) -> None:
        """#1597 review B3: `²` is a digit to `str.isdigit` but not to `int` or `\\d`.

        Killed by: src/uclone_x/story/timeline.py :: if not (token[0].isdecimal() or (token[0] == "-" and len(token) > 1)):
        Becomes: if not (token[0].isdigit() or (token[0] == "-" and len(token) > 1)):
        """
        assert time_key("²x") == ((1, "²x"),)
        assert time_key("3²x") == time_key("3") + ((1, "²x"),)
        assert time_key("day ²") < time_key("day ³")

    def test_a_number_of_any_length_is_ordered_by_its_value(self) -> None:
        """#1601: `int` refuses more than 4300 digits, and a `story_time` is free text.

        Killed by: src/uclone_x/story/timeline.py :: plain = "".join(str(unicodedata.decimal(ch)) for ch in digits).lstrip("0")
        Becomes: plain = str(int(digits)).lstrip("0")
        Killed by: src/uclone_x/story/timeline.py :: return (0, -len(plain), plain.translate(_COMPLEMENT))
        Becomes: return (0, -len(plain), plain)
        Killed by: src/uclone_x/story/timeline.py :: return (0, -len(plain), plain.translate(_COMPLEMENT))
        Becomes: return (0, len(plain), plain.translate(_COMPLEMENT))
        """
        long_times = ["-" + "9" * 5000, "-" + "1" * 5000, "0", "9" * 4999, "1" * 5000, "2" * 5000]
        keys = [time_key(t) for t in long_times]
        assert all(earlier < later for earlier, later in zip(keys, keys[1:], strict=False))
        numbers = ["-19", "-12", "-10", "-9", "-0", "01", "9", "10", "12", "19"]
        keys = [time_key(t) for t in numbers]
        assert all(earlier <= later for earlier, later in zip(keys, keys[1:], strict=False))
        assert [int(t) for t in sorted(numbers, key=time_key)] == sorted(int(t) for t in numbers)
        assert time_key("-0") == time_key("0") and time_key("01") == time_key("1")
        assert time_key("１２") == time_key("12")

    def test_times_compare_as_numbers_and_an_untimed_scene_continues_the_one_before(
        self,
    ) -> None:
        """Killed by: src/uclone_x/story/timeline.py :: key.append((0, number))
        Becomes: key.append((1, str(number)))
        Killed by: src/uclone_x/story/timeline.py :: placements[scene.id] = Placement(scene.id, index, current, inherited_from=source)
        Becomes: placements[scene.id] = Placement(scene.id, index, (), untimed=True)
        """
        assert time_key("day 3") < time_key("day 10")
        assert time_key(5) == time_key("5")
        assert time_key(None) < time_key("day 1")

        outline = Outline.model_validate(
            {
                "chapters": [
                    {
                        "id": "c",
                        "title": "C",
                        "scenes": [
                            {"id": "a", "title": "A"},
                            {"id": "b", "title": "B", "story_time": "day 10"},
                            {"id": "c", "title": "C"},
                            {"id": "d", "title": "D", "story_time": "day 3"},
                        ],
                    }
                ]
            }
        )
        placed = place_scenes(outline)
        order = sorted(placed, key=lambda s: placed[s].position)
        assert order == ["a", "d", "b", "c"]
        assert assumptions(placed) == [
            {
                "scene_id": "a",
                "placed": "first: it has no story_time and no scene before it has one",
            },
            {
                "scene_id": "c",
                "placed": "at the same time as 'b': it has no story_time of its own",
            },
        ]

    async def test_a_flashback_s_context_shows_the_entry_as_it_was_then(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/timeline.py :: elif placement.position > here.position:
        Becomes: elif False:
        """
        _, ctx = await _timed_story(tmp_path)
        context = StoryContextTool()

        later = await _ok(context, ctx, action="for_scene", scene_id="ch01.s03")
        assert [e["state"] for e in later["codex"] if e["id"] == "vane"] == [{"status": "dead"}]
        assert later["manifest"]["story_time_assumptions"] == [
            {
                "scene_id": "ch01.s03",
                "placed": "at the same time as 'ch01.s02': it has no story_time of its own",
            }
        ]

        flashback = await _ok(context, ctx, action="for_scene", scene_id="ch01.s04")
        assert [e["state"] for e in flashback["codex"] if e["id"] == "vane"] == [
            {"status": "alive"}
        ]
        assert "progressions_applied" not in flashback["manifest"]


# --------------------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------------------


class TestTheAuditChecksAgainstStoryTime:
    async def test_dead_in_a_later_scene_contradicts_and_alive_in_a_flashback_does_not(
        self, tmp_path: Path
    ) -> None:
        """The acceptance of #1557: the same fact, two scenes, two answers.

        Killed by: src/uclone_x/story/audit.py :: snapshot = entry_snapshot(item.entry, placements, scene_id, through_scene=True)
        Becomes: snapshot = entry_snapshot(item.entry, placements, "ch01.s03", through_scene=True)
        Killed by: src/uclone_x/story/audit.py :: pred, obj = "type", self.class_name(value)
        Becomes: obj = self.class_name(value)
        """
        _, ctx = await _timed_story(tmp_path)
        audit = StoryAuditTool()

        flashback = await _ok(audit, ctx, action="check", scene_id="ch01.s04", facts=[_alive()])
        assert flashback["contradictions"] == []
        assert flashback["summary"] == (
            "1 submitted fact(s) were checked against 2 rule(s): 0 contradiction(s) found."
        )

        later = await _ok(audit, ctx, action="check", scene_id="ch01.s03", facts=[_alive()])
        assert later["summary"] == (
            "1 submitted fact(s) were checked against 2 rule(s): 1 contradiction(s) found."
        )
        [contradiction] = later["contradictions"]
        assert contradiction["kind"] == "disjoint"
        assert contradiction["involves_this_scene"] is True
        sources = {
            part["fact"]: part["sources"] for part in contradiction["facts"] if "fact" in part
        }
        assert sources["vane type Alive"] == [
            {"from": "scene", "scene_id": "ch01.s03", "fact": 1, "quote": "Lord Vane laughed"}
        ]
        assert sources["vane type Dead"] == [
            {
                "from": "codex",
                "file": "codex/characters/vane.yaml",
                "field": "state.status",
                "set_by_progression_at": "ch01.s02",
            }
        ]
        assert later["story_time_assumptions"] == [
            {
                "scene_id": "ch01.s03",
                "placed": "at the same time as 'ch01.s02': it has no story_time of its own",
            }
        ]

    async def test_the_scene_s_own_change_is_in_force_when_it_is_checked(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/timeline.py :: if not through_scene:
        Becomes: if True:
        """
        _, ctx = await _timed_story(tmp_path)
        out = await _ok(
            StoryAuditTool(),
            ctx,
            action="check",
            scene_id="ch01.s02",
            facts=[
                {"subject": "Vane", "predicate": "status", "object": "dead", "quote": "he died"}
            ],
        )
        assert out["contradictions"] == []

    async def test_a_fact_without_a_quote_or_with_one_the_scene_lacks_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/audit.py :: if fact.quote is None or not fact.quote.strip():
        Becomes: if False:
        Killed by: src/uclone_x/story/audit.py :: elif not quote_found(fact.quote, scene_text):
        Becomes: elif False:
        Killed by: src/uclone_x/story/audit.py :: elif quote_too_short(fact.quote):
        Becomes: elif False:
        """
        _, ctx = await _timed_story(tmp_path)
        out = await _ok(
            StoryAuditTool(),
            ctx,
            action="check",
            scene_id="ch01.s03",
            facts=[
                {"subject": "Lord Vane", "predicate": "status", "object": "alive"},
                _alive(quote="Lord Vane danced"),
                # Case and spacing are not the model's to get exactly right.
                _alive(quote="lord  vane LAUGHED"),
                # #1584 item 3: two letters, and words cut in the middle, show nothing.
                _alive(quote="at"),
                _alive(quote="ne laughed"),
            ],
        )
        assert out["rejected_facts"] == [
            {"fact": 1, "reason": "Fact 1 has no quote from the scene, so it was not checked."},
            {
                "fact": 2,
                "reason": "Fact 2's quote is not in scene 'ch01.s03', so it was not checked.",
            },
            {
                "fact": 4,
                "reason": "Fact 4's quote is too short to show anything: a quote needs at "
                "least 3 letters, so it was not checked.",
            },
            {
                "fact": 5,
                "reason": "Fact 5's quote is not in scene 'ch01.s03', so it was not checked.",
            },
        ]
        assert out["summary"] == (
            "1 submitted fact(s) were checked against 2 rule(s): 1 contradiction(s) found; "
            "4 fact(s) were rejected and not checked."
        )

    async def test_one_owner_at_a_time_is_a_default_rule(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/schemas.py :: kind="inverseFunctional",
        Becomes: kind="inverseFunctionaX",
        """
        story_id, ctx = await _timed_story(tmp_path)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {"id": "mara", "name": "Mara", "state": {"possesses": "moon-sword"}},
        )
        _codex(tmp_path, story_id, "items", {"id": "moon-sword", "name": "the moon sword"})
        out = await _ok(
            StoryAuditTool(),
            ctx,
            action="check",
            scene_id="ch01.s01",
            facts=[
                {
                    "subject": "Lord Vane",
                    "predicate": "possesses",
                    "object": "The Moon Sword",
                    "quote": "Lord Vane counted the toll",
                }
            ],
        )
        assert [c["kind"] for c in out["contradictions"]] == ["inverse_functional"]
        assert out["rules_source"] == "the defaults (story.yaml has no axioms)"

    async def test_story_yaml_s_rules_replace_the_defaults(self, tmp_path: Path) -> None:
        """An empty list is the story's choice: nothing is checked against a rule.

        Killed by: src/uclone_x/story/work.py :: return story_axioms(extra.get("axioms"), present="axioms" in extra, where=STORY_FILE)
        Becomes: return story_axioms(None, present=False, where=STORY_FILE)
        """
        story_id, ctx = await _timed_story(tmp_path)
        _set_story_yaml(tmp_path, story_id, axioms=[])
        out = await _ok(
            StoryAuditTool(), ctx, action="check", scene_id="ch01.s03", facts=[_alive()]
        )
        assert out["contradictions"] == []
        assert out["rules"] == []
        assert out["rules_source"] == "story.yaml"

    async def test_a_rule_is_read_whatever_its_case_and_reported_as_written(
        self, tmp_path: Path
    ) -> None:
        """#1584 item 2: a lowercase rule applies, and one that cannot is named as written.

        Killed by: src/uclone_x/story/audit.py :: return class_name(subject), class_name(obj)
        Becomes: return subject, obj
        Killed by: src/uclone_x/story/audit.py :: return self._spelling.setdefault(spelled.casefold(), spelled)
        Becomes: return spelled
        Killed by: src/uclone_x/story/audit.py :: {"rule": written.get(u.source, u.source), "reason": _plain_reason(u.reason)}
        Becomes: {"rule": u.source, "reason": _plain_reason(u.reason)}
        Killed by: src/uclone_x/story/audit.py :: {"rule": written.get(u.source, u.source), "reason": _plain_reason(u.reason)}
        Becomes: {"rule": written.get(u.source, u.source), "reason": u.reason}
        Killed by: src/uclone_x/story/audit.py :: return _KINDS.get(_WORD_JOIN.sub("", _KIND_PREFIX.sub("", raw.strip())).casefold())
        Becomes: return _KINDS.get(raw.strip())
        """
        story_id, ctx = await _timed_story(tmp_path)
        _set_story_yaml(
            tmp_path,
            story_id,
            axioms=[
                {"kind": "owl:Disjoint_With", "subject": "ALIVE", "object": "dead"},
                {"kind": "sameAs", "subject": "Alive", "object": "Living"},
                {"kind": "disjointWith", "subject": "Alive"},
            ],
        )
        out = await _ok(
            StoryAuditTool(), ctx, action="check", scene_id="ch01.s03", facts=[_alive()]
        )
        [contradiction] = out["contradictions"]
        assert contradiction["kind"] == "disjoint"
        assert out["summary"] == (
            "1 submitted fact(s) were checked against 1 rule(s): 1 contradiction(s) found; "
            "2 rule(s) could not be applied."
        )
        not_applied = out["rules_not_applied"]
        assert [r["rule"] for r in not_applied] == ["sameAs Alive Living", "disjointWith Alive"]
        assert not_applied[1]["reason"] == (
            "axiom kind 'disjointWith' needs an object, which is empty"
        )
        shown = repr(not_applied)
        for internal in ("story_axiom", "subject_entity", "object_value"):
            assert internal not in shown

    @pytest.mark.parametrize("rule_object", ["HalfDead", "half dead", "HALF_DEAD"])
    async def test_a_class_of_several_words_is_one_class_however_it_is_spelled(
        self, tmp_path: Path, rule_object: str
    ) -> None:
        """#1597 review B1: `HalfDead` in a rule meets `half dead` in the codex.

        Killed by: src/uclone_x/story/audit.py :: return self._spelling.setdefault(spelled.casefold(), spelled)
        Becomes: return spelled
        """
        story_id, ctx = await _timed_story(tmp_path)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {
                "id": "vane",
                "name": "Lord Vane",
                "aliases": ["Vane"],
                "state": {"status": "half dead"},
            },
        )
        _set_story_yaml(
            tmp_path,
            story_id,
            axioms=[{"kind": "disjointWith", "subject": "Alive", "object": rule_object}],
        )
        out = await _ok(
            StoryAuditTool(), ctx, action="check", scene_id="ch01.s03", facts=[_alive()]
        )
        [contradiction] = out["contradictions"]
        assert contradiction["kind"] == "disjoint"
        assert not out.get("rules_not_applied")

    async def test_a_class_a_rule_names_is_shown_as_the_rule_spells_it(
        self, tmp_path: Path
    ) -> None:
        """#1601: the codex's `npc alive` is shown as the rule's `NPCAlive`, not `NpcAlive`.

        Killed by: src/uclone_x/story/audit.py :: for axiom in axioms:
        Becomes: for axiom in ():
        """
        story_id, ctx = await _timed_story(tmp_path)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {
                "id": "vane",
                "name": "Lord Vane",
                "aliases": ["Vane"],
                "state": {"status": "npc alive"},
            },
        )
        _set_story_yaml(
            tmp_path,
            story_id,
            axioms=[{"kind": "disjointWith", "subject": "ALIVE", "object": "NPCAlive"}],
        )
        out = await _ok(
            StoryAuditTool(), ctx, action="check", scene_id="ch01.s03", facts=[_alive()]
        )
        [contradiction] = out["contradictions"]
        assert {"rule": "ALIVE disjointWith NPCAlive"} in contradiction["facts"]
        shown = repr(contradiction)
        assert "NpcAlive" not in shown and "'Alive'" not in shown

    async def test_a_composed_and_a_decomposed_letter_are_one_class(self, tmp_path: Path) -> None:
        """#1601: `é` typed as one code point or as `e` and an accent is the same letter.

        Killed by: src/uclone_x/story/audit.py :: return unicodedata.normalize("NFC", raw.strip())
        Becomes: return raw.strip()
        """
        story_id, ctx = await _timed_story(tmp_path)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {
                "id": "vane",
                "name": "Lord Vane",
                "aliases": ["Vane"],
                "state": {"status": "d\u00e9c\u00e9d\u00e9"},
            },
        )
        _set_story_yaml(
            tmp_path,
            story_id,
            axioms=[
                {"kind": "disjointWith", "subject": "Alive", "object": "de\u0301ce\u0301de\u0301"}
            ],
        )
        out = await _ok(
            StoryAuditTool(), ctx, action="check", scene_id="ch01.s03", facts=[_alive()]
        )
        [contradiction] = out["contradictions"]
        assert contradiction["kind"] == "disjoint"

    async def test_a_property_rule_is_read_whatever_its_case(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/audit.py :: return _predicate(subject), obj
        Becomes: return subject, obj
        """
        story_id, ctx = await _timed_story(tmp_path)
        _set_story_yaml(
            tmp_path, story_id, axioms=[{"kind": "inverseFunctional", "subject": "Possesses"}]
        )
        _codex(tmp_path, story_id, "characters", {"id": "mara", "name": "Mara"})
        _codex(tmp_path, story_id, "items", {"id": "toll", "name": "The toll", "state": {}})
        out = await _ok(
            StoryAuditTool(),
            ctx,
            action="check",
            scene_id="ch01.s01",
            facts=[
                {
                    "subject": subject,
                    "predicate": "possesses",
                    "object": "The toll",
                    "quote": "Lord Vane counted the toll",
                }
                for subject in ("Mara", "Lord Vane")
            ],
        )
        assert [c["kind"] for c in out["contradictions"]] == ["inverse_functional"]

    async def test_a_rule_that_does_not_fit_names_story_yaml_and_the_field(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/schemas.py :: parsed = StoryAxioms.model_validate({"axioms": raw})
        Becomes: parsed = StoryAxioms.model_validate({"axioms": []})
        """
        story_id, ctx = await _timed_story(tmp_path)
        _set_story_yaml(tmp_path, story_id, axioms=[{"subject": "Alive"}])
        error = await _refused(
            StoryAuditTool(), ctx, action="check", scene_id="ch01.s03", facts=[_alive()]
        )
        assert error.startswith("story.yaml does not fit its shape: 'axioms[0].kind'")

    async def test_a_scene_with_no_text_is_refused_in_words(self, tmp_path: Path) -> None:
        story_id, ctx = await _timed_story(tmp_path)
        await _ok(StoryOutlineTool(), ctx, action="set_scene", chapter_id="ch01", title="Empty")
        error = await _refused(
            StoryAuditTool(), ctx, action="check", scene_id="ch01.s05", facts=[_alive()]
        )
        assert error == "Scene 'ch01.s05' has no text yet, so there is nothing to check."
        assert story_id


# --------------------------------------------------------------------------------------
# Proposals: the model proposes, a person applies
# --------------------------------------------------------------------------------------


async def _proposed(ctx: ToolContext) -> str:
    out = await _ok(
        StoryCodexTool(),
        ctx,
        action="propose",
        entry_id="vane",
        at="ch01.s01",
        set={"wounded": True},
        quote="Lord Vane counted the toll",
    )
    proposal_id = out["proposed"]
    assert isinstance(proposal_id, str)
    return proposal_id


class TestProposals:
    async def test_a_proposal_needs_a_quote_the_scene_has(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/tools.py :: if not quote_found(quote, text.text):
        Becomes: if False:
        """
        story_id, ctx = await _timed_story(tmp_path)
        codex = StoryCodexTool()
        args: dict[str, Any] = {
            "action": "propose",
            "entry_id": "vane",
            "at": "ch01.s01",
            "set": {"wounded": True},
        }
        assert await _refused(codex, ctx, **args) == (
            "A proposal needs a quote from the scene that shows the change, so nothing was "
            "proposed."
        )
        assert await _refused(codex, ctx, **args, quote="Vane bled") == (
            "The quote is not in scene 'ch01.s01', so nothing was proposed. Copy whole words "
            "exactly as the scene has them."
        )
        assert not (_story_dir(tmp_path, story_id) / "proposals").exists()

    async def test_a_quote_is_whole_words_and_long_enough_to_show_something(
        self, tmp_path: Path
    ) -> None:
        """#1584 item 3: a letter, or part of a word, is in almost every scene.

        Killed by: src/uclone_x/story/quotes.py :: return sum(1 for ch in folded(quote) if ch.isalnum()) < MIN_QUOTE_CHARACTERS
        Becomes: return not folded(quote)
        Killed by: src/uclone_x/story/quotes.py :: if not _cuts_a_word(before, needle[0]) and not _cuts_a_word(after, needle[-1]):
        Becomes: if not _cuts_a_word(after, needle[-1]):
        Killed by: src/uclone_x/story/quotes.py :: if not _cuts_a_word(before, needle[0]) and not _cuts_a_word(after, needle[-1]):
        Becomes: if not _cuts_a_word(before, needle[0]):
        Killed by: src/uclone_x/story/quotes.py :: start = haystack.find(needle, start + 1)
        Becomes: break
        Killed by: src/uclone_x/story/tools.py :: if quote_too_short(quote):
        Becomes: if False:
        Killed by: src/uclone_x/story/quotes.py :: "IDEOGRAPHIC",
        Becomes: "IDEOGRAPHIX",
        Killed by: src/uclone_x/story/quotes.py :: "VERTICAL IDEOGRAPHIC",
        Becomes: "VERTICAL IDEOGRAPHIX",
        Killed by: src/uclone_x/story/quotes.py :: "HALFWIDTH KATAKANA",
        Becomes: "HALFWIDTH KATAKANX",
        """
        story_id, ctx = await _timed_story(tmp_path)
        codex = StoryCodexTool()
        args: dict[str, Any] = {
            "action": "propose",
            "entry_id": "vane",
            "at": "ch01.s01",
            "set": {"wounded": True},
        }
        # ch01.s01: "Mara waited at the gate while Lord Vane counted the toll."
        short = await _refused(codex, ctx, **args, quote="a")
        assert short == (
            "The quote is too short to show the change: it needs at least 3 letters, so "
            "nothing was proposed."
        )
        for part_of_a_word in ("ate at th", "Vane count", "ted the toll"):
            error = await _refused(codex, ctx, **args, quote=part_of_a_word)
            assert error.startswith("The quote is not in scene 'ch01.s01'"), part_of_a_word
        assert not (_story_dir(tmp_path, story_id) / "proposals").exists()
        # The first place the words appear may be inside a word; a later one still counts.
        assert quote_found("ate", "Kate ate")
        assert quote_found("the toll.", "Lord Vane counted the toll.")
        # Scripts written without spaces have no word edge, so only the length counts.
        assert quote_found("上死了", "他在桥上死了。")
        assert not quote_found("死了", "他在桥上死了。")
        # Marks written among those characters have no word edge either (#1601).
        assert quote_found("〇〇年に", "二〇〇〇年に")
        assert quote_found("々々年", "年々々々年")
        assert quote_found("〻〻年", "〻〻〻年")
        assert quote_found("ｲｳｴ", "ｱｲｳｴｵ")
        assert await _proposed(ctx) == "p001"

    async def test_a_proposal_file_that_does_not_load_is_reported_and_the_rest_listed(
        self, tmp_path: Path
    ) -> None:
        """#1584 item 7: one broken file under proposals/ neither hides the others nor
        vanishes; it is named with what is wrong with it.

        Killed by: src/uclone_x/story/work.py :: unreadable.append(problem)
        Becomes: raise
        Killed by: src/uclone_x/story/work.py :: unreadable.append(problem)
        Becomes: pass
        """
        story_id, ctx = await _timed_story(tmp_path)
        assert await _proposed(ctx) == "p001"
        broken = _story_dir(tmp_path, story_id) / "proposals" / "p002.yaml"
        broken.write_text("id: p002\nstatus: pending\n", encoding="utf-8")

        out = await _ok(StoryCodexTool(), ctx, action="proposals")

        assert [line["id"] for line in out["pending"]] == ["p001"]
        [problem] = out["unreadable_files"]
        assert problem["file"] == "proposals/p002.yaml"
        assert problem["reason"].startswith("proposals/p002.yaml does not fit its shape:")
        assert "Traceback" not in problem["reason"]

    async def test_apply_without_a_person_s_approval_changes_nothing(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/tools.py :: if not context.approved_by_person:
        Becomes: if False:
        """
        story_id, ctx = await _timed_story(tmp_path)
        proposal_id = await _proposed(ctx)
        entry = _story_dir(tmp_path, story_id) / "codex/characters/vane.yaml"
        before = entry.read_text(encoding="utf-8")

        error = await _refused(StoryCodexTool(), ctx, action="apply", proposal_id=proposal_id)
        assert error == (
            f"Applying proposal '{proposal_id}' needs a person's approval, and this call was "
            "not approved, so the codex was not changed."
        )
        assert entry.read_text(encoding="utf-8") == before
        # Nor can the model say so in the call: the argument is not one the tool takes.
        forged = await _call(
            StoryCodexTool(), ctx, action="apply", proposal_id=proposal_id, approved_by_person=True
        )
        assert not forged.success
        assert entry.read_text(encoding="utf-8") == before

    async def test_an_approved_apply_changes_the_entry_and_closes_the_proposal(
        self, tmp_path: Path
    ) -> None:
        story_id, ctx = await _timed_story(tmp_path)
        proposal_id = await _proposed(ctx)
        approved = _ctx(tmp_path, story=story_id, approved=True)
        out = await _ok(StoryCodexTool(), approved, action="apply", proposal_id=proposal_id)
        assert out["applied"] == proposal_id

        entry = yaml.safe_load(
            (_story_dir(tmp_path, story_id) / "codex/characters/vane.yaml").read_text()
        )
        assert {"at": "ch01.s01", "set": {"wounded": True}} in entry["progressions"]
        listed = await _ok(StoryCodexTool(), ctx, action="proposals")
        assert listed["pending"] == []
        assert listed["decided"] == [{"id": proposal_id, "status": "applied", "entry_id": "vane"}]
        again = await _refused(StoryCodexTool(), approved, action="apply", proposal_id=proposal_id)
        assert again == f"Proposal '{proposal_id}' was already applied, so nothing was changed."

    async def test_an_entry_edited_since_the_proposal_is_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/proposals.py :: if digest != proposal.entry_digest:
        Becomes: if False:
        """
        story_id, ctx = await _timed_story(tmp_path)
        proposal_id = await _proposed(ctx)
        edited = _codex(tmp_path, story_id, "characters", {"id": "vane", "name": "Vane the Grey"})
        error = await _refused(
            StoryCodexTool(),
            _ctx(tmp_path, story=story_id, approved=True),
            action="apply",
            proposal_id=proposal_id,
        )
        assert error == (
            f"The entry 'vane' changed after proposal '{proposal_id}' was made, so it was not "
            "applied. Reject it and propose the change again."
        )
        assert "Vane the Grey" in edited.read_text(encoding="utf-8")

    async def test_reject_needs_no_approval_and_closes_the_proposal(self, tmp_path: Path) -> None:
        _, ctx = await _timed_story(tmp_path)
        proposal_id = await _proposed(ctx)
        await _ok(StoryCodexTool(), ctx, action="reject", proposal_id=proposal_id, reason="no")
        error = await _refused(
            StoryCodexTool(), ctx, action="reject", proposal_id=proposal_id, reason="no"
        )
        assert error == f"Proposal '{proposal_id}' was already rejected, so nothing was changed."


# --------------------------------------------------------------------------------------
# Through the agent: only a person's yes applies
# --------------------------------------------------------------------------------------


def _agent(bus: EventBus | None, **kwargs: Any) -> BaseAgent:
    tools = ToolRegistry()
    tools.register(StoryCodexTool())
    config = AgentConfig(agent_id="writer", name="writer", approval_timeout_seconds=2.0)
    return BaseAgent(config=config, bus=bus, tools=tools, **kwargs)


def _apply_call(proposal_id: str) -> ToolCallRequest:
    return ToolCallRequest(
        id="c1", name="story_codex", arguments={"action": "apply", "proposal_id": proposal_id}
    )


def _vane_progressions(workspace: Path, story_id: str) -> list[dict[str, Any]]:
    path = _story_dir(workspace, story_id) / "codex/characters/vane.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["progressions"]


async def _run_with_answer(
    tmp_path: Path, answer: str, **agent_kwargs: Any
) -> tuple[str, str, dict[str, Any]]:
    """Runs one `apply` call through the agent while a person answers `answer`."""
    story_id, ctx = await _timed_story(tmp_path)
    proposal_id = await _proposed(ctx)
    bus = EventBus()
    await bus.start()
    agent = _agent(bus, **agent_kwargs)
    topic = f"session.{agent._context.session_id}"  # pyright: ignore[reportPrivateUsage]
    sub = bus.subscribe({topic})
    asked: dict[str, Any] = {}

    async def person() -> None:
        while True:
            event = await sub.get()
            if event.type == EventType.TOOL_APPROVAL_REQUEST:
                asked.update(event.payload)
                await bus.publish(
                    AgentEvent(
                        type=EventType.TOOL_APPROVAL_RESPONSE,
                        topic=topic,
                        sender_id="ui",
                        payload={"request_id": event.payload["request_id"], "action": answer},
                    )
                )
                return

    await agent.start()
    answering = asyncio.create_task(person())
    try:
        _, record = await agent._execute_single_tool(  # pyright: ignore[reportPrivateUsage]
            _apply_call(proposal_id), ctx
        )
        await asyncio.wait_for(answering, timeout=2.0)
    finally:
        await agent.stop()
        await bus.stop()
    return story_id, record.status, asked


class TestOnlyAPersonApproves:
    async def test_a_person_s_yes_applies_it(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: approved_by_person = decision.action in (HookAction.ALLOW, HookAction.MODIFY)
        Becomes: approved_by_person = False
        """
        story_id, status, asked = await _run_with_answer(tmp_path, "allow")
        assert status == "success"
        assert asked["tool_name"] == "story_codex"
        assert (
            asked["reason"] == "story_codex runs this action only when a person approves the call."
        )
        assert {"at": "ch01.s01", "set": {"wounded": True}} in _vane_progressions(
            tmp_path, story_id
        )

    async def test_a_hook_s_rewrite_is_what_the_person_approves_and_what_runs(
        self, tmp_path: Path
    ) -> None:
        """#1584 item 6: a hook that rewrites an apply call keeps its rewrite.

        The runtime asks a person about every apply, whatever the hooks said; before this,
        that forced question dropped the hook's `MODIFY`, so the person approved -- and the
        tool ran -- the model's call rather than the one the hooks left.

        Killed by: src/uclone_x/agent/hooks/runner.py :: decision.modified_payload if decision.action == HookAction.MODIFY else None
        Becomes: None
        Killed by: src/uclone_x/agent/base.py :: if decision.action in (HookAction.ALLOW, HookAction.MODIFY)
        Becomes: if False
        Killed by: src/uclone_x/agent/base.py :: if hook_arguments is not None
        Becomes: if False
        """

        class Redirect(BaseHook):
            async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
                arguments = dict(context.payload.get("arguments", {}))
                arguments["proposal_id"] = "p002"
                return HookDecision(
                    action=HookAction.MODIFY, modified_payload={"arguments": arguments}
                )

        story_id, ctx = await _timed_story(tmp_path)
        assert await _proposed(ctx) == "p001"
        second = await _ok(
            StoryCodexTool(),
            ctx,
            action="propose",
            entry_id="vane",
            at="ch01.s01",
            set={"hooded": True},
            quote="Lord Vane counted the toll",
        )
        assert second["proposed"] == "p002"
        bus = EventBus()
        await bus.start()
        agent = _agent(bus, hooks=[Redirect()])
        topic = f"session.{agent._context.session_id}"  # pyright: ignore[reportPrivateUsage]
        sub = bus.subscribe({topic})
        asked: dict[str, Any] = {}

        async def person() -> None:
            while True:
                event = await sub.get()
                if event.type == EventType.TOOL_APPROVAL_REQUEST:
                    asked.update(event.payload)
                    await bus.publish(
                        AgentEvent(
                            type=EventType.TOOL_APPROVAL_RESPONSE,
                            topic=topic,
                            sender_id="ui",
                            payload={"request_id": event.payload["request_id"], "action": "allow"},
                        )
                    )
                    return

        await agent.start()
        answering = asyncio.create_task(person())
        try:
            _, record = await agent._execute_single_tool(  # pyright: ignore[reportPrivateUsage]
                _apply_call("p001"), ctx
            )
            await asyncio.wait_for(answering, timeout=2.0)
        finally:
            await agent.stop()
            await bus.stop()
        assert record.status == "success", record.error
        assert asked["arguments"] == {"action": "apply", "proposal_id": "p002"}
        progressions = _vane_progressions(tmp_path, story_id)
        assert {"at": "ch01.s01", "set": {"hooded": True}} in progressions
        assert {"at": "ch01.s01", "set": {"wounded": True}} not in progressions

    async def test_an_empty_rewrite_is_what_the_person_sees_and_what_runs(
        self, tmp_path: Path
    ) -> None:
        """#1597 review B2: a hook's rewrite to `{}` is shown, so `{}` is what runs.

        An empty rewrite read for its truth was dropped after the person said yes, and the
        model's original call ran, approved, although the person had been shown `{}`.

        Killed by: src/uclone_x/agent/base.py :: if approved_arguments is not None and decision.action == HookAction.ALLOW
        Becomes: if approved_arguments and decision.action == HookAction.ALLOW
        """

        class Empty(BaseHook):
            async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
                return HookDecision(action=HookAction.MODIFY, modified_payload={"arguments": {}})

        story_id, ctx = await _timed_story(tmp_path)
        assert await _proposed(ctx) == "p001"
        bus = EventBus()
        await bus.start()
        agent = _agent(bus, hooks=[Empty()])
        topic = f"session.{agent._context.session_id}"  # pyright: ignore[reportPrivateUsage]
        sub = bus.subscribe({topic})
        asked: dict[str, Any] = {}

        async def person() -> None:
            while True:
                event = await sub.get()
                if event.type == EventType.TOOL_APPROVAL_REQUEST:
                    asked.update(event.payload)
                    await bus.publish(
                        AgentEvent(
                            type=EventType.TOOL_APPROVAL_RESPONSE,
                            topic=topic,
                            sender_id="ui",
                            payload={"request_id": event.payload["request_id"], "action": "allow"},
                        )
                    )
                    return

        await agent.start()
        answering = asyncio.create_task(person())
        try:
            _, record = await agent._execute_single_tool(  # pyright: ignore[reportPrivateUsage]
                _apply_call("p001"), ctx
            )
            await asyncio.wait_for(answering, timeout=2.0)
        finally:
            await agent.stop()
            await bus.stop()
        assert asked["arguments"] == {}
        assert record.status != "success"
        assert {"at": "ch01.s01", "set": {"wounded": True}} not in _vane_progressions(
            tmp_path, story_id
        )

    async def test_a_person_s_no_leaves_the_codex_alone(self, tmp_path: Path) -> None:
        story_id, status, _ = await _run_with_answer(tmp_path, "block")
        assert status == "error"
        assert _vane_progressions(tmp_path, story_id) == [
            {"at": "ch01.s02", "set": {"status": "dead"}}
        ]

    async def test_a_hook_that_allows_everything_still_leaves_it_to_a_person(
        self, tmp_path: Path
    ) -> None:
        """Auto mode allows every tool; the runner asks anyway, and nobody is there to answer.

        Killed by: src/uclone_x/agent/hooks/runner.py :: if needs_approval and decision.action not in (HookAction.BLOCK, HookAction.ASK):
        Becomes: if False:
        Killed by: src/uclone_x/agent/base.py :: pre_payload["needs_approval"] = tool_call_needs_approval(
        Becomes: pre_payload["needs_approval"] = False and tool_call_needs_approval(

        The refusal is the person's answer in the desktop app, which does not ask during a
        conversation: it says so in plain words, not that a request "timed out", and names
        where the person decides instead (the story view, #1560).

        Killed by: src/uclone_x/agent/base.py :: timeout_note or "Approval request timed out (denied fail-closed)",
        Becomes: "Approval request timed out (denied fail-closed)",
        Killed by: src/uclone_x/story/tools.py :: approval_timeout_note: ClassVar[str | None] = APPLY_NOT_APPROVED_NOTE
        Becomes: approval_timeout_note: ClassVar[str | None] = None
        """
        story_id, ctx = await _timed_story(tmp_path)
        proposal_id = await _proposed(ctx)
        agent = _agent(None, hooks=[HumanApprovalHook(permission_mode=PermissionMode.AUTO)])
        _, record = await agent._execute_single_tool(  # pyright: ignore[reportPrivateUsage]
            _apply_call(proposal_id), ctx
        )
        assert record.status == "error"
        assert record.error == (
            "The change was not applied, and nothing in the story changed: applying a "
            "proposal needs a person to approve the call, and no one did. Where the app does "
            "not ask during a conversation, as the desktop app does not, a person approves or "
            "rejects proposals in the story's view, under Files."
        )
        assert _vane_progressions(tmp_path, story_id) == [
            {"at": "ch01.s02", "set": {"status": "dead"}}
        ]

    async def test_the_unanswered_note_is_true_for_an_id_that_names_no_proposal(
        self, tmp_path: Path
    ) -> None:
        """#1584 item 4: the note is said before the tool runs, so it cannot know the id.

        It must not say a proposal is kept when there is none, nor that this shell is the
        desktop app.

        Killed by: src/uclone_x/story/tools.py :: "in the story's view, under Files."
        Becomes: "in the story's view, under Files. The proposal is kept."
        """
        story_id, ctx = await _timed_story(tmp_path)
        agent = _agent(None, hooks=[HumanApprovalHook(permission_mode=PermissionMode.AUTO)])
        _, record = await agent._execute_single_tool(  # pyright: ignore[reportPrivateUsage]
            _apply_call("p999"), ctx
        )
        assert record.status == "error"
        assert record.error is not None
        assert "kept" not in record.error
        assert "The desktop app does not ask" not in record.error
        assert "p999" not in record.error
        assert not (_story_dir(tmp_path, story_id) / "proposals").exists()

    async def test_a_task_from_another_persona_answers_input_required(self, tmp_path: Path) -> None:
        """Nobody can approve during an A2A task, so the call is blocked and named as pending."""
        story_id, ctx = await _timed_story(tmp_path)
        proposal_id = await _proposed(ctx)
        runner = _DeferredApprovalRunner()
        agent = _agent(None, hook_runner=runner)
        _, record = await agent._execute_single_tool(  # pyright: ignore[reportPrivateUsage]
            _apply_call(proposal_id), ctx
        )
        assert record.status == "error"
        assert runner.pending == ["story_codex"]
        assert len(_vane_progressions(tmp_path, story_id)) == 1

    async def test_propose_is_not_held_for_approval(self, tmp_path: Path) -> None:
        _, ctx = await _timed_story(tmp_path)
        agent = _agent(None)
        call = ToolCallRequest(
            id="c1",
            name="story_codex",
            arguments={
                "action": "propose",
                "entry_id": "vane",
                "at": "ch01.s01",
                "set": {"wounded": True},
                "quote": "Lord Vane counted the toll",
            },
        )
        _, record = await agent._execute_single_tool(call, ctx)  # pyright: ignore[reportPrivateUsage]
        assert record.status == "success"


# --------------------------------------------------------------------------------------
# character_sheet over a story (#1576 items 2 and 3)
# --------------------------------------------------------------------------------------


class TestAReadIsNotRecordedAsAWrite:
    """#1584 item 5: a room counts a writing tool's call that names no path as a possible
    unnamed write, so a read of the codex, outline or manuscript must not be recorded as a
    call that writes. The tool is still a write tool (`enable_write_tools` refuses it)."""

    async def test_the_story_tools_reads_are_recorded_as_reads(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: declared_writes = tool_call_writes_files(tool_inst, unwrapped_args)
        Becomes: declared_writes = tool_writes_files(tool_inst)
        Killed by: src/uclone_x/tools/base.py :: return not (isinstance(action, str) and action in cast(frozenset[object], declared))
        Becomes: return True
        Killed by: src/uclone_x/story/tools.py :: read_actions: ClassVar[frozenset[str]] = frozenset({"get", "search", "proposals"})
        Becomes: read_actions: ClassVar[frozenset[str]] = frozenset({"search", "proposals"})
        Killed by: src/uclone_x/story/tools.py :: read_actions: ClassVar[frozenset[str]] = frozenset({"get"})
        Becomes: read_actions: ClassVar[frozenset[str]] = frozenset()
        Killed by: src/uclone_x/story/tools.py :: read_actions: ClassVar[frozenset[str]] = frozenset({"read", "list"})
        Becomes: read_actions: ClassVar[frozenset[str]] = frozenset({"list"})
        """
        _, ctx = await _timed_story(tmp_path)
        tools = ToolRegistry()
        for tool in (StoryCodexTool(), StoryOutlineTool(), StoryManuscriptTool()):
            tools.register(tool)
        config = AgentConfig(agent_id="writer", name="writer", approval_timeout_seconds=2.0)
        agent = BaseAgent(config=config, bus=None, tools=tools)

        async def recorded(name: str, **arguments: Any) -> bool:
            call = ToolCallRequest(id="c1", name=name, arguments=arguments)
            _, record = await agent._execute_single_tool(call, ctx)  # pyright: ignore[reportPrivateUsage]
            assert record.status == "success", record.error
            return record.writes_files

        assert not await recorded("story_codex", action="get", entry_id="vane")
        assert not await recorded("story_codex", action="proposals")
        assert not await recorded("story_outline", action="get")
        assert not await recorded("story_manuscript", action="read", scene_id="ch01.s01")
        assert not await recorded("story_manuscript", action="list")
        # A call that changes the story is still recorded as one that may write.
        assert await recorded(
            "story_codex",
            action="propose",
            entry_id="vane",
            at="ch01.s01",
            set={"wounded": True},
            quote="Lord Vane counted the toll",
        )
        assert tool_writes_files(StoryCodexTool())


class TestTheRuleKindsAreTheReasoners:
    def test_the_audit_reads_every_kind_spelling_the_reasoner_reads(self) -> None:
        """#1597 review: `story/audit.py` keeps its own list of rule kinds, because the
        layering keeps it from importing `ontology/rules.py`; this pins the two together."""
        from uclone_x.ontology.rules import (
            _KIND_ALIASES,  # pyright: ignore[reportPrivateUsage]
        )
        from uclone_x.story.audit import _KINDS  # pyright: ignore[reportPrivateUsage]

        assert dict(_KINDS) == dict(_KIND_ALIASES)


class TestTheWriterIsToldWhatTheCheckReads:
    def test_the_writer_prompt_says_the_check_reads_the_end_of_the_scene(self) -> None:
        """#1584 item 9: by design the audit checks the state once the scene has ended, so
        "alive" in the death scene itself is a contradiction; the Writer is told so, and
        told to report it with that explanation rather than hide it."""
        writer = PersonaRegistry().get_persona("writer")
        assert writer is not None
        prompt = writer.system_prompt
        assert (
            "The check compares the scene with the story as it stands once the scene has "
            "ended, including the changes the codex records at that scene"
        ) in prompt
        death = (
            "when the codex records a character's death at a scene, a fact that they are "
            "alive, quoted from that scene, is reported as a contradiction"
        )
        assert death in prompt
        assert "rather than leaving it out" in prompt


class TestCharacterSheetSaysWhatItLacks:
    async def test_a_broken_workspace_sheet_is_reported_not_taken_for_missing(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/tools/builtin/character.py :: result["workspace_sheet_unreadable"] = legacy_problem
        Becomes: pass
        """
        story_id, _ = await _timed_story(tmp_path)
        (tmp_path / "characters").mkdir()
        (tmp_path / "characters" / "mara.yaml").write_text("name: [unclosed", encoding="utf-8")
        out = await _ok(
            CharacterSheetTool(), _ctx(tmp_path, story=story_id), action="get", character_id="mara"
        )
        assert out["workspace_sheet_unreadable"] == {
            "file": "characters/mara.yaml",
            "reason": "It could not be read as YAML text, so it was not shown.",
        }

    async def test_looks_the_codex_does_not_give_are_not_invented(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/builtin/character.py :: "gender": visual.gender if visual else None,
        Becomes: "gender": (visual.gender if visual else None) or "other",
        Killed by: src/uclone_x/tools/builtin/character.py :: "default_style": visual.default_style if visual else None,
        Becomes: "default_style": (visual.default_style if visual else None) or "anime",
        """
        story_id, _ = await _timed_story(tmp_path)
        _codex(tmp_path, story_id, "characters", {"id": "mara", "name": "Mara"})
        out = await _ok(
            CharacterSheetTool(), _ctx(tmp_path, story=story_id), action="get", character_id="mara"
        )
        assert out["character"]["gender"] is None
        assert out["character"]["default_style"] is None

    async def test_compose_says_which_characters_brought_no_looks(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/builtin/character.py :: composed["notes"] = notes
        Becomes: pass
        """
        story_id, _ = await _timed_story(tmp_path)
        _codex(tmp_path, story_id, "characters", {"id": "mara", "name": "Mara"})
        _codex(
            tmp_path,
            story_id,
            "characters",
            {"id": "ilse", "name": "Ilse", "visual": {"tags": ["silver hair"]}},
        )
        out = await _ok(
            CharacterSheetTool(),
            _ctx(tmp_path, story=story_id),
            action="compose",
            character_ids=["mara", "ilse"],
        )
        assert out["notes"] == [
            "'mara' has no 'visual' block in the story's codex, so no tags of its own went "
            "into the prompt.",
            "'ilse' has no gender in its 'visual' block, so the prompt counts it as a person "
            "rather than a girl or a boy.",
        ]
