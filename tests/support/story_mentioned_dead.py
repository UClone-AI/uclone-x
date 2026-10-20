"""#2311's three false findings, and their replay of recorded live scene readings.

Each case is a scene qwen3:8b wrote in a live run after 이진호 had died, in which he is
only spoken of: his legacy (`legacy`), a memory of him alive (`memory`), or what another
character says of him (`reported`). Read with the reading of the time, the store reported
him alive again (`disjoint`, alive vs dead). The story here is that run's, cut to what the
reading is given: the cast as pending proposals, the death as his progression at the scene
that wrote it, and the scene's own outline line and text (`tests/fixtures/
story_mentioned_dead/cases.json`).

Shared by `tests/unit/test_story_mentioned_dead.py` and the recorder that records the
readings again. The recorder drives a live model through the private eval harness, so it is
kept with the recording's run record in the eval ledger, not in this tree.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from uclone_x.knowledge.extraction import ALIAS_MARKER
from uclone_x_story.continuity import Finding, checked_store, findings_from
from uclone_x_story.enrich import (
    ENRICH_MARKER,
    PRESENCE_MARKER,
    Enrichment,
    SceneModel,
    enrich_scene,
)
from uclone_x_story.library import StoryLibrary
from uclone_x_story.schemas import Outline, Proposal, dump_file
from uclone_x_story.work import StoryWork

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "story_mentioned_dead"
ROOM = "room_a"
#: The man who died, in every case.
DEAD = "ijinho"
#: Each set of cases and its recording: #2311's three scenes, and #2361's two, from a later
#: live run, kept in files of their own so neither recording is rewritten for the other.
SETS: dict[str, tuple[str, str]] = {
    "2311": ("cases.json", "readings.json"),
    "2361": ("cases-2361.json", "readings-2361.json"),
}


def cases(cases_set: str = "2311") -> dict[str, dict[str, Any]]:
    """Each case: the scene that wrote the death, the scene read, its text, and the cast."""
    data = json.loads((FIXTURES / SETS[cases_set][0]).read_text(encoding="utf-8"))
    return data["cases"]


def recording(cases_set: str = "2311") -> dict[str, Any]:
    """The recorded live readings: `{"model", "cases": {case: [{"marker",
    "prompt_sha256", "reply"}]}}`."""
    return json.loads((FIXTURES / SETS[cases_set][1]).read_text(encoding="utf-8"))


def reading_marker(prompt: str) -> str | None:
    """Which reading call `prompt` is, by the marker it starts with; `None` for any other."""
    for marker in (ENRICH_MARKER, ALIAS_MARKER, PRESENCE_MARKER):
        if prompt.startswith(marker):
            return marker
    return None


def story(
    workspace: Path, case: str, *, holder: str | None = None, cases_set: str = "2311"
) -> StoryWork:
    """The case's story: two scenes outlined, nothing written, nothing approved, and the
    death a progression of 이진호 at the scene that wrote it. With `holder`, that character
    holds the watch from the same scene on."""
    found = cases(cases_set)[case]
    death, scene = found["death_scene"], found["scene"]
    library = StoryLibrary(workspace)
    record = library.create("Clock", conversation_id=ROOM)
    work = StoryWork(library, record.story_id)
    chapters: dict[str, list[dict[str, str]]] = {}
    for line in (death, scene):
        chapters.setdefault(line["id"].split(".")[0], []).append(line)
    outline = Outline.model_validate(
        {
            "chapters": [
                {"id": chapter, "title": chapter, "scenes": lines}
                for chapter, lines in chapters.items()
            ]
        }
    )
    work.write("outline.yaml", dump_file(outline), room_id=ROOM, expected_digest=None)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for raw in found["cast"]:
        entry: dict[str, Any] = dict(raw)
        kind = "items" if entry["id"] == "eunsigye" else "characters"
        if kind == "characters":
            entry["state"] = {"status": "alive"}
        if entry["id"] == DEAD:
            entry["progressions"] = [{"at": death["id"], "set": {"status": "dead"}}]
        if entry["id"] == holder:
            entry["progressions"] = [{"at": death["id"], "set": {"possesses": ["eunsigye"]}}]
        work.add_proposal(
            Proposal.model_validate(
                {
                    "id": "p000",
                    "kind": kind,
                    "entry_id": entry["id"],
                    "change": {"new_entry": entry},
                    "proposed_at": now,
                    "room_id": ROOM,
                }
            ),
            room_id=ROOM,
        )
    return work


async def read_and_check(
    work: StoryWork,
    model: SceneModel,
    case: str,
    text: str | None = None,
    *,
    cases_set: str = "2311",
) -> tuple[list[Finding], Enrichment]:
    """The case's scene written (as `text` when given), read once, and checked against the
    store, as `story_start` does after writing a chapter."""
    found = cases(cases_set)[case]
    scene_id: str = found["scene"]["id"]
    if text is None:
        text = str(found["text"])
    work.write_scene(
        scene_id, text, room_id=ROOM, expected_digest=None, agent_id=None, turn_index=None
    )
    read = await enrich_scene(work, model=model, scene_id=scene_id, text=text, room_id=ROOM)
    outline, _ = work.require_outline()
    store = checked_store(work.story_id, outline, work.codex(), [p for p, _ in work.proposals()[0]])
    findings = findings_from(
        store=store,
        outline=outline,
        axioms=[],
        axioms_are_defaults=False,
        scene_id=scene_id,
        text=text,
        facts=read.facts,
    )
    return findings, read


class Replay:
    """Every reading call answered from the recording, in order.

    A reading whose prompt is not the one recorded is kept in `mismatched` (the story's
    reading catches what a model raises, so a raise here would only look like a reading
    that failed), and the tests hold it to none.
    """

    def __init__(self, calls: list[dict[str, str]]) -> None:
        self.calls = list(calls)
        self.replayed = 0
        self.mismatched: list[str] = []
        #: Presence questions the recording has no answer for, answered with nothing.
        self.unrecorded: list[str] = []
        #: Presence questions' prompts, as sent.
        self.presence_prompts: list[str] = []

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        del system, temperature, max_tokens
        marker = reading_marker(prompt)
        if marker is None:
            raise AssertionError(f"not a reading: {prompt[:60]}")
        if marker == PRESENCE_MARKER:
            self.presence_prompts.append(prompt)
        if marker == PRESENCE_MARKER and (not self.calls or self.calls[0]["marker"] != marker):
            # A presence question (#2317) the recording was made without: left unanswered,
            # so the reading's own labels decide, as they did when it was recorded.
            self.unrecorded.append(marker)
            return "{}"
        if not self.calls:
            self.mismatched.append(f"{marker}: a call the recording does not have")
            return "{}"
        call = self.calls.pop(0)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if (call["marker"], call["prompt_sha256"]) != (marker, digest):
            self.mismatched.append(f"{marker}: not the prompt recorded")
        self.replayed += 1
        return call["reply"]
