"""The #2014 seeded recall cases and their replay of recorded live scene readings.

Shared by `tests/unit/test_story_seeded_recall.py` and the recorder that records the
readings again. The recorder drives a live model through the private eval harness, so it
is kept with the recording's run record in the eval ledger, not in this tree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tests.unit import test_story_start as T
from uclone_x.knowledge.extraction import ALIAS_MARKER
from uclone_x.tools.models import ToolResult
from uclone_x_story.enrich import ENRICH_MARKER, PRESENCE_MARKER

#: Where each case's recorded live readings are:
#: `{"model", "calls": [{"marker", "prompt_sha256", "reply"}]}`.
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "story_seeded_recall"

#: The seeded scenes of the 2026-10-11 acceptance run.
GRAVE = (
    "마을 장로는 조각난 시계를 손에 쥐고 무덤 앞에 섰다. 이진호는 무덤 아래로 떨어졌고, "
    "그 자리에서 숨을 거두었다. 장로는 시계를 쥔 채 오래 서 있었다."
)
VILLAGE = (
    "다음 날 아침, 이진호는 마을 한가운데로 걸어갔다. 이진호는 조각난 시계를 품에 넣고 "
    "다녔다. 사람들은 그를 보고 수군거렸다."
)


#: The handover case (#2312 review): the elder gives the clock to 이진호 in one scene, and
#: in the next takes it out and uses it again. Nobody dies, so only the one-holder rule
#: can catch it.
HANDOVER = (
    "마을 장로는 조각난 시계를 이진호에게 건넸다. 이진호는 시계를 받아 품에 넣었다. "
    "장로는 빈손으로 집에 돌아갔다."
)
OLD_HOLDER = (
    "그날 밤, 마을 장로는 조각난 시계를 꺼내 시간을 확인했다. 장로는 시계를 손에 쥔 채 "
    "창밖을 내다보았다."
)

#: Each seeded case: its two scenes, in order, and its recording.
CASES: dict[str, tuple[tuple[str, str], str]] = {
    "death": ((GRAVE, VILLAGE), "readings.json"),
    "handover": ((HANDOVER, OLD_HOLDER), "handover.json"),
}


def reading_marker(prompt: str) -> str | None:
    """Which reading call `prompt` is, by the marker it starts with; `None` for any other."""
    for marker in (ENRICH_MARKER, ALIAS_MARKER, PRESENCE_MARKER):
        if prompt.startswith(marker):
            return marker
    return None


def scripted_parts(case: str = "death") -> dict[str, Any]:
    """The scripted model's parts for a seeded case: the clock's cast and codex, and the
    case's two scenes, in order."""
    scenes, _ = CASES[case]
    return {"cast_replies": [T._CLOCK_CAST], "kg": T._CLOCK_KG, "scenes": list(scenes)}  # pyright: ignore[reportPrivateUsage]


async def run_seeded(model: T.ScriptedModel, workspace: Path) -> ToolResult:
    """The seeded story, asked for at once, so its first chapter is written and checked."""
    return await T._tool(model).execute(  # pyright: ignore[reportPrivateUsage]
        {"action": "start", "request": T.REQUEST + " 바로 써줘."},
        T._ctx(workspace),  # pyright: ignore[reportPrivateUsage]
    )


class Replay(T.ScriptedModel):
    """The scripted model, with every reading call answered from the recording in order.

    A reading whose prompt is not the one recorded is kept in `mismatched` (the story's
    reading catches what a model raises, so a raise here would only look like a reading
    that failed), and the tests hold it to none.
    """

    def __init__(self, calls: list[dict[str, str]], case: str = "death") -> None:
        super().__init__(**scripted_parts(case))
        self.calls = list(calls)
        self.replayed = 0
        self.mismatched: list[str] = []
        #: Presence questions the recording has no answer for, answered with nothing.
        self.unrecorded: list[str] = []
        #: Those questions' prompts, as sent.
        self.presence_prompts: list[str] = []

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        marker = reading_marker(prompt)
        if marker is None:
            return await super().complete(
                prompt, system=system, temperature=temperature, max_tokens=max_tokens
            )
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


def stale_message(case: str) -> str:
    """What a replay of `case` against prompts nobody sends any more says to do: record that
    case's own file again (#2317 item 1)."""
    return (
        "the reading prompts are not the ones recorded: record "
        f"tests/fixtures/story_seeded_recall/{CASES[case][1]} again"
    )


def recording(case: str = "death") -> dict[str, Any]:
    """The recorded live readings of a seeded case."""
    _, name = CASES[case]
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
