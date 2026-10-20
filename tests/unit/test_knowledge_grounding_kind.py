"""#1842: a fact's grounding is held to the words, not to the line kind the model names.

The 2026-09-27 ruling: another clone's messages are context only, never a fact source. The
concern of #1842 was a model that reads a claim from another clone's line and labels it
`source: "person"` while a person line is also in the span, so a check on the label alone
passes it. Since #2235 the label is not read at all: a fact must quote words that are in a
person or tool line (`knowledge.extraction.verify_facts` over those lines only), and which
of the two holds the quote is its source (`memory.extractor._source`). These pin that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit import test_knowledge_extractor as K
from tests.unit.test_knowledge_extractor import (
    SAGE,
    SCOUT,
    _fact,  # pyright: ignore[reportPrivateUsage]
    _lesson,  # pyright: ignore[reportPrivateUsage]
    _memory,  # pyright: ignore[reportPrivateUsage]
    _Model,  # pyright: ignore[reportPrivateUsage]
    _read_reply,  # pyright: ignore[reportPrivateUsage]
    _room,  # pyright: ignore[reportPrivateUsage]
    _say,  # pyright: ignore[reportPrivateUsage]
    _seat,  # pyright: ignore[reportPrivateUsage]
    _Seats,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.memory.extractor import SpanLine


@pytest.fixture(autouse=True)
def _sessions_under_tmp(tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    """The seats keep their sessions under this test's `tmp_path`, not the working tree."""
    K._SESSIONS[0] = tmp_path / "sessions"  # pyright: ignore[reportPrivateUsage]


async def test_another_clones_claim_labelled_as_the_persons_is_not_saved(tmp_path: Path) -> None:
    """Sage says where Alice lives; Scout's extraction quotes Sage and calls it the person's.

    Killed by: src/uclone_x/memory/extractor.py :: grounds = [line for line in lesson.lines if line.kind in _SOURCE_ORIGIN]
    Becomes: grounds = list(lesson.lines)
    """
    scout_memory = _memory(tmp_path, "scout")
    sage_llm = _Model(reply="Alice told me she moved to Seoul.")
    scout_llm = _Model(
        extractions=[
            [
                _fact("user", "lives_in", "Seoul", quote="she moved to Seoul", source="person"),
                _fact("user", "likes", "maps", quote="I like maps", source="person"),
            ]
        ]
    )
    orch = _room(
        tmp_path,
        _Seats((SAGE, _seat(SAGE, sage_llm, None)), (SCOUT, _seat(SCOUT, scout_llm, scout_memory))),
        SAGE.id,
        SCOUT.id,
    )

    await _say(orch, "I like maps.")

    span = str(scout_llm.extraction_requests[0].messages[1].content)
    assert "[other clone: Sage] Alice told me she moved to Seoul." in span
    assert "[person: Alice] I like maps." in span
    facts = scout_memory.list_facts()
    assert [(f.predicate, f.object_value, f.origin) for f in facts] == [("likes", "maps", "told")]


def test_the_source_is_the_line_that_holds_the_quote_not_the_label() -> None:
    """A tool's result labelled as the person's is filed as found, from the tool line.

    Killed by: src/uclone_x/memory/extractor.py :: source = _source(fact.quote, grounds)
    Becomes: source = str(item.get("source") or "")
    """
    lesson = _lesson(
        SpanLine("person", "Alice", "What's the weather in Busan?"),
        SpanLine("tool", "weather", "Busan: 21C, light rain."),
    )
    reply = json.dumps(
        {"facts": [_fact("project", "weather_busan", "21C", quote="Busan: 21C", source="person")]}
    )

    assert _read_reply(reply, lesson) == [("project", "weather_busan", "21C", "tool")]
