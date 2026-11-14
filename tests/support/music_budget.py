"""A full-size song and a scripted three-turn Composer conversation over it (#2429).

Shared by `tests/unit/test_music_context_budget.py` and by the measurement this module
runs as a script, which prints the estimated input tokens of every model call the scripted
conversation makes:

    PYTHONPATH=$PWD:$PWD/src:$PWD/examples/music/src .venv/bin/python -m tests.support.music_budget

The model is scripted, so the same calls run before and after a change; only what the
tools return and what the runtime assembles around it differs. Token figures come from
the repo's own estimator (`estimate_request_tokens`, UTF-8 bytes / 4).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.compactor import estimate_request_tokens
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.tools.registry import ToolRegistry
from uclone_x_music import MusicLifecycleHook
from uclone_x_music.library import SongLibrary
from uclone_x_music.schemas import (
    InstrumentEntry,
    ScoreOutline,
    ScoreSection,
    TrackBar,
    TrackFile,
)
from uclone_x_music.section_turn import MusicTurnHook
from uclone_x_music.ucx_extension import EXTENSION
from uclone_x_music.work import SongWork, track_id_for

__all__ = [
    "BARS",
    "INSTRUMENTS",
    "ROOM",
    "SECTIONS",
    "SONG",
    "ScriptedComposer",
    "bar_fingerprints",
    "bar_notes",
    "build_full_song",
    "composer_agent",
    "repeated_bodies",
    "request_text",
    "run_scenario",
    "scenario_script",
    "scripted_reply",
]

SONG = "budget-song"
ROOM = "room-budget"
SECTIONS = ("theme_a", "theme_b", "bridge")
BARS = 32
INSTRUMENTS: tuple[tuple[str, str, str, int], ...] = (
    ("violin_solo", "strings", "lead_melody", 72),
    ("grand_piano", "piano", "harmonic_comping", 60),
    ("bass_synth", "bass", "bass_foundation", 36),
    ("drum_kit", "drums", "rhythm_percussion", 48),
)
#: Sections with no tracks, so the score outline is full size: nine more 32-bar sections
#: and a 120-bar finale.
EXTRA_SECTIONS: tuple[tuple[str, int], ...] = (
    *((f"movement_{n}", BARS) for n in range(1, 10)),
    ("finale", 120),
)
#: Instruments in the orchestra with no tracks yet.
EXTRA_INSTRUMENTS = tuple(f"pad_{n:02d}" for n in range(1, 13))
_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="scripted", model="m"),
    served_by=ServiceRef(provider="scripted", model="m"),
    attempts=(),
)
_USAGE = TokenUsage(provider="scripted", model="m", input_tokens=0, output_tokens=0)


def _pitch(midi: int) -> str:
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def bar_notes(section_index: int, instrument_index: int, bar: int, *, shift: int = 0) -> list[str]:
    """Four quarter notes, unique to (section, instrument, bar): the velocity names the bar."""
    base = INSTRUMENTS[instrument_index][3]
    velocity = 1 + section_index * BARS + (bar - 1)
    return [
        f"{beat}.0:{_pitch(base + (bar + beat + shift) % 12)}:1.0:v{velocity}"
        for beat in range(1, 5)
    ]


def build_full_song(workspace: Path) -> None:
    """Thirteen sections, sixteen instruments, twelve full 32-bar tracks, held by `ROOM`."""
    library = SongLibrary(workspace)
    library.create(title="Budget Song", room_id=ROOM, song_id=SONG)
    work = SongWork(library, SONG, ROOM)
    chords = ["Dm7", "Gm7", "Cmaj7", "Fmaj7", "Bbmaj7", "Em7b5", "A7", "Dm7"]
    outline = ScoreOutline(
        tempo=96,
        time_signature="4/4",
        key_signature="D minor",
        sections=[
            ScoreSection(
                id=sec,
                title=sec.replace("_", " ").title(),
                kind="verse" if i < 2 else "bridge",
                bars=BARS,
                chords=[chords[(b + i) % len(chords)] for b in range(BARS)],
            )
            for i, sec in enumerate(SECTIONS)
        ]
        + [
            ScoreSection(
                id=sec,
                title=sec.replace("_", " ").title(),
                kind="movement",
                bars=bars,
                chords=[chords[b % len(chords)] for b in range(bars)],
            )
            for sec, bars in EXTRA_SECTIONS
        ],
    )
    existing = work.read_score()
    work.write_score(outline, expected_digest=existing[1] if existing else None)
    for inst_id, category, role, _ in INSTRUMENTS:
        entry = InstrumentEntry.model_validate(
            {"id": inst_id, "name": inst_id.replace("_", " "), "category": category, "role": role}
        )
        have = work.read_instrument(inst_id)
        work.write_instrument(entry, expected_digest=have[1] if have else None)
    for inst_id in EXTRA_INSTRUMENTS:
        pad = InstrumentEntry.model_validate(
            {"id": inst_id, "name": inst_id, "category": "synth", "role": "pad_texture"}
        )
        work.write_instrument(pad, expected_digest=None)
    for s_index, sec in enumerate(SECTIONS):
        for i_index, (inst_id, _, _, _) in enumerate(INSTRUMENTS):
            track = TrackFile(
                track_id=track_id_for(sec, inst_id),
                section_id=sec,
                instrument_id=inst_id,
                bars=[
                    TrackBar(
                        bar=b,
                        chord=outline.sections[s_index].chords[b - 1],
                        notes=bar_notes(s_index, i_index, b),
                    )
                    for b in range(1, BARS + 1)
                ],
            )
            have_track = work.read_track(sec, inst_id)
            work.write_track(track, expected_digest=have_track[1] if have_track else None)


def bar_fingerprints() -> list[str]:
    """Every bar of the song, and every bar the scenario writes, as compact JSON notes."""
    song = [
        bar_notes(s, i, b)
        for s in range(len(SECTIONS))
        for i in range(len(INSTRUMENTS))
        for b in range(1, BARS + 1)
    ]
    written = [bar_notes(0, 0, 4, shift=5), bar_notes(0, 2, 2, shift=5)]
    return [json.dumps(notes, separators=(",", ":")) for notes in (*song, *written)]


_SPACE = re.compile(r"\s+")


def request_text(request: LLMRequest) -> str:
    """Everything a request carries that the model reads, whitespace removed."""
    parts: list[str] = []
    for message in request.messages:
        parts.append(message.content or "")
        for call in message.tool_calls or ():
            parts.append(json.dumps(unwrap_immutable(call.arguments), separators=(",", ":")))
    return _SPACE.sub("", "".join(parts))


def repeated_bodies(request: LLMRequest, fingerprints: Sequence[str]) -> dict[str, int]:
    """The bars that `request` carries more than once, with how many times."""
    text = request_text(request)
    counts = Counter({fp: text.count(fp) for fp in fingerprints})
    return {fp: n for fp, n in counts.items() if n > 1}


def _call(call_id: str, name: str, **arguments: Any) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def scripted_reply(content: str | None, *calls: ToolCallRequest) -> ModelResponse:
    return ModelResponse(
        content=content,
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        tool_calls=calls,
        usage=_USAGE,
        provenance=_PROV,
    )


def _new_bar(bar: int, instrument_index: int, section_index: int = 0) -> dict[str, Any]:
    """A changed bar: the same rhythm, pitches shifted so it differs from every song bar."""
    return {"bar": bar, "notes": bar_notes(section_index, instrument_index, bar, shift=5)}


def scenario_script() -> list[tuple[str, str, list[ModelResponse]]]:
    """Three turns: (turn id, the person's words, what the scripted model answers)."""
    return [
        (
            "t1",
            "Look at the violin in bars 1 to 8 and make bar 4 brighter.",
            [
                scripted_reply(
                    None,
                    _call(
                        "c1",
                        "music_track",
                        action="read",
                        section_id="theme_a",
                        instrument_id="violin_solo",
                        start_bar=1,
                        end_bar=8,
                    ),
                ),
                scripted_reply(
                    None,
                    _call(
                        "c2",
                        "music_track",
                        action="patch_bars",
                        section_id="theme_a",
                        instrument_id="violin_solo",
                        start_bar=4,
                        end_bar=4,
                        bars=[_new_bar(4, 0)],
                    ),
                ),
                scripted_reply(None, _call("c3", "music_proposal", action="list")),
                scripted_reply("I proposed a brighter bar 4 for the violin."),
            ],
        ),
        (
            "t2",
            "Apply it.",
            [
                scripted_reply(
                    None,
                    _call(
                        "c4",
                        "music_track",
                        action="read",
                        section_id="theme_a",
                        instrument_id="violin_solo",
                        start_bar=1,
                        end_bar=8,
                    ),
                ),
                scripted_reply(
                    None,
                    _call(
                        "c5",
                        "music_score",
                        action="upsert_section",
                        section_id="theme_a",
                        bars=BARS,
                    ),
                ),
                scripted_reply(
                    None, _call("c6", "music_proposal", action="apply", proposal_id="p001")
                ),
                scripted_reply("Please approve the proposal in MusicView."),
            ],
        ),
        (
            "t3",
            "Make bass bar 2 of theme A move, then check the section.",
            [
                scripted_reply(
                    None,
                    _call(
                        "c7",
                        "music_track",
                        action="patch_bars",
                        section_id="theme_a",
                        instrument_id="bass_synth",
                        start_bar=2,
                        end_bar=2,
                        propose=False,
                        bars=[_new_bar(2, 2)],
                    ),
                ),
                scripted_reply(None, _call("c8", "music_context")),
                scripted_reply(None, _call("c9", "music_audit", section_id="theme_a")),
                scripted_reply("Bass bar 2 now moves; the audit is in."),
            ],
        ),
    ]


class ScriptedComposer(BaseLLMConnector):
    """Answers each request with the next scripted response, and keeps every request."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            return scripted_reply("(script ended)")
        return self._responses.pop(0)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        resp = await self.generate(request)
        yield StreamChunk(
            delta_content=resp.content,
            finish_reason=resp.finish_reason,
            tool_calls=resp.tool_calls,
        )


def composer_agent(workspace: Path, llm: BaseLLMConnector) -> BaseAgent:
    """A Composer-like agent: the music tools and hooks, and nobody to answer approvals."""
    tools = ToolRegistry()
    for tool in EXTENSION.tools():
        tools.register(tool)
    return BaseAgent(
        config=AgentConfig(
            agent_id="composer", name="Composer", llm_config=AgentLLMConfig(model_name="m")
        ),
        llm=llm,
        tools=tools,
        context=AgentContext(agent_id="composer", session_id="s-budget", workspace_root=workspace),
        lifecycle_hooks=(MusicLifecycleHook(), MusicTurnHook()),
        approvals_answered=False,
    )


async def run_scenario(workspace: Path) -> tuple[ScriptedComposer, list[Any]]:
    """Build the song, run the three scripted turns, and return the model and the results."""
    build_full_song(workspace)
    script = scenario_script()
    llm = ScriptedComposer([resp for _, _, responses in script for resp in responses])
    agent = composer_agent(workspace, llm)
    results: list[Any] = []
    for turn_id, words, _ in script:
        results.append(
            await agent.execute_turn(
                words,
                caller_turn_id=turn_id,
                room_id=ROOM,
                story_id=SONG,
                workspace_root=workspace,
            )
        )
    return llm, results


def _main() -> int:
    with tempfile.TemporaryDirectory(prefix="music-budget-") as tmp:
        llm, _ = asyncio.run(run_scenario(Path(tmp)))
    per_call = [estimate_request_tokens(r) for r in llm.requests]
    fingerprints = bar_fingerprints()
    repeats = [sum(n - 1 for n in repeated_bodies(r, fingerprints).values()) for r in llm.requests]
    print("call  est_input_tokens  repeated_bar_copies")
    for index, (tokens, extra) in enumerate(zip(per_call, repeats, strict=True), start=1):
        print(f"{index:>4}  {tokens:>16}  {extra:>19}")
    print(f"calls={len(per_call)} total_est_input_tokens={sum(per_call)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
