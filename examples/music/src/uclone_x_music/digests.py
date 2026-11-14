"""The short forms the music tools return to the model (#2429).

Owner ruling, 2026-10-30: a tool result in context is a digest and a short summary, or the
slice the model asked for; a score appears in context at most once. So a write returns what
it changed and the file's new digest, never the file; a list returns one line per item; an
audit returns its first findings and the count of the rest. The fix is in what the tools
return, never in trimming history afterwards.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uclone_x_music.audit import AuditFinding
    from uclone_x_music.context import ScoreContextBundle
    from uclone_x_music.schemas import InstrumentEntry, MusicProposal, ScoreSection, TrackBar

__all__ = [
    "AUDIT_FINDINGS_SHOWN",
    "LIST_ITEMS_SHOWN",
    "NOTHING_CHANGED",
    "SECTION_CHORDS_SHOWN",
    "audit_digest",
    "bar_view",
    "bundle_view",
    "changed_bar_numbers",
    "digest_of_value",
    "instrument_line",
    "proposal_line",
    "section_line",
    "write_result",
]

#: The opening words of every result whose write left the file as it was.
NOTHING_CHANGED = "Nothing changed"
#: How many chords one section's summary names before it says how many more there are.
SECTION_CHORDS_SHOWN = 32
#: How many findings an audit result lists; the rest are counted by kind.
AUDIT_FINDINGS_SHOWN = 8
#: How many items a list result shows before it says how many more there are.
LIST_ITEMS_SHOWN = 20


def digest_of_value(value: object) -> str:
    """SHA-256 of `value`'s canonical JSON."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_result(
    path: str,
    previous: str | None,
    digest: str,
    summary: str,
    **extra: Any,
) -> dict[str, Any]:
    """A write's result: whether it changed the file, the digests, and what changed.

    `previous` is the file's digest before the write (`None` when it did not exist). When
    the digests match the file already held this content and nothing was written, and the
    summary says so in plain words, so a repeated write never reads as a fresh success.
    """
    changed = previous != digest
    text = (
        summary
        if changed
        else f"{NOTHING_CHANGED}: {path} already held this, so nothing was written."
    )
    return {
        "changed": changed,
        "path": path,
        "digest": digest,
        "previous_digest": previous,
        "summary": text,
        **extra,
    }


def section_line(section: ScoreSection) -> str:
    """`'<kind>, <n> bars, chords: A B C …'`, naming at most `SECTION_CHORDS_SHOWN` chords."""
    chords = list(section.chords)
    shown = " ".join(chords[:SECTION_CHORDS_SHOWN])
    more = len(chords) - SECTION_CHORDS_SHOWN
    tail = f" (+{more} more)" if more > 0 else ""
    tempo = f", {len(section.tempo_map)} tempo changes" if section.tempo_map else ""
    return f"{section.kind}, {section.bars} bars{tempo}, chords: {shown or 'none'}{tail}"


def instrument_line(entry: InstrumentEntry) -> dict[str, Any]:
    """One instrument as the fields a mix decision needs."""
    return {
        "id": entry.id,
        "name": entry.name,
        "category": entry.category,
        "role": entry.role,
        "volume": entry.volume,
        "pan": entry.pan,
        "muted": entry.muted,
        "solo": entry.solo,
    }


def proposal_line(proposal: MusicProposal, digest: str) -> dict[str, Any]:
    """One proposal as its id, target, bar range and summary: never its bars."""
    line: dict[str, Any] = {
        "id": proposal.id,
        "status": proposal.status,
        "kind": proposal.kind,
        "summary": proposal.summary,
        "digest": digest,
    }
    if proposal.section_id is not None:
        line["section_id"] = proposal.section_id
    if proposal.instrument_id is not None:
        line["instrument_id"] = proposal.instrument_id
    if proposal.start_bar is not None:
        line["bars"] = f"{proposal.start_bar}..{proposal.end_bar or proposal.start_bar}"
    return line


def bar_view(bar: TrackBar) -> dict[str, Any]:
    """A bar without the fields that hold their defaults."""
    return bar.model_dump(mode="json", exclude_defaults=True) | {"bar": bar.bar}


def bundle_view(bundle: ScoreContextBundle) -> dict[str, Any]:
    """A score context bundle with its bars' default fields left out."""
    payload: dict[str, Any] = bundle.model_dump(mode="json", exclude_none=True)
    payload["tail_bars"] = {
        inst: [bar_view(b) for b in bars] for inst, bars in bundle.tail_bars.items()
    }
    return payload


def changed_bar_numbers(current: Sequence[TrackBar], proposed: Sequence[TrackBar]) -> list[int]:
    """The bar numbers in `proposed` whose content differs from the same bar in `current`."""
    by_number = {b.bar: b for b in current}
    return [b.bar for b in proposed if by_number.get(b.bar) != b]


def audit_digest(findings: Sequence[AuditFinding], *, passed: bool) -> dict[str, Any]:
    """An audit as its verdict, its first findings, and a count of every finding by kind."""
    ordered = sorted(findings, key=lambda f: (f.severity != "error", f.bar))
    shown = [
        {
            "code": f.code,
            "severity": f.severity,
            "bar": f.bar,
            "message": f.message,
        }
        for f in ordered[:AUDIT_FINDINGS_SHOWN]
    ]
    result: dict[str, Any] = {
        "passed": passed,
        "findings_total": len(findings),
        "findings": shown,
    }
    if len(findings) > len(shown):
        result["counts_by_code"] = dict(Counter(f.code for f in findings))
        result["not_shown"] = len(findings) - len(shown)
    return result
