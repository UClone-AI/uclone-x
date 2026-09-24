"""Budget guard for Tier 2 VCR cassettes.

Built before the first cassette exists, deliberately. Measured 2026-09-03 in `uclone2`
(`~/uclone-git/uclone2` @ `14e806c`): three live tests carry **18.3 MB** of committed
cassettes, one of them a single 4.1 MB file holding **397 interactions** — of which 109
are the application's own API and its Langfuse telemetry ingestion, neither of which is
an LLM request or response. Nothing measured any of that at commit time, so it
accumulated silently and a re-record produces an unreviewable diff.

A cassette is an artifact a human is expected to read a diff of. That is what these
limits encode.

The interaction count is taken from the `- request:` line count rather than by parsing
YAML: vcrpy writes one such top-level sequence entry per interaction, and counting lines
keeps this guard free of a YAML dependency it would otherwise need only to count.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MAX_CASSETTE_BYTES: Final = 256 * 1024
MAX_INTERACTIONS: Final = 20
REDACTION_PLACEHOLDER: Final = "REDACTED"

_INTERACTION_RE: Final = re.compile(r"^- request:", re.MULTILINE)

# Deliberately narrow: each pattern matches a *shape* of live credential, and the
# redaction placeholder written by the Tier 2 `filter_headers` config never matches any
# of them. A broad "looks secret" heuristic would fire on recorded prose and train
# people to ignore this guard.
_SECRET_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("openai-style key", re.compile(r"sk-[A-Za-z0-9]{16,}")),
    ("google api key", re.compile(r"AIza[A-Za-z0-9_\-]{20,}")),
    ("anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    (
        "bearer token",
        re.compile(rf"(?i)bearer\s+(?!{REDACTION_PLACEHOLDER})[A-Za-z0-9._\-]{{16,}}"),
    ),
)

__all__ = [
    "MAX_CASSETTE_BYTES",
    "MAX_INTERACTIONS",
    "REDACTION_PLACEHOLDER",
    "Violation",
    "check_cassette",
    "check_cassette_dir",
    "count_interactions",
    "format_violations",
    "iter_cassettes",
]


@dataclass(frozen=True)
class Violation:
    """One budget rule broken by one cassette."""

    path: Path
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}: {self.rule} — {self.detail}"


def count_interactions(text: str) -> int:
    """Count recorded interactions in cassette YAML text."""
    return len(_INTERACTION_RE.findall(text))


def iter_cassettes(root: Path) -> Iterator[Path]:
    """Yield every cassette under `root`, sorted, or nothing if `root` does not exist."""
    if not root.is_dir():
        return
    yield from sorted(p for p in root.rglob("*.yaml") if p.is_file())


def check_cassette(path: Path) -> tuple[Violation, ...]:
    """Check one cassette against every budget rule, reporting all violations at once."""
    violations: list[Violation] = []

    size = path.stat().st_size
    if size > MAX_CASSETTE_BYTES:
        violations.append(
            Violation(
                path=path,
                rule="size",
                detail=f"{size} bytes exceeds the {MAX_CASSETTE_BYTES}-byte budget. "
                "Narrow the recording to the provider host, or split the scenario.",
            )
        )

    text = path.read_text(encoding="utf-8", errors="replace")

    interactions = count_interactions(text)
    if interactions > MAX_INTERACTIONS:
        violations.append(
            Violation(
                path=path,
                rule="interactions",
                detail=f"{interactions} interactions exceeds the {MAX_INTERACTIONS} budget. "
                "Telemetry and first-party API traffic must not be recorded.",
            )
        )

    for label, pattern in _SECRET_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            line = text.count("\n", 0, match.start()) + 1
            violations.append(
                Violation(
                    path=path,
                    rule="secret",
                    detail=f"possible {label} at line {line}; "
                    "extend filter_headers/filter_query_parameters and re-record.",
                )
            )

    return tuple(violations)


def check_cassette_dir(root: Path) -> tuple[Violation, ...]:
    """Check every cassette under `root`. An empty or absent tree yields no violations."""
    return tuple(v for cassette in iter_cassettes(root) for v in check_cassette(cassette))


def format_violations(violations: tuple[Violation, ...]) -> str:
    """Render violations for an assertion message."""
    return "\n".join(f"  • {v}" for v in violations)
