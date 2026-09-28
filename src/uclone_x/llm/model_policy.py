"""Which listed model to suggest first: family patterns, never model ids (#1631).

A pattern names a family ("Gemini Flash, newest version"), so it keeps choosing correctly as
the provider publishes new versions and retires old ones. It can only pick from the entries
the provider listed; it can never add one. The first rule for each provider is the mid tier
(Flash, Sonnet, the non-mini GPT) -- owner ruling 2026-09-25.

When no rule matches, because a provider renamed its families, the suggestion is the newest
chat model in the listing. That may be a less suitable choice, visible in the picker, but it
is never a model that does not exist and never nothing while a chat model is listed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from uclone_x.llm.catalog import CatalogEntry


@dataclass(frozen=True)
class FamilyRule:
    """A model family, by id pattern; its capture groups are the version, newest first."""

    pattern: re.Pattern[str]


def _rule(pattern: str) -> FamilyRule:
    return FamilyRule(re.compile(pattern))


#: Per provider settings id, the families in the order they are preferred.
PREFERENCE: Final[dict[str, tuple[FamilyRule, ...]]] = {
    "gemini": (
        _rule(r"^gemini-(\d+)(?:\.(\d+))?-flash$"),
        _rule(r"^gemini-(\d+)(?:\.(\d+))?-pro$"),
        _rule(r"^gemini-(\d+)(?:\.(\d+))?-flash-lite$"),
    ),
    "anthropic": (
        _rule(r"^claude-sonnet-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?$"),
        _rule(r"^claude-(\d+)-(\d+)-sonnet(?:-\d{8})?$"),
        _rule(r"^claude-opus-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?$"),
        _rule(r"^claude-haiku-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?$"),
    ),
    "openai": (
        _rule(r"^gpt-(\d+)(?:\.(\d+))?$"),
        _rule(r"^gpt-(\d+)(?:\.(\d+))?-mini$"),
    ),
}

#: Never suggested, though still listed and still selectable: previews and experiments change
#: or vanish without notice, `-latest` aliases move under the user, and the rest do not chat.
EXCLUDE: Final = re.compile(
    r"preview|exp|-latest$|tts|embed|image|audio|realtime|transcribe|search|live",
)

_OLDEST: Final = datetime.min.replace(tzinfo=UTC)


def _provider_key(provider: str) -> str:
    key = provider.strip().lower()
    return "gemini" if key == "google" else key


def _version(match: re.Match[str]) -> tuple[int, ...]:
    return tuple(int(group) for group in match.groups() if group is not None)


def _newest_key(entry: CatalogEntry) -> tuple[datetime, str]:
    return (entry.created_at or _OLDEST, entry.id)


def recommend(provider: str, entries: Sequence[CatalogEntry]) -> str | None:
    """The id to suggest from `entries`, or None only when no entry can chat."""
    chat = [entry for entry in entries if entry.chat_capable]
    if not chat:
        return None
    candidates = [entry for entry in chat if not EXCLUDE.search(entry.id)] or chat
    for rule in PREFERENCE.get(_provider_key(provider), ()):
        matched = [
            (_version(match), entry)
            for entry in candidates
            if (match := rule.pattern.match(entry.id)) is not None
        ]
        if matched:
            # Newest version first; within one version, the most recently published listing
            # entry (a dated snapshot and its alias are the same model).
            return max(matched, key=lambda pair: (pair[0], _newest_key(pair[1])))[1].id
    return max(candidates, key=_newest_key).id
