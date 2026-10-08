"""The lexical half of host binding, and the answer to a call that names no tool (#2190).

Host binding (`tool_binder.ToolBinder`) ranks catalog tools against a user message. A
message of two requests ("look up PR #42, then make a Notion page") embedded whole lets the
stronger request take every slot, so the binder ranks each clause on its own as well, and
fuses each embedding ranking with a BM25 ranking over the tools' names and descriptions.
This module holds the parts of that which need no model:

- `split_clauses` -- the message cut at English and Korean sequence markers;
- `BM25` -- Okapi BM25 over each tool's name and description terms;
- `rrf` and `interleave` -- reciprocal-rank fusion, and a round-robin over rankings;
- `dedupe` -- one tool per action across servers (`create_issue` on GitHub and GitLab),
  unless the message names the second server;
- `closest_tool_names` and `unknown_tool_message` -- the plain tool result for a call to a
  name no tool has, listing the closest tools the agent may use.

The `tool_binding` eval scores these on 239 labelled messages; the eval imports them from
here, so it measures this code.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Final

__all__ = [
    "BM25",
    "CLOSEST_LIMIT",
    "RRF_K",
    "SERVER_ALIASES",
    "closest_tool_names",
    "dedupe",
    "interleave",
    "normalize_terms",
    "rrf",
    "server_and_action",
    "split_clauses",
    "stem",
    "unknown_tool_message",
]

#: The constant of reciprocal-rank fusion (Cormack et al.), as the eval measured it.
RRF_K: Final = 60
_BM25_K1: Final = 1.2
_BM25_B: Final = 0.75

_STOPWORDS: Final = frozenset(
    {
        "a", "an", "the", "to", "for", "of", "and", "or", "in", "on", "at", "by", "with",
        "from", "into", "is", "are", "be", "it", "this", "that", "my", "me", "i", "you",
        "your", "tool", "tools", "please", "can", "do", "what", "how",
    }
)  # fmt: skip
_STEM_SUFFIXES: Final[tuple[str, ...]] = ("ing", "ion", "es", "s", "e")


def stem(word: str) -> str:
    """Strip the first matching suffix of ``_STEM_SUFFIXES`` if three letters remain.

    Light on purpose: ``translation``, ``translate`` and ``translating`` all become
    ``translat``; ``issues`` and ``issue`` become ``issu``.
    """
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def normalize_terms(text: str) -> list[str]:
    """Lowercased word terms of ``text``, split on ``_`` and punctuation, stopwords dropped,
    each stemmed. Non-Latin words pass through unstemmed."""
    return [stem(w) for w in re.findall(r"[^\W_]+", text.lower()) if w not in _STOPWORDS]


# Sequence markers a clause boundary is cut at. English words need word boundaries; the
# Korean connectives are attached to the verb before them ("확인하고", "만든 다음"), so they
# are matched as suffixes of a word and the cut falls after them.
_EN_SPLIT: Final = re.compile(
    r"\s*(?:[,;]?\s*\b(?:and then|then|after that|afterwards|once (?:that's|that is|it's|you've|"
    r"you have) done|once done|also|and also|plus|next)\b|;|\.\s+(?=[A-Z]))\s*",
    re.IGNORECASE,
)
_KO_SPLIT: Final = re.compile(
    r"(?<=[가-힣])(?:하고|고 나서|한 다음에?|한 뒤에?|한 후에?|하면|해서|하고 나서|다음에|그리고|그 다음|끝나면|"
    r"고)(?=[\s,]|$)|(?:\s|^)(?:그리고|그 다음에?|그러고 나서|다음으로|마지막으로)(?=\s)|[.?!]\s+"
)


def split_clauses(text: str) -> list[str]:
    """`text` cut at sequence markers, each piece stripped, pieces of under two terms
    dropped. A message with no marker is one clause."""
    pieces = [p for p in _EN_SPLIT.split(text) if p]
    out: list[str] = []
    for piece in pieces:
        out.extend(p for p in _KO_SPLIT.split(piece) if p)
    clauses = [c.strip(" ,.;") for c in out]
    return [c for c in clauses if len(c.split()) >= 2 or len(c) >= 6] or [text]


class BM25:
    """Okapi BM25 over documents' terms (`normalize_terms`, without ``mcp``)."""

    def __init__(self, documents: Sequence[str]) -> None:
        self.docs = [[t for t in normalize_terms(d) if t != "mcp"] for d in documents]
        self.avg_len = sum(len(d) for d in self.docs) / len(self.docs) if self.docs else 0.0
        df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        """One score per document, in document order; zero where no query term occurs."""
        terms = normalize_terms(query)
        out: list[float] = []
        for doc in self.docs:
            score = 0.0
            for term in terms:
                tf = doc.count(term)
                if not tf:
                    continue
                denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * len(doc) / (self.avg_len or 1))
                score += self.idf.get(term, 0.0) * tf * (_BM25_K1 + 1) / denom
            out.append(score)
        return out

    def ranking(self, query: str, names: Sequence[str]) -> list[str]:
        """`names` (one per document) whose document shares a term with `query`, best
        first; ties keep document order."""
        scores = self.scores(query)
        ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
        return [names[i] for i in ranked]


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal-rank fusion of `rankings`; ties keep first-seen order."""
    score: dict[str, float] = {}
    order: list[str] = []
    for ranking in rankings:
        for rank, name in enumerate(ranking):
            if name not in score:
                score[name] = 0.0
                order.append(name)
            score[name] += 1.0 / (k + rank + 1)
    return sorted(order, key=lambda n: (-score[n], order.index(n)))


def interleave(rankings: Sequence[Sequence[str]]) -> list[str]:
    """Round-robin over `rankings`, first entries first, each name once."""
    out: list[str] = []
    for depth in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if depth < len(ranking) and ranking[depth] not in out:
                out.append(ranking[depth])
    return out


#: How a message may name a server, for `dedupe`'s "the message names this server" check:
#: the server's own name, and its Korean spellings.
SERVER_ALIASES: Final[Mapping[str, tuple[str, ...]]] = {
    "github": ("github", "깃허브", "깃헙"),
    "gitlab": ("gitlab", "깃랩"),
    "slack": ("slack", "슬랙"),
    "discord": ("discord", "디스코드"),
    "teams": ("teams", "팀즈"),
    "notion": ("notion", "노션"),
    "jira": ("jira", "지라"),
    "linear": ("linear", "리니어"),
    "confluence": ("confluence", "컨플루언스"),
    "email": ("email", "mail", "메일", "이메일"),
    "postgres": ("postgres", "포스트그레스"),
    "mysql": ("mysql",),
    "sqlite": ("sqlite",),
    "calendar": ("calendar", "캘린더", "일정"),
}


def server_and_action(name: str) -> tuple[str, str]:
    """``("github", "create_issue")`` for ``mcp__github__create_issue``; a name with no
    server part is ``("", name)``."""
    parts = name.split("__")
    return (parts[-2], parts[-1]) if len(parts) >= 3 else ("", name)


def dedupe(ranking: Sequence[str], text: str) -> list[str]:
    """`ranking` without a tool whose action is already taken from another server, unless
    `text` names that tool's server."""
    lowered = text.lower()
    taken: dict[str, str] = {}
    out: list[str] = []
    for name in ranking:
        server, action = server_and_action(name)
        if server and action in taken and taken[action] != server:
            aliases = SERVER_ALIASES.get(server, (server,))
            if not any(a in lowered for a in aliases):
                continue
        taken.setdefault(action, server)
        out.append(name)
    return out


#: How many tools `unknown_tool_message` suggests.
CLOSEST_LIMIT: Final = 3


def closest_tool_names(
    name: str,
    argument_names: Sequence[str],
    candidates: Sequence[tuple[str, str, Sequence[str]]],
    limit: int = CLOSEST_LIMIT,
) -> list[str]:
    """The `limit` candidates closest to a call of `name` with `argument_names`.

    Each candidate is ``(name, description, parameter names)``. A candidate scores the
    similarity of its action part to the called name's (difflib's ratio, so a misspelled
    or re-prefixed name finds its tool), plus the share of the call's terms -- its name's
    and its argument names' -- that occur in the candidate's name, description or
    parameters. Ties keep candidate order. No model is asked: this runs where an embedder
    may not.
    """
    _, wanted_action = server_and_action(name)
    wanted = set(normalize_terms(f"{name} {' '.join(argument_names)}")) - {"mcp"}
    scored: list[tuple[float, int, str]] = []
    for index, (candidate, description, params) in enumerate(candidates):
        _, action = server_and_action(candidate)
        ratio = difflib.SequenceMatcher(None, wanted_action.lower(), action.lower()).ratio()
        have = set(normalize_terms(f"{candidate} {description} {' '.join(params)}"))
        overlap = len(wanted & have) / len(wanted) if wanted else 0.0
        scored.append((ratio + overlap, index, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored[:limit]]


def unknown_tool_message(name: str, closest: Sequence[str]) -> str:
    """The tool result for a call to `name`, which no tool has (R4).

    Plain words for the model that made the call, and for a person who reads it: what
    happened, and the closest tools it may call instead. No agent id, permission list,
    registry or exception text.
    """
    if not closest:
        return (
            f"There is no tool named '{name}', so nothing was run. "
            "Use only the tools listed in this request."
        )
    return (
        f"There is no tool named '{name}', so nothing was run. "
        f"The closest tools you can use are: {', '.join(closest)}."
    )
