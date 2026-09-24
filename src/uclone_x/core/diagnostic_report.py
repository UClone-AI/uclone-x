"""Rendering a recorded failure into something a person can send.

The journal collects; this renders. Nothing here transmits: it produces the
markdown a user reviews, and a URL that opens a *pre-filled* issue form which
the user still has to submit. That split is the privacy design. A report leaves
the machine because someone read it and pressed a button, not because a
background task decided to.

The URL form matters more than it looks. It needs no token, so nothing
embeddable can be stolen from the distribution and no server of ours has to
exist to receive anything; and the content is visible in the browser before it
is posted, so "what is being sent" is answerable by looking.
"""

from __future__ import annotations

import platform
import urllib.parse
from collections import Counter
from typing import Final

from uclone_x.core.failure_journal import FailureEntry, JournalRead, read_journal

#: Where reports go. The public repository, not the development one.
ISSUE_REPO: Final = "UClone-AI/uclone-x"

#: GitHub issue forms pre-fill from query parameters named for the form's field
#: ids, so these names are a contract with `.github/ISSUE_TEMPLATE/bug.yml`.
ISSUE_TEMPLATE: Final = "bug.yml"

#: A pre-filled issue travels in a URL, and a URL has a practical ceiling before
#: servers and browsers begin refusing it. Past this the report is written to a
#: file and the user attaches it instead, which loses nothing but a click.
MAX_URL_CHARS: Final = 6000


def environment_lines() -> list[str]:
    """The environment facts a report needs, and only those."""
    from uclone_x import __version__

    return [
        f"- UClone-X: {__version__}",
        f"- Python: {platform.python_version()}",
        f"- OS: {platform.system()} {platform.release()}",
    ]


def render_entry(entry: FailureEntry) -> str:
    """One failure, as markdown."""
    lines = [
        f"**`{entry.code}`** (`{entry.fingerprint}`) at {entry.ts}",
        "",
        f"> {entry.message}" if entry.message else "> (no message)",
    ]
    if entry.frames:
        lines += ["", "```", *entry.frames, "```"]
    if entry.context:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(entry.context.items()))
        lines += ["", f"Context: {rendered}"]
    return "\n".join(lines)


def summarise(entries: list[FailureEntry]) -> list[tuple[str, str, int]]:
    """`(fingerprint, code, count)` per distinct failure, most frequent first.

    The count is the part worth reporting. One occurrence of a configuration
    error is a user's afternoon; the same fingerprint forty times is a defect in
    how the product explains itself.
    """
    counts = Counter((entry.fingerprint, entry.code) for entry in entries)
    return [(fingerprint, code, count) for (fingerprint, code), count in counts.most_common()]


def render_report(read: JournalRead | None = None) -> str:
    """The full report body: environment, a summary, and the newest failure.

    Takes a `JournalRead`, not a list of entries, and that is the correction
    review asked for last. With a list, the renderer could not know that
    recording had stopped, so the banner appeared on screen while the body --
    the artifact that is saved, submitted, or copied out of "show what would be
    sent", and the only part a maintainer ever receives -- said nothing about
    it. The same substitution this module keeps being repaired for, displaced
    one surface outward onto the thing that actually travels.
    """
    read = read_journal() if read is None else read

    lines = ["## Environment", "", *environment_lines(), ""]

    if read.recording_blocked is not None:
        lines += [
            "## Recording is not working",
            "",
            f"**{read.recording_blocked}**",
            "",
            "Anything below is what was kept *before* that started.",
            "",
        ]

    if read.error is not None:
        lines += [
            "## Failures",
            "",
            f"**The failure journal could not be read.** {read.error}",
            "",
            "This is not the same as having nothing to report: whatever was recorded "
            "is still on disk and unreadable to this command.",
        ]
        if read.entries:
            lines += ["", f"{len(read.entries)} entries were readable before the error."]
        return "\n".join(lines)

    entries = list(read.entries)
    if read.unreadable_lines:
        lines += [
            f"> {read.unreadable_lines} journal line(s) could not be parsed and are not "
            "included below.",
            "",
        ]

    if not entries:
        lines += ["## Failures", "", "No failures recorded."]
        return "\n".join(lines)

    lines += ["## Failures recorded", ""]
    for fingerprint, code, count in summarise(entries):
        occurrence = "1 time" if count == 1 else f"{count} times"
        lines.append(f"- `{fingerprint}` {code} — {occurrence}")

    lines += ["", "## Most recent", "", render_entry(entries[-1])]
    return "\n".join(lines)


def report_title(entries: list[FailureEntry] | None = None) -> str:
    """An issue title carrying the fingerprint, so duplicates are searchable."""
    entries = list(read_journal().entries) if entries is None else entries
    if not entries:
        return "[ucx] Report"
    newest = entries[-1]
    return f"[ucx {newest.fingerprint}] {newest.code}"


def issue_url(
    body: str,
    title: str,
    *,
    repo: str = ISSUE_REPO,
    template: str = ISSUE_TEMPLATE,
) -> str | None:
    """A pre-filled `new issue` URL, or `None` when the body will not fit.

    `None` is not a failure to handle silently -- the caller falls back to
    writing the report to a file, and says so. Returning a URL that the browser
    or GitHub would truncate would post a report missing the part that mattered.
    """
    query = urllib.parse.urlencode(
        {"template": template, "title": title, "environment": body},
        quote_via=urllib.parse.quote,
    )
    url = f"https://github.com/{repo}/issues/new?{query}"
    return None if len(url) > MAX_URL_CHARS else url


def search_url(fingerprint: str, *, repo: str = ISSUE_REPO) -> str:
    """Where to check whether this failure is already reported.

    Offered instead of querying the API from the client: a search needs no
    consent and no rate-limit budget when the person does it, and the answer is
    more useful in a browser than as a line of CLI output.
    """
    query = urllib.parse.quote(f'repo:{repo} "{fingerprint}"')
    return f"https://github.com/search?q={query}&type=issues"
