"""What a turn's tools did, said plainly when the model then wrote nothing (#2322).

A turn whose reply is still empty after the re-ask ends with the plain `step.empty_reply`
sentence (#2264): nothing to show, send the message again. That is right when nothing was
done. It is wrong after a picture was made or a file written: told to send again, the
person asks for the same work twice. So when the turn's tools did work, the reply says
what was done instead, and the turn ends as an answered one.

Read from structure only, never from words: the files the turn's calls say they produced
(`ToolExecutionRecord.produced_paths`, nothing for a call that failed), and a produced
file's suffix to tell a picture from another file. Work that made no file is not counted:
`writes_files` is what a tool may do, declared for approval, not what a call did -- it is
true for `bash_run`, every MCP tool and `a2a_call`, so reading it said "I saved the
changes" after an `ls` (#2325 review). A turn that produced nothing keeps the stop.

The sentences carry no path, file name or tool name. English lives here
(`WORK_DONE_TEXT`); every other language is `locales/<language>/work_done.json`, keyed the
same, with the same `{count}`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import PurePosixPath
from typing import Final, Literal, cast

from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.i18n.language import Language

__all__ = [
    "PICTURE_SUFFIXES",
    "WORK_DONE_TEXT",
    "WorkDone",
    "WorkDoneKey",
    "work_done",
    "work_done_lines",
    "work_done_text",
]

#: A produced file with one of these suffixes is a picture.
PICTURE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"})

WorkDoneKey = Literal["work.picture", "work.pictures", "work.file", "work.files"]

#: Each line in English. Worded for the person: what was done and where it is, in the
#: conversation's own terms, never a path or a tool's name.
WORK_DONE_TEXT: Final[Mapping[WorkDoneKey, str]] = {
    "work.picture": "I made the picture and saved it with this conversation's files.",
    "work.pictures": "I made {count} pictures and saved them with this conversation's files.",
    "work.file": "I wrote the file and saved it with this conversation's files.",
    "work.files": "I wrote {count} files and saved them with this conversation's files.",
}


@dataclass(frozen=True)
class WorkDone:
    """What a turn's calls made: pictures, and other files."""

    pictures: int = 0
    files: int = 0


def work_done(records: Sequence[ToolExecutionRecord]) -> WorkDone:
    """What `records` did, read from their structure (see the module docstring)."""
    produced: list[str] = []
    for record in records:
        produced.extend(p for p in record.produced_paths if p not in produced)
    pictures = sum(1 for path in produced if PurePosixPath(path).suffix.lower() in PICTURE_SUFFIXES)
    return WorkDone(pictures=pictures, files=len(produced) - pictures)


@cache
def _korean_catalog() -> Mapping[WorkDoneKey, str]:
    source = resources.files("uclone_x.i18n").joinpath("locales").joinpath("ko")
    text = source.joinpath("work_done.json").read_text(encoding="utf-8")
    return cast("dict[WorkDoneKey, str]", json.loads(text))


def work_done_text(key: WorkDoneKey, language: Language) -> str:
    """The line `key` names, as a person reading `language` is told it."""
    if language == "ko":
        return _korean_catalog()[key]
    return WORK_DONE_TEXT[key]


def work_done_lines(done: WorkDone, language: Language) -> list[str]:
    """The lines saying what `done` did, in `language`; none when it did nothing."""
    lines: list[str] = []
    if done.pictures:
        key: WorkDoneKey = "work.picture" if done.pictures == 1 else "work.pictures"
        lines.append(work_done_text(key, language).format(count=done.pictures))
    if done.files:
        key = "work.file" if done.files == 1 else "work.files"
        lines.append(work_done_text(key, language).format(count=done.files))
    return lines
