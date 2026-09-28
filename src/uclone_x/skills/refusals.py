"""Why a skill is not used, as codes the Settings Skills panel words (#1720, #1777).

A refused skill is listed in the panel with the reason it is not used. The reason is shown
in the person's language, so the Core does not send a sentence for the panel to show: it
sends a `code` and its `params`, and the head writes the sentence from its catalog,
`frontend/src/i18n/locales/<language>/skills.json` under `notLoaded`, key for key. This is
the rule the conversation notices follow (`uclone_x.room.notices`, multilingual-ui.md §3.3).

`not_loaded_reason` is still filled, in English, from `SKILL_REFUSAL_FALLBACK`. It is what
the log, the CLI and a head that predates a code show. A code added here and not to the
head's catalog fails `tests/unit/test_skill_approval_pins.py`, not a user's screen.

Every sentence is plain words for the person: no digest, no path, no exception text. The one
technical thing a sentence may carry is the command to run, `ucx skill approve {name}`.
"""

from __future__ import annotations

from typing import Final, Literal

__all__ = [
    "SKILL_REFUSAL_FALLBACK",
    "SkillRefusalCode",
    "refusal_reason",
]

SkillRefusalCode = Literal[
    "changed_after_approval",
    "approved_before_pins",
    "never_approved",
    "failed_safety_check",
    "check_not_finished",
    "unreadable",
]
"""Every reason the store refuses a skill for. Closed: a head words exactly these."""

SKILL_REFUSAL_FALLBACK: Final[dict[SkillRefusalCode, str]] = {
    "changed_after_approval": (
        "This skill was changed after it was approved, so it is not used. To use it, check "
        "what changed, then approve it again in a terminal window: ucx skill approve {name}"
    ),
    "approved_before_pins": (
        "This skill was approved before UClone-X began keeping a record of approvals on this "
        "computer, so it is not used until you approve it again. In a terminal window, run: "
        "ucx skill approve {name}"
    ),
    "never_approved": (
        "This skill is marked ready to use, but it was never approved on this computer, so it "
        "is not used. To use it, approve it in a terminal window: ucx skill approve {name}"
    ),
    "failed_safety_check": "The safety check did not pass this skill, so it is not used.",
    "check_not_finished": (
        "The safety check could not read all of this skill's files, so it is not used. Check "
        "that its files can be opened, then start UClone-X again."
    ),
    "unreadable": (
        "This skill's instructions could not be read, so it is not used. They may be "
        "incomplete or edited by hand. Fix them, or install the skill again."
    ),
}


def refusal_reason(code: SkillRefusalCode, name: str) -> str:
    """The English sentence for `code`, filled with the skill's `name`."""
    return SKILL_REFUSAL_FALLBACK[code].format(name=name)
