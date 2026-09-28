"""The approved digest of each skill this repository ships in `ucx-agent-skills/` (#1720).

A person approves their own skills with `ucx skill approve`, which pins the approved digest
in their approvals ledger (`uclone_x.skills.approvals`). The skills that ship with the code
cannot be approved that way: nobody is at a terminal when they are installed. Their approval
is this table instead. It is code, so a change to it is reviewed like any other change, and
it lives outside the skill packages, so editing a package cannot also edit its approval.

A shipped skill is loaded only while its bytes still hash, under the rule in
`compute_skill_sha256`, to the digest here. Changing a shipped `SKILL.md` therefore means
changing its line here in the same change, which is the reviewable act of approving it; a
unit test recomputes every shipped skill's digest and fails when a skill changed and its
digest here did not.

The published tree ships no skill packages, so there these entries match nothing and
approve nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

__all__ = ["SHIPPED_SKILL_PINS"]

#: Skill name -> the digest `compute_skill_sha256` gives its approved package.
SHIPPED_SKILL_PINS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "avatar": "32610294684963ee70a259549aeba6ac37e5f3046104aae8f388e943d4555fb2",
        "character-consistency": "40921a01394d5c69b7fa72a8a6a44928971f22a48bcaa06aabbfae904c43b595",
        "media-architecture": "f256d0a10ea619fbe4bce794d5f084c89168b64feb327521cfc5aeeb642bba3a",
        "media-character": "8bd8c0f04009569343c4c0e22b88ea2b6f51b05365c1da77625e0aed3f012087",
        "media-engineering": "91225f24df306133e22824d52c5f726c25a253d5744317be54c1a623d12893eb",
        "remote_gpu_recovery": "c1f598b77d47981aee14d7961973135f5e5e6069bfd34bc3f599c8d5ef3b5044",
    }
)
