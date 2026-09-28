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
        "avatar": "0a2a85e5d5bd4ac8e3a2f2577818b2f0b0b5eecc390f943c609aa49367c3eb9b",
        "character-consistency": "8e25a4c8b94cb003e199e486e3e4e5a09d3e3e12f944794e8bda2f1a43666a4b",
        "media-architecture": "18b43d798492e1e2dfe957e6ea450c5cc2b04fb724197d543245d8f1c3df0cb6",
        "media-character": "1e9c5abac180f0a2af498c79e1d7026c2b85b57f48b9feef52ce61ec1f8310ea",
        "media-engineering": "054e8ac12a1fc610c49696e4fcf7468b7c2d03cee370486c33595c51a7caa836",
        "remote_gpu_recovery": "c1f598b77d47981aee14d7961973135f5e5e6069bfd34bc3f599c8d5ef3b5044",
    }
)
