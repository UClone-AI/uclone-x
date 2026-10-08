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

The published tree ships the same packages, so the same table approves them there.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

__all__ = ["SHIPPED_SKILL_PINS"]

#: Skill name -> the digest `compute_skill_sha256` gives its approved package.
SHIPPED_SKILL_PINS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "art-brief-expansion": "5c7fa91a147979e0f148ad36797499ab3af6d0dfafd5fd20a7059aaf25a24053",
        "art-genre-vocab": "e87792ff66887cbfa24b0d0c521c72906c419edf1e08a719dcd1a3a92d215c93",
        "art-iterative-edit": "86044977ec4f7609687763565cc94951f817e7d74d7be521936960d1855c65c7",
        "art-literal-spec": "f6b175ae66adae88732307ef5443cf53af6be29d07b01b5320f9a29b345dfb95",
        "art-medium-restraint": "44fa39cc9c576be6e4c6f0997a6fe68e9c88c1e47efc06f6b5c88b71cc249e4b",
        "avatar": "d2ebe7f28075489746cb7bddb98f3427a4cbcf52b8299f08db74661d726ced6b",
        "character-consistency": "b5bc1c64072ec8f065e5efc065b5fec11117d1da0caf109e11e218168eb4f4e1",
        "media-architecture": "cdf0a7955d6e79ee3f6c5b279c159a1ea0021195e2539a2ee4a1e9d4aa3b4fb0",
        "media-character": "2c04d628e32a968803edf6aeaaaa49dda299579428c5cb8092c4e7e7815472a2",
        "media-engineering": "35efcaa8812c98bade1a0b8a61038c75a31de7a809fb1b52661df22bd29589c9",
        "remote_gpu_recovery": "0eeb339253abc7e765e6c8c5245c0625d4d7963e6dcc4cb100080f500fd93ce6",
    }
)
