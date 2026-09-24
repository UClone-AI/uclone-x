"""Skills subsystem: SKILL.md dynamic packages, autonomous synthesis, and security auditor."""

from uclone_x.skills.auditor import (
    Skill,
    SkillAuditor,
    SkillRegistry,
    compute_skill_sha256,
    load_skill_from_dir,
    manifest_from_dict,
    parse_skill_markdown,
    save_skill,
    serialize_skill_markdown,
)
from uclone_x.skills.models import (
    AuditVerdict,
    AutoApprovalPolicy,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.skills.protocols import (
    SkillAuditorProtocol,
    SkillProtocol,
    SkillRegistryProtocol,
    SkillSynthesizerProtocol,
)
from uclone_x.skills.synthesizer import SkillSynthesizer

__all__ = [
    "AuditVerdict",
    "AutoApprovalPolicy",
    "Skill",
    "SkillAuditReport",
    "SkillAuditor",
    "SkillAuditorProtocol",
    "SkillManifest",
    "SkillOrigin",
    "SkillProtocol",
    "SkillRegistry",
    "SkillRegistryProtocol",
    "SkillStatus",
    "SkillSynthesizer",
    "SkillSynthesizerProtocol",
    "compute_skill_sha256",
    "load_skill_from_dir",
    "manifest_from_dict",
    "parse_skill_markdown",
    "save_skill",
    "serialize_skill_markdown",
]
