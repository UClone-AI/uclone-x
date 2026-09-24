# Principle 9: Self-Evolving & Pluggable Skills

## Detailed Specification & Rationale

* **Core Law**: Agents must possess dynamic, modular skill capabilities that can be autonomously synthesized from experience or explicitly augmented by developers.
* **Strict Rule**: 
  - **Autonomous Skill Synthesis**: When an agent discovers a successful multi-step workflow, automation pattern, or custom tool combination, it can autonomously package and persist it as a modular skill (`ucx-agent-skills/<name>/SKILL.md` + scripts/tests).
  - **Auditor Agent & Configurable Auto-Approval**: Newly synthesized skills undergo automated safety audit by a **Skill Auditor ucx agent**. Auto-approval behavior is fully configurable by the developer (`always`, `safe_only`, `never`) with human notification gates for high-risk actions.
  - **Developer Extensibility**: Developers can inject, modify, or hot-reload skills on-demand via declarative files or CLI instructions.
  - **Sandboxed Execution**: Skills execute with configurable sandbox boundaries and schema validations.

* **Why**: Enables agents to progressively accumulate reusable capabilities without hardcoding tools into the core codebase.
* **Implementation Reference**: `docs/skill-system-architecture.md`
