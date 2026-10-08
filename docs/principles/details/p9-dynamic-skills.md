# Principle 9: Self-Evolving & Pluggable Skills

## Detailed Specification & Rationale

* **Core Law**: Agents must possess dynamic, modular skill capabilities that can be autonomously synthesized from experience or explicitly augmented by developers.
* **Strict Rule**: 
  - **Autonomous Skill Synthesis**: When an agent discovers a successful multi-step workflow, automation pattern, or custom tool combination, it can autonomously propose it as a modular skill. A skill written this way is prompt-only: a single `SKILL.md` (frontmatter + instructions), with no scripts or tests. A clone's proposal (`propose_skill`) is written only into the store's `.pending/` area (`ucx-agent-skills/.pending/<name>/<version>/SKILL.md`) and is not loaded while it waits there. A person running `ucx skill synthesize` gets a package written with status `pending` (`ucx-agent-skills/<name>/SKILL.md`), which is not loaded until it is approved.
  - **Auditor Agent & Human Approval**: A proposed skill undergoes automated safety audit by a **Skill Auditor ucx agent**, and becomes active only by a person's decision. A clone's proposal becomes active when a person approves it in Settings, and approval installs exactly the version the person saw there, bound by its content digest, or nothing. A package from `ucx skill synthesize` becomes active when a person runs `ucx skill approve`. Neither path approves a skill automatically, and there is no setting that turns auto-approval on.
  - **Developer Extensibility**: Developers can inject, modify, or hot-reload skills on-demand via declarative files or CLI instructions.
  - **Sandboxed Execution**: Skills execute with configurable sandbox boundaries and schema validations.

* **Why**: Enables agents to progressively accumulate reusable capabilities without hardcoding tools into the core codebase.
* **Implementation Reference**: [`docs/public/overview.md`](../../public/overview.md)
