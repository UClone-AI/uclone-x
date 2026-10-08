# `ucx-agent-skills/` — the runtime skill store

This directory is the skill store that the agent runtime reads. The `ucx skill` commands
(`list`, `audit`, `approve`, `synthesize`) resolve it as `<repository root>/ucx-agent-skills`,
and the web app and the CLI heads (`run`, `loop`, `room`, `a2a`, `acp`) load every package
whose manifest says `status: active` into the agent's skill registry at startup, after
auditing it again. A package that is not `active` is not loaded.

Each package is `ucx-agent-skills/<name>/SKILL.md` (YAML frontmatter followed by the
workflow in Markdown), with optional `scripts/` and `tests/` folders.

Because a loaded skill reaches every agent's `load_skill` tool and system prompt, this
directory is the most valuable place in a checkout for an attacker to persist. Treat a
change here as a change to the agent's behaviour, and review it as one.

## What ships here

Eleven runtime skills, each pinned to its approved digest in
`src/uclone_x/skills/shipped_pins.py`. A shipped skill whose bytes no longer match its pin
is not loaded, so changing a `SKILL.md` here means changing its pin too.

- `avatar`: how a clone draws candidates for its own profile picture, or one it sets
  directly, with `generate_image` and `set_avatar`.
- `media-character`, `media-architecture`, `media-engineering`: the image prompt rules for
  each subject domain, with one section per prompt family (`danbooru`, `prose`,
  `generic`). `load_skill` returns the preamble plus the active image model's section.
- `character-consistency`: character sheets and Visual DNA rules for keeping a character
  consistent across images.
- `remote_gpu_recovery`: diagnostic and recovery playbooks for a remote GPU machine
  running Ollama and ComfyUI.
- `art-brief-expansion`, `art-literal-spec`, `art-medium-restraint`, `art-iterative-edit`,
  `art-genre-vocab`: case skills for drawing. The model never picks these:
  `src/uclone_x/agent/artist_skill_router.py` chooses the one that fits the latest message
  and places its text after that message, for that turn only.

A skill whose steps call tools names them in `requires_tools`, and a clone is offered the
skill only when every one of them is in its tool scope. **Settings › Skills** shows which
clones cannot use a skill, and why.

Skills you add yourself are approved with `ucx skill approve`, which records their digest
in your own approvals ledger rather than in the pin table above.
