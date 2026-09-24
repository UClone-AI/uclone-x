# Changelog

What changed in each released version of `uclone-x`, and what that change means for
someone installing it. Internal refactors, test work and build tooling are deliberately
absent: a reader of this file is deciding whether to upgrade, not reviewing the diff.

**Entries are assembled when the version is bumped, not when a change merges.** One entry
per released version, written from the changes that went into that release. A repository
check refuses a version bump that adds no entry, so a release with no entry cannot ship.
There is no `Unreleased` section for the same reason: an entry exists once the version it
names does.

Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Each date is
the date that version was published.

## [0.2.1] - 2026-09-24

### Added

- A one-line install. `curl -fsSL https://raw.githubusercontent.com/UClone-AI/uclone-x/main/install.sh | bash`
  asks its questions on the terminal even when piped, puts a `ucx` command in
  `~/.local/bin` (and says how to add that folder to `PATH` when it is not there), and
  ends by offering to start UClone-X. `--no-start` skips that last question.
- Autonomous discussion in group chats. A toggle in the conversation header lets the
  clones keep talking among themselves. It pauses while you are not viewing the
  conversation, resumes when you come back, and stops after 20 turns in a row without you.
- While a clone runs a tool, the conversation shows which one instead of "Thinking...".
- The artist clone can remember characters: a `character_sheet` tool saves each
  character's look (tags, description, seed) as a file under `characters/` in the
  workspace, reuses it in later images, and combines several saved characters into one
  scene prompt. It now writes image prompts in English and chooses the aspect ratio from
  the subject.
- In a group chat, the model that picks the next speaker is now told to let another clone
  answer a question one clone asks, instead of ending the exchange, and to start a
  discussion when you ask the clones to talk among themselves.
- With developer mode on, a turn shows the model calls it made: model, tokens, duration
  and tools called for each, and on expanding one, the request, the tools offered and the
  response.

### Fixed

- Choosing the next speaker in a group chat now asks Ollama to turn a reasoning model's
  thinking off (for example Qwen3), so it replies without spending its budget on
  reasoning first. A selector that runs out of tokens now says so instead of reporting a
  missing JSON object.
- The message shown when no model is available points to Settings, and to `ucx llm status`
  from a terminal, instead of a `./ucx` command that does not exist in an installed copy.

## [0.2.0] - 2026-09-24

### Added

- A beginner installer. `install.sh` sets up uv and Python on a Mac that has neither,
  installs the core, reports how much memory the machine has and which local models fit
  in it, and offers the image engine with its download size stated.
- `ucx install` downloads the local language model and the image checkpoint, then checks
  that each one actually works. It exits successfully only when every requested part is
  ready, and reports a partial setup as incomplete. An interrupted checkpoint download
  resumes, and a truncated or corrupt file is refused.
- `ucx start` inspects the machine's hardware, prefers a private local model through
  Ollama, and opens the dashboard scoped to the current folder.
- The dashboard lists your conversations: one-to-one chats with each clone and group
  chats with several, most recent first, with pinning, rename and delete
  on each row.
- Six built-in clones ship with the package: clone (the default), writer, artist,
  guardian, pioneer and scout, each with its own avatar. A clone can be created and edited
  from Settings or from the workspace rail, including drafting its instructions with the
  connected model, and is saved as a YAML file in the workspace.
- Replies stream as they are produced, including the clone's reasoning and each tool call
  it makes. Messages render Markdown and math, and generated images and search results
  appear as cards in the conversation.
- A context window gauge shows how full the conversation is, and suggests compacting it
  before it overflows. Conversation history can be rewound or cleared from the
  conversation itself.
- The workspace dock shows, for the conversation on screen, each turn's tools and files,
  the documents and images it produced, and a Remembers tab listing the facts each clone
  saved to its memory.
- Every clone has memory tools to record, look up and retract facts, and remembers across
  conversations, in its own directory under `UCLONE_AGENTS_DIR`. Every clone can also
  list folders and read files, including folders you mark as read-only in Settings;
  writing stays limited to the workspace.
- Connect external MCP servers from Settings, either a remote server by URL or a local
  one by command, or by pasting a vendor's `mcpServers` snippet. Their tools are
  available from the next turn, without a restart.
- vLLM is a language model provider in its own right, rather than being reported as
  OpenAI.
- Ollama models can be installed and removed from Settings, and removed with
  `ucx llm rm`.
- A different model can be chosen for one clone in the current session without changing
  the default for the others.
- Image generation runs in-process by default with no separate service to start, and uses
  a running ComfyUI instead when one is found. Prompts and image sizes are adapted to the
  model being used.
- A clone can install a missing optional component itself, from a fixed list of
  packages, instead of telling you which command to run.
- `ucx loop run` and the `/loop` command repeat a prompt on an interval.
- `ucx acp serve` lets an editor that speaks the Agent Client Protocol talk to a clone.

### Changed

- Settings gains Skills, a Developer mode switch, and a Diagnostics section. Developer
  mode is off by default, and the developer tools in the dock appear only when it is on.
- A half-written message is kept when you switch conversations, and the dashboard is
  usable in a narrow window.
- Agent names must be lowercase letters, digits, `-` and `_`. A name that breaks the
  rule, and a request that names no agent at all, is refused with a message saying why,
  instead of being answered by a default agent.

### Removed

- Cost tracking. Token usage is still counted and limited, but no prices are calculated,
  and the dashboard no longer shows a cost ceiling or per-provider spend.

### Fixed

- Failures in conversations, Settings, MCP connections and Diagnostics are described in
  plain words, and a read that fails says so instead of showing an empty list.
- A failed or cancelled turn leaves nothing half-done behind: it is shown as failed, is
  not saved as if it had succeeded, and Retry is not offered when retrying cannot help.
- A group chat's clones resume where they left off after a restart, and what each has
  learned in the conversation is kept.
- A long tool result or a nearly full context window no longer breaks a turn; the
  conversation is compacted against the context window the local model actually serves.
  A step whose tool results cannot fit is refused with a message saying why, and the turns
  after it stay within the window.
- The conversation no longer jumps while a reply streams, or pulls you back to the bottom
  while you are reading earlier messages.
- A model download that stalls times out, instead of waiting forever.
- A clone's declared tools are the tools it can actually use, and a clone never gains
  write or sub-agent tools it was not given.
- `ucx run` prints the reply verbatim, reports a failed turn as an error on stderr with a
  nonzero exit, and exits 3 when the session could not be saved.

## [0.1.2] - 2026-09-12

### Fixed

- `pip install uclone-x` followed by `ucx` no longer ends in a traceback. An install
  without the optional extra a command needs now reports the missing extra as one
  sentence naming what to install. In 0.1.1 the same situation produced a twenty-line
  traceback.
- `ucx --version` is an option. The contributing guide had been telling bug reporters to
  run it while it did not exist.
- Resetting a session clears the durable event queue, so events queued before the reset
  are no longer delivered after it.
- Tool-call arguments that arrive wrapped in an extra layer are unwrapped before they are
  JSON-encoded, instead of being re-encoded nested.

### Added

- Multi-agent rooms: several agents share one conversation, with a selection seam, a store
  and an orchestrator deciding who speaks next.
- A tool call that found nothing now says so, so a miss is distinguishable from a tool
  that was never consulted.
- Failures can be recorded locally, with consent, and reported in one step.

## [0.1.1] - 2026-09-11

### Added

- Installs on Python 3.13. 0.1.0 does not.
- Personas can be defined and read at runtime, through `define_persona` and `get_persona`.
- Turn-loop events are written on the durable path, and the event log carries its version
  from the first line, so an unrecognised event fails closed rather than being ignored.

### Fixed

- The A2A server and the UI start without the `http` extra installed, and say which extra
  to install, instead of failing on an import.
- Credentials are redacted as they are written to the append-only log and to session
  state, rather than only when they are read back.
- The server no longer hangs on shutdown while a stream is still open.
- A missing local-model CLI, or a daemon that is not reachable, is diagnosed before a
  model pull is attempted rather than during it.
- A run that specifies conflicting step and turn limits is rejected instead of silently
  choosing one of them.
- Steps taken by a child agent are deducted from its parent's budget.

## [0.1.0] - 2026-09-09

First open-source release.

**No itemised record of this version's contents exists.** Nothing was kept per change
before this release, so the list above starts at 0.1.1. This entry is a marker rather than
a reconstruction: anything more specific would be invented.
