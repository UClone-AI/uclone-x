/**
 * The persona editor's words (#892), passed to the components as a prop so they carry none.
 * Each refusal names its reason and its remedy; each absence names its cause.
 *
 * The user-read noun is **clone** (design doc `unified-conversations-and-room-ui.md` §3.2.2
 * R3, owner ruling). It replaces `agent` and `persona`, which were two words this head used
 * for one thing a user sees. The sweep stops at the wire: `PersonaEditorCopy`,
 * `PERSONA_EDITOR_COPY`, `AgentInfo`, `PersonaInfo` and `GET /api/personas` keep their names,
 * because a second head reads the same API (P8). What is renamed is what is read, never what
 * is typed.
 */
import type { PersonaEditorCopy } from './personaDraft';

export const PERSONA_EDITOR_COPY: PersonaEditorCopy = {
  sectionTitle: 'Clones',
  savedTo: (dir) => `Saved as files in ${dir}`,
  noWorkspace: 'This runtime has no workspace folder, so clones cannot be saved here.',
  loadFailed: (reason) =>
    ['Could not load clones.', reason, 'Close and reopen Settings to retry.'].filter(Boolean).join(' '),
  emptyList: 'No clones are set up yet. Press New clone to add one.',
  newPersona: 'New clone',
  edit: 'Edit',
  builtinBadge: 'Built-in',
  overrideBadge: 'Customized',
  createTitle: 'New clone',
  editTitle: (name) => `Edit ${name}`,
  builtinEditNote:
    'This clone ships with the app. Saving keeps your version in the workspace folder and leaves the original untouched.',
  fields: {
    name: 'Name',
    nameHint: 'Lowercase letters, digits, - and _. It is also the file name.',
    role: 'Role',
    description: 'Description',
    systemPrompt: 'Instructions',
    appendDefaultPrompt: "Add the app's standard instructions after these",
    tools: 'Tools',
    toolsHint:
      'By default, all runtime tools are available. Select tools below only to restrict this clone to a specific allowlist.',
    toolsAllAllowedBadge: 'All tools allowed',
    toolsRestrictedBadge: (count: number, total: number) => `Restricted allowlist (${count}/${total})`,
    toolsResetToAll: 'Allow all tools',
    toolsRestrictedNotice: (count: number) =>
      `Restricted mode: Only the ${count} selected tool${count === 1 ? '' : 's'} will be available. All other tools are disabled.`,
    model: 'Model',
    modelDefault: 'Use the runtime default',
    modelOther: 'Other model…',
    modelOtherPlaceholder: 'Model tag, e.g. qwen3:8b',
    tier: 'Tier',
    temperature: 'Temperature',
    temperaturePresets: {
      precise: 'Precise (0.2)',
      balanced: 'Balanced (0.7)',
      creative: 'Creative (1.0)',
    },
    advancedSettings: 'Advanced settings',
    generateWithAi: 'Generate with AI',
    generatingPrompt: 'Generating…',
    generateHint: 'Write the instructions from the name, role and description, using the connected model',
    draftedByModel: (model) => `Drafted by ${model}. Read it over before saving.`,
    draftedFromTemplate: (reason) =>
      `No model answered (${reason}), so this is a basic template filled in from your fields. Check the model connection in Settings for a written draft.`,
    draftFailed: (reason) => `Could not draft instructions (${reason}). Your text was not changed.`,
    undoDraft: 'Undo',
    maxTokens: 'Max tokens',
    maxTokensHint: 'Leave blank for no limit.',
    modelSettingsHint: 'Model settings apply to conversations started after saving.',
    // These two read exactly as `BaseAgent._capability_refusal` quotes them: a refused tool
    // tells the person to turn on "Allow file-writing tools" / "Allow sub-agents" here, and
    // that instruction is only followable while the switch is named the same (#1167).
    capabilities: 'Capabilities',
    enableWriteTools: 'Allow file-writing tools',
    enableSubagentTools: 'Allow sub-agents',
    capabilitiesHint:
      'Turning one off removes those tools from this clone, and from any helper it starts, even when the list above names them.',
  },
  tierLabels: {
    inherit: 'Same as the runtime',
    fast: 'Fast',
    pro: 'Pro',
    flash_lite: 'Flash Lite',
    custom: 'Custom',
  },
  noToolsSelected: 'No tools selected — this clone can use every tool the runtime has.',
  noToolsAvailable: 'The runtime reports no tools, so there are none to choose from.',
  save: 'Save',
  saving: 'Saving…',
  cancel: 'Cancel',
  nameRule:
    'The name must be lowercase letters, digits, - or _, start and end with a letter or digit, and be at most 64 characters. Change the name to fit.',
  nameTaken: (name) => `A clone named ${name} already exists. Choose another name, or edit that one.`,
  roleRequired: 'The role is empty. Describe what this clone does in a few words.',
  promptRequired: 'The instructions are empty. Write what this clone should do.',
  fieldsRefused: (fields) => `The server refused these fields: ${fields}. Correct them and save again.`,
  saveFailed: (status) => `The save failed (HTTP ${status}). Check the server log, then save again.`,
  unreachable: (reason) => `The server could not be reached (${reason}). Check it is running, then save again.`,
  saved: (name, liveAgents) =>
    liveAgents > 0
      ? `Saved ${name}. ${liveAgents} open conversation${liveAgents === 1 ? '' : 's'} will use it from the next message.`
      : `Saved ${name}.`,
};
