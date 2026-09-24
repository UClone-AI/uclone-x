/**
 * The persona write path's head side (#892): the draft an editor holds, the rules it can
 * check before a round trip, and the shape of the words it shows.
 *
 * Pure: no request is made from this module. The components in `components/personas/`
 * import only from here, and take data, callbacks, copy and icons as props; the requests
 * are in `personasApi.ts` and the words in `personaCopy.ts`, and `SettingsModal` is the one
 * place that wires the three together.
 *
 * The server is the authority on every rule here. The name rule is repeated so the editor
 * can say what is wrong before a round trip, but a save the server refuses is shown with the
 * server's own sentence, which names the file, the tool or the field it refused.
 */
import type { PersonaInfo, PersonaModelTier } from '../types';

/** What `POST /api/personas` and `PUT /api/personas/{name}` accept, field for field. */
export interface PersonaDraft {
  name: string;
  role: string;
  description: string;
  system_prompt: string;
  append_default_prompt: boolean;
  allowed_tools: string[];
  model_name: string | null;
  model_tier: PersonaModelTier;
  temperature: number;
  max_tokens: number | null;
  enable_write_tools: boolean;
  enable_subagent_tools: boolean;
}

export type PersonaEditMode = 'create' | 'edit';

export type PersonaSaveResult =
  | { ok: true; persona: PersonaInfo; liveAgentsUpdated: number }
  | { ok: false; message: string };

export const MODEL_TIERS: readonly PersonaModelTier[] = [
  'inherit',
  'fast',
  'pro',
  'flash_lite',
  'custom',
];

export const emptyPersonaDraft = (): PersonaDraft => ({
  name: '',
  role: '',
  description: '',
  system_prompt: '',
  append_default_prompt: false,
  allowed_tools: [],
  model_name: null,
  model_tier: 'inherit',
  temperature: 0.7,
  max_tokens: null,
  enable_write_tools: false,
  enable_subagent_tools: false,
});

/** The draft an edit starts from: exactly what the catalogue says the persona is. */
export const draftFromPersona = (persona: PersonaInfo): PersonaDraft => ({
  name: persona.name,
  role: persona.role ?? '',
  description: persona.description ?? '',
  system_prompt: persona.system_prompt ?? '',
  append_default_prompt: persona.append_default_prompt ?? false,
  allowed_tools: [...(persona.allowed_tools ?? [])],
  model_name: persona.model_name ?? null,
  model_tier: persona.model_tier ?? 'inherit',
  temperature: persona.temperature ?? 0.7,
  max_tokens: persona.max_tokens ?? null,
  enable_write_tools: persona.enable_write_tools ?? false,
  enable_subagent_tools: persona.enable_subagent_tools ?? false,
});

/**
 * The loader's name rule (`refuse_an_unusable_username`): the name is a file name and the
 * agent id, so lowercase letters, digits, `-` and `_`, starting and ending with a letter
 * or digit, at most 64 characters.
 */
const NAME_RULE = /^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$/;

/**
 * Why a draft cannot be saved yet, as one sentence with its remedy, or null when it can.
 * Checks only what the editor can know without the server: the name rule, a name already
 * taken on create, and the two fields a persona file must have.
 */
export const draftProblem = (
  draft: PersonaDraft,
  mode: PersonaEditMode,
  existingNames: readonly string[],
  copy: PersonaEditorCopy,
): string | null => {
  if (!NAME_RULE.test(draft.name)) return copy.nameRule;
  if (mode === 'create' && existingNames.includes(draft.name)) return copy.nameTaken(draft.name);
  if (!draft.role.trim()) return copy.roleRequired;
  if (!draft.system_prompt.trim()) return copy.promptRequired;
  return null;
};

interface ValidationIssue {
  loc?: (string | number)[];
  msg?: string;
}

/** A refusal's `detail` as one sentence: the server's own, or its field errors joined. */
export const describeRefusal = (detail: unknown, status: number, copy: PersonaEditorCopy): string => {
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const fields = (detail as ValidationIssue[]).map((issue) => {
      const field = issue.loc && issue.loc.length > 0 ? String(issue.loc[issue.loc.length - 1]) : '';
      return field ? `${field}: ${issue.msg ?? 'invalid'}` : (issue.msg ?? 'invalid');
    });
    return copy.fieldsRefused(fields.join('; '));
  }
  return copy.saveFailed(status);
};

/** Every word the persona editor shows. Passed in, so the components carry none of their own. */
export interface PersonaEditorCopy {
  sectionTitle: string;
  savedTo: (dir: string) => string;
  noWorkspace: string;
  /** `reason` is the Core's own words for the failure, or `''` when it gave none (#1436). */
  loadFailed: (reason: string) => string;
  emptyList: string;
  newPersona: string;
  edit: string;
  builtinBadge: string;
  overrideBadge: string;
  createTitle: string;
  editTitle: (name: string) => string;
  builtinEditNote: string;
  fields: {
    name: string;
    nameHint: string;
    role: string;
    description: string;
    systemPrompt: string;
    appendDefaultPrompt: string;
    tools: string;
    toolsHint?: string;
    toolsAllAllowedBadge?: string;
    toolsRestrictedBadge?: (count: number, total: number) => string;
    toolsResetToAll?: string;
    toolsRestrictedNotice?: (count: number) => string;
    model: string;
    modelDefault: string;
    modelOther: string;
    modelOtherPlaceholder: string;
    tier: string;
    temperature: string;
    temperaturePresets?: {
      precise: string;
      balanced: string;
      creative: string;
    };
    advancedSettings?: string;
    generateWithAi: string;
    generatingPrompt: string;
    generateHint: string;
    draftedByModel: (model: string) => string;
    draftedFromTemplate: (reason: string) => string;
    draftFailed: (reason: string) => string;
    undoDraft: string;
    maxTokens: string;
    maxTokensHint: string;
    modelSettingsHint: string;
    capabilities: string;
    enableWriteTools: string;
    enableSubagentTools: string;
    capabilitiesHint: string;
  };
  tierLabels: Record<PersonaModelTier, string>;
  noToolsSelected: string;
  noToolsAvailable: string;
  save: string;
  saving: string;
  cancel: string;
  nameRule: string;
  nameTaken: (name: string) => string;
  roleRequired: string;
  promptRequired: string;
  fieldsRefused: (fields: string) => string;
  saveFailed: (status: number) => string;
  unreachable: (reason: string) => string;
  saved: (name: string, liveAgents: number) => string;
}


/**
 * What the instructions-draft button got back. A template draft is a distinct outcome, not
 * a model draft with a different label: the editor must say that no model wrote it, and why.
 */
export type PromptDraftResult =
  | { ok: true; prompt: string; source: 'llm'; model: string }
  | { ok: true; prompt: string; source: 'template'; fallbackReason: string }
  | { ok: false; message: string };
