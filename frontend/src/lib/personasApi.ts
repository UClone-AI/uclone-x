/**
 * The two requests behind the persona editor (#892): read the catalogue, save a draft.
 * Kept apart from `personaDraft.ts` so the editor components never import a request.
 */
import type { PersonaCatalog, PersonaInfo } from '../types';
import { failureOf } from './coreFailure';
import {
  describeRefusal,
  draftFromPersona,
  type PersonaDraft,
  type PersonaEditMode,
  type PersonaEditorCopy,
  type PersonaSaveResult,
  type PromptDraftResult,
} from './personaDraft';

/** `GET /api/clones`, whose `clones` are the catalogue's entries. */
export const fetchPersonaCatalog = async (): Promise<PersonaCatalog> => {
  const res = await fetch('/api/clones');
  if (!res.ok) throw await failureOf(res);
  const data = (await res.json()) as Partial<Omit<PersonaCatalog, 'personas'>> & { clones?: PersonaInfo[] };
  return {
    personas: data.clones ?? [],
    available_tools: data.available_tools ?? [],
    personas_dir: data.personas_dir ?? null,
    base_tools: data.base_tools ?? [],
    write_tools: data.write_tools ?? [],
  };
};

export const savePersona = async (
  draft: PersonaDraft,
  mode: PersonaEditMode,
  copy: PersonaEditorCopy,
): Promise<PersonaSaveResult> => {
  // An edit addresses the clone by its id; the body carries only what the clone is.
  const { id, ...body } = draft;
  const url = mode === 'create' ? '/api/clones' : `/api/clones/${encodeURIComponent(id || draft.name)}`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: mode === 'create' ? 'POST' : 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (_err) {
    return { ok: false, message: copy.unreachable };
  }
  const answer = (await res.json().catch(() => ({}))) as {
    persona?: PersonaInfo;
    live_agents_updated?: number;
    detail?: unknown;
  };
  if (!res.ok || !answer.persona) {
    return { ok: false, message: describeRefusal(answer.detail, res.status, copy) };
  }
  return { ok: true, persona: answer.persona, liveAgentsUpdated: answer.live_agents_updated ?? 0 };
};

export const synthesizePersonaPrompt = async (params: {
  name: string;
  role: string;
  description: string;
  allowed_tools: readonly string[];
}): Promise<PromptDraftResult> => {
  try {
    const res = await fetch('/api/clones/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = (await res.json().catch(() => ({}))) as {
      system_prompt?: string;
      source?: string;
      model?: string;
      fallback_reason?: string;
      detail?: string;
    };
    if (!res.ok) {
      return { ok: false, message: data.detail || 'The draft could not be generated.' };
    }
    const prompt = data.system_prompt ?? '';
    if (!prompt) return { ok: false, message: 'the server returned no draft' };
    if (data.source === 'llm') {
      return { ok: true, prompt, source: 'llm', model: data.model || 'the model' };
    }
    return {
      ok: true,
      prompt,
      source: 'template',
      fallbackReason: data.fallback_reason || 'no reason given',
    };
  } catch (err) {
    return { ok: false, message: String(err) };
  }
};


/**
 * "Use system default" on a turn that failed on a clone's own model (model-gateway.md §3.6):
 * the clone's slots that name `ref` are cleared, so they follow the default again, and every
 * other field is sent back as the Core holds it. A clone with no slot naming `ref` any more
 * has its conversation model cleared, the slot the failure is about. Rejects with the Core's
 * refusal.
 */
export const clearCloneModel = async (clone: string, ref: string): Promise<PersonaInfo> => {
  const read = await fetch(`/api/clones/${encodeURIComponent(clone)}`);
  if (!read.ok) throw await failureOf(read);
  const { persona } = (await read.json()) as { persona: PersonaInfo };
  const draft = draftFromPersona(persona);
  const slots = ['model_name', 'fast_model', 'image_model'] as const;
  const named = slots.filter((slot) => draft[slot] === ref);
  for (const slot of named.length > 0 ? named : (['model_name'] as const)) draft[slot] = null;
  const { id, ...fields } = draft;
  const saved = await fetch(`/api/clones/${encodeURIComponent(id || draft.name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(fields),
  });
  if (!saved.ok) throw await failureOf(saved);
  return ((await saved.json()) as { persona: PersonaInfo }).persona;
};
