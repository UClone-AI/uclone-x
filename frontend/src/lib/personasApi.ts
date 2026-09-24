/**
 * The two requests behind the persona editor (#892): read the catalogue, save a draft.
 * Kept apart from `personaDraft.ts` so the editor components never import a request.
 */
import type { PersonaCatalog, PersonaInfo } from '../types';
import { failureOf } from './coreFailure';
import {
  describeRefusal,
  type PersonaDraft,
  type PersonaEditMode,
  type PersonaEditorCopy,
  type PersonaSaveResult,
  type PromptDraftResult,
} from './personaDraft';

export const fetchPersonaCatalog = async (): Promise<PersonaCatalog> => {
  const res = await fetch('/api/personas');
  if (!res.ok) throw await failureOf(res);
  const data = (await res.json()) as Partial<PersonaCatalog>;
  return {
    personas: data.personas ?? [],
    available_tools: data.available_tools ?? [],
    personas_dir: data.personas_dir ?? null,
  };
};

export const savePersona = async (
  draft: PersonaDraft,
  mode: PersonaEditMode,
  copy: PersonaEditorCopy,
): Promise<PersonaSaveResult> => {
  const url = mode === 'create' ? '/api/personas' : `/api/personas/${encodeURIComponent(draft.name)}`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: mode === 'create' ? 'POST' : 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(draft),
    });
  } catch (err) {
    return { ok: false, message: copy.unreachable(String(err)) };
  }
  const body = (await res.json().catch(() => ({}))) as {
    persona?: PersonaInfo;
    live_agents_updated?: number;
    detail?: unknown;
  };
  if (!res.ok || !body.persona) {
    return { ok: false, message: describeRefusal(body.detail, res.status, copy) };
  }
  return { ok: true, persona: body.persona, liveAgentsUpdated: body.live_agents_updated ?? 0 };
};

export const synthesizePersonaPrompt = async (params: {
  name: string;
  role: string;
  description: string;
  allowed_tools: readonly string[];
}): Promise<PromptDraftResult> => {
  try {
    const res = await fetch('/api/personas/synthesize', {
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
      return { ok: false, message: data.detail || `HTTP ${res.status}` };
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

