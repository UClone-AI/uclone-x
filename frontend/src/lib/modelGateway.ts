/**
 * The model gateway's head side (model-gateway.md §3.7.1): the shapes of its routes, the
 * requests, and the pure rules a picker needs to say what a saved model ref means now.
 *
 * A model ref is `<connection id>/<model id>`, split at the **first** `/`: a model id may
 * contain `/` (`meta-llama/Llama-3.3-70B`), a connection id may not (§3.1).
 *
 * Every refusal is the Core's plain-words body, read through `failureOf`; a caller shows it
 * with `plainFailure`, never the status code.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { failureOf } from './coreFailure';

/** `unsupported`: a saved row of a kind this version does not know, listed so it is never lost. */
export type ConnectionStatus = 'connected' | 'no_key' | 'key_rejected' | 'unreachable' | 'unchecked' | 'unsupported';

export const CONNECTION_STATUSES: readonly ConnectionStatus[] = [
  'connected',
  'no_key',
  'key_rejected',
  'unreachable',
  'unchecked',
  'unsupported',
];

/** A status the head has words for; a value added later reads as not checked yet. */
export const statusOf = (raw: unknown): ConnectionStatus =>
  (CONNECTION_STATUSES as readonly unknown[]).includes(raw) ? (raw as ConnectionStatus) : 'unchecked';

export type Capability = 'chat' | 'image_create' | 'image_edit' | 'image_input';

/** `GET /api/connections`'s `Connection`. */
export interface Connection {
  id: string;
  kind: string;
  label: string;
  base_url: string | null;
  key_set: boolean;
  key_masked: string | null;
  source: 'settings' | 'env';
  /** The variable that sets it, when `source` is `"env"` (§3.7.1 extension, step 4). */
  env_var?: string | null;
  /**
   * The variable that supplies the key of a saved (`"settings"`) row (§3.7.1 Rev 1): the row
   * stays editable, but a key saved here is not the one used while the variable is set.
   */
  key_env_var?: string | null;
  paid: boolean;
  status: ConnectionStatus;
  detail: string | null;
  model_count: number | null;
}

/** `GET /api/connections`'s `Kind`: what a new connection of this kind asks for. */
export interface ConnectionKind {
  kind: string;
  label: string;
  needs_key: boolean;
  needs_base_url: boolean;
  default_base_url: string | null;
  key_url: string | null;
  capabilities: Capability[];
}

export interface ConnectionsResponse {
  connections: Connection[];
  kinds: ConnectionKind[];
}

export type ModelSlot = 'model_name' | 'fast_model' | 'image_model';
export type DefaultSlot = 'deep' | 'fast' | 'image';

/** `GET /api/connections/{id}/dependents`: what removing it would leave without a model. */
export interface ConnectionDependents {
  clones: { id: string; name: string; slots: ModelSlot[] }[];
  defaults: DefaultSlot[];
}

export interface ModelEntry {
  ref: string;
  id: string;
  /** The listing's own name for it; `null` when it gives none (the id is shown then). */
  display_name: string | null;
  capabilities: string[];
  context_window: number | null;
}

/** One connection's part of the model set: its models, or why it has none to offer. */
export interface ModelGroup {
  connection_id: string;
  label: string;
  kind: string;
  status: ConnectionStatus;
  detail: string | null;
  models: ModelEntry[];
}

export interface DefaultModels {
  deep: string | null;
  fast: string | null;
  /** A ref, or `auto`. */
  image: string;
}

/** `GET /api/models?capability=chat|image`. */
export interface ModelSet {
  groups: ModelGroup[];
  defaults: DefaultModels;
  recommended: { deep: string | null; fast: string | null };
}

export type ModelSetCapability = 'chat' | 'image';

/** The picture slot's value meaning "the first one that is ready" (§3.5). */
export const AUTO_IMAGE = 'auto';

// ---------------------------------------------------------------------------------------------
// Refs
// ---------------------------------------------------------------------------------------------

/** A ref's two halves, split at the first `/`; `null` when it names no connection. */
export const parseRef = (ref: string): { connectionId: string; modelId: string } | null => {
  const at = ref.indexOf('/');
  if (at <= 0 || at === ref.length - 1) return null;
  return { connectionId: ref.slice(0, at), modelId: ref.slice(at + 1) };
};

/** The listed entry a ref names, or `null` when no connection lists it now. */
export const findModel = (set: ModelSet | null, ref: string | null): ModelEntry | null => {
  if (!set || !ref) return null;
  for (const group of set.groups) {
    const hit = group.models.find((m) => m.ref === ref);
    if (hit) return hit;
  }
  return null;
};

/** What a person reads for a ref: its listed display name, else the ref as written. */
export const modelName = (set: ModelSet | null, ref: string): string => {
  const hit = findModel(set, ref);
  return hit ? hit.display_name || hit.id : ref;
};

/**
 * Whether a saved ref can be used now. `listed` when a connection lists it; otherwise
 * `unavailable` (its connection is gone, offline, or does not list that model). Only asked
 * once the set has loaded: an unloaded set says nothing about any ref.
 */
export const refState = (set: ModelSet, ref: string): 'listed' | 'unavailable' =>
  findModel(set, ref) ? 'listed' : 'unavailable';

/** The groups that have models to pick from. */
export const pickableGroups = (set: ModelSet): ModelGroup[] =>
  set.groups.filter((g) => g.models.length > 0);

/** The groups that cannot offer a model now, each to be shown with its reason. */
export const silentGroups = (set: ModelSet): ModelGroup[] =>
  set.groups.filter((g) => g.models.length === 0);

/** A pasted API key without the whitespace and surrounding quotes a copy often brings along. */
export const sanitizeApiKey = (key: string): string => {
  let cleaned = key.trim();
  if ((cleaned.startsWith('"') && cleaned.endsWith('"')) || (cleaned.startsWith("'") && cleaned.endsWith("'"))) {
    cleaned = cleaned.slice(1, -1).trim();
  }
  return cleaned;
};

// ---------------------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------------------

const jsonInit = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  ...(body === undefined ? {} : { body: JSON.stringify(body) }),
});

const answer = async <T>(res: Response): Promise<T> => {
  if (!res.ok) throw await failureOf(res);
  return (await res.json()) as T;
};

const connectionPath = (id: string) => `/api/connections/${encodeURIComponent(id)}`;

export const fetchConnections = async (): Promise<ConnectionsResponse> => {
  const data = await answer<Partial<ConnectionsResponse>>(await fetch('/api/connections'));
  return { connections: data.connections ?? [], kinds: data.kinds ?? [] };
};

/** Adds a connection. The key travels in the body, never the URL. */
export const addConnection = async (fields: {
  kind: string;
  label?: string;
  base_url?: string;
  key?: string;
}): Promise<Connection> => answer<Connection>(await fetch('/api/connections', jsonInit('POST', fields)));

export const updateConnection = async (
  id: string,
  fields: { label?: string; base_url?: string; key?: string },
): Promise<Connection> => answer<Connection>(await fetch(connectionPath(id), jsonInit('PATCH', fields)));

export const checkConnection = async (id: string): Promise<Connection> =>
  answer<Connection>(await fetch(`${connectionPath(id)}/check`, jsonInit('POST')));

export const fetchDependents = async (id: string): Promise<ConnectionDependents> => {
  const data = await answer<Partial<ConnectionDependents>>(await fetch(`${connectionPath(id)}/dependents`));
  return { clones: data.clones ?? [], defaults: data.defaults ?? [] };
};

export const removeConnection = async (id: string): Promise<void> => {
  await answer<unknown>(await fetch(connectionPath(id), jsonInit('DELETE')));
};

export const fetchModelSet = async (capability: ModelSetCapability, refresh = false): Promise<ModelSet> => {
  const data = await answer<Partial<ModelSet>>(
    await fetch(`/api/models?capability=${capability}&refresh=${refresh ? 1 : 0}`),
  );
  return {
    groups: (data.groups ?? []).map((g) => ({ ...g, status: statusOf(g.status), models: g.models ?? [] })),
    defaults: {
      deep: data.defaults?.deep ?? null,
      fast: data.defaults?.fast ?? null,
      image: data.defaults?.image ?? AUTO_IMAGE,
    },
    recommended: { deep: data.recommended?.deep ?? null, fast: data.recommended?.fast ?? null },
  };
};

/** One model an Ollama connection has installed; `chat` false for an embedder (#2167). */
export interface InstalledModel {
  id: string;
  chat: boolean;
}

/** What the Ollama connection `id` has installed, embedders included. */
export const fetchInstalledModels = async (id: string): Promise<InstalledModel[]> => {
  const data = await answer<{ models?: InstalledModel[] }>(
    await fetch(`/api/models/installed?connection_id=${encodeURIComponent(id)}`),
  );
  return data.models ?? [];
};

/**
 * Installs `model` on the Ollama connection `id`. The request waits for the whole download;
 * `signal` stops the wait, not the download. `joined` is true when an install of the same
 * model was already running and this one waited for it.
 */
export const installModel = async (id: string, model: string, signal?: AbortSignal): Promise<{ joined: boolean }> => {
  const data = await answer<{ joined?: boolean }>(
    await fetch('/api/models/pull', { ...jsonInit('POST', { model, connection_id: id }), signal }),
  );
  return { joined: data.joined === true };
};

/** Removes `model` from the Ollama connection `id`. */
export const removeInstalledModel = async (id: string, model: string): Promise<void> => {
  await answer<unknown>(await fetch('/api/models/delete', jsonInit('POST', { model, connection_id: id })));
};

/** Saves one default slot alone (S1: one writer, one field per save). */
export const saveDefaultModel = async (slot: DefaultSlot, ref: string | null): Promise<void> => {
  await answer<unknown>(await fetch('/api/settings', jsonInit('POST', { default_models: { [slot]: ref } })));
};

/**
 * One capability's model set, read when `enabled` and again on `reload()` or a change of
 * `version` (a connection added, checked or removed elsewhere on the screen).
 */
export const useModelSet = (
  capability: ModelSetCapability,
  enabled: boolean,
  version = 0,
): { set: ModelSet | null; failed: unknown; loading: boolean; reload: (refresh?: boolean) => void } => {
  const [set, setSet] = useState<ModelSet | null>(null);
  const [failed, setFailed] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const seq = useRef(0);
  const load = useCallback(
    (refresh = false) => {
      const mine = ++seq.current;
      setLoading(true);
      fetchModelSet(capability, refresh).then(
        (data) => {
          if (mine !== seq.current) return;
          setSet(data);
          setFailed(null);
          setLoading(false);
        },
        (err: unknown) => {
          if (mine !== seq.current) return;
          console.error(`Failed to read the ${capability} models:`, err);
          setFailed(err ?? new Error('failed'));
          setLoading(false);
        },
      );
    },
    [capability],
  );
  useEffect(() => {
    if (enabled) load();
  }, [enabled, load, version]);
  return { set, failed, loading, reload: load };
};
