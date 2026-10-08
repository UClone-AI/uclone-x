import React, { useCallback, useEffect, useState } from 'react';
import { ExternalLink, Loader2, Plug, Plus, RefreshCw, Trash2 } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { plainFailure } from '../../lib/coreFailure';
import {
  addConnection,
  checkConnection,
  fetchConnections,
  fetchDependents,
  removeConnection,
  sanitizeApiKey,
  updateConnection,
  type Connection,
  type ConnectionDependents,
  type ConnectionKind,
  type ConnectionsResponse,
} from '../../lib/modelGateway';
import { Button } from '../ui/Button';
import { OllamaModels } from './OllamaModels';
import { ReadLoading } from './ReadState';
import { statusLine } from './gatewayWords';

const inputClass =
  'w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors';

type Notice = { tone: 'ok' | 'error'; text: string } | null;

/** A removal in progress: reading what depends on it, then asking. */
type Removal =
  | { id: string; stage: 'reading' }
  | { id: string; stage: 'asking'; dependents: ConnectionDependents }
  | { id: string; stage: 'removing'; dependents: ConnectionDependents };

/**
 * Settings › Model connections (model-gateway.md §3.7): one row per connection with its status
 * in words, Check connection and Remove, and Add connection, which asks for the kind first and
 * then only the fields that kind needs.
 *
 * A connection set by an environment variable is shown read-only, naming the variable (S4):
 * the variable wins while it is set, so a change here would not be the one in use.
 *
 * `onChanged` is called after any change, so the model pickers on the screen read again.
 * `paidOffer` (the usage-limit offer) is shown only while a paid connection exists: a limit
 * counts paid connections only (model-gateway.md §3.8).
 */
export const ConnectionsSection: React.FC<{
  onChanged?: () => void;
  paidOffer?: React.ReactNode;
  /**
   * Raised when something else on the screen changed the connections (the GPU computer's
   * connect or disconnect adds or removes its own), so the list reads them again.
   */
  version?: number;
}> = ({ onChanged, paidOffer, version = 0 }) => {
  const t = useCopy().gateway;
  const [data, setData] = useState<ConnectionsResponse | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const [removal, setRemoval] = useState<Removal | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);

  const load = useCallback(async () => {
    setLoadFailed(false);
    try {
      setData(await fetchConnections());
    } catch (err) {
      console.error('Failed to read the connections:', err);
      setLoadFailed(true);
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load, version]);

  /** One row replaced by the Core's answer, everything else kept. */
  const replaceRow = (next: Connection) =>
    setData((prev) =>
      prev ? { ...prev, connections: prev.connections.map((c) => (c.id === next.id ? next : c)) } : prev,
    );

  const handleCheck = async (row: Connection) => {
    setBusy(row.id);
    setNotice(null);
    try {
      replaceRow(await checkConnection(row.id));
      onChanged?.();
    } catch (err) {
      console.error('Failed to check the connection:', err);
      setNotice({ tone: 'error', text: plainFailure(err, t.connections.checkFailed) });
    } finally {
      setBusy(null);
    }
  };

  /** After an install or removal: the row's count from a fresh check, then the pickers. */
  const afterModelsChanged = async (row: Connection) => {
    try {
      const recounted = await checkConnection(row.id);
      replaceRow(recounted);
    } catch (err) {
      console.error('Failed to check the connection after a model change:', err);
    }
    onChanged?.();
  };

  const startRemove = async (row: Connection) => {
    setNotice(null);
    setEditing(null);
    setRemoval({ id: row.id, stage: 'reading' });
    try {
      const dependents = await fetchDependents(row.id);
      setRemoval({ id: row.id, stage: 'asking', dependents });
    } catch (err) {
      // Never removed without the list: removing blind could leave clones without a model.
      console.error('Failed to read what uses the connection:', err);
      setRemoval(null);
      setNotice({ tone: 'error', text: t.remove.readFailed });
    }
  };

  const confirmRemove = async (row: Connection, dependents: ConnectionDependents) => {
    setRemoval({ id: row.id, stage: 'removing', dependents });
    try {
      await removeConnection(row.id);
      setRemoval(null);
      setData((prev) => (prev ? { ...prev, connections: prev.connections.filter((c) => c.id !== row.id) } : prev));
      setNotice({ tone: 'ok', text: fmt(t.remove.removed, { name: row.label }) });
      onChanged?.();
    } catch (err) {
      console.error('Failed to remove the connection:', err);
      setRemoval({ id: row.id, stage: 'asking', dependents });
      setNotice({ tone: 'error', text: plainFailure(err, t.remove.failed) });
    }
  };

  const kindOf = (kind: string): ConnectionKind | undefined => data?.kinds.find((k) => k.kind === kind);

  return (
    <div className="space-y-3" data-testid="settings-connections">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Plug className="w-3.5 h-3.5 text-cyan-400" />
          {t.connections.title}
        </h3>
      </div>
      <p className="text-[11px] text-slate-500">{t.connections.intro}</p>

      {notice && (
        <p
          data-testid="settings-connections-notice"
          role={notice.tone === 'error' ? 'alert' : 'status'}
          className={`text-[11px] ${notice.tone === 'error' ? 'text-rose-300' : 'text-emerald-400/90'}`}
        >
          {notice.text}
        </p>
      )}

      {data === null && !loadFailed && <ReadLoading testId="settings-connections-loading">{t.connections.loading}</ReadLoading>}
      {loadFailed && (
        <div data-testid="settings-connections-error" role="alert" className="flex items-center gap-2 text-[11px] text-rose-300">
          <span>{t.connections.loadFailed}</span>
          <button type="button" onClick={() => void load()} className="underline text-slate-300 hover:text-white">
            {t.connections.retry}
          </button>
        </div>
      )}

      {data !== null && data.connections.length === 0 && (
        <p data-testid="settings-connections-empty" className="text-[11px] text-slate-400">
          {t.connections.empty}
        </p>
      )}

      {data !== null && data.connections.length > 0 && (
        <ul className="space-y-2">
          {data.connections.map((row) => {
            const fromEnv = row.source === 'env';
            // A kind this version does not know: listed, checked and removable, never changed.
            const unsupported = row.status === 'unsupported';
            const ownRemoval = removal?.id === row.id ? removal : null;
            const hint =
              row.status === 'key_rejected'
                ? t.connections.keyRejectedHint
                : row.status === 'unreachable'
                  ? t.connections.unreachableHint
                  : row.status === 'no_key'
                    ? t.connections.noKeyHint
                    : unsupported
                      ? t.connections.unsupportedHint
                      : null;
            const kindLabel = kindOf(row.kind)?.label ?? row.kind;
            return (
              <li
                key={row.id}
                data-testid={`connection-row-${row.id}`}
                className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2"
              >
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-slate-100 truncate">{row.label}</div>
                    <div className="text-[11px] text-slate-500">
                      {kindLabel}
                      {row.base_url ? <span className="font-mono"> · {row.base_url}</span> : null}
                    </div>
                    <div
                      data-testid={`connection-status-${row.id}`}
                      data-status={row.status}
                      className={`text-[11px] ${row.status === 'connected' ? 'text-emerald-400/90' : 'text-amber-300'}`}
                    >
                      {statusLine(t, row.status, row.model_count)}
                    </div>
                    {row.key_set && row.key_masked && (
                      <div className="text-[11px] text-slate-500 font-mono">
                        {fmt(t.connections.keySaved, { masked: row.key_masked })}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <Button
                      variant="bordered"
                      data-testid={`connection-check-${row.id}`}
                      disabled={busy === row.id}
                      onClick={() => void handleCheck(row)}
                      className="text-[11px] px-2.5 py-1 rounded-lg"
                    >
                      {busy === row.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                      {busy === row.id ? t.connections.checking : t.connections.check}
                    </Button>
                    {!fromEnv && (
                      <>
                        {!unsupported && (
                        <Button
                          variant="bordered"
                          data-testid={`connection-edit-${row.id}`}
                          onClick={() => setEditing(editing === row.id ? null : row.id)}
                          className="text-[11px] px-2.5 py-1 rounded-lg"
                        >
                          {t.connections.edit}
                        </Button>
                        )}
                        <Button
                          variant="bordered"
                          data-testid={`connection-remove-${row.id}`}
                          disabled={ownRemoval !== null}
                          onClick={() => void startRemove(row)}
                          className="text-[11px] px-2.5 py-1 rounded-lg text-rose-300 border-rose-900/60 hover:bg-rose-950/40"
                        >
                          <Trash2 className="w-3 h-3" />
                          {t.connections.remove}
                        </Button>
                      </>
                    )}
                  </div>
                </div>

                {fromEnv && (
                  <p data-testid={`connection-env-${row.id}`} className="text-[11px] text-amber-300/90">
                    {row.env_var
                      ? fmt(t.connections.fromEnv, { variable: row.env_var })
                      : t.connections.fromEnvUnnamed}
                  </p>
                )}
                {row.key_env_var && !fromEnv && (
                  <p data-testid={`connection-key-env-${row.id}`} className="text-[11px] text-amber-300/90">
                    {fmt(t.connections.keyFromEnv, { variable: row.key_env_var })}
                  </p>
                )}
                {hint && !fromEnv && (
                  <p data-testid={`connection-hint-${row.id}`} className="text-[11px] text-slate-400">
                    {hint}
                  </p>
                )}

                {/* Install and remove models on an Ollama (#2167), once it answered or before
                    it was first checked (the panel says so itself if it does not answer); its
                    row's model count and the pickers read again after each change. */}
                {row.kind === 'ollama' && (row.status === 'connected' || row.status === 'unchecked') && (
                  <OllamaModels row={row} onChanged={() => void afterModelsChanged(row)} />
                )}

                {editing === row.id && !fromEnv && !unsupported && (
                  <EditConnection
                    row={row}
                    kind={kindOf(row.kind)}
                    onDone={(next) => {
                      setEditing(null);
                      if (next) {
                        replaceRow(next);
                        onChanged?.();
                      }
                    }}
                  />
                )}

                {ownRemoval && (
                  <RemoveConfirm
                    row={row}
                    removal={ownRemoval}
                    onConfirm={(dependents) => void confirmRemove(row, dependents)}
                    onKeep={() => setRemoval(null)}
                  />
                )}
              </li>
            );
          })}
        </ul>
      )}

      {data !== null && data.connections.some((c) => c.paid) && paidOffer}

      {data !== null &&
        (adding ? (
          <AddConnection
            kinds={data.kinds}
            onCancel={() => setAdding(false)}
            onAdded={(row) => {
              setAdding(false);
              setData((prev) => (prev ? { ...prev, connections: [...prev.connections, row] } : prev));
              setNotice({
                tone: row.status === 'connected' ? 'ok' : 'error',
                text: fmt(t.add.added, { name: row.label, status: statusLine(t, row.status, row.model_count) }),
              });
              onChanged?.();
            }}
          />
        ) : (
          <Button
            variant="bordered"
            data-testid="connection-add-open"
            onClick={() => {
              setNotice(null);
              setAdding(true);
            }}
            className="text-xs px-3 py-1.5 rounded-xl"
          >
            <Plus className="w-3.5 h-3.5" />
            {t.add.open}
          </Button>
        ))}
    </div>
  );
};

/** The confirm step of Remove: what will lose its model, named before anything is removed. */
const RemoveConfirm: React.FC<{
  row: Connection;
  removal: Removal;
  onConfirm: (dependents: ConnectionDependents) => void;
  onKeep: () => void;
}> = ({ row, removal, onConfirm, onKeep }) => {
  const copy = useCopy();
  const t = copy.gateway;
  if (removal.stage === 'reading') {
    return <ReadLoading testId={`connection-remove-reading-${row.id}`}>{t.remove.reading}</ReadLoading>;
  }
  const { dependents } = removal;
  const defaultName = { deep: t.defaults.deep, fast: t.defaults.fast, image: t.defaults.image };
  const unused = dependents.clones.length === 0 && dependents.defaults.length === 0;
  return (
    <div
      data-testid={`connection-remove-confirm-${row.id}`}
      className="p-3 rounded-lg border border-rose-900/50 bg-rose-950/20 space-y-2 text-[11px]"
    >
      <p className="text-slate-200 font-medium">{fmt(t.remove.confirm, { name: row.label })}</p>
      {unused && <p className="text-slate-400">{t.remove.unused}</p>}
      {dependents.clones.length > 0 && (
        <div>
          <p className="text-slate-300">{t.remove.clonesIntro}</p>
          <ul className="list-disc pl-5 text-slate-300" data-testid={`connection-remove-clones-${row.id}`}>
            {dependents.clones.map((clone) => (
              <li key={clone.id}>
                {clone.name}
                {clone.slots.length > 0 && (
                  <span className="text-slate-500"> ({clone.slots.map((s) => t.remove.slots[s] ?? s).join(', ')})</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      {dependents.defaults.length > 0 && (
        <div>
          <p className="text-slate-300">{t.remove.defaultsIntro}</p>
          <ul className="list-disc pl-5 text-slate-300" data-testid={`connection-remove-defaults-${row.id}`}>
            {dependents.defaults.map((slot) => (
              <li key={slot}>{defaultName[slot] ?? slot}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="flex items-center gap-2">
        <Button
          variant="bordered"
          data-testid={`connection-remove-yes-${row.id}`}
          disabled={removal.stage === 'removing'}
          onClick={() => onConfirm(dependents)}
          className="text-[11px] px-2.5 py-1 rounded-lg text-rose-300 border-rose-900/60 hover:bg-rose-950/40"
        >
          {removal.stage === 'removing' && <Loader2 className="w-3 h-3 animate-spin" />}
          {t.remove.confirmButton}
        </Button>
        <Button variant="bordered" onClick={onKeep} className="text-[11px] px-2.5 py-1 rounded-lg">
          {t.remove.keep}
        </Button>
      </div>
    </div>
  );
};

/** Change a saved connection: a new key, or a new address. Nothing is sent until Save. */
const EditConnection: React.FC<{
  row: Connection;
  kind: ConnectionKind | undefined;
  onDone: (next: Connection | null) => void;
}> = ({ row, kind, onDone }) => {
  const t = useCopy().gateway;
  // A kind the list no longer names: a key when it holds one or has no address (a cloud API).
  const wantsKey = kind ? kind.needs_key : row.key_set || row.base_url === null;
  const wantsAddress = kind ? kind.needs_base_url : row.base_url !== null;
  const [key, setKey] = useState('');
  const [address, setAddress] = useState(row.base_url ?? '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const changes: { key?: string; base_url?: string } = {};
  if (wantsKey && sanitizeApiKey(key)) changes.key = sanitizeApiKey(key);
  if (wantsAddress && address.trim() && address.trim() !== (row.base_url ?? '')) changes.base_url = address.trim();
  const nothing = Object.keys(changes).length === 0;

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      onDone(await updateConnection(row.id, changes));
    } catch (err) {
      console.error('Failed to change the connection:', err);
      setError(plainFailure(err, t.connections.updateFailed));
      setSaving(false);
    }
  };

  return (
    <div data-testid={`connection-edit-form-${row.id}`} className="space-y-2">
      {wantsAddress && (
        <input
          type="text"
          aria-label={t.connections.address}
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          className={inputClass}
        />
      )}
      {wantsKey && (
        <input
          type="password"
          aria-label={t.connections.newKey}
          placeholder={t.connections.newKeyPlaceholder}
          value={key}
          onChange={(e) => setKey(e.target.value)}
          className={inputClass}
        />
      )}
      <div className="flex items-center gap-2">
        <Button
          variant="bordered"
          data-testid={`connection-edit-save-${row.id}`}
          disabled={nothing || saving}
          onClick={() => void save()}
          className="text-[11px] px-2.5 py-1 rounded-lg"
        >
          {saving && <Loader2 className="w-3 h-3 animate-spin" />}
          {t.connections.saveChanges}
        </Button>
        <Button variant="bordered" onClick={() => onDone(null)} className="text-[11px] px-2.5 py-1 rounded-lg">
          {t.connections.cancel}
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-[11px] text-rose-300">
          {error}
        </p>
      )}
    </div>
  );
};

/** Add connection: the kind first, then only the fields that kind needs. */
const AddConnection: React.FC<{
  kinds: ConnectionKind[];
  onCancel: () => void;
  onAdded: (row: Connection) => void;
}> = ({ kinds, onCancel, onAdded }) => {
  const t = useCopy().gateway;
  const [kindId, setKindId] = useState('');
  const [label, setLabel] = useState('');
  const [key, setKey] = useState('');
  const [address, setAddress] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const kind = kinds.find((k) => k.kind === kindId);

  const pickKind = (next: string) => {
    setKindId(next);
    setError(null);
    setKey('');
    setAddress(kinds.find((k) => k.kind === next)?.default_base_url ?? '');
  };

  const ready =
    kind !== undefined &&
    (!kind.needs_key || sanitizeApiKey(key) !== '') &&
    (!kind.needs_base_url || address.trim() !== '');

  const submit = async () => {
    if (!kind) return;
    setSaving(true);
    setError(null);
    try {
      onAdded(
        await addConnection({
          kind: kind.kind,
          ...(label.trim() ? { label: label.trim() } : {}),
          ...(kind.needs_base_url && address.trim() ? { base_url: address.trim() } : {}),
          ...(kind.needs_key ? { key: sanitizeApiKey(key) } : {}),
        }),
      );
    } catch (err) {
      console.error('Failed to add the connection:', err);
      setError(plainFailure(err, t.add.failed));
      setSaving(false);
    }
  };

  return (
    <div data-testid="connection-add-form" className="p-3 rounded-xl border border-slate-800 bg-slate-950/40 space-y-2.5">
      <p className="text-xs font-medium text-slate-200">{t.add.title}</p>
      <div className="space-y-1">
        <label htmlFor="connection-add-kind" className="block text-[11px] text-slate-400">
          {t.add.kind}
        </label>
        <select
          id="connection-add-kind"
          data-testid="connection-add-kind"
          value={kindId}
          onChange={(e) => pickKind(e.target.value)}
          className={`${inputClass} font-sans cursor-pointer`}
        >
          <option value="" disabled>
            {t.add.kindPlaceholder}
          </option>
          {kinds.map((k) => (
            <option key={k.kind} value={k.kind}>
              {k.label}
            </option>
          ))}
        </select>
      </div>

      {kind && (
        <>
          <div className="space-y-1">
            <label htmlFor="connection-add-name" className="block text-[11px] text-slate-400">
              {t.add.name}
            </label>
            <input
              id="connection-add-name"
              type="text"
              value={label}
              placeholder={t.add.namePlaceholder}
              onChange={(e) => setLabel(e.target.value)}
              className={`${inputClass} font-sans`}
            />
          </div>
          {kind.needs_base_url && (
            <div className="space-y-1">
              <label htmlFor="connection-add-address" className="block text-[11px] text-slate-400">
                {t.add.address}
              </label>
              <input
                id="connection-add-address"
                data-testid="connection-add-address"
                type="text"
                value={address}
                placeholder={t.add.addressPlaceholder}
                onChange={(e) => setAddress(e.target.value)}
                className={inputClass}
              />
            </div>
          )}
          {kind.needs_key && (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <label htmlFor="connection-add-key" className="block text-[11px] text-slate-400">
                  {t.add.key}
                </label>
                {kind.key_url && (
                  <a
                    href={kind.key_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    data-testid="connection-add-key-link"
                    className="inline-flex items-center gap-1 text-[11px] text-cyan-400 hover:text-cyan-300"
                  >
                    {fmt(t.add.getKey, { name: kind.label })}
                    <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
              <input
                id="connection-add-key"
                data-testid="connection-add-key"
                type="password"
                value={key}
                placeholder={t.add.keyPlaceholder}
                onChange={(e) => setKey(e.target.value)}
                className={inputClass}
              />
            </div>
          )}
        </>
      )}

      <div className="flex items-center gap-2">
        <Button
          variant="bordered"
          data-testid="connection-add-submit"
          disabled={!ready || saving}
          onClick={() => void submit()}
          className="text-xs px-3 py-1.5 rounded-xl text-cyan-300 border-cyan-800/60 hover:bg-cyan-950/50"
        >
          {saving && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
          {saving ? t.add.submitting : t.add.submit}
        </Button>
        <Button variant="bordered" onClick={onCancel} className="text-xs px-3 py-1.5 rounded-xl">
          {t.add.cancel}
        </Button>
      </div>
      {error && (
        <p role="alert" data-testid="connection-add-error" className="text-[11px] text-rose-300">
          {error}
        </p>
      )}
    </div>
  );
};
