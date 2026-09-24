import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Plug, Plus, RefreshCw, Trash2, Upload, X } from 'lucide-react';
import type { McpServer, McpServerDraft, McpServerList } from '../../types';
import {
  MCP_NAME_PATTERN,
  addMcpServer,
  fetchMcpServers,
  importMcpServers,
  reconnectMcpServer,
  removeMcpServer,
  setMcpServerEnabled,
} from '../../lib/mcpServersApi';
import { coreReason } from '../../lib/coreFailure';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';

/**
 * External tool servers (MCP), as a section of Settings.
 *
 * A user connects a tool server here -- by its web address, or by a command that starts it on
 * this computer -- and clones can then use the tools it offers. The runtime owns the list and
 * each server's connection (P8); this section only shows it and asks for changes.
 *
 * Secret values (header and environment values) are sent once, when a server is added, and
 * never come back: the server reports only their names, and the form forgets them on success.
 */

const NAME_RULE = 'Use 1 to 32 letters, digits, - or _ (no spaces).';

const inputClass =
  'bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 transition-colors';

/** How a command line reads: arguments with spaces are quoted so the line is unambiguous. */
const commandLine = (server: McpServer): string =>
  [server.command ?? '', ...server.args.map((a) => (/\s/.test(a) || a === '' ? JSON.stringify(a) : a))]
    .join(' ')
    .trim();

const toolCount = (n: number) => (n === 1 ? '1 tool' : `${n} tools`);

// ---------------------------------------------------------------------------------------------
// Key/value rows (headers, environment variables). Values are secret: password inputs.
// ---------------------------------------------------------------------------------------------

interface Pair {
  key: string;
  value: string;
}

const KeyValueRows: React.FC<{
  label: string;
  keyPlaceholder: string;
  addLabel: string;
  pairs: Pair[];
  onChange: (pairs: Pair[]) => void;
}> = ({ label, keyPlaceholder, addLabel, pairs, onChange }) => (
  <fieldset className="space-y-1.5">
    <legend className="text-xs font-medium text-slate-300 mb-1">{label}</legend>
    {pairs.map((pair, i) => (
      <div key={i} className="flex flex-wrap items-center gap-2">
        <input
          type="text"
          aria-label={`${label} name ${i + 1}`}
          value={pair.key}
          placeholder={keyPlaceholder}
          onChange={(e) => onChange(pairs.map((p, j) => (j === i ? { ...p, key: e.target.value } : p)))}
          className={`${inputClass} flex-1 min-w-[8rem] font-mono`}
        />
        <input
          type="password"
          autoComplete="off"
          aria-label={`${label} value ${i + 1}`}
          value={pair.value}
          placeholder="Value (kept secret)"
          onChange={(e) => onChange(pairs.map((p, j) => (j === i ? { ...p, value: e.target.value } : p)))}
          className={`${inputClass} flex-1 min-w-[8rem] font-mono`}
        />
        <button
          type="button"
          aria-label={`Remove ${label.toLowerCase()} row ${i + 1}`}
          title="Remove this row"
          onClick={() => onChange(pairs.filter((_, j) => j !== i))}
          className="text-slate-500 hover:text-rose-400"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    ))}
    <Button onClick={() => onChange([...pairs, { key: '', value: '' }])}>
      <Plus className="w-3 h-3" />
      {addLabel}
    </Button>
  </fieldset>
);

/** The rows as a record, or the reason they cannot be one. Blank rows are dropped. */
const pairsToRecord = (pairs: Pair[], what: string): Record<string, string> | string => {
  const out: Record<string, string> = {};
  for (const { key, value } of pairs) {
    const k = key.trim();
    if (!k && !value) continue;
    if (!k) return `Each ${what} value needs a name.`;
    if (k in out) return `The ${what} "${k}" is listed twice.`;
    out[k] = value;
  }
  return out;
};

// ---------------------------------------------------------------------------------------------
// Add form
// ---------------------------------------------------------------------------------------------

const AddServerForm: React.FC<{ onAdded: (server: McpServer) => void; onCancel: () => void }> = ({
  onAdded,
  onCancel,
}) => {
  const [transport, setTransport] = useState<'http' | 'stdio'>('http');
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [command, setCommand] = useState('');
  const [args, setArgs] = useState('');
  const [headers, setHeaders] = useState<Pair[]>([]);
  const [env, setEnv] = useState<Pair[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const nameInvalid = name.trim() !== '' && !MCP_NAME_PATTERN.test(name.trim());

  const submit = async () => {
    setError(null);
    const trimmedName = name.trim();
    if (!MCP_NAME_PATTERN.test(trimmedName)) {
      setError(`The name is not usable. ${NAME_RULE}`);
      return;
    }
    const draft: McpServerDraft = { name: trimmedName, transport };
    if (transport === 'http') {
      if (!url.trim()) {
        setError('Enter the server address, starting with http:// or https://.');
        return;
      }
      const record = pairsToRecord(headers, 'header');
      if (typeof record === 'string') {
        setError(record);
        return;
      }
      draft.url = url.trim();
      if (Object.keys(record).length) draft.headers = record;
    } else {
      if (!command.trim()) {
        setError('Enter the command that starts the server, for example npx.');
        return;
      }
      const record = pairsToRecord(env, 'environment variable');
      if (typeof record === 'string') {
        setError(record);
        return;
      }
      draft.command = command.trim();
      draft.args = args
        .split('\n')
        .map((a) => a.trim())
        .filter(Boolean);
      if (Object.keys(record).length) draft.env = record;
    }

    setBusy(true);
    const result = await addMcpServer(draft);
    setBusy(false);
    if (!result.ok) {
      setError(result.message);
      return;
    }
    // Forget every secret the moment the server has them.
    setName('');
    setUrl('');
    setCommand('');
    setArgs('');
    setHeaders([]);
    setEnv([]);
    onAdded(result.value);
  };

  return (
    <div
      className="p-3 rounded-xl border border-slate-800 bg-slate-950/40 space-y-3"
      data-testid="mcp-add-form"
    >
      <fieldset className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-300">
        <legend className="sr-only">Kind of server</legend>
        <label className="flex items-center gap-1.5">
          <input
            type="radio"
            name="mcp-transport"
            checked={transport === 'http'}
            onChange={() => setTransport('http')}
          />
          Remote (URL)
        </label>
        <label className="flex items-center gap-1.5">
          <input
            type="radio"
            name="mcp-transport"
            checked={transport === 'stdio'}
            onChange={() => setTransport('stdio')}
          />
          On this computer (command)
        </label>
      </fieldset>

      <div className="space-y-1">
        <label className="block text-xs font-medium text-slate-300" htmlFor="mcp-name">
          Name
        </label>
        <input
          id="mcp-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. github"
          className={`${inputClass} w-full font-mono`}
        />
        <p className={`text-[11px] ${nameInvalid ? 'text-amber-400' : 'text-slate-500'}`}>{NAME_RULE}</p>
      </div>

      {transport === 'http' ? (
        <>
          <div className="space-y-1">
            <label className="block text-xs font-medium text-slate-300" htmlFor="mcp-url">
              Server address (URL)
            </label>
            <input
              id="mcp-url"
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/mcp"
              className={`${inputClass} w-full font-mono`}
            />
          </div>
          <KeyValueRows
            label="Headers"
            keyPlaceholder="e.g. Authorization"
            addLabel="Add header"
            pairs={headers}
            onChange={setHeaders}
          />
        </>
      ) : (
        <>
          <p
            className="flex items-start gap-2 text-[11px] text-amber-300"
            data-testid="mcp-command-warning"
          >
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            This runs a program on this computer with your permissions. Only add servers you trust.
          </p>
          <div className="space-y-1">
            <label className="block text-xs font-medium text-slate-300" htmlFor="mcp-command">
              Command
            </label>
            <input
              id="mcp-command"
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="e.g. npx"
              className={`${inputClass} w-full font-mono`}
            />
          </div>
          <div className="space-y-1">
            <label className="block text-xs font-medium text-slate-300" htmlFor="mcp-args">
              Arguments, one per line
            </label>
            <textarea
              id="mcp-args"
              rows={3}
              value={args}
              onChange={(e) => setArgs(e.target.value)}
              placeholder={'-y\n@modelcontextprotocol/server-filesystem\n/Users/me/Documents'}
              className={`${inputClass} w-full font-mono`}
            />
          </div>
          <KeyValueRows
            label="Environment variables"
            keyPlaceholder="e.g. API_KEY"
            addLabel="Add variable"
            pairs={env}
            onChange={setEnv}
          />
        </>
      )}

      {error !== null && (
        <p role="alert" className="text-[11px] text-rose-300" data-testid="mcp-add-error">
          {error}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="solid" onClick={() => void submit()} disabled={busy}>
          {busy ? 'Adding…' : 'Add server'}
        </Button>
        <Button onClick={onCancel} disabled={busy}>
          Cancel
        </Button>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------------------------
// Import from JSON
// ---------------------------------------------------------------------------------------------

const ImportServers: React.FC<{ onAdded: (servers: McpServer[]) => void; onCancel: () => void }> = ({
  onAdded,
  onCancel,
}) => {
  const [json, setJson] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<{ added: string[]; skipped: { name: string; reason: string }[] } | null>(
    null,
  );

  const submit = async () => {
    setError(null);
    setOutcome(null);
    setBusy(true);
    const result = await importMcpServers(json);
    setBusy(false);
    if (!result.ok) {
      setError(result.message);
      return;
    }
    const added = result.value.added ?? [];
    const skipped = result.value.skipped ?? [];
    setOutcome({ added: added.map((s) => s.name), skipped });
    // The snippet may carry secret values; do not keep them on screen once they are saved.
    setJson('');
    onAdded(added);
  };

  return (
    <div
      className="p-3 rounded-xl border border-slate-800 bg-slate-950/40 space-y-3"
      data-testid="mcp-import"
    >
      <div className="space-y-1">
        <label className="block text-xs font-medium text-slate-300" htmlFor="mcp-import-json">
          Paste a server list in JSON
        </label>
        <p className="text-[11px] text-slate-500">
          Many tools publish a setup snippet that starts with {'{"mcpServers": …}'}. Paste it here as it is.
          Servers in it that run a command run on this computer with your permissions, so only import
          from sources you trust.
        </p>
        <textarea
          id="mcp-import-json"
          rows={6}
          value={json}
          onChange={(e) => setJson(e.target.value)}
          placeholder={'{\n  "mcpServers": {\n    "name": { "command": "npx", "args": ["…"] }\n  }\n}'}
          className={`${inputClass} w-full font-mono`}
        />
      </div>

      {error !== null && (
        <p role="alert" className="text-[11px] text-rose-300" data-testid="mcp-import-error">
          {error}
        </p>
      )}

      {outcome !== null && (
        <div className="space-y-1 text-[11px]" data-testid="mcp-import-outcome">
          <p className="text-slate-300">
            {outcome.added.length > 0
              ? `Added: ${outcome.added.join(', ')}`
              : 'No servers were added.'}
          </p>
          {outcome.skipped.length > 0 && (
            <ul className="space-y-0.5" aria-label="Skipped servers">
              {outcome.skipped.map((s) => (
                <li key={s.name} className="text-amber-300">
                  Skipped {s.name}: {s.reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="solid" onClick={() => void submit()} disabled={busy || !json.trim()}>
          {busy ? 'Importing…' : 'Import'}
        </Button>
        <Button onClick={onCancel} disabled={busy}>
          Close
        </Button>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------------------------
// One server row
// ---------------------------------------------------------------------------------------------

const ServerRow: React.FC<{
  server: McpServer;
  onChanged: (server: McpServer) => void;
  onRemoved: (name: string) => void;
}> = ({ server, onChanged, onRemoved }) => {
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (action: () => Promise<{ ok: true; value: McpServer } | { ok: false; message: string }>) => {
    setBusy(true);
    setError(null);
    const result = await action();
    setBusy(false);
    if (result.ok) onChanged(result.value);
    else setError(result.message);
  };

  const remove = async () => {
    setBusy(true);
    setError(null);
    const result = await removeMcpServer(server.name);
    setBusy(false);
    setConfirming(false);
    if (result.ok) onRemoved(server.name);
    else setError(result.message);
  };

  const status =
    server.status === 'connected' ? (
      <Badge tone="success">Connected · {toolCount(server.tools.length)}</Badge>
    ) : server.status === 'disabled' ? (
      <Badge tone="neutral">Disabled</Badge>
    ) : server.status === 'connecting' ? (
      <Badge tone="neutral">Connecting…</Badge>
    ) : (
      <Badge tone="danger">Error</Badge>
    );

  const secretKeys = server.transport === 'http' ? server.header_keys : server.env_keys;

  return (
    <li
      className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg space-y-1.5"
      data-testid={`mcp-server-${server.name}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-200 font-mono">{server.name}</span>
        <span className="text-[11px] text-slate-500">
          {server.transport === 'http' ? 'Remote' : 'On this computer'}
        </span>
        {status}
        <span className="ml-auto flex items-center gap-2">
          <button
            type="button"
            role="switch"
            aria-checked={server.enabled}
            aria-label={`Use ${server.name}`}
            title={server.enabled ? 'Turn off' : 'Turn on'}
            disabled={busy}
            onClick={() => void run(() => setMcpServerEnabled(server.name, !server.enabled))}
            className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors disabled:opacity-50 ${
              server.enabled ? 'bg-cyan-700 border-cyan-600' : 'bg-slate-800 border-slate-700'
            }`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 rounded-full bg-slate-100 transition-transform ${
                server.enabled ? 'translate-x-4' : 'translate-x-0.5'
              }`}
            />
          </button>
          <Button
            variant="ghost"
            size="icon"
            aria-label={`Reconnect ${server.name}`}
            title="Reconnect"
            disabled={busy}
            onClick={() => void run(() => reconnectMcpServer(server.name))}
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label={`Remove ${server.name}`}
            title="Remove"
            disabled={busy}
            onClick={() => setConfirming(true)}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </span>
      </div>

      <p className="text-[11px] font-mono text-slate-400 break-all">
        {server.transport === 'http' ? server.url : commandLine(server)}
      </p>
      {secretKeys.length > 0 && (
        <p className="text-[11px] text-slate-500">
          {server.transport === 'http' ? 'Sends headers' : 'Sets environment variables'}:{' '}
          <span className="font-mono">{secretKeys.join(', ')}</span> (values hidden)
        </p>
      )}

      {server.status === 'error' && (
        <div className="text-[11px] space-y-0.5" data-testid="mcp-server-error">
          <p className="text-rose-300">Not connected. The server reported:</p>
          <p className="font-mono text-rose-200 break-all">{server.error || 'No reason was given.'}</p>
          <p className="text-slate-500">
            It is still saved. Check the {server.transport === 'http' ? 'address and headers' : 'command'}, then
            press Reconnect, or remove it and add it again.
          </p>
        </div>
      )}
      {server.status === 'connecting' && (
        <p className="text-[11px] text-slate-500">Starting the connection. Its tools are listed once it connects.</p>
      )}
      {server.status === 'disabled' && (
        <p className="text-[11px] text-slate-500">Turned off. Clones cannot use its tools until you turn it on.</p>
      )}

      {server.status === 'connected' && (
        <details className="text-[11px]">
          <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
            Show tools ({server.tools.length})
          </summary>
          {server.tools.length > 0 ? (
            <ul className="mt-1 space-y-1" aria-label={`Tools from ${server.name}`}>
              {server.tools.map((tool) => (
                <li key={tool.name}>
                  <span className="font-mono text-slate-200">{tool.name}</span>
                  {tool.description && <span className="text-slate-500"> — {tool.description}</span>}
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1 text-slate-500">This server is connected but offers no tools.</p>
          )}
        </details>
      )}

      {confirming && (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300" data-testid="mcp-remove-confirm">
          <span>Remove {server.name}? Clones will no longer be able to use its tools.</span>
          <Button onClick={() => void remove()} disabled={busy}>
            Remove
          </Button>
          <Button onClick={() => setConfirming(false)} disabled={busy}>
            Keep
          </Button>
        </div>
      )}

      {error !== null && (
        <p role="alert" className="text-[11px] text-rose-300">
          {error}
        </p>
      )}
    </li>
  );
};

// ---------------------------------------------------------------------------------------------
// The section
// ---------------------------------------------------------------------------------------------

/** How often the list is read again while any server is still connecting. */
const CONNECTING_POLL_MS = 2000;

export const McpServersSection: React.FC<{ pollMs?: number }> = ({ pollMs = CONNECTING_POLL_MS }) => {
  const [list, setList] = useState<McpServerList | null>(null);
  /** Why the last read failed, or `null` when it did not; `reason` is the Core's own words, if any. */
  const [loadError, setLoadError] = useState<{ reason: string | null } | null>(null);
  const [panel, setPanel] = useState<'none' | 'add' | 'import'>('none');

  const load = useCallback(async () => {
    try {
      setList(await fetchMcpServers());
      setLoadError(null);
    } catch (err) {
      // The cause is for the console. The reader is shown the Core's own words or none (#1436):
      // a browser's "Failed to fetch" and a status line are not a reason they can act on.
      console.error('Failed to read the tool servers:', err);
      setLoadError({ reason: coreReason(err) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const upsert = (incoming: McpServer[]) =>
    setList((prev) => {
      const servers = [...(prev?.servers ?? [])];
      for (const server of incoming) {
        const i = servers.findIndex((s) => s.name === server.name);
        if (i >= 0) servers[i] = server;
        else servers.push(server);
      }
      return { config_path: prev?.config_path ?? '', servers };
    });

  const removeLocal = (name: string) =>
    setList((prev) => (prev ? { ...prev, servers: prev.servers.filter((s) => s.name !== name) } : prev));

  const servers = list?.servers ?? [];

  // Servers connect in the background (at app start, say), so a row can read "Connecting…".
  // Read the list again, quietly, until none is; otherwise it would say so until reopened.
  const anyConnecting = servers.some((s) => s.status === 'connecting');
  useEffect(() => {
    if (!anyConnecting || loadError !== null) return;
    const timer = window.setTimeout(() => void load(), pollMs);
    return () => window.clearTimeout(timer);
  }, [anyConnecting, list, loadError, load, pollMs]);

  return (
    <div className="space-y-3" data-testid="settings-mcp">
      <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <Plug className="w-3.5 h-3.5 text-cyan-400" />
        External tools (MCP)
      </label>
      <p className="text-[11px] text-slate-500">
        Connect a tool server so clones can use its tools.
      </p>

      {loadError !== null && (
        <div
          role="alert"
          data-testid="settings-mcp-error"
          className="p-3 rounded-xl border border-rose-800/80 bg-rose-950/50 space-y-1"
        >
          <div className="flex items-center gap-2 text-xs text-rose-200">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>The list of tool servers could not be read.</span>
          </div>
          {loadError.reason !== null && <p className="text-[11px] text-rose-300/80">{loadError.reason}</p>}
          <Button onClick={() => void load()}>Try again</Button>
        </div>
      )}

      {loadError === null && list !== null && servers.length === 0 && (
        <p className="text-[11px] text-slate-400" data-testid="settings-mcp-empty">
          No tool servers are connected yet. Choose Add a server to connect one by its web address or by a
          command, or Import from JSON to paste the setup snippet a tool publishes.
        </p>
      )}

      {servers.length > 0 && (
        <ul className="space-y-1.5" aria-label="Tool servers">
          {servers.map((server) => (
            <ServerRow
              key={server.name}
              server={server}
              onChanged={(s) => upsert([s])}
              onRemoved={removeLocal}
            />
          ))}
        </ul>
      )}

      {list?.config_path ? (
        <p className="text-[11px] text-slate-600 font-mono break-all" data-testid="settings-mcp-config-path">
          Saved in {list.config_path}
        </p>
      ) : null}

      {panel === 'add' && (
        <AddServerForm
          onAdded={(server) => {
            upsert([server]);
            setPanel('none');
          }}
          onCancel={() => setPanel('none')}
        />
      )}
      {panel === 'import' && <ImportServers onAdded={upsert} onCancel={() => setPanel('none')} />}

      {panel === 'none' && (
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="bordered" onClick={() => setPanel('add')} className="px-3 py-1.5 rounded-xl">
            <Plus className="w-3.5 h-3.5 text-cyan-400" />
            Add a server
          </Button>
          <Button variant="bordered" onClick={() => setPanel('import')} className="px-3 py-1.5 rounded-xl">
            <Upload className="w-3.5 h-3.5 text-cyan-400" />
            Import from JSON
          </Button>
        </div>
      )}
    </div>
  );
};
