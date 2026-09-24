import React from 'react';
import { AlertTriangle, Check, PlugZap, RefreshCw, ShieldAlert, X } from 'lucide-react';
import { AcpMethod, AcpMethodStatus, AcpStatusData } from '../types';

interface AcpTabProps {
  acpStatus: AcpStatusData | null;
  onRefresh: () => void;
  isLoading: boolean;
}

const STATUS_STYLE: Record<AcpMethodStatus, { label: string; className: string }> = {
  implemented: { label: 'impl', className: 'bg-emerald-950/80 text-emerald-400' },
  not_implemented: { label: 'todo', className: 'bg-slate-800 text-slate-400' },
  not_implementable: { label: 'blocked', className: 'bg-rose-950/80 text-rose-400' },
  out_of_scope: { label: 'n/a', className: 'bg-slate-900 text-slate-500' },
};

const StatusPill: React.FC<{ status: AcpMethodStatus }> = ({ status }) => {
  const style = STATUS_STYLE[status];
  return (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase shrink-0 ${style.className}`}
    >
      {style.label}
    </span>
  );
};

const MethodRow: React.FC<{ method: AcpMethod }> = ({ method }) => (
  <div
    data-testid={`acp-method-${method.side}-${method.name}`}
    className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/60 space-y-1"
  >
    <div className="flex items-center justify-between gap-2">
      <span className="font-mono text-[12px] text-slate-200">{method.name}</span>
      <div className="flex items-center gap-1.5">
        {method.spec_section && (
          <span className="text-[10px] font-mono text-slate-600">§{method.spec_section}</span>
        )}
        <StatusPill status={method.status} />
      </div>
    </div>
    {method.counterpart && (
      <div className="text-[11px] text-slate-500 font-mono truncate" title={method.counterpart}>
        → {method.counterpart}
      </div>
    )}
    {method.note && <p className="text-[11px] text-slate-400 leading-snug">{method.note}</p>}
  </div>
);

/**
 * The ACP surface: what this build answers of the Agent Client Protocol, and whether anything
 * is serving it.
 *
 * Most ACP screens belong to the *client* — a connected editor draws the conversation, the
 * permission dialog, the terminal and the file buffers. What has no client-side counterpart,
 * and therefore has to live here, is the state of the protocol itself: whether a shell is
 * present at all, which methods it can honestly claim, and what happens to an MCP descriptor
 * that arrives over the connection.
 *
 * The presence block is rendered before anything else and states its reason verbatim. An ACP
 * surface showing an empty session list is indistinguishable from one showing a running shell
 * that nobody has connected to, and P6 forbids exactly that substitution.
 */
export const AcpTab: React.FC<AcpTabProps> = ({ acpStatus, onRefresh, isLoading }) => {
  if (!acpStatus) {
    return (
      <div data-testid="acp-unavailable" className="text-center py-12 text-slate-500 space-y-2">
        <PlugZap className="w-8 h-8 mx-auto text-slate-600" />
        <p>ACP status has not been read from the runtime.</p>
        <button
          type="button"
          onClick={onRefresh}
          className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700"
        >
          Retry
        </button>
      </div>
    );
  }

  const agentMethods = acpStatus.methods.filter((m) => m.side === 'agent');
  const clientMethods = acpStatus.methods.filter((m) => m.side === 'client');

  return (
    <div data-testid="acp-panel" className="space-y-4">
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-1">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-sm font-bold text-white flex items-center gap-2">
            <PlugZap className="w-4 h-4 text-cyan-400" />
            Agent Client Protocol
          </h2>
          <button
            type="button"
            onClick={onRefresh}
            disabled={isLoading}
            className="p-1.5 rounded-lg bg-slate-800/90 hover:bg-slate-700 text-slate-300 disabled:opacity-50"
            title="Re-read ACP status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
        <p className="text-[11px] text-slate-400">
          Transport <span className="font-mono text-slate-300">{acpStatus.transport}</span> · SDK{' '}
          <span className="font-mono text-slate-300">
            agent-client-protocol=={acpStatus.sdk_version_specified}
          </span>
        </p>
      </div>

      {/* Presence. Always first, and never collapsed into an empty list. */}
      <div
        data-testid="acp-presence"
        className={`p-4 rounded-2xl border space-y-2 ${
          acpStatus.serving
            ? 'bg-emerald-950/20 border-emerald-800/60'
            : 'bg-amber-950/20 border-amber-800/60'
        }`}
      >
        <div className="flex items-center gap-2">
          {acpStatus.serving ? (
            <Check className="w-4 h-4 text-emerald-400" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          )}
          <span
            data-testid="acp-serving-state"
            className={`text-xs font-bold ${acpStatus.serving ? 'text-emerald-300' : 'text-amber-300'}`}
          >
            {acpStatus.serving ? 'Serving ACP' : 'Not serving ACP'}
          </span>
        </div>
        <p className="text-[11px] text-slate-300 leading-snug">{acpStatus.presence.reason}</p>
        <div className="flex flex-wrap gap-3 text-[10px] font-mono text-slate-400 pt-1">
          <span className="flex items-center gap-1">
            {acpStatus.presence.shell_module_present ? (
              <Check className="w-3 h-3 text-emerald-400" />
            ) : (
              <X className="w-3 h-3 text-rose-400" />
            )}
            uclone_x.acp.server
          </span>
          <span className="flex items-center gap-1">
            {acpStatus.presence.sdk_installed ? (
              <Check className="w-3 h-3 text-emerald-400" />
            ) : (
              <X className="w-3 h-3 text-rose-400" />
            )}
            acp SDK
          </span>
        </div>
      </div>

      {/* Sessions. Deliberately not a list until a shell can produce one. */}
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-2">
        <div className="text-[10px] text-slate-500 uppercase font-sans">Live ACP sessions</div>
        {acpStatus.serving ? (
          <p data-testid="acp-sessions-live" className="text-[11px] text-slate-400">
            Session reporting arrives with the shell; this build serves ACP but does not yet
            enumerate its sessions.
          </p>
        ) : (
          <p data-testid="acp-sessions-unavailable" className="text-[11px] text-slate-400">
            Unavailable — nothing is serving ACP, so there is no session map to report. This is
            not an empty list: a running shell with no client attached would say so here.
          </p>
        )}
      </div>

      {/* Conformance. The same source `initialize` derives its capability response from. */}
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-3">
        <div className="flex items-center justify-between">
          <div className="text-[10px] text-slate-500 uppercase font-sans">
            Agent side — a client calls these on us
          </div>
          <span data-testid="acp-agent-counts" className="text-[10px] font-mono text-slate-400">
            {acpStatus.counts.agent.implemented}/{acpStatus.counts.agent.total} implemented
          </span>
        </div>
        <div className="grid grid-cols-1 gap-2">
          {agentMethods.map((m) => (
            <MethodRow key={`agent-${m.name}`} method={m} />
          ))}
        </div>
      </div>

      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-3">
        <div className="flex items-center justify-between">
          <div className="text-[10px] text-slate-500 uppercase font-sans">
            Client side — we call these on the client
          </div>
          <span data-testid="acp-client-counts" className="text-[10px] font-mono text-slate-400">
            {acpStatus.counts.client.implemented}/{acpStatus.counts.client.total} implemented
          </span>
        </div>
        <div className="grid grid-cols-1 gap-2">
          {clientMethods.map((m) => (
            <MethodRow key={`client-${m.name}`} method={m} />
          ))}
        </div>
      </div>

      {/* MCP descriptors arriving over the protocol. */}
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-3">
        <div className="text-[10px] text-slate-500 uppercase font-sans">
          MCP descriptors carried by the protocol
        </div>
        <div className="grid grid-cols-1 gap-2">
          {acpStatus.mcp_descriptors.map((d) => (
            <div
              key={d.name}
              data-testid={`acp-descriptor-${d.name}`}
              className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/60 space-y-1"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-[12px] text-slate-200">{d.name}</span>
                <StatusPill status={d.status} />
              </div>
              <p className="text-[11px] text-slate-400 leading-snug">{d.note}</p>
            </div>
          ))}
        </div>
        <div className="p-2.5 rounded-xl bg-rose-950/25 border border-rose-900/40 flex gap-2">
          <ShieldAlert className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
          <p data-testid="acp-loader-warning" className="text-[11px] text-rose-200 leading-snug">
            {acpStatus.mcp_loader_warning}
          </p>
        </div>
      </div>
    </div>
  );
};
