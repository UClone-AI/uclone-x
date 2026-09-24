import { useState, useMemo } from 'react';
import {
  Terminal,
  Search,
  Play,
  Pause,
  Trash2,
  ChevronDown,
  ChevronRight,
  Database,
  Clock,
  Copy,
  Check,
  Shield,
  Activity,
} from 'lucide-react';
import { EventEnvelope } from '../types';
import { Button } from './ui/Button';

interface LedgerTabProps {
  events: EventEnvelope[];
  isPaused: boolean;
  onTogglePause: () => void;
  onClearEvents: () => void;
  lastHeartbeat: string;
}

export function LedgerTab({
  events,
  isPaused,
  onTogglePause,
  onClearEvents,
  lastHeartbeat,
}: LedgerTabProps) {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedType, setSelectedType] = useState<string>('ALL');
  const [selectedPriority, setSelectedPriority] = useState<string>('ALL');
  const [expandedEventId, setExpandedEventId] = useState<string | null>(null);
  const [copiedEventId, setCopiedEventId] = useState<string | null>(null);

  const filteredEvents = useMemo(() => {
    return events.filter((ev) => {
      const matchesType =
        selectedType === 'ALL' ||
        ev.type === selectedType ||
        ev.event_type === selectedType;

      const matchesPriority =
        selectedPriority === 'ALL' || ev.priority === selectedPriority;

      const term = searchTerm.toLowerCase();
      const matchesSearch =
        !term ||
        ev.type.toLowerCase().includes(term) ||
        (ev.source && ev.source.toLowerCase().includes(term)) ||
        (ev.sender_id && ev.sender_id.toLowerCase().includes(term)) ||
        (ev.recipient_id && ev.recipient_id.toLowerCase().includes(term)) ||
        (ev.topic && ev.topic.toLowerCase().includes(term)) ||
        (ev.event_type && ev.event_type.toLowerCase().includes(term)) ||
        JSON.stringify(ev.payload || '').toLowerCase().includes(term);

      return matchesType && matchesPriority && matchesSearch;
    });
  }, [events, selectedType, selectedPriority, searchTerm]);

  const handleCopyPayload = (ev: EventEnvelope) => {
    navigator.clipboard.writeText(JSON.stringify(ev, null, 2));
    setCopiedEventId(ev.id);
    setTimeout(() => setCopiedEventId(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Top Stream Status & Controls Bar */}
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-tr from-emerald-600 to-teal-600 rounded-xl shadow-emerald-500/20 shadow-md">
            <Terminal className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              EventBus SSE Stream Ledger
              <span className="text-[10px] font-mono bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 px-2 py-0.5 rounded">
                Live SSE ({filteredEvents.length} events)
              </span>
            </h2>
            <p className="text-[11px] text-slate-400">
              Low-overhead SSE event stream with sequence numbers, priorities, and provenance
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="text-xs font-mono text-slate-400 bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span>Last Heartbeat: <strong className="text-slate-200">{lastHeartbeat}</strong></span>
          </div>

          <Button
            variant="bordered"
            onClick={onTogglePause}
            className={`px-3 py-1.5 rounded-xl gap-1.5 transition-all ${
              isPaused
                ? 'bg-amber-950/80 text-amber-300 border-amber-800 shadow-md shadow-amber-950/30 hover:bg-amber-950/80 hover:text-amber-300'
                : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700 hover:text-slate-300'
            }`}
          >
            {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            {isPaused ? 'Resume Stream' : 'Pause Stream'}
          </Button>

          <Button
            variant="bordered"
            onClick={onClearEvents}
            className="px-3 py-1.5 rounded-xl gap-1.5 bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300 hover:text-slate-300"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear
          </Button>
        </div>
      </div>

      {/* Filter & Search Toolbar */}
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg flex flex-wrap gap-3">
        <div className="relative grow basis-48">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
          <input
            type="text"
            placeholder="Search topic, sender, recipient, type, or payload JSON..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 shadow-inner"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            value={selectedPriority}
            onChange={(e) => setSelectedPriority(e.target.value)}
            className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500 font-mono shadow-inner"
          >
            {/*
              The three the stream can actually carry: `src/uclone_x/ui/app.py` stamps
              "P1" on SYSTEM_CONNECTED, "P2" on AGENT_EVENT and "P3" on HEARTBEAT, and
              stamps nothing else. A fourth option ("P0 - Critical") was offered here and
              could only ever filter the ledger down to nothing (#1027).
            */}
            <option value="ALL">All Priorities</option>
            <option value="P1">P1 - High</option>
            <option value="P2">P2 - Normal</option>
            <option value="P3">P3 - Low</option>
          </select>

          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500 font-mono shadow-inner"
          >
            <option value="ALL">All Event Types</option>
            <option value="AGENT_EVENT">Agent Events</option>
            <option value="SYSTEM_CONNECTED">System Connects</option>
            <option value="HEARTBEAT">Heartbeats</option>
            <option value="user_input">User Input</option>
            <option value="agent_reply">Agent Reply</option>
            <option value="subagent_spawn">Subagent Spawn</option>
          </select>
        </div>
      </div>

      {/* Event Stream List */}
      <div className="p-5 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg space-y-3">
        {filteredEvents.length === 0 ? (
          <div className="p-16 text-center text-slate-500 text-xs">
            <Database className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p className="font-semibold text-slate-400">No events found matching current criteria.</p>
            <p className="text-[11px] text-slate-500 mt-1">
              Events published to EventBus or FastPath A2A channels will appear here in real-time.
            </p>
          </div>
        ) : (
          filteredEvents.map((ev) => {
            const isExpanded = expandedEventId === ev.id;
            const isCopied = copiedEventId === ev.id;
            const isAgentEvent = ev.type === 'AGENT_EVENT';
            const isHeartbeat = ev.type === 'HEARTBEAT';

            return (
              <div
                key={ev.id}
                className="p-3.5 bg-slate-950/80 border border-slate-800/80 rounded-xl hover:border-slate-700 transition-colors text-xs"
              >
                <div
                  className="flex flex-wrap items-center justify-between gap-3 cursor-pointer select-none"
                  onClick={() => setExpandedEventId(isExpanded ? null : ev.id)}
                >
                  <div className="flex items-center gap-2.5">
                    {isExpanded ? (
                      <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                    )}

                    {/* Sequence Badge */}
                    {ev.seq !== undefined && (
                      <span className="font-mono text-[10px] text-slate-500 font-bold">
                        #{ev.seq}
                      </span>
                    )}

                    {/* Priority Badge */}
                    {ev.priority && (
                      <span
                        className={`text-[9px] font-mono font-bold px-1.5 py-0.2 rounded ${
                          ev.priority === 'P1'
                            ? 'bg-amber-950 text-amber-300 border border-amber-800'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {ev.priority}
                      </span>
                    )}

                    {/* Type Badge */}
                    <span
                      className={`font-mono text-[10px] font-bold px-2 py-0.5 rounded ${
                        isAgentEvent
                          ? 'bg-cyan-950 text-cyan-300 border border-cyan-800/60'
                          : isHeartbeat
                          ? 'bg-slate-800 text-slate-400 border border-slate-700'
                          : 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                      }`}
                    >
                      {ev.event_type || ev.type}
                    </span>

                    {/* Topic */}
                    <span className="font-mono text-slate-300 font-medium">
                      {ev.topic || 'runtime.default'}
                    </span>
                  </div>

                  {/* Routing: Sender -> Recipient */}
                  <div className="flex items-center gap-4 text-slate-400 text-[11px] font-mono">
                    {(ev.sender_id || ev.source) && (
                      <span className="text-slate-300">
                        {ev.sender_id || ev.source}
                        {ev.recipient_id ? ` ➔ ${ev.recipient_id}` : ''}
                      </span>
                    )}

                    <span className="flex items-center gap-1 text-[10px] text-slate-500">
                      <Clock className="w-3 h-3" />
                      {typeof ev.timestamp === 'number'
                        ? `${ev.timestamp.toFixed(1)}s`
                        : ev.timestamp.slice(11, 19)}
                    </span>
                  </div>
                </div>

                {/* Expanded Payload & Provenance Details */}
                {isExpanded && (
                  <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-3">
                    <div className="flex items-center justify-between text-[11px] text-slate-400">
                      <div className="flex items-center gap-3">
                        <span>Event ID: <strong className="font-mono text-slate-300">{ev.event_id || ev.id}</strong></span>
                        {ev.provenance?.component && (
                          <span className="text-purple-400 font-mono text-[10px] flex items-center gap-1">
                            <Shield className="w-3 h-3" />
                            {ev.provenance.component}
                          </span>
                        )}
                      </div>

                      <Button
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCopyPayload(ev);
                        }}
                        className="px-2.5 py-1 gap-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-slate-300 font-mono text-[10px]"
                      >
                        {isCopied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                        {isCopied ? 'Copied' : 'Copy JSON'}
                      </Button>
                    </div>

                    <div className="bg-slate-900/90 p-3 rounded-xl border border-slate-800 font-mono text-[11px] text-slate-200 overflow-x-auto shadow-inner">
                      <pre>{JSON.stringify(ev, null, 2)}</pre>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
