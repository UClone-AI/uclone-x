import React, { useState, useEffect, useRef } from 'react';
import {
  FileText,
  Brain,
  Activity,
  Gauge,
  X,
  Network,
  Terminal,
  Bot,
  MessageSquare,
  BookOpen,
} from 'lucide-react';
import { DocViewer } from '../artifacts/DocViewer';
import { KnowledgeGraphViewer } from '../artifacts/KnowledgeGraphViewer';
import { ActivityTimeline } from '../artifacts/ActivityTimeline';
import { ResourceSummary } from '../artifacts/ResourceSummary';
import { RemembersPanel } from '../artifacts/RemembersPanel';
import { agentSeats } from '../../lib/roomDock';
// The dock is one of the workspace's top-level regions and its parent spans the window, so the
// quantity that decides whether it fits on screen is the window's own width. The panels
// *inside* it follow the dock's width instead, through `.dock-scope` (ui-authoring §3).
import { useWindowWidth } from '../../lib/rail';
import { shownSurface } from '../../lib/developerMode';
import { TopologyTab } from '../TopologyTab';
import { LedgerTab } from '../LedgerTab';
import { OntologyTab } from '../OntologyTab';
import { TurnDetail } from './TurnDetail';
import { CloneProfile } from '../clones/CloneProfile';
import type { PersonaDraft, PersonaEditMode, PersonaSaveResult } from '../../lib/personaDraft';
import {
  BudgetData,
  DockSurface,
  EventEnvelope,
  OntologyData,
  PersonaInfo,
  RoomContext,
  RoomState,
} from '../../types';

const MIN_WIDTH = 360;
const MAX_WIDTH = 900;
const DEFAULT_WIDTH = 520;
const WIDTH_KEY = 'uclone-x.dock.width';

const readStoredWidth = (): number => {
  try {
    const raw = window.localStorage.getItem(WIDTH_KEY);
    if (raw === null) return DEFAULT_WIDTH;
    const parsed = Number.parseInt(raw, 10);
    if (Number.isNaN(parsed)) return DEFAULT_WIDTH;
    return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, parsed));
  } catch {
    return DEFAULT_WIDTH;
  }
};

const storeWidth = (width: number): void => {
  try {
    window.localStorage.setItem(WIDTH_KEY, String(width));
  } catch {
    /* Safe fallback */
  }
};

interface SurfaceTabDef {
  id: DockSurface;
  label: string;
  icon: React.ReactNode;
  accent: string;
  description?: string;
}

export interface ArtifactsDockProps {
  activeSurface: DockSurface;
  onSelectSurface: (surface: DockSurface) => void;
  onClose: () => void;
  /**
   * How much of the workspace row is not the dock's to take: the session rail's width while
   * it is open, 0 while it is closed.
   *
   * The dock cannot read this off the window, and the window is not the quantity that decides
   * whether it fits. A 400px window seats a 360px dock perfectly well with the rail closed and
   * not at all with it open, and the difference is 240px the dock cannot see. Without it the
   * dock had to guess from the window alone, and every rule expressible that way is wrong in
   * one direction or the other (#1030).
   *
   * Required, with no default. It was optional and defaulted to `0` until #1042, which meant a
   * mount site that forgot it got the pre-#1030 window-only rule back — silently, with no type
   * error and no test failing, because `reservedWidth = 0` is exactly that rule. That is the
   * silent-fallback shape P6 names, and the only mount site is `App.tsx`, so requiring it costs
   * that site nothing and turns the next one's omission into a build error.
   */
  reservedWidth: number;
  /**
   * Whether the developer drawer is offered (Settings -> Developer mode; off by default).
   *
   * Required for the reason `reservedWidth` is: a mount site that forgot it must not quietly
   * get either answer.
   */
  developerMode: boolean;
  /**
   * The conversation the centre column shows. Docs, Activity, Remembers and the DAG read
   * it and nothing else (owner ruling 2026-09-22, #1356); `null` when no conversation is
   * open, which each surface says in words.
   */
  roomId: string | null;
  /**
   * Why `roomId` is `null` while a conversation *is* open: it is still loading, or it could
   * not be read, with the cause (#1389). `null` when no conversation is open (each surface
   * says so itself) or when it is on screen.
   *
   * Without it every conversation-scoped surface said "No conversation is open" through
   * each load and after a failed read: an absence stated with the wrong cause (P6).
   * Required for the reason `reservedWidth` is.
   */
  roomPending: string | null;
  /**
   * The seat Activity, Remembers and the Knowledge Graph describe, or `null` when no clone
   * is seated. Chosen by `App` (the inspected clone, else the last to speak); the picker
   * here changes it and appears only when there is more than one seat to choose between.
   */
  seatId: string | null;
  onSelectSeat?: (seatId: string) => void;
  /** The file Docs shows; set by "Open in Docs" anywhere on the page. */
  selectedArtifactPath?: string | null;
  /** Front Docs on a file (an Activity row that wrote one). */
  onOpenInDocs?: (path: string) => void;
  events: EventEnvelope[];
  ontology: OntologyData | null;
  budgetData: BudgetData | null;
  /**
   * The conversation on screen and how full its seats are, for the Resource surface (#1272).
   *
   * These replace `runSteps` / `stepBudgetMax` / `conversationTurns` / `tokenTotal`, which
   * were derived from `chatMessages` — the retired single-agent history, not the room the
   * centre column shows.
   */
  room?: RoomState | null;
  roomContext?: RoomContext | null;
  /**
   * The turn `why ›` was pressed on, by `seq` in `room`'s transcript, or `null`.
   *
   * A `seq` rather than the message, so the Turn surface re-reads it from `room` on every
   * render: a retried turn and a rewound one both change what that `seq` means, and a copy
   * taken when the control was pressed would go on showing the old reading.
   */
  selectedTurnSeq?: number | null;
  isPaused: boolean;
  onTogglePause: () => void;
  onClearEvents: () => void;
  lastHeartbeat: string;
  /**
   * The clone the rail has picked, and the installed definitions to find it among.
   *
   * The dock is handed both rather than the resolved persona, because a clone can be picked
   * that has no definition installed -- the Clone surface says so in words, and it cannot
   * tell that case from "not loaded yet" if the caller has already resolved it to
   * `undefined`.
   */
  selectedClone: string;
  personas: PersonaInfo[];
  /** Open a new conversation seating the named clone. */
  onStartConversation: (cloneId: string) => void;
  /** Current mode of the clone panel. Defaults to 'view'. */
  cloneMode?: 'view' | 'edit' | 'create';
  /** Switch between view, edit, and create clone modes. */
  onCloneModeChange?: (mode: 'view' | 'edit' | 'create') => void;
  /** Available tools from runtime. */
  availableTools?: readonly string[];
  /** Available models from runtime. */
  availableModels?: readonly string[];
  /** Whether the workspace directory is writable. */
  canWritePersonas?: boolean;
  /** Handler to persist the persona draft. */
  onSavePersona?: (draft: PersonaDraft, mode: PersonaEditMode) => Promise<PersonaSaveResult>;
  /**
   * The clone editor has the workspace to itself (Studio mode). `App` hides the conversation
   * column while this is true, and the dock takes the row's remaining width in normal flow
   * instead of its dragged width -- never a `fixed` layer, which this aside's `backdrop-blur`
   * would confine to its own box.
   */
  cloneStudio?: boolean;
  onCloneStudioChange?: (studio: boolean) => void;
  onRefresh: () => void;
  isRefreshing: boolean;
}

/**
 * ArtifactsDock: the workspace's side panel. Its tab row offers the user surfaces -- Clone,
 * Turn, Docs & Artifacts, Remembers, Activity & Tools, Resource -- and, only while developer
 * mode is on, a drawer below them offers the developer instruments (`DEVELOPER_SURFACES`).
 */
export const ArtifactsDock: React.FC<ArtifactsDockProps> = ({
  activeSurface,
  onSelectSurface,
  onClose,
  reservedWidth,
  developerMode,
  roomId,
  roomPending,
  seatId,
  onSelectSeat,
  selectedArtifactPath,
  onOpenInDocs,
  events,
  ontology,
  budgetData,
  room = null,
  roomContext = null,
  selectedTurnSeq = null,
  isPaused,
  onTogglePause,
  onClearEvents,
  lastHeartbeat,
  selectedClone,
  personas,
  onStartConversation,
  cloneMode = 'view',
  onCloneModeChange,
  availableTools,
  availableModels,
  canWritePersonas,
  onSavePersona,
  cloneStudio = false,
  onCloneStudioChange,
  onRefresh,
  isRefreshing,
}) => {
  const [width, setWidth] = useState<number>(readStoredWidth);
  const draggingRef = useRef<boolean>(false);
  const windowWidth = useWindowWidth();
  // What is on screen, which is the selection unless developer mode is off and the selection is
  // one of its surfaces: then Docs & Artifacts, never a blank panel.
  const surface = shownSurface(activeSurface, developerMode);
  // The dock re-reads when the conversation moves on: a new message is a new record to show.
  const refreshKey = room?.room_id === roomId ? (room?.transcript.length ?? 0) : 0;
  // Only the agents in *this* conversation, and only while `room` is the one on screen.
  const seats = room && room.room_id === roomId ? agentSeats(room) : [];
  const seatName = (() => {
    const seat = seats.find((s) => s.id === seatId);
    return seat ? (seat.display_name ?? seat.id) : undefined;
  })();
  const seatScoped =
    surface === 'activity' || surface === 'remembers' || surface === 'knowledge_graph';
  // The surfaces that read the open conversation. While it is opening, or could not be read,
  // they are replaced by the sentence that says which, rather than each claiming none is open.
  const roomScoped =
    seatScoped || surface === 'artifacts' || surface === 'topology' || surface === 'resource';
  const pending = roomId === null && roomScoped ? roomPending : null;

  // What is left of the row once the rail has taken its share is what decides whether the
  // dock can sit *in* the row; when it cannot, the dock is drawn over the workspace instead
  // (#1019). Reading the window alone could not express this, and the two ways of getting it
  // wrong are the whole of #1030: a rule that only compares the window with the dock's own
  // width over-fires above ~600px, where narrowing back into the row would have been fine,
  // and under-fires at 400px with the rail open, where no permitted width fits beside it.
  //
  // Nothing else changes state: the rail stays exactly where it was and is back when the
  // dock closes, which keeps this clear of the auto-collapse question open on #1013/#1018.
  const overlaid = windowWidth - reservedWidth <= width;

  // Overlaid, the dock covers the window, so there is no partially-covering state to read;
  // in the row it takes the width it was dragged to, which the line above has already
  // established the row can seat. Either way it never renders wider than the window.
  const renderedWidth = overlaid ? windowWidth : width;

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!draggingRef.current) return;
      const next = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, window.innerWidth - e.clientX));
      // The drag is never refused. An earlier revision of #1030 froze the width whenever the
      // dock was overlaid, which made it unrecoverable: one drag to the left edge at any
      // window up to MAX_WIDTH stored that window as the width, and from there the handle was
      // inert at that size forever -- through further drags, a reload, and closing and
      // reopening the dock. Refusing the gesture was the wrong half of the problem to fix;
      // with the row's own arithmetic above, a width that cannot be seated is drawn over the
      // workspace rather than off the edge of it, so the drag needs no guard at all.
      setWidth(next);
    };
    const handleUp = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      setWidth((current) => {
        storeWidth(current);
        return current;
      });
    };
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, []);

  const startDrag = () => {
    draggingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };

  // The user surfaces. The developer instruments are not here at all: they are the drawer's,
  // and the drawer exists only in developer mode.
  const primaryTabs: SurfaceTabDef[] = [
    // First, because it is the one surface about the thing the user picked rather than about
    // the runtime: picking a clone in the rail opens the dock here.
    { id: 'clone', label: 'Clone', icon: <Bot className="w-3.5 h-3.5" />, accent: 'indigo' },
    // Beside Clone for the same reason: it is about what the reader picked -- the turn
    // `why ›` was pressed on -- and not about the runtime at large.
    {
      id: 'turn',
      label: 'Turn',
      icon: <MessageSquare className="w-3.5 h-3.5" />,
      accent: 'sky',
    },
    {
      id: 'artifacts',
      label: 'Docs & Artifacts',
      icon: <FileText className="w-3.5 h-3.5" />,
      accent: 'cyan',
    },
    // What the seat remembers, in sentences (#1357). The graph behind it is a developer
    // instrument; this is the everyday reading of the same store.
    {
      id: 'remembers',
      label: 'Remembers',
      icon: <BookOpen className="w-3.5 h-3.5" />,
      accent: 'violet',
    },
    {
      id: 'activity',
      label: 'Activity & Tools',
      icon: <Activity className="w-3.5 h-3.5" />,
      accent: 'amber',
    },
    {
      id: 'resource',
      label: 'Resource',
      icon: <Gauge className="w-3.5 h-3.5" />,
      accent: 'emerald',
    },
  ];

  // The developer instruments, in `DEVELOPER_SURFACES` order. Rendered only in developer mode.
  const devTabs: SurfaceTabDef[] = [
    {
      id: 'knowledge_graph',
      label: 'Knowledge Graph',
      icon: <Brain className="w-3.5 h-3.5" />,
      accent: 'violet',
    },
    { id: 'topology', label: 'DAG', icon: <Network className="w-3.5 h-3.5" />, accent: 'indigo' },
    { id: 'ledger', label: 'EventBus', icon: <Terminal className="w-3.5 h-3.5" />, accent: 'cyan' },
    { id: 'ontology', label: 'Ontology', icon: <Brain className="w-3.5 h-3.5" />, accent: 'violet' },
  ];

  const accentClass = (accent: string, isActive: boolean): string => {
    if (!isActive) return 'text-slate-400 hover:text-slate-200 hover:bg-slate-900';
    const byAccent: Record<string, string> = {
      cyan: 'bg-cyan-600/30 text-cyan-300 border-cyan-500/50',
      violet: 'bg-violet-600/30 text-violet-300 border-violet-500/50',
      amber: 'bg-amber-600/30 text-amber-300 border-amber-500/50',
      emerald: 'bg-emerald-600/30 text-emerald-300 border-emerald-500/50',
      indigo: 'bg-indigo-600/30 text-indigo-300 border-indigo-500/50',
      teal: 'bg-teal-600/30 text-teal-300 border-teal-500/50',
      sky: 'bg-sky-600/30 text-sky-300 border-sky-500/50',
      rose: 'bg-rose-600/30 text-rose-300 border-rose-500/50',
    };
    return `${byAccent[accent] ?? 'bg-slate-700/40 text-slate-200'} font-semibold border`;
  };

  return (
    <aside
      data-testid="artifacts-dock"
      data-studio={cloneStudio ? 'true' : 'false'}
      style={cloneStudio ? undefined : { width: `${renderedWidth}px` }}
      className={`${
        cloneStudio ? 'relative flex-1 min-w-0' : overlaid ? 'absolute inset-y-0 right-0' : 'relative'
      } dock-scope bg-slate-950/95 border-l border-slate-800/80 flex flex-col h-full min-h-0 shrink-0 backdrop-blur-md shadow-2xl z-20`}
    >
      {/* Resizable handle -- not in Studio mode, where the dock has no width of its own */}
      {!cloneStudio && <div
        data-testid="dock-resize-handle"
        role="separator"
        aria-orientation="vertical"
        onMouseDown={startDrag}
        title="Drag to resize workspace dock"
        className="absolute left-0 top-0 h-full w-1 hover:w-1.5 cursor-col-resize bg-transparent hover:bg-cyan-500/60 transition-all z-30"
      />}

      {/* Primary Tab Navigation. The row wraps rather than scrolls: with six surfaces a
          scrolled row hides Activity and Resource past the edge of a default-width dock. */}
      <div className="px-3 py-2 flex items-start justify-between gap-2 shrink-0 border-b border-slate-800/80 select-none bg-slate-900/50">
        <div data-testid="dock-primary-tabs" className="flex flex-wrap items-center gap-1 text-xs min-w-0">
          {primaryTabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              data-testid={`tab-${tab.id}`}
              onClick={() => onSelectSurface(tab.id)}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 whitespace-nowrap transition-all ${accentClass(
                tab.accent,
                surface === tab.id
              )}`}
              title={tab.label}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        <button
          type="button"
          data-testid="dock-close"
          onClick={onClose}
          className="p-1 rounded-lg text-slate-500 hover:text-white hover:bg-slate-900 transition-colors shrink-0"
          title="Close workspace dock"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Which clone the seat-scoped surfaces describe. Offered only when there is a choice:
          with one seat there is nothing to pick, and a control with one option is noise. */}
      {seatScoped && seats.length > 1 && (
        <div className="px-3 pt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-400 shrink-0">
          <label htmlFor="dock-seat-picker">Showing</label>
          <select
            id="dock-seat-picker"
            data-testid="dock-seat-picker"
            value={seatId ?? ''}
            onChange={(e) => onSelectSeat?.(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-xs text-slate-200 max-w-full"
          >
            {seats.map((s) => (
              <option key={s.id} value={s.id}>
                {s.display_name ?? s.id}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Main Surface Display Area */}
      <div data-testid="dock-surface" className="flex-1 overflow-y-auto min-h-0">
        {pending !== null && (
          <p
            data-testid="dock-room-pending"
            className="text-xs text-slate-300 text-center max-w-sm mx-auto px-3 py-12"
          >
            {pending}
          </p>
        )}
        {/* Surface 0: the clone the rail has picked */}
        {surface === 'clone' && (
          <div className="h-full p-3">
            <CloneProfile
              cloneId={selectedClone}
              persona={personas.find((p) => p.name === selectedClone)}
              onStartConversation={onStartConversation}
              mode={cloneMode}
              onModeChange={onCloneModeChange}
              availableTools={availableTools}
              availableModels={availableModels}
              existingNames={personas.map((p) => p.name)}
              canWrite={canWritePersonas}
              onSave={onSavePersona}
              studio={cloneStudio}
              onStudioChange={onCloneStudioChange}
            />
          </div>
        )}

        {/* Surface 1: Docs & Artifacts */}
        {pending === null && surface === 'artifacts' && (
          <DocViewer
            roomId={roomId}
            selectedArtifactPath={selectedArtifactPath}
            refreshKey={refreshKey}
          />
        )}

        {/* Surface 2: Knowledge Graph (Interactive LinkML / triples) */}
        {pending === null && surface === 'knowledge_graph' && (
          <div className="h-full flex flex-col">
            <KnowledgeGraphViewer roomId={roomId} seatId={seatId} refreshKey={refreshKey} />
          </div>
        )}

        {/* Surface 3: Activity & Tools */}
        {pending === null && surface === 'activity' && (
          <div className="h-full flex flex-col p-3">
            <ActivityTimeline
              roomId={roomId}
              seatId={seatId}
              seatName={seatName}
              events={events}
              refreshKey={refreshKey}
              onOpenInDocs={onOpenInDocs}
            />
          </div>
        )}

        {pending === null && surface === 'remembers' && (
          <div className="h-full flex flex-col p-3">
            <RemembersPanel
              roomId={roomId}
              seatId={seatId}
              seatName={seatName}
              refreshKey={refreshKey}
            />
          </div>
        )}

        {/* Surface 4: Resource & Status */}
        {pending === null && surface === 'resource' && (
          <div className="h-full flex flex-col p-3">
            <ResourceSummary
              budgetData={budgetData}
              room={room}
              roomContext={roomContext}
              onRefresh={onRefresh}
              isRefreshing={isRefreshing}
            />
          </div>
        )}

        {/* Developer surfaces, reached from the drawer; `surface` is never one of them out of developer mode */}
        {pending === null && surface === 'topology' && (
          <div className="h-full p-3"><TopologyTab roomId={roomId} refreshKey={refreshKey} /></div>
        )}
        {surface === 'ledger' && (
          <div className="h-full p-3">
            <LedgerTab
              events={events}
              isPaused={isPaused}
              onTogglePause={onTogglePause}
              onClearEvents={onClearEvents}
              lastHeartbeat={lastHeartbeat}
            />
          </div>
        )}
        {surface === 'turn' && (
          <div className="h-full p-3">
            <TurnDetail
              room={room}
              seq={selectedTurnSeq}
              events={events}
              developerMode={developerMode}
            />
          </div>
        )}
        {surface === 'ontology' && (
          <div className="h-full p-3"><OntologyTab ontology={ontology} onRefresh={onRefresh} isLoading={isRefreshing} /></div>
        )}
      </div>

      {/* Developer drawer: absent, label and all, unless developer mode is on */}
      {developerMode && (
        <div data-testid="dev-drawer" className="border-t border-slate-800/80 bg-slate-950 shrink-0">
          <div className="px-3 pt-1.5 text-[11px] text-slate-500 font-mono">
            Developer Tools
          </div>
          <div className="p-2 bg-slate-950/90 flex flex-wrap gap-1">
            {devTabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                data-testid={`dev-tab-${tab.id}`}
                onClick={() => onSelectSurface(tab.id)}
                className={`px-2 py-1 rounded text-[10px] font-medium flex items-center gap-1 transition-all ${accentClass(
                  tab.accent,
                  surface === tab.id
                )}`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
};
