import React, { useState, useEffect, useRef } from 'react';
import {
  FileText,
  Brain,
  Activity,
  Gauge,
  X,
  Bot,
  MessageSquare,
  BookOpen,
  Globe,
} from 'lucide-react';
import { DocViewer } from '../artifacts/DocViewer';
import { MusicView } from '../artifacts/MusicView';
import { StoryView } from '../artifacts/StoryView';
import { VideoView } from '../artifacts/VideoView';
import { BrowserPanel } from '../browser/BrowserPanel';
import type { BrowserLive } from '../../lib/browserLive';
import { KnowledgeGraphViewer } from '../artifacts/KnowledgeGraphViewer';
import { ActivityTimeline } from '../artifacts/ActivityTimeline';
import { ResourceSummary } from '../artifacts/ResourceSummary';
import { RemembersPanel } from '../artifacts/RemembersPanel';
import { agentSeats, transcriptRefreshKey } from '../../lib/roomDock';
import { useCopy } from '../../i18n';
// The dock is one of the workspace's top-level regions and its parent spans the window, so the
// quantity that decides whether it fits on screen is the window's own width. The panels
// *inside* it follow the dock's width instead, through `.dock-scope` (ui-authoring §3).
import { useWindowWidth } from '../../lib/rail';
import { cloneIdOf } from '../../lib/cloneLabel';
import { shownSurface } from '../../lib/developerMode';
import { TurnDetail } from './TurnDetail';
import { CloneProfile } from '../clones/CloneProfile';
import type { PersonaDraft, PersonaEditMode, PersonaSaveResult } from '../../lib/personaDraft';
import {
  BudgetData,
  DockSurface,
  EventEnvelope,
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
  /** Current mode of the clone panel. Defaults to 'view'. */
  cloneMode?: 'view' | 'edit' | 'create';
  /** Switch between view, edit, and create clone modes. */
  onCloneModeChange?: (mode: 'view' | 'edit' | 'create') => void;
  /** Available tools from runtime. */
  availableTools?: readonly string[];
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
  /**
   * The open conversation's browser, live (`lib/browserLive.ts`). `App` holds it so the step
   * lines under a turn read the same socket; without it the Browser tab opens its own.
   */
  browserLive?: BrowserLive;
  /** Open Settings ▸ Browser to connect U0's Chrome. */
  onConnectBrowser?: () => void;
  /** Open Settings ▸ Browser on the Saved sign-ins form for `site` (the Save offer). */
  onSaveBrowserSignIn?: (site: string) => void;
  /** Send a user message into the open conversation (e.g. from MusicView Voice Rhythm or Fast Edit). */
  onSendRoomMessage?: (content: string) => Promise<void> | void;
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
  budgetData,
  room = null,
  roomContext = null,
  selectedTurnSeq = null,
  selectedClone,
  personas,
  cloneMode = 'view',
  onCloneModeChange,
  availableTools,
  canWritePersonas,
  onSavePersona,
  cloneStudio = false,
  onCloneStudioChange,
  onRefresh,
  isRefreshing,
  browserLive,
  onConnectBrowser,
  onSaveBrowserSignIn,
  onSendRoomMessage,
}) => {
  const [width, setWidth] = useState<number>(readStoredWidth);
  const [artifactsMode, setArtifactsMode] = useState<'story' | 'music' | 'video' | 'files'>('story');
  const [localVideoProjectId, setLocalVideoProjectId] = useState<string | null>(null);
  const [availableVideoProjects, setAvailableVideoProjects] = useState<
    Array<{ project_id: string; title: string; total_timecode: string }>
  >([]);
  const draggingRef = useRef<boolean>(false);
  const windowWidth = useWindowWidth();
  // The product surfaces' names and the dock's own chrome. The developer drawer's tabs stay
  // English with the instruments they open (`multilingual-ui.md` step 7).
  const t = useCopy().dock;
  // What is on screen, which is the selection unless developer mode is off and the selection is
  // one of its surfaces: then Docs & Artifacts, never a blank panel.
  const surface = shownSurface(activeSurface, developerMode);
  const activeStoryId = room && room.room_id === roomId && room.story_id ? room.story_id : null;
  // The dock re-reads when the conversation moves on: a new message is a new record to show.
  // So is a clone learning from a turn after it (#1404): the row gains what was saved while
  // the transcript keeps its length, so each saved fact and each failure moves the key too.
  const refreshKey = room && room.room_id === roomId ? transcriptRefreshKey(room.transcript) : 0;
  // Only the agents in *this* conversation, and only while `room` is the one on screen.
  const seats = room && room.room_id === roomId ? agentSeats(room) : [];
  const composerIds = new Set<string>(['composer']);
  const directorIds = new Set<string>(['director']);
  for (const p of personas) {
    if (p.name === 'composer') composerIds.add(cloneIdOf(p));
    if (p.name === 'director') directorIds.add(cloneIdOf(p));
  }
  const isMusicRoom =
    composerIds.has(selectedClone) ||
    seats.some(
      (s) =>
        composerIds.has(s.id) ||
        s.handle === 'composer' ||
        s.display_name?.toLowerCase() === 'composer' ||
        s.display_name === '\uc791\uace1\uac00',
    );
  const hasMusicOption =
    isMusicRoom || personas.some((p) => p.name === 'composer');
  const isVideoRoom =
    directorIds.has(selectedClone) ||
    seats.some(
      (s) =>
        directorIds.has(s.id) ||
        s.handle === 'director' ||
        s.display_name?.toLowerCase() === 'director' ||
        s.display_name === '\uac10\ub3c5',
    );
  const hasVideoOption =
    isVideoRoom || personas.some((p) => p.name === 'director');
  const shownArtifactsMode =
    artifactsMode === 'story' && isVideoRoom
      ? 'video'
      : artifactsMode === 'story' && isMusicRoom
        ? 'music'
        : artifactsMode;
  useEffect(() => {
    if (selectedArtifactPath) {
      setArtifactsMode('files');
    }
  }, [selectedArtifactPath]);
  useEffect(() => {
    if (selectedArtifactPath) return;
    if (isVideoRoom) {
      setArtifactsMode('video');
    } else if (isMusicRoom) {
      setArtifactsMode('music');
    } else if (activeStoryId) {
      setArtifactsMode('story');
    }
  }, [activeStoryId, isMusicRoom, isVideoRoom, selectedArtifactPath]);

  const effectiveVideoProjectId = activeStoryId ?? localVideoProjectId;

  useEffect(() => {
    if (artifactsMode !== 'video' || !hasVideoOption || effectiveVideoProjectId) return;
    let cancelled = false;
    void fetch('/api/artifacts/library/videos')
      .then(async (res) => {
        if (!res.ok) return [];
        const data = (await res.json()) as Array<{
          project_id: string;
          title: string;
          total_timecode: string;
        }>;
        return Array.isArray(data) ? data : [];
      })
      .then((items) => {
        if (!cancelled) setAvailableVideoProjects(items);
      })
      .catch(() => {
        if (!cancelled) setAvailableVideoProjects([]);
      });
    return () => {
      cancelled = true;
    };
  }, [artifactsMode, effectiveVideoProjectId, hasVideoOption]);
  const seatName = (() => {
    const seat = seats.find((s) => s.id === seatId);
    return seat ? (seat.display_name ?? seat.id) : undefined;
  })();
  const seatScoped =
    surface === 'activity' || surface === 'remembers' || surface === 'knowledge_graph';
  // The surfaces that read the open conversation. While it is opening, or could not be read,
  // they are replaced by the sentence that says which, rather than each claiming none is open.
  const roomScoped =
    seatScoped || surface === 'artifacts' || surface === 'browser' || surface === 'resource';
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
    { id: 'clone', label: t.tabs.clone, icon: <Bot className="w-3.5 h-3.5" />, accent: 'indigo' },
    // Beside Clone for the same reason: it is about what the reader picked -- the turn
    // `why ›` was pressed on -- and not about the runtime at large.
    {
      id: 'turn',
      label: t.tabs.turn,
      icon: <MessageSquare className="w-3.5 h-3.5" />,
      accent: 'sky',
    },
    {
      id: 'artifacts',
      label: t.tabs.artifacts,
      icon: <FileText className="w-3.5 h-3.5" />,
      accent: 'cyan',
    },
    {
      id: 'browser',
      label: t.tabs.browser,
      icon: <Globe className="w-3.5 h-3.5" />,
      accent: 'blue',
    },
    // What the seat remembers, in sentences (#1357). The graph behind it is a developer
    // instrument; this is the everyday reading of the same store.
    {
      id: 'remembers',
      label: t.tabs.remembers,
      icon: <BookOpen className="w-3.5 h-3.5" />,
      accent: 'violet',
    },
    {
      id: 'activity',
      label: t.tabs.activity,
      icon: <Activity className="w-3.5 h-3.5" />,
      accent: 'amber',
    },
    {
      id: 'resource',
      label: t.tabs.resource,
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
  ];

  const accentClass = (accent: string, isActive: boolean): string => {
    if (!isActive) return 'text-slate-400 hover:text-slate-200 hover:bg-slate-900';
    const byAccent: Record<string, string> = {
      cyan: 'bg-cyan-600/30 text-cyan-300 border-cyan-500/50',
      blue: 'bg-blue-600/30 text-blue-300 border-blue-500/50',
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
        title={t.chrome.resize}
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
          title={t.chrome.close}
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Which clone the seat-scoped surfaces describe. Offered only when there is a choice:
          with one seat there is nothing to pick, and a control with one option is noise. */}
      {seatScoped && seats.length > 1 && (
        <div className="px-3 pt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-400 shrink-0">
          <label htmlFor="dock-seat-picker">{t.chrome.showing}</label>
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
              persona={personas.find((p) => cloneIdOf(p) === selectedClone)}
              mode={cloneMode}
              onModeChange={onCloneModeChange}
              availableTools={availableTools}
              developerMode={developerMode}
              existingNames={personas.map((p) => p.name)}
              canWrite={canWritePersonas}
              onSave={onSavePersona}
              studio={cloneStudio}
              onStudioChange={onCloneStudioChange}
            />
          </div>
        )}

        {/* Surface 1: Docs & Artifacts (with inline World KG & Story Navigator when a story is open) */}
        {pending === null && surface === 'artifacts' && (
          activeStoryId || hasMusicOption || hasVideoOption ? (
            <div className="h-full flex flex-col min-h-0">
              <div
                className="px-3 py-1.5 flex items-center gap-1 border-b border-slate-800/80 bg-slate-900/40 shrink-0"
                data-testid="dock-story-switcher"
              >
                {activeStoryId && !isMusicRoom && !isVideoRoom && (
                  <button
                    type="button"
                    data-testid="dock-mode-story"
                    onClick={() => setArtifactsMode('story')}
                    className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                      shownArtifactsMode === 'story'
                        ? 'bg-cyan-600/30 text-cyan-300 border border-cyan-500/50'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    }`}
                  >
                    {t.story.dockStoryMode}
                  </button>
                )}
                {hasMusicOption && (
                  <button
                    type="button"
                    data-testid="dock-mode-music"
                    onClick={() => setArtifactsMode('music')}
                    className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                      shownArtifactsMode === 'music'
                        ? 'bg-cyan-600/30 text-cyan-300 border border-cyan-500/50'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    }`}
                  >
                    🎵 Music Score
                  </button>
                )}
                {hasVideoOption && (
                  <button
                    type="button"
                    data-testid="dock-mode-video"
                    onClick={() => setArtifactsMode('video')}
                    className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                      shownArtifactsMode === 'video'
                        ? 'bg-cyan-600/30 text-cyan-300 border border-cyan-500/50'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    }`}
                  >
                    🎬 Video Studio
                  </button>
                )}
                <button
                  type="button"
                  data-testid="dock-mode-files"
                  onClick={() => setArtifactsMode('files')}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                    shownArtifactsMode === 'files'
                      ? 'bg-cyan-600/30 text-cyan-300 border border-cyan-500/50'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                  }`}
                >
                  {t.story.dockFilesMode}
                </button>
              </div>
              <div className="flex-1 min-h-0 flex flex-col">
                {shownArtifactsMode === 'video' && hasVideoOption ? (
                  effectiveVideoProjectId ? (
                    <VideoView
                      projectId={effectiveVideoProjectId}
                      conversationId={roomId}
                      refreshKey={refreshKey}
                      onBack={() => setArtifactsMode('files')}
                    />
                  ) : (
                    <div data-testid="video-no-project" className="p-4 space-y-3 text-xs text-slate-400">
                      <p>
                        No video project is open in this conversation yet. Ask Director to start one
                        or open one.
                      </p>
                      <div className="space-y-1.5">
                        {availableVideoProjects.map((vp) => (
                          <button
                            key={vp.project_id}
                            type="button"
                            data-testid={`video-open-project-${vp.project_id}`}
                            onClick={() => {
                              void fetch(
                                `/api/artifacts/library/videos/${encodeURIComponent(vp.project_id)}/open`,
                                {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify({ room_id: roomId ?? undefined }),
                                }
                              ).finally(() => {
                                setLocalVideoProjectId(vp.project_id);
                                onRefresh();
                              });
                            }}
                            className="w-full px-3 py-2 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-left flex items-center justify-between text-slate-200"
                          >
                            <span className="font-medium">🎬 {vp.title}</span>
                            <span className="text-[10px] font-mono text-cyan-300">
                              {vp.project_id} • {vp.total_timecode}
                            </span>
                          </button>
                        ))}
                        <button
                          type="button"
                          data-testid="video-create-sample-project-btn"
                          onClick={() => {
                            void fetch('/api/artifacts/library/videos', {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({
                                title: 'Seoul Cyber Cut',
                                project_id: 'seoul-cyber-cut',
                                room_id: roomId ?? undefined,
                              }),
                            })
                              .then(async (res) => {
                                if (!res.ok) return null;
                                return (await res.json()) as { project_id?: string };
                              })
                              .then((created) => {
                                const pid = created?.project_id ?? 'seoul-cyber-cut';
                                setLocalVideoProjectId(pid);
                                onRefresh();
                              });
                          }}
                          className="px-3 py-1.5 rounded bg-cyan-600/30 hover:bg-cyan-600/50 border border-cyan-500/40 text-cyan-200 font-medium"
                        >
                          + Start Sample Video Studio Project
                        </button>
                      </div>
                    </div>
                  )
                ) : shownArtifactsMode === 'music' && hasMusicOption ? (
                  activeStoryId !== null ? (
                    <MusicView
                      songId={activeStoryId}
                      conversationId={roomId}
                      refreshKey={refreshKey}
                      onBack={() => setArtifactsMode('files')}
                      onFastEditPrompt={
                        onSendRoomMessage
                          ? (prompt) => {
                              void onSendRoomMessage(prompt);
                            }
                          : undefined
                      }
                    />
                  ) : (
                    <p data-testid="music-no-song" className="p-4 text-xs text-slate-400">
                      No song is open in this conversation yet. Ask Composer to start one or
                      open one.
                    </p>
                  )
                ) : shownArtifactsMode === 'story' && activeStoryId && !isMusicRoom && !isVideoRoom ? (
                  <StoryView
                    storyId={activeStoryId}
                    refreshKey={refreshKey}
                    onBack={() => setArtifactsMode('files')}
                  />
                ) : (
                  <DocViewer
                    roomId={roomId}
                    selectedArtifactPath={selectedArtifactPath}
                    refreshKey={refreshKey}
                  />
                )}
              </div>
            </div>
          ) : (
            <DocViewer
              roomId={roomId}
              selectedArtifactPath={selectedArtifactPath}
              refreshKey={refreshKey}
            />
          )
        )}

        {/* Surface: Browser */}
        {pending === null && surface === 'browser' && (
          <BrowserPanel
            roomId={roomId}
            room={room && room.room_id === roomId ? room : null}
            live={browserLive}
            onConnect={onConnectBrowser}
            onSaveSignIn={onSaveBrowserSignIn}
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
              developerMode={developerMode}
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
