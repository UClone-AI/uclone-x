import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { Header } from './components/Header';
import { RAIL_WIDTH_PX, WorkspaceSidebar } from './components/layout/WorkspaceSidebar';
import { RoomConversation } from './components/rooms/RoomConversation';
import {
  NEW_CONVERSATION_TITLE,
  CONTEXT_READ_ATTEMPTS,
  roomsApi,
  roomReadReason,
  roomFailureReason,
  seededTitle,
  applyRoomEvent,
  roomResponseIsStale,
  RoomReadOrder,
  RoomSendError,
  RoomsNoAnswerError,
  EMPTY_LIVE,
  lastSeq,
  messageArrived,
  sendWasRefused,
  type RoomListing,
  type RoomLiveState,
  type RoomReadTicket,
} from './lib/rooms';
import { ArtifactsDock } from './components/layout/ArtifactsDock';
import {
  initialGroupRooms,
  initialPinnedClones,
  initialRailOpen,
  railOverlays,
  storeGroupRooms,
  storePinnedClones,
  storeRailOpen,
  useWindowWidth,
} from './lib/rail';
import { readDeveloperMode, storeDeveloperMode } from './lib/developerMode';
import { SettingsModal } from './components/SettingsModal';
import { ArtifactLibrary } from './components/artifacts/ArtifactLibrary';
import { artifactLibraryApi } from './lib/artifactLibrary';
import { CoreFailure } from './lib/coreFailure';
import { OpenInDocsContext, defaultSeat } from './lib/roomDock';
import { savePersona } from './lib/personasApi';
import { PERSONA_EDITOR_COPY } from './lib/personaCopy';
import type { PersonaDraft, PersonaEditMode, PersonaSaveResult } from './lib/personaDraft';
import {
  DockSurface,
  AgentInfo,
  EventEnvelope,
  OntologyData,
  BudgetData,
  RoomSummary,
  RoomContext,
  RoomHistoryAnswer,
  RoomState,
  RoomTopicEvent,
  PersonaInfo,
} from './types';

export function App() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [personas, setPersonas] = useState<PersonaInfo[]>([]);
  /**
   * Clones available for room invites: prioritizes installed personas and falls back to
   * runtime agents, matching the Rail's population logic (#1088).
   */
  const availableClones = useMemo<AgentInfo[]>(() => {
    if (personas.length > 0) {
      const personaClones: AgentInfo[] = personas.map((p) => {
        const live = agents.find((a) => a.id === p.name);
        return {
          id: p.name,
          label: p.name,
          role: p.role || 'Clone',
          status: live?.status || 'idle',
          tier: live?.tier || 'standard',
          isolation_level: live?.isolation_level || 'shared',
          capabilities: p.allowed_tools || [],
          uptime_s: live?.uptime_s || 0,
          current_task: live?.current_task || '',
          parent_id: live?.parent_id || null,
          subagents: live?.subagents || [],
        };
      });
      const personaIds = new Set(personas.map((p) => p.name));
      const extraAgents = agents.filter((a) => !personaIds.has(a.id));
      return [...personaClones, ...extraAgents];
    }
    return agents;
  }, [personas, agents]);
  const [cloneDockMode, setCloneDockMode] = useState<'view' | 'edit' | 'create'>('view');
  const [availableTools, setAvailableTools] = useState<string[]>([]);
  const [personasDir, setPersonasDir] = useState<string | null>(null);
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [ontology, setOntology] = useState<OntologyData | null>(null);
  const [budgetData, setBudgetData] = useState<BudgetData | null>(null);

  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [lastHeartbeat, setLastHeartbeat] = useState<string>('-');
  const [healthStatus, setHealthStatus] = useState<string>('healthy');

  // Interactive Chat State (FR-13)
  // Empty until the server names an agent. There is no built-in name to fall back on:
  // `champion` was a literal here, so a fresh install opened on an agent it did not have.
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [currentModel, setCurrentModel] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  /**
   * Per-agent model overrides, keyed by agent id. Session-scoped and never persisted:
   * this state lives only in the browser tab and is lost on reload, unlike `currentModel`
   * (the global default persisted via `/api/settings`). An agent with no entry here sends
   * `currentModel` as before (#1138).
   */
  const [agentModelOverrides, setAgentModelOverrides] = useState<Record<string, string>>({});
  /**
   * Whether the runtime has a model configured — `/api/models`'s `current_model`, which
   * is `settings["llm_model"]` and is `''` when nothing has been chosen.
   *
   * `null` until the first reply lands, so the rail does not accuse a healthy install of
   * having no model during startup. This used to be passed as `agents.length > 0`, which
   * made "no model" and "no agents" the same condition and left the no-agents sentence
   * unreachable.
   */
  const [modelConfigured, setModelConfigured] = useState<boolean | null>(null);

  // Two-region workspace: a conversation that is always present, and a dock of runtime
  // internals beside it. Neither is a tab of the other — that arrangement is what made
  // reading the DAG cost you the conversation.
  //
  // The rail's first state is decided from the window, and after that it is the user's: a
  // first run on a phone starts with it closed, and a collapse survives a reload (#1062).
  const windowWidth = useWindowWidth();
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(() =>
    initialRailOpen(window.innerWidth),
  );
  // Below `RAIL_OVERLAY_BELOW_PX` the open rail is drawn over the conversation rather than
  // taking 240px of a row that cannot spare it. Read on every render, so a resize moves the
  // rail between the two without changing whether it is open.
  const railIsOverlaid = railOverlays(windowWidth);
  const chooseSidebarOpen = useCallback((open: boolean) => {
    setIsSidebarOpen(open);
    storeRailOpen(open);
  }, []);
  const [pinnedClones, setPinnedClones] = useState<string[]>(() => initialPinnedClones('clone'));
  const handleTogglePinClone = useCallback((cloneId: string) => {
    setPinnedClones((prev) => {
      const next = prev.includes(cloneId)
        ? prev.filter((id) => id !== cloneId)
        : [...prev, cloneId];
      storePinnedClones(next);
      return next;
    });
  }, []);
  const [groupRoomIds, setGroupRoomIds] = useState<string[]>(initialGroupRooms);
  // Navigating from an overlaid rail closes it, as a phone-width drawer does: it covers the left
  // 240px of the conversation just opened. A rail in the row covers nothing and stays. Not
  // stored -- the user did not choose it, so a reload opens the rail as they last left it.
  const closeOverlaidRail = (): void => {
    if (railIsOverlaid) setIsSidebarOpen(false);
  };
  const [isDockOpen, setIsDockOpen] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  // The Files screen (#1554): a user feature, so not behind developer mode.
  const [isFilesOpen, setIsFilesOpen] = useState<boolean>(false);
  const [activeDockSurface, setActiveDockSurface] = useState<DockSurface>('artifacts');
  // Developer mode, off by default (owner ruling 2026-09-22): whether the dock offers its
  // developer instruments. A head preference kept in this browser, like the rail's open state.
  // Every programmatic `setActiveDockSurface` below opens a user surface ('clone', 'turn'), so
  // nothing here sends a first-time user to an instrument; the dock also falls back to Docs &
  // Artifacts should the selection be one while the mode is off.
  const [developerMode, setDeveloperMode] = useState<boolean>(readDeveloperMode);
  const changeDeveloperMode = useCallback((on: boolean): void => {
    setDeveloperMode(on);
    storeDeveloperMode(on);
  }, []);
  // Studio mode: the clone editor takes the conversation column and the dock, in the row, with
  // the rail left where it is. Screen state, so the head's (P8). It holds only while the editor
  // is what the dock shows; leaving the editor by any route ends it.
  const [cloneStudioRequested, setCloneStudioRequested] = useState<boolean>(false);
  const cloneStudio =
    cloneStudioRequested && isDockOpen && activeDockSurface === 'clone' && cloneDockMode !== 'view';
  useEffect(() => {
    if (cloneDockMode === 'view') setCloneStudioRequested(false);
  }, [cloneDockMode]);
  /**
   * The turn the dock's Turn surface is showing, by `seq` in the open conversation.
   *
   * The `seq` and not the message: the Turn surface re-reads it from `room` every render,
   * so a retry or a rewind changes what it shows instead of leaving it reporting a turn
   * the conversation no longer has.
   */
  const [selectedTurnSeq, setSelectedTurnSeq] = useState<number | null>(null);
  /** The clone the reader last picked or inspected in the rail; the dock's default seat. */
  const [preferredClone, setPreferredClone] = useState<string | null>(null);
  /** A seat picked in the dock, and the conversation it was picked in. */
  const [seatChoice, setSeatChoice] = useState<{ roomId: string; seatId: string } | null>(null);
  /** The file Docs fronts, set by any "Open in Docs" on the page (#1354). */
  const [selectedArtifactPath, setSelectedArtifactPath] = useState<string | null>(null);
  const openInDocs = useCallback((path: string) => {
    setSelectedArtifactPath(path);
    setActiveDockSurface('artifacts');
    setIsDockOpen(true);
  }, []);
  const [gitCommit, setGitCommit] = useState<string | undefined>(undefined);

  const eventSourceRef = useRef<EventSource | null>(null);
  const isPausedRef = useRef<boolean>(false);
  isPausedRef.current = isPaused;

  // Conversations, which seat one agent or several. Since #1208 retired the
  // single-agent surface there is no second thing the centre column can show, so a
  // null `currentRoomId` is a moment being passed through, not a place to rest in:
  // the effect below opens one.
  const [rooms, setRooms] = useState<RoomSummary[]>([]);
  const [roomsListed, setRoomsListed] = useState<boolean>(false);
  const [metadataListed, setMetadataListed] = useState<boolean>(false);
  const autoOpenRef = useRef<boolean>(false);
  /**
   * Whether a conversation is being created right now: from the click to the answer (#1288).
   *
   * `currentRoomId` was meant to be what tells the auto-open below that a conversation is
   * already coming, and it is set two awaits too late to do it -- `handleNewRoom` waits
   * for `POST /api/rooms` and then for the list re-read before `openRoom` names one. For
   * the whole of a user's New click the guard therefore read "nothing open, latch free",
   * and on a fresh install, where the list is empty, an agent list landing inside that
   * window made the effect start a second conversation beside the one being created.
   *
   * Held separately from `autoOpenRef` rather than folded into it, because the two are
   * released by different things. This one is released whatever the answer, so no single
   * create can disable the auto-open for the rest of the session; the latch is released
   * only by a delete. Taking the latch on entry to `handleNewRoom` instead would stop the
   * second create too, and would spend the auto-open's one turn on a click that may
   * produce no conversation at all.
   */
  const createInFlightRef = useRef<boolean>(false);
  const [currentRoomId, setCurrentRoomId] = useState<string | null>(null);
  const [room, setRoom] = useState<RoomState | null>(null);
  const [roomLive, setRoomLive] = useState<RoomLiveState>(EMPTY_LIVE);
  // Every room failure the user should see. `lib/rooms.ts` keeps the Core's own `detail` on
  // the error, and `roomFailureReason` shows that or, when the Core gave none, a plain
  // sentence of ours -- never the transport text or status line an error's `message` falls
  // back to (#1411). Swallowing the failure left a surface that could only go quiet.
  const [roomNotice, setRoomNotice] = useState<string | null>(null);
  /**
   * Why the rail's list of conversations is not current: the last list read failed (#1439).
   *
   * Its own state rather than a `roomNotice`, because opening or leaving a conversation
   * clears that notice, and on a fresh install the auto-open does exactly that a moment
   * after a failed list read -- the failure vanished while the rail still showed nothing.
   * Cleared only by a list read that lands, or by the user dismissing it.
   */
  const [roomsListError, setRoomsListError] = useState<string | null>(null);
  /** Ids of conversation records the Core has and cannot read, from the last list read (#1440). */
  const [unreadableRoomIds, setUnreadableRoomIds] = useState<string[]>([]);
  /**
   * Why the conversation being opened is not on screen: the read's own failure, or `null`
   * while it is still loading or once it has landed (#1389).
   *
   * Held apart from `roomNotice`, which the user can dismiss. Dismissing the banner does
   * not make the conversation readable, and without this the dock and the centre column
   * would go back to saying it is opening.
   *
   * It holds `roomReadReason`'s sentence, never a raw transport message: the Core's own
   * refusal, or a plain sentence of ours when there is none.
   */
  const [roomOpenError, setRoomOpenError] = useState<string | null>(null);
  /**
   * How full the open conversation is, read separately from the conversation itself.
   *
   * `null` until the first read lands, which is what the banner and the dock's per-seat
   * rows treat as "not asked yet" rather than as "not full" -- the two are different, and
   * a surface that drew one as the other would be naming a state nobody reported.
   */
  const [roomContext, setRoomContext] = useState<RoomContext | null>(null);
  const [isRoomCompacting, setIsRoomCompacting] = useState(false);
  /**
   * Seats the last rewind or clear did not reach, from that call's own answer.
   *
   * Held rather than derived: no later read of the room reports it, and a seat that kept
   * its turns is exactly what the surface must not render as a clean cut.
   */
  const [participantsNotReset, setParticipantsNotReset] = useState<readonly string[]>([]);
  /**
   * What the user has typed and not sent, per conversation (#1290).
   *
   * Held here rather than in the composer because the composer does not outlive a switch:
   * `openRoom` nulls `room` synchronously, `RoomConversation` unmounts, and state inside
   * it goes with it -- so a draft owned there is destroyed by the act of glancing at
   * another conversation, with no warning and no undo. Keyed by room id, so returning
   * gives back that conversation's own text and never another's.
   *
   * Only non-empty drafts are kept: the composer reports a cleared box as `''`, and
   * storing that would grow the map by one dead key per conversation ever opened.
   */
  const [drafts, setDrafts] = useState<Readonly<Record<string, string>>>({});
  const rememberDraft = useCallback((roomId: string, next: string) => {
    setDrafts((prev) => {
      if ((prev[roomId] ?? '') === next) return prev;
      if (next === '') {
        if (!(roomId in prev)) return prev;
        const { [roomId]: _gone, ...rest } = prev;
        return rest;
      }
      return { ...prev, [roomId]: next };
    });
  }, []);
  /**
   * `readRoomContext`, reachable from the event stream without being a dependency of it.
   * Naming the callback there would reconnect the stream whenever it was re-created.
   */
  const readRoomContextRef = useRef<(roomId: string) => void>(() => {});
  const currentRoomIdRef = useRef<string | null>(null);
  currentRoomIdRef.current = currentRoomId;

  /**
   * Which conversation the head is currently showing, as a number.
   *
   * Every read of a room resolves against the generation it started in, and a reply from
   * an older one is dropped. Without it `openRoom(A)` then `openRoom(B)` raced: A's
   * transcript could land after B's and stay on screen while `currentRoomId` was already
   * B -- so the next Send posted into a conversation the user was not reading.
   */
  const roomGenerationRef = useRef(0);

  /**
   * The order this head's reads of each room went out in (#1412). A read lands only while
   * it is the last one issued for its room, so neither an older transcript nor one from
   * before the stream's `final` replaces the one that `final`'s own read brings.
   */
  const roomReadsRef = useRef(new RoomReadOrder());

  /**
   * The open conversation's one seated clone, when it has exactly one.
   *
   * With a second clone seated there is no single model the header could state, so the
   * picker is not offered and nothing here is keyed.
   */
  const seatedClones = (room?.participants ?? []).filter((p) => p.kind === 'agent');
  const seatedCloneId = seatedClones.length === 1 ? seatedClones[0].id : null;

  /**
   * The seat the dock describes (#1356).
   *
   * A pick in the dock's seat picker holds for the conversation it was made in, while that
   * clone is still seated there. Otherwise the dock follows the clone the reader last picked
   * or inspected in the rail, else whoever spoke last, else the first seat (`defaultSeat`).
   * Switching conversations needs nothing here: the pick is keyed by room, so another room
   * falls back to its own default and the dock re-reads for it.
   */
  const dockSeat =
    room && seatChoice && seatChoice.roomId === room.room_id &&
    seatedClones.some((p) => p.id === seatChoice.seatId)
      ? seatChoice.seatId
      : defaultSeat(room, preferredClone);
  /** The conversation the dock reads: the open one, once its record is on screen. */
  const dockRoomId = room && room.room_id === currentRoomId ? currentRoomId : null;
  /**
   * Why the dock has no conversation to read while one is open (#1389, P6): it is still
   * loading, or it could not be read, with the cause. `null` when none is open, which the
   * dock's surfaces say themselves, or when it is on screen.
   */
  const dockRoomPending =
    currentRoomId !== null && dockRoomId === null
      ? roomOpenError !== null
        ? `Could not read this conversation: ${roomOpenError} Pick it again in the rail to retry.`
        : 'Opening this conversation…'
      : null;
  const chooseDockSeat = useCallback((seatId: string) => {
    const roomId = currentRoomIdRef.current;
    if (roomId) setSeatChoice({ roomId, seatId });
  }, []);

  /** Number a read of the open room that is going out now, for `commitRoom`. */
  const issueRoomRead = useCallback(
    (roomId: string): RoomReadTicket => roomReadsRef.current.issue(roomId, roomGenerationRef.current),
    [],
  );

  /** Whether a read's answer is to be dropped: the room moved on, or a later read went out. */
  const roomReadIsStale = useCallback(
    (ticket: RoomReadTicket) =>
      roomReadsRef.current.isStale(ticket, roomGenerationRef.current, currentRoomIdRef.current),
    [],
  );

  /**
   * Accept a re-read of a room only if it is still the one on screen and no later read of it
   * has gone out since (#1412).
   */
  const commitRoom = useCallback(
    (ticket: RoomReadTicket, next: RoomState) => {
      if (roomReadIsStale(ticket)) return;
      setRoom(next);
    },
    [roomReadIsStale],
  );

  /**
   * Put on screen the record an act of the user's produced, and retire every read older than it.
   *
   * `commitRoom` asks only whether this is still the conversation on screen. That is the
   * right question for a re-read and the wrong one for an act, because a re-read is in
   * flight at the exact moment the user regains the controls: the stream's landed turn
   * clears the live row -- which is what re-enables Clear and Rewind -- and calls
   * `refreshRoom` in the same breath. A Clear sent in that moment is answered, applied,
   * and then overwritten by the older read, which knows nothing about it, and the
   * conversation the user emptied fills back in.
   *
   * Bumping the generation is what retires those reads: each carries the number it started
   * with, and it is no longer the current one. Until this existed nothing bumped it but
   * opening or leaving a conversation, so the counter tracked *which* room was on screen
   * and never what had been done to it.
   *
   * Two acts racing each other is not a case this orders. The mutating controls are
   * disabled while a turn runs, and where that leaves a gap the later commit wins.
   */
  const commitRoomAct = useCallback((roomId: string, generation: number, next: RoomState) => {
    if (roomResponseIsStale(generation, roomGenerationRef.current, roomId, currentRoomIdRef.current)) {
      return;
    }
    // `generation` is the current one, or the guard above would have returned.
    roomGenerationRef.current = generation + 1;
    setRoom(next);
  }, []);

  /**
   * Close the open conversation, carrying nothing across.
   *
   * Only a delete calls this now. It used to be the way back to the single-agent
   * surface, which #1208 retired; what follows it is the effect below opening the
   * next conversation, once `handleDeleteRoom` has a list fresh enough to pick from.
   */
  const leaveRoom = useCallback(() => {
    roomGenerationRef.current += 1;
    setCurrentRoomId(null);
    setRoom(null);
    setRoomOpenError(null);
    setRoomLive(EMPTY_LIVE);
    setRoomNotice(null);
    setRoomContext(null);
    setParticipantsNotReset([]);
  }, []);

  /**
   * Read the list of conversations. A listing that lands goes on screen and retires the
   * failure of any read before it; one that fails says so, and answers `null`.
   */
  const readListing = useCallback(async (): Promise<RoomListing | null> => {
    try {
      const listing = await roomsApi.list();
      setRooms(listing.rooms);
      setUnreadableRoomIds(listing.unreadable);
      setRoomsListError(null);
      return listing;
    } catch (err) {
      setRoomsListError(`Could not list your conversations: ${roomFailureReason(err)}`);
      return null;
    }
  }, []);

  const fetchRooms = useCallback(async () => {
    try {
      await readListing();
    } finally {
      // Set whether or not the read landed: a failure still answers "this install has
      // no conversation on screen and none is coming", which the auto-open below acts on.
      setRoomsListed(true);
    }
  }, [readListing]);

  const openRoom = useCallback(async (roomId: string) => {
    const generation = (roomGenerationRef.current += 1);
    setCurrentRoomId(roomId);
    setRoom(null);
    setRoomLive(EMPTY_LIVE);
    setRoomNotice(null);
    setRoomOpenError(null);
    setRoomContext(null);
    setParticipantsNotReset([]);
    const ticket = roomReadsRef.current.issue(roomId, generation);
    try {
      const next = await roomsApi.get(roomId);
      if (roomGenerationRef.current !== generation) return;
      // Overtaken only by a read of this same room issued since, which lands in its place.
      if (!roomReadsRef.current.overtaken(ticket)) setRoom(next);
      void readRoomContextRef.current(roomId);
    } catch (err) {
      if (roomGenerationRef.current !== generation) return;
      if (roomReadsRef.current.overtaken(ticket)) return;
      const reason = roomReadReason(err);
      setRoomOpenError(reason);
      setRoomNotice(`Could not open that conversation: ${reason} Pick it again to retry.`);
    }
  }, []);

  /** Re-read the open conversation, dropping the answer if it is no longer open. */
  const refreshRoom = useCallback(
    async (roomId: string) => {
      const ticket = issueRoomRead(roomId);
      try {
        commitRoom(ticket, await roomsApi.get(roomId));
      } catch (err) {
        // A later read of this room is out, and it answers for this one.
        const answeredFor = roomReadIsStale(ticket);
        if (answeredFor) return;
        setRoomNotice(
          `Could not refresh this conversation: ${roomFailureReason(err)} It may be behind what the agents have said.`,
        );
      }
    },
    [commitRoom, issueRoomRead, roomReadIsStale],
  );

  /**
   * Ask how full the open conversation is. Unconditionally, on every occasion that asks.
   *
   * **There is no cadence any more (#1286).** A `contextReadIsDue` predicate used to hold
   * one: at most every four landed turns, and never again once a seat was saturated. That
   * cadence bought one thing -- avoiding `SessionManager.active_turns` loading a seat's
   * persisted session -- and the maintainers' decision in #1286 is that this install never pays
   * it. The agent being talked to is already alive in memory as `live_agent`, so the read
   * is answered from the object the turn just ran on and loads nothing from disk. There is
   * therefore no cost left to sample against, and #1256's objection, which was that cost,
   * no longer applies.
   *
   * What the cadence *did* cost was correctness of the dock's per-seat turn counts, which
   * #1272 (PR #1283) put on screen as numbers. A number a reader watches while they talk
   * is read as current; four turns stale, it is simply wrong, and PR #1283 could only
   * label the staleness rather than remove it. Reading on every landed turn removes it,
   * and the label went with it.
   *
   * So this function no longer decides *whether* to ask -- `refreshRoom` and this are now
   * called together on the same occasions. It still issues the call, drops an answer for a
   * conversation that is no longer open, and re-asks when the open conversation's record
   * moved while the read was out.
   *
   * That last one is the whole of #1256, and it survives the gate's removal because it was
   * never about the gate: the read *was* issued, and then its answer was thrown away with
   * nothing recording that the question had gone unanswered. Nothing else re-asks it, since
   * the turn that prompted it may have been the conversation's last.
   *
   * A failure says nothing on screen, and warns on the console. The banner it feeds is
   * advisory -- compaction runs by itself and does not wait for it -- so a notice here would
   * report a problem the user has no act to take about, on a read they never asked for. The
   * console line is for whoever is looking at why the banner stopped moving.
   */
  const readRoomContext = useCallback(async (roomId: string) => {
    for (let attempt = 0; attempt < CONTEXT_READ_ATTEMPTS; attempt += 1) {
      const generation = roomGenerationRef.current;
      try {
        const next = await roomsApi.context(roomId);
        // The two halves of `roomResponseIsStale` are deliberately *not* asked together
        // here, because they no longer have the same consequence. This is the only caller
        // for which that is true, which is why the helper still stands and still serves
        // `commitRoom` and `commitRoomAct` unchanged.
        //
        // Left the conversation: no surface is waiting for this answer and the state that
        // would hold it has already been cleared. Drop it, and do not ask again.
        if (roomId !== currentRoomIdRef.current) return;
        // Still the conversation on screen, but its record moved while the read was out.
        // The *answer* is stale; the *question* is not -- and nothing else will ask it. The
        // read is issued on a landed turn, and the turn that prompted this one may have been
        // the conversation's last, after which no event arrives at all. So the guard marks
        // the read as owed and re-asks, rather than merely refusing (#1256).
        // Measured before this: 2 of 8 compaction runs ended with `GET /context` answering
        // `is_saturated: true` and no banner on screen.
        if (generation === roomGenerationRef.current) {
          setRoomContext(next);
          return;
        }
      } catch (err) {
        console.warn('Could not read how full this conversation is:', err);
        return;
      }
    }
  }, []);

  readRoomContextRef.current = readRoomContext;

  /** Shorten every seat's context, then re-read what that left. */
  const handleCompactRoom = useCallback(
    async (roomId: string) => {
      setIsRoomCompacting(true);
      try {
        const done = await roomsApi.compact(roomId);
        // Merged here, and only here. The route answers a result per seat and never a
        // merged figure, because each carries the provenance of whichever producer wrote
        // its ledger and one averaged figure would name none of them (`ui/rooms.py`,
        // `compact_room`). What is merged below is the *sentence*, not the record: a
        // one-line notice cannot name four producers, and `done.results` is still per seat
        // for anything that wants them. The seam is at the head because that is where the
        // loss of attribution is paid for, by a reader who asked for one number.
        const saved = done.results.reduce((total, seat) => total + seat.saved_tokens, 0);
        setRoomNotice(
          done.results.length === 0
            ? 'There was no one here to shorten, so nothing changed.'
            : `Shortened ${done.results.length} ${done.results.length === 1 ? 'clone' : 'clones'}, freeing about ${saved} tokens.`,
        );
      } catch (err) {
        setRoomNotice(`Could not shorten this conversation: ${roomFailureReason(err)} Try again in a moment.`);
      } finally {
        setIsRoomCompacting(false);
      }
      await readRoomContext(roomId);
      await refreshRoom(roomId);
    },
    [readRoomContext, refreshRoom],
  );

  /**
   * Apply what a rewind or a clear answered.
   *
   * Both answer the room plus `participants_not_reset`, so both land through here: the
   * transcript is replaced from the answer rather than re-read, and the seats the cut
   * missed are kept for the surface to name.
   */
  const commitHistoryChange = useCallback(
    (roomId: string, answer: RoomHistoryAnswer) => {
      const { participants_not_reset: notReset, ...next } = answer;
      commitRoomAct(roomId, roomGenerationRef.current, next);
      setParticipantsNotReset(notReset);
      void readRoomContext(roomId);
    },
    [commitRoomAct, readRoomContext],
  );

  const handleClearRoomHistory = useCallback(
    async (roomId: string) => {
      try {
        commitHistoryChange(roomId, await roomsApi.clearHistory(roomId));
        await fetchRooms();
      } catch (err) {
        setRoomNotice(`Could not clear this conversation: ${roomFailureReason(err)} Try again in a moment.`);
      }
    },
    [commitHistoryChange, fetchRooms],
  );

  /**
   * Start a conversation.
   *
   * `cloneId` names who it seats. The rail's New omits it and the selected clone is used;
   * an expanded clone row that is in no conversation yet passes its own id, because the
   * selection is state this call would not yet see if the row set it first. Passing one
   * also makes that clone the selection, so the next message goes where the user just
   * looked.
   */
  const handleNewRoom = useCallback(
    async (cloneId?: string, isGroup?: boolean) => {
      // No prompt (F1). The Core refuses a blank title, so the conversation opens under
      // `NEW_CONVERSATION_TITLE` and its first message names it -- see `seededTitle`. A
      // native `window.prompt` used to stand here, gating the first thing a new user does
      // behind a browser modal, and it is also why the rail carried a second New control.
      const opener = cloneId ?? selectedAgent;
      // Taken before the first await, which is the whole point of it: everything below
      // this line is a window in which nothing else may conclude that the head has no
      // conversation and none coming (#1288).
      createInFlightRef.current = true;
      try {
        const created = await roomsApi.create(NEW_CONVERSATION_TITLE, opener ? [opener] : []);
        if (isGroup) {
          setGroupRoomIds((prev) => {
            const next = prev.includes(created.room_id) ? prev : [...prev, created.room_id];
            storeGroupRooms(next);
            return next;
          });
        }
        if (cloneId) setSelectedAgent(cloneId);
        await fetchRooms();
        await openRoom(created.room_id);
      } catch (err) {
        // A Core that refuses to make conversations is not asked again on the way past:
        // the auto-open would either repeat this same notice or, worse, open an older
        // conversation on top of it and leave the user's New looking like it did nothing.
        // Held exactly as a refused auto-open create holds it, and released by the same
        // thing -- a delete.
        autoOpenRef.current = true;
        setRoomNotice(`Could not start a conversation: ${roomFailureReason(err)}`);
      } finally {
        createInFlightRef.current = false;
      }
    },
    [fetchRooms, openRoom, selectedAgent],
  );

  /**
   * "Open in a new conversation" from the Files screen (#1554): a new conversation titled
   * after the story, with the story open in it. A conversation whose story could not be
   * opened is removed again, so a refusal leaves nothing half-made behind; the refusal
   * propagates to the Files screen, which shows it. The Core's note -- that another
   * conversation is still writing the story -- is shown once the conversation is open.
   */
  const handleOpenStoryInNewRoom = useCallback(
    async (storyId: string, title: string) => {
      // The guard New takes, held through an alias so that this release is a line of its
      // own: App.test.tsx declares a kill on New's release by its exact text.
      const guard = createInFlightRef;
      guard.current = true;
      try {
        let created: RoomState;
        try {
          created = await roomsApi.create(title, selectedAgent ? [selectedAgent] : []);
        } catch (err) {
          console.error('Files: a conversation for the story could not be made', err);
          throw new CoreFailure(0, `Could not start a conversation: ${roomFailureReason(err)}`);
        }
        let note: string | null;
        try {
          note = (await artifactLibraryApi.openStory(storyId, created.room_id)).note;
        } catch (err) {
          await roomsApi.delete(created.room_id).catch((cleanup: unknown) => {
            console.error('Files: the unused conversation could not be removed', cleanup);
          });
          await fetchRooms();
          throw err;
        }
        await fetchRooms();
        await openRoom(created.room_id);
        if (note) setRoomNotice(note);
      } finally {
        guard.current = false;
      }
    },
    [fetchRooms, openRoom, selectedAgent],
  );

  /**
   * Rename from the rail. A refusal propagates to the row, which shows its message: the
   * Core's own words, or a plain sentence when the Core gave none (#1411). Only the title
   * is taken from the answer: the open transcript may have moved on since the Core wrote
   * the record it replied with.
   */
  const handleRenameRoom = useCallback(
    async (roomId: string, title: string) => {
      let renamed: RoomState;
      try {
        renamed = await roomsApi.rename(roomId, title);
      } catch (err) {
        throw new Error(roomFailureReason(err));
      }
      setRoom((open) => (open && open.room_id === roomId ? { ...open, title: renamed.title } : open));
      await fetchRooms();
    },
    [fetchRooms],
  );

  /**
   * Delete from the rail's confirmation dialog.
   *
   * The list is re-read whatever the answer, because the likeliest refusal is that the
   * conversation is already gone -- deleted from a second window -- and its row must not
   * outlive it. Deleting the open conversation leaves it the way a fresh load finds the
   * head: nothing open, no notice, nothing of the room still folded in from its topic.
   * A refusal is re-thrown for the dialog to show, as a plain sentence (#1411).
   */
  const handleDeleteRoom = useCallback(
    async (roomId: string) => {
      let refusal: unknown = null;
      try {
        await roomsApi.delete(roomId);
      } catch (err) {
        refusal = err;
      }
      if (refusal === null) {
        setGroupRoomIds((prev) => {
          const next = prev.filter((id) => id !== roomId);
          storeGroupRooms(next);
          return next;
        });
      }
      // Leave a deleted open conversation before the list re-read, not after: `leaveRoom`
      // clears the room notice, and would wipe the re-read's own failure with it.
      if (refusal === null && currentRoomIdRef.current === roomId) leaveRoom();
      const listed = await readListing();
      // Only now may the auto-open act: on a list this read, not on the one that still
      // holds the conversation just deleted. Held when the re-read fails, so the head does
      // not open a conversation off a list known to be stale.
      if (listed !== null) autoOpenRef.current = false;
      // A refused delete of a room the list no longer holds: it is gone all the same. The
      // list was read here, so there is no failure notice for `leaveRoom` to clear.
      const goneAnyway =
        refusal !== null &&
        listed !== null &&
        !listed.rooms.some((r) => r.room_id === roomId) &&
        !listed.unreadable.includes(roomId);
      if (goneAnyway) {
        setGroupRoomIds((prev) => {
          const next = prev.filter((id) => id !== roomId);
          storeGroupRooms(next);
          return next;
        });
      }
      if (goneAnyway && currentRoomIdRef.current === roomId) leaveRoom();
      // A conversation that is gone can never be sent into, so its unsent text goes with
      // it rather than sitting in the map for a room id nothing will ask for again.
      if (refusal === null || goneAnyway) rememberDraft(roomId, '');
      if (refusal !== null) throw new Error(roomFailureReason(refusal));
    },
    [leaveRoom, readListing, rememberDraft],
  );

  /**
   * The last message whose send got no answer, and whose re-read failed too (#1441).
   *
   * Sending the same words again into the same conversation looks first, so a message the
   * Core did store is not stored twice. Anything else sent clears it.
   */
  const unconfirmedSendRef = useRef<{ roomId: string; content: string; afterSeq: number } | null>(
    null,
  );

  /**
   * Send a message, and settle what happened to it when the send is not answered (#1441).
   *
   * Resolves when the message is in the conversation: the Core took it, or a re-read after
   * a failed send found it there. It resolves to the failure in that second case, unless the
   * send only went unanswered: a 5xx after the message was stored (a `/loop` command whose
   * note or schedule failed) is still a failure the reader is told about. Rejects with a
   * `RoomSendError` saying which of the other outcomes it was. There is no idempotency key on `POST /messages`; the re-read is
   * how a missing answer is told apart from a message that never arrived.
   */
  const deliverMessage = useCallback(
    async (roomId: string, content: string, afterSeq: number): Promise<unknown> => {
      const pending = unconfirmedSendRef.current;
      unconfirmedSendRef.current = null;
      if (pending && pending.roomId === roomId && pending.content === content) {
        let reread: RoomState;
        try {
          reread = await roomsApi.get(roomId);
        } catch (err) {
          unconfirmedSendRef.current = pending;
          throw new RoomSendError('unconfirmed', err);
        }
        if (messageArrived(reread, content, pending.afterSeq)) return null;
      }
      try {
        await roomsApi.send(roomId, content);
        return null;
      } catch (sendErr) {
        if (sendWasRefused(sendErr)) throw new RoomSendError('refused', sendErr);
        let reread: RoomState;
        try {
          reread = await roomsApi.get(roomId);
        } catch {
          unconfirmedSendRef.current = { roomId, content, afterSeq };
          throw new RoomSendError('unconfirmed', sendErr);
        }
        if (!messageArrived(reread, content, afterSeq)) throw new RoomSendError('absent', sendErr);
        return sendErr instanceof RoomsNoAnswerError ? null : sendErr;
      }
    },
    [],
  );

  const fetchAllMetadata = useCallback(async () => {
    setIsRefreshing(true);
    try {
      // Skills, ACP status and evaluations are not read here any more: their sections in
      // Settings read their own routes when shown (#1358).
      const [healthRes, agentsRes, personasRes, ontologyRes, budgetRes, modelsRes] =
        await Promise.all([
          fetch('/api/health'),
          fetch('/api/agents'),
          fetch('/api/personas'),
          fetch('/api/ontology'),
          fetch('/api/budget'),
          fetch('/api/models'),
        ]);

      if (healthRes.ok) {
        const hData = await healthRes.json();
        setHealthStatus(hData.status || 'healthy');
        if (hData.git_commit) {
          setGitCommit(hData.git_commit);
        }
      }
      // The first name the server offers, if we are not already on one. The composer
      // lists personas first and falls back to agents, so the name adopted here is the
      // one its dropdown will show as selected.
      let firstNamed: string | null = null;
      if (agentsRes.ok) {
        const aData = await agentsRes.json();
        const loaded: AgentInfo[] = aData.agents || [];
        setAgents(loaded);
        if (loaded.length > 0) firstNamed = loaded[0].id;
      }
      if (personasRes.ok) {
        const pData = await personasRes.json();
        const loaded: PersonaInfo[] = pData.personas || [];
        setPersonas(loaded);
        if (pData.available_tools) setAvailableTools(pData.available_tools);
        if (pData.personas_dir !== undefined) setPersonasDir(pData.personas_dir);
        if (loaded.length > 0) firstNamed = loaded[0].name;
      }
      if (firstNamed !== null) {
        const named = firstNamed;
        setSelectedAgent((current) => current || named);
      }
      if (ontologyRes.ok) {
        const oData = await ontologyRes.json();
        setOntology(oData);
      }
      if (budgetRes.ok) {
        const bData = await budgetRes.json();
        setBudgetData(bData);
      }
      if (modelsRes.ok) {
        const mData = await modelsRes.json();
        setModelConfigured(Boolean(mData.current_model));
        if (mData.models && Array.isArray(mData.models)) {
          setAvailableModels(mData.models);
        }
        if (mData.current_model) {
          setCurrentModel(mData.current_model);
        } else if (mData.models && mData.models.length > 0) {
          setCurrentModel(mData.models[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch UI metadata:', err);
    } finally {
      setIsRefreshing(false);
      setMetadataListed(true);
    }
  }, []);

  /**
   * Sets or clears one agent's session-scoped model override (`model: null` clears it,
   * falling back to `currentModel`). Pure local state: nothing about this choice is
   * meant to persist or to change any other agent's requests (#1138).
   *
   * Nothing reads it today. `POST /api/chat/stream` was the only caller that ever put this
   * override on the wire, and #1208 deletes it; `POST /api/rooms/{room_id}/messages` takes
   * a body of `{content}` only, and the turn is served by the seated persona's own
   * configured model. So the picker this fed is held off the screen rather than left to
   * state a choice nothing honours (`A_ROOM_SEAT_CAN_CARRY_A_MODEL` in `RoomConversation`),
   * and the state below stays wired for the day a seat can carry one.
   *
   * Whether it should is an open product question, not this PR's to settle:
   * 
   */
  const handleSelectAgentModel = useCallback((agentId: string, model: string | null) => {
    setAgentModelOverrides((prev) => {
      if (model === null) {
        if (!(agentId in prev)) return prev;
        const next = { ...prev };
        delete next[agentId];
        return next;
      }
      return { ...prev, [agentId]: model };
    });
  }, []);

  useEffect(() => {
    fetchAllMetadata();
    const interval = setInterval(fetchAllMetadata, 15000);
    return () => clearInterval(interval);
  }, [fetchAllMetadata]);

  useEffect(() => {
    void fetchRooms();
  }, [fetchRooms]);

  /**
   * Open a conversation, because there is no longer a second surface to fall back to.
   *
   * Retiring `PlaygroundTab` (#1208) makes the conversation the centre column
   * unconditionally, so `currentRoomId === null` stops being a state the head can
   * render and becomes one it has to leave: the most recent conversation is reopened
   * -- `rooms[0]`, since `RoomService.list_rooms` orders by `updated_at` for every
   * head at once (#1053) --
   * and an install that has none is given one under `NEW_CONVERSATION_TITLE`, which
   * its first message renames (F1, the same path the rail's New takes).
   *
   * Both lists are waited for. Firing before `/api/rooms` answers would start a second
   * conversation beside the one the server already has; firing before `/api/agents`
   * does would seat it with nobody, since `handleNewRoom` reads `selectedAgent`.
   *
   * The latch is a ref, not a "done once" flag: deleting the open conversation calls
   * `leaveRoom`, which releases it so the next one opens. A create that fails leaves
   * it held, so a Core that refuses is asked once and its notice stands, rather than
   * being retried on every render.
   *
   * `createInFlightRef` is a third guard and not a second reading of the latch: a
   * conversation the user's own click is making is one the head is already getting, and
   * `currentRoomId` does not say so until two awaits after the click (#1288).
   */
  useEffect(() => {
    if (currentRoomId !== null || autoOpenRef.current || createInFlightRef.current) return;
    if (!roomsListed || !metadataListed) return;
    autoOpenRef.current = true;
    const mostRecent = rooms[0];
    void (mostRecent ? openRoom(mostRecent.room_id) : handleNewRoom());
  }, [currentRoomId, roomsListed, metadataListed, rooms, openRoom, handleNewRoom]);

  // Server-Sent Events (SSE) live connection
  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connectStream = () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const es = new EventSource('/api/stream');
      eventSourceRef.current = es;

      es.onopen = () => {
        setIsConnected(true);
      };

      es.onmessage = (e) => {
        try {
          const raw = JSON.parse(e.data);
          const timestamp = raw.timestamp || new Date().toISOString();
          const eventItem: EventEnvelope = {
            id: raw.event_id || `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
            seq: raw.seq,
            type: raw.type || 'UNKNOWN',
            priority: raw.priority,
            event_type: raw.event_type,
            source: raw.source || 'runtime',
            sender_id: raw.sender_id,
            recipient_id: raw.recipient_id,
            target: raw.target || '*',
            topic: raw.topic || 'engine',
            payload: raw.payload,
            timestamp,
            provenance: raw.provenance,
          };

          // A room's events arrive on the stream that is already open -- there is no
          // per-room endpoint. Only the open conversation's topic is folded in; the
          // others are ordinary events in the dock like everything else.
          const topic: string = raw.topic || '';
          const openRoomId = currentRoomIdRef.current;
          if (openRoomId && topic === `room.${openRoomId}` && raw.payload) {
            // The envelope's `event_type` travels with the payload: the topic carries a
            // composing notice and an interrupt beside the replies, and only the type says
            // which one this is. `applyRoomEvent` refuses, out loud, anything it does not
            // name -- this cast is a claim about the Core, checked there.
            const roomEvent = { event_type: raw.event_type, payload: raw.payload } as RoomTopicEvent;
            setRoomLive((live) => applyRoomEvent(live, roomEvent));
            if (roomEvent.event_type === 'AGENT_REPLY' && roomEvent.payload.status === 'final') {
              // Nothing announces that the cascade is over, so the record is re-read on
              // every landed row rather than waited on for an event that never comes.
              // Guarded: these land out of order, and an unguarded `setRoom` could put an
              // older transcript back over a newer one.
              void refreshRoom(openRoomId);
              // How full the seats are changed with this turn, so it is re-read on the same
              // occasion as the record and never sampled behind it (#1286). The seat that
              // took this turn is the `live_agent` still in memory, so the read is answered
              // from that object rather than by loading a session off disk.
              void readRoomContextRef.current(openRoomId);
            }
          }

          if (raw.type === 'HEARTBEAT') {
            setLastHeartbeat(new Date().toLocaleTimeString());
          }

          if (!isPausedRef.current) {
            setEvents((prev) => [eventItem, ...prev.slice(0, 299)]);
          }
        } catch (err) {
          console.error('Failed to parse SSE event data:', err);
        }
      };

      es.onerror = () => {
        setIsConnected(false);
        es.close();
        reconnectTimer = setTimeout(connectStream, 3000);
      };
    };

    connectStream();

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      clearTimeout(reconnectTimer);
    };
    // `refreshRoom` is stable, so naming it here does not reconnect the stream.
  }, [refreshRoom]);

  /**
   * Picking a clone in the rail opens or starts a conversation with that clone.
   *
   * UClone2 parity (Option 2): clicking a clone row opens its most recently updated
   * conversation if it has one, or creates a new conversation seating that clone if
   * none exists yet.
   */
  const handleSelectAgent = useCallback(
    async (agentId: string) => {
      setSelectedAgent(agentId);
      setPreferredClone(agentId);

      const cloneRooms = rooms
        .filter((r) => r.agent_ids.includes(agentId))
        .sort((a, b) => (a.updated_at < b.updated_at ? 1 : a.updated_at > b.updated_at ? -1 : 0));

      if (cloneRooms.length > 0) {
        if (currentRoomId !== cloneRooms[0].room_id) {
          await openRoom(cloneRooms[0].room_id);
        }
      } else {
        await handleNewRoom(agentId);
      }
      if (railIsOverlaid) {
        closeOverlaidRail();
      }
    },
    [closeOverlaidRail, currentRoomId, handleNewRoom, openRoom, railIsOverlaid, rooms],
  );

  /**
   * Inspecting a clone from the rail's inspect button opens the dock on that clone.
   */
  const handleInspectAgent = useCallback(
    (agentId: string) => {
      setSelectedAgent(agentId);
      setPreferredClone(agentId);
      setCloneDockMode('view');
      setActiveDockSurface('clone'); // handleInspectAgent
      setIsDockOpen(true);
    },
    [],
  );

  const handleEditAgent = useCallback(
    (agentId: string) => {
      setSelectedAgent(agentId);
      setCloneDockMode('edit');
      setActiveDockSurface('clone'); // handleEditAgent
      setIsDockOpen(true);
    },
    [],
  );

  const handleNewClone = useCallback(() => {
    setCloneDockMode('create');
    setActiveDockSurface('clone');
    setIsDockOpen(true);
  }, []);

  const handleSavePersona = useCallback(
    async (draft: PersonaDraft, mode: PersonaEditMode): Promise<PersonaSaveResult> => {
      const result = await savePersona(draft, mode, PERSONA_EDITOR_COPY);
      if (result.ok) {
        await fetchAllMetadata();
        setSelectedAgent(result.persona.name);
        setCloneDockMode('view');
      }
      return result;
    },
    [fetchAllMetadata],
  );

  // The step, turn and token readings the dock's Resource surface shows are not derived
  // here (#1272): the surface reads `room` and `roomContext` directly, and says in words
  // where a room reports no such quantity.

  // There is no second dock opener floating over the conversation any more.
  //
  // It was a cyan-glowing circle pinned above the composer, and it opened the dock — which
  // is what the header's Workspace toggle beside it already does, from a place that does
  // not sit on top of the reader's own text. Two controls for one act, one of them
  // overlapping the thing it is next to, and the glow (`shadow-[0_0_20px_rgba(6,182,212,
  // 0.3)]`) is the neon drop-shadow this surface's visual restraint forbids outright.
  //
  // What it carried that the toggle does not is nothing: reopening on the surface the
  // reader last chose rather than resetting to Docs & Artifacts (#1055) is `setIsDockOpen`
  // alone, which is exactly what the header toggle calls.

  return (
    <OpenInDocsContext.Provider value={openInDocs}>
    <div className="h-screen overflow-hidden bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Top Application Header */}
      <Header
        isConnected={isConnected}
        healthStatus={healthStatus}
        gitCommit={gitCommit}
        onRefresh={fetchAllMetadata}
        isRefreshing={isRefreshing}
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={() => chooseSidebarOpen(!isSidebarOpen)}
        isDockOpen={isDockOpen}
        onToggleDock={() => setIsDockOpen(!isDockOpen)}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenFiles={() => setIsFilesOpen(true)}
      />

      {/* Workspace body: rail, conversation, dock. The conversation is not a tab and has no
          condition on it — it is the centre column whenever the app is open. */}
      <div className="flex-1 flex min-h-0 h-full overflow-hidden relative">
        {isSidebarOpen && (
          <WorkspaceSidebar
            agents={agents}
            personas={personas}
            selectedAgent={selectedAgent}
            onSelectAgent={(agentId) => {
              void handleSelectAgent(agentId);
            }}
            onInspectAgent={handleInspectAgent}
            onEditAgent={handleEditAgent}
            onNewClone={handleNewClone}
            // No budget, turn or token props here since #1059: those readings are the dock's
            // Resource surface, one keystroke from U0's default screen rather than on it.
            onClose={() => chooseSidebarOpen(false)}
            overlay={railIsOverlaid}
            rooms={rooms}
            unreadableRoomIds={unreadableRoomIds}
            currentRoomId={currentRoomId}
            onSelectRoom={(roomId) => {
              closeOverlaidRail();
              void openRoom(roomId);
            }}
            // The clone the row belongs to, forwarded. The rail sends it when a clone's
            // expansion starts its first conversation, and dropping it here seated that
            // conversation with whichever clone happened to be *selected* -- which is the
            // exact confusion the argument exists to prevent. Absent (the list's New), the
            // head falls back to its selection.
            onNewRoom={(cloneId, isGroup) => {
              closeOverlaidRail();
              void handleNewRoom(cloneId, isGroup);
            }}
            onRenameRoom={handleRenameRoom}
            onDeleteRoom={handleDeleteRoom}
            modelConfigured={modelConfigured}
            pinnedCloneIds={pinnedClones}
            onTogglePinClone={handleTogglePinClone}
            groupRoomIds={groupRoomIds}
          />
        )}

        {/* Hidden, not unmounted, in Studio mode: the conversation keeps its scroll and state. */}
        <main
          className={`${cloneStudio ? 'hidden' : 'flex'} flex-1 min-h-0 h-full flex-col overflow-hidden p-0`}
        >
          {/* Every room failure the surface can reach: the server's own words when it gave
              some, a plain sentence of ours when it did not (#1411). */}
          {roomNotice ? (
            <div
              data-testid="room-notice"
              className="flex items-start justify-between gap-3 px-4 py-2 border-b border-slate-800/60 text-xs text-slate-300"
            >
              <span>{roomNotice}</span>
              <button
                type="button"
                data-testid="dismiss-room-notice"
                onClick={() => setRoomNotice(null)}
                className="shrink-0 text-slate-500 hover:text-slate-300"
              >
                Dismiss
              </button>
            </div>
          ) : null}
          {roomsListError ? (
            <div
              data-testid="rooms-list-notice"
              className="flex items-start justify-between gap-3 px-4 py-2 border-b border-slate-800/60 text-xs text-slate-300"
            >
              <span>{roomsListError}</span>
              <button
                type="button"
                data-testid="dismiss-rooms-list-notice"
                onClick={() => setRoomsListError(null)}
                className="shrink-0 text-slate-500 hover:text-slate-300"
              >
                Dismiss
              </button>
            </div>
          ) : null}

          {room && currentRoomId ? (
            <RoomConversation
              room={room}
              availableAgents={availableClones}
              live={roomLive}
              // Owned here so it outlives the composer, which `openRoom` unmounts (#1290).
              draft={drafts[currentRoomId] ?? ''}
              onDraftChange={(next) => rememberDraft(currentRoomId, next)}
              onSend={async (content) => {
                // Rethrown on purpose: the composer keeps the draft when this rejects.
                const generation = roomGenerationRef.current;
                // Decided from the record as it stood before this message, which is the
                // only moment "nothing has been said yet" is still true.
                const seed = seededTitle(room, content);
                const storedDespite = await deliverMessage(currentRoomId, content, lastSeq(room));
                if (storedDespite !== null) {
                  setRoomNotice(
                    `Your message is in this conversation. ${roomFailureReason(storedDespite)}`,
                  );
                }
                if (seed) {
                  // After the send, not before: a message the Core refuses must not name
                  // the conversation it was never recorded in.
                  try {
                    await roomsApi.rename(currentRoomId, seed);
                  } catch (err) {
                    setRoomNotice(
                      `Your message was sent, but this conversation could not be named after it: ${roomFailureReason(err)}`,
                    );
                  }
                }
                // Every send, not only the first: the message moved this conversation's
                // `updated_at`, so it now heads the rail and reads "just now" (#1053).
                // `fetchRooms` reports its own failure.
                await fetchRooms();
                // A read like any other, not an act: the send was the act, and the Core had
                // answered it before this went out. Committed as an act, it retired the read
                // the stream's `final` fired and could land after that read, taking the
                // landed reply off screen (#1412).
                // Asked before the ticket: a ticket is a promise of a read, and one never
                // made would overtake a read issued since the act, with nothing in its place.
                if (generation !== roomGenerationRef.current) return;
                const ticket = issueRoomRead(currentRoomId);
                try {
                  commitRoom(ticket, await roomsApi.get(currentRoomId));
                } catch (err) {
                  if (roomReadIsStale(ticket)) return;
                  setRoomNotice(
                    `Your message was sent, but this conversation could not be re-read: ${roomFailureReason(err)} Open it again to see the latest.`,
                  );
                }
              }}
              onStop={() => {
                const generation = roomGenerationRef.current;
                void roomsApi
                  .stop(currentRoomId)
                  .then((next) => commitRoomAct(currentRoomId, generation, next))
                  .catch((err) =>
                    setRoomNotice(
                      `Could not stop this turn: ${roomFailureReason(err)} It may finish on its own.`,
                    ),
                  );
              }}
              onRetry={() => {
                const generation = roomGenerationRef.current;
                void roomsApi
                  .retry(currentRoomId)
                  .then((next) => commitRoomAct(currentRoomId, generation, next))
                  .catch((err) =>
                    setRoomNotice(`Could not run that turn again: ${roomFailureReason(err)} Try again in a moment.`),
                  );
              }}
              onAddAgent={(agentId) => {
                const generation = roomGenerationRef.current;
                void roomsApi
                  .addAgent(currentRoomId, agentId)
                  .then((next) => {
                    commitRoomAct(currentRoomId, generation, next);
                    return fetchRooms();
                  })
                  .catch((err) =>
                    setRoomNotice(
                      `Could not add ${agentId} to this conversation: ${roomFailureReason(err)} Try again in a moment.`,
                    ),
                  );
              }}
              context={roomContext}
              compacting={isRoomCompacting}
              onCompact={() => handleCompactRoom(currentRoomId)}
              onClearHistory={() => handleClearRoomHistory(currentRoomId)}
              participantsNotReset={participantsNotReset}
              availableModels={availableModels}
              currentModel={currentModel}
              // Keyed on the seated clone, not on whatever the rail has selected: the
              // header's picker is a statement about who answers *here*.
              agentModelOverride={seatedCloneId ? agentModelOverrides[seatedCloneId] : undefined}
              onSelectAgentModel={handleSelectAgentModel}
              onToggleAutonomous={(enabled) => {
                const generation = roomGenerationRef.current;
                void roomsApi
                  .setAutonomous(currentRoomId, enabled)
                  .then((next) => commitRoomAct(currentRoomId, generation, next))
                  .catch((err) =>
                    setRoomNotice(`Could not update autonomous discussion: ${roomFailureReason(err)}`),
                  );
              }}
              // `why ›` on a turn. The dock is opened, not merely switched: a control
              // that changed a hidden panel's surface and left the screen unchanged is the
              // invisible selection `handleSelectAgent` was written to stop doing.
              onOpenTurn={(seq) => {
                setSelectedTurnSeq(seq);
                setActiveDockSurface('turn');
                setIsDockOpen(true);
              }}
              onTyping={() => {
                // The one room call whose failure is not surfaced. It reports a timestamp
                // the room uses to hold the floor for `hesitation_seconds`; the user asked
                // for nothing and there is no remedy to offer, so a notice here would be
                // noise on a keystroke. It is logged, and it is the only exception.
                void roomsApi
                  .typing(currentRoomId)
                  .catch((err) => console.warn('Could not report composing:', err));
              }}
            />
          ) : (
            // Not an empty container: either it is still loading or `roomNotice` above
            // has already said why it is not.
            <p data-testid="room-loading" className="px-4 py-3 text-xs text-slate-500">
              {roomNotice
                ? 'This conversation is not on screen.'
                : roomOpenError !== null
                  ? // The banner was dismissed; the failure it reported still stands.
                    `Could not open this conversation: ${roomOpenError} Pick it again to retry.`
                  : currentRoomId
                  ? 'Opening this conversation…'
                  : 'Opening a conversation…'}
            </p>
          )}
        </main>

        {isDockOpen && (
          <ArtifactsDock
            activeSurface={activeDockSurface}
            onSelectSurface={setActiveDockSurface}
            onClose={() => setIsDockOpen(false)}
            // What the rest of the row needs, which only this component knows: the rail is
            // rendered here and is 240px when open. The dock subtracts it to decide whether
            // the row can seat it, rather than guessing from the window (#1030). An overlaid
            // rail is drawn over the row and holds none of it (#1062).
            reservedWidth={isSidebarOpen && !railIsOverlaid ? RAIL_WIDTH_PX : 0}
            developerMode={developerMode}
            // The conversation on screen and one clone seated in it: every dock read is
            // scoped to these, never to a session or the whole workspace (#1356).
            roomId={dockRoomId}
            roomPending={dockRoomPending}
            seatId={dockSeat}
            onSelectSeat={chooseDockSeat}
            selectedArtifactPath={selectedArtifactPath}
            onOpenInDocs={openInDocs}
            events={events}
            ontology={ontology}
            budgetData={budgetData}
            // The conversation on screen, not the retired single-agent history (#1272).
            room={room}
            roomContext={roomContext}
            selectedTurnSeq={selectedTurnSeq}
            isPaused={isPaused}
            onTogglePause={() => setIsPaused(!isPaused)}
            onClearEvents={() => setEvents([])}
            lastHeartbeat={lastHeartbeat}
            // The Clone surface's subject: the rail's pick, and the installed definitions to
            // find it among. `onStartConversation` is `handleNewRoom` with the clone named,
            // the same call the rail's own `Start one` makes -- one conversation, reachable
            // from either place, never two.
            selectedClone={selectedAgent}
            personas={personas}
            onStartConversation={(cloneId) => {
              closeOverlaidRail();
              void handleNewRoom(cloneId);
            }}
            cloneMode={cloneDockMode}
            onCloneModeChange={setCloneDockMode}
            availableTools={availableTools}
            cloneToolsNeedingConversation={
              agents.find((a) => a.id === selectedAgent)?.capabilities_needing_room
            }
            availableModels={availableModels}
            canWritePersonas={personasDir !== null}
            onSavePersona={handleSavePersona}
            cloneStudio={cloneStudio}
            onCloneStudioChange={setCloneStudioRequested}
            onRefresh={fetchAllMetadata}
            isRefreshing={isRefreshing}
          />
        )}

      </div>

      {/* Settings & Hot-Reload Modal */}
      <ArtifactLibrary
        isOpen={isFilesOpen}
        onClose={() => setIsFilesOpen(false)}
        onOpenStory={handleOpenStoryInNewRoom}
      />
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSettingsSaved={() => fetchAllMetadata()}
        developerMode={developerMode}
        onDeveloperModeChange={changeDeveloperMode}
      />
    </div>
    </OpenInDocsContext.Provider>
  );
}

export default App;

