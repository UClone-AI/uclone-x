/**
 * The dock's reads, scoped to the conversation on screen and the seat picked in it.
 *
 * Owner ruling, 2026-09-22 (#1356): the workspace dock describes the room the centre
 * column shows and one agent seated in it -- never the retired single-agent session
 * (its fixed default session id) and never the whole workspace. The Core half is
 * `src/uclone_x/ui/room_dock.py` (#1366); the shapes below are its answers, documented in
 * §3.5 and §3.6 of the room UI design document.
 *
 * **Absence is `null` plus a sentence, never an empty list.** On these routes an empty
 * list means *recorded as empty*; where nothing was recorded the field is `null` and a
 * `reason` says why (P6). The surfaces that render these keep that distinction.
 */
import { createContext, useContext, useEffect, useRef, useState } from 'react';
import type {
  EventEnvelope,
  KnowledgeGraphResponse,
  RoomProvenance,
  RoomSpeakerDecision,
  RoomState,
} from '../types';
import type { ReadFault } from './useApiRead';

/** One tool call a seat made, as the room recorded it after the turn landed. */
export interface RoomToolUse {
  turn_id: string;
  participant_id: string;
  tool_name: string;
  tool_call_id: string | null;
  /** `ToolResultStatus`: success, error, timeout... */
  status: string;
  error: string | null;
  duration_ms: number;
  arguments_preview: string;
  output_preview: string;
  truncated: boolean;
  /** The file the call wrote, read from the tool's output. */
  written_path: string | null;
  /**
   * The call *may* have written without naming a path: a declared writer that did not both
   * succeed and name one (a shell, one that failed partway), or any call that started a
   * helper. It is a possibility the Core counts, never a record that a write happened.
   */
  wrote_unnamed: boolean;
  subagent_id: string | null;
  recorded_at: string;
  /** The transcript `seq` of the turn that made the call. */
  seq: number | null;
}

export interface TurnDocument {
  path: string;
  writes: number;
}

export interface TurnSummary {
  seq: number;
  turn_id: string | null;
  sender_id: string;
  created_at: string;
  completed: boolean;
  error: string | null;
  refusal: string | null;
  provenance: RoomProvenance | null;
  decision: RoomSpeakerDecision | null;
  rendered_through: string | null;
  usage: {
    provider?: string;
    model?: string;
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
    count_source?: string;
  } | null;
  notable: string[];
  steps: RoomToolUse[];
  documents: TurnDocument[];
  unnamed_writes: number;
  subagent_steps: RoomToolUse[];
  steps_absent_reason: 'not_recorded' | 'not_an_agent_turn' | null;
}

export interface SeatTurn {
  seq: number;
  turn_id: string | null;
  created_at: string;
  status: 'answered' | 'failed' | 'interrupted';
  error: string | null;
  content_preview: string;
  /** `null` when the turn's tools were not recorded; `tools_not_recorded_reason` says why. */
  tools: RoomToolUse[] | null;
  tools_not_recorded_reason: string | null;
}

/** `GET /api/rooms/{id}/seats/{participant}/history`. */
export interface SeatHistory {
  room_id: string;
  participant_id: string;
  display_name: string;
  session_id: string;
  live: boolean;
  turns: SeatTurn[];
  tool_uses: RoomToolUse[];
  /**
   * What `tools` covers, said on every read: the calls the seat's saved turns reported, not
   * what a helper it started did. An empty `tools` is "no tool calls listed", never "used
   * no tools".
   */
  tools_note: string;
  /** Turns anywhere in the room that started and were never saved (room-wide count). */
  unsaved_turns: number;
  /** Why this seat's list may be missing a turn; `null` when no turn was lost. */
  unsaved_note: string | null;
  /** Why `turns` is empty, from the room's record; `null` when it is not. */
  reason: string | null;
}

export interface RoomArtifact {
  id: string;
  path: string;
  name: string;
  title: string;
  type: 'image' | 'document' | 'file';
  participant_id: string;
  tool_name: string;
  turn_id: string;
  tool_call_id: string | null;
  written_at: string;
  write_count: number;
  writers: string[];
  /** `null` when the disk did not answer; `exists_reason` says why. */
  exists: boolean | null;
  exists_reason: string | null;
  size_bytes: number | null;
}

/**
 * `GET /api/rooms/{id}/artifacts`.
 *
 * **It never says that nothing was written** (#1366): a shell, an MCP server or a helper
 * can write without the room seeing it. `scope_note` says what the list covers on every
 * read, and `record_gaps` names each *known* reason it may be missing a write. There is no
 * completeness flag: an empty `record_gaps` is no known gap, never a complete list.
 */
export interface RoomArtifacts {
  room_id: string;
  artifacts: RoomArtifact[];
  total: number;
  unattributed_writes: number;
  unattributed_note: string | null;
  unrecorded_turns: number;
  unrecorded_note: string | null;
  unsaved_turns: number;
  /** Each known gap as a clause, e.g. "its history was cleared". */
  record_gaps: string[];
  scope_note: string;
  /** A turn is running; the files it saves are listed when it finishes. */
  turn_running: boolean;
  /** Why the list is empty, with the scope and every known gap; `null` when it is not. */
  reason: string | null;
}

export type RoomTopologyNode =
  | {
      id: string;
      kind: 'seat';
      participant_id: string;
      display_name: string;
      session_id: string | null;
      status: 'idle' | 'answered' | 'left';
      turn_count: number;
      live: boolean;
    }
  | {
      id: string;
      kind: 'turn';
      participant_id: string;
      seq: number;
      turn_id: string | null;
      status: 'answered' | 'failed' | 'interrupted';
      tools_recorded: boolean;
    }
  | {
      id: string;
      kind: 'tool';
      participant_id: string;
      tool_name: string;
      tool_call_id: string | null;
      status: string;
      turn_id: string;
    }
  | {
      id: string;
      kind: 'subagent';
      subagent_id: string;
      parent_participant_id: string;
    };

export interface RoomTopologyEdge {
  id: string;
  source: string;
  target: string;
  kind: 'took_turn' | 'followed_by' | 'called' | 'spawned';
}

/** `GET /api/rooms/{id}/topology`. */
export interface RoomTopology {
  room_id: string;
  nodes: RoomTopologyNode[];
  edges: RoomTopologyEdge[];
  /**
   * True when no turn *row* can have been removed or lost. It says nothing about tool
   * completeness: a turn's tool calls can still be unrecorded or made by a helper (#1388
   * N3), so `summary.tool_calls` is the calls listed, never all calls made.
   */
  history_complete: boolean;
  /** Each known reason turns may be missing from the graph, as a clause. */
  history_gaps: string[];
  /**
   * Each known reason a turn in the graph may show fewer tool calls than it made, as a
   * clause (#1388 N3). Independent of `history_complete`: every turn can be present while
   * some of their calls are not.
   */
  tool_call_gaps: string[];
  summary: { seats: number; turns: number; tool_calls: number; subagents: number };
  reason: string | null;
}

export interface RememberedStatement {
  statement: string;
  subject: string;
  relation: string;
  object: string;
  /** The conversation it was learned in; `null` is "not recorded", not "nowhere". */
  source_session_id: string | null;
  confidence: number | null;
  /** True when the clone worked it out from other things it knew, rather than was told. */
  learned: boolean;
}

/** A fact the clone saved to its own memory (#1401). */
export interface SavedFact {
  statement: string;
  subject: string;
  relation: string;
  object: string;
  /** The conversation it was saved in; `null` is "not recorded", not "nowhere". */
  source_session_id: string | null;
  /** True when it was saved in the conversation on screen. */
  saved_here: boolean;
  confidence: number | null;
}

/**
 * `GET /api/rooms/{id}/knowledge?agent_id=<seat>`. Only `ok` carries lists. A seat that is
 * not running is read from its saved knowledge record (#1367): `not_recorded` when there is
 * none, `unreadable` when a record is there and could not be read. The read leaves that
 * record as it is. `reason` is
 * plain copy: where the record is and why it failed are in the log, not here.
 */
export type SeatKnowledge = {
  room_id: string;
  participant_id: string;
  session_id: string;
  status: 'ok' | 'not_recorded' | 'unreadable' | 'no_ontology';
  reason: string | null;
  remembers: RememberedStatement[] | null;
  /**
   * The facts this clone saved to its own memory, on every status (#1401): they belong to
   * the clone, not to this conversation's record. `null` when they could not be read;
   * `saved_facts_reason` says so, and says when none are listed.
   */
  saved_facts: SavedFact[] | null;
  saved_facts_reason: string | null;
} & (
  | ({ status: 'ok' } & KnowledgeGraphResponse)
  | {
      status: 'not_recorded' | 'unreadable' | 'no_ontology';
      triples: null;
      nodes: null;
      edges: null;
      summary: null;
    }
);

const enc = encodeURIComponent;

export const roomDockUrls = {
  seatHistory: (roomId: string, seatId: string): string =>
    `/api/rooms/${enc(roomId)}/seats/${enc(seatId)}/history`,
  turn: (roomId: string, seq: number): string =>
    `/api/rooms/${enc(roomId)}/turns/${seq}`,
  artifacts: (roomId: string): string => `/api/rooms/${enc(roomId)}/artifacts`,
  topology: (roomId: string): string => `/api/rooms/${enc(roomId)}/topology`,
  knowledge: (roomId: string, seatId: string): string =>
    `/api/rooms/${enc(roomId)}/knowledge?agent_id=${enc(seatId)}`,
};

/** A failed read: `message` is technical, `fault` is what a U0 surface may say. */
class RoomReadError extends Error {
  constructor(
    message: string,
    readonly fault: ReadFault,
  ) {
    super(message);
  }
}

/** The Core's refusal, in its own words where it gave any (404 names the roster). */
async function readOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`.trim();
    let detail: string | null = null;
    try {
      const body = await res.json();
      if (typeof body?.detail === 'string' && body.detail.trim() !== '') {
        detail = body.detail as string;
        message = body.detail as string;
      }
    } catch {
      // No JSON body: the status line is still a reason, for a developer.
    }
    throw new RoomReadError(message, { kind: 'status', detail });
  }
  try {
    return (await res.json()) as T;
  } catch (err) {
    throw new RoomReadError(err instanceof Error ? err.message : String(err), {
      kind: 'unreadable',
      detail: null,
    });
  }
}

export interface RoomRead<T> {
  data: T | null;
  /** Why the read failed, in technical words; for developer surfaces. */
  error: string | null;
  /**
   * The same failure, classified, with the Core's own plain `detail` where it gave one.
   * A surface every user sees says `fault.detail` or a fixed sentence of its own, never
   * `error` (transport text such as "Failed to fetch").
   */
  fault: ReadFault | null;
  loading: boolean;
}

/**
 * Read `url`, and read it again whenever `refreshKey` changes.
 *
 * `url === null` means there is nothing to scope the read to (no conversation open, or no
 * seat), and no request is made. An answer that arrives after the scope moved on is
 * dropped, so switching conversations never shows the previous one's record.
 */
export function useRoomRead<T>(url: string | null, refreshKey: unknown = 0): RoomRead<T> {
  // Each answer remembers the address it answered, so a read for a scope the dock has
  // since left is never shown as the current one's.
  const [state, setState] = useState<RoomRead<T> & { url: string | null }>({
    url: null,
    data: null,
    error: null,
    fault: null,
    loading: false,
  });

  useEffect(() => {
    if (url === null) return;
    let current = true;
    // The same scope re-reading keeps what is on screen; a new scope starts blank.
    setState((prev) =>
      prev.url === url
        ? { ...prev, error: null, fault: null, loading: true }
        : { url, data: null, error: null, fault: null, loading: true },
    );
    (async () => {
      try {
        const data = await readOrThrow<T>(await fetch(url));
        if (current) setState({ url, data, error: null, fault: null, loading: false });
      } catch (err) {
        if (current) {
          setState({
            url,
            data: null,
            error: err instanceof Error ? err.message : String(err),
            // `fetch` itself rejecting is the only failure without a fault of its own.
            fault: err instanceof RoomReadError ? err.fault : { kind: 'unreachable', detail: null },
            loading: false,
          });
        }
      }
    })();
    return () => {
      current = false;
    };
  }, [url, refreshKey]);

  if (url === null) return { data: null, error: null, fault: null, loading: false };
  if (state.url !== url) return { data: null, error: null, fault: null, loading: true };
  return { data: state.data, error: state.error, fault: state.fault, loading: state.loading };
}

/**
 * Read a turn's summary, and refetch when a tool event for this turn or a final reply lands.
 *
 * Relies on the existing SSE stream; there is no timer (G6, P1).
 */
export function useTurnSummary(
  roomId: string | null,
  seq: number | null,
  turnId?: string | null,
  events: EventEnvelope[] = [],
): RoomRead<TurnSummary> {
  const url = roomId && seq !== null ? roomDockUrls.turn(roomId, seq) : null;
  const [refreshKey, setRefreshKey] = useState(0);
  const seenEventIds = useRef(new Set<string>());
  const summaryRead = useRoomRead<TurnSummary>(url, refreshKey);
  const effectiveTurnId = turnId ?? summaryRead.data?.turn_id ?? null;

  useEffect(() => {
    if (!roomId || (seq === null && !effectiveTurnId)) return;
    let bumped = false;
    for (const ev of events) {
      const id =
        ev.id ||
        `${ev.topic || ''}:${ev.timestamp || ''}:${JSON.stringify(ev.payload || {})}`;
      if (seenEventIds.current.has(id)) continue;
      seenEventIds.current.add(id);

      // 1) Matching room.{id}.tool event for this turn_id
      if (effectiveTurnId && isRoomToolEvent(ev, roomId)) {
        const p = ev.payload as { turn_id?: string } | undefined;
        if (p?.turn_id === effectiveTurnId) {
          bumped = true;
        }
      }

      // 2) AGENT_REPLY final event for this seq
      const evType = ev.event_type ?? ev.type;
      if (evType === 'AGENT_REPLY') {
        const p = ev.payload as
          | { status?: string; seq?: number; turn_id?: string }
          | undefined;
        if (p?.status === 'final') {
          if (
            (seq !== null && p.seq === seq) ||
            (effectiveTurnId && p.turn_id === effectiveTurnId)
          ) {
            bumped = true;
          }
        }
      }
    }
    if (bumped) {
      setRefreshKey((k) => k + 1);
    }
  }, [events, roomId, seq, effectiveTurnId]);

  return summaryRead;
}

/** The agents seated in `room`, in roster order. */
export const agentSeats = (room: RoomState | null | undefined) =>
  (room?.participants ?? []).filter((p) => p.kind === 'agent');

/**
 * The seat the dock describes when the reader has not picked one.
 *
 * In order: the clone the reader last picked or inspected in the rail, if it is seated
 * here; else the agent that spoke last; else the first agent seated. `null` only when no
 * agent is seated, which the surfaces say in words.
 */
export function defaultSeat(
  room: RoomState | null | undefined,
  preferredClone: string | null | undefined,
): string | null {
  const seats = agentSeats(room);
  if (seats.length === 0) return null;
  if (preferredClone && seats.some((s) => s.id === preferredClone)) return preferredClone;
  const seated = new Set(seats.map((s) => s.id));
  const transcript = room?.transcript ?? [];
  for (let i = transcript.length - 1; i >= 0; i -= 1) {
    const m = transcript[i];
    if (m.kind === 'utterance' && seated.has(m.sender_id)) return m.sender_id;
  }
  return seats[0].id;
}

/**
 * Whether an SSE envelope is one of `roomId`'s tool announcements (§3.6).
 *
 * The server writes the event type in upper case (`TOOL_CALL`, `TOOL_RESULT`); the
 * comparison ignores case so a head never again drops every row over a spelling (#1353).
 */
export function isRoomToolEvent(
  ev: { event_type?: string; type?: string; topic?: string },
  roomId: string,
): boolean {
  if (ev.topic !== `room.${roomId}.tool`) return false;
  const kind = String(ev.event_type ?? ev.type ?? '').toUpperCase();
  return kind === 'TOOL_CALL' || kind === 'TOOL_RESULT';
}

/**
 * Front the dock's Docs surface on a file (#1354).
 *
 * Provided once by the page that owns the dock; read by whatever draws a file in the
 * conversation (an inline generated image, a media card) so "open in Docs" works without
 * threading a handler through every message row. `null` where there is no dock, and the
 * button is then not drawn.
 */
export const OpenInDocsContext = createContext<((path: string) => void) | null>(null);
export const useOpenInDocs = () => useContext(OpenInDocsContext);
