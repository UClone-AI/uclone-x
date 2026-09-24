/**
 * What `App` itself decides, now that the conversation is the only centre column (#1208).
 *
 * `PlaygroundTab` and the `currentRoomId ? … : …` fork are gone, and with them every case
 * here that drove a composer, a stream or a history control through `App`. What is left is
 * the head's own wiring, which no component test can reach:
 *
 * - which conversation is open on a load, and where one comes from when there is none;
 * - which agent the page adopts when the server names one, and what it asks for before it
 *   has a name (#1125);
 * - the dock opener, which the conversation renders but `App` owns (#1055);
 * - deleting the open conversation (#1058);
 * - whose model override reaches the conversation header (#1138);
 * - what picking a clone in the rail puts on screen (#1300);
 * - which conversation and seat the dock reads, and "Open in Docs" (#1356, #1354).
 *
 * What the conversation does with any of it is pinned in
 * `components/rooms/RoomConversation.test.tsx` and in the room e2e modules.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { App } from './App';
import { NEW_CONVERSATION_TITLE } from './lib/rooms';
import { DEVELOPER_MODE_KEY } from './lib/developerMode';
import { expectPlain } from './test/plainCopy';

/** A JSON response the page can read, without a body stream to drain. */
const answer = (body: unknown, status = 200) =>
  ({
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ 'content-type': 'application/json' }),
    json: () => Promise.resolve(body),
  }) as unknown as Response;

/**
 * A request that failed without the Core saying why: the fetch itself rejected (the Core is
 * not answering), or a 500 came back with no JSON body.
 */
const faulted = (fault: 'no-answer' | 'bare-500'): Promise<Response> =>
  fault === 'no-answer'
    ? Promise.reject(new TypeError('Failed to fetch'))
    : Promise.resolve({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        headers: new Headers({ 'content-type': 'text/plain' }),
        json: () => Promise.reject(new SyntaxError('Unexpected token I in JSON')),
      } as unknown as Response);

/** Let every continuation a delivery queued run, then commit React's updates. */
const settle = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));

class SilentEventSource {
  /**
   * The instance `App` is currently listening on.
   *
   * Kept so a case can deliver a room event the way the field does -- through the stream
   * the page opened for itself -- rather than reaching for the handler some other way.
   * `App` re-opens the stream on a reconnect, so the newest instance is the live one.
   */
  static current: SilentEventSource | null = null;
  onopen: (() => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) {
    SilentEventSource.current = this;
  }
  addEventListener() {}
  removeEventListener() {}
  close() {}
}

const summaryOf = (roomId: string, title: string) => ({
  room_id: roomId,
  title,
  agent_ids: ['champion'],
  human_ids: ['user'],
  message_count: 2,
  updated_at: '2026-09-19T00:00:00Z',
});

const stateOf = (roomId: string, title: string) => ({
  room_id: roomId,
  title,
  participants: [
    { id: 'user', kind: 'human' },
    ...(seatsOnServer.get(roomId) ?? ['champion']).map((id) => ({ id, kind: 'agent' })),
  ],
  transcript: transcriptsOnServer.get(roomId) ?? [],
  turn_state: { agent_turns_since_human: 0 },
  policy: {
    max_agent_turns_per_human_message: 3,
    max_span_messages: 40,
    transcript_window: 15,
    hesitation_seconds: 0,
    default_responder_id: '',
  },
});

const missing = (roomId: string) =>
  `No room '${roomId}' in the store: it has been deleted, or was never created. List the rooms to see the ones that exist.`;

/** The conversations the Core holds, by id and title, most recently updated first. */
let roomsOnServer: Map<string, string>;
/** Ids of records the Core has and will not load, which it lists apart (#1440). */
let unreadableOnServer: Set<string>;
/**
 * How `POST /api/rooms/{id}/messages` goes wrong, when it does (#1441): the message is
 * stored and the answer never arrives, it is stored and a bare 500 comes back (a `/loop`
 * command whose note or schedule failed after `accept`), it never arrives at all, or the Core
 * refuses it.
 */
let sendFault:
  | 'stored-no-answer'
  | 'stored-then-500'
  | 'lost'
  | { status: number; detail: string }
  | null;
/** The agents seated in a conversation, by room id; `champion` alone where unnamed. */
let seatsOnServer: Map<string, string[]>;
/**
 * How `GET /api/rooms/{id}` fails for a room, when it fails without the Core saying why:
 * the fetch itself rejects (the Core is not answering), or a 500 comes back with no JSON body.
 */
let roomReadFault: Map<string, 'no-answer' | 'bare-500'>;
/**
 * The same two faults for any route, keyed by `METHOD path` (#1411): the notices of every
 * room act must be plain whichever request failed and however it failed.
 */
let routeFault: Map<string, 'no-answer' | 'bare-500'>;
/** A conversation's rows, by room id; empty where unnamed. */
let transcriptsOnServer: Map<string, unknown[]>;
let agentsOnServer: { id: string; name: string }[];
let personasOnServer: { name: string }[];
let modelsOnServer: string[];
let currentModelOnServer: string;
/** What `POST /api/rooms` refuses with, when it refuses. */
let createRefusal: string | null;
/**
 * Whether `GET /api/rooms/{id}/context` reports the conversation full.
 *
 * A single flag rather than a fixture, because what #1256 is about is *when* the head
 * asks and what it does with the answer, not the shape of the answer. The seat is named
 * `champion` so the banner resolves it against the room's own roster.
 */
let saturatedOnServer: boolean;
/**
 * How many turns `GET /api/rooms/{id}/context` says the `champion` seat is holding.
 *
 * `null` leaves the answer seatless, which is what every case predating #1286 wants. A
 * number serves one `champion` seat holding it, saturated at or above the threshold of 40,
 * so a case can move the figure between turns and watch the screen follow it.
 */
let seatTurnsOnServer: number | null;
let requests: { url: string; method: string; body?: string }[];
/** What `GET /api/rooms/{id}/seats/{seat}/history` reports, beyond its identity. */
let seatHistoryOnServer: { turns: unknown[]; tool_uses: unknown[] };
let created: number;

/**
 * Answers withheld until the test releases them, keyed by `METHOD path`.
 *
 * The auto-open waits on two lists it did not order; holding one back is the only way to
 * observe that it waits, rather than to hope the unheld ordering is the one that runs.
 */
let gates: Map<string, Promise<void>>;
let openers: Map<string, () => void>;

const hold = (key: string) => {
  gates.set(
    key,
    new Promise<void>((resolve) => openers.set(key, resolve)),
  );
};

const release = async (key: string) => {
  gates.delete(key);
  openers.get(key)!();
  await settle();
};

/**
 * Turn developer mode on the way a user does: Settings, the switch, close. Off is the default
 * (owner ruling 2026-09-22), so a case about a developer surface starts here.
 */
const turnOnDeveloperMode = async () => {
  fireEvent.click(screen.getByTestId('open-settings'));
  const toggle = await screen.findByRole('switch', { name: /developer mode/i });
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute('aria-checked', 'true');
  fireEvent.click(screen.getByTitle('Close'));
  await settle();
};

const sent = (method: string, path: string) =>
  requests.filter((r) => r.method === method && r.url.split('?')[0] === path);

beforeEach(() => {
  // jsdom keeps localStorage for the whole file: a case that turned developer mode on must not
  // hand it to the next one, which would then test a state no fresh install opens in.
  window.localStorage.removeItem(DEVELOPER_MODE_KEY);
  window.localStorage.removeItem('uclone-x.rail.open');
  roomsOnServer = new Map([
    ['r1', 'Index tuning'],
    ['r2', 'Release notes'],
  ]);
  unreadableOnServer = new Set();
  sendFault = null;
  seatsOnServer = new Map();
  roomReadFault = new Map();
  routeFault = new Map();
  transcriptsOnServer = new Map();
  agentsOnServer = [{ id: 'champion', name: 'champion' }];
  personasOnServer = [];
  modelsOnServer = [];
  currentModelOnServer = '';
  createRefusal = null;
  saturatedOnServer = false;
  seatTurnsOnServer = null;
  requests = [];
  seatHistoryOnServer = { turns: [], tool_uses: [] };
  created = 0;
  gates = new Map();
  openers = new Map();
  vi.stubGlobal('EventSource', SilentEventSource);
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const path = url.split('?')[0];
      const method = (init?.method ?? 'GET').toUpperCase();
      requests.push({ url, method, body: typeof init?.body === 'string' ? init.body : undefined });
      const gated = gates.get(`${method} ${path}`);
      const serve = (): Promise<Response> => {
        const routeFailure = routeFault.get(`${method} ${path}`);
        if (routeFailure) return faulted(routeFailure);
        const seatHistory = /^\/api\/rooms\/([^/]+)\/seats\/([^/]+)\/history$/.exec(path);
        if (seatHistory) {
          return Promise.resolve(
            answer({
              room_id: seatHistory[1],
              participant_id: seatHistory[2],
              display_name: seatHistory[2],
              session_id: `room-${seatHistory[1]}-${seatHistory[2]}`,
              live: false,
              reason: null,
              ...seatHistoryOnServer,
            }),
          );
        }
        const roomArtifacts = /^\/api\/rooms\/([^/]+)\/artifacts$/.exec(path);
        if (roomArtifacts) {
          return Promise.resolve(
            answer({
              room_id: roomArtifacts[1],
              artifacts: [],
              total: 0,
              unattributed_writes: 0,
              unattributed_note: null,
              reason: 'Nobody has written a file in this conversation yet.',
            }),
          );
        }
        if (path === '/api/artifacts/content') {
          return Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('# Plan') } as unknown as Response);
        }
        if (path === '/api/rooms' && method === 'GET') {
          return Promise.resolve(
            answer({
              rooms: [...roomsOnServer].map(([id, title]) => summaryOf(id, title)),
              unreadable: [...unreadableOnServer],
            }),
          );
        }
        if (path === '/api/rooms' && method === 'POST') {
          if (createRefusal !== null) return Promise.resolve(answer({ detail: createRefusal }, 500));
          created += 1;
          const roomId = `new-${created}`;
          roomsOnServer.set(roomId, NEW_CONVERSATION_TITLE);
          return Promise.resolve(answer(stateOf(roomId, NEW_CONVERSATION_TITLE), 201));
        }
        const context = /^\/api\/rooms\/([^/]+)\/context$/.exec(path);
        if (context) {
          // `saturatedOnServer` is the old single flag, kept as the 40-turn case of the
          // figure so #1256's cases read exactly as they did. `seatTurnsOnServer` is the
          // same answer with the number under the caller's control (#1286).
          const seatTurns = seatTurnsOnServer ?? (saturatedOnServer ? 40 : null);
          return Promise.resolve(
            answer({
              room_id: context[1],
              seats:
                seatTurns === null
                  ? []
                  : [
                      {
                        participant_id: 'champion',
                        session_id: 'champion-session',
                        active_turns: seatTurns,
                        is_saturated: seatTurns >= 40,
                        used_tokens: 12345,
                        live: true,
                      },
                    ],
              is_saturated: seatTurns !== null && seatTurns >= 40,
              saturation_threshold: 40,
            }),
          );
        }
        const messages = /^\/api\/rooms\/([^/]+)\/messages$/.exec(path);
        if (messages && method === 'POST') {
          if (sendFault === 'lost') return Promise.reject(new TypeError('Failed to fetch'));
          if (sendFault === 'stored-no-answer' || sendFault === 'stored-then-500') {
            const rows = transcriptsOnServer.get(messages[1]) ?? [];
            transcriptsOnServer.set(messages[1], [
              ...rows,
              {
                seq: rows.length + 1,
                sender_id: 'user',
                content: JSON.parse(String(init?.body)).content,
                kind: 'utterance',
                created_at: '2026-09-19T00:00:00Z',
                completed: true,
              },
            ]);
            return faulted(sendFault === 'stored-no-answer' ? 'no-answer' : 'bare-500');
          }
          if (sendFault !== null) {
            return Promise.resolve(answer({ detail: sendFault.detail }, sendFault.status));
          }
          return Promise.resolve(answer({ seq: 1 }));
        }
        // The composer announces itself as the user types. Served so a case driving the
        // composer does not print a 404 the page correctly shrugs off.
        if (/^\/api\/rooms\/([^/]+)\/typing$/.test(path) && method === 'POST') {
          return Promise.resolve(answer(null, 204));
        }
        const cleared = /^\/api\/rooms\/([^/]+)\/history$/.exec(path);
        if (cleared && method === 'DELETE') {
          transcriptsOnServer.set(cleared[1], []);
          return Promise.resolve(
            answer({ ...stateOf(cleared[1], roomsOnServer.get(cleared[1]) ?? ''), participants_not_reset: [] }),
          );
        }
        const seat = /^\/api\/rooms\/([^/]+)\/participants$/.exec(path);
        if (seat && method === 'POST') {
          const agentId = JSON.parse(String(init?.body)).agent_id as string;
          seatsOnServer.set(seat[1], [...(seatsOnServer.get(seat[1]) ?? ['champion']), agentId]);
          return Promise.resolve(answer(stateOf(seat[1], roomsOnServer.get(seat[1]) ?? '')));
        }
        const compact = /^\/api\/rooms\/([^/]+)\/compact$/.exec(path);
        if (compact && method === 'POST') {
          return Promise.resolve(answer({ room_id: compact[1], results: [] }));
        }
        const room = /^\/api\/rooms\/([^/]+)$/.exec(path);
        if (room) {
          const fault = method === 'GET' ? roomReadFault.get(room[1]) : undefined;
          if (fault) return faulted(fault);
          if (unreadableOnServer.has(room[1]) && method === 'DELETE') {
            unreadableOnServer.delete(room[1]);
            return Promise.resolve(answer(null, 204));
          }
          const title = roomsOnServer.get(room[1]);
          if (title === undefined) return Promise.resolve(answer({ detail: missing(room[1]) }, 404));
          if (method === 'DELETE') {
            roomsOnServer.delete(room[1]);
            return Promise.resolve(answer(null, 204));
          }
          return Promise.resolve(answer(stateOf(room[1], title)));
        }
        switch (path) {
          case '/api/agents':
            return Promise.resolve(answer({ agents: agentsOnServer }));
          case '/api/personas':
            return Promise.resolve(answer({ personas: personasOnServer }));
          case '/api/models':
            return Promise.resolve(
              answer({ current_model: currentModelOnServer, models: modelsOnServer }),
            );
          default:
            return Promise.resolve(answer({}, 404));
        }
      };
      return gated ? gated.then(serve) : serve();
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/**
 * The conversation `App` opened by itself, once it is on screen.
 *
 * Every test waits on this rather than clicking a row: since #1208 there is no load that
 * does not end in a conversation, so arriving at one is the behaviour, not the setup.
 */
const openedConversation = async () => {
  await screen.findByTestId('room-conversation');
};

/**
 * One landed reply on the open conversation's topic, delivered through the page's stream.
 *
 * `AGENT_REPLY` with `status: 'final'` is the only event the head treats as a turn having
 * landed, and it is what drives both `refreshRoom` and the seat read. Published through
 * `es.onmessage` rather than by calling a handler directly, so the envelope goes through
 * the same parse and the same topic match the Core's events do.
 */
const landTurn = async (roomId: string, seq: number) => {
  const stream = SilentEventSource.current;
  if (stream?.onmessage == null) throw new Error('the page has not opened its stream yet');
  const onmessage = stream.onmessage;
  await act(async () => {
    onmessage({
      data: JSON.stringify({
        event_id: `turn-${seq}`,
        seq,
        type: 'AGENT_EVENT',
        topic: `room.${roomId}`,
        event_type: 'AGENT_REPLY',
        source: 'runtime',
        payload: {
          room_id: roomId,
          agent_id: 'champion',
          turn_id: `t-${seq}`,
          seq,
          status: 'final',
          content: `reply ${seq}`,
          completed: true,
        },
      }),
    } as MessageEvent);
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  });
};

/**
 * The seat counts are read on every landed turn, with nothing sampling between them (#1286).
 *
 * `contextReadIsDue` used to gate this read: at most once every four landed turns, and never
 * again once a seat was saturated. The cadence bought one thing, avoiding a Core session
 * load per seat, and the maintainers' decision in #1286 is that this install never pays it — the
 * seat that just answered is alive in memory as `live_agent`, so the read is answered from
 * that object and loads nothing from disk. #1256's cost objection was that load, so with the
 * load gone the objection goes with it.
 *
 * What the cadence cost was the dock's per-seat counts, which #1272 (PR #1283) put on screen
 * as numbers a reader watches while they talk. PR #1283 could only label them — "counted up
 * to 4 turns ago" — and that label is gone with the staleness it described.
 */
describe('App reads the seats on every landed turn (#1286)', () => {
  // Deleted rather than re-gated: the predicate the mutation would restore no longer
  // exists, and removing the call is the cadence taken to its limit — never re-read on a
  // landed turn at all, which is the state every count on screen would then be frozen in.
  // Killed by: frontend/src/App.tsx :: void readRoomContextRef.current(openRoomId);
  // Becomes:
  it('asks once per landed turn, and not once per several', async () => {
    seatTurnsOnServer = 3;
    render(<App />);
    await openedConversation();

    // `openRoom` reads the context itself, so that one is the baseline.
    expect(sent('GET', '/api/rooms/r1/context')).toHaveLength(1);

    await landTurn('r1', 3);
    expect(sent('GET', '/api/rooms/r1/context')).toHaveLength(2);
    await landTurn('r1', 5);
    expect(sent('GET', '/api/rooms/r1/context')).toHaveLength(3);
    await landTurn('r1', 7);
    // Three turns, three reads. Under the four-turn cadence this was still 1.
    expect(sent('GET', '/api/rooms/r1/context')).toHaveLength(4);
  });

  // The other half of the old rule, and the one a reader actually sees: `contextReadIsDue`
  // returned `false` outright once `is_saturated` was true, so the count froze at whatever
  // turn crossed the threshold and never moved again. The mutation restores exactly that by
  // refusing to commit a saturated answer.
  // Killed by: frontend/src/App.tsx :: setRoomContext(next);
  // Becomes: if (!next.is_saturated) setRoomContext(next);
  it('keeps counting past the threshold, where the cadence used to stop for good', async () => {
    seatTurnsOnServer = 38;
    render(<App />);
    await openedConversation();

    // The figure is put on screen by a landed turn rather than by the open. `openRoom`
    // clears `roomContext`, sets `currentRoomId` and only then reads, and here the read is
    // answered before React has committed that open -- so `readRoomContext` finds
    // `currentRoomIdRef` still empty and drops its own answer, exactly as it should for an
    // answer about a conversation nobody is looking at. In the field the round trip is far
    // longer than the commit; in this environment the mock answers within the same
    // microtask queue. Either way the turn below is what this case is about.
    await landTurn('r1', 38);
    await waitFor(() =>
      expect(screen.getByTestId('answering-context-count')).toHaveTextContent('38/40 turns'),
    );

    // The turn that saturates the seat. The banner goes up here either way; the count is
    // what the old rule stopped maintaining.
    seatTurnsOnServer = 40;
    await landTurn('r1', 40);
    await waitFor(() =>
      expect(screen.getByTestId('answering-context-count')).toHaveTextContent('40/40 turns'),
    );

    // And the turn after it, which the old rule never read at all.
    seatTurnsOnServer = 41;
    await landTurn('r1', 41);
    await waitFor(() =>
      expect(screen.getByTestId('answering-context-count')).toHaveTextContent('41/40 turns'),
    );
  });
});

describe('App opens a conversation by itself (#1208)', () => {
  it('opens the most recent one, because there is no longer a surface to fall back to', async () => {
    // Killed by: frontend/src/App.tsx :: const mostRecent = rooms[0];
    // Becomes: const mostRecent = undefined;
    render(<App />);

    await openedConversation();
    // `RoomService.list_rooms` orders by `updated_at` for every head at once (#1053), so
    // the head takes the first row rather than sorting for itself.
    expect(sent('GET', '/api/rooms/r1')).toHaveLength(1);
    expect(sent('GET', '/api/rooms/r2')).toHaveLength(0);
    expect(sent('POST', '/api/rooms')).toHaveLength(0);
  });

  it('starts one on an install that has none, under the name its first message takes', async () => {
    // Killed by: frontend/src/App.tsx :: void (mostRecent ? openRoom(mostRecent.room_id) : handleNewRoom());
    // Becomes: void (mostRecent ? openRoom(mostRecent.room_id) : Promise.resolve());
    roomsOnServer = new Map();
    render(<App />);

    await openedConversation();
    const create = sent('POST', '/api/rooms');
    expect(create).toHaveLength(1);
    // No prompt and no blank title: `NEW_CONVERSATION_TITLE` stands until the first
    // message renames it (F1), which is the rail's New taken by a different route.
    expect(JSON.parse(create[0].body ?? '{}')).toMatchObject({
      title: NEW_CONVERSATION_TITLE,
      agent_ids: ['champion'],
    });
  });

  it('waits for the conversation list, rather than starting a second one beside it', async () => {
    // Killed by: frontend/src/App.tsx :: if (!roomsListed || !metadataListed) return;
    // Becomes: if (!metadataListed) return;
    hold('GET /api/rooms');
    render(<App />);
    await settle();

    // The list has not answered, so "this install has no conversation" is not yet known.
    expect(sent('POST', '/api/rooms')).toHaveLength(0);

    await release('GET /api/rooms');
    await openedConversation();
    expect(sent('POST', '/api/rooms')).toHaveLength(0);
  });

  it('waits for the agent list, so the conversation it starts is not seated with nobody', async () => {
    // Killed by: frontend/src/App.tsx :: if (!roomsListed || !metadataListed) return;
    // Becomes: if (!roomsListed) return;
    roomsOnServer = new Map();
    hold('GET /api/agents');
    render(<App />);
    await settle();

    expect(sent('POST', '/api/rooms')).toHaveLength(0);

    await release('GET /api/agents');
    await openedConversation();
    // Seated, not empty: `handleNewRoom` reads `selectedAgent`, which the agent list sets.
    expect(JSON.parse(sent('POST', '/api/rooms')[0].body ?? '{}')).toMatchObject({
      agent_ids: ['champion'],
    });
  });

  it('asks once when the Core refuses to start one, and says why', async () => {
    // The latch is not what holds this case to one ask, and the declaration that said so
    // did not kill (#1246): the auto-open effect has a dependency array, and a refused
    // create changes none of it, so the effect does not re-run and dropping
    // `|| autoOpenRef.current` leaves this case green. The latch is load-bearing where a
    // dependency does move -- `keeps the notice when the open conversation is deleted but
    // the list cannot be re-read` dies of that same mutation -- so the claim is true
    // there and not here. What this case pins is the second half of its own title:
    // the refusal is said, in the Core's words, rather than swallowed into a silent
    // empty workspace.
    // Killed by: frontend/src/App.tsx :: setRoomNotice(`Could not start a conversation: ${roomFailureReason(err)}`);
    // Becomes:
    roomsOnServer = new Map();
    createRefusal = 'The room store is read-only.';
    render(<App />);

    expect(await screen.findByTestId('room-notice')).toHaveTextContent(
      'Could not start a conversation: The room store is read-only.',
    );
    await settle();
    expect(sent('POST', '/api/rooms')).toHaveLength(1);
    expect(screen.queryByTestId('room-conversation')).toBeNull();
  });

  it('leaves the conversation a New click is creating alone, rather than starting a second', async () => {
    // Killed by: frontend/src/App.tsx :: || createInFlightRef.current) return;
    // Becomes: ) return;
    roomsOnServer = new Map();
    hold('GET /api/agents');
    hold('POST /api/rooms');
    render(<App />);
    await settle();

    // The auto-open is still waiting on the agent list, so nothing of its own exists yet.
    expect(sent('POST', '/api/rooms')).toHaveLength(0);

    fireEvent.click(screen.getByTestId('new-conversation-button'));
    await settle();
    expect(sent('POST', '/api/rooms')).toHaveLength(1);

    // The agent list lands while that create is still in flight. `currentRoomId` is set
    // two awaits later -- after the create and after the list re-read -- so for the whole
    // of the user's click the effect's own guard reads "nothing open, latch free", which
    // is the reading that starts a second conversation beside the one being made (#1288).
    await release('GET /api/agents');
    await release('POST /api/rooms');
    await openedConversation();

    expect(sent('POST', '/api/rooms')).toHaveLength(1);
    // One row in the rail, not the clicked conversation plus a stranded empty one.
    expect(screen.getByTestId('conversation-new-1')).toBeVisible();
    expect(screen.queryByTestId('conversation-new-2')).toBeNull();
  });

  it('lets go of the guard when the Core refuses the New, so a later open still happens', async () => {
    // Killed by: frontend/src/App.tsx :: createInFlightRef.current = false;
    // Becomes:
    //
    // A guard taken on the click and never given back is the failure this pins. It would
    // hide the second create just as well and cost the whole auto-open: nothing below
    // would open a conversation again for the rest of the session, because the effect
    // would read "a create is in flight" forever. This is also why the guard is separate
    // from the latch rather than being it -- the latch is spent deliberately and released
    // by a delete, and a click that produced no conversation may not spend it by accident.
    createRefusal = 'The room store is read-only.';
    hold('GET /api/agents');
    hold('POST /api/rooms');
    render(<App />);
    await settle();

    fireEvent.click(screen.getByTestId('new-conversation-button'));
    await settle();
    await release('GET /api/agents');
    // Held while the create is in flight -- the guard doing its job.
    expect(sent('GET', '/api/rooms/r1')).toHaveLength(0);

    await release('POST /api/rooms');
    // The refusal stands, in the Core's words, and nothing was opened over the top of it.
    expect(await screen.findByTestId('room-notice')).toHaveTextContent(
      'Could not start a conversation: The room store is read-only.',
    );
    expect(sent('POST', '/api/rooms')).toHaveLength(1);
    expect(screen.queryByTestId('room-conversation')).toBeNull();

    // A delete releases the latch, as it always has. The guard must be free by now too,
    // or that release reaches an effect that still refuses to act on it.
    fireEvent.click(screen.getByRole('button', { name: 'Delete “Release notes”' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete conversation' }));

    await waitFor(() => expect(sent('GET', '/api/rooms/r1')).toHaveLength(1));
    await openedConversation();
  });
});

/**
 * The dock describes the conversation on screen and a clone seated in it (#1356).
 *
 * It used to read a fixed default session, or whichever session the rail's clone pick
 * pointed at, so what it showed and what the centre column showed were two different
 * conversations. What is pinned here is the wiring only `App` has: which room and seat it
 * hands the dock, and that "Open in Docs" fronts Docs on the file.
 */
describe('App dock scope (#1356)', () => {
  const openDockOn = async (tab: string) => {
    if (screen.queryByTestId('artifacts-dock') === null) {
      fireEvent.click(screen.getByTestId('toggle-dock'));
    }
    fireEvent.click(await screen.findByTestId(tab));
  };

  it('reads Activity and Docs for the open room and its seat, and follows a room switch', async () => {
    // Killed by: frontend/src/App.tsx :: roomId={dockRoomId}
    // Becomes: roomId={null}
    window.localStorage.setItem('uclone-x.rail.open', 'true');
    render(<App />);
    await openedConversation();

    await openDockOn('tab-activity');
    await waitFor(() =>
      expect(sent('GET', '/api/rooms/r1/seats/champion/history')).toHaveLength(1),
    );
    // Docs is the dock's first surface, so it has read once already by now.
    await openDockOn('tab-artifacts');
    await waitFor(() => expect(sent('GET', '/api/rooms/r1/artifacts').length).toBeGreaterThan(0));

    // Another conversation, picked in the rail: the dock re-reads for it, no reload.
    fireEvent.click(await screen.findByTestId('conversation-r2'));
    await waitFor(() => expect(sent('GET', '/api/rooms/r2/artifacts')).toHaveLength(1));

    // Nothing asked for a session or for the whole workspace's files.
    expect(requests.some((r) => /\/api\/sessions?\b|session_id=|\/api\/artifacts\?/.test(r.url))).toBe(
      false,
    );
  });

  // The seat picker's choice (#1389): it holds for the conversation it was made in, is not
  // carried into another, comes back with its own conversation, and gives way once that
  // clone is no longer seated there.
  // Killed by: frontend/src/App.tsx :: room && seatChoice && seatChoice.roomId === room.room_id &&
  // Becomes: room && seatChoice &&
  // Killed by: frontend/src/App.tsx :: seatedClones.some((p) => p.id === seatChoice.seatId)
  // Becomes: true
  it('keeps a picked seat to its own conversation, and drops it once that clone leaves', async () => {
    seatsOnServer.set('r1', ['champion', 'scout']);
    seatsOnServer.set('r2', ['champion', 'scout']);
    const picker = () => screen.getByTestId('dock-seat-picker') as HTMLSelectElement;
    const lastSeatRead = (roomId: string) =>
      requests
        .map((r) => new RegExp(`^/api/rooms/${roomId}/seats/([^/]+)/history$`).exec(r.url))
        .filter((m) => m !== null)
        .map((m) => m[1])
        .pop();
    render(<App />);
    await openedConversation();
    await openDockOn('tab-activity');

    // Nobody has spoken and nothing is picked: the first seat.
    await waitFor(() => expect(picker().value).toBe('champion'));
    fireEvent.change(picker(), { target: { value: 'scout' } });
    await waitFor(() => expect(lastSeatRead('r1')).toBe('scout'));
    expect(picker().value).toBe('scout');

    // Another conversation seating the same clones starts from its own default.
    fireEvent.click(screen.getByTestId('conversation-r2'));
    await waitFor(() => expect(lastSeatRead('r2')).toBe('champion'));
    expect(picker().value).toBe('champion');
    expect(sent('GET', '/api/rooms/r2/seats/scout/history')).toHaveLength(0);

    // Back in the first, the pick made there holds.
    fireEvent.click(screen.getByTestId('conversation-r1'));
    await waitFor(() => expect(picker().value).toBe('scout'));
    await waitFor(() => expect(lastSeatRead('r1')).toBe('scout'));

    // Scout leaves the first conversation; the next read of it seats someone else.
    seatsOnServer.set('r1', ['champion', 'critic']);
    fireEvent.click(screen.getByTestId('conversation-r2'));
    await waitFor(() => expect(picker().value).toBe('champion'));
    fireEvent.click(screen.getByTestId('conversation-r1'));
    await waitFor(() => expect(lastSeatRead('r1')).toBe('champion'));
    expect(picker().value).toBe('champion');
    expect([...picker().options].map((o) => o.value)).toEqual(['champion', 'critic']);
  });

  // Opening a conversation clears the dock's scope until the read lands (#1389). The dock
  // said "No conversation is open" through that wait, and went on saying it when the read
  // failed: a wrong cause, where the right one was known.
  // Killed by: frontend/src/App.tsx :: : 'Opening this conversation…'
  // Becomes: : null
  // Killed by: frontend/src/App.tsx :: ? roomOpenError !== null
  // Becomes: ? false
  // Killed by: frontend/src/App.tsx :: : roomOpenError !== null
  // Becomes: : false
  it('says a conversation is opening while it loads, and why when it could not be read', async () => {
    render(<App />);
    await openedConversation();
    await openDockOn('tab-artifacts');
    await screen.findByTestId('doc-reason');

    hold('GET /api/rooms/r2');
    fireEvent.click(screen.getByTestId('conversation-r2'));
    await settle();
    const pending = await screen.findByTestId('dock-room-pending');
    expect(pending).toHaveTextContent('Opening this conversation');
    expect(screen.getByTestId('artifacts-dock')).not.toHaveTextContent('No conversation is open');

    // The read fails: the dock names the Core's cause, and what to do about it.
    roomsOnServer.delete('r2');
    await release('GET /api/rooms/r2');
    await waitFor(() =>
      expect(screen.getByTestId('dock-room-pending')).toHaveTextContent(
        'Could not read this conversation',
      ),
    );
    expect(screen.getByTestId('dock-room-pending')).toHaveTextContent(missing('r2'));
    expect(screen.getByTestId('artifacts-dock')).not.toHaveTextContent('No conversation is open');
    expect(screen.getByTestId('artifacts-dock')).not.toHaveTextContent('Opening this conversation');

    // Dismissing the notice does not turn the failure back into a load.
    fireEvent.click(screen.getByTestId('dismiss-room-notice'));
    expect(screen.getByTestId('room-loading')).not.toHaveTextContent('Opening this conversation');
    expect(screen.getByTestId('room-loading')).toHaveTextContent(missing('r2'));
    expect(screen.getByTestId('dock-room-pending')).toHaveTextContent(missing('r2'));

    // Another conversation that reads fine puts the dock's own surfaces back.
    fireEvent.click(screen.getByTestId('conversation-r1'));
    expect(await screen.findByTestId('doc-reason')).toBeInTheDocument();
    expect(screen.queryByTestId('dock-room-pending')).toBeNull();
  });

  // A read that fails without the Core saying why must not put the browser's transport
  // message or an HTTP status line in front of someone who does not read either (P0), and
  // must still say that it failed (P6).
  // Killed by: frontend/src/App.tsx :: const reason = roomReadReason(err);
  // Becomes: const reason = describeError(err);
  it.each([
    ['no-answer', "The app's background service didn't answer."],
    ['bare-500', 'Something went wrong reading it.'],
  ])('says a %s read failed in a sentence of ours, not in transport text', async (fault, sentence) => {
    render(<App />);
    await openedConversation();
    await openDockOn('tab-artifacts');
    await screen.findByTestId('doc-reason');

    roomReadFault.set('r2', fault as 'no-answer' | 'bare-500');
    fireEvent.click(screen.getByTestId('conversation-r2'));
    await waitFor(() =>
      expect(screen.getByTestId('dock-room-pending')).toHaveTextContent(
        'Could not read this conversation',
      ),
    );
    fireEvent.click(screen.getByTestId('dismiss-room-notice'));
    for (const surface of [screen.getByTestId('dock-room-pending'), screen.getByTestId('room-loading')]) {
      expect(surface).toHaveTextContent(`${sentence} Pick it again`);
      expect(surface).not.toHaveTextContent('Failed to fetch');
      expect(surface.textContent).not.toMatch(/\b\d{3} [A-Z]/);
    }
  });

  // "Pick it again to retry" is a promise: picking the same conversation re-reads it.
  it('re-reads a conversation that could not be read when it is picked again, and clears the failure', async () => {
    render(<App />);
    await openedConversation();
    await openDockOn('tab-artifacts');
    await screen.findByTestId('doc-reason');

    roomReadFault.set('r2', 'no-answer');
    fireEvent.click(screen.getByTestId('conversation-r2'));
    await screen.findByText(/Could not read this conversation/);
    const readsBefore = requests.filter((r) => r.method === 'GET' && r.url === '/api/rooms/r2').length;

    roomReadFault.delete('r2');
    hold('GET /api/rooms/r2');
    fireEvent.click(screen.getByTestId('conversation-r2'));
    await settle();
    // While the re-read is out, the old failure is not still stated as the current one.
    expect(screen.getByTestId('dock-room-pending')).toHaveTextContent('Opening this conversation');
    expect(screen.getByTestId('dock-room-pending')).not.toHaveTextContent('Could not read');
    await release('GET /api/rooms/r2');
    expect(await screen.findByTestId('doc-reason')).toBeInTheDocument();
    expect(
      requests.filter((r) => r.method === 'GET' && r.url === '/api/rooms/r2').length,
    ).toBe(readsBefore + 1);
    expect(screen.queryByTestId('dock-room-pending')).toBeNull();
    expect(screen.queryByText(/Could not (read|open) this conversation/)).toBeNull();
  });

  it('fronts Docs on the file an Activity row wrote', async () => {
    // Killed by: frontend/src/App.tsx :: setActiveDockSurface('artifacts');
    // Becomes:
    seatHistoryOnServer = {
      turns: [],
      tool_uses: [
        {
          turn_id: 't1',
          participant_id: 'champion',
          tool_name: 'write_file',
          tool_call_id: 'w1',
          status: 'success',
          error: null,
          duration_ms: 3,
          arguments_preview: '{}',
          output_preview: 'ok',
          truncated: false,
          written_path: 'notes/plan.md',
          wrote_unnamed: false,
          subagent_id: null,
          recorded_at: '2026-09-22T10:00:00Z',
          seq: 2,
        },
      ],
    };
    render(<App />);
    await openedConversation();
    await openDockOn('tab-activity');

    fireEvent.click(await screen.findByTestId('activity-open-doc-w1'));
    expect(await screen.findByTestId('doc-viewer')).toBeInTheDocument();
    await waitFor(() =>
      expect(requests.some((r) => r.url === '/api/artifacts/content?path=notes%2Fplan.md')).toBe(
        true,
      ),
    );
  });
});

describe('App no default agent (#1125)', () => {
  it('adopts the first agent the server names, rather than opening on a literal name', async () => {
    // The head opened on `champion` whether or not this install had ever registered that
    // name, so a fresh install asked for a conversation addressed to nobody.
    // Killed by: frontend/src/App.tsx :: if (loaded.length > 0) firstNamed = loaded[0].id;
    // Becomes: if (loaded.length > 0) firstNamed = 'champion';
    agentsOnServer = [{ id: 'novelist', name: 'novelist' }];
    roomsOnServer = new Map();
    render(<App />);

    // The conversation it starts seats that agent, not the name the head once assumed.
    await openedConversation();
    expect(JSON.parse(sent('POST', '/api/rooms')[0].body ?? '{}')).toMatchObject({
      agent_ids: ['novelist'],
    });
  });

});

describe('App workspace dock opener (#1055)', () => {
  it('reopens on the surface the user left, instead of resetting to Docs & Artifacts', async () => {
    render(<App />);
    await openedConversation();

    // Open the dock and switch off its default surface (Docs & Artifacts, which renders
    // `doc-viewer`) onto Knowledge Graph, which is in the developer drawer since 2026-09-22.
    await turnOnDeveloperMode();
    fireEvent.click(screen.getByTestId('toggle-dock'));
    fireEvent.click(await screen.findByTestId('dev-tab-knowledge_graph'));
    expect(await screen.findByTestId('knowledge-graph-viewer')).toBeInTheDocument();

    // Close and reopen. This used to be asserted through a second, floating opener pinned
    // over the conversation; that control is gone as a duplicate of this toggle, and what
    // it was here to pin -- that reopening does not force the surface back to 'artifacts'
    // -- is a property of reopening itself, whichever control does it.
    fireEvent.click(screen.getByTestId('dock-close'));
    fireEvent.click(screen.getByTestId('toggle-dock'));

    expect(await screen.findByTestId('knowledge-graph-viewer')).toBeInTheDocument();
    expect(screen.queryByTestId('doc-viewer')).toBeNull();
  });

  // No mutation declared: what this pins is an absence, and there is no line to break in
  // order to bring a deleted control back. It guards the deletion against a reinstatement,
  // which is a review question rather than a scoring one.
  it('offers one control for opening the workspace, not a second one over the text', async () => {
    render(<App />);
    await openedConversation();

    expect(screen.getByTestId('toggle-dock')).toBeInTheDocument();
    expect(screen.queryByTestId('floating-dock-btn')).toBeNull();
  });
});

describe('App developer mode (owner ruling 2026-09-22)', () => {
  it('opens with developer mode off: no developer drawer in the dock', async () => {
    // Killed by: frontend/src/lib/developerMode.ts :: return window.localStorage.getItem(DEVELOPER_MODE_KEY) === 'true';
    // Becomes: return window.localStorage.getItem(DEVELOPER_MODE_KEY) !== 'false';
    render(<App />);
    await openedConversation();
    fireEvent.click(screen.getByTestId('toggle-dock'));

    expect(await screen.findByTestId('doc-viewer')).toBeInTheDocument();
    expect(screen.queryByTestId('dev-drawer')).toBeNull();
    expect(screen.queryByTestId('dev-tab-knowledge_graph')).toBeNull();
  });

  it('keeps developer mode across a reload once it is switched on', async () => {
    // Killed by: frontend/src/App.tsx :: storeDeveloperMode(on);
    // Becomes:
    const first = render(<App />);
    await openedConversation();
    await turnOnDeveloperMode();
    fireEvent.click(screen.getByTestId('toggle-dock'));
    expect(await screen.findByTestId('dev-drawer')).toBeInTheDocument();
    first.unmount();

    render(<App />);
    await openedConversation();
    fireEvent.click(screen.getByTestId('toggle-dock'));
    expect(await screen.findByTestId('dev-drawer')).toBeInTheDocument();
    expect(screen.getByTestId('dev-tab-ontology')).toBeInTheDocument();
  });
});

describe('App start-up reads (#1358)', () => {
  it('reads Skills, ACP and Evals only when Settings shows them, not at start-up', async () => {
    // Killed by: frontend/src/App.tsx :: fetch('/api/budget'),
    // Becomes: fetch('/api/budget'), fetch('/api/skills'), fetch('/api/acp/status'), fetch('/api/evaluations/latest'),
    render(<App />);
    await openedConversation();
    // Start-up's metadata read did run, so the absences below are not a read that never began.
    await waitFor(() => expect(sent('GET', '/api/budget').length).toBeGreaterThan(0));
    expect(sent('GET', '/api/skills')).toHaveLength(0);
    expect(sent('GET', '/api/acp/status')).toHaveLength(0);
    expect(sent('GET', '/api/evaluations/latest')).toHaveLength(0);

    // Settings reads the skill catalogue; its Diagnostics section, which developer mode adds,
    // reads the other two.
    await turnOnDeveloperMode();
    fireEvent.click(screen.getByTestId('open-settings'));
    await screen.findByTestId('settings-diagnostics');
    await waitFor(() => expect(sent('GET', '/api/skills').length).toBeGreaterThan(0));
    await waitFor(() => expect(sent('GET', '/api/acp/status').length).toBeGreaterThan(0));
    await waitFor(() => expect(sent('GET', '/api/evaluations/latest').length).toBeGreaterThan(0));
  });
});

describe('App conversation delete (#1058)', () => {
  const confirmDelete = (title: string) => {
    fireEvent.click(screen.getByRole('button', { name: `Delete “${title}”` }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete conversation' }));
  };

  it('opens the next conversation when the open one is deleted', async () => {
    render(<App />);
    await openedConversation();

    confirmDelete('Index tuning');

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull());
    expect(screen.queryByTestId('conversation-r1')).toBeNull();
    // Since #1208 there is nothing to land on but another conversation, so the head opens
    // the one that is now most recent rather than showing an empty column.
    await waitFor(() => expect(sent('GET', '/api/rooms/r2')).toHaveLength(1));
    await openedConversation();
    expect(screen.queryByTestId('room-notice')).toBeNull();
  });

  it('starts one when the conversation deleted was the last', async () => {
    roomsOnServer = new Map([['r1', 'Index tuning']]);
    render(<App />);
    await openedConversation();

    confirmDelete('Index tuning');

    await waitFor(() => expect(sent('POST', '/api/rooms')).toHaveLength(1));
    await openedConversation();
  });

  it('keeps the open conversation open when another one is deleted', async () => {
    render(<App />);
    await openedConversation();

    confirmDelete('Release notes');

    await waitFor(() => expect(screen.queryByTestId('conversation-r2')).toBeNull());
    expect(screen.getByTestId('room-conversation')).toBeInTheDocument();
    // The open one was never left, so nothing re-read it and nothing was started.
    expect(sent('GET', '/api/rooms/r1')).toHaveLength(1);
    expect(sent('POST', '/api/rooms')).toHaveLength(0);
  });

  it('shows the Core`s refusal for a conversation already deleted elsewhere, and drops its row', async () => {
    render(<App />);
    await openedConversation();
    roomsOnServer.delete('r1');

    confirmDelete('Index tuning');

    expect(await screen.findByRole('alert')).toHaveTextContent(missing('r1'));
    // The refusal says the conversation is gone, so the head stops showing it as there and
    // opens the one that is left.
    await waitFor(() => expect(screen.queryByTestId('conversation-r1')).toBeNull());
    await waitFor(() => expect(sent('GET', '/api/rooms/r2')).toHaveLength(1));
    expect(screen.getByRole('alertdialog')).toBeInTheDocument();
  });

  it('keeps the notice when the open conversation is deleted but the list cannot be re-read', async () => {
    // Leaving the deleted conversation clears the room notice, so it has to happen before
    // the list re-read reports its failure, not after: otherwise the failure is wiped in the
    // same tick and the deleted row stays in the rail with nothing to explain it. The
    // auto-open is held for the same reason — it would ask for the conversation just
    // deleted, off a list known to be stale, and overwrite the notice with its 404.
    //
    // The anchor is the latch's own test, not the line that clears it: `leaveRoom` moves
    // `currentRoomId`, which is a dependency of the auto-open effect, so the effect re-runs
    // here and only the latch stops it. The clearing line sits in the *successful* branch of
    // the re-read, which this case never reaches, so deleting it leaves the case green
    // (#1246).
    // Re-pointed by #1288, which added a third guard to that line: the anchor names the
    // latch alone so the mutation still drops the latch and nothing else.
    // Killed by: frontend/src/App.tsx :: || autoOpenRef.current || createInFlightRef.current) return;
    // Becomes: || createInFlightRef.current) return;
    render(<App />);
    await openedConversation();
    const before = requests.length;
    let deleted = false;
    const served = globalThis.fetch;
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input).split('?')[0];
        const method = (init?.method ?? 'GET').toUpperCase();
        if (method === 'DELETE') deleted = true;
        if (deleted && path === '/api/rooms' && method === 'GET') {
          requests.push({ url: String(input), method });
          return Promise.resolve(answer({ detail: 'The room store could not be read.' }, 500));
        }
        return served(input, init);
      }),
    );

    confirmDelete('Index tuning');

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull());
    expect(screen.queryByTestId('room-conversation')).toBeNull();
    expect(screen.getByTestId('rooms-list-notice')).toHaveTextContent(
      'Could not list your conversations: The room store could not be read.',
    );
    await settle();
    expect(requests.slice(before).filter((r) => r.url.startsWith('/api/rooms/'))).toHaveLength(1);
  });
});

/**
 * A saturation read whose answer arrives into a conversation that moved underneath it (#1256).
 *
 * The read is issued, the record is bumped while it is out, and the answer can no longer be
 * committed against the generation it was asked in. The question, though, is still owed --
 * and before #1256 nothing re-asked it: the read is issued on a landed turn, and the turn
 * that prompted this one may have been the conversation's last, after which no event arrives
 * at all. Measured on the compaction flow: 2 of 8 runs ended with `GET /context` answering
 * `is_saturated: true` and no banner on screen.
 *
 * **This survives #1286's removal of the sampling cadence**, because it was never about the
 * cadence. The gate is gone and every landed turn now reads, and a read whose answer is
 * dropped still needs re-asking when no further turn is coming.
 *
 * **The interleaving is forced, not hoped for.** A race reproduced by timing proves nothing
 * on the run where it does not fire. `hold` parks `GET /api/rooms/r1/context` before the
 * page is rendered, so the read `openRoom` issues is provably still in flight -- the test
 * asserts the request was made and the banner is absent. The generation is then bumped by
 * seating a clone, which commits through `commitRoomAct`; the test asserts that request
 * was made before releasing anything. Only then is the context answer delivered. Every
 * step is a state the test checks, so there is no ordering the run can take that silently
 * skips the race.
 *
 * `refreshRoom` commits through `commitRoom`, which does **not** bump the generation.
 */
describe('App re-asks a saturation read the record moved under (#1256)', () => {
  // Neutered at the bound rather than at the branch: one attempt *is* the old code --
  // issue the read, refuse a stale answer, and stop -- so the mutation restores the
  // defect exactly instead of approximating it.
  // Killed by: frontend/src/lib/rooms.ts :: export const CONTEXT_READ_ATTEMPTS = 3;
  // Becomes: export const CONTEXT_READ_ATTEMPTS = 1;
  it('raises the banner without waiting for a turn that may never come', async () => {
    saturatedOnServer = true;
    // A clone not yet seated, for the act below to seat.
    agentsOnServer = [...agentsOnServer, { id: 'scout', name: 'scout' }];
    // Parked before the render, so `openRoom`'s forced read cannot answer until released.
    hold('GET /api/rooms/r1/context');
    render(<App />);
    await openedConversation();

    // The read is out: asked, unanswered, and nothing on screen from it yet.
    expect(sent('GET', '/api/rooms/r1/context')).toHaveLength(1);
    expect(screen.queryByTestId('saturation-banner')).toBeNull();

    // Bump the generation under it, the way the field does: seating a clone is an act, and
    // its answer commits through `commitRoomAct`, which advances the generation by one. (A
    // send used to be the act here; since #1412 its re-read is an ordinary read.)
    fireEvent.click(screen.getByTestId('add-someone'));
    fireEvent.click(await screen.findByTestId('invite-scout'));
    await waitFor(() => expect(sent('POST', '/api/rooms/r1/participants')).toHaveLength(1));
    await settle();

    // Only now does the read come back, into a conversation whose generation has moved.
    // Before #1256 this answer was dropped and no further read was ever issued.
    await release('GET /api/rooms/r1/context');

    await waitFor(() => expect(screen.getByTestId('saturation-banner')).toBeInTheDocument());
    expect(screen.getByTestId('saturation-banner')).toHaveTextContent(
      'champion reached the 40-turn limit on context',
    );
    // Re-asked rather than answered from the dropped read: the second call is the evidence
    // that the guard marked the question owed instead of merely refusing an answer.
    expect(sent('GET', '/api/rooms/r1/context').length).toBeGreaterThan(1);
  });

  it('drops the answer for good when the conversation it asked about was left', async () => {
    // The other half of the same guard, and the reason the retry is not unconditional: a
    // read that comes back after the user moved to another conversation has nobody waiting
    // for it, and re-asking would put one conversation's fullness under another's name.
    // Killed by: frontend/src/App.tsx :: if (roomId !== currentRoomIdRef.current) return;
    // Becomes: if (false) return;
    saturatedOnServer = true;
    hold('GET /api/rooms/r1/context');
    render(<App />);
    await openedConversation();
    expect(sent('GET', '/api/rooms/r1/context')).toHaveLength(1);

    // Move to the other conversation, which answers `is_saturated` for itself.
    saturatedOnServer = false;
    fireEvent.click(screen.getByTestId('conversation-r2'));
    await settle();

    await release('GET /api/rooms/r1/context');
    await settle();

    expect(screen.queryByTestId('saturation-banner')).toBeNull();
    // Dropped, not retried: r1's fullness is nobody's question once r1 is not on screen.
    expect(sent('GET', '/api/rooms/r1/context')).toHaveLength(1);
  });
});

/**
 * Only the last read of a room that went out may land (#1412).
 *
 * After a send two reads of the room are out at once: the send's own re-read, issued before
 * the reply was recorded, and the one the stream's `final` fires, issued after. The server
 * answers each with the transcript as it stood when the read arrived; only *when* the page
 * receives those answers is this test's. Either order must leave the reply on screen.
 */
describe('App lands only the last read of a room it issued (#1412)', () => {
  const HUMAN = {
    seq: 1,
    sender_id: 'user',
    content: 'Tell me something.',
    kind: 'utterance',
    created_at: '2026-09-23T00:00:00Z',
    completed: true,
  };
  const REPLY = {
    seq: 2,
    sender_id: 'champion',
    content: 'The landed reply.',
    kind: 'utterance',
    created_at: '2026-09-23T00:00:01Z',
    completed: true,
    turn_id: 'turn-1',
  };

  let streams: { onmessage: ((e: MessageEvent) => void) | null }[];
  /** Answers to `GET /api/rooms/r1`, made by the server when asked and handed over on call. */
  let owed: (() => void)[];
  /** Set to hold `POST /api/rooms/r1/messages` until released. */
  let sendHeld: Promise<void> | null;

  const emit = (payload: Record<string, unknown>) =>
    act(() => {
      const data = JSON.stringify({
        topic: 'room.r1',
        event_type: 'AGENT_REPLY',
        payload: { room_id: 'r1', agent_id: 'champion', turn_id: 'turn-1', ...payload },
      });
      for (const stream of streams) stream.onmessage?.({ data } as MessageEvent);
    });

  const deliver = async (index: number) => {
    owed[index]();
    await settle();
  };

  /** Render with a stream the test can speak on, and hold every read of r1 until delivered. */
  const deferReads = async () => {
    streams = [];
    owed = [];
    const held = sendHeld;
    vi.stubGlobal(
      'EventSource',
      class extends SilentEventSource {
        constructor(url: string) {
          super(url);
          streams.push(this);
        }
      },
    );
    render(<App />);
    await openedConversation();

    const serve = globalThis.fetch;
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const answered = serve(input, init);
        const method = (init?.method ?? 'GET').toUpperCase();
        if (held && method === 'POST' && String(input) === '/api/rooms/r1/messages') {
          return held.then(() => answered);
        }
        if (method !== 'GET' || String(input).split('?')[0] !== '/api/rooms/r1') return answered;
        return answered.then((res) => new Promise<Response>((resolve) => owed.push(() => resolve(res))));
      }),
    );
  };

  beforeEach(() => {
    sendHeld = null;
  });

  /** Send, stream the reply, land it: both reads are out and neither is answered yet. */
  const bothReadsOwed = async () => {
    await deferReads();
    transcriptsOnServer.set('r1', [HUMAN]);
    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: HUMAN.content } });
    fireEvent.click(screen.getByTestId('send-message'));
    // The send's re-read: answered before the reply was recorded.
    await waitFor(() => expect(owed).toHaveLength(1));

    await emit({ status: 'generating' });
    await emit({ status: 'streaming', delta: 'The landed' });
    expect(screen.getByTestId('live-turn')).toBeInTheDocument();

    transcriptsOnServer.set('r1', [HUMAN, REPLY]);
    await emit({ status: 'final' });
    // The read `final` fired: answered with the reply's row in it.
    await waitFor(() => expect(owed).toHaveLength(2));
  };

  /** The reply is on screen: as its row, or as the words held until the row arrives. */
  const replyOnScreen = () =>
    screen.queryByTestId('row-2') !== null || screen.queryByTestId('live-turn') !== null;

  // The ordering does not replace the generation: a read that is the last one out, and so
  // passes the ordering, is still dropped when a Clear lands before it does.
  // Killed by: frontend/src/App.tsx :: roomGenerationRef.current = generation + 1;
  // Becomes: roomGenerationRef.current = generation;
  it('drops the last read issued when a Clear lands before it', async () => {
    await deferReads();
    transcriptsOnServer.set('r1', [HUMAN]);
    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: HUMAN.content } });
    fireEvent.click(screen.getByTestId('send-message'));
    await waitFor(() => expect(owed).toHaveLength(1));
    await deliver(0);
    expect(screen.getByTestId('row-1')).toHaveTextContent(HUMAN.content);

    // The reply lands and `final` fires its read, which is now the last one issued.
    transcriptsOnServer.set('r1', [HUMAN, REPLY]);
    await emit({ status: 'final' });
    await waitFor(() => expect(owed).toHaveLength(2));

    fireEvent.click(screen.getByTestId('clear-history'));
    fireEvent.click(await screen.findByTestId('confirm-history-change-yes'));
    await waitFor(() => expect(screen.getByTestId('transcript-empty')).toBeInTheDocument());

    await deliver(1);
    expect(screen.queryByTestId('row-1')).toBeNull();
    expect(screen.queryByTestId('row-2')).toBeNull();
    expect(screen.getByTestId('transcript-empty')).toBeInTheDocument();
  });

  // A send an act overtook makes no read of its own, so it must not take a ticket either:
  // one would overtake the read `final` fired after the act, and nothing would land.
  // Killed by: frontend/src/App.tsx :: if (generation !== roomGenerationRef.current) return;
  // Becomes: if (issueRoomRead(currentRoomId).generation !== generation) return;
  it('lands the read `final` fired when an act came in while the send was out', async () => {
    let release = () => {};
    sendHeld = new Promise<void>((resolve) => {
      release = resolve;
    });
    agentsOnServer = [...agentsOnServer, { id: 'scout', name: 'scout' }];
    await deferReads();
    transcriptsOnServer.set('r1', [HUMAN]);
    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: HUMAN.content } });
    fireEvent.click(screen.getByTestId('send-message'));
    await waitFor(() => expect(sent('POST', '/api/rooms/r1/messages')).toHaveLength(1));

    // The act, while the send is still out.
    fireEvent.click(screen.getByTestId('add-someone'));
    fireEvent.click(await screen.findByTestId('invite-scout'));
    await waitFor(() => expect(screen.getByTestId('row-1')).toHaveTextContent(HUMAN.content));

    transcriptsOnServer.set('r1', [HUMAN, REPLY]);
    await emit({ status: 'generating' });
    await emit({ status: 'final' });
    await waitFor(() => expect(owed).toHaveLength(1));

    release();
    await settle();
    // The send made no read of its own.
    expect(owed).toHaveLength(1);

    await deliver(0);
    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY.content);
  });

  // Picking the open conversation again re-reads it through `openRoom`, which takes a
  // ticket like any other read.
  // Killed by: frontend/src/App.tsx :: if (!roomReadsRef.current.overtaken(ticket)) setRoom(next);
  // Becomes: setRoom(next);
  it('drops an open read that a later refresh overtook, rather than putting an older transcript back', async () => {
    transcriptsOnServer.set('r1', [HUMAN]);
    await deferReads();
    fireEvent.click(screen.getByTestId('conversation-r1'));
    await waitFor(() => expect(owed).toHaveLength(1));

    transcriptsOnServer.set('r1', [HUMAN, REPLY]);
    await emit({ status: 'generating' });
    await emit({ status: 'final' });
    await waitFor(() => expect(owed).toHaveLength(2));

    await deliver(1);
    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY.content);

    await deliver(0);
    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY.content);
  });

  // A refresh that fails after a later one was issued says nothing: the later one answers.
  // Killed by: frontend/src/App.tsx :: if (answeredFor) return;
  // Becomes:
  it('says nothing about a refresh that failed after a later one was issued', async () => {
    transcriptsOnServer.set('r1', [HUMAN]);
    await deferReads();
    roomReadFault.set('r1', 'bare-500');
    await emit({ status: 'generating' });
    await emit({ status: 'final' });
    await waitFor(() => expect(owed).toHaveLength(1));

    roomReadFault.delete('r1');
    transcriptsOnServer.set('r1', [HUMAN, REPLY]);
    await emit({ status: 'final' });
    await waitFor(() => expect(owed).toHaveLength(2));

    await deliver(1);
    await deliver(0);
    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY.content);
    expect(screen.queryByTestId('dismiss-room-notice')).toBeNull();
  });

  // Killed by: frontend/src/App.tsx :: commitRoom(ticket, await roomsApi.get(currentRoomId));
  // Becomes: commitRoomAct(ticket.roomId, ticket.generation, await roomsApi.get(currentRoomId));
  it('keeps the reply on screen when the earlier read resolves between `final` and the later one', async () => {
    await bothReadsOwed();

    await deliver(0);
    expect(replyOnScreen()).toBe(true);
    expect(screen.queryByTestId('row-2')).toBeNull();

    await deliver(1);
    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY.content);
    expect(screen.queryByTestId('live-turn')).toBeNull();
  });

  // Killed by: frontend/src/App.tsx :: roomReadsRef.current.isStale(ticket, roomGenerationRef.current, currentRoomIdRef.current)
  // Becomes: roomResponseIsStale(ticket.generation, roomGenerationRef.current, ticket.roomId, currentRoomIdRef.current)
  it('drops the earlier read when it resolves after the later one, rather than putting an older transcript back', async () => {
    await bothReadsOwed();

    await deliver(1);
    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY.content);

    await deliver(0);
    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY.content);
    expect(screen.getByTestId('row-1')).toHaveTextContent(HUMAN.content);
  });
});

/**
 * A half-written message survives leaving the conversation and coming back (#1290).
 *
 * `draft` is `RoomConversation`'s own state, and `App` renders that component behind
 * `room && currentRoomId`. `openRoom` sets `room` to `null` **synchronously**, before its
 * `GET` is even issued, so every conversation switch unmounts the composer and takes the
 * draft with it. The user is left facing an empty box and a disabled Send, with nothing
 * said about the text that was there -- and no undo.
 */
describe('App keeps a half-written message across a conversation switch (#1290)', () => {
  it('gives the draft back when the user returns to the conversation they left', async () => {
    // Restores the defect where it lived rather than approximating it: a draft destroyed
    // by every `openRoom` is precisely what component-local state gave, since `openRoom`
    // nulls the room and unmounts the composer on its first line.
    // Killed by: frontend/src/App.tsx :: const generation = (roomGenerationRef.current += 1);
    // Becomes: const generation = (roomGenerationRef.current += 1); setDrafts({});
    render(<App />);
    await openedConversation();

    fireEvent.change(screen.getByTestId('room-composer'), {
      target: { value: 'half-written message' },
    });
    await settle();
    expect(screen.getByTestId('send-message')).toBeEnabled();

    fireEvent.click(screen.getByTestId('conversation-r2'));
    await settle();
    expect(screen.getByTestId('room-composer')).toHaveValue('');

    fireEvent.click(screen.getByTestId('conversation-r1'));
    await settle();

    expect(screen.getByTestId('room-composer')).toHaveValue('half-written message');
    expect(screen.getByTestId('send-message')).toBeEnabled();
  });
});

/**
 * Picking a clone (#1300).
 *
 * The rail owns the choosing and `App` owns what the choice changes. Before this, clicking a
 * clone moved a highlight in a 240px column and nothing else: the reader had asked a question
 * -- who is this? -- that the screen never answered. It answers in the dock, which is where
 * this head keeps everything that is one deliberate action away from the conversation
 * (ui-authoring §2).
 */
describe('App opens the picked clone (#1300)', () => {
  // Killed by: frontend/src/App.tsx :: setActiveDockSurface('clone'); // handleInspectAgent
  // Becomes: setActiveDockSurface('activity'); // handleInspectAgent
  it('opens the dock on that clone when clicking inspect, rather than switching a surface nobody can see', async () => {
    // One case, not two. The dock is closed on a first run -- U0's default state, and it
    // stays that way -- so opening it and putting the clone on it are one sentence: a
    // surface switched behind a closed dock is the same silence this issue is about. The
    // line beside the declared one, `setIsDockOpen(true)`, is what the first assertion
    // holds, and replacing it with `false` fails this case and no other.
    personasOnServer = [{ name: 'champion' }, { name: 'surveyor' }];
    render(<App />);
    await openedConversation();
    expect(screen.queryByTestId('artifacts-dock')).toBeNull();

    fireEvent.click(await screen.findByTestId('clone-avatar-surveyor'));

    expect(screen.getByTestId('artifacts-dock')).toBeInTheDocument();
    expect(screen.getByTestId('clone-profile-surveyor')).toBeInTheDocument();
  });

  it('opens the dock on that clone in edit mode when clicking settings', async () => {
    personasOnServer = [{ name: 'champion' }, { name: 'surveyor' }];
    render(<App />);
    await openedConversation();
    expect(screen.queryByTestId('artifacts-dock')).toBeNull();

    fireEvent.click(await screen.findByTestId('clone-menu-surveyor'));
    fireEvent.click(screen.getByTestId('persona-edit-surveyor'));

    expect(screen.getByTestId('artifacts-dock')).toBeInTheDocument();
    expect(screen.getByTestId('persona-editor')).toBeInTheDocument();
  });

  it('opens or starts a conversation when clicking a clone row', async () => {
    personasOnServer = [{ name: 'champion' }, { name: 'surveyor' }];
    render(<App />);
    await openedConversation();
    expect(screen.queryByTestId('artifacts-dock')).toBeNull();

    fireEvent.click(await screen.findByTestId('persona-item-surveyor'));

    expect(screen.queryByTestId('artifacts-dock')).toBeNull();
    const postRooms = sent('POST', '/api/rooms');
    expect(postRooms.length).toBeGreaterThan(0);
    const lastBody = JSON.parse(postRooms[postRooms.length - 1].body || '{}');
    expect(lastBody.agent_ids).toContain('surveyor');
  });
});

/**
 * Every room notice, for the two failures that carry no reason of the Core's (#1411).
 *
 * `describeError` printed `err.message`, so the Core not answering read "Failed to fetch" and
 * a bodyless 500 read "500 Internal Server Error" in a banner meant for someone who does not
 * read code. Each family below provokes its own notice by failing its own request both ways,
 * and the banner must say what happened in plain words: no transport text, no status line.
 */
describe('App room notices say what happened plainly (#1411)', () => {
  const FAULTS = ['no-answer', 'bare-500'] as const;
  type Fault = (typeof FAULTS)[number];

  class TalkingEventSource extends SilentEventSource {
    static last: TalkingEventSource | null = null;
    constructor(url: string) {
      super(url);
      TalkingEventSource.last = this;
    }
  }

  /** The banner, once it starts with `opening`, checked for everything U0 must not see. */
  const expectPlainNotice = async (opening: string, testId = 'room-notice') => {
    await waitFor(() => expect(screen.getByTestId(testId)).toHaveTextContent(opening));
    const text = screen.getByTestId(testId).querySelector('span')?.textContent ?? '';
    expect(text.startsWith(opening), text).toBe(true);
    expect(text).not.toContain('Failed to fetch');
    expect(text).not.toMatch(/\b\d{3} [A-Z]/);
    expectPlain(text);
  };

  /** The rail's own refusal line, checked the same way. */
  const expectPlainAlert = async () => {
    const alert = await screen.findByRole('alert');
    const text = alert.textContent ?? '';
    expect(text).not.toContain('Failed to fetch');
    expect(text).not.toMatch(/\b\d{3} [A-Z]/);
    expectPlain(text);
  };

  /**
   * Open `r1` with its saturation banner up, which is where Shorten is offered. The context
   * read is held until the conversation is on screen, as the #1256 cases hold it: answered
   * at once, the banner did not appear in this harness.
   */
  const openSaturated = async () => {
    saturatedOnServer = true;
    hold('GET /api/rooms/r1/context');
    render(<App />);
    await openedConversation();
    await release('GET /api/rooms/r1/context');
    await screen.findByTestId('saturation-banner');
  };

  const send = (words: string) => {
    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: words } });
    fireEvent.click(screen.getByTestId('send-message'));
  };

  // Killed by: frontend/src/App.tsx :: `Could not list your conversations: ${roomFailureReason(err)}
  // Becomes: `Could not list your conversations: ${(err as Error).message}
  it.each([
    ...FAULTS,
  ])('the list read (%s)', async (fault: Fault) => {
    routeFault.set('GET /api/rooms', fault);
    hold('POST /api/rooms');
    render(<App />);
    await expectPlainNotice('Could not list your conversations:', 'rooms-list-notice');
  });

  it.each([
    ...FAULTS,
  ])('the list re-read after a delete (%s)', async (fault: Fault) => {
    render(<App />);
    await openedConversation();
    routeFault.set('GET /api/rooms', fault);
    fireEvent.click(screen.getByRole('button', { name: 'Delete “Release notes”' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete conversation' }));
    await expectPlainNotice('Could not list your conversations:', 'rooms-list-notice');
  });

  // Killed by: frontend/src/App.tsx :: setRoomNotice(`Could not start a conversation: ${roomFailureReason(err)}`);
  // Becomes: setRoomNotice(`Could not start a conversation: ${String(err)}`);
  it.each([
    ...FAULTS,
  ])('starting a conversation (%s)', async (fault: Fault) => {
    render(<App />);
    await openedConversation();
    routeFault.set('POST /api/rooms', fault);
    fireEvent.click(screen.getByTestId('new-conversation-button'));
    await expectPlainNotice('Could not start a conversation:');
  });

  // Killed by: frontend/src/App.tsx :: `Could not shorten this conversation: ${roomFailureReason(err)}
  // Becomes: `Could not shorten this conversation: ${(err as Error).message}
  it.each([
    ...FAULTS,
  ])('shortening (%s)', async (fault: Fault) => {
    await openSaturated();
    routeFault.set('POST /api/rooms/r1/compact', fault);
    fireEvent.click(await screen.findByTestId('compact-room'));
    await expectPlainNotice('Could not shorten this conversation:');
  });

  // Killed by: frontend/src/App.tsx :: `Could not refresh this conversation: ${roomFailureReason(err)}
  // Becomes: `Could not refresh this conversation: ${(err as Error).message}
  it.each([
    ...FAULTS,
  ])('the re-read after shortening (%s)', async (fault: Fault) => {
    await openSaturated();
    routeFault.set('GET /api/rooms/r1', fault);
    fireEvent.click(await screen.findByTestId('compact-room'));
    await expectPlainNotice('Could not refresh this conversation:');
  });

  // Killed by: frontend/src/App.tsx :: `Could not clear this conversation: ${roomFailureReason(err)}
  // Becomes: `Could not clear this conversation: ${(err as Error).message}
  it.each([
    ...FAULTS,
  ])('clearing (%s)', async (fault: Fault) => {
    render(<App />);
    await openedConversation();
    routeFault.set('DELETE /api/rooms/r1/history', fault);
    fireEvent.click(screen.getByTestId('clear-history'));
    fireEvent.click(await screen.findByTestId('confirm-history-change-yes'));
    await expectPlainNotice('Could not clear this conversation:');
  });

  // Killed by: frontend/src/App.tsx :: could not be named after it: ${roomFailureReason(err)}
  // Becomes: could not be named after it: ${(err as Error).message}
  it.each([
    ...FAULTS,
  ])('naming a new conversation after its first message (%s)', async (fault: Fault) => {
    roomsOnServer = new Map();
    render(<App />);
    await openedConversation();
    routeFault.set('PATCH /api/rooms/new-1', fault);
    send('Plan the release');
    await expectPlainNotice('Your message was sent, but this conversation could not be named after it:');
  });

  // Killed by: frontend/src/App.tsx :: could not be re-read: ${roomFailureReason(err)}
  // Becomes: could not be re-read: ${(err as Error).message}
  it.each([
    ...FAULTS,
  ])('the re-read after a send (%s)', async (fault: Fault) => {
    render(<App />);
    await openedConversation();
    routeFault.set('GET /api/rooms/r1', fault);
    send('Any news?');
    await expectPlainNotice('Your message was sent, but this conversation could not be re-read:');
  });

  // Killed by: frontend/src/App.tsx :: `Could not stop this turn: ${roomFailureReason(err)}
  // Becomes: `Could not stop this turn: ${(err as Error).message}
  it.each([
    ...FAULTS,
  ])('stopping a turn (%s)', async (fault: Fault) => {
    vi.stubGlobal('EventSource', TalkingEventSource);
    render(<App />);
    await openedConversation();
    // A turn starts, as the Core announces one on the room's topic: Stop is offered.
    await act(async () => {
      TalkingEventSource.last!.onmessage!({
        data: JSON.stringify({
          topic: 'room.r1',
          event_type: 'AGENT_REPLY',
          payload: { room_id: 'r1', agent_id: 'champion', turn_id: 'turn-1', status: 'generating' },
        }),
      } as MessageEvent);
    });
    routeFault.set('POST /api/rooms/r1/stop', fault);
    fireEvent.click(await screen.findByTestId('stop-turn'));
    await expectPlainNotice('Could not stop this turn:');
  });

  // Killed by: frontend/src/App.tsx :: `Could not run that turn again: ${roomFailureReason(err)}
  // Becomes: `Could not run that turn again: ${(err as Error).message}
  it.each([
    ...FAULTS,
  ])('running a failed turn again (%s)', async (fault: Fault) => {
    transcriptsOnServer.set('r1', [
      { seq: 1, sender_id: 'user', kind: 'utterance', content: 'go', created_at: '2026-09-19T00:00:00Z', completed: true },
      {
        seq: 2,
        sender_id: 'champion',
        kind: 'utterance',
        content: '',
        created_at: '2026-09-19T00:00:01Z',
        completed: true,
        error: 'RuntimeError: provider dropped',
      },
    ]);
    render(<App />);
    await openedConversation();
    routeFault.set('POST /api/rooms/r1/retry', fault);
    fireEvent.click(await screen.findByTestId('retry-turn'));
    await expectPlainNotice('Could not run that turn again:');
  });

  // Killed by: frontend/src/App.tsx :: to this conversation: ${roomFailureReason(err)}
  // Becomes: to this conversation: ${(err as Error).message}
  it.each([
    ...FAULTS,
  ])('adding someone (%s)', async (fault: Fault) => {
    agentsOnServer = [
      { id: 'champion', name: 'champion' },
      { id: 'critic', name: 'critic' },
    ];
    render(<App />);
    await openedConversation();
    routeFault.set('POST /api/rooms/r1/participants', fault);
    fireEvent.click(screen.getByTestId('add-someone'));
    fireEvent.click(await screen.findByTestId('invite-critic'));
    await expectPlainNotice('Could not add critic to this conversation:');
  });

  // Killed by: frontend/src/App.tsx :: throw new Error(roomFailureReason(err));
  // Becomes: throw err;
  it.each([
    ...FAULTS,
  ])('renaming from the rail (%s)', async (fault: Fault) => {
    render(<App />);
    await openedConversation();
    routeFault.set('PATCH /api/rooms/r2', fault);
    fireEvent.click(screen.getByTestId('rename-conversation-r2'));
    const editor = await screen.findByTestId('conversation-title-editor');
    fireEvent.change(editor.querySelector('input')!, { target: { value: 'Launch notes' } });
    fireEvent.submit(editor);
    await expectPlainAlert();
  });

  // Killed by: frontend/src/App.tsx :: if (refusal !== null) throw new Error(roomFailureReason(refusal));
  // Becomes: if (refusal !== null) throw refusal;
  it.each([
    ...FAULTS,
  ])('deleting from the rail (%s)', async (fault: Fault) => {
    render(<App />);
    await openedConversation();
    routeFault.set('DELETE /api/rooms/r2', fault);
    fireEvent.click(screen.getByRole('button', { name: 'Delete “Release notes”' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete conversation' }));
    await expectPlainAlert();
  });
});

/**
 * A failed list read stays said until a list read lands or the reader dismisses it (#1439).
 *
 * It was a room notice, and opening a conversation clears those. With no list, the head
 * starts a conversation and opens it, so on the install where the list failed the failure
 * was taken off screen a moment later and the rail was left empty with nothing to say why.
 */
describe('App keeps a failed list read on screen (#1439)', () => {
  // Found by its words, not its container, so that moving it back into the room notice
  // fails only because it is cleared.
  // Killed by: frontend/src/App.tsx :: setRoomsListError(`Could not list your conversations: ${roomFailureReason(err)}`);
  // Becomes: setRoomNotice(`Could not list your conversations: ${roomFailureReason(err)}`);
  it('past the conversation the head starts when it has no list', async () => {
    routeFault.set('GET /api/rooms', 'no-answer');
    render(<App />);
    await openedConversation();
    await settle();
    expect(sent('POST', '/api/rooms')).toHaveLength(1);
    const notice = screen.getByText(/^Could not list your conversations:/);
    expect(notice).toHaveTextContent(
      "Could not list your conversations: The app's background service didn't answer.",
    );
    expectPlain(notice.textContent ?? '');
  });

  it('until a list read lands', async () => {
    routeFault.set('GET /api/rooms', 'bare-500');
    render(<App />);
    await openedConversation();
    await screen.findByTestId('rooms-list-notice');

    routeFault.delete('GET /api/rooms');
    fireEvent.click(screen.getByTestId('new-conversation-button'));
    await waitFor(() => expect(screen.queryByTestId('rooms-list-notice')).toBeNull());
  });

  it('until the reader dismisses it', async () => {
    routeFault.set('GET /api/rooms', 'no-answer');
    render(<App />);
    await openedConversation();
    fireEvent.click(await screen.findByTestId('dismiss-rooms-list-notice'));
    expect(screen.queryByTestId('rooms-list-notice')).toBeNull();
  });
});

/**
 * A conversation whose record will not load is listed as such, and can be deleted (#1440).
 *
 * The Core used to leave it out of the listing, so the rail could neither show it nor
 * offer to delete it, and the file stayed on disk with no way to reach it from the app.
 */
describe('App lists a conversation it cannot read (#1440)', () => {
  // Killed by: frontend/src/App.tsx :: setUnreadableRoomIds(listing.unreadable);
  // Becomes: setUnreadableRoomIds([]);
  it('shows a row for it, and deletes it from there', async () => {
    unreadableOnServer = new Set(['r_bad']);
    render(<App />);
    await openedConversation();
    const row = await screen.findByTestId('unreadable-conversation-r_bad');
    expect(row).toHaveTextContent('A conversation that could not be read');

    fireEvent.click(screen.getByTestId('delete-unreadable-conversation-r_bad'));
    expect(screen.getByRole('alertdialog')).toHaveTextContent(
      'Delete the conversation that could not be read?',
    );
    fireEvent.click(screen.getByRole('button', { name: 'Delete conversation' }));

    await waitFor(() => expect(screen.queryByTestId('unreadable-conversation-r_bad')).toBeNull());
    expect(screen.queryByRole('alertdialog')).toBeNull();
    expect(sent('DELETE', '/api/rooms/r_bad')).toHaveLength(1);
    // The conversation on screen was not the one deleted, and stays.
    expect(screen.getByTestId('room-conversation')).toBeInTheDocument();
  });
});

/**
 * A send that got no answer is not said to be "not sent" (#1441).
 *
 * The Core stores the message before it answers, so a missing answer can hide a message
 * that is there. The head re-reads the conversation and says only what that read showed,
 * and sending the same words again looks before it posts, so they are not stored twice.
 */
describe('App says what it knows about a send that got no answer (#1441)', () => {
  const WORDS = 'Ship the index change tonight';

  const send = () => {
    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: WORDS } });
    fireEvent.click(screen.getByTestId('send-message'));
  };

  const notice = async () => (await screen.findByTestId('send-error')).textContent ?? '';

  /** On the server: how many times `r1` holds the words. */
  const copiesOnServer = () =>
    (transcriptsOnServer.get('r1') ?? []).filter(
      (row) => (row as { content: string }).content === WORDS,
    ).length;

  it('treats a message the re-read finds as sent', async () => {
    render(<App />);
    await openedConversation();
    sendFault = 'stored-no-answer';
    send();

    await waitFor(() => expect(screen.getByTestId('room-composer')).toHaveValue(''));
    expect(screen.queryByTestId('send-error')).toBeNull();
    expect(sent('POST', '/api/rooms/r1/messages')).toHaveLength(1);
  });

  // Killed by: frontend/src/App.tsx :: return sendErr instanceof RoomsNoAnswerError ? null : sendErr;
  // Becomes: return null;
  it('says the message is there, and that something failed, when the Core stored it and then failed', async () => {
    render(<App />);
    await openedConversation();
    sendFault = 'stored-then-500';
    send();

    await waitFor(() => expect(screen.getByTestId('room-composer')).toHaveValue(''));
    const text = (await screen.findByTestId('room-notice')).textContent ?? '';
    expect(text).toContain(
      "Your message is in this conversation. Something went wrong in the app's background service.",
    );
    expect(text).not.toMatch(/not sent|Internal Server Error|500|SyntaxError/i);
    expectPlain(text);
    expect(screen.queryByTestId('send-error')).toBeNull();
    expect(copiesOnServer()).toBe(1);
  });

  // Killed by: frontend/src/lib/rooms.ts :: if (err.outcome === 'refused') return `Not sent: ${reason}`;
  // Becomes: return `Not sent: ${reason}`;
  it('says the message is not there when the re-read does not have it', async () => {
    render(<App />);
    await openedConversation();
    sendFault = 'lost';
    send();

    const text = await notice();
    expect(text).toBe(
      "The app's background service didn't answer. Your message is not in this conversation.",
    );
    expect(text).not.toMatch(/not sent/i);
    expectPlain(text);
    expect(screen.getByTestId('room-composer')).toHaveValue(WORDS);
  });

  // Killed by: frontend/src/App.tsx :: if (messageArrived(reread, content, pending.afterSeq)) return null;
  // Becomes: if (messageArrived(reread, content, pending.afterSeq)) {}
  it('claims neither way when the re-read fails too, and a resend does not store it twice', async () => {
    render(<App />);
    await openedConversation();
    sendFault = 'stored-no-answer';
    roomReadFault.set('r1', 'no-answer');
    send();

    const text = await notice();
    expect(text).toBe(
      "Could not confirm whether this conversation has your message: The app's background service didn't answer.",
    );
    expect(text).not.toMatch(/\bsent\b/i);
    expectPlain(text);
    expect(screen.getByTestId('room-composer')).toHaveValue(WORDS);

    sendFault = null;
    roomReadFault.delete('r1');
    fireEvent.click(screen.getByTestId('send-message'));

    await waitFor(() => expect(screen.getByTestId('room-composer')).toHaveValue(''));
    expect(screen.queryByTestId('send-error')).toBeNull();
    expect(sent('POST', '/api/rooms/r1/messages')).toHaveLength(1);
    expect(copiesOnServer()).toBe(1);
  });

  it('sends a resend whose first try the re-read shows never arrived', async () => {
    render(<App />);
    await openedConversation();
    sendFault = 'lost';
    roomReadFault.set('r1', 'bare-500');
    send();
    await notice();

    sendFault = null;
    roomReadFault.delete('r1');
    fireEvent.click(screen.getByTestId('send-message'));

    await waitFor(() => expect(screen.getByTestId('room-composer')).toHaveValue(''));
    expect(sent('POST', '/api/rooms/r1/messages')).toHaveLength(2);
  });

  // Killed by: frontend/src/lib/rooms.ts :: return err instanceof RoomsApiError && err.status >= 400 && err.status < 500;
  // Becomes: return false;
  it('still says "Not sent" when the Core refused it', async () => {
    render(<App />);
    await openedConversation();
    sendFault = { status: 400, detail: "Room 'r1' has no human named 'guest'." };
    send();

    const text = await notice();
    expect(text).toBe("Not sent: Room 'r1' has no human named 'guest'.");
    expectPlain(text);
    // A refusal is an answer: nothing is re-read to look for the message.
    expect(sent('GET', '/api/rooms/r1')).toHaveLength(1);
  });
});

describe('App room clone invites and group chat creation', () => {
  it('populates room invite choices from personas when agents is empty', async () => {
    agentsOnServer = [];
    personasOnServer = [{ name: 'champion' }, { name: 'surveyor' }];

    render(<App />);
    await openedConversation();

    // champion is seated by default fixture in r1, surveyor is not seated
    const addSomeone = screen.getByTestId('add-someone');
    fireEvent.click(addSomeone);

    // surveyor should be available to invite
    expect(screen.getByTestId('invite-surveyor')).toBeInTheDocument();
    // champion is already seated, so it should not be invitable
    expect(screen.queryByTestId('invite-champion')).toBeNull();

    // Clicking invite adds surveyor to the room
    fireEvent.click(screen.getByTestId('invite-surveyor'));
    await settle();
    expect(sent('POST', '/api/rooms/r1/participants')).toHaveLength(1);
  });

  it('seats the active clone when clicking + on Group Chats and retains it in group chats', async () => {
    personasOnServer = [{ name: 'champion' }, { name: 'surveyor' }];

    render(<App />);
    await openedConversation();

    const newGroupBtn = screen.getByTestId('new-conversation-button');
    fireEvent.click(newGroupBtn);
    await settle();

    // POST /api/rooms should have been called with the active persona
    const createReqs = sent('POST', '/api/rooms');
    expect(createReqs.length).toBeGreaterThan(0);
    const lastCreateBody = JSON.parse(createReqs[createReqs.length - 1].body ?? '{}');
    expect(lastCreateBody.agent_ids).toEqual(['champion']);
  });
});
