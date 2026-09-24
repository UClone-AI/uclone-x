import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  answerHint,
  applyRoomEvent,
  attributionLines,
  conversationShape,
  headerAttribution,
  mentionCandidates,
  mentionTokens,
  mentionUnderCaret,
  NEW_CONVERSATION_TITLE,
  participantLabel,
  resolveMention,
  roomResponseIsStale,
  RoomReadOrder,
  roomReadReason,
  roomFailureReason,
  ROOM_ACTION_NO_REASON,
  ROOM_APP_FAULT,
  RoomListingShapeError,
  RoomSendError,
  RoomsNoAnswerError,
  lastSeq,
  messageArrived,
  sendFailureNotice,
  sendWasRefused,
  roomsApi,
  RoomsApiError,
  ROOM_READ_NO_ANSWER,
  ROOM_READ_NO_REASON,
  seededTitle,
  senderOf,
  servedByLabel,
  serviceRefLabel,
  unansweredSilence,
  EMPTY_LIVE,
} from './rooms';
import type { RoomLiveState } from './rooms';
import type {
  RoomParticipant,
  RoomProvenance,
  RoomSpeakerDecision,
  RoomState,
  RoomReplyPayload,
  RoomTopicEvent,
  RoomTranscriptMessage,
} from '../types';
import publishedRoomEvents from '../test/room-stream-events.json';
import { expectPlain } from '../test/plainCopy';

const roster = (...ids: string[]): RoomParticipant[] => [
  { id: 'user', kind: 'human' },
  ...ids.map((id) => ({ id, kind: 'agent' as const })),
];

/** A ceiling high enough that it is not the thing under test. */
const NO_CEILING = 99;

describe('senderOf', () => {
  it('resolves a message by its own sender, not by whichever agent is selected', () => {
    // uclone2's group chat fell back to the selected clone when the lookup missed and
    // rendered one agent's name over another's words -- permanently, on message rows.
    const participants = roster('scout', 'critic');

    expect(senderOf(participants, 'critic')?.id).toBe('critic');
  });

  it('returns nothing rather than a guess when the sender is unknown', () => {
    expect(senderOf(roster('scout'), 'someone-who-left')).toBeNull();
  });
});

describe('participantLabel', () => {
  it('prefers the roster name and falls back to the raw id', () => {
    expect(participantLabel({ id: 'scout', kind: 'agent', display_name: 'Scout' })).toBe('Scout');
    expect(participantLabel({ id: 'scout', kind: 'agent' })).toBe('scout');
  });
});

describe('serviceRefLabel / servedByLabel', () => {
  it('builds a string from the provider and model, because `served_by` is an object', () => {
    // `Provenance.served_by` is a `ServiceRef` (`core/provenance.py`). Handing the object
    // to React threw "Objects are not valid as a React child" and blank-screened the
    // conversation on the first agent reply.
    expect(serviceRefLabel({ provider: 'ollama', model: 'qwen3:8b' })).toBe('ollama:qwen3:8b');
  });

  it('names the provider alone when there is no model to name', () => {
    expect(serviceRefLabel({ provider: 'ollama', model: null })).toBe('ollama');
  });

  it('has nothing to say when there is no provenance at all', () => {
    expect(servedByLabel(null)).toBeNull();
    expect(servedByLabel(undefined)).toBeNull();
  });

  it('states what was asked for when the Core reports a substitution', () => {
    const degraded: RoomProvenance = {
      path: 'primary',
      requested: { provider: 'ollama', model: 'qwen3:8b' },
      served_by: { provider: 'ollama', model: 'qwen3:4b' },
      degraded: true,
    };

    expect(servedByLabel(degraded)).toBe('ollama:qwen3:4b (asked for ollama:qwen3:8b)');
  });

  it('says only what served when nothing was substituted', () => {
    const clean: RoomProvenance = {
      path: 'primary',
      requested: { provider: 'ollama', model: 'qwen3:8b' },
      served_by: { provider: 'ollama', model: 'qwen3:8b' },
      degraded: false,
    };

    expect(servedByLabel(clean)).toBe('ollama:qwen3:8b');
  });
});

describe('conversationShape', () => {
  // Killed by: frontend/src/lib/rooms.ts :: return participants.filter((p) => p.kind === 'agent').length > 1 ? 'multi' : 'solo';
  // Becomes: return participants.length > 1 ? 'multi' : 'solo';
  it('counts agents, not participants: the human seat is not a second speaker', () => {
    expect(conversationShape(roster('scout'))).toBe('solo');
    expect(conversationShape(roster('scout', 'critic'))).toBe('multi');
  });

  // Killed by: frontend/src/lib/rooms.ts :: length > 1 ? 'multi' : 'solo'
  // Becomes: length !== 1 ? 'multi' : 'solo'
  it('calls a conversation nobody is in a 1:1, because there is nothing to tell apart', () => {
    expect(conversationShape(roster())).toBe('solo');
  });
});

describe('headerAttribution / attributionLines', () => {
  const turn = (seq: number, model: string | null): RoomTranscriptMessage => ({
    seq,
    sender_id: model === null ? 'user' : 'scout',
    content: `turn ${seq}`,
    kind: 'utterance',
    created_at: '2026-09-19T00:00:00Z',
    completed: true,
    provenance:
      model === null
        ? null
        : {
            path: 'primary',
            requested: { provider: 'ollama', model },
            served_by: { provider: 'ollama', model },
            degraded: false,
          },
  });

  /**
   * A turn the Core substituted a model on: served by one, asked for another.
   *
   * `path: 'failover'` and `degraded: true` are what the Core sets together; the display
   * label folds the requested model in, which is exactly what the change rule must not
   * compare on.
   */
  const degradedTurn = (
    seq: number,
    servedModel: string,
    requestedModel: string,
  ): RoomTranscriptMessage => ({
    ...turn(seq, servedModel),
    provenance: {
      path: 'failover',
      requested: { provider: 'ollama', model: requestedModel },
      served_by: { provider: 'ollama', model: servedModel },
      degraded: true,
    },
  });

  // Killed by: frontend/src/lib/rooms.ts :: else if (only !== model) return null;
  // Becomes: else if (false) return null;
  it('states a model in the header only while one model has served every turn', () => {
    // The header is a standing claim about every row beneath it. One model throughout,
    // and it is true; a second model, and there is no value that is true of both, so the
    // header carries none rather than the first turn's.
    expect(headerAttribution([turn(1, null), turn(2, 'qwen3:8b'), turn(3, 'qwen3:8b')])).toBe(
      'ollama:qwen3:8b',
    );
    expect(
      headerAttribution([turn(1, null), turn(2, 'qwen3:8b'), turn(3, 'hermes3:8b')]),
    ).toBeNull();
  });

  // The default this case refuses is the one `headerAttribution` would have to invent, so
  // that is where the mutation goes: a fallback string where the code returns `null`.
  // Killed by: frontend/src/lib/rooms.ts :: return only;
  // Becomes: return only ?? 'unknown';
  it('reports no model at all until a turn has been served, rather than a default', () => {
    expect(headerAttribution([turn(1, null)])).toBeNull();
    expect(headerAttribution([])).toBeNull();
  });

  // Killed by: frontend/src/lib/rooms.ts :: const perTurn = shape === 'multi' || headerAttribution(transcript) === null;
  // Becomes: const perTurn = true;
  it('prints nothing under a 1:1 every turn of which one model served', () => {
    const lines = attributionLines([turn(1, 'qwen3:8b'), turn(2, 'qwen3:8b')], 'solo');

    expect([...lines.keys()]).toEqual([]);
  });

  // Killed by: frontend/src/lib/rooms.ts :: else if (only !== model) return null;
  // Becomes: else if (false) return null;
  it('prints every served turn of a 1:1 once a second model appears, the early ones too', () => {
    // The rows the first-turn anchor left silent are the point. With turns 1-2 on one
    // model and 3-4 on another, a header naming the first and a line on the third leaves
    // rows 4 onward standing under a claim that is false of them.
    const transcript = [
      turn(1, 'qwen3:8b'),
      turn(2, 'qwen3:8b'),
      turn(3, 'hermes3:8b'),
      turn(4, 'hermes3:8b'),
    ];

    expect(headerAttribution(transcript)).toBeNull();
    const lines = attributionLines(transcript, 'solo');
    expect([...lines.keys()]).toEqual([1, 2, 3, 4]);
    expect(lines.get(2)).toBe('ollama:qwen3:8b');
    expect(lines.get(4)).toBe('ollama:hermes3:8b');
  });

  // Killed by: frontend/src/lib/rooms.ts :: const model = serviceRefLabel(message.provenance?.served_by);
  // Becomes: const model = servedByLabel(message.provenance);
  it('does not read a changed request as a changed model', () => {
    // One model served both turns; only what was asked for changed. Comparing the
    // rendered labels -- which fold "(asked for ...)" in -- made this read as a change,
    // took the header away and printed "came from ollama:qwen3:8b, not ollama:qwen3:8b".
    // The header is what this case is about: turn 2 prints its own line under [Rev 25]
    // (see below), and that is a fact about the turn, not a change between two.
    const transcript = [turn(1, 'qwen3:8b'), degradedTurn(2, 'qwen3:8b', 'hermes3:8b')];

    expect(headerAttribution(transcript)).toBe('ollama:qwen3:8b');
  });

  // [Rev 25] The header answers "what has served this conversation" and the turn line
  // answers "what happened here". Turn 2 asked for one model and was answered by the one
  // already in the header, so nothing but the line distinguishes it from an ordinary
  // turn -- the case a reader cannot infer. Turn 1 stays silent: the header speaks for it.
  // Killed by: frontend/src/lib/rooms.ts :: if (perTurn || message.provenance?.degraded) lines.set(message.seq, display);
  // Becomes: if (perTurn) lines.set(message.seq, display);
  it('prints a degraded serve on its own turn even when the model did not change', () => {
    const transcript = [turn(1, 'qwen3:8b'), degradedTurn(2, 'qwen3:8b', 'hermes3:8b')];

    expect(headerAttribution(transcript)).toBe('ollama:qwen3:8b');
    expect([...attributionLines(transcript, 'solo').keys()]).toEqual([2]);
    expect(attributionLines(transcript, 'solo').get(2)).toBe(
      'ollama:qwen3:8b (asked for ollama:hermes3:8b)',
    );
  });

  // Killed by: frontend/src/lib/rooms.ts :: yield [message, model, display];
  // Becomes: yield [message, model, model];
  it('carries what was asked for into the line, where a line is printed at all', () => {
    const transcript = [turn(1, 'qwen3:8b'), degradedTurn(2, 'hermes3:8b', 'qwen3:8b')];

    expect(attributionLines(transcript, 'solo').get(2)).toBe(
      'ollama:hermes3:8b (asked for ollama:qwen3:8b)',
    );
  });

  // The clause this case pins is `shape === 'multi'`, not the header one: a group prints
  // per turn whatever the header would say, so dropping the *second* clause leaves this
  // case green (#1246). Dropping the first is what kills it.
  // Killed by: frontend/src/lib/rooms.ts :: const perTurn = shape === 'multi' || headerAttribution(transcript) === null;
  // Becomes: const perTurn = headerAttribution(transcript) === null;
  it('states a group`s turns whatever its models, because the speaker changes anyway', () => {
    const lines = attributionLines([turn(1, 'qwen3:8b'), turn(2, 'qwen3:8b')], 'multi');

    expect([...lines.keys()]).toEqual([1, 2]);
    expect(lines.get(2)).toBe('ollama:qwen3:8b');
  });

  // Killed by: frontend/src/lib/rooms.ts :: if (message.kind !== 'utterance') continue;
  // Becomes: if (false) continue;
  it('reads only speech: a roster change is not a turn, whatever it carries', () => {
    // The row is skipped for being a roster change, not for happening to carry nothing.
    // Given a provenance -- which the wire is free to attach -- a join read as speech
    // would strip the header off a conversation whose model never changed, and print a
    // line under every turn of it.
    const join: RoomTranscriptMessage = {
      ...turn(2, 'hermes3:8b'),
      kind: 'join',
      sender_id: 'critic',
    };
    const transcript = [turn(1, 'qwen3:8b'), join, turn(3, 'qwen3:8b')];

    expect(headerAttribution(transcript)).toBe('ollama:qwen3:8b');
    expect([...attributionLines(transcript, 'solo').keys()]).toEqual([]);
  });

  // What this case pins is the skip itself: drop it and the unserved turn is yielded with
  // no model, which reads as a second model and takes the header away.
  // Killed by: frontend/src/lib/rooms.ts :: if (!model || !display) continue;
  // Becomes:
  it('skips a turn nothing reported a model for, rather than reading it as a model', () => {
    // A failed turn has no provenance. Treating its absence as a value of its own would
    // read as a second model, take the header away and print a line under every turn.
    const transcript = [turn(1, 'qwen3:8b'), turn(2, null), turn(3, 'qwen3:8b')];

    expect(headerAttribution(transcript)).toBe('ollama:qwen3:8b');
    expect([...attributionLines(transcript, 'solo').keys()]).toEqual([]);
  });
});

describe('mentionTokens', () => {
  it('reads addresses the way `MENTION_PATTERN` does', () => {
    expect(mentionTokens('(@scout) and **@critic**, ask @d.b-a. please')).toEqual([
      'scout',
      'critic',
      'd.b-a.',
    ]);
  });

  it('is not fooled by an email address or a nested handle', () => {
    expect(mentionTokens('mail me@example.com or ping @@scout')).toEqual([]);
  });
});

describe('resolveMention', () => {
  it('matches case-insensitively, as the Core does on both sides', () => {
    expect(resolveMention(roster('scout'), 'SCOUT')?.id).toBe('scout');
  });

  it('falls back to aliases, which `MentionSelector` also resolves', () => {
    const participants: RoomParticipant[] = [
      { id: 'database_reviewer', kind: 'agent', aliases: ['dba', 'DB'] },
    ];

    expect(resolveMention(participants, 'db')?.id).toBe('database_reviewer');
  });

  it('prefers an id over another participant`s alias', () => {
    const participants: RoomParticipant[] = [
      { id: 'scout', kind: 'agent', aliases: [] },
      { id: 'critic', kind: 'agent', aliases: ['scout'] },
    ];

    expect(resolveMention(participants, 'scout')?.id).toBe('scout');
  });

  it('retries a token with trailing punctuation removed', () => {
    expect(resolveMention(roster('scout'), 'scout.')?.id).toBe('scout');
  });

  it('resolves an id containing a regex metacharacter instead of throwing', () => {
    // The Core permits it: `room/service.py` refuses only blank, whitespace-padded and
    // `__` ids. Interpolating one into `new RegExp()` raised `SyntaxError` during render.
    expect(resolveMention(roster('a('), 'a(')?.id).toBe('a(');
    expect(resolveMention(roster('a('), 'a')).toBeNull();
  });
});

describe('answerHint', () => {
  it('names the only agent in the conversation', () => {
    expect(answerHint(roster('scout'), 'check the index', '', NO_CEILING)).toBe(
      'scout will answer',
    );
  });

  it('names an addressed agent over the designated responder', () => {
    expect(answerHint(roster('scout', 'dba'), '@dba please look', 'scout', NO_CEILING)).toBe(
      'dba will answer',
    );
  });

  it('names the designated responder when nobody is addressed', () => {
    expect(answerHint(roster('scout', 'dba'), 'anyone?', 'scout', NO_CEILING)).toBe(
      'scout will answer',
    );
  });

  it('refuses to guess when a model will choose', () => {
    // Several agents, no address, no designated responder: the LLM selector decides and
    // the head cannot know. A likely-looking name here would be a guess shown as a fact.
    expect(answerHint(roster('scout', 'dba', 'critic'), 'anyone?', '', NO_CEILING)).toBe(
      'Someone will pick this up',
    );
  });

  it('says the conversation is empty rather than promising an answer', () => {
    expect(answerHint(roster(), 'hello', '', NO_CEILING)).toBe(
      'No one is in this conversation yet.',
    );
  });

  it('does not promise the only agent for an address the Core will refuse', () => {
    // `MentionSelector` runs FIRST, ahead of `SoleAgentSelector`, and raises
    // `UnknownRoomParticipantError` on a name it cannot resolve. The hint used to read
    // "scout will answer" here -- a promise for a message nobody ever answers.
    expect(answerHint(roster('scout'), '@phantm help', '', NO_CEILING)).toBe(
      'No one here is called @phantm, so this message will not be answered.',
    );
  });

  it('honours the address whatever case it is written in', () => {
    expect(answerHint(roster('scout', 'dba'), '@DBA look', 'scout', NO_CEILING)).toBe(
      'dba will answer',
    );
  });

  it('honours an alias, which the Core resolves after ids', () => {
    const participants: RoomParticipant[] = [
      { id: 'user', kind: 'human' },
      { id: 'scout', kind: 'agent' },
      { id: 'database_reviewer', kind: 'agent', display_name: 'DBA', aliases: ['dba'] },
    ];

    expect(answerHint(participants, '@dba look', '', NO_CEILING)).toBe('DBA will answer');
  });

  it('answers every addressed agent once, in the order they were named', () => {
    expect(answerHint(roster('a', 'b', 'c'), '@b then @a', '', NO_CEILING)).toBe(
      'b, a will answer, in that order',
    );
  });

  it('counts a repeated address once, as the Core serves it once', () => {
    expect(answerHint(roster('a', 'b'), '@a and @a again', '', NO_CEILING)).toBe('a will answer');
  });

  it('does not promise more answers than the turn ceiling allows', () => {
    // `max_agent_turns_per_human_message` bounds the whole cascade, so a fourth address
    // under a ceiling of three never gets the floor. Naming all four claimed a turn the
    // ceiling had already refused.
    expect(answerHint(roster('a', 'b', 'c', 'd'), '@a @b @c @d', '', 3)).toBe(
      'a, b, c will answer, in that order. Paused after 3 replies, so the rest of those you named will not.',
    );
  });

  it('does not build a regex from a participant id', () => {
    const participants: RoomParticipant[] = [
      { id: 'user', kind: 'human' },
      { id: 'a(', kind: 'agent' },
    ];

    expect(() => answerHint(participants, 'hello', '', NO_CEILING)).not.toThrow();
    expect(answerHint(participants, 'hello', '', NO_CEILING)).toBe('a( will answer');
  });
});

describe('mentionCandidates / mentionUnderCaret', () => {
  it('offers this conversation`s agents, filtered by what is typed', () => {
    const participants: RoomParticipant[] = [
      { id: 'user', kind: 'human' },
      { id: 'scout', kind: 'agent' },
      { id: 'critic', kind: 'agent' },
    ];

    expect(mentionCandidates(participants, '').map((p) => p.id)).toEqual(['scout', 'critic']);
    expect(mentionCandidates(participants, 'sc').map((p) => p.id)).toEqual(['scout']);
    expect(mentionCandidates(participants, 'zz')).toEqual([]);
  });

  it('matches an alias too, so a remembered short name still completes', () => {
    const participants: RoomParticipant[] = [
      { id: 'database_reviewer', kind: 'agent', aliases: ['dba'] },
    ];

    expect(mentionCandidates(participants, 'db').map((p) => p.id)).toEqual(['database_reviewer']);
  });

  it('finds the token the caret sits in, and where it starts', () => {
    expect(mentionUnderCaret('ask @sc', 7)).toEqual({ start: 4, prefix: 'sc' });
    expect(mentionUnderCaret('@', 1)).toEqual({ start: 0, prefix: '' });
  });

  it('offers nothing when the caret is not in an address', () => {
    expect(mentionUnderCaret('ask scout', 9)).toBeNull();
    expect(mentionUnderCaret('me@example.com', 14)).toBeNull();
  });
});

/**
 * The room's own payloads, not ones written here.
 *
 * The accumulation test used to hand-write a `seq` onto `generating` and `streaming`
 * events -- a field the room sends on neither -- and a fold keyed on that absent field
 * replaced the live bubble on every delta while the test stayed green over an input the
 * server cannot produce. This file is generated from `RoomOrchestrator` and pinned to it
 * by `test_the_heads_payload_fixture_is_what_the_room_publishes`, so a payload change on
 * the Core side fails a test instead of passing these by.
 */
const published = publishedRoomEvents.events as RoomTopicEvent[];

/** The published reply for `turnId` with `status`; the `n`th when a turn has several. */
const eventOf = (turnId: string, status: RoomReplyPayload['status'], n = 0): RoomTopicEvent => {
  const found = published.filter(
    (e) =>
      e.event_type === 'AGENT_REPLY' &&
      'turn_id' in e.payload &&
      e.payload.turn_id === turnId &&
      e.payload.status === status,
  )[n];
  if (!found) throw new Error(`the fixture has no ${status} #${n} for ${turnId}`);
  return found;
};

/** The one published event of `eventType` -- or, for a reply, with `status` -- and no other. */
const onlyEvent = (match: (e: RoomTopicEvent) => boolean, what: string): RoomTopicEvent => {
  const found = published.filter(match);
  if (found.length !== 1) throw new Error(`the fixture has ${found.length} ${what}, not one`);
  return found[0];
};

const typing = () => onlyEvent((e) => e.event_type === 'USER_INPUT', 'composing notices');
const interrupt = () => onlyEvent((e) => e.event_type === 'INTERRUPT', 'interrupts');
const cascadeFailure = () =>
  onlyEvent(
    (e) => e.event_type === 'AGENT_REPLY' && e.payload.status === 'error',
    'cascade failures',
  );

const fold = (events: RoomTopicEvent[], from: RoomLiveState = EMPTY_LIVE): RoomLiveState =>
  events.reduce(applyRoomEvent, from);

describe('seededTitle', () => {
  // New opens a conversation without asking for a name -- F1's "no modal, no title
  // prompt" -- and the first message names it, once. These are the rules for "once".
  const fresh = (over: Partial<RoomState> = {}): RoomState => ({
    room_id: 'r1',
    title: NEW_CONVERSATION_TITLE,
    participants: roster('scout'),
    transcript: [
      { seq: 1, sender_id: 'scout', content: '', kind: 'join', created_at: '', completed: true },
    ],
    turn_state: { agent_turns_since_human: 0 },
    policy: {
      max_agent_turns_per_human_message: 3,
      max_span_messages: 40,
      transcript_window: 15,
      hesitation_seconds: 0,
      default_responder_id: '',
    },
    ...over,
  });

  it('names a new conversation after its first message', () => {
    expect(seededTitle(fresh(), 'check the index on users')).toBe('check the index on users');
  });

  it('never renames a conversation someone already named', () => {
    expect(seededTitle(fresh({ title: 'Index tuning' }), 'check the index')).toBeNull();
  });

  it('never renames a conversation once something has been said in it', () => {
    const spoken = fresh({
      transcript: [
        { seq: 1, sender_id: 'user', content: 'hi', kind: 'utterance', created_at: '', completed: true },
      ],
    });

    expect(seededTitle(spoken, 'check the index')).toBeNull();
  });

  it('takes one line, without the characters the Core refuses in a title', () => {
    // `_clean_title` refuses newlines and ANSI escapes; a seed carrying either is a rename
    // the Core answers with 400.
    expect(seededTitle(fresh(), '  \u001b[31mcheck\u001b[0m   the\tindex\nand then the rest')).toBe(
      'check the index',
    );
  });

  it('shortens a long first message at a word', () => {
    const seed = seededTitle(fresh(), 'why does the composite index on tenant and created at get ignored by the planner');

    expect(seed).toBe('why does the composite index on tenant and created at get…');
    expect(seed!.length).toBeLessThanOrEqual(60);
  });

  it('seeds nothing from a message with nothing to name it by', () => {
    expect(seededTitle(fresh(), '   \n  ')).toBeNull();
  });
});

describe('applyRoomEvent', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('opens a live turn when an agent takes the floor', () => {
    const live = applyRoomEvent(EMPTY_LIVE, eventOf('turn-1', 'generating'));

    expect(live.turn).toEqual({ agentId: 'scout', turnId: 'turn-1', text: '' });
  });

  it('updates live turn status when a status_update event arrives', () => {
    const started = applyRoomEvent(EMPTY_LIVE, {
      event_type: 'AGENT_REPLY',
      payload: { room_id: 'r1', agent_id: 'scout', turn_id: 'turn-1', status: 'generating' },
    });
    expect(started.turn?.statusText).toBeUndefined();

    const updated = applyRoomEvent(started, {
      event_type: 'AGENT_REPLY',
      payload: {
        room_id: 'r1',
        agent_id: 'scout',
        turn_id: 'turn-1',
        status: 'status_update',
        detail: 'Running tool: search...',
      },
    });
    expect(updated.turn?.statusText).toBe('Running tool: search...');
    expect(updated.turn?.text).toBe('');
  });

  it('accumulates every delta of a turn, as the room publishes them', () => {
    // The shipped defect: the delta was matched against the event's `seq` rather than
    // against the turn it belongs to, so every delta opened a bubble of its own.
    // Killed by: frontend/src/lib/rooms.ts :: current.turnId !== reply.turn_id) {
    // Becomes: current.turnId !== String((reply as { seq?: number }).seq ?? 0)) {
    // Everything up to the landed row: the floor taken and three deltas.
    const live = fold(published.slice(0, published.indexOf(eventOf('turn-1', 'final'))));

    expect(live.turn?.agentId).toBe('scout');
    expect(live.turn?.text).toBe('scout says TTL');
  });

  it('accumulates a turn whose start it never saw', () => {
    // A head that opens the conversation mid-turn receives deltas with no `generating`
    // before them. The first opens the bubble; the rest must still append to it.
    const live = fold([eventOf('turn-1', 'streaming', 0), eventOf('turn-1', 'streaming', 1)]);

    expect(live.turn?.text).toBe('scout says ');
  });

  it('clears the live turn when that turn lands, because no event says the room is done', () => {
    const live = fold(published.slice(0, published.indexOf(eventOf('turn-1', 'final')) + 1));

    expect(live.turn).toBeNull();
  });

  it('follows the room from one speaker to the next and ends with nothing live', () => {
    const critic = fold(published.slice(0, published.indexOf(eventOf('turn-2', 'final'))));
    expect(critic.turn?.agentId).toBe('critic');
    expect(critic.turn?.text).toBe('critic disagrees');

    expect(fold(published).turn).toBeNull();
  });

  it('does not append one turn`s delta to another turn`s bubble', () => {
    // Killed by: frontend/src/lib/rooms.ts :: if (!current || current.turnId !== reply.turn_id) {
    // Becomes: if (!current) {
    // turn-1 is still live when turn-2's first delta arrives -- its `final` was never
    // received. Appending would put critic's words under scout's name.
    const live = fold([
      eventOf('turn-1', 'generating'),
      eventOf('turn-1', 'streaming', 0),
      eventOf('turn-2', 'streaming', 0),
    ]);

    expect(live.turn?.agentId).toBe('critic');
    expect(live.turn?.text).toBe('critic ');
  });

  it('leaves a live turn alone when a different turn lands', () => {
    // Killed by: frontend/src/lib/rooms.ts :: if (live.turn && live.turn.turnId === reply.turn_id) return
    // Becomes: if (live.turn) return
    const live = fold([eventOf('turn-2', 'generating'), eventOf('turn-1', 'final')]);

    expect(live.turn).toEqual({ agentId: 'critic', turnId: 'turn-2', text: '' });
  });

  it('reports a cascade that stopped before anyone spoke', () => {
    // Killed by: frontend/src/lib/rooms.ts :: error: reply.error ?? 'This conversation stopped.'
    // Becomes: error: 'This conversation stopped.'
    // What `ui/rooms.py`'s `_announce_failure` published when an address named nobody: no
    // agent, no turn -- taken from the fixture, where it used to be written out here.
    const failure = cascadeFailure();
    const live = applyRoomEvent(EMPTY_LIVE, failure);

    expect(failure.payload).not.toHaveProperty('turn_id');
    expect(live.turn).toBeNull();
    expect(live.error).toContain('@phantm');
  });

  it('ends a turn on a cascade failure even while one is open', () => {
    const live = fold([eventOf('turn-1', 'generating'), cascadeFailure()]);

    expect(live.turn).toBeNull();
    expect(live.error).toContain('@phantm');
  });

  it('recognises every event the room publishes, and reports none of them as unknown', () => {
    // Killed by: frontend/src/lib/rooms.ts :: evt.payload.status === 'typing' ? live
    // Becomes: evt.payload.status === ('composing' as string) ? live
    // The topic carries the human's composing notice and Stop's interrupt beside the
    // replies; a head whose event type named only the replies read both as landed rows.
    const reported = vi.spyOn(console, 'error').mockImplementation(() => {});

    fold(published);

    expect(reported).not.toHaveBeenCalled();
  });

  it('leaves the live turn alone when the human is composing', () => {
    // Killed by: frontend/src/lib/rooms.ts :: evt.payload.status === 'typing' ? live :
    // Becomes: evt.payload.status === 'typing' ? EMPTY_LIVE :
    const open = fold([eventOf('turn-1', 'generating'), eventOf('turn-1', 'streaming', 0)]);

    expect(applyRoomEvent(open, typing())).toBe(open);
  });

  it('leaves a stopped turn to the landed row the stop produces', () => {
    // Killed by: frontend/src/lib/rooms.ts :: case 'INTERRUPT':
    // Becomes: case 'INTERRUPT': return { turn: null, error: live.error };
    // Stop publishes INTERRUPT and cancels the turn; the cancelled turn is then recorded and
    // lands like any other, `completed: false`, and that row is what clears the bubble. A
    // Stop with no turn running produces no row and has no bubble to clear.
    const writing = fold([eventOf('turn-3', 'generating'), interrupt()]);
    expect(writing.turn).toEqual({ agentId: 'scout', turnId: 'turn-3', text: '' });

    const landed = applyRoomEvent(writing, eventOf('turn-3', 'final'));
    expect(landed.turn).toBeNull();
    expect(eventOf('turn-3', 'final').payload).toMatchObject({ completed: false });
  });

  it('refuses to read an unrecognised status as a landed reply, and says so', () => {
    // Killed by: frontend/src/lib/rooms.ts :: return ignoreUnrecognised(live, reply);
    // Becomes: return live.turn && live.turn.turnId === (reply as { turn_id?: string }).turn_id ? { turn: null, error: live.error } : live;
    // The fold used to treat any status it did not name as `final`, so a status the Core
    // added would silently end whichever bubble shared its turn.
    const reported = vi.spyOn(console, 'error').mockImplementation(() => {});
    const open = fold([eventOf('turn-1', 'generating'), eventOf('turn-1', 'streaming', 0)]);
    const unknown = {
      event_type: 'AGENT_REPLY',
      payload: { ...eventOf('turn-1', 'final').payload, status: 'retracted' },
    } as unknown as RoomTopicEvent;

    expect(applyRoomEvent(open, unknown)).toBe(open);
    expect(reported).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(reported.mock.calls[0])).toContain('retracted');
  });

  it('refuses an event type the room topic is not known to carry, and says so', () => {
    // Killed by: frontend/src/lib/rooms.ts :: return ignoreUnrecognised(live, evt);
    // Becomes: return live;
    const reported = vi.spyOn(console, 'error').mockImplementation(() => {});
    const open = fold([eventOf('turn-1', 'generating')]);
    const unknown = {
      event_type: 'ROOM_FLOOR_YIELDED',
      payload: { room_id: 'r1' },
    } as unknown as RoomTopicEvent;

    expect(applyRoomEvent(open, unknown)).toBe(open);
    expect(reported).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(reported.mock.calls[0])).toContain('ROOM_FLOOR_YIELDED');
  });
});

// ---------------------------------------------------------------------------------
// The HTTP client. Ten calls, and the refusal-reading they share.
// ---------------------------------------------------------------------------------

interface FakeResponse {
  ok: boolean;
  status: number;
  statusText: string;
  json: () => Promise<unknown>;
}

const ok = (body: unknown, status = 200): FakeResponse => ({
  ok: true,
  status,
  statusText: 'OK',
  json: async () => body,
});

const noContent = (): FakeResponse => ({
  ok: true,
  status: 204,
  statusText: 'No Content',
  json: async () => {
    throw new Error('a 204 has no body to read');
  },
});

const refusal = (status: number, statusText: string, body?: unknown): FakeResponse => ({
  ok: false,
  status,
  statusText,
  json: async () => {
    if (body === undefined) throw new SyntaxError('Unexpected end of JSON input');
    return body;
  },
});

const stubFetch = (...responses: FakeResponse[]) => {
  const mock = vi.fn();
  for (const response of responses) mock.mockResolvedValueOnce(response);
  vi.stubGlobal('fetch', mock);
  return mock;
};

/** The URL, method and parsed body of the nth `fetch` call. */
const callOf = (mock: ReturnType<typeof vi.fn>, index = 0) => {
  const [url, init] = mock.mock.calls[index] as [string, RequestInit | undefined];
  return {
    url,
    method: init?.method ?? 'GET',
    body: init?.body === undefined ? undefined : JSON.parse(String(init.body)),
  };
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('readOrThrow', () => {
  it('keeps the server`s own reason, which is the only remedy the surface has', async () => {
    stubFetch(refusal(400, 'Bad Request', { detail: "Room 'r1' has no participant matching @phantm" }));

    await expect(roomsApi.get('r1')).rejects.toThrow(
      "Room 'r1' has no participant matching @phantm",
    );
  });

  it('falls back to the status line when a refusal carries no JSON', async () => {
    stubFetch(refusal(502, 'Bad Gateway'));

    await expect(roomsApi.get('r1')).rejects.toThrow('502 Bad Gateway');
  });

  it('does not try to read a body out of a 204', async () => {
    const mock = stubFetch(noContent());

    await expect(roomsApi.typing('r1')).resolves.toBeUndefined();
    expect(callOf(mock)).toMatchObject({ url: '/api/rooms/r1/typing', method: 'POST' });
  });

  it('reports a refused typing report rather than resolving', async () => {
    stubFetch(refusal(404, 'Not Found', { detail: "Room 'gone' does not exist" }));

    await expect(roomsApi.typing('gone')).rejects.toThrow("Room 'gone' does not exist");
  });
});

describe('roomsApi', () => {
  it('lists conversations out of the envelope the route returns', async () => {
    const mock = stubFetch(ok({ rooms: [{ room_id: 'r1' }], unreadable: ['r_bad'] }));

    await expect(roomsApi.list()).resolves.toEqual({
      rooms: [{ room_id: 'r1' }],
      unreadable: ['r_bad'],
    });
    expect(callOf(mock)).toMatchObject({ url: '/api/rooms', method: 'GET' });
  });

  it('creates a conversation with its title and roster', async () => {
    const mock = stubFetch(ok({ room_id: 'r2' }, 201));

    await expect(roomsApi.create('Index tuning', ['scout'])).resolves.toEqual({ room_id: 'r2' });
    expect(callOf(mock)).toEqual({
      url: '/api/rooms',
      method: 'POST',
      body: { title: 'Index tuning', agent_ids: ['scout'] },
    });
  });

  it('reads one conversation by id', async () => {
    const mock = stubFetch(ok({ room_id: 'r1' }));

    await expect(roomsApi.get('r1')).resolves.toEqual({ room_id: 'r1' });
    expect(callOf(mock)).toMatchObject({ url: '/api/rooms/r1', method: 'GET' });
  });

  it('renames a conversation with PATCH, not by recreating it', async () => {
    const mock = stubFetch(ok({ room_id: 'r1', title: 'Renamed' }));

    await roomsApi.rename('r1', 'Renamed');
    expect(callOf(mock)).toEqual({
      url: '/api/rooms/r1',
      method: 'PATCH',
      body: { title: 'Renamed' },
    });
  });

  it('deletes a conversation with DELETE and resolves on the 204 without reading a body', async () => {
    const mock = stubFetch(noContent());

    await expect(roomsApi.delete('r1')).resolves.toBeUndefined();
    expect(callOf(mock)).toEqual({ url: '/api/rooms/r1', method: 'DELETE', body: undefined });
  });

  it('posts a message and returns the seq the route answered with', async () => {
    const mock = stubFetch(ok({ room_id: 'r1', seq: 7 }, 202));

    await expect(roomsApi.send('r1', 'check the index')).resolves.toMatchObject({ seq: 7 });
    expect(callOf(mock)).toEqual({
      url: '/api/rooms/r1/messages',
      method: 'POST',
      body: { content: 'check the index' },
    });
  });

  it('seats an agent through the participants route', async () => {
    const mock = stubFetch(ok({ room_id: 'r1' }));

    await roomsApi.addAgent('r1', 'dba');
    expect(callOf(mock)).toEqual({
      url: '/api/rooms/r1/participants',
      method: 'POST',
      body: { agent_id: 'dba' },
    });
  });

  it('removes a participant by id, with DELETE', async () => {
    const mock = stubFetch(ok({ room_id: 'r1' }));

    await roomsApi.removeParticipant('r1', 'dba');
    expect(callOf(mock)).toMatchObject({
      url: '/api/rooms/r1/participants/dba',
      method: 'DELETE',
    });
  });

  it('takes the floor back through the stop route', async () => {
    const mock = stubFetch(ok({ room_id: 'r1' }));

    await roomsApi.stop('r1');
    expect(callOf(mock)).toMatchObject({ url: '/api/rooms/r1/stop', method: 'POST' });
  });

  it('runs a failed turn again through the retry route', async () => {
    const mock = stubFetch(ok({ room_id: 'r1' }));

    await roomsApi.retry('r1');
    expect(callOf(mock)).toMatchObject({ url: '/api/rooms/r1/retry', method: 'POST' });
  });

  it('carries a Core refusal out of every write, not just the reads', async () => {
    const reasons = [
      [() => roomsApi.create('', []), 'a conversation needs a title'],
      [() => roomsApi.send('r1', ''), 'a message needs content'],
      [() => roomsApi.addAgent('r1', 'ghost'), "'ghost' cannot be resolved to an agent"],
      [() => roomsApi.stop('r1'), 'nothing is running in this conversation'],
      [() => roomsApi.retry('r1'), 'there is no failed turn to run again'],
      [() => roomsApi.removeParticipant('r1', 'user'), 'a conversation keeps its human'],
      [() => roomsApi.rename('r1', ''), 'a conversation needs a title'],
      [() => roomsApi.delete('gone'), "No room 'gone' in the store: it has been deleted"],
    ] as const;

    for (const [call, detail] of reasons) {
      stubFetch(refusal(400, 'Bad Request', { detail }));
      await expect(call()).rejects.toThrow(detail);
      vi.unstubAllGlobals();
    }
  });
});

describe('RoomReadOrder (#1412)', () => {
  // Killed by: frontend/src/lib/rooms.ts :: this.latest.set(roomId, this.issued);
  // Becomes: if (!this.latest.has(roomId)) this.latest.set(roomId, this.issued);
  it('lands a later read that resolves first, and drops the earlier one when it resolves late', () => {
    const reads = new RoomReadOrder();
    const sendsReread = reads.issue('r1', 3);
    const finalsReread = reads.issue('r1', 3);
    expect(reads.isStale(finalsReread, 3, 'r1')).toBe(false);
    expect(reads.isStale(sendsReread, 3, 'r1')).toBe(true);
  });

  // Killed by: frontend/src/lib/rooms.ts :: return this.latest.get(ticket.roomId) !== ticket.seq;
  // Becomes: return false;
  it('drops an earlier read that resolves before a later one it was overtaken by', () => {
    // The send's re-read went out before the stream's `final`, and resolves between that
    // `final` and the read `final` fired: the newest answer so far, but not the newest
    // question, and landing it takes the reply off screen until the later read arrives.
    const reads = new RoomReadOrder();
    const sendsReread = reads.issue('r1', 3);
    reads.issue('r1', 3);
    expect(reads.isStale(sendsReread, 3, 'r1')).toBe(true);
  });

  it('lands the only read that is out', () => {
    const reads = new RoomReadOrder();
    const only = reads.issue('r1', 3);
    expect(reads.isStale(only, 3, 'r1')).toBe(false);
  });

  // Killed by: frontend/src/lib/rooms.ts :: roomResponseIsStale(ticket.generation, currentGeneration, ticket.roomId, openRoomId) ||
  // Becomes: false ||
  it('still drops a read that a Clear retired, and lands the read issued after it', () => {
    const reads = new RoomReadOrder();
    const beforeClear = reads.issue('r1', 3);
    // The Clear commits through `commitRoomAct`, which bumps the generation to 4.
    expect(reads.isStale(beforeClear, 4, 'r1')).toBe(true);
    const afterClear = reads.issue('r1', 4);
    expect(reads.isStale(afterClear, 4, 'r1')).toBe(false);
    expect(reads.isStale(beforeClear, 4, 'r1')).toBe(true);
  });

  it('orders each room on its own, and still drops a read of a room no longer open', () => {
    const reads = new RoomReadOrder();
    const a = reads.issue('a', 1);
    const b = reads.issue('b', 1);
    expect(reads.overtaken(a)).toBe(false);
    expect(reads.overtaken(b)).toBe(false);
    expect(reads.isStale(a, 1, 'b')).toBe(true);
    expect(reads.isStale(b, 1, 'b')).toBe(false);
  });
});

describe('roomResponseIsStale', () => {
  it('accepts a reply for the conversation that is still open', () => {
    expect(roomResponseIsStale(4, 4, 'r1', 'r1')).toBe(false);
  });

  it('drops a reply the user has already navigated away from', () => {
    // `openRoom(A)` then `openRoom(B)`: A's record resolving second used to leave A's
    // transcript on screen while the open id was B, so the next Send posted into B.
    expect(roomResponseIsStale(4, 5, 'r1', 'r1')).toBe(true);
  });

  it('drops a reply for a conversation other than the open one', () => {
    expect(roomResponseIsStale(4, 4, 'r1', 'r2')).toBe(true);
  });

  it('drops a reply that arrives after the single-agent surface was returned to', () => {
    expect(roomResponseIsStale(4, 4, 'r1', null)).toBe(true);
  });
});

describe('unansweredSilence', () => {
  /** The silence the chain records when it has nobody left to give the floor to. */
  const exhausted: RoomSpeakerDecision = {
    verdict: 'silence',
    speaker_id: null,
    confidence: 1,
    selector: 'orchestrator',
    reasoning: 'the selector chain was exhausted without a judgement',
  };

  const said = (
    seq: number,
    sender_id: string,
    over: Partial<RoomTranscriptMessage> = {},
  ): RoomTranscriptMessage => ({
    seq,
    sender_id,
    content: `row ${seq}`,
    kind: 'utterance',
    created_at: '',
    completed: true,
    ...over,
  });

  /**
   * The rows the room's own `final` events land as, after the human message at seq 1.
   *
   * Taken from the generated fixture rather than written here, so the agent rows carry the
   * `seq` and `agent_id` the Core actually publishes for a landed reply.
   */
  const landed = (turnIds: string[]): RoomTranscriptMessage[] =>
    turnIds.map((turnId) => {
      const evt = eventOf(turnId, 'final');
      if (evt.event_type !== 'AGENT_REPLY' || evt.payload.status !== 'final') {
        throw new Error(`the fixture's final for ${turnId} is not a landed reply`);
      }
      return said(evt.payload.seq, evt.payload.agent_id, { content: evt.payload.content });
    });

  const settled = (participants: RoomParticipant[], transcript: RoomTranscriptMessage[]) => ({
    participants,
    transcript,
    last_decision: exhausted,
  });

  it('reports a silence that follows the human message with nothing after it', () => {
    expect(unansweredSilence(settled(roster('scout', 'critic'), [said(1, 'user')]))).toBe(
      exhausted,
    );
  });

  it('reports nothing once the only agent has answered (#920)', () => {
    // Killed by: frontend/src/lib/rooms.ts :: message.rendered_through === undefined ||
    // Becomes: false ||
    const transcript = [said(1, 'user'), ...landed(['turn-1'])];

    expect(unansweredSilence(settled(roster('scout'), transcript))).toBeNull();
  });

  it('reports nothing when a legacy row arrives with rendered_through 0, not undefined (B1)', () => {
    // A room stored before this field existed carries `rendered_through: 0` on every such
    // row on the wire (`room/models.py`'s default, serialised by `model_dump`) -- never as
    // an absent/`undefined` key, which is all the `landed()` fixture above exercises.
    // Treating that `0` as evidence against an answer, rather than as the absence of
    // evidence it is documented to be, reopens #920 for exactly the histories this fix
    // protects: a conversation ending human -> answer -> silence.
    // Killed by: frontend/src/lib/rooms.ts :: message.rendered_through === 0
    // Becomes: false
    const transcript = [said(1, 'user'), said(2, 'scout', { rendered_through: 0 })];

    expect(unansweredSilence(settled(roster('scout'), transcript))).toBeNull();
  });

  it('reports nothing once several agents have answered', () => {
    const transcript = [said(1, 'user'), ...landed(['turn-1', 'turn-2'])];

    expect(unansweredSilence(settled(roster('scout', 'critic'), transcript))).toBeNull();
  });

  it('reports the latest message going unanswered even when an earlier one was answered', () => {
    const transcript = [said(1, 'user'), ...landed(['turn-1']), said(3, 'user')];

    expect(unansweredSilence(settled(roster('scout', 'critic'), transcript))).toBe(exhausted);
  });

  it('does not take someone joining for an answer', () => {
    // Killed by: frontend/src/lib/rooms.ts :: room.transcript.filter((message) => message.kind === 'utterance')
    // Becomes: room.transcript.filter(() => true)
    // A message sent into a conversation nobody was in, and then someone added: the join
    // is the room's bookkeeping, and nobody has said anything back.
    const transcript = [said(1, 'user'), said(2, 'scout', { kind: 'join', content: '' })];

    expect(unansweredSilence(settled(roster('scout'), transcript))).toBe(exhausted);
  });

  it('counts a turn that failed as an answer, because that row says so itself', () => {
    const transcript = [
      said(1, 'user'),
      said(2, 'scout', { error: 'provider refused', completed: false }),
    ];

    expect(unansweredSilence(settled(roster('scout'), transcript))).toBeNull();
  });

  it('does not take an agent that has since left for the human', () => {
    // Killed by: frontend/src/lib/rooms.ts :: senderOf(room.participants, utterances[i].sender_id)?.kind === 'human'
    // Becomes: senderOf(room.participants, utterances[i].sender_id)?.kind !== 'agent'
    const transcript = [said(1, 'user'), ...landed(['turn-1'])];

    expect(unansweredSilence(settled(roster(), transcript))).toBeNull();
  });

  it('does not count a reply that landed too late to have seen the message it answers (#945)', () => {
    // A reply already in flight when a second human message lands can complete afterward and
    // land after it, looking like an answer to the newer message when the turn that produced
    // it started before that message existed and never saw it.
    // Killed by: frontend/src/lib/rooms.ts :: message.rendered_through >= seq
    // Becomes: true
    const transcript = [said(1, 'user'), said(2, 'user'), said(3, 'scout', { rendered_through: 1 })];

    expect(unansweredSilence(settled(roster('scout'), transcript))).toBe(exhausted);
  });

  it('counts a reply that rendered through the message it answers, even sent after a later one', () => {
    const transcript = [said(1, 'user'), said(2, 'user'), said(3, 'scout', { rendered_through: 2 })];

    expect(unansweredSilence(settled(roster('scout'), transcript))).toBeNull();
  });

  it('reports nothing when the last decision gave somebody the floor, or there is none', () => {
    const transcript = [said(1, 'user')];
    const speak: RoomSpeakerDecision = { ...exhausted, verdict: 'speak', speaker_id: 'scout' };

    const answered = settled(roster('scout'), transcript);

    expect(unansweredSilence({ ...answered, last_decision: speak })).toBeNull();
    expect(unansweredSilence({ ...answered, last_decision: null })).toBeNull();
  });
});

describe('roomsApi, the retirement slice`s four routes', () => {
  // Killed by: frontend/src/lib/rooms.ts :: readOrThrow(await ask(`/api/rooms/${roomId}/context`)),
  // Becomes: readOrThrow(await ask(`/api/rooms/${roomId}`)),
  it('asks how full a conversation is on its own route, not on the one re-read every turn', async () => {
    const mock = stubFetch(ok({ room_id: 'r1', seats: [], is_saturated: false, saturation_threshold: 20 }));

    await expect(roomsApi.context('r1')).resolves.toMatchObject({ saturation_threshold: 20 });
    expect(callOf(mock)).toMatchObject({ url: '/api/rooms/r1/context', method: 'GET' });
  });

  // Killed by: frontend/src/lib/rooms.ts :: await ask(`/api/rooms/${roomId}/compact`, {
  // Becomes: await ask(`/api/rooms/${roomId}/history`, {
  it('shortens the whole conversation, naming no seat', async () => {
    // The per-seat half of this case went with the `participantId` argument (#1230): the
    // head never passed it, so what the second stub proved was that a test could call it.
    // The route still takes `participant_id` and `tests/unit/test_ui_room_api.py` posts it;
    // what is checked here is the one body the head actually sends.
    const roomWide = stubFetch(ok({ room_id: 'r1', results: [] }));
    await roomsApi.compact('r1');
    expect(callOf(roomWide)).toEqual({ url: '/api/rooms/r1/compact', method: 'POST', body: {} });
  });

  // Killed by: frontend/src/lib/rooms.ts :: body: JSON.stringify({ seq }),
  // Becomes: body: JSON.stringify({ index: seq }),
  it('rewinds by the message`s seq, which is the key the route refuses anything else on', async () => {
    const mock = stubFetch(ok({ room_id: 'r1', participants_not_reset: [] }));

    await roomsApi.truncateHistory('r1', 7);
    expect(callOf(mock)).toEqual({
      url: '/api/rooms/r1/history/truncate',
      method: 'POST',
      body: { seq: 7 },
    });
  });

  // Killed by: frontend/src/lib/rooms.ts :: await ask(`/api/rooms/${roomId}/history`, { method: 'DELETE' })
  // Becomes: await ask(`/api/rooms/${roomId}`, { method: 'DELETE' })
  it('empties a conversation without deleting it, and keeps what the cut missed', async () => {
    // `DELETE /api/rooms/{id}` removes the conversation; this one empties the conversation
    // you are in. Two routes, because they are two operations.
    const mock = stubFetch(ok({ room_id: 'r1', participants_not_reset: ['scout'] }));

    await expect(roomsApi.clearHistory('r1')).resolves.toMatchObject({
      participants_not_reset: ['scout'],
    });
    expect(callOf(mock)).toMatchObject({ url: '/api/rooms/r1/history', method: 'DELETE' });
  });
});

describe('roomReadReason (#1389)', () => {
  // Killed by: frontend/src/lib/rooms.ts :: if (err instanceof RoomsApiError) reason = err.coreDetail ?? noReason;
  // Becomes: if (err instanceof RoomsApiError) reason = err.message;
  it("passes the Core's own refusal through, and nothing else it did not write", () => {
    expect(roomReadReason(new RoomsApiError("No room 'r2' in the store.", 404, "No room 'r2' in the store."))).toBe(
      "No room 'r2' in the store.",
    );
    expect(roomReadReason(new RoomsApiError('500 Internal Server Error', 500, null))).toBe(ROOM_READ_NO_REASON);
  });

  // Killed by: frontend/src/lib/rooms.ts :: else if (err instanceof RoomsNoAnswerError) reason = ROOM_READ_NO_ANSWER;
  // Becomes: else if (err instanceof RoomsNoAnswerError) reason = ROOM_APP_FAULT;
  it('says the service did not answer when the fetch itself failed', () => {
    expect(roomReadReason(new RoomsNoAnswerError(new TypeError('Failed to fetch')))).toBe(ROOM_READ_NO_ANSWER);
    expect(roomReadReason(new RoomsNoAnswerError(new TypeError('Load failed')))).toBe(ROOM_READ_NO_ANSWER);
  });

  // A `TypeError` is also what the head's own bugs throw, and the service did nothing (#1441).
  // Killed by: frontend/src/lib/rooms.ts :: else reason = ROOM_APP_FAULT;
  // Becomes: else reason = ROOM_READ_NO_ANSWER;
  it("does not blame the service for the app's own fault", () => {
    for (const fault of [new TypeError("Cannot read properties of undefined (reading 'seq')"), 'something odd']) {
      const reason = roomReadReason(fault);
      expect(reason).toBe(ROOM_APP_FAULT);
      expect(reason).not.toMatch(/service/);
      expectPlain(reason);
    }
  });

  // Killed by: frontend/src/lib/rooms.ts :: return /[.!?…]$/.test(reason) ? reason : `${reason}.`;
  // Becomes: return reason;
  it('always ends a sentence, so the copy after it does not run on', () => {
    expect(roomReadReason(new RoomsApiError('Room is locked', 409, 'Room is locked'))).toBe('Room is locked.');
  });

  it('keeps the Core detail and the status on what roomsApi throws', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.reject(new SyntaxError('not JSON')),
      })),
    );
    const err = await roomsApi.get('r2').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(RoomsApiError);
    expect((err as RoomsApiError).message).toBe('500 Internal Server Error');
    expect((err as RoomsApiError).coreDetail).toBeNull();
    expect(roomReadReason(err)).toBe(ROOM_READ_NO_REASON);
    vi.unstubAllGlobals();
  });
});

describe('roomFailureReason (#1411)', () => {
  afterEach(() => vi.unstubAllGlobals());

  /** What `roomsApi.compact` rejects with when `fetch` answers `answer` (or rejects with it). */
  const compactFailure = async (answer: () => Promise<unknown>): Promise<unknown> => {
    vi.stubGlobal('fetch', vi.fn(answer));
    return roomsApi.compact('r1').catch((e: unknown) => e);
  };

  // Killed by: frontend/src/lib/rooms.ts :: return roomFailureReason(err, ROOM_READ_NO_REASON);
  // Becomes: return roomFailureReason(err);
  it('says a read failed and an act failed in different words when the Core gave no reason', () => {
    const bare = new RoomsApiError('500 Internal Server Error', 500, null);
    expect(roomReadReason(bare)).toBe(ROOM_READ_NO_REASON);
    expect(roomFailureReason(bare)).toBe(ROOM_ACTION_NO_REASON);
  });

  // Killed by: frontend/src/lib/rooms.ts :: throw new RoomsNoAnswerError(err);
  // Becomes: throw err;
  it('replaces a fetch that never got an answer with a plain sentence', async () => {
    const err = await compactFailure(() => Promise.reject(new TypeError('Failed to fetch')));
    expect(err).toBeInstanceOf(RoomsNoAnswerError);
    const reason = roomFailureReason(err);
    expect(reason).toBe(ROOM_READ_NO_ANSWER);
    expectPlain(reason);
  });

  it('replaces a bodyless 500 with a plain sentence rather than its status line', async () => {
    const err = await compactFailure(async () => ({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: () => Promise.reject(new SyntaxError('Unexpected end of JSON input')),
    }));
    expect((err as Error).message).toBe('500 Internal Server Error');
    const reason = roomFailureReason(err);
    expect(reason).toBe(ROOM_ACTION_NO_REASON);
    expectPlain(reason);
  });

  // Killed by: frontend/src/lib/rooms.ts :: if (typeof body?.detail === 'string' && body.detail.trim() !== '') detail = body.detail;
  // Becomes: if (body?.detail) detail = String(body.detail);
  it("does not take FastAPI's list-shaped validation detail for the Core's words", async () => {
    // What FastAPI answers for a request body it could not validate: a list of objects.
    const err = await compactFailure(async () => ({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Entity',
      json: async () => ({ detail: [{ loc: ['body', 'seq'], msg: 'Field required', type: 'missing' }] }),
    }));
    expect((err as RoomsApiError).coreDetail).toBeNull();
    const reason = roomFailureReason(err);
    expect(reason).toBe(ROOM_ACTION_NO_REASON);
    expectPlain(reason);
  });

  it("passes the Core's own detail through", async () => {
    const err = await compactFailure(async () => ({
      ok: false,
      status: 409,
      statusText: 'Conflict',
      json: async () => ({ detail: 'A turn is running; stop it first' }),
    }));
    expect(roomFailureReason(err)).toBe('A turn is running; stop it first.');
  });
});

describe('roomsApi.list reports what it could not read (#1440)', () => {
  afterEach(() => vi.unstubAllGlobals());

  // An answer without its `unreadable` list is refused, not read as "nothing unreadable".
  // Killed by: frontend/src/lib/rooms.ts :: if (!Array.isArray(body?.rooms) || !Array.isArray(body?.unreadable)) {
  // Becomes: if (!Array.isArray(body?.rooms)) {
  it('refuses a listing that does not carry both lists', async () => {
    stubFetch(ok({ rooms: [{ room_id: 'r1' }] }));
    await expect(roomsApi.list()).rejects.toBeInstanceOf(RoomListingShapeError);
    expectPlain(roomFailureReason(new RoomListingShapeError()));
  });
});

describe('what a failed send can say (#1441)', () => {
  const said = (over: Partial<RoomState['transcript'][number]>) => ({
    seq: 1,
    sender_id: 'user',
    content: 'ship it',
    kind: 'utterance' as const,
    created_at: '',
    completed: true,
    ...over,
  });
  const roomWith = (transcript: RoomState['transcript']): RoomState => ({
    room_id: 'r1',
    title: 'Index tuning',
    participants: roster('scout'),
    transcript,
    turn_state: { agent_turns_since_human: 0 },
    policy: {
      max_agent_turns_per_human_message: 3,
      max_span_messages: 40,
      transcript_window: 15,
      hesitation_seconds: 0,
      default_responder_id: '',
    },
  });

  // Killed by: frontend/src/lib/rooms.ts :: message.seq > afterSeq &&
  // Becomes: message.seq >= 0 &&
  it('finds only a human message after the last one the head had seen', () => {
    const earlier = roomWith([said({ seq: 1 })]);
    expect(messageArrived(earlier, 'ship it', 1)).toBe(false);
    expect(messageArrived(roomWith([said({ seq: 1 }), said({ seq: 2 })]), 'ship it', 1)).toBe(true);
    // The same words from an agent are not the user's message.
    expect(messageArrived(roomWith([said({ seq: 2, sender_id: 'scout' })]), 'ship it', 1)).toBe(false);
    // Stored unchanged, so the words must match exactly.
    expect(messageArrived(roomWith([said({ seq: 2, content: 'ship it ' })]), 'ship it', 1)).toBe(false);
  });

  it('numbers from the highest seq the head holds', () => {
    expect(lastSeq(null)).toBe(0);
    expect(lastSeq(roomWith([said({ seq: 3 }), said({ seq: 7 })]))).toBe(7);
  });

  it('counts only a 4xx as a refusal', () => {
    expect(sendWasRefused(new RoomsApiError('no', 400, 'no'))).toBe(true);
    expect(sendWasRefused(new RoomsApiError('409 Conflict', 409, null))).toBe(true);
    // A 5xx can come after `accept` stored the message.
    expect(sendWasRefused(new RoomsApiError('boom', 503, 'The turn could not start.'))).toBe(false);
    expect(sendWasRefused(new RoomsNoAnswerError(new TypeError('Failed to fetch')))).toBe(false);
  });

  // Killed by: frontend/src/lib/rooms.ts :: if (err.outcome === 'absent') return `${reason} Your message is not in this conversation.`;
  // Becomes: if (err.outcome === 'absent') return `Not sent: ${reason}`;
  it('says "Not sent" for a refusal alone, and never shows raw error text', () => {
    const noAnswer = new RoomsNoAnswerError(new TypeError('Failed to fetch'));
    const bare500 = new RoomsApiError('500 Internal Server Error', 500, null);
    const refused = sendFailureNotice(
      new RoomSendError('refused', new RoomsApiError("Room 'r1' is closed.", 409, "Room 'r1' is closed.")),
    );
    expect(refused).toBe("Not sent: Room 'r1' is closed.");

    const absent = sendFailureNotice(new RoomSendError('absent', bare500));
    expect(absent).toBe(
      "Something went wrong in the app's background service. Your message is not in this conversation.",
    );
    const unconfirmed = sendFailureNotice(new RoomSendError('unconfirmed', noAnswer));
    expect(unconfirmed).toBe(
      "Could not confirm whether this conversation has your message: The app's background service didn't answer.",
    );
    const ownFault = sendFailureNotice(new Error('x is not a function'));
    expect(ownFault).toBe(
      'Could not confirm whether this conversation has your message: Something went wrong in the app.',
    );
    for (const text of [absent, unconfirmed, ownFault]) {
      expect(text).not.toMatch(/\bsent\b/i);
    }
    for (const text of [refused, absent, unconfirmed, ownFault]) expectPlain(text);
  });
});
