/**
 * The head's half of `/api/rooms`, plus the two decisions the conversation UI rests on.
 *
 * Both decisions are pure functions here rather than logic inside a component, because
 * both are the kind that a component test reaches only through a render and that a
 * refactor then quietly changes: which participant a message is attributed to, and what
 * the composer is allowed to promise about who will answer.
 */
import type {
  RoomCompaction,
  RoomContext,
  RoomHistoryAnswer,
  RoomParticipant,
  RoomProvenance,
  RoomServiceRef,
  RoomSpeakerDecision,
  RoomReplyPayload,
  RoomState,
  RoomTopicEvent,
  RoomSummary,
  RoomTranscriptMessage,
} from '../types';

/**
 * A non-2xx answer from `/api/rooms`, and whether the Core itself said why.
 *
 * `message` is unchanged from what callers always got — the Core's `detail`, or the status
 * line when there was none — so existing notices read the same. `coreDetail` is what lets a
 * surface meant for someone who does not read HTTP tell the two apart (#1389).
 */
export class RoomsApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly coreDetail: string | null,
  ) {
    super(message);
    this.name = 'RoomsApiError';
  }
}

/**
 * A request to `/api/rooms` that got no answer at all: `fetch` itself rejected (#1441).
 *
 * Its own class so that only this case is said as "the service didn't answer". A bare
 * `TypeError` used to stand for it, and a bug in the head's own code throws that too.
 */
export class RoomsNoAnswerError extends Error {
  constructor(readonly transportCause: unknown) {
    super('No answer from /api/rooms');
    this.name = 'RoomsNoAnswerError';
  }
}

/** `fetch`, with a rejection turned into `RoomsNoAnswerError`. */
async function ask(url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (err) {
    throw new RoomsNoAnswerError(err);
  }
}

async function readOrThrow(res: Response): Promise<any> {
  if (!res.ok) {
    let detail: string | null = null;
    try {
      const body = await res.json();
      // A string only. FastAPI's own request validation answers `detail` as a list of
      // objects, which `String()` renders as "[object Object]" -- not a reason, and not
      // the Core's words either.
      if (typeof body?.detail === 'string' && body.detail.trim() !== '') detail = body.detail;
    } catch {
      // A refusal with no JSON body keeps the status line, which is still a reason.
    }
    throw new RoomsApiError(detail ?? `${res.status} ${res.statusText}`, res.status, detail);
  }
  return res.status === 204 ? null : res.json();
}

/** Said when a request never got an answer: the Core is stopped, restarting, or unreachable. */
export const ROOM_READ_NO_ANSWER = "The app's background service didn't answer.";
/** Said when the Core answered a read with a failure but did not say why. */
export const ROOM_READ_NO_REASON = 'Something went wrong reading it.';
/** Said when the Core answered any other request with a failure but did not say why. */
export const ROOM_ACTION_NO_REASON = "Something went wrong in the app's background service.";
/**
 * Said for a failure that is neither the Core's answer nor a missing one: the head's own
 * code threw (#1441). Not blamed on the background service, which may have done nothing.
 */
export const ROOM_APP_FAULT = 'Something went wrong in the app.';

/**
 * Why a request to `/api/rooms` failed, in a sentence fit for someone who does not read code.
 *
 * The one mapping the room notices go through (#1411), in place of `err.message`, which fell
 * back to transport text. Only the Core's own `detail` is passed through. A browser's
 * transport message ("Failed to fetch", "Load failed"), a bare status line ("500 Internal
 * Server Error") and any other exception's text are replaced by one of ours, rather than
 * shown (P0); the cause is still stated, not dropped (P6). The result always ends a sentence,
 * so copy can follow it.
 *
 * `noReason` is what to say when the Core answered with a failure and no reason: a read and
 * an action fail in different words.
 */
export function roomFailureReason(err: unknown, noReason: string = ROOM_ACTION_NO_REASON): string {
  let reason: string;
  if (err instanceof RoomsApiError) reason = err.coreDetail ?? noReason;
  else if (err instanceof RoomsNoAnswerError) reason = ROOM_READ_NO_ANSWER;
  else reason = ROOM_APP_FAULT;
  reason = reason.trim();
  return /[.!?…]$/.test(reason) ? reason : `${reason}.`;
}

/** `roomFailureReason` for reading a conversation (#1389). */
export function roomReadReason(err: unknown): string {
  return roomFailureReason(err, ROOM_READ_NO_REASON);
}

/**
 * What the head knows about a message it tried to send and did not get a 202 for (#1441).
 *
 * * `refused` -- the Core answered 4xx. Every such answer is given before `accept` stores
 *   the message, so it is not in the conversation.
 * * `absent` -- there was no answer, or a 5xx, and a re-read of the conversation afterwards
 *   did not have it.
 * * `unconfirmed` -- there was no answer, or a 5xx, and the re-read failed too. The message
 *   may or may not be there, and nothing on screen may say which.
 */
export type RoomSendOutcome = 'refused' | 'absent' | 'unconfirmed';

export class RoomSendError extends Error {
  constructor(
    readonly outcome: RoomSendOutcome,
    readonly sendCause: unknown,
  ) {
    super(`Send ${outcome}`);
    this.name = 'RoomSendError';
  }
}

/**
 * Whether a failed send was answered by a refusal, which the Core gives before it stores
 * anything. A 5xx is not one: `send_message` can fail after `accept` has saved the message.
 */
export function sendWasRefused(err: unknown): boolean {
  return err instanceof RoomsApiError && err.status >= 400 && err.status < 500;
}

/** The highest `seq` in a conversation's transcript, or `0` for an empty or missing one. */
export function lastSeq(room: RoomState | null): number {
  if (!room) return 0;
  return room.transcript.reduce((highest, message) => Math.max(highest, message.seq), 0);
}

/**
 * Whether `room` holds a human utterance numbered after `afterSeq` whose text is `content`.
 *
 * Exact text, because `accept` stores what it was sent unchanged. `afterSeq` is the last
 * `seq` the head had seen when the message went out, so an earlier message with the same
 * words is not taken for this one.
 */
export function messageArrived(room: RoomState, content: string, afterSeq: number): boolean {
  return room.transcript.some(
    (message) =>
      message.seq > afterSeq &&
      message.kind === 'utterance' &&
      message.content === content &&
      senderOf(room.participants, message.sender_id)?.kind === 'human',
  );
}

/**
 * The composer's line for a send that did not go through, true in the outcome it is for.
 *
 * "Not sent" only when the Core refused; for a missing answer the head cannot say that, and
 * says what it knows instead.
 */
export function sendFailureNotice(err: unknown): string {
  if (!(err instanceof RoomSendError)) {
    return `Could not confirm whether this conversation has your message: ${ROOM_APP_FAULT}`;
  }
  const reason = roomFailureReason(err.sendCause);
  if (err.outcome === 'refused') return `Not sent: ${reason}`;
  if (err.outcome === 'absent') return `${reason} Your message is not in this conversation.`;
  return `Could not confirm whether this conversation has your message: ${reason}`;
}

/**
 * `GET /api/rooms`: the conversations that load, and the ids of records that do not (#1440).
 *
 * An answer without both lists is refused rather than read as "none": an empty rail for
 * an answer this head cannot read would be a claim that there is nothing to list.
 */
export interface RoomListing {
  rooms: RoomSummary[];
  unreadable: string[];
}

/** Thrown for a 2xx listing that is not the shape `RoomListing` describes. */
export class RoomListingShapeError extends Error {
  constructor() {
    super('GET /api/rooms answered without its rooms and unreadable lists');
    this.name = 'RoomListingShapeError';
  }
}

export const roomsApi = {
  list: async (): Promise<RoomListing> => {
    const body = await readOrThrow(await ask('/api/rooms'));
    if (!Array.isArray(body?.rooms) || !Array.isArray(body?.unreadable)) {
      throw new RoomListingShapeError();
    }
    return { rooms: body.rooms, unreadable: body.unreadable };
  },

  create: async (title: string, agentIds: string[]): Promise<RoomState> =>
    readOrThrow(
      await ask('/api/rooms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, agent_ids: agentIds }),
      }),
    ),

  get: async (roomId: string): Promise<RoomState> =>
    readOrThrow(await ask(`/api/rooms/${roomId}`)),

  rename: async (roomId: string, title: string): Promise<RoomState> =>
    readOrThrow(
      await ask(`/api/rooms/${roomId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      }),
    ),

  /** Remove a conversation for good; the Core also stops anything still running in it. */
  delete: async (roomId: string): Promise<void> => {
    await readOrThrow(await ask(`/api/rooms/${roomId}`, { method: 'DELETE' }));
  },

  send: async (roomId: string, content: string): Promise<{ seq: number }> =>
    readOrThrow(
      await ask(`/api/rooms/${roomId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      }),
    ),

  addAgent: async (roomId: string, agentId: string): Promise<RoomState> =>
    readOrThrow(
      await ask(`/api/rooms/${roomId}/participants`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: agentId }),
      }),
    ),

  removeParticipant: async (roomId: string, participantId: string): Promise<RoomState> =>
    readOrThrow(
      await ask(`/api/rooms/${roomId}/participants/${participantId}`, { method: 'DELETE' }),
    ),

  stop: async (roomId: string): Promise<RoomState> =>
    readOrThrow(await ask(`/api/rooms/${roomId}/stop`, { method: 'POST' })),

  retry: async (roomId: string): Promise<RoomState> =>
    readOrThrow(await ask(`/api/rooms/${roomId}/retry`, { method: 'POST' })),

  typing: async (roomId: string): Promise<void> => {
    await readOrThrow(await ask(`/api/rooms/${roomId}/typing`, { method: 'POST' }));
  },

  /**
   * How full each seat is, read on its own and not as part of `get`.
   *
   * `get` is re-issued after every landed turn; this answer costs one Core session load
   * per seat and the banner it feeds asks far less often, so folding the two would pay
   * that cost on every reconciliation.
   */
  context: async (roomId: string): Promise<RoomContext> =>
    readOrThrow(await ask(`/api/rooms/${roomId}/context`)),

  /**
   * Shorten every seat's context.
   *
   * Room-level because that is the order the question is asked in: *"this conversation is
   * full, shorten it"* names no seat, and the only control that reaches this is the one
   * button in the saturation banner.
   *
   * **The route also takes an optional `participant_id` for one seat, and this does not
   * send it (#1230).** It used to carry a `participantId` argument for that, which nothing
   * in the head ever passed -- `roomsApi` is the head's own fetch layer and not a published
   * client, so an argument no head code passes is unreachable code with a test for its only
   * caller. The capability stays where it is exercised: the route keeps it, and
   * `tests/unit/test_ui_room_api.py` posts `participant_id` to it. When a per-seat control
   * is built it comes back as one argument, which is what the unified-conversations
   * design note means in §4 by *"stays one argument away"* -- a statement about the
   * route, which is still true of it.
   */
  compact: async (roomId: string): Promise<RoomCompaction> =>
    readOrThrow(
      await ask(`/api/rooms/${roomId}/compact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      }),
    ),

  /**
   * Rewind the conversation to a message it holds, and cut every seat back to match.
   *
   * `seq` is the number of a message, never an index into the rendered list: the two
   * diverge the moment a row is a join or a failure, and a rewind aimed at a position
   * cuts a different message than the one the reader pointed at.
   */
  truncateHistory: async (roomId: string, seq: number): Promise<RoomHistoryAnswer> =>
    readOrThrow(
      await ask(`/api/rooms/${roomId}/history/truncate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seq }),
      }),
    ),

  /** Empty this conversation, keeping its id, its title and who is in it. */
  clearHistory: async (roomId: string): Promise<RoomHistoryAnswer> =>
    readOrThrow(await ask(`/api/rooms/${roomId}/history`, { method: 'DELETE' })),
};

/**
 * How many times a saturation read re-asks after the record moved underneath it.
 *
 * A read that comes back into a conversation whose generation has advanced is answering a
 * question about a record that no longer exists, so its answer cannot be committed. The
 * question, though, is still owed -- and nothing else will ask it, because the head reads
 * the context on a landed turn and the turn that prompted this read may have been the
 * conversation's last, after which no event arrives at all (#1256).
 *
 * Bounded rather than a `while`: what bumps the generation is a discrete act -- a stop,
 * a retry, a seat added, a rewind, a clear -- so a second act landing
 * inside the retry is possible and a third is not worth arming a loop for. An unbounded
 * retry would turn anything that bumped the generation on a timer into a request spin.
 */
export const CONTEXT_READ_ATTEMPTS = 3;

/**
 * The participant a message came from, or `null` when nothing in the roster says.
 *
 * Deliberately **not** falling back to "whichever agent is selected". uclone2's group
 * chat did exactly that and rendered one agent's name and face over another's words --
 * in the typing bubble, where it was transient, and on message rows, where it became a
 * permanent part of the transcript. A name we cannot resolve is shown as the raw sender
 * id; a name we invent is worse than no name.
 */
export function senderOf(
  participants: RoomParticipant[],
  senderId: string,
): RoomParticipant | null {
  return participants.find((p) => p.id === senderId) ?? null;
}

/**
 * A served-by line built from the room's own `Provenance`, or `null` when there is none.
 *
 * `served_by` is a `ServiceRef` object (`core/provenance.py`), not a string. Rendering the
 * object itself as a React child threw `Objects are not valid as a React child` and took
 * the whole conversation down on the first agent reply -- and `tsc` could not see it,
 * because the row was typed with the flattened chat shape.
 */
export function serviceRefLabel(ref: RoomServiceRef | null | undefined): string | null {
  if (!ref || !ref.provider) return null;
  return ref.model ? `${ref.provider}:${ref.model}` : ref.provider;
}

/**
 * What served this turn, and -- when it was not what was asked for -- what was.
 *
 * `degraded` is computed by the Core and means `served_by` differs from `requested`. P6
 * exists so that substitution is visible rather than inferred from the text, so when the
 * two differ the line says both.
 */
export function servedByLabel(provenance: RoomProvenance | null | undefined): string | null {
  const served = serviceRefLabel(provenance?.served_by);
  if (!served) return null;
  const requested = serviceRefLabel(provenance?.requested);
  if (provenance?.degraded && requested && requested !== served) {
    return `${served} (asked for ${requested})`;
  }
  return served;
}

/**
 * The two shapes a conversation is rendered in, chosen by how many agents are seated.
 *
 * Design doc `unified-conversations-and-room-ui.md` §3.2.3 [Rev 20]. It is one predicate,
 * derived here rather than in each component, because two regions computing "is this a
 * group?" from two different fields is exactly the split #1088 was opened for.
 *
 * Nought agents is `solo`: there is no second speaker on the left to tell apart.
 */
export type ConversationShape = 'solo' | 'multi';

export function conversationShape(participants: readonly RoomParticipant[]): ConversationShape {
  return participants.filter((p) => p.kind === 'agent').length > 1 ? 'multi' : 'solo';
}

/**
 * Every turn whose model is known, with the **model** and the **label** kept apart.
 *
 * Both readers below ask the same question of a transcript, and asking it in two places is
 * how they drift: a roster change skipped by one and read as speech by the other would put
 * a model change under a conversation whose model never changed. Roster rows are excluded
 * for being roster rows, not for happening to carry no provenance -- the wire is free to
 * attach one -- and a turn nothing reported a model for is left out rather than standing in
 * as a value, because a failed turn's silence is not a new model (P6).
 *
 * The two strings are **not** interchangeable, and conflating them was a defect on this
 * surface: `servedByLabel` folds the *requested* model into the display label when the
 * Core reports `degraded`, so a turn served by the same model as its predecessor under a
 * different request compared unequal, and the change line read "came from ollama:qwen3:8b,
 * not ollama:qwen3:8b". Anything asking *has the model changed* compares `model`; anything
 * printing compares nothing and prints `display`.
 */
function* servedTurns(
  transcript: readonly RoomTranscriptMessage[],
): Generator<readonly [RoomTranscriptMessage, string, string]> {
  for (const message of transcript) {
    if (message.kind !== 'utterance') continue;
    const model = serviceRefLabel(message.provenance?.served_by);
    const display = servedByLabel(message.provenance);
    if (!model || !display) continue;
    yield [message, model, display];
  }
}

/**
 * The model a 1:1's header may carry, or `null` when no single model is true of every row.
 *
 * §3.2.3 puts the value in the header "where it is true of every row beneath it", and a
 * header is a standing claim: it says this of the rows below it, all of them, including
 * the ones not yet on screen when the reader last looked. So it survives exactly as long
 * as there is one value to state. The moment a second model serves a turn, there is no
 * string that is true of both, and the honest reading of the doc's own rule -- print it
 * where printing carries information -- is that the value has stopped being a constant
 * and become content, which belongs on the rows (see `attributionLines`).
 *
 * This replaces an anchor on the *first* served turn. That kept the header true of the
 * rows above the change and left it false of every row below, which is the one thing a
 * header may not be: with turns 1-2 on `qwen3:8b` and 3-5 on `hermes3:8b`, the header
 * named `qwen3:8b` and rows 4 and 5 printed nothing at all, so the only standing statement
 * about them named a model that had not served them.
 *
 * `null` when nothing has been served yet, so a fresh conversation states no model. There
 * is no default to fall back on: which model answers is the Core's to report (P6).
 *
 * **The comparison key is the flattened `provider:model` string, and that is deliberate.**
 * `serviceRefLabel` joins the pair with a colon, so `{provider: 'ollama:qwen3:8b', model:
 * null}` and `{provider: 'ollama', model: 'qwen3:8b'}` compare equal and a 1:1 alternating
 * between the two shapes keeps its header. Comparing the pair instead would be *worse* on
 * this surface, not better: the header prints the flattened string, so two refs that
 * compare unequal here would go on rendering identically on the rows. The header would
 * vanish and forty turns would each print the same label -- exactly the decoration §3.2.3
 * removed -- with nothing on screen for the reader to read the change off. A surface may
 * only claim a difference it can show, and this layer's unit of identity is what it
 * renders. What would make the collision real is a `served_by.provider` containing a
 * colon, and then the two refs render the same string too, so the repair belongs in
 * `serviceRefLabel` -- an unambiguous rendering -- and the comparison follows it for free.
 * Measured at this commit, nothing emits one: every `provider_name` is a literal without a
 * colon (`ollama`, `vllm`, `openai`, `anthropic`, `gemini`, `mock`), and the two composite
 * providers in the Core are dotted (`uclone_x.llm.compactor`, `agent.core`). The one
 * identifier that flows into a `provider` field is `agent/base.py`'s `provider=self.agent_id`,
 * and it lands on `requested`, which nothing here compares.
 */
export function headerAttribution(
  transcript: readonly RoomTranscriptMessage[],
): string | null {
  let only: string | null = null;
  for (const [, model] of servedTurns(transcript)) {
    if (only === null) only = model;
    else if (only !== model) return null;
  }
  return only;
}

/**
 * The model that served the most recent answered turn, or `null` when none has.
 *
 * What the strip beside the composer names. It is deliberately *not*
 * `headerAttribution`: that one reports `null` the moment two models have served, because
 * a header is a standing claim about every row beneath it. This is not a claim about the
 * rows -- it is the last thing that actually happened, which stays true however many
 * models have served, and so it keeps naming one where the header has gone quiet.
 *
 * Read from the turn's own `served_by` and never from a model chosen in this head. A room
 * turn is served by the seated persona's configured model, resolved in
 * `src/uclone_x/room/resolver.py`, so the workspace default and any override held here
 * name a model that does not serve it (see `A_ROOM_SEAT_CAN_CARRY_A_MODEL`, #1235).
 * `null` is therefore a real answer -- nothing has served yet -- and the surface has to
 * say that in words rather than leave the slot blank (P6).
 */
export function lastServedModel(
  transcript: readonly RoomTranscriptMessage[],
): string | null {
  let latest: string | null = null;
  for (const [, model] of servedTurns(transcript)) latest = model;
  return latest;
}

/**
 * Which turns print their own attribution line, keyed by `seq`, and what each one prints.
 *
 * FR-13.4 as amended (PRD 0.5.0, `97e2d60a`) separates the record from its printing: every
 * turn carries who served it, and the surface prints it where printing carries information.
 *
 * * **`multi`** -- every served turn. The speaker changes turn to turn, so the model is
 *   content rather than a constant.
 * * **`solo`** -- nothing at all while one model has served every turn, because the header
 *   states it and a line repeating the header under turn one is the decoration this
 *   replaced; and then **every** served turn from the moment a second model appears,
 *   exactly as a group prints, because the header has stopped speaking for them.
 *
 * There is deliberately no "only the turn that changed" case any more. A line on the
 * changing turn alone leaves the turns after it carrying nothing while the header names
 * the model they were *not* served by -- a reader would have to scan upward past rows that
 * print nothing to find the nearest line, which FR-13.4's "recoverable from the nearest
 * one at or above it" permits only when the header above is itself true.
 *
 * **[Rev 25] A degraded serve prints on its own turn, whether or not the model changed**,
 * and that is a second rule rather than a case of the first. The header answers *what has
 * served this conversation* and is decided by `served_by` alone, deliberately, so one
 * substitution does not strip a label that stays true of every other row. The turn line
 * answers *what happened here*, and a substitution is only ever the second question's
 * answer. Inheriting the header's comparison collapsed the two: a degraded turn answered
 * by the model already in the header rendered nothing at all, and nothing else on screen
 * distinguished it from an ordinary turn -- which is exactly the case a reader cannot
 * infer, and which FR-13.4 names among those that print. Reaching it behind `why ›`
 * satisfies the *reachability* clause, which is separate and weaker.
 *
 * Resolved from `provenance`, never from the message content -- a local model once
 * reported itself as another vendor's.
 */
export function attributionLines(
  transcript: readonly RoomTranscriptMessage[],
  shape: ConversationShape,
): ReadonlyMap<number, string> {
  const lines = new Map<number, string>();
  const perTurn = shape === 'multi' || headerAttribution(transcript) === null;
  for (const [message, , display] of servedTurns(transcript)) {
    if (perTurn || message.provenance?.degraded) lines.set(message.seq, display);
  }
  return lines;
}

/** Characters `MentionSelector._candidates` retries a token without. */
const MENTION_TRAILING = /[.-]+$/;

/**
 * Mention tokens in a draft, in the order they were written.
 *
 * A transcription of `selectors.py`'s `MENTION_PATTERN` -- `(?<![\w@/])@(\w[\w.\-]*)` --
 * without the lookbehind, which Safari only learned in 16.4. The head reads addresses the
 * same way the Core does or it promises answers the Core will not give.
 */
export function mentionTokens(draft: string): string[] {
  const tokens: string[] = [];
  const scanner = /@(\w[\w.\-]*)/g;
  let match: RegExpExecArray | null;
  while ((match = scanner.exec(draft)) !== null) {
    const before = match.index === 0 ? '' : draft[match.index - 1];
    if (before && /[\w@/]/.test(before)) continue;
    tokens.push(match[1]);
  }
  return tokens;
}

/**
 * The participant a token addresses, resolved the way `MentionSelector._resolve` does.
 *
 * Ids before aliases, case-insensitively on both sides, and the token retried once with
 * trailing `.`/`-` removed. **No regex is built from the id.** It used to be -- a
 * participant id the Core permits, such as `a(`, made `new RegExp()` throw a
 * `SyntaxError` during render and took the screen with it.
 */
export function resolveMention(
  participants: readonly RoomParticipant[],
  token: string,
): RoomParticipant | null {
  const trimmed = token.replace(MENTION_TRAILING, '');
  const candidates = trimmed && trimmed !== token ? [token, trimmed] : [token];
  for (const candidate of candidates) {
    const lowered = candidate.toLowerCase();
    const byId = participants.find((p) => p.id.toLowerCase() === lowered);
    if (byId) return byId;
    const byAlias = participants.find((p) =>
      (p.aliases ?? []).some((alias) => alias.toLowerCase() === lowered),
    );
    if (byAlias) return byAlias;
  }
  return null;
}

/**
 * The title a conversation carries from the moment New opens it until its first message.
 *
 * New does not ask for a name (F1: no modal, no title prompt), and the Core refuses a
 * blank title, so a conversation starts under this one and `seededTitle` replaces it once.
 */
export const NEW_CONVERSATION_TITLE = 'New conversation';

/** The longest seeded title, ellipsis included. A rail row is one line. */
const SEEDED_TITLE_MAX = 60;

/** CSI and two-character escapes: the shapes `service.py`'s `_clean_title` refuses. */
const ANSI_ESCAPE = /\u001b(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])/g;

/**
 * The title to give a conversation on its first message, or `null` to leave it alone.
 *
 * Seeded **once** (D3): only a conversation still under `NEW_CONVERSATION_TITLE`, and only
 * while nobody has said anything in it. A title somebody chose is never overwritten, and a
 * conversation already under way is never renamed from whatever happens to be said next --
 * which is the property `RoomState.title` protects by refusing a *derived* title.
 *
 * One line, whitespace collapsed, ANSI escapes removed, cut at a word. Each of those is a
 * refusal the Core would otherwise answer the rename with.
 */
export function seededTitle(room: RoomState, firstMessage: string): string | null {
  if (room.title !== NEW_CONVERSATION_TITLE) return null;
  if (room.transcript.some((message) => message.kind === 'utterance')) return null;
  const line = firstMessage
    .replace(ANSI_ESCAPE, '')
    .split(/[\r\n]/)
    .map((part) => part.replace(/\s+/g, ' ').trim())
    .find((part) => part !== '');
  if (!line) return null;
  if (line.length <= SEEDED_TITLE_MAX) return line;
  const cut = line.slice(0, SEEDED_TITLE_MAX - 1);
  const atWord = cut.lastIndexOf(' ');
  return `${(atWord > 0 ? cut.slice(0, atWord) : cut).trimEnd()}…`;
}

/** How a participant is named on screen: the roster's name, else the raw id. */
export function participantLabel(participant: RoomParticipant): string {
  return participant.display_name || participant.id;
}

/** A row's sender, shown as the roster knows it -- or as the raw id, never as a guess. */
export function senderLabel(room: RoomState, senderId: string): string {
  const participant = senderOf(room.participants, senderId);
  if (!participant) return senderId;
  if (participant.kind === 'human') return 'you';
  return participantLabel(participant);
}

/**
 * A row's sender kind, for the face beside its label -- 'agent' when the roster has no
 * record of them, since an unresolved id in this room is far more likely a departed agent
 * than the one human it seats (#763).
 */
export function senderKind(room: RoomState, senderId: string): 'agent' | 'human' {
  return senderOf(room.participants, senderId)?.kind ?? 'agent';
}

/**
 * What the composer may promise about who answers next, given what is typed so far.
 *
 * It says what the selector chain in `room/selectors.py` will actually do, which is not
 * the same as what the chain's *order* suggests:
 *
 * * **`MentionSelector` runs first, ahead of `SoleAgentSelector`.** So an unresolvable
 *   address is refused rather than answered by the room's only agent. This used to read
 *   `scout will answer` for `@phantm` in a one-agent room -- a promise the Core refuses,
 *   and the exact reroute-to-a-default the chain order was changed to prevent.
 * * **Matching is case-insensitive and includes aliases.** The Core lowercases both
 *   sides and falls back to `Participant.aliases`.
 * * **Every addressed agent answers once, bounded by the turn ceiling.** Four addresses
 *   under a ceiling of three is three answers and one that never gets the floor, and a
 *   line naming all four claims a turn the ceiling already refused.
 *
 * Honest ambiguity where the chain is undetermined: with several agents, no address and
 * no designated responder, the model selector decides and this cannot know.
 */
export type Answerers =
  /** Nobody will answer, and `reason` says why. */
  | { kind: 'none'; reason: 'empty-room'; token?: undefined }
  | { kind: 'none'; reason: 'unknown-name'; token: string }
  /** The room has not been told who, and will choose when the message lands. */
  | { kind: 'unresolved' }
  /**
   * These seats will answer, in this order.
   *
   * `cutOff` is how many more were named than the per-message ceiling admits -- nought on
   * an ordinary address. Those seats are already absent from `seats`.
   */
  | { kind: 'seats'; seats: RoomParticipant[]; cutOff: number };

/**
 * Who will answer the message now in the box.
 *
 * Exported because two regions need it and neither may compute it for itself: the hint
 * below the composer says who in words, and the strip beside it shows that seat's context
 * and the model it last answered on. A second copy of this resolution is how the two come
 * to disagree in front of the reader -- the same failure `conversationShape` was pulled
 * out of two components to prevent.
 *
 * It resolves mentions exactly as the Core does, which is why an unresolved name is
 * `none` rather than a fallback to the only agent in the room: `MentionSelector` refuses
 * the whole address before serving any of it.
 */
export function answeringSeats(
  participants: readonly RoomParticipant[],
  draft: string,
  defaultResponderId: string,
  maxAgentTurnsPerHumanMessage: number,
): Answerers {
  const agents = participants.filter((p) => p.kind === 'agent');
  if (agents.length === 0) return { kind: 'none', reason: 'empty-room' };

  const addressed: RoomParticipant[] = [];
  for (const token of mentionTokens(draft)) {
    const resolved = resolveMention(participants, token);
    if (resolved === null) return { kind: 'none', reason: 'unknown-name', token };
    if (resolved.kind === 'agent' && !addressed.some((a) => a.id === resolved.id)) {
      addressed.push(resolved);
    }
  }

  const ceiling = maxAgentTurnsPerHumanMessage > 0 ? maxAgentTurnsPerHumanMessage : addressed.length;
  if (addressed.length > ceiling) {
    return { kind: 'seats', seats: addressed.slice(0, ceiling), cutOff: addressed.length - ceiling };
  }
  if (addressed.length > 0) return { kind: 'seats', seats: addressed, cutOff: 0 };
  if (agents.length === 1) return { kind: 'seats', seats: [agents[0]], cutOff: 0 };
  const responder = defaultResponderId
    ? agents.find((a) => a.id === defaultResponderId)
    : undefined;
  if (responder) return { kind: 'seats', seats: [responder], cutOff: 0 };
  return { kind: 'unresolved' };
}

export function answerHint(
  participants: RoomParticipant[],
  draft: string,
  defaultResponderId: string,
  maxAgentTurnsPerHumanMessage: number,
): string {
  const who = answeringSeats(
    participants,
    draft,
    defaultResponderId,
    maxAgentTurnsPerHumanMessage,
  );
  if (who.kind === 'none') {
    // The Core refuses the whole address before serving any of it, so nothing here
    // may promise an answer -- not even the only agent in the room.
    return who.reason === 'empty-room'
      ? 'No one is in this conversation yet.'
      : `No one here is called @${who.token}, so this message will not be answered.`;
  }
  if (who.kind === 'unresolved') return 'Someone will pick this up';

  const answering = who.seats.map(participantLabel).join(', ');
  if (who.cutOff > 0) {
    return `${answering} will answer, in that order. Paused after ${who.seats.length} replies, so the rest of those you named will not.`;
  }
  if (who.seats.length === 1) return `${answering} will answer`;
  return `${answering} will answer, in that order`;
}

/**
 * The silence to report under the transcript, or `null` when there is none to report.
 *
 * `last_decision` alone does not say that a message went unanswered. The chain records a
 * silence whenever it has nobody left to give the floor to -- and that is how every
 * answered exchange ends too: the one agent replies, `SoleAgentSelector` abstains because
 * the last utterance is no longer a human's, and the chain is exhausted. Reading the
 * verdict by itself told a new user, under their first answer, that nobody had answered
 * and nobody was going to (#920).
 *
 * So the silence is reported only while the **latest human message has nothing that
 * answers it**. Checking merely that the transcript's last row is the human's own missed
 * a real case: a reply already in flight when a second human message lands can complete
 * afterward and land *after* it, so the last row is an agent's and looks like an answer to
 * the newer message -- when the turn that produced it started before that message existed
 * and never saw it (#945). `rendered_through` is that turn's own account of what it saw;
 * a follower is an answer only when it covers the human row it would have to have seen.
 * A follower with no evidence of what it rendered -- a fixture that omits the field, or a
 * row stored before the field existed -- is read permissively rather than accused. `0` is
 * that no-evidence value on the wire (`room/models.py`'s `rendered_through` defaults to
 * `0`, and `model_dump` always serialises it, so a legacy row arrives as a concrete `0`,
 * never as `undefined`): treating `0` as evidence *against* an answer, rather than as the
 * absence of evidence it is documented to be, reopened #920 for every room stored before
 * this field existed, directly under the answer already on screen. Joins and leaves are
 * not answers. A sender the roster no longer knows is not taken for the human; a room
 * seats exactly one, and a departed agent's reply is still a reply.
 */
export function unansweredSilence(
  room: Pick<RoomState, 'participants' | 'transcript' | 'last_decision'>,
): RoomSpeakerDecision | null {
  const decision = room.last_decision;
  if (!decision || decision.verdict !== 'silence') return null;
  const utterances = room.transcript.filter((message) => message.kind === 'utterance');
  let humanSeq: number | null = null;
  for (let i = utterances.length - 1; i >= 0; i -= 1) {
    if (senderOf(room.participants, utterances[i].sender_id)?.kind === 'human') {
      humanSeq = utterances[i].seq;
      break;
    }
  }
  if (humanSeq === null) return null;
  const seq = humanSeq;
  const followers = utterances.filter((message) => message.seq > seq);
  if (followers.length === 0) return decision;
  const answered = followers.some(
    (message) =>
      message.rendered_through === undefined ||
      message.rendered_through === 0 ||
      message.rendered_through >= seq,
  );
  return answered ? null : decision;
}

/**
 * The participants an `@` completion should offer for the token being typed.
 *
 * Agents only, and only this conversation's: §3.3 F3's "the current participants only".
 * A user who has to remember and type a runtime id gets defect 4's wrong promise and
 * defect 5's silent failure for their trouble (ui-authoring, recognition over recall).
 */
export function mentionCandidates(
  participants: RoomParticipant[],
  prefix: string,
): RoomParticipant[] {
  const lowered = prefix.toLowerCase();
  return participants.filter((p) => {
    if (p.kind !== 'agent') return false;
    if (lowered === '') return true;
    if (p.id.toLowerCase().startsWith(lowered)) return true;
    return (p.aliases ?? []).some((alias) => alias.toLowerCase().startsWith(lowered));
  });
}

/**
 * The `@` token the caret sits inside, or `null` when it does not sit inside one.
 *
 * Returns the token's start offset so an accepted completion can replace exactly the
 * characters the user typed rather than re-deriving them.
 */
export function mentionUnderCaret(
  draft: string,
  caret: number,
): { start: number; prefix: string } | null {
  let index = caret - 1;
  while (index >= 0 && /[\w.\-]/.test(draft[index])) index -= 1;
  if (index < 0 || draft[index] !== '@') return null;
  const before = index === 0 ? '' : draft[index - 1];
  if (before && /[\w@/]/.test(before)) return null;
  return { start: index, prefix: draft.slice(index + 1, caret) };
}

/**
 * Whether a room response that is now resolving should be thrown away.
 *
 * Two independent ways a reply can be stale, and both were reachable:
 *
 * * the head has moved on to another conversation since the request went out
 *   (`openRoom(A)` then `openRoom(B)` -- A's transcript landing second left A on screen
 *   while the open id was already B, so the next Send posted into B); and
 * * the reply is for a conversation that is not the open one at all, which the several
 *   unguarded refetches on the `final` event could each produce.
 *
 * A pure rule rather than three inline comparisons, because a guard that exists in three
 * copies is a guard that will shortly exist in two.
 */
export function roomResponseIsStale(
  startedGeneration: number,
  currentGeneration: number,
  roomId: string,
  openRoomId: string | null,
): boolean {
  return startedGeneration !== currentGeneration || roomId !== openRoomId;
}

/** One read of a room, as `RoomReadOrder.issue` numbered it when it went out. */
export interface RoomReadTicket {
  roomId: string;
  /** The generation it started in, for `roomResponseIsStale`. */
  generation: number;
  /** Its place in the order the reads of this room went out in. */
  seq: number;
}

/**
 * The order the reads of each room went out in, so that only the last one lands (#1412).
 *
 * `roomResponseIsStale` drops a read when the conversation moved on: another room is open,
 * or an act (Clear, Rewind, Stop, Retry, seating a clone) bumped the generation. It cannot
 * drop a read that was simply *overtaken*. Two reads of the same room in the same
 * generation -- the send's re-read and the `refreshRoom` a `final` fires -- can resolve in
 * either order, and whichever lands second is what stays on screen.
 *
 * The rule here is stricter than "not older than what is on screen": a read may land only
 * while it is the **last one issued** for its room. A read that went out before the stream's
 * `final` can still resolve after it and before the read `final` fired; it is then the
 * newest *answer* but not the newest *question*, and committing it takes the landed reply
 * off screen until the later read arrives. Under the stricter rule it is dropped, and the
 * later read, which is already out, is the one that lands.
 *
 * A read overtaken by one that then fails is lost with it; the failed read's caller puts
 * its notice on screen.
 */
export class RoomReadOrder {
  private issued = 0;
  private readonly latest = new Map<string, number>();

  /** Number a read of `roomId` that is going out now. */
  issue(roomId: string, generation: number): RoomReadTicket {
    this.issued += 1;
    this.latest.set(roomId, this.issued);
    return { roomId, generation, seq: this.issued };
  }

  /** Whether a later read of the same room went out after this one did. */
  overtaken(ticket: RoomReadTicket): boolean {
    return this.latest.get(ticket.roomId) !== ticket.seq;
  }

  /** Whether this read's answer must be dropped rather than put on screen. */
  isStale(ticket: RoomReadTicket, currentGeneration: number, openRoomId: string | null): boolean {
    return (
      roomResponseIsStale(ticket.generation, currentGeneration, ticket.roomId, openRoomId) ||
      this.overtaken(ticket)
    );
  }
}

/** The live state of one turn, accumulated from the room topic. */
export interface RoomLiveTurn {
  agentId: string;
  /** The Core's `turn_id`: what every delta and the landed row of this turn carry. */
  turnId: string;
  text: string;
  statusText?: string;
}

export interface RoomLiveState {
  turn: RoomLiveTurn | null;
  error: string | null;
}

export const EMPTY_LIVE: RoomLiveState = { turn: null, error: null };

/**
 * Fold one room event into the live turn the transcript renders underneath its rows.
 *
 * **Keyed on `turn_id`, and only on it.** The Core mints it before the turn runs and puts
 * it on the floor being taken, on every delta, and on the landed row. It used to be keyed
 * on `seq`, which the Core sends on `final` alone: every delta compared `0` against
 * `undefined`, so each one replaced the bubble instead of appending to it, and a user
 * watched single words flicker past. The test that should have caught it supplied a `seq`
 * the server never sends; the tests now read `test/room-stream-events.json`, which the
 * Core's own suite generates.
 *
 * **There is no "the room is finished" event.** The topic carries a floor being taken, a
 * token delta, a landed utterance and a cascade-level failure -- not a silence, not the
 * ceiling being reached, not a roster change. So the live turn is cleared by the landed
 * row of the same turn, and a head that waited for a "yielded" event would show
 * "Writing..." forever. It is also why nothing here substitutes a generic placeholder
 * when it has no status: an unattributed "Thinking..." is indistinguishable from a
 * stream that died, which is the failure uclone2 recorded and then had to stop hiding.
 *
 * **Every kind of event is named, and anything else is refused out loud** (#929). The
 * topic also carries the human's composing notice and Stop's interrupt; neither changes
 * the live turn, and each is handled as itself. This used to fold every event it did not
 * name as a landed reply, so a status the Core added would have ended whichever bubble
 * shared its turn, with nothing to say why. An event this head does not recognise is now
 * reported on the console and leaves the live state exactly as it was.
 */
export function applyRoomEvent(live: RoomLiveState, evt: RoomTopicEvent): RoomLiveState {
  switch (evt.event_type) {
    case 'AGENT_REPLY':
      return applyReply(live, evt.payload);
    case 'USER_INPUT':
      // The human composing: a hint for the Core's hesitation pause, not a turn.
      return evt.payload.status === 'typing' ? live : ignoreUnrecognised(live, evt.payload);
    case 'INTERRUPT':
      // Stop. A turn it cancels is still recorded and lands like any other, with
      // `completed: false`, and that row clears the bubble; a Stop with no turn running
      // has no bubble to clear. Clearing here as well would race the landed row.
      return live;
    default:
      return ignoreUnrecognised(live, evt);
  }
}

function applyReply(live: RoomLiveState, reply: RoomReplyPayload): RoomLiveState {
  switch (reply.status) {
    case 'error':
      return { turn: null, error: reply.error ?? 'This conversation stopped.' };
    case 'generating':
      return {
        turn: {
          agentId: reply.agent_id,
          turnId: reply.turn_id,
          text: '',
          ...(reply.detail !== undefined ? { statusText: reply.detail } : {}),
        },
        error: null,
      };
    case 'status_update': {
      const current = live.turn;
      const turnMismatch = !current || current.turnId !== reply.turn_id;
      if (turnMismatch) {
        return {
          turn: {
            agentId: reply.agent_id,
            turnId: reply.turn_id,
            text: '',
            statusText: reply.detail || 'Thinking...',
          },
          error: live.error,
        };
      }
      return { turn: { ...current, statusText: reply.detail }, error: live.error };
    }
    case 'streaming': {
      const current = live.turn;
      if (!current || current.turnId !== reply.turn_id) {
        // A delta for a turn this head did not see start -- opened mid-turn, or the previous
        // turn's landed row was missed. It opens its own bubble rather than lending its
        // words to whoever was live before.
        return {
          turn: { agentId: reply.agent_id, turnId: reply.turn_id, text: reply.delta },
          error: live.error,
        };
      }
      return { turn: { ...current, text: current.text + reply.delta }, error: live.error };
    }
    case 'final':
      // The row is in the transcript now, so this turn's bubble is over. A landed row for a
      // *different* turn leaves the live one alone -- one speaker's row must not erase
      // another's words mid-sentence.
      if (live.turn && live.turn.turnId === reply.turn_id) return { turn: null, error: live.error };
      return live;
    default:
      return ignoreUnrecognised(live, reply);
  }
}

/**
 * Refuse an event this head has no case for: say so, and change nothing.
 *
 * Ignored rather than guessed at, because every guess available is a wrong one -- a
 * landed row clears somebody's bubble, a failure puts copy in front of the user -- and
 * reported, because an ignored event nobody hears about is how the head drifts from the
 * Core unnoticed. The console, not the conversation: the reader of the conversation has
 * no remedy for a head older than its Core.
 */
function ignoreUnrecognised(live: RoomLiveState, event: unknown): RoomLiveState {
  console.error('Ignoring a room event this head does not recognise:', event);
  return live;
}
