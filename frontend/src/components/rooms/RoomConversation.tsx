import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown, Bot, Cpu, Loader2, Mic, Plus, Send, Sparkles, Square, Trash2 } from 'lucide-react';
import type {
  AgentInfo,
  RoomContext,
  RoomState,
  RoomTranscriptMessage,
  RoomParticipant,
} from '../../types';
import {
  answerHint,
  answeringSeats,
  attributionLines,
  conversationShape,
  headerAttribution,
  roomsApi,
  lastServedModel,
  mentionCandidates,
  mentionUnderCaret,
  participantLabel,
  servedByLabel,
  senderKind,
  senderLabel,
  unansweredSilence,
  sendFailureNotice,
  type ConversationShape,
  type RoomLiveState,
  type RoomLiveTurn,
} from '../../lib/rooms';
import { memoryUnsavedNotice, refusalRemedy, turnFailureSentence } from '../../lib/turnOutcome';
import { useEscapeOwner } from '../../lib/escapePrecedence';
import { cn } from '../../lib/utils';
import { appendPhrase, useDictation } from '../../lib/useDictation';
import { personaAvatarUrl } from '../../lib/personaAvatar';
import { Avatar, FillRing } from '../../ui-kit';
import { Button } from '../ui/Button';
import { RichText } from '../RichText';

interface RoomConversationProps {
  room: RoomState;
  /** Agents the runtime can offer, for the invite list. Never a typed id. */
  availableAgents: AgentInfo[];
  live: RoomLiveState;
  /**
   * What is in the composer, and where a keystroke goes (#1290).
   *
   * Controlled from above rather than held here, because this component does not survive
   * a conversation switch: `openRoom` nulls the room synchronously, so anything typed and
   * kept in local state is destroyed by the switch. The owner keys it by conversation, so
   * coming back gives the text back.
   */
  draft: string;
  onDraftChange: (next: string) => void;
  /**
   * Post the message.
   *
   * Returns a promise so the composer can keep the draft when the post is refused. It
   * used to clear the box before the promise settled, so a 400 the user never saw also
   * destroyed what they had written.
   */
  onSend: (content: string) => void | Promise<void>;
  onStop: () => void;
  onRetry: () => void;
  onAddAgent: (agentId: string) => void;
  onTyping: () => void;
  /**
   * How full the conversation is, or `null` while nothing has been read.
   *
   * Read separately from the room (`GET /api/rooms/{id}/context`) and passed in rather
   * than derived here: the figure is the Core's per-seat turn count, and a head that
   * estimated it from the transcript would report a number no seat agrees with.
   */
  context?: RoomContext | null;
  /** Shorten every seat's context. Absent when the surface cannot offer it. */
  onCompact?: () => void | Promise<void>;
  compacting?: boolean;
  /** Empty this conversation, keeping its id, its title and who is in it. */
  onClearHistory?: () => void | Promise<void>;
  /**
   * Seats whose own session did not reset with the last rewind or clear.
   *
   * Empty on an ordinary cut. A seat named here is still answering from turns the
   * transcript no longer holds, which is what these controls exist to prevent -- so it is
   * said, rather than being left to look like a clean result.
   */
  participantsNotReset?: readonly string[];
  /** Models the runtime enumerated. Never a tag typed from memory (P0). */
  availableModels?: string[];
  /** The app-wide default, so the "use it" option can name what it is. */
  currentModel?: string;
  /**
   * This conversation's own model, when one has been chosen for it. `undefined` means it
   * answers on `currentModel` like everything else.
   */
  agentModelOverride?: string;
  /** Sets (or clears, with `null`) the model for the one clone seated here. */
  onSelectAgentModel?: (agentId: string, model: string | null) => void;
  /** Toggle autonomous discussion mode. */
  onToggleAutonomous?: (enabled: boolean) => void;
  /**
   * Show one turn's record, by `seq`, on the workspace dock's Turn surface.
   *
   * Required rather than optional: `why ›` is the only way onto that surface, and an
   * owner that forgot to wire it would render a control that does nothing -- which reads
   * as a broken turn rather than as a missing feature. The `seq`, never the message, so
   * what the dock shows is whatever the transcript holds at that seq now; a copied object
   * would keep showing a failed turn that has since been retried.
   */
  onOpenTurn: (seq: number) => void;
}

/**
 * The quiet period that collapses a burst of keystrokes into one report.
 *
 * `/api/rooms/{id}/typing` is a read-modify-write of the whole room plus a bus publish,
 * and it used to run once per keystroke. The report is a timestamp
 * (`note_human_activity`), so one per burst carries exactly as much information as forty.
 */
export const TYPING_REPORT_INTERVAL_MS = 1500;

/**
 * What a row's words are drawn in, for the two shapes §3.2.3 distinguishes.
 *
 * A bubble is a distinguishing device, so it appears where there is something to
 * distinguish -- which is two things, not one.
 *
 * *Who is speaking*: in a multi-agent conversation the left side holds two or more senders,
 * so those rows need a boundary of their own. In a 1:1 the left side holds exactly one, and
 * an agent's rows there carry no bubble -- it would draw a border around a fact the layout
 * already states.
 *
 * *Which words are the reader's own*: the user's rows carry a bubble in **both** shapes. The
 * 1:1 case used to rely on right alignment alone, which separates the two senders but does
 * not make either findable at a glance -- a wall of alternating prose, each paragraph flush
 * to one edge or the other. The bubble is what a reader scrolling back is looking for, so it
 * is not conditional on how many agents happen to be seated.
 *
 * One fill for both sides, deliberately. A second colour for the user's rows would be a
 * third device saying what the bubble and the alignment already say together.
 */
const bubbleClass = (shape: ConversationShape, mine: boolean): string | false =>
  (mine || shape === 'multi') && 'rounded-lg bg-slate-800/50 px-3 py-2';

/**
 * What a speaker said, rendered as markdown (#1225).
 *
 * Retiring `PlaygroundTab` (#1234) took the conversation's markdown with it. That surface put
 * every message through `RichText`; the row that replaced it printed `message.content` into a
 * `<p>`, so a fenced block arrived as its own backticks and a table as a column of pipes.
 * §3.2.5's "what must survive" list named the composer, the history controls and the model
 * selection, and not this -- which is how the loss passed review rather than being argued.
 *
 * One component for both render sites, the settled row and the turn in flight, because they
 * are the same body at two moments. Fixing only the settled one leaves the reader watching a
 * reply stream in as raw pipes and then snap into a table when it lands.
 *
 * `RichText` carries the rest and is not re-implemented here: `TableScroll` on a table wider
 * than the column (#1010, #1015, #1018), `PreBlock` on a fenced block, and
 * `[overflow-wrap:anywhere]` for the unbroken token `break-words` used to catch (#1007).
 *
 * Model output is untrusted and is treated as such by the renderer already: `RichText` mounts
 * no `rehype-raw`, so raw HTML in a reply is never parsed into elements, and react-markdown's
 * default `urlTransform` strips a `javascript:` href before the anchor override is reached.
 * Adding `rehype-raw` here would undo both.
 *
 * What the `<p>` did that markdown does not is keep a single newline as a line break. That is
 * the retired surface's behaviour too, not a new loss: a paragraph break is two newlines.
 */
const MessageBody: React.FC<{ text: string }> = ({ text }) => (
  <RichText
    content={text}
    className={cn(
      'text-sm text-slate-200 prose-sm',
      'prose-p:text-slate-200 prose-li:text-slate-200 prose-strong:text-slate-100',
      'prose-headings:text-slate-100 prose-td:text-slate-200 prose-th:text-slate-100',
      // A bubble sizes itself; prose's outer margins would pad it a second time.
      '[&>*:first-child]:mt-0 [&>*:last-child]:mb-0',
    )}
  />
);

const TranscriptRow: React.FC<{
  room: RoomState;
  message: RoomTranscriptMessage;
  shape: ConversationShape;
  /** Present only on a turn that prints its attribution -- see `attributionLines`. */
  attribution: string | undefined;
  onRetry: () => void;
  /**
   * Show this turn's record in the workspace dock. The row keeps the control and gives up
   * the rendering: a turn's provenance is what the reader picked, and the dock is where
   * this workspace already puts what has been picked -- the same move the rail's clone
   * selection makes (#1300).
   */
  onOpenTurn: (seq: number) => void;
}> = ({ room, message, shape, attribution, onRetry, onOpenTurn }) => {
  const label = senderLabel(room, message.sender_id);
  const mine = senderKind(room, message.sender_id) === 'human';
  const isAgent = senderKind(room, message.sender_id) === 'agent';
  // This turn's own model, whether or not the turn prints a line. FR-13.4 requires it to
  // be reachable *on the turn*, and the header is a statement about the conversation, not
  // an answer for a particular row. Read from `provenance`, never from what was said.
  const servedHere = servedByLabel(message.provenance);
  const showWhy = Boolean(message.decision || servedHere || (isAgent && message.turn_id));
  const unsavedNotice = memoryUnsavedNotice(
    label,
    message.memory_facts_tried,
    message.memory_facts_unsaved,
  );

  return (
    <div
      data-testid={`row-${message.seq}`}
      className={cn('py-2 flex flex-col', mine ? 'items-end' : 'items-start')}
    >
      {/* The sender is named where the name is news. The user's own turns are aligned to
          the right, which is the one distinction that never needs a label -- they are the
          only participant whose messages they already know the author of. In a 1:1 the
          agent needs no label either: the header names it, and the two sides alternate. */}
      {shape === 'multi' && !mine ? (
        <div className="flex items-center gap-1.5 pb-1 text-xs font-medium text-slate-300">
          <Avatar
            label={label}
            kind={senderKind(room, message.sender_id)}
            agentIcon={Bot}
            size="xs"
            // A clone wears the same picture in the transcript as in the rail and on its
            // profile, because a reader who learned it in one place has learned it. Only a
            // clone has one: the endpoint 404s for anything else, and the kit falls back to
            // its single default, which is a kind marker rather than a likeness.
            imageSrc={
              senderKind(room, message.sender_id) === 'agent'
                ? personaAvatarUrl(message.sender_id)
                : undefined
            }
          />
          {label}
        </div>
      ) : null}

      {/* A turn that failed is marked on the row, not only in its words (#1228).
          `PlaygroundTab` marked one with a `turn-chip` carrying `data-state`; #1234 retired
          that surface and the prose came across without the marker, which left a reader
          scanning a long conversation for the turn that failed with nothing to scan -- in a
          1:1 `bubbleClass` draws an agent no bubble either, so a failed row and a good
          one were the same box with different sentences in it.

          A flat rule down the left edge, in the kit's `danger` border, and `data-state` for
          anything reading the DOM. No pulse and no glow: the row already says what happened
          and offers the way forward, so the mark's whole job is to be findable while the
          reader is looking past it (Visual Restraint; #1061 names the same rule for the
          rail). It is additive -- the prose is untouched, because that is what a screen
          reader reads, and a border is not something it can announce. `pl-3` matches the
          gutter `bubbleClass` gives a group's bubble, so the rule costs the text no room. */}
      <div
        data-testid={`row-body-${message.seq}`}
        data-state={message.error ? 'failed' : undefined}
        className={cn(
          'min-w-0',
          mine || shape === 'multi' ? 'max-w-[85%]' : 'w-full',
          bubbleClass(shape, mine),
          message.error && 'border-l-2 border-rose-800 pl-3',
        )}
      >
        {message.error ? (
          <div data-testid={`row-error-${message.seq}`} className="text-sm text-slate-300">
            {/* A plain sentence, never `message.error`: that field is the raw cause, kept
                for the log's reader (#1408). */}
            <p>{turnFailureSentence(label, message.refusal, message.completed)}</p>
            {message.refusal ? (
              <p data-testid={`row-remedy-${message.seq}`} className="mt-1 text-xs text-slate-400">
                {refusalRemedy(message.refusal)}
              </p>
            ) : (
              <Button data-testid="retry-turn" onClick={onRetry} className="mt-1">
                Retry
              </Button>
            )}
          </div>
        ) : message.content.trim() === '' ? (
          /* A turn that said nothing is not a turn with nothing to say (P6). The room
             recorded three of these on 2026-09-21: `completed: true`, `error: null`,
             `content: ""` -- while the image tool had written a 600KB file. The row drew an
             empty bubble, which reads as a rendering bug rather than as what happened.
             Two sentences, because a turn that ended early and a turn that finished with no
             words are different facts, and only the reader can tell which one they meant to
             get. Retry either way: neither is a refusal, so asking again is the remedy. */
          <div data-testid={`row-silent-${message.seq}`} className="text-sm text-slate-300">
            <p>
              {message.completed
                ? `${label} finished this turn without sending any text.`
                : `${label} stopped partway through this turn and sent no text.`}
            </p>
            <Button data-testid="retry-turn" onClick={onRetry} className="mt-1">
              Retry
            </Button>
          </div>
        ) : (
          <MessageBody text={message.content} />
        )}

        {/* The reply stands, but the clone's own record of this turn was not saved, so it
            will not remember it after a restart (#1366). One quiet line, not an error: the
            turn did not fail, and retrying would answer again rather than save this. The
            field's text is the cause, for the log's reader; it is not shown here (#1408). */}
        {message.persist_error ? (
          <p data-testid={`row-unsaved-${message.seq}`} className="mt-1 text-[11px] text-slate-500">
            {label} will not remember this turn after a restart; it could not be saved.
          </p>
        ) : null}

        {/* Its knowledge is a second record, saved beside the session and lost on its own
            (#1367). The same quiet line, naming what is lost: what it learned, not the turn.
            The field's text is the cause, for the log's reader; it is not shown here. */}
        {message.knowledge_persist_error ? (
          <p
            data-testid={`row-knowledge-unsaved-${message.seq}`}
            className="mt-1 text-[11px] text-slate-500"
          >
            {label} will not know what it learned in this turn after a restart; it could not be
            saved.
          </p>
        ) : null}

        {/* Its knowledge record for this conversation could not be read, so it was set
            aside in this turn -- kept under another name, not deleted (#1367). Only that
            record: the clone's saved memory facts are another file, untouched (#1434). Said
            once, on this row: a record that could not be read is not one that never was (P6). */}
        {message.knowledge_set_aside ? (
          <p
            data-testid={`row-knowledge-reset-${message.seq}`}
            className="mt-1 text-[11px] text-slate-500"
          >
            {label}&apos;s knowledge record for this conversation could not be read, so it was
            set aside and kept.
          </p>
        ) : null}

        {/* The clone asked to remember something and some of it was never saved --
            whatever the reply above says (#1375). The Core counts this from the turn's own
            tool results; the reply is the model's and is not edited. Quiet, like the line
            above: the turn did not fail. The tool's own error is written for the model and
            stays in the tool history, never here. */}
        {unsavedNotice ? (
          <p data-testid={`row-memory-unsaved-${message.seq}`} className="mt-1 text-[11px] text-slate-500">
            {unsavedNotice}
          </p>
        ) : null}

        {/* Attribution comes from the turn, never from the text or from whichever agent
            happens to be selected. A local model once reported itself as another vendor's.
            `served_by` is a provider/model pair, and is rendered as a string built from it
            -- handing React the object itself blank-screened the conversation.

            Which turns carry a line at all is `attributionLines`': every one of them in a
            group, none in a 1:1 whose model has never changed, and every one of them again
            once it has. `why ›` is offered wherever there is anything to disclose, which is
            a wider condition than "prints a line": a 1:1 reply prints no line and its model
            still has to be reachable on it.

            It **opens the dock's Turn surface** rather than disclosing under the row. The
            row is not the place: it held three short lines of a record with more than
            three in it, and expanding one moved every turn below it while the reader was
            reading. The dock is what this workspace already opens for the thing that was
            picked, it keeps the record on screen while the conversation is scrolled, and
            it leaves one rendering of the turn rather than two -- which is the reason
            the retired `ToolDetail` surface gave for its own existence, having watched the
            pair it replaced drift apart. */}
        {attribution || showWhy ? (
          <div
            className={cn(
              'mt-1 flex flex-wrap items-center gap-x-2 text-[11px] text-slate-500',
              shape === 'multi' ? 'justify-end' : 'justify-start',
            )}
          >
            {attribution ? (
              <span data-testid={`served-by-${message.seq}`} className="flex items-center gap-1">
                {attribution}
              </span>
            ) : null}

            {showWhy ? (
              <button
                type="button"
                data-testid={`why-${message.seq}`}
                onClick={() => onOpenTurn(message.seq)}
                className="hover:text-slate-300"
              >
                why ›
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
};

/**
 * A roster change, in the conversation where the question about it gets asked.
 *
 * `RoomMessageKind` is `utterance | join | leave` and the three are not interchangeable:
 * putting a membership row through `TranscriptRow` renders the room's own bookkeeping as
 * something a participant said.
 */
const MembershipRow: React.FC<{ room: RoomState; message: RoomTranscriptMessage }> = ({
  room,
  message,
}) => (
  <p data-testid={`membership-${message.seq}`} className="py-1.5 text-[11px] text-slate-500">
    {senderLabel(room, message.sender_id)}{' '}
    {message.kind === 'join' ? 'joined this conversation' : 'left this conversation'}
  </p>
);

/**
 * Whether a seat in a room can carry a model of its own.
 *
 * It cannot, and the picker below is held off the screen until it can. `POST
 * /api/rooms/{room_id}/messages` takes a body of `{content}` alone, and the turn is served
 * by the seated persona's own configured model, resolved server-side in
 * `src/uclone_x/room/resolver.py`. So a choice made here changed nothing, and the option
 * that named the fallback -- `Use default (<the workspace default>)` -- named a model that
 * does not serve the turn either. Both halves of the control lied, which is worse than
 * the control's absence.
 *
 * Held rather than deleted: whether a seat *should* carry a model is an open product
 * question on  When the room route
 * carries one, this flips to `true` and the control comes back with its wiring intact.
 */
const A_ROOM_SEAT_CAN_CARRY_A_MODEL: boolean = false;

/**
 * How far above the transcript's bottom edge still counts as "at the bottom".
 *
 * A reader who stops a few pixels short -- a trackpad's last flick, a line's worth -- is
 * still following, and is treated as such.
 */
const FOLLOW_BOTTOM_SLACK_PX = 32;

/**
 * One conversation, whether it seats one agent or four.
 *
 * The header carries the participants, because they belong to this conversation rather
 * than to the application. The composer says who will answer *before* the message is
 * sent -- the first thing a person wants to know in a room with several agents, and the
 * thing uclone2's group chat never answered.
 */
export const RoomConversation: React.FC<RoomConversationProps> = ({
  room,
  availableAgents,
  live,
  draft,
  onDraftChange,
  onSend,
  onStop,
  onRetry,
  onAddAgent,
  onTyping,
  context = null,
  onCompact,
  compacting = false,
  onClearHistory,
  participantsNotReset = [],
  availableModels,
  currentModel,
  agentModelOverride,
  onSelectAgentModel,
  onToggleAutonomous,
  onOpenTurn,
}) => {
  const isAutonomous = Boolean(room.policy?.autonomous);

  useEffect(() => {
    if (!isAutonomous) return;

    const report = (active: boolean) => {
      void roomsApi.reportPresence(room.room_id, active).catch(() => {});
    };

    if (document.visibilityState === 'visible') {
      report(true);
    }

    const onVisibility = () => {
      report(document.visibilityState === 'visible');
    };
    const onFocus = () => report(true);
    const onBlur = () => {
      if (document.visibilityState !== 'visible') {
        report(false);
      }
    };

    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('focus', onFocus);
    window.addEventListener('blur', onBlur);

    const interval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        report(true);
      }
    }, 15000);

    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('focus', onFocus);
      window.removeEventListener('blur', onBlur);
      clearInterval(interval);
      report(false);
    };
  }, [room.room_id, isAutonomous]);

  const [caret, setCaret] = useState(0);
  const [completionOpen, setCompletionOpen] = useState(false);
  const [inviting, setInviting] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [showSilenceReason, setShowSilenceReason] = useState(false);
  /**
   * The destructive act waiting for a second click, if any.
   *
   * A clear removes messages for good, and used to be one click on a control whose label
   * did not say how much would go.
   */
  const [pending, setPending] = useState<{ kind: 'clear' } | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  // What is in the box right now, for the browser's speech callbacks. They close over the
  // render that built them, and a phrase heard two words later would otherwise be appended
  // to the draft as it stood when the mic was pressed -- discarding everything typed since.
  const draftRef = useRef(draft);
  draftRef.current = draft;

  // ---- chat auto-scroll (adapted from uclone2 MessageList) ------------------------
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const lastRoomIdRef = useRef<string | undefined>(room.room_id);
  const isInitialLoadRef = useRef(true);
  const prevTranscriptLengthRef = useRef(room.transcript.length);
  const prevLiveTurnRef = useRef(Boolean(live.turn));
  const prevLiveTextRef = useRef(live.turn?.text ?? '');
  /**
   * Whether the reader is following the conversation's bottom edge.
   *
   * Auto-scroll is for a reader who is watching the newest words arrive. A reader who has
   * scrolled up is reading something older, and every streamed delta used to put them back
   * at the bottom -- so the history could not be read while a clone was answering, and a
   * reply landing late snapped away whatever they had scrolled to.
   */
  const followingRef = useRef(true);
  /** Where the transcript was the last time this component or the reader left it. */
  const lastTopRef = useRef(0);
  /**
   * Whether words have arrived below a reader who scrolled up, since they did (#1380).
   *
   * Not following is correct -- the reader is reading something older -- but it left them
   * no way to know the reply had moved on. State rather than a ref because it is drawn: it
   * is what puts "Jump to latest" on screen. It is the head's own, like the scroll offset
   * it is about (P8); a second head would not need it.
   */
  const [unseenBelow, setUnseenBelow] = useState(false);

  // Reset initial load flag when conversation room changes
  useLayoutEffect(() => {
    if (room.room_id !== lastRoomIdRef.current) {
      lastRoomIdRef.current = room.room_id;
      isInitialLoadRef.current = true;
      followingRef.current = true;
      setUnseenBelow(false);
    }
  }, [room.room_id]);

  /**
   * The turn in flight, as the transcript draws it: once, whichever channel got here first.
   *
   * One reply reaches this component by two routes that do not arrive in step (#1379). The
   * landed row comes with a read of the room -- the send's own re-read can return it while
   * the stream is still delivering deltas -- and the bubble is cleared by the stream's
   * `final`, which can be far behind. So the bubble is dropped as soon as the record holds a
   * row carrying its `turn_id`: the Core's record is what the conversation is (P8), and the
   * row holds the whole of what the bubble was still spelling out. Keyed on the turn, never
   * on "some row is newer", because a second clone's reply streaming under the first one's
   * landed row is a different reply.
   *
   * The other order leaves a gap: `final` clears the bubble at once and the row arrives with
   * the re-read it triggers, so the reply vanished for a round trip and came back. The last
   * words streamed are held until the record next changes -- to the row, which replaces
   * them, or to anything else (a clear, a rewind), which is the Core's answer and wins. Only
   * words are held, never a status: a "Thinking..." left behind a turn that has ended would
   * claim work nobody is doing (P6). A cascade failure drops them, since that has its own line.
   */
  const heldRef = useRef<{ turn: RoomLiveTurn; transcript: RoomTranscriptMessage[] } | null>(
    null,
  );
  if (live.turn) {
    heldRef.current =
      live.turn.text.trim() !== '' ? { turn: live.turn, transcript: room.transcript } : null;
  } else if (
    heldRef.current &&
    (heldRef.current.transcript !== room.transcript || live.error !== null)
  ) {
    heldRef.current = null;
  }
  const inFlight = live.turn ?? heldRef.current?.turn ?? null;
  const liveTurn =
    inFlight && !room.transcript.some((row) => row.turn_id === inFlight.turnId) ? inFlight : null;

  /**
   * Is the reader still following the bottom edge, as of this instant?
   *
   * Read from the element at the moment it is needed, not only from the scroll event: a
   * scroll the reader made lands its event on the next frame, and a streamed delta rendered
   * before then would otherwise read the old answer and scroll them straight back. Moving
   * back to the bottom resumes following; moving up from where the transcript was left
   * stops it. Only the reader moves it up -- this component only ever scrolls down, and a
   * transcript that shrinks under a reader at the bottom leaves them at the bottom.
   */
  const followReader = useCallback((el: HTMLElement): boolean => {
    const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
    const currentTop = Math.min(el.scrollTop, maxScroll);
    const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
    const isScrollingUp = currentTop < lastTopRef.current - 1;
    const isScrollingDown = currentTop > lastTopRef.current + 1;
    const isAtBottom = gap <= FOLLOW_BOTTOM_SLACK_PX;

    if (isScrollingUp) {
      // The reader deliberately moved up from where the transcript was left.
      // Auto-scroll stops immediately, even if they are still within the bottom slack.
      followingRef.current = false;
    } else if (isAtBottom && (followingRef.current || isScrollingDown)) {
      // The reader reached the bottom edge while following, or scrolled back down to it.
      followingRef.current = true;
      setUnseenBelow(false);
    }
    lastTopRef.current = currentTop;
    return followingRef.current;
  }, []);

  /**
   * Put the transcript's bottom edge in view.
   *
   * Every caller decides for itself whether the reader is following, so this only scrolls.
   * It used to carry a throttle for smooth scrolls that were not forced, with a timer that
   * re-asked `followReader` when it fired; no caller ever asked for one -- a new row, a
   * started turn and a send all force, and a streamed delta is instant -- so the timer and
   * the check inside it never ran (#1380).
   */
  const scrollToBottom = useCallback((behavior: ScrollBehavior) => {
    const container = transcriptRef.current;
    if (!container) return;
    if (typeof container.scrollTo === 'function') {
      container.scrollTo({ top: container.scrollHeight, behavior });
    } else {
      container.scrollTop = container.scrollHeight;
    }
    const maxScroll = Math.max(0, container.scrollHeight - container.clientHeight);
    lastTopRef.current = Math.min(container.scrollTop, maxScroll);
  }, []);

  useLayoutEffect(() => {
    const hasItems = room.transcript.length > 0 || Boolean(liveTurn);
    const liveText = liveTurn?.text ?? '';
    if (hasItems) {
      const container = transcriptRef.current;
      const isNewMessage = room.transcript.length > prevTranscriptLengthRef.current;
      const isTurnStarted = Boolean(liveTurn) && !prevLiveTurnRef.current;
      if (isInitialLoadRef.current) {
        followingRef.current = true;
        scrollToBottom('auto');
        isInitialLoadRef.current = false;
      } else if (container && !followReader(container)) {
        // The reader scrolled up to read something older. New words do not take them away
        // from it; scrolling back to the bottom, sending, or "Jump to latest" resumes
        // following. What they are told is that there is more below.
        if (isNewMessage || isTurnStarted || liveText.length > prevLiveTextRef.current.length) {
          setUnseenBelow(true);
        }
      } else if (isNewMessage || isTurnStarted) {
        scrollToBottom('smooth');
      } else if (liveTurn) {
        // Instant for streamed deltas: a smooth scroll per token jitters.
        scrollToBottom('auto');
      }
    }
    prevTranscriptLengthRef.current = room.transcript.length;
    prevLiveTurnRef.current = Boolean(liveTurn);
    prevLiveTextRef.current = liveText;
  }, [room.transcript, liveTurn, scrollToBottom, followReader]);

  /** Back to the newest words, following again -- what the reader asked for by pressing it. */
  const jumpToLatest = () => {
    followingRef.current = true;
    setUnseenBelow(false);
    scrollToBottom('smooth');
  };

  const agents = room.participants.filter((p) => p.kind === 'agent');

  // One conversation, two shapes, chosen by participant count -- not two components
  // (§3.2.3, §3.2.5). Seating a second clone re-renders the whole transcript, including
  // the turns already in it, because the thing that changed is what needs distinguishing.
  const shape = conversationShape(room.participants);
  const attribution = useMemo(
    () => attributionLines(room.transcript, shape),
    [room.transcript, shape],
  );
  // The 1:1 states its model once, above every row it is true of -- and only while there
  // is one model to state, because a header is a standing claim about every row beneath
  // it. Once a second model has served, `headerAttribution` reports none and the rows
  // carry their own, exactly as a group's do.
  const headerServedBy = shape === 'solo' ? headerAttribution(room.transcript) : null;

  /**
   * Who will answer what is in the box, resolved once for the two things that say so.
   *
   * The hint says it in words and the strip beside it shows that seat's figures; both read
   * this, because a strip that named a different seat than the sentence next to it would
   * be a disagreement the reader has no way to settle.
   */
  const who = useMemo(
    () =>
      answeringSeats(
        room.participants,
        draft,
        room.policy.default_responder_id,
        room.policy.max_agent_turns_per_human_message,
      ),
    [
      room.participants,
      room.policy.default_responder_id,
      room.policy.max_agent_turns_per_human_message,
      draft,
    ],
  );

  /**
   * The one seat about to answer, when exactly one is.
   *
   * The strip is keyed off this rather than off `shape`, which is the whole of the answer
   * to "what does a group do here": a group addressed at one clone is in the same position
   * as a 1:1 and gets the same strip, and a 1:1 is simply the case where there is never
   * anyone else to address. Where several will answer there is no single model and no
   * single context to show, and inventing a combined figure for them would attribute to
   * one ceiling what belongs to several (P6).
   */
  const soleAnswerer = who.kind === 'seats' && who.seats.length === 1 ? who.seats[0] : null;
  const answererContext =
    soleAnswerer && context
      ? (context.seats.find((seat) => seat.participant_id === soleAnswerer.id) ?? null)
      : null;
  /**
   * The ring's denominator, when there is a measured one.
   *
   * Both halves must be present: a window with no count, or a count with no window, is
   * not a fraction, and `?? 0` on either would draw a full ring for an empty seat or an
   * empty ring for a full one. `?? null` rather than `=== null` on each, because a server
   * that predates these fields sends neither, and "absent" and "explicitly unknown" are
   * the same fact here -- nothing was measured.
   */
  const tokenWindow =
    answererContext &&
    (answererContext.used_tokens ?? null) !== null &&
    (answererContext.max_context_tokens ?? null) !== null &&
    (answererContext.max_context_tokens ?? 0) > 0
      ? {
          used: answererContext.used_tokens ?? 0,
          max: answererContext.max_context_tokens ?? 0,
          source: answererContext.context_window_source ?? null,
        }
      : null;
  /**
   * The model named beside the composer: the last one that actually served.
   *
   * Not the workspace default and not any override held in this head. A room turn is
   * served by the seated persona's own configured model (`A_ROOM_SEAT_CAN_CARRY_A_MODEL`),
   * so those name something else, and this surface may only name what a turn reported
   * (FR-13.4).
   */
  const servedModel = lastServedModel(room.transcript);
  /**
   * What the strip beside Send names, or `null` when the header is already naming it.
   *
   * In the ordinary 1:1 -- one seat, one model, every turn served by it -- `servedModel`
   * and `headerServedBy` are the same string, and the screen printed it twice: once under
   * the conversation's title and once again an inch above the button. Two copies of one
   * fact do not make it twice as true; they make a reader check whether they differ.
   *
   * The header keeps it, because it is the standing claim -- true of every row beneath it.
   * The strip speaks exactly when the header cannot: no turn has been served yet, or a
   * second model has served and the header has gone quiet. `null` is not the same as
   * "nothing has served": that case still says so in words (P6), which is why it is tested
   * against `servedModel` rather than folded into a `??`.
   */
  const stripModel =
    servedModel !== null && servedModel === headerServedBy
      ? null
      : (servedModel ?? 'No answer yet');

  const dictation = useDictation(
    useCallback(
      (phrase: string) => onDraftChange(appendPhrase(draftRef.current, phrase)),
      [onDraftChange],
    ),
  );

  const hint = useMemo(
    () =>
      answerHint(
        room.participants,
        draft,
        room.policy.default_responder_id,
        room.policy.max_agent_turns_per_human_message,
      ),
    [
      room.participants,
      room.policy.default_responder_id,
      room.policy.max_agent_turns_per_human_message,
      draft,
    ],
  );

  const ceilingReached =
    room.turn_state.agent_turns_since_human >= room.policy.max_agent_turns_per_human_message;
  const running = live.turn !== null;
  const hasSpoken = room.transcript.some((message) => message.kind === 'utterance');

  // The conversation decided nobody speaks. Nothing is published for that, so without
  // this the surface shows your own message and then nothing, forever. Only while that
  // message is still the last thing said: a silence also ends every answered exchange.
  const silence = unansweredSilence(room);

  /**
   * The one seated clone, when there is exactly one.
   *
   * The model picker is a standing claim about who answers here, in the same way the
   * header's attribution line is a standing claim about what served the rows beneath it.
   * With two clones seated there is no single answer, so the header states none and the
   * control is not offered.
   */
  const soleAgent = agents.length === 1 ? agents[0] : null;

  const saturatedSeats = (context?.seats ?? []).filter((seat) => seat.is_saturated);
  const seatLabel = (participantId: string): string =>
    senderLabel(room, participantId);

  /**
   * How many messages the asked-about change would remove, which is what the confirmation
   * states. The control that opens it cannot carry the number, and clearing is final.
   */
  const removedByPending = pending === null ? 0 : room.transcript.length;

  const invitable = availableAgents.filter(
    (agent) => !room.participants.some((p) => p.id === agent.id),
  );

  // ---- typing, collapsed to one report per burst ---------------------------------
  const typingArmedRef = useRef(true);
  const typingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (typingTimerRef.current !== null) clearTimeout(typingTimerRef.current);
    },
    [],
  );
  const reportTyping = useCallback(() => {
    if (typingTimerRef.current !== null) clearTimeout(typingTimerRef.current);
    typingTimerRef.current = setTimeout(() => {
      typingArmedRef.current = true;
      typingTimerRef.current = null;
    }, TYPING_REPORT_INTERVAL_MS);
    if (!typingArmedRef.current) return;
    typingArmedRef.current = false;
    onTyping();
  }, [onTyping]);

  // ---- `@` completion ------------------------------------------------------------
  const mention = completionOpen ? mentionUnderCaret(draft, caret) : null;
  const candidates = mention ? mentionCandidates(room.participants, mention.prefix) : [];

  // Escape closes the mention menu, but only when no higher-precedence layer claims it
  // first (#1036) -- a dialog or a running turn elsewhere must win over this.
  // The confirmation strip is the other claimant at this layer, and the two are made
  // mutually exclusive here rather than left to the "most recently registered wins"
  // tie-break, which `escapePrecedence` documents as an undesigned case.
  useEscapeOwner('overlay', Boolean(mention && candidates.length > 0) && pending === null, () =>
    setCompletionOpen(false),
  );

  // Escape abandons a clear that has been asked about and not answered. It is
  // an overlay and not a dialog: a running turn's Stop must still outrank it, and while a
  // turn runs the confirmation's own action is refused anyway.
  useEscapeOwner('overlay', pending !== null, () => setPending(null));

  const acceptCompletion = (participant: RoomParticipant) => {
    if (!mention) return;
    const head = draft.slice(0, mention.start);
    const tail = draft.slice(mention.start + 1 + mention.prefix.length);
    const inserted = `@${participant.id} `;
    onDraftChange(`${head}${inserted}${tail}`);
    setCaret(mention.start + inserted.length);
    setCompletionOpen(false);
    composerRef.current?.focus();
  };

  const submit = async () => {
    const content = draft;
    if (content.trim() === '' || sending) return;
    // Sending is the reader returning to the conversation's edge: what comes back is followed
    // even if they had scrolled up to read before writing.
    followingRef.current = true;
    setSending(true);
    setSendError(null);
    try {
      await onSend(content);
      onDraftChange('');
      setCaret(0);
      setCompletionOpen(false);
      setUnseenBelow(false);
      scrollToBottom('smooth');
    } catch (err) {
      // The draft is deliberately left where it was. A refusal that also erases what you
      // wrote costs you the message twice. The line says only what is known: "Not sent" for
      // a refusal, and never for a send that got no answer (#1441).
      setSendError(sendFailureNotice(err));
    } finally {
      setSending(false);
    }
  };

  return (
    <section data-testid="room-conversation" className="column-scope flex flex-col h-full min-h-0">
      {/* The header holds more than a narrow pane fits, so the order it gives way in is
          stated rather than left to whichever item happens to be least shrinkable. The
          title keeps a floor and truncates; the controls wrap to their own row and the
          model name truncates there. Before this the title was the item that yielded, and
          a pane under ~300px showed a conversation with no name on it. */}
      <header className="flex flex-wrap items-center gap-x-2 gap-y-1 px-4 py-2 border-b border-slate-800/60">
        <h2 className="min-w-[8rem] max-w-full truncate text-sm font-medium text-slate-200">
          {room.title}
        </h2>
        <div
          data-testid="participant-strip"
          className="flex min-w-0 shrink items-center gap-2 overflow-hidden text-xs text-slate-400"
        >
          {agents.map((agent) => (
            <span key={agent.id} className="flex items-center gap-1">
              <Avatar
                label={participantLabel(agent)}
                kind="agent"
                agentIcon={Bot}
                size="2xs"
                imageSrc={personaAvatarUrl(agent.id)}
              />
              <span>{participantLabel(agent)}</span>
              {live.turn?.agentId === agent.id ? (
                <span data-testid={`writing-${agent.id}`} className="text-slate-500">
                  Writing…
                </span>
              ) : null}
            </span>
          ))}
          {/* FR-13.4 as amended: the record is kept on every turn, and printed where
              printing carries information. In a 1:1 that is here -- one value, true of
              every row beneath it -- and on the turns that depart from it. */}
          {headerServedBy ? (
            <span data-testid="header-attribution" className="text-slate-500">
              · {headerServedBy}
            </span>
          ) : null}
        </div>
        {/* Which model answers here, chosen from what the runtime reports rather than
            typed (P0, recognition over recall). It is the conversation's, not the
            workspace's: the app-wide default is one control in Settings, and a second
            copy of it in every conversation header would read as a local choice while
            writing a global one.

            Not on screen today -- see `A_ROOM_SEAT_CAN_CARRY_A_MODEL` above and #1235. */}
        <div className="ml-auto flex min-w-0 shrink items-center gap-2">
        {A_ROOM_SEAT_CAN_CARRY_A_MODEL &&
        soleAgent &&
        onSelectAgentModel &&
        availableModels &&
        availableModels.length > 0 ? (
          <label className="flex min-w-0 shrink items-center gap-1.5 text-xs text-slate-400">
            <Cpu className="w-3.5 h-3.5 shrink-0" />
            <span className="sr-only">Model for this conversation</span>
            <select
              data-testid="room-model-select"
              value={agentModelOverride ?? '__default__'}
              onChange={(event) =>
                onSelectAgentModel(
                  soleAgent.id,
                  event.target.value === '__default__' ? null : event.target.value,
                )
              }
              className="min-w-0 max-w-full truncate bg-transparent text-xs text-slate-300 outline-none"
            >
              <option value="__default__" className="bg-slate-950 text-slate-200">
                {currentModel ? `Use default (${currentModel})` : 'Use the default model'}
              </option>
              {availableModels.map((model) => (
                <option key={model} value={model} className="bg-slate-950 text-slate-200">
                  {model}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        {onToggleAutonomous ? (
          <Button
            data-testid="toggle-autonomous"
            onClick={() => onToggleAutonomous(!isAutonomous)}
            aria-label={isAutonomous ? '자율 토론 모드 켜짐' : '자율 토론 모드 꺼짐'}
            title={
              isAutonomous
                ? '자율 토론 활성화됨 (화면을 보고 있는 동안 에이전트 간 연속 대화)'
                : '자율 토론 켜기 (화면을 보고 있는 동안 에이전트 간 연속 대화)'
            }
            className={cn(
              'shrink-0 whitespace-nowrap text-xs flex items-center gap-1.5 transition-colors',
              isAutonomous
                ? 'bg-emerald-600 hover:bg-emerald-500 text-white font-medium shadow-sm'
                : 'text-slate-400 hover:text-slate-200',
            )}
          >
            <Sparkles className={cn('w-3 h-3', isAutonomous ? 'text-amber-300' : 'text-slate-400')} />
            <span className="column-icon-only">자율 토론</span>
            <span className="text-[10px] uppercase tracking-wider font-semibold opacity-90">
              {isAutonomous ? 'ON' : 'OFF'}
            </span>
          </Button>
        ) : null}

        <Button
          data-testid="add-someone"
          onClick={() => setInviting((open) => !open)}
          aria-label="Add someone"
          className="shrink-0 whitespace-nowrap"
        >
          <Plus className="w-3 h-3" />
          <span className="column-icon-only">Add someone</span>
        </Button>

        {onClearHistory ? (
          <Button
            data-testid="clear-history"
            onClick={() => setPending({ kind: 'clear' })}
            aria-label="Clear this conversation"
            title="Clear this conversation"
            className="shrink-0 whitespace-nowrap"
          >
            <Trash2 className="w-3 h-3" />
            <span className="column-icon-only">Clear</span>
          </Button>
        ) : null}
        </div>
      </header>

      {inviting ? (
        <div data-testid="invite-list" className="px-4 py-2 border-b border-slate-800/60 flex flex-wrap gap-1">
          {invitable.length === 0 ? (
            // Two different absences with two different remedies. The runtime reports no
            // clones at all until one has been used, so a conversation opened on a fresh
            // install was told everybody was already here -- which was not true, and left
            // nothing to do about it.
            <span data-testid="invite-empty-cause" className="text-xs text-slate-500">
              {availableAgents.length === 0
                ? 'No clones are running yet. Send a message in a chat first, and whoever answers can be added here.'
                : 'Everyone available is already in this conversation.'}
            </span>
          ) : (
            invitable.map((agent) => (
              <Button
                key={agent.id}
                data-testid={`invite-${agent.id}`}
                onClick={() => {
                  onAddAgent(agent.id);
                  setInviting(false);
                }}
              >
                <Avatar
                  label={agent.label || agent.id}
                  kind="agent"
                  agentIcon={Bot}
                  size="2xs"
                  imageSrc={personaAvatarUrl(agent.id)}
                />
                {agent.label || agent.id}
              </Button>
            ))
          )}
        </div>
      ) : null}

      {/* A reading column, not the window.

          Both this and the composer below are capped at the same 768px and centred. A line
          of our 16px prose runs about 7.4px per character, so an unconstrained row on a
          1920px screen was 222 characters wide -- three times the width prose is readable
          at, and the reader's eye lost its place returning to the left edge. The cap is the
          sibling product's `max-w-3xl`, and the composer carries it too so that the column
          the reader writes into is the column they just read. */}
      <div
        ref={transcriptRef}
        data-testid="transcript"
        onScroll={(e) => {
          followReader(e.currentTarget);
        }}
        className="flex-1 overflow-y-auto min-h-0 px-4 mx-auto w-full max-w-3xl"
      >
        {room.transcript.map((message) =>
          message.kind === 'utterance' ? (
            <TranscriptRow
              key={message.seq}
              room={room}
              message={message}
              shape={shape}
              attribution={attribution.get(message.seq)}
              onRetry={onRetry}
              onOpenTurn={onOpenTurn}
            />
          ) : (
            <MembershipRow key={message.seq} room={room} message={message} />
          ),
        )}

        {/* A conversation New has just opened has nothing in it, and a transcript with no
            branch for that rendered the screen's primary region as a blank box. Joins are
            not speech, so a conversation holding only those is still waiting to begin, and
            the line sits under them. Two sentences, because "nobody has spoken" and
            "nobody is here to speak" have different remedies. */}
        {!hasSpoken && !inFlight && !live.error ? (
          <p data-testid="transcript-empty" className="py-6 text-sm text-slate-500">
            {agents.length === 0
              ? 'No one is in this conversation yet. Add someone to get a reply.'
              : 'Nothing has been said here yet. Send a message to start.'}
          </p>
        ) : null}

        {liveTurn ? (
          // A turn in flight is drawn the way the turn it becomes will be, or the
          // transcript reflows under the reader the moment it lands.
          <div data-testid="live-turn" className="py-2 flex flex-col items-start">
            {shape === 'multi' ? (
              <div className="flex items-center gap-1.5 pb-1 text-xs font-medium text-slate-300">
                <Avatar
                  label={senderLabel(room, liveTurn.agentId)}
                  kind="agent"
                  agentIcon={Bot}
                  size="xs"
                  imageSrc={personaAvatarUrl(liveTurn.agentId)}
                />
                {senderLabel(room, liveTurn.agentId)}
              </div>
            ) : null}
            <div
              data-testid="live-turn-body"
              className={cn(
                'min-w-0',
                shape === 'multi' ? 'max-w-[85%]' : 'w-full',
                // Always an agent's words, never the reader's: a turn in flight is one the
                // room is answering with.
                bubbleClass(shape, false),
              )}
            >
              {!liveTurn.text.trim() ? (
                <div
                  data-testid="live-turn-indicator"
                  className="flex items-center gap-2 text-slate-400 py-1"
                >
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-400 shrink-0" aria-label="Thinking" />
                  <span className="text-xs font-mono text-slate-400">
                    {liveTurn.statusText || 'Thinking...'}
                  </span>
                </div>
              ) : (
                <>
                  <MessageBody text={liveTurn.text} />
                  {liveTurn.statusText && liveTurn.statusText !== 'Thinking...' ? (
                    <div
                      data-testid="live-turn-status"
                      className="flex items-center gap-1.5 pt-1.5 mt-1 border-t border-slate-700/40 text-[11px] font-mono text-cyan-400"
                    >
                      <Loader2 className="w-3 h-3 animate-spin text-cyan-400 shrink-0" />
                      <span>{liveTurn.statusText}</span>
                    </div>
                  ) : null}
                </>
              )}
            </div>
          </div>
        ) : null}

        {live.error ? (
          <p data-testid="cascade-error" className="py-2 text-sm text-slate-300">
            This conversation stopped: {live.error}
          </p>
        ) : null}

        {ceilingReached && !running ? (
          <p data-testid="ceiling-notice" className="py-2 text-xs text-slate-500">
            Paused after {room.policy.max_agent_turns_per_human_message} replies. Send a message
            to continue.
          </p>
        ) : null}

        {silence && !running && !ceilingReached && !live.error ? (
          <div data-testid="silence-notice" className="py-2 text-xs text-slate-500">
            <p>
              No one answered your last message, and no one is going to. Address someone with
              @ to get a reply.
            </p>
            <button
              type="button"
              data-testid="why-silence"
              onClick={() => setShowSilenceReason((open) => !open)}
              className="mt-1 text-slate-500 hover:text-slate-300"
            >
              why ›
            </button>
            {showSilenceReason ? (
              <div data-testid="why-silence-detail" className="mt-1">
                <p>Decided by {silence.selector}.</p>
                {silence.reasoning ? <p>{silence.reasoning}</p> : null}
              </div>
            ) : null}
          </div>
        ) : null}

        {/* New words below a reader who scrolled up (#1380). The transcript leaves them where
            they are, which is right, and used to leave them with no way to know the reply
            had moved on. One quiet control, drawn only once there is something below to
            point at -- "new text below" before any arrived would be a claim with nothing
            behind it. No pulse and nothing that grows: it is found by being there when the
            reader looks, not by moving. Sticky inside the scroller rather than a layer over
            it, so it costs the column no structure and cannot cover the composer. A plain
            button, so Tab reaches it and Enter or Space presses it. */}
        {unseenBelow ? (
          <div className="sticky bottom-2 flex h-0 items-end justify-center">
            <Button
              data-testid="jump-to-latest"
              onClick={jumpToLatest}
              className="bg-slate-900 text-slate-200 shadow-sm"
            >
              <ArrowDown className="w-3 h-3" aria-hidden="true" />
              Jump to latest
            </Button>
          </div>
        ) : null}
      </div>

      {/* A seat the last cut did not reach. Named rather than left silent: it is still
          answering from turns this transcript no longer holds, which is the divergence
          the rewind and the clear exist to prevent. */}
      {participantsNotReset.length > 0 ? (
        <div className="border-t border-slate-800/60">
          <p
            data-testid="participants-not-reset"
            className="px-4 py-2 text-[11px] text-slate-300 mx-auto w-full max-w-3xl"
          >
            {participantsNotReset.map(seatLabel).join(', ')}{' '}
            {participantsNotReset.length === 1 ? 'still remembers' : 'still remember'} what was
            removed here, so a reply may refer to it. Send a message to start again, or clear
            this conversation.
          </p>
        </div>
      ) : null}

      {/* Read from the Core's per-seat turn count, never estimated from the transcript.
          The conversation is what stops being able to continue, so one full seat is
          enough for the banner. */}
      {context?.is_saturated ? (
        <div
          data-testid="saturation-banner"
          className="border-t border-slate-800/60"
        >
          <div className="px-4 py-2 text-[11px] text-slate-300 mx-auto w-full max-w-3xl">
            <p>
              {saturatedSeats.map((seat) => seatLabel(seat.participant_id)).join(', ')} reached the{' '}
              {context.saturation_threshold}-turn limit on context. Shorten this conversation to
              keep going, or start a new one.
            </p>
            {/* Which copy the figure came from. A seat nothing has spoken in since this
                server started answers from its saved record, which is behind any turn an
                earlier run did not write (P6). */}
            {saturatedSeats.some((seat) => !seat.live) ? (
              <p data-testid="saturation-from-record" className="mt-1 text-slate-500">
                Counted from the saved record for{' '}
                {saturatedSeats
                  .filter((seat) => !seat.live)
                  .map((seat) => seatLabel(seat.participant_id))
                  .join(', ')}
                , which nothing has spoken in since this server started.
              </p>
            ) : null}
            {onCompact ? (
              <Button
                data-testid="compact-room"
                onClick={() => void onCompact()}
                disabled={compacting || running}
                className="mt-1"
              >
                {compacting ? 'Shortening\u2026' : 'Shorten this conversation'}
              </Button>
            ) : null}
            {running ? (
              <p data-testid="compact-blocked" className="mt-1 text-slate-500">
                A turn is running. Shortening waits until it finishes, so nothing is cut out
                from under it.
              </p>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* The destructive control asks once, and the question names how much goes. A
          clear removes messages for good; the label on the control that starts it
          cannot carry that number, so the confirmation does. */}
      {pending ? (
        <div
          data-testid="confirm-history-change"
          className="border-t border-slate-800/60"
        >
          <div className="px-4 py-2 text-[11px] text-slate-300 mx-auto w-full max-w-3xl">
            <p>
              {`Clear this conversation? Its ${
                removedByPending === 1 ? 'one message is' : `${removedByPending} messages are`
              } removed for good. It keeps its name and who is in it.`}
            </p>
            {running ? (
              <p data-testid="confirm-blocked" className="mt-1 text-slate-500">
                A turn is running, so this would be refused. It can be done once that
                finishes.
              </p>
            ) : null}
            <div className="mt-1 flex items-center gap-2">
              <Button
                data-testid="confirm-history-change-yes"
                disabled={running}
                onClick={() => {
                  setPending(null);
                  void onClearHistory?.();
                }}
              >
                Clear
              </Button>
              <Button data-testid="confirm-history-change-no" onClick={() => setPending(null)}>
                Keep everything
              </Button>
            </div>
          </div>
        </div>
      ) : null}

      <div data-testid="composer-column" className="px-4 pt-2 pb-3 mx-auto w-full max-w-3xl">
        {mention && candidates.length > 0 ? (
          <div data-testid="mention-completion" className="mb-1 flex flex-wrap gap-1">
            {candidates.map((participant, index) => (
              <Button
                key={participant.id}
                data-testid={`mention-option-${participant.id}`}
                onClick={() => acceptCompletion(participant)}
                // Enter takes the first of these. A keystroke that picks one item out of a
                // row which draws them all alike is a rule the reader can only learn by
                // being surprised, so the one it picks is lit -- quietly, a brighter border
                // and brighter text, no halo and nothing that moves.
                data-enter-takes={index === 0 ? 'true' : undefined}
                className={index === 0 ? 'border-slate-400 text-slate-100' : undefined}
              >
                @{participant.id}
                {participant.display_name && participant.display_name !== participant.id ? (
                  <span className="text-slate-500"> · {participant.display_name}</span>
                ) : null}
              </Button>
            ))}
          </div>
        ) : null}

        {/* The box the composer is typed into, drawn rather than implied.

            It was a bare transparent textarea under a hairline, which is the one shape on
            this screen that states nothing: a reader arriving at the bottom could not see
            where the input began or how much of the width was theirs. The rounded border
            is the same device the reader's own rows now carry above it -- one shape for
            "these are your words", in the transcript and in the box.

            The hairline above it goes with it. Two separations for one boundary is one too
            many, and the box's own border is the stronger of the two. */}
        <div
          data-testid="composer-box"
          className="rounded-2xl border border-slate-700/70 bg-slate-900/40 px-3 py-2"
        >
          <textarea
            data-testid="room-composer"
            ref={composerRef}
            // Ported from the retired `PlaygroundTab` (#1208), which carried #114/#339's
            // load-time focus. The conversation is the screen's one subject and the
            // composer is its one input, so a reader who starts typing types into it
            // rather than nowhere. It is the only focusable thing that autofocuses, so
            // nothing else is stolen from.
            autoFocus
            value={draft}
            rows={2}
            onChange={(event) => {
              onDraftChange(event.target.value);
              setCaret(event.target.selectionStart ?? event.target.value.length);
              setCompletionOpen(true);
              setSendError(null);
              reportTyping();
            }}
            onClick={(event) => setCaret(event.currentTarget.selectionStart ?? 0)}
            // Enter sends; Shift+Enter starts a line. The box had no key handler at all, so
            // Enter only ever grew the draft and the message could be sent one way: by
            // travelling to a button at the far corner of the box. That is the opposite of
            // the convention every chat surface a reader arrives from already taught them,
            // and the cost falls on the most ordinary action the screen has.
            onKeyDown={(event) => {
              if (event.key !== 'Enter' || event.shiftKey) return;
              // A Korean, Japanese or Chinese keyboard spends Enter on the characters it is
              // still assembling -- the key that settles which hanja or kana was meant. The
              // browser marks that keystroke as composing, and sending on it would post a
              // half-written word *and* swallow the press the writer aimed at the IME.
              if (event.nativeEvent.isComposing) return;
              // An open name menu is what makes Enter mean "take this one". `@cr` resolves to
              // nobody, so sending it produces a message that looks addressed and reaches no
              // seat; the first candidate is the one taken, and it is drawn as such.
              if (mention && candidates.length > 0) {
                event.preventDefault();
                acceptCompletion(candidates[0]);
                return;
              }
              // While a turn is running the control in the strip is Stop; there is no Send to
              // mirror. Enter falls through to the textarea and starts a line, which keeps
              // what was typed rather than spending the key on nothing.
              if (running) return;
              event.preventDefault();
              void submit();
            }}
            placeholder="Send a message"
            className="w-full bg-transparent text-sm text-slate-200 outline-none resize-none placeholder:text-slate-600"
          />

          {/* Inside the box, along its bottom edge: how to put words in it, what the room
              will do with them, who will answer and on what, and the control that sends it.

              These were a separate row underneath, which drew the send button as far from
              the text it sends as the screen allows and left the box looking like a field
              with an unrelated toolbar beneath it. The reference product keeps one shape --
              the box holds the words and everything done to them, with send at its bottom
              right -- and that is what a reader's hand expects after typing. */}
          {/* Wrapping, because in a 160px column the strip cannot hold a mic, a sentence,
              a model name, a ring and a send button on one line, and a row that does not wrap
              put the send button 47px past the column's right edge, where a click reached the
              transcript behind it (#1029's property, pinned at 160px in
              `tests/e2e/test_room_conversation_layout_e2e.py`). Truncating each piece toward
              nothing instead would leave the model named by a few pixels of ellipsis, which is
              the empty container P6 forbids. So the pieces keep their size and take a second
              line, and send stays at the right of whichever line it lands on. */}
          <div
            data-testid="composer-controls"
            className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1"
          >
            <Button
              data-testid="dictate"
              onClick={() => {
                if (dictation.state.kind === 'error') dictation.dismissError();
                dictation.toggle();
              }}
              aria-label={dictation.state.kind === 'listening' ? 'Stop listening' : 'Speak the message'}
              aria-pressed={dictation.state.kind === 'listening'}
              className={cn(
                'shrink-0 py-1',
                // Plain amber while it is listening. The sibling pulses the button and draws
                // a pinging halo on it; this surface forbids both, and a state that is either
                // on or off does not need an animation to say which.
                dictation.state.kind === 'listening' ? 'text-amber-300' : 'text-slate-400',
              )}
            >
              <Mic className="w-3 h-3" />
              <span className="column-icon-only">
                {dictation.state.kind === 'listening' ? 'Listening' : 'Speak'}
              </span>
            </Button>

            <span
              data-testid="answer-hint"
              className="min-w-0 flex-1 truncate text-[11px] text-slate-500"
            >
              {hint}
            </span>

            {/* What the answer will cost and what it last came from, next to the button that
                asks for it. This is not the rail's budget panel returning: that was a standing
                readout of every clone whether or not the reader was about to spend anything,
                which is the instrument #1059 took off U0's default screen. This is one seat's
                figure, at the moment of spending, for the seat that is about to spend it. */}
            {soleAnswerer ? (
              /* Wrapping for the same reason the row above it wraps, and shrinking where that
                 row's pieces do not: at a 320px column this readout is wider than the whole
                 column, so as one unshrinkable piece it took a line of its own and still ran
                 16px past the edge (#1010's property, the four narrow cases in
                 `tests/e2e/test_room_conversation_layout_e2e.py`). The pieces inside keep
                 their size and take a further line rather than each being squeezed toward an
                 ellipsis. */
              <span
                data-testid="answering-seat"
                className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-slate-500"
              >
                {stripModel !== null ? (
                  <span
                    data-testid="answering-model"
                    className="min-w-0 max-w-[12rem] shrink-0 truncate"
                  >
                    {stripModel}
                  </span>
                ) : null}
                {answererContext && context ? (
                  <span className="flex shrink-0 items-center gap-1">
                    {/* The digits the ring is a picture of, **and the unit it counts in**.
                        A ring alone is a proportion of nothing the reader can name, which
                        is the empty container P6 forbids -- and a bare `3/20` beside a
                        composer is read as tokens, which is what it was read as.

                        It counts tokens when the seat's window was measured, because that
                        is the ceiling a long conversation reaches first and the one the
                        reader is watching for. Where no window was measured it counts
                        turns, against the ceiling this runtime enforces, and the unit is
                        written out either way so the two readings can never be confused
                        for one another. */}
                    <FillRing
                      data-testid="answering-context-ring"
                      filled={
                        tokenWindow
                          ? tokenWindow.used / tokenWindow.max
                          : answererContext.active_turns / context.saturation_threshold
                      }
                      warn={
                        answererContext.is_saturated ||
                        (tokenWindow !== null && tokenWindow.used / tokenWindow.max >= 0.9)
                      }
                      label={
                        tokenWindow
                          ? `${participantLabel(soleAnswerer)} has used ${tokenWindow.used.toLocaleString()} of ${tokenWindow.max.toLocaleString()} tokens of context`
                          : `${participantLabel(soleAnswerer)} has used ${answererContext.active_turns} of ${context.saturation_threshold} turns of context`
                      }
                    />
                    <span
                      data-testid="answering-context-count"
                      title={
                        tokenWindow === null
                          ? undefined
                          : tokenWindow.source === 'loaded'
                            ? 'The server serving this model reported this context window when it loaded it.'
                            : tokenWindow.source === 'published'
                              ? 'The published context window for this model, which its provider enforces.'
                              : undefined
                      }
                    >
                      {tokenWindow
                        ? `${tokenWindow.used.toLocaleString()}/${tokenWindow.max.toLocaleString()} tokens`
                        : `${answererContext.active_turns}/${context.saturation_threshold} turns`}
                    </span>
                  </span>
                ) : null}
                {answererContext && context ? (
                  <span
                    data-testid="answering-tokens"
                    className="shrink-0"
                    title={
                      // `?? null` and not `=== null`: a server that predates the field, or
                      // any read that did not carry it, leaves `undefined` here, and that
                      // is the same fact as an explicit `null` -- no count was reported.
                      // Treating the two differently is how the missing-key case reached
                      // the browser as a crash rather than as a sentence.
                      tokenWindow !== null
                        ? (answererContext.cumulative_tokens ?? null) !== null
                          ? `Active context: ${tokenWindow.used.toLocaleString()} / ${tokenWindow.max.toLocaleString()} tokens. Session spend: ${(answererContext.cumulative_tokens ?? 0).toLocaleString()} tokens.`
                          : undefined
                        : (answererContext.used_tokens ?? null) === null
                          ? 'Token counts are kept for the current run. This seat has not answered since it started, so nothing has been booked against it here.'
                          : 'No context window was reported for this model, so the ring counts turns instead of tokens.'
                    }
                  >
                    {/* Whichever fact the ring is not showing. Both ceilings are real and a
                        seat can reach either first, so neither is dropped: with a window
                        measured the ring takes the tokens and this keeps the turns; with no
                        window it keeps the token count, which is what this said before the
                        ring had a denominator at all. */}
                    {tokenWindow
                      ? `${answererContext.active_turns}/${context.saturation_threshold} turns`
                      : (answererContext.used_tokens ?? null) === null
                        ? 'tokens not counted this run'
                        : // Grouped the way the Budget surface writes the same quantity. A
                          // second format for one figure is how two readings of it start
                          // looking like two quantities.
                          `${(answererContext.used_tokens ?? 0).toLocaleString()} tokens`}
                  </span>
                ) : null}
              </span>
            ) : who.kind === 'seats' ? (
              <span
                data-testid="answering-several"
                className="shrink-0 text-[11px] text-slate-500"
              >
                Each answers from its own context
              </span>
            ) : null}
            {running ? (
              <Button
                data-testid="stop-turn"
                onClick={onStop}
                aria-label="Stop"
                className="ml-auto shrink-0 text-slate-200 py-1"
              >
                <Square className="w-3 h-3" />
                <span className="column-icon-only">Stop</span>
              </Button>
            ) : (
              <Button
                data-testid="send-message"
                disabled={draft.trim() === '' || sending}
                onClick={() => {
                  void submit();
                }}
                aria-label="Send (Enter)"
                className="ml-auto shrink-0 text-slate-200 py-1"
              >
                <Send className="w-3 h-3" />
                <span className="column-icon-only">Send</span>
              </Button>
            )}
          </div>
        </div>

        {sendError ? (
          <p data-testid="send-error" className="pt-1.5 text-[11px] text-slate-300">
            {sendError}
          </p>
        ) : null}

        {dictation.state.kind === 'unsupported' || dictation.state.kind === 'error' ? (
          // The mic's refusals, in words on the surface. The sibling product writes these
          // to the console and puts the unsupported case in a `title` on a greyed button,
          // which a touch device cannot reach at all -- so a reader who presses the mic and
          // gets nothing is told nothing about why, or about what would let it through.
          <p data-testid="dictation-message" className="pt-1.5 text-[11px] text-slate-300">
            {dictation.state.kind === 'unsupported'
              ? dictation.state.reason
              : dictation.state.message}
          </p>
        ) : null}
      </div>
    </section>
  );
};
