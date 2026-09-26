import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import { RoomConversation, TYPING_REPORT_INTERVAL_MS } from './RoomConversation';
import type { AgentInfo, RoomContext, RoomState, RoomTranscriptMessage } from '../../types';
import { EMPTY_LIVE, RoomSendError, RoomsApiError, RoomsNoAnswerError } from '../../lib/rooms';
import { ABSENCE_CLAIM } from '../../lib/absenceGuard';
import { makeAgentInfo } from '../../test/fixtures';
import { expectPlain } from '../../test/plainCopy';
import failedRows from '../../test/room-failed-rows.json';

const message = (over: Partial<RoomTranscriptMessage>): RoomTranscriptMessage => ({
  seq: 1,
  sender_id: 'user',
  content: 'hello',
  kind: 'utterance',
  created_at: '2026-09-14T00:00:00Z',
  completed: true,
  ...over,
});

/**
 * The roster carries `display_name`s on purpose.
 *
 * Every attribution test below used ids alone, so `senderLabel` could be replaced with
 * `(_room, senderId) => senderId` and the whole file still passed -- the fixtures made
 * the resolved name and the raw id the same string.
 */
const room = (over: Partial<RoomState> = {}): RoomState => ({
  room_id: 'r1',
  title: 'Architecture triage',
  participants: [
    { id: 'user', kind: 'human', display_name: 'Kenny' },
    { id: 'scout', kind: 'agent', display_name: 'Scout' },
    { id: 'critic', kind: 'agent', display_name: 'Critic' },
  ],
  transcript: [message({ seq: 1 })],
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

const agents: AgentInfo[] = [makeAgentInfo({ id: 'dba', label: 'DBA', role: 'dba' })];

/**
 * The composer's text, owned above the component the way `App` owns it (#1290).
 *
 * `draft` is a controlled prop now, because the component does not survive a conversation
 * switch and state kept inside it was destroyed by one. A fixed `draft=""` here would make
 * the box unwritable and every typing case below vacuous, so the harness holds the value
 * for the same reason the real owner does.
 */
const DraftHost: React.FC<{
  initial: string;
  props: Omit<React.ComponentProps<typeof RoomConversation>, 'draft' | 'onDraftChange'>;
}> = ({ initial, props }) => {
  const [draft, setDraft] = React.useState(initial);
  return <RoomConversation {...props} draft={draft} onDraftChange={setDraft} />;
};

const renderRoom = (over: Partial<React.ComponentProps<typeof RoomConversation>> = {}) => {
  const { draft = '', onDraftChange: _ignored, ...rest } = over;
  return render(
    <DraftHost
      initial={draft}
      props={{
        room: room(),
        availableAgents: agents,
        live: EMPTY_LIVE,
        onSend: () => {},
        onStop: () => {},
        onRetry: () => {},
        onAddAgent: () => {},
        onTyping: () => {},
        onOpenTurn: () => {},
        ...rest,
      }}
    />,
  );
};

afterEach(() => {
  vi.useRealTimers();
});

describe('RoomConversation attribution', () => {
  it('names each row by its own sender, never by another participant', () => {
    // The defect this pins was paid for in uclone2: resolving a row through "the selected
    // agent" rendered one agent's name over another's words, permanently. Two agent rows
    // with different names, so one label for all rows cannot pass.
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: 'the composite index is fine' }),
          message({ seq: 2, sender_id: 'critic', content: 'the partial index is missing' }),
        ],
      }),
    });

    expect(screen.getByTestId('row-1')).toHaveTextContent('Scout');
    expect(screen.getByTestId('row-2')).toHaveTextContent('Critic');
    // No `not.toHaveTextContent('scout')` here: there is no "selected agent" concept in
    // this component for a regression to come from, so that assertion could not fail.
  });

  it('calls the human "you" rather than by the roster name', () => {
    // On a roster row, which is where the human is still named at all: their own turns
    // carry no name now, because the right-hand side of the screen is the name (§3.2.3).
    renderRoom({
      room: room({ transcript: [message({ seq: 1, sender_id: 'user', kind: 'join', content: '' })] }),
    });

    expect(screen.getByTestId('membership-1').textContent).toContain('you');
    expect(screen.getByTestId('membership-1').textContent).not.toContain('Kenny');
  });

  it('shows an unresolvable sender as its id rather than inventing a name', () => {
    renderRoom({
      room: room({ transcript: [message({ seq: 1, sender_id: 'departed', content: 'bye' })] }),
    });

    expect(screen.getByTestId('row-1')).toHaveTextContent('departed');
  });

  it('renders who served the turn as a string built from the provider and model', () => {
    // `provenance.served_by` is a `ServiceRef` object. Rendering it as a React child
    // threw "Objects are not valid as a React child" and blank-screened the conversation
    // on the very first agent reply.
    renderRoom({
      room: room({
        transcript: [
          message({
            seq: 1,
            sender_id: 'scout',
            content: 'the index is fine',
            provenance: {
              path: 'primary',
              requested: { provider: 'ollama', model: 'qwen3:8b' },
              served_by: { provider: 'ollama', model: 'qwen3:8b' },
              degraded: false,
            },
          }),
        ],
      }),
    });

    expect(screen.getByTestId('served-by-1')).toHaveTextContent('ollama:qwen3:8b');
  });

  it('keeps the selector trail out of the default view, and hands it to the dock', () => {
    const opened: number[] = [];
    renderRoom({
      onOpenTurn: (seq) => opened.push(seq),
      room: room({
        transcript: [
          message({
            seq: 1,
            sender_id: 'scout',
            decision: {
              verdict: 'speak',
              speaker_id: 'scout',
              confidence: 0.94,
              selector: 'LLMSpeakerSelector',
              reasoning: 'the question is about indexes',
            },
          }),
        ],
      }),
    });

    // The trail is not under the row in either state now. `why ›` names the turn to the
    // owner, which opens the dock's Turn surface on it; the transcript is not pushed around
    // to read a record that never fitted under a row anyway.
    expect(screen.queryByTestId('why-detail-1')).toBeNull();
    fireEvent.click(screen.getByTestId('why-1'));
    expect(opened).toEqual([1]);
    expect(screen.queryByTestId('why-detail-1')).toBeNull();
  });

  it('renders a roster change as a roster change, not as something somebody said', () => {
    // `RoomMessageKind` is `utterance | join | leave`. Putting a membership row through
    // the speech renderer puts the room's own bookkeeping in a participant's mouth.
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'user', content: 'check it' }),
          message({ seq: 2, sender_id: 'critic', kind: 'join', content: '' }),
        ],
      }),
    });

    expect(screen.getByTestId('membership-2')).toHaveTextContent('Critic joined this conversation');
    expect(screen.queryByTestId('row-2')).toBeNull();
  });
});

/**
 * The two shapes of §3.2.3 [Rev 20], and the rules that choose between them.
 *
 * Revision 3's surface had every row left-aligned behind a name column -- the user's own
 * included -- and an attribution line under every reply. In a 1:1 that spent a fifth of
 * the width restating two facts that alternate strictly, and printed one unchanging model
 * name under each of forty turns. Both are changed, and the change is the *same*
 * component rendering either shape: what a 1:1 and a group differ by is participant
 * count, not which file drew them (§3.2.5).
 */
describe('RoomConversation shape', () => {
  const solo = (over: Partial<RoomState> = {}): RoomState =>
    room({
      participants: [
        { id: 'user', kind: 'human', display_name: 'Kenny' },
        { id: 'scout', kind: 'agent', display_name: 'Scout' },
      ],
      ...over,
    });

  const served = (provider: string, model: string) => ({
    path: 'primary' as const,
    requested: { provider, model },
    served_by: { provider, model },
    degraded: false,
  });

  const exchange = [
    message({ seq: 1, sender_id: 'user', content: 'check the index' }),
    message({
      seq: 2,
      sender_id: 'scout',
      content: 'the composite index is unused',
      provenance: served('ollama', 'qwen3:8b'),
    }),
  ];

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: const mine = senderKind(room, message.sender_id) === 'human';
  // Becomes: const mine = false;
  it('aligns the user`s own turns right and every agent`s left', () => {
    // The one distinction that never needs a label: the user is the only participant
    // whose messages they already know the author of.
    renderRoom({ room: solo({ transcript: exchange }) });

    expect(screen.getByTestId('row-1')).toHaveClass('items-end');
    expect(screen.getByTestId('row-2')).toHaveClass('items-start');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {shape === 'multi' && !mine ? (
  // Becomes: {!mine ? (
  it('names nobody in a 1:1, where the two sides alternate and the header says who', () => {
    renderRoom({ room: solo({ transcript: exchange }) });

    expect(screen.getByTestId('row-2')).toHaveTextContent('the composite index is unused');
    expect(screen.getByTestId('row-2').textContent).not.toContain('Scout');
    expect(screen.getByTestId('row-1').textContent).not.toContain('you');
  });

  // Killed by: frontend/src/lib/rooms.ts :: return participants.filter((p) => p.kind === 'agent').length > 1 ? 'multi' : 'solo';
  // Becomes: return 'solo';
  it('names every agent turn once a second clone is seated, and re-draws the ones already there', () => {
    // R2: seating a second clone re-renders the whole transcript, including turns taken
    // while it was still a 1:1. The thing that changed is what needs distinguishing, and
    // that is a property of the conversation, not of the individual message.
    renderRoom({
      room: room({
        transcript: [
          ...exchange,
          message({ seq: 3, sender_id: 'critic', content: 'the partial one is still hit' }),
        ],
      }),
    });

    expect(screen.getByTestId('row-2')).toHaveTextContent('Scout');
    expect(screen.getByTestId('row-3')).toHaveTextContent('Critic');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: ? personaAvatarUrl(message.sender_id)
  // Becomes: ? undefined
  it('draws each clone`s own picture beside its name in the transcript', () => {
    // A clone wears one picture everywhere it appears -- the rail, its profile, and here --
    // because a reader who learned it in one place has learned it. The transcript row was
    // the one place that could not take one: the two clones were told apart by their names
    // alone while the rail beside them showed their faces.
    //
    // Both rows are read, and they are different clones, because a picture keyed on
    // anything but the sender passes on one row.
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: 'the partial index is missing' }),
          message({ seq: 2, sender_id: 'critic', content: 'the partial one is still hit' }),
        ],
      }),
    });

    expect(screen.getByTestId('row-1').querySelector('img')).toHaveAttribute(
      'src',
      '/api/personas/scout/avatar',
    );
    expect(screen.getByTestId('row-2').querySelector('img')).toHaveAttribute(
      'src',
      '/api/personas/critic/avatar',
    );
  });

  // A bubble answers two questions, and the two mutations below drop one each: the first
  // takes the reader's own rows out of a 1:1, the second takes every agent out of a group.
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: (mine || shape === 'multi') && 'rounded-lg bg-slate-800/50 px-3 py-2';
  // Becomes: (shape === 'multi') && 'rounded-lg bg-slate-800/50 px-3 py-2';
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: (mine || shape === 'multi') && 'rounded-lg bg-slate-800/50 px-3 py-2';
  // Becomes: (mine) && 'rounded-lg bg-slate-800/50 px-3 py-2';
  it('bubbles the reader`s own turns in either shape, and an agent`s only in a group', () => {
    // In a 1:1 the alignment already separates the two senders, so an agent's rows carry
    // no bubble -- but alignment alone does not make the reader's own words findable on
    // the way back up a long conversation, which is the bubble's other job.
    const { unmount } = renderRoom({ room: solo({ transcript: exchange }) });
    expect(screen.getByTestId('row-body-1')).toHaveClass('rounded-lg');
    expect(screen.getByTestId('row-body-2')).not.toHaveClass('rounded-lg');
    unmount();

    // In a group the left side holds more than one sender, so those rows need a boundary
    // of their own as well.
    renderRoom({ room: room({ transcript: exchange }) });
    expect(screen.getByTestId('row-body-1')).toHaveClass('rounded-lg');
    expect(screen.getByTestId('row-body-2')).toHaveClass('rounded-lg');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: className="rounded-2xl border border-slate-700/70 bg-slate-900/40 px-3 py-2"
  // Becomes: className="px-3 py-2"
  it('draws the box the composer is typed into, rather than implying it', () => {
    // The same device as the row above it: the composer was a transparent textarea under a
    // hairline, so where the input began and how much of the width was the reader's were
    // both left to be guessed at. Asserted on the box's own classes, because jsdom resolves
    // no stylesheet and a rendered border is not a thing it can be asked about.
    renderRoom({ room: solo({ transcript: exchange }) });

    const box = screen.getByTestId('composer-box');
    expect(box).toHaveClass('rounded-2xl');
    expect(box).toHaveClass('border');
    expect(box).toContainElement(screen.getByTestId('room-composer'));
  });

  // No mutation declared: what this pins is where a node sits in the tree, and there is no
  // single line whose replacement moves the strip back out of the box without unbalancing
  // the JSX -- a parse error fails the whole file, which reads as a kill and is not one.
  it('keeps send and the rest of the strip inside the box the words are typed into', () => {
    // The strip was a row underneath the box, which put the send button as far from the
    // text it sends as the row allows and made the box read as a field with an unrelated
    // toolbar below it. One shape: the box holds the words and everything done to them.
    renderRoom({ room: solo({ transcript: exchange }) });

    const box = screen.getByTestId('composer-box');
    const controls = screen.getByTestId('composer-controls');
    expect(box).toContainElement(controls);
    expect(controls).toContainElement(screen.getByTestId('send-message'));
    expect(controls).toContainElement(screen.getByTestId('dictate'));
    expect(controls).toContainElement(screen.getByTestId('answer-hint'));
    // Send is the strip's last control, so it lands at the bottom right corner of the box.
    expect(controls.lastElementChild).toContainElement(screen.getByTestId('send-message'));
  });

  /** How full each seat is, as `GET /api/rooms/{id}/context` answers it. */
  const contextFor = (
    turns: Record<string, number>,
    threshold = 20,
    tokens: Record<string, number | null> = {},
    windows: Record<string, number | null> = {},
  ): RoomContext => ({
    room_id: 'r1',
    seats: Object.entries(turns).map(([participant_id, active_turns]) => ({
      participant_id,
      session_id: `s-${participant_id}`,
      active_turns,
      is_saturated: active_turns >= threshold,
      // `null` by default, which is what a seat that has not answered in this process
      // reports -- the case the strip has to say something about rather than draw as zero.
      used_tokens: tokens[participant_id] ?? null,
      // Also `null` by default, and for the same reason: a window is reported only where
      // the Core measured one, so the unmeasured seat is the case to default to.
      max_context_tokens: windows[participant_id] ?? null,
      context_window_source: (windows[participant_id] ?? null) === null ? null : 'loaded',
      live: true,
    })),
    is_saturated: Object.values(turns).some((t) => t >= threshold),
    saturation_threshold: threshold,
  });

  // The ring is the only figure on this screen drawn rather than written, so the assertion
  // is on the fraction it was given -- an SVG arc's own geometry is not something jsdom
  // resolves, and a ring drawn to the wrong length still renders.
  // Killed by: frontend/src/ui-kit/primitives/FillRing.tsx :: data-filled={fraction.toFixed(3)}
  // Becomes: data-filled={(1).toFixed(3)}
  it('shows how full the answering seat`s context is, beside Send', () => {
    renderRoom({
      room: solo({ transcript: exchange }),
      context: contextFor({ scout: 5 }),
    });

    // 5 of 20, so a quarter -- and the digits beside it, because a ring on its own is a
    // proportion of something the reader cannot name.
    expect(screen.getByTestId('answering-context-ring')).toHaveAttribute('data-filled', '0.250');
    expect(screen.getByTestId('answering-context-count')).toHaveTextContent('5/20');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: : `${answererContext.active_turns}/${context.saturation_threshold} turns`}
  // Becomes: : `${answererContext.active_turns}/${context.saturation_threshold}`}
  it('names the unit the ring counts in, so its digits cannot be read as tokens', () => {
    // A bare `5/20` an inch above Send is read as tokens -- it was. The ring counts turns,
    // because turns are the ceiling this runtime enforces and compacts against, and a
    // figure whose unit is not on screen is a number the reader has to guess the meaning
    // of. `toHaveTextContent` is a substring match, so `5/20` alone would not fail here;
    // the regex anchors the unit to the digits.
    renderRoom({ room: solo({ transcript: exchange }), context: contextFor({ scout: 5 }) });

    expect(screen.getByTestId('answering-context-count').textContent).toMatch(/5\/20\s*turns/);
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: `${(answererContext.used_tokens ?? 0).toLocaleString()} tokens`}
  // Becomes: `${(answererContext.used_tokens ?? 0).toLocaleString()}`}
  it('writes the seat`s token count as tokens, beside the turns the ring draws', () => {
    // The two quantities sit next to each other, so each one says what it is. This seat
    // has no measured window -- `contextFor` reports none unless one is given -- so the
    // token figure is written and not drawn: the ring is counting turns, and
    // `TokenBudget.max_tokens` is a spend ceiling rather than a ceiling on context.
    renderRoom({
      room: solo({ transcript: exchange }),
      context: contextFor({ scout: 5 }, 20, { scout: 12345 }),
    });

    expect(screen.getByTestId('answering-tokens').textContent).toMatch(/12,345\s*tokens/);
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: ? 'tokens not counted this run'
  // Becomes: ? '0 tokens'
  it('says a seat`s tokens were not counted this run rather than reporting none spent', () => {
    // P6: the budget manager holds one process's bookings, so a conversation reopened
    // after a restart has a full transcript and no record of what it cost. Drawing that as
    // `0 tokens` reports a fresh conversation where there is an expensive one.
    renderRoom({
      room: solo({ transcript: exchange }),
      context: contextFor({ scout: 5 }, 20, { scout: null }),
    });

    expect(screen.getByTestId('answering-tokens')).toHaveTextContent(
      'tokens not counted this run',
    );
  });

  /**
   * The ring's denominator, when the Core measured one.
   *
   * Every case below picks a token fraction that is *not* the turn fraction, because the
   * two are drawn into the same attribute: with `5/20` turns beside `16,384/32,768`
   * tokens, a ring that silently kept counting turns would still read `0.250`, and an
   * assertion that accepted it would be pinning nothing.
   */
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: ? tokenWindow.used / tokenWindow.max
  // Becomes: ? answererContext.active_turns / context.saturation_threshold
  it('draws the ring against the seat`s measured context window when there is one', () => {
    renderRoom({
      room: solo({ transcript: exchange }),
      context: contextFor({ scout: 5 }, 20, { scout: 16_384 }, { scout: 32_768 }),
    });

    // Half the window, against a quarter of the turn ceiling. The ring takes the window.
    expect(screen.getByTestId('answering-context-ring')).toHaveAttribute('data-filled', '0.500');
    expect(screen.getByTestId('answering-context-count').textContent).toMatch(
      /16,384\/32,768\s*tokens/,
    );
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: ? `${participantLabel(soleAnswerer)} has used ${tokenWindow.used.toLocaleString()} of ${tokenWindow.max.toLocaleString()} tokens of context`
  // Becomes: ? `${participantLabel(soleAnswerer)} has used ${tokenWindow.used.toLocaleString()} of ${tokenWindow.max.toLocaleString()} turns of context`
  it('reads the ring out in the unit it is actually counting', () => {
    // The ring is decoration to a screen reader and its label is the whole of what that
    // reader gets. A label that says turns over a ring drawn to tokens tells one reader
    // something the screen does not say to the other.
    renderRoom({
      room: solo({ transcript: exchange }),
      context: contextFor({ scout: 5 }, 20, { scout: 16_384 }, { scout: 32_768 }),
    });

    expect(screen.getByTestId('answering-context-ring')).toHaveAttribute(
      'aria-label',
      'Scout has used 16,384 of 32,768 tokens of context',
    );
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: ? `${answererContext.active_turns}/${context.saturation_threshold} turns`
  // Becomes: ? `${answererContext.active_turns}`
  it('keeps the turn count on screen when the ring has taken the tokens', () => {
    // Both ceilings are real and a seat can reach either first: a short conversation can
    // fill a small window, and a long one can hit the turn limit with the window barely
    // touched. Moving one figure into the ring must not take the other off the screen.
    renderRoom({
      room: solo({ transcript: exchange }),
      context: contextFor({ scout: 5 }, 20, { scout: 16_384 }, { scout: 32_768 }),
    });

    expect(screen.getByTestId('answering-tokens').textContent).toMatch(/5\/20\s*turns/);
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: (answererContext.max_context_tokens ?? 0) > 0
  // Becomes: true
  it('refuses a window of zero as a denominator rather than dividing by it', () => {
    // A server that reports `0` has not reported a window. Taken as one it divides to
    // `Infinity`, which `FillRing` clamps to a full ring -- a seat with 100 tokens drawn
    // as out of room. The turn count is the honest reading and it is what is drawn.
    renderRoom({
      room: solo({ transcript: exchange }),
      context: contextFor({ scout: 5 }, 20, { scout: 100 }, { scout: 0 }),
    });

    expect(screen.getByTestId('answering-context-ring')).toHaveAttribute('data-filled', '0.250');
    // The digits, not the word beside them: naming the unit is *names the unit the ring
    // counts in*'s claim, and asserting it here too would make that test's mutation kill
    // this one as well, which reports one defect as two.
    expect(screen.getByTestId('answering-context-count').textContent).toMatch(/5\/20/);
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: (tokenWindow !== null && tokenWindow.used / tokenWindow.max >= 0.9)
  // Becomes: false
  it('warns on a nearly full window even while the turn count is low', () => {
    // Two turns in, and the window is nine tenths gone -- a seat that pasted a large file
    // is exactly this. Warning only on turns would leave the ring quiet until the model
    // started dropping the conversation's beginning.
    renderRoom({
      room: solo({ transcript: exchange }),
      context: contextFor({ scout: 2 }, 20, { scout: 30_000 }, { scout: 32_768 }),
    });

    expect(
      screen.getByTestId('answering-context-ring').querySelector('.stroke-amber-400'),
    ).not.toBeNull();
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: : 'No context window was reported for this model, so the ring counts turns instead of tokens.'
  // Becomes: : undefined
  it('says why the ring is counting turns when no window was reported', () => {
    // P6: the ring changing unit between two seats is a fact about what could be measured,
    // not a rendering accident, and the surface has to be able to say which.
    renderRoom({
      room: solo({ transcript: exchange }),
      context: contextFor({ scout: 5 }, 20, { scout: 12_345 }),
    });

    expect(screen.getByTestId('answering-tokens')).toHaveAttribute(
      'title',
      'No context window was reported for this model, so the ring counts turns instead of tokens.',
    );
  });

  /**
   * The model is named once on the screen, by whichever of the two places can still say it.
   *
   * In the ordinary 1:1 the header's standing claim and the strip's last-served figure are
   * the same string, and both were printed: once under the title and once an inch above the
   * button. Two copies of one fact do not make it twice as true -- they make a reader stop
   * and check whether they differ.
   */
  describe('naming the model once', () => {
    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: servedModel !== null && servedModel === headerServedBy
    // Becomes: false
    it('leaves it to the header while the header can still speak for every turn', () => {
      renderRoom({ room: solo({ transcript: exchange }), context: contextFor({ scout: 5 }) });

      expect(screen.getByTestId('header-attribution')).toHaveTextContent('qwen3:8b');
      expect(screen.queryByTestId('answering-model')).toBeNull();
      // The ring is not the model, and it is not the header's: it stays.
      expect(screen.getByTestId('answering-context-count')).toHaveTextContent('5/20');
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: servedModel !== null && servedModel === headerServedBy
    // Becomes: servedModel !== null
    it('names it once a second model has served and the header has gone quiet', () => {
      // `headerAttribution` reports `null` the moment two models have served, because a
      // header is a claim about every row beneath it. Nothing would name a model at all
      // if the strip stayed silent here, which is the failure this rule risks.
      renderRoom({
        room: solo({
          transcript: [
            ...exchange,
            message({ seq: 3, sender_id: 'user', content: 'again' }),
            message({
              seq: 4,
              sender_id: 'scout',
              content: 'the partial index is missing',
              provenance: served('ollama', 'hermes3:8b'),
            }),
          ],
        }),
        context: contextFor({ scout: 5 }),
      });

      expect(screen.queryByTestId('header-attribution')).toBeNull();
      expect(screen.getByTestId('answering-model')).toHaveTextContent('hermes3:8b');
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx ::       : (servedModel ?? 'No answer yet');
    // Becomes:       : servedModel;
    it('says nothing has served yet, rather than going blank with the header', () => {
      // Both are silent here for the same reason, and one of them has to say why: an
      // absence states its cause (P6). It must not fall back to a model this head merely
      // has to hand -- a room turn is served by the seated persona's own.
      renderRoom({
        room: solo({ transcript: [message({ seq: 1, sender_id: 'user', content: 'hi' })] }),
        context: contextFor({ scout: 0 }),
      });

      expect(screen.queryByTestId('header-attribution')).toBeNull();
      expect(screen.getByTestId('answering-model')).toHaveTextContent('No answer yet');
    });
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: const soleAnswerer = who.kind === 'seats' && who.seats.length === 1 ? who.seats[0] : null;
  // Becomes: const soleAnswerer = who.kind === 'seats' ? who.seats[0] : null;
  it('shows one seat`s figures only when one seat will answer', () => {
    // A group addressed at one clone is in the same position as a 1:1 -- one model, one
    // context -- so it gets the same strip. The shape of the room is not what decides it.
    const { unmount } = renderRoom({
      room: room({ transcript: exchange }),
      context: contextFor({ scout: 5, critic: 12 }),
      draft: '@scout have another look',
    });
    expect(screen.getByTestId('answer-hint')).toHaveTextContent('Scout will answer');
    expect(screen.getByTestId('answering-context-count')).toHaveTextContent('5/20');
    unmount();

    // Two will answer, from two contexts against two ceilings. One ring here would have to
    // pick a seat or average them, and both state a figure no seat reported.
    renderRoom({
      room: room({ transcript: exchange }),
      context: contextFor({ scout: 5, critic: 12 }),
      draft: '@scout @critic both of you',
    });
    expect(screen.getByTestId('answer-hint')).toHaveTextContent('Scout, Critic will answer');
    expect(screen.queryByTestId('answering-context-ring')).toBeNull();
    expect(screen.queryByTestId('answering-model')).toBeNull();
    expect(screen.getByTestId('answering-several')).toHaveTextContent(
      'Each answers from its own context',
    );
  });

  // jsdom carries no `SpeechRecognition`, which is the unsupported browser the sibling
  // product renders as a greyed button with a `title` -- unreachable on a touch device and
  // silent everywhere. Pressing it here has to put the cause on the screen.
  // Killed by: frontend/src/lib/useDictation.ts :: setState({ kind: 'unsupported', reason: DICTATION_UNSUPPORTED });
  // Becomes: setState({ kind: 'idle' });
  it('says why it cannot listen, rather than a mic that does nothing', () => {
    renderRoom({ room: solo({ transcript: exchange }) });

    expect(screen.queryByTestId('dictation-message')).toBeNull();
    fireEvent.click(screen.getByTestId('dictate'));

    const said = screen.getByTestId('dictation-message').textContent ?? '';
    expect(said).toContain('cannot listen');
    // A cause with no remedy leaves the reader nothing to do about it.
    expect(said).toContain('Type the message');
  });

  /**
   * A browser's recognition object, driven by the test rather than by a queue.
   *
   * It fires nothing on its own: every result and every error below is pushed in by the
   * case that wants it. A fake that plays a script instead would keep passing after the
   * component stopped listening to it.
   */
  class FakeRecognition {
    static last: FakeRecognition | null = null;
    lang = '';
    continuous = false;
    interimResults = true;
    started = false;
    onresult: ((event: { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> }) => void) | null = null;
    onerror: ((event: { error: string }) => void) | null = null;
    onend: (() => void) | null = null;
    constructor() {
      FakeRecognition.last = this;
    }
    start() {
      this.started = true;
    }
    stop() {
      this.started = false;
      this.onend?.();
    }
    abort() {
      this.started = false;
    }
  }

  const heard = (transcript: string, isFinal: boolean) => ({
    resultIndex: 0,
    results: [Object.assign([{ transcript }], { isFinal })],
  });

  // Killed by: frontend/src/lib/useDictation.ts :: if (result.isFinal) phraseRef.current(result[0].transcript);
  // Becomes: if (false) phraseRef.current(result[0].transcript);
  // Killed by: frontend/src/lib/useDictation.ts :: session.interimResults = false;
  // Becomes: session.interimResults = true;
  it('adds what was heard to the end of what is already in the box', () => {
    vi.stubGlobal('SpeechRecognition', FakeRecognition);
    try {
      renderRoom({ room: solo({ transcript: exchange }), draft: 'check the' });
      fireEvent.click(screen.getByTestId('dictate'));

      const session = FakeRecognition.last;
      expect(session?.started).toBe(true);
      // Interim results off: a box that rewrites itself while the reader is still
      // speaking cannot be edited, and the half-heard words in it are not what is sent.
      expect(session?.interimResults).toBe(false);

      act(() => session?.onresult?.(heard('index', true)));
      expect(screen.getByTestId('room-composer')).toHaveValue('check the index');
    } finally {
      vi.unstubAllGlobals();
    }
  });

  // Killed by: frontend/src/lib/useDictation.ts :: setState({ kind: 'error', message: dictationErrorMessage(event.error) });
  // Becomes: setState({ kind: 'idle' });
  it('says on the screen why listening stopped, and what would let it through', () => {
    vi.stubGlobal('SpeechRecognition', FakeRecognition);
    try {
      renderRoom({ room: solo({ transcript: exchange }) });
      fireEvent.click(screen.getByTestId('dictate'));

      const session = FakeRecognition.last;
      act(() => session?.onerror?.({ error: 'not-allowed' }));
      // `onend` follows an error in every implementation, and must not wipe the sentence
      // the error just put up -- that is the silent mic this whole control exists to avoid.
      act(() => session?.onend?.());

      expect(screen.getByTestId('dictation-message')).toHaveTextContent('Allow it for this site');
    } finally {
      vi.unstubAllGlobals();
    }
  });

  // Measured on the running UI at a 1920px viewport before this cap existed: a row of the
  // conversation was 1648px, which at that text's own 7.43px average advance is 222
  // characters. The two mutations below drop the cap from one column each, and each leaves
  // that column free to run the whole window again.
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: className="flex-1 overflow-y-auto min-h-0 px-4 mx-auto w-full max-w-3xl"
  // Becomes: className="flex-1 overflow-y-auto min-h-0 px-4"
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: className="px-4 pt-2 pb-3 mx-auto w-full max-w-3xl"
  // Becomes: className="px-4 pt-2 pb-3"
  it('reads and writes in one centred column rather than across the whole window', () => {
    renderRoom({ room: solo({ transcript: exchange }) });

    // Both columns, not one: a capped transcript over a full-width composer would put the
    // words the reader writes outside the column they just read.
    for (const id of ['transcript', 'composer-column']) {
      const column = screen.getByTestId(id);
      expect(column).toHaveClass('max-w-3xl');
      expect(column).toHaveClass('mx-auto');
    }
  });

  // The anchor is the live-turn block's own shape test, not the transcript row's.
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {shape === 'multi' ? (
  // Becomes: {true ? (
  it('draws a turn still in flight the way the turn it becomes will be drawn', () => {
    // The in-flight block is a second copy of the shape rules, and nothing pinned it: a
    // named, bubbled live turn that loses both the moment it lands reflows the
    // transcript under the reader, which is the defect the copy exists to avoid.
    const live = { turn: { agentId: 'scout', turnId: 't1', text: 'the composite index' }, error: null };

    const { unmount } = renderRoom({ room: solo({ transcript: exchange }), live });
    expect(screen.getByTestId('live-turn').textContent).not.toContain('Scout');
    expect(screen.getByTestId('live-turn-body')).not.toHaveClass('rounded-lg');
    unmount();

    renderRoom({ room: room({ transcript: exchange }), live });
    expect(screen.getByTestId('live-turn')).toHaveTextContent('Scout');
    expect(screen.getByTestId('live-turn-body')).toHaveClass('rounded-lg');
  });

  it('renders thinking indicator while agent is preparing a reply before tokens arrive', () => {
    const live = { turn: { agentId: 'scout', turnId: 't1', text: '', statusText: 'Thinking...' }, error: null };

    renderRoom({ room: solo({ transcript: exchange }), live });
    expect(screen.getByTestId('live-turn-indicator')).toHaveTextContent('Thinking...');
  });

  it('renders intermediate status update when tool is executing during live turn', () => {
    const live = {
      turn: {
        agentId: 'scout',
        turnId: 't1',
        text: 'Partial text',
        statusText: 'Running tool: bash...',
      },
      error: null,
    };

    renderRoom({ room: solo({ transcript: exchange }), live });
    expect(screen.getByTestId('live-turn-status')).toHaveTextContent('Running tool: bash...');
  });
});

/**
 * FR-13.4 as amended (PRD 0.5.0, `97e2d60a`): every turn carries who served it, and the
 * surface prints it where printing carries information.
 */
describe('RoomConversation attribution, printed where it is news', () => {
  const served = (model: string) => ({
    path: 'primary' as const,
    requested: { provider: 'ollama', model },
    served_by: { provider: 'ollama', model },
    degraded: false,
  });

  const soloRoom = (transcript: RoomTranscriptMessage[]): RoomState =>
    room({
      participants: [
        { id: 'user', kind: 'human', display_name: 'Kenny' },
        { id: 'scout', kind: 'agent', display_name: 'Scout' },
      ],
      transcript,
    });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: const headerServedBy = shape === 'solo' ? headerAttribution(room.transcript) : null;
  // Becomes: const headerServedBy = null;
  it('states a 1:1`s model once, in the header, instead of under every turn', () => {
    renderRoom({
      room: soloRoom([
        message({ seq: 1, sender_id: 'scout', content: 'one', provenance: served('qwen3:8b') }),
        message({ seq: 2, sender_id: 'scout', content: 'two', provenance: served('qwen3:8b') }),
      ]),
    });

    expect(screen.getByTestId('header-attribution')).toHaveTextContent('ollama:qwen3:8b');
    expect(screen.queryByTestId('served-by-1')).toBeNull();
    expect(screen.queryByTestId('served-by-2')).toBeNull();
  });

  // Killed by: frontend/src/lib/rooms.ts :: else if (only !== model) return null;
  // Becomes: else if (false) return null;
  it('drops the header and prints every turn once a second model has served', () => {
    // The header is a standing claim about every row beneath it, so it survives only
    // while one model is true of all of them. Rows 1 and 2 print their own model too:
    // under the first-turn anchor they were the rows the header was still true of, and
    // rows 3 and 4 were the rows it silently was not.
    renderRoom({
      room: soloRoom([
        message({ seq: 1, sender_id: 'scout', content: 'one', provenance: served('qwen3:8b') }),
        message({ seq: 2, sender_id: 'scout', content: 'two', provenance: served('qwen3:8b') }),
        message({ seq: 3, sender_id: 'scout', content: 'three', provenance: served('hermes3:8b') }),
        message({ seq: 4, sender_id: 'scout', content: 'four', provenance: served('hermes3:8b') }),
      ]),
    });

    expect(screen.queryByTestId('header-attribution')).toBeNull();
    expect(screen.getByTestId('served-by-1')).toHaveTextContent('ollama:qwen3:8b');
    expect(screen.getByTestId('served-by-2')).toHaveTextContent('ollama:qwen3:8b');
    expect(screen.getByTestId('served-by-3')).toHaveTextContent('ollama:hermes3:8b');
    expect(screen.getByTestId('served-by-4')).toHaveTextContent('ollama:hermes3:8b');
  });

  /** The card's transcript (#1201): one degraded turn landing on the header's own model. */
  const degradedOntoTheHeadersOwnModel = (): RoomTranscriptMessage[] => [
    message({ seq: 1, sender_id: 'scout', content: 'one', provenance: served('qwen3:8b') }),
    message({
      seq: 2,
      sender_id: 'scout',
      content: 'two',
      provenance: {
        path: 'failover',
        requested: { provider: 'ollama', model: 'hermes3:8b' },
        served_by: { provider: 'ollama', model: 'qwen3:8b' },
        degraded: true,
      },
    }),
  ];

  // Killed by: frontend/src/lib/rooms.ts :: const model = serviceRefLabel(message.provenance?.served_by);
  // Becomes: const model = servedByLabel(message.provenance);
  it('does not call a changed request a changed model', () => {
    // One model served both turns; only the request changed. Comparing rendered labels
    // read that as a change, dropped the header, and printed "This reply came from
    // ollama:qwen3:8b, not ollama:qwen3:8b" here. The header is what this case holds:
    // what turn 2 prints is the next case's, and it is a different rule.
    renderRoom({ room: soloRoom(degradedOntoTheHeadersOwnModel()) });

    expect(screen.getByTestId('header-attribution')).toHaveTextContent('ollama:qwen3:8b');
    expect(screen.queryByTestId('served-by-1')).toBeNull();
  });

  // [Rev 25], and #1201's acceptance: a degraded serve prints on its own turn whether or
  // not the model changed. Both halves are the case. The header keeps `ollama:qwen3:8b`
  // because `served_by` is what decides it and that stays true of every row; turn 2 still
  // prints, because it asked for one model and was answered by another and nothing else
  // on screen says so. A fix that also takes the header away is the wrong fix: it would
  // let one substitution strip a label that is true of forty other turns.
  // Killed by: frontend/src/lib/rooms.ts :: if (perTurn || message.provenance?.degraded) lines.set(message.seq, display);
  // Becomes: if (perTurn) lines.set(message.seq, display);
  it('prints a degraded serve on its own turn, under a header it did not change', () => {
    renderRoom({ room: soloRoom(degradedOntoTheHeadersOwnModel()) });

    expect(screen.getByTestId('header-attribution')).toHaveTextContent('ollama:qwen3:8b');
    expect(screen.queryByTestId('served-by-1')).toBeNull();
    expect(screen.getByTestId('served-by-2')).toHaveTextContent(
      'ollama:qwen3:8b (asked for ollama:hermes3:8b)',
    );
  });

  // Killed by: frontend/src/lib/rooms.ts :: const perTurn = shape === 'multi' || headerAttribution(transcript) === null;
  // Becomes: const perTurn = headerAttribution(transcript) === null;
  it('prints it on every turn of a group, where the speaker changes turn to turn', () => {
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: 'one', provenance: served('qwen3:8b') }),
          message({ seq: 2, sender_id: 'critic', content: 'two', provenance: served('qwen3:8b') }),
        ],
      }),
    });

    expect(screen.getByTestId('served-by-1')).toHaveTextContent('ollama:qwen3:8b');
    expect(screen.getByTestId('served-by-2')).toHaveTextContent('ollama:qwen3:8b');
    // Nothing constant to hoist: a header here would be true of one row and false of the next.
    expect(screen.queryByTestId('header-attribution')).toBeNull();
  });

  it('says nothing about a model in a conversation nothing has been served in yet', () => {
    // P6: an absent attribution is absent, not a default. No turn has reported one.
    renderRoom({ room: soloRoom([message({ seq: 1, sender_id: 'user', content: 'hello' })]) });

    expect(screen.queryByTestId('header-attribution')).toBeNull();
  });
});

/**
 * FR-13.4's other half: the record is kept on **every** turn, and each turn's own
 * attribution stays reachable *from* that turn. `why ›` is that affordance, so it is
 * offered wherever there is anything to disclose -- a selector trail, a served model, or
 * both -- and it names that turn to the owner, which opens the dock's Turn surface on it.
 *
 * **What the record says is `TurnDetail`'s to render and `TurnDetail.test.tsx`'s to check.**
 * It used to be disclosed under the row and asserted here; the cases that moved went with
 * it rather than being kept in both places, which is how two renderings of one record start
 * disagreeing. §3.2.3's "not on a separate screen" is still met: the dock is a region of
 * the same screen, opened beside the conversation, which stays where it is and keeps its
 * scroll.
 */
describe('RoomConversation why ›', () => {
  const soloRoom = (transcript: RoomTranscriptMessage[]): RoomState =>
    room({
      participants: [
        { id: 'user', kind: 'human', display_name: 'Kenny' },
        { id: 'scout', kind: 'agent', display_name: 'Scout' },
      ],
      transcript,
    });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: message.decision || servedHere ||
  // Becomes: message.decision ||
  it('offers the turn`s own record on a turn the header speaks for', () => {
    // FR-13.4's acceptance criterion, exactly: the reply's text names one model and the
    // turn was served by another, and what the surface shows for that turn is the served
    // one -- without leaving the conversation. What a `why ›` conditioned on the decision
    // alone left unreachable was not the *decision* -- a solo reply carries one: the
    // Core's `SoleAgentSelector.select` returns SPEAK with `selector="sole_agent"`,
    // `_take_turn(..., decision: SpeakerDecision)` is non-optional and writes it onto the
    // message, and `GET /api/rooms/{id}` dumps the whole state. It was the *model*: at
    // `d84fda5a` the disclosure body read `Chosen by {selector}` and named no model on any
    // turn, so the served model was reachable from nowhere on the turn itself. That is the
    // FR-13.4 defect this case pins.
    const opened: number[] = [];
    renderRoom({
      onOpenTurn: (seq) => opened.push(seq),
      room: soloRoom([
        message({
          seq: 1,
          sender_id: 'scout',
          content: 'I am OpenAI GPT-4.',
          provenance: {
            path: 'primary',
            requested: { provider: 'ollama', model: 'qwen3:8b' },
            served_by: { provider: 'ollama', model: 'qwen3:8b' },
            degraded: false,
          },
        }),
      ]),
    });

    // This turn carries no decision at all, so a control conditioned on the decision alone
    // is not offered here -- which is the FR-13.4 defect. What the surface then shows for
    // it is `TurnDetail.test.tsx`'s `names the model the turn was served by`.
    fireEvent.click(screen.getByTestId('why-1'));
    expect(opened).toEqual([1]);
  });

  // Both of the cases that used to sit here -- a turn that reported no model, and a
  // degraded turn whose requested model is still reachable -- moved to
  // `TurnDetail.test.tsx` with the lines they assert. Their `Killed by:` declarations went
  // with them, renamed onto the file that now holds those lines.

  it('offers nothing to open on a turn with neither a model nor a decision', () => {
    // A `why ›` that opens on "Chosen by undefined" is worse than no control.
    renderRoom({
      room: soloRoom([message({ seq: 1, sender_id: 'user', content: 'hello' })]),
    });

    expect(screen.queryByTestId('why-1')).toBeNull();
  });
});

describe('RoomConversation before anything is said', () => {
  // The first thing New does is open a conversation with nothing in it. With no branch for
  // that, the primary region of the screen rendered as a blank box -- an absence that
  // stated no cause, on the first action a new user takes.
  it('says the conversation is waiting for its first message', () => {
    renderRoom({ room: room({ transcript: [] }) });

    expect(screen.getByTestId('transcript-empty')).toHaveTextContent(
      'Nothing has been said here yet. Send a message to start.',
    );
  });

  it('says why nobody will answer when nobody is here, with the control that fixes it', () => {
    renderRoom({ room: room({ participants: [{ id: 'user', kind: 'human' }], transcript: [] }) });

    expect(screen.getByTestId('transcript-empty')).toHaveTextContent(
      'No one is in this conversation yet. Add someone to get a reply.',
    );
  });

  it('does not count a join as something said', () => {
    renderRoom({
      room: room({ transcript: [message({ seq: 1, sender_id: 'scout', kind: 'join', content: '' })] }),
    });

    expect(screen.getByTestId('transcript-empty')).toBeInTheDocument();
  });

  it('goes away once anyone has spoken', () => {
    renderRoom();

    expect(screen.queryByTestId('transcript-empty')).toBeNull();
  });
});

describe('RoomConversation while a turn runs', () => {
  it('marks the agent that has the floor and nobody else', () => {
    renderRoom({ live: { turn: { agentId: 'scout', turnId: 'turn-1', text: 'The ' }, error: null } });

    expect(screen.getByTestId('writing-scout')).toBeInTheDocument();
    expect(screen.queryByTestId('writing-critic')).toBeNull();
  });

  it('offers Stop instead of Send while the room is running, and Stop takes the floor back', () => {
    const onStop = vi.fn();
    renderRoom({ live: { turn: { agentId: 'scout', turnId: 'turn-1', text: '' }, error: null }, onStop });

    expect(screen.queryByTestId('send-message')).toBeNull();
    fireEvent.click(screen.getByTestId('stop-turn'));

    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it('says a cascade stopped, rather than going quiet', () => {
    renderRoom({ live: { turn: null, error: "'phantm' is not a participant" } });

    expect(screen.getByTestId('cascade-error')).toHaveTextContent('phantm');
  });
});

describe('RoomConversation composer', () => {
  it('promises the only agent by name and refuses to guess among several', () => {
    const single = room({
      participants: [
        { id: 'user', kind: 'human' },
        { id: 'scout', kind: 'agent', display_name: 'Scout' },
      ],
    });
    const { unmount } = renderRoom({ room: single });
    expect(screen.getByTestId('answer-hint')).toHaveTextContent('Scout will answer');
    unmount();

    renderRoom();
    expect(screen.getByTestId('answer-hint')).toHaveTextContent('Someone will pick this up');
  });

  it('names the agent an address points at', () => {
    renderRoom();

    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: '@critic look' } });

    expect(screen.getByTestId('answer-hint')).toHaveTextContent('Critic will answer');
  });

  it('does not promise more answers than this conversation`s own ceiling allows', () => {
    // The number comes from `room.policy`, so a hint computed against a constant -- or
    // against no ceiling at all -- states a bound this conversation does not have.
    renderRoom({
      room: room({
        participants: [
          { id: 'user', kind: 'human' },
          { id: 'scout', kind: 'agent' },
          { id: 'critic', kind: 'agent' },
          { id: 'dba', kind: 'agent' },
        ],
        policy: { ...room().policy, max_agent_turns_per_human_message: 2 },
      }),
    });

    fireEvent.change(screen.getByTestId('room-composer'), {
      target: { value: '@scout @critic @dba' },
    });

    expect(screen.getByTestId('answer-hint')).toHaveTextContent('Paused after 2 replies');
  });

  it('will not promise an answer for a name the conversation does not have', () => {
    renderRoom({
      room: room({
        participants: [
          { id: 'user', kind: 'human' },
          { id: 'scout', kind: 'agent', display_name: 'Scout' },
        ],
      }),
    });

    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: '@phantm help' } });

    expect(screen.getByTestId('answer-hint')).toHaveTextContent(
      'No one here is called @phantm',
    );
  });

  it('offers the conversation`s participants when an address is being typed', () => {
    // §3.3 F3. Typing a runtime id from memory is the recall the head exists to avoid,
    // and getting one wrong lands in the refusal above.
    renderRoom();

    expect(screen.queryByTestId('mention-completion')).toBeNull();
    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'ask @cr' } });

    expect(screen.getByTestId('mention-completion')).toBeInTheDocument();
    expect(screen.queryByTestId('mention-option-scout')).toBeNull();
    fireEvent.click(screen.getByTestId('mention-option-critic'));

    expect(screen.getByTestId('room-composer')).toHaveValue('ask @critic ');
    expect(screen.queryByTestId('mention-completion')).toBeNull();
  });

  it('closes the mention menu on Escape without sending or clearing the draft (#1036)', () => {
    renderRoom();

    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'ask @cr' } });
    expect(screen.getByTestId('mention-completion')).toBeInTheDocument();

    // Escape now goes through the shared owner (`useEscapeOwner('overlay', ...)`) rather
    // than the composer's own `onKeyDown`, so this fires on the composer element like a
    // real keypress would, and relies on it bubbling to the shared window listener.
    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: useEscapeOwner('overlay', Boolean(mention && candidates.length > 0) && pending === null, () =>
    // Becomes: useEscapeOwner('overlay', false, () =>
    fireEvent.keyDown(screen.getByTestId('room-composer'), { key: 'Escape' });

    expect(screen.queryByTestId('mention-completion')).toBeNull();
    expect(screen.getByTestId('room-composer')).toHaveValue('ask @cr');
  });

  /**
   * Enter sends.
   *
   * The box had no key handler at all: Enter grew the draft, and the only way to send was
   * the button in the far corner of the box. Every chat surface a reader arrives from --
   * Claude Desktop, Slack, Discord, iMessage -- has already taught their hands otherwise,
   * so the most ordinary action on the screen was the one that did not work.
   *
   * What these guard is mostly the exceptions, because the exceptions are where a send
   * key does damage: a line break must stay reachable, an IME must keep the key it needs
   * to settle a word, a half-typed name must not go out addressed to nobody, and the key
   * must not do what no button on screen is offering.
   */
  describe('the send key', () => {
    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: if (event.key !== 'Enter' || event.shiftKey) return;
    // Becomes: if (true) return;
    it('sends on Enter', async () => {
      const onSend = vi.fn().mockResolvedValue(undefined);
      renderRoom({ onSend });

      fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'ship it' } });
      fireEvent.keyDown(screen.getByTestId('room-composer'), { key: 'Enter' });

      await waitFor(() => expect(onSend).toHaveBeenCalledWith('ship it'));
      expect(screen.getByTestId('room-composer')).toHaveValue('');
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: || event.shiftKey
    // Becomes:
    it('starts a line on Shift+Enter, and leaves the draft where it was', () => {
      // A message of more than one line is not an edge case here -- the box is two rows
      // tall by default. If Shift+Enter sent, a paragraph could not be written at all.
      const onSend = vi.fn();
      renderRoom({ onSend });

      fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'first line' } });
      fireEvent.keyDown(screen.getByTestId('room-composer'), { key: 'Enter', shiftKey: true });

      expect(onSend).not.toHaveBeenCalled();
      expect(screen.getByTestId('room-composer')).toHaveValue('first line');
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: if (event.nativeEvent.isComposing) return;
    // Becomes:
    it('leaves Enter to the IME while a word is still being composed', () => {
      // Typing Korean, Japanese or Chinese spends Enter on the characters being assembled:
      // it is the key that settles which word was meant. Sending on it costs the writer
      // twice -- a half-written word goes out, and the keystroke that would have finished
      // it is gone. The browser marks such a press `isComposing`, and it is the only
      // signal there is; without this branch the surface is unusable in those languages.
      const onSend = vi.fn();
      renderRoom({ onSend });

      fireEvent.change(screen.getByTestId('room-composer'), { target: { value: '한글' } });
      fireEvent.keyDown(screen.getByTestId('room-composer'), { key: 'Enter', isComposing: true });

      expect(onSend).not.toHaveBeenCalled();
      expect(screen.getByTestId('room-composer')).toHaveValue('한글');
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: acceptCompletion(candidates[0]);
    // Becomes:
    it('takes the offered name instead of sending, while the name menu is open', () => {
      // `@cr` is nobody. Sent as it stands it produces a message that looks addressed and
      // reaches no seat -- the refusal this surface already has a test for. The menu being
      // open is precisely the moment Enter means "that one".
      const onSend = vi.fn();
      renderRoom({ onSend });

      fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'ask @cr' } });
      fireEvent.keyDown(screen.getByTestId('room-composer'), { key: 'Enter' });

      expect(onSend).not.toHaveBeenCalled();
      expect(screen.getByTestId('room-composer')).toHaveValue('ask @critic ');
      expect(screen.queryByTestId('mention-completion')).toBeNull();
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: data-enter-takes={index === 0 ? 'true' : undefined}
    // Becomes:
    it('marks the one candidate Enter will take, rather than making the reader find out', () => {
      // A key that picks one item out of a row drawn uniformly is a rule learned only by
      // being surprised by it. The marked option and the taken option are read from the
      // same render here, so a mark on the wrong one cannot pass.
      renderRoom({
        room: room({
          participants: [
            { id: 'user', kind: 'human' },
            { id: 'critic', kind: 'agent' },
            { id: 'critique-bot', kind: 'agent' },
          ],
        }),
      });

      fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'ask @cri' } });
      const marked = screen
        .getByTestId('mention-completion')
        .querySelectorAll('[data-enter-takes="true"]');
      expect(marked).toHaveLength(1);
      const markedId = marked[0].getAttribute('data-testid');

      fireEvent.keyDown(screen.getByTestId('room-composer'), { key: 'Enter' });

      expect(markedId).toBe('mention-option-critic');
      expect(screen.getByTestId('room-composer')).toHaveValue('ask @critic ');
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: if (running) return;
    // Becomes:
    it('does not send while a turn is running, because no Send button is offered', () => {
      // The strip holds Stop, not Send, for the whole of a running turn. A key that does
      // what no control on screen offers is a second, invisible interface.
      const onSend = vi.fn();
      renderRoom({
        onSend,
        live: { turn: { agentId: 'scout', turnId: 'turn-1', text: '' }, error: null },
      });

      fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'and also' } });
      fireEvent.keyDown(screen.getByTestId('room-composer'), { key: 'Enter' });

      expect(onSend).not.toHaveBeenCalled();
      expect(screen.getByTestId('room-composer')).toHaveValue('and also');
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: aria-label="Send (Enter)"
    // Becomes: aria-label="Send"
    it('names the key on the button that does the same thing', () => {
      renderRoom();

      expect(screen.getByTestId('send-message')).toHaveAttribute('aria-label', 'Send (Enter)');
    });
  });

  it('collapses a burst of keystrokes into one composing report', () => {
    // The route is a read-modify-write of the whole conversation plus a bus publish, and
    // this used to fire on every `onChange`.
    vi.useFakeTimers();
    const onTyping = vi.fn();
    renderRoom({ onTyping });
    const composer = screen.getByTestId('room-composer');

    fireEvent.change(composer, { target: { value: 'h' } });
    fireEvent.change(composer, { target: { value: 'he' } });
    fireEvent.change(composer, { target: { value: 'hel' } });

    expect(onTyping).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(TYPING_REPORT_INTERVAL_MS + 1);
    });
    fireEvent.change(composer, { target: { value: 'hell' } });

    expect(onTyping).toHaveBeenCalledTimes(2);
  });

  it('clears the box once the message is actually recorded', async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    renderRoom({ onSend });

    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'check it' } });
    fireEvent.click(screen.getByTestId('send-message'));

    await waitFor(() => expect(screen.getByTestId('room-composer')).toHaveValue(''));
    expect(onSend).toHaveBeenCalledWith('check it');
  });

  it('keeps the draft and says why when the send is refused', async () => {
    // It used to clear the box before the promise settled, so a refusal the user never
    // saw also destroyed what they had written.
    // Rejected as the head rejects a refused send: the Core's own `detail` on a
    // `RoomsApiError`, inside the outcome the head settled on (#1441).
    const onSend = vi
      .fn()
      .mockRejectedValue(
        new RoomSendError(
          'refused',
          new RoomsApiError("Room 'r1' has no participant @phantm.", 400, "Room 'r1' has no participant @phantm."),
        ),
      );
    renderRoom({ onSend });

    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: '@phantm help' } });
    fireEvent.click(screen.getByTestId('send-message'));

    expect(await screen.findByTestId('send-error')).toHaveTextContent(
      "Not sent: Room 'r1' has no participant @phantm.",
    );
    expect(screen.getByTestId('room-composer')).toHaveValue('@phantm help');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: autoFocus
  // Becomes:
  it('takes the caret when the conversation opens, so typing goes somewhere (#114, #339)', () => {
    // Ported by hand from the retired `PlaygroundTab`, whose own case is deleted with it
    // (#1208). Named here rather than left to `test_dashboard_e2e.py`, which asserts that
    // the document's first `input, textarea` has focus without naming which element it
    // wanted -- true of this composer today, and of whatever else lands above it tomorrow.
    renderRoom();

    expect(screen.getByTestId('room-composer')).toHaveFocus();
  });
});

describe('RoomConversation stops and failures', () => {
  it('explains the pause with the policy number, not a constant', () => {
    renderRoom({
      room: room({
        turn_state: { agent_turns_since_human: 5 },
        policy: { ...room().policy, max_agent_turns_per_human_message: 5 },
      }),
    });

    expect(screen.getByTestId('ceiling-notice')).toHaveTextContent('Paused after 5 replies');
  });

  it('does not announce the pause while a turn is still running', () => {
    // `agent_turns_since_human` reaches the ceiling before the last turn lands, so the
    // notice without `!running` appeared under an agent that was still writing.
    renderRoom({
      room: room({
        turn_state: { agent_turns_since_human: 3 },
        policy: { ...room().policy, max_agent_turns_per_human_message: 3 },
      }),
      live: { turn: { agentId: 'scout', turnId: 'turn-1', text: 'Well…' }, error: null },
    });

    expect(screen.queryByTestId('ceiling-notice')).toBeNull();
  });

  it('says so when the conversation decided nobody speaks', () => {
    // The selector chain can reach SILENCE and publish nothing at all, so the surface
    // showed the human's own message and then nothing, forever.
    renderRoom({
      room: room({
        last_decision: {
          verdict: 'silence',
          speaker_id: null,
          confidence: 1,
          selector: 'orchestrator',
          reasoning: 'the selector chain was exhausted without a judgement',
        },
      }),
    });

    expect(screen.getByTestId('silence-notice')).toHaveTextContent('No one answered');
    expect(screen.queryByTestId('why-silence-detail')).toBeNull();
    fireEvent.click(screen.getByTestId('why-silence'));
    expect(screen.getByTestId('why-silence-detail')).toHaveTextContent('orchestrator');
  });

  it('does not say nobody answered under the reply the only agent gave (#920)', () => {
    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: const silence = unansweredSilence(room);
    // Becomes: const silence = room.last_decision && room.last_decision.verdict === 'silence' ? room.last_decision : null;
    // The chain records a silence after every answered exchange: the one agent has
    // replied, and nobody is left to give the floor to.
    renderRoom({
      room: room({
        participants: [
          { id: 'user', kind: 'human', display_name: 'Kenny' },
          { id: 'scout', kind: 'agent', display_name: 'Scout' },
        ],
        transcript: [
          message({ seq: 1, sender_id: 'user', content: 'check the index' }),
          message({ seq: 2, sender_id: 'scout', content: 'the index is fine' }),
        ],
        turn_state: { agent_turns_since_human: 1 },
        last_decision: {
          verdict: 'silence',
          speaker_id: null,
          confidence: 1,
          selector: 'orchestrator',
          reasoning: 'the selector chain was exhausted without a judgement',
        },
      }),
    });

    expect(screen.getByTestId('row-2')).toHaveTextContent('the index is fine');
    expect(screen.queryByTestId('silence-notice')).toBeNull();
  });

  it('still says nobody answered in a conversation nobody is in', () => {
    renderRoom({
      room: room({
        participants: [{ id: 'user', kind: 'human', display_name: 'Kenny' }],
        transcript: [message({ seq: 1, sender_id: 'user', content: 'hello?' })],
        last_decision: {
          verdict: 'silence',
          speaker_id: null,
          confidence: 1,
          selector: 'orchestrator',
          reasoning: 'the selector chain was exhausted without a judgement',
        },
      }),
    });

    expect(screen.getByTestId('silence-notice')).toHaveTextContent('No one answered');
  });

  it('does not claim silence when somebody was chosen to speak', () => {
    renderRoom({
      room: room({
        last_decision: {
          verdict: 'speak',
          speaker_id: 'scout',
          confidence: 1,
          selector: 'sole_agent',
          reasoning: "the room's only agent",
        },
      }),
    });

    expect(screen.queryByTestId('silence-notice')).toBeNull();
  });

  // A reply whose seat could not be saved stands, but says so in one line (#1366, P6).
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {message.persist_error ? (
  // Becomes: {false && message.persist_error ? (
  it('says, under a reply, that its turn was not saved for after a restart', () => {
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: 'here it is', persist_error: 'OSError: disk full' }),
          message({ seq: 2, sender_id: 'scout', content: 'and this one saved' }),
        ],
      }),
    });

    const notice = screen.getByTestId('row-unsaved-1');
    // The cause stays on the field, for the log's reader; the line is plain (#1408).
    expect(notice).not.toHaveTextContent('OSError');
    expect(notice).not.toHaveTextContent('disk full');
    expect(notice).toHaveTextContent(/after a restart; it could not be saved\.$/);
    expect(screen.getByTestId('row-body-1')).toHaveTextContent('here it is');
    expect(screen.queryByTestId('row-unsaved-2')).toBeNull();
    expect(screen.queryByTestId('row-error-1')).toBeNull();
  });

  // What the clone learned in a turn could not be saved: said on its own line, apart from
  // the session's, because the two records are lost independently (#1367, P6).
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {message.knowledge_persist_error ? (
  // Becomes: {false && message.knowledge_persist_error ? (
  it('says, under a reply, that what the clone learned in it was not saved', () => {
    renderRoom({
      room: room({
        transcript: [
          message({
            seq: 1,
            sender_id: 'scout',
            content: 'here it is',
            knowledge_persist_error: 'OSError: disk full',
          }),
          message({ seq: 2, sender_id: 'scout', content: 'and this one saved' }),
        ],
      }),
    });

    const notice = screen.getByTestId('row-knowledge-unsaved-1');
    // The cause is for the log and the field, not the reader: no class name, no path.
    expect(notice).not.toHaveTextContent(/OSError|disk full|Error/);
    expect(notice).toHaveTextContent(/learned/);
    expect(notice).toHaveTextContent(/after a restart/);
    expect(screen.getByTestId('row-body-1')).toHaveTextContent('here it is');
    expect(screen.queryByTestId('row-unsaved-1')).toBeNull();
    expect(screen.queryByTestId('row-knowledge-unsaved-2')).toBeNull();
    expect(screen.queryByTestId('row-error-1')).toBeNull();
  });

  // A knowledge record that could not be read was set aside: said on the row of the turn that
  // set it aside, and on no other, in plain words (#1367, P6). It speaks of that record only,
  // never of the clone's saved memory, which the set-aside does not touch (#1434).
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {message.knowledge_set_aside ? (
  // Becomes: {false && message.knowledge_set_aside ? (
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {label}&apos;s knowledge record for this conversation could not be read, so it was
  // Becomes: {label}&apos;s saved memory could not be read, so it was
  it('says, under a reply, that the knowledge record for this conversation was set aside', () => {
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: 'hello', knowledge_set_aside: true }),
          message({ seq: 2, sender_id: 'scout', content: 'hello again' }),
        ],
      }),
    });

    const notice = screen.getByTestId('row-knowledge-reset-1');
    expect(notice).toHaveTextContent(/could not be read/);
    expect(notice).toHaveTextContent(/kept/);
    expect(notice).toHaveTextContent(/knowledge record for this conversation/);
    expect(notice).not.toHaveTextContent(/saved memory|start(?:ing)? over/i);
    expect(notice.textContent ?? '').not.toMatch(ABSENCE_CLAIM);
    expect(notice).not.toHaveTextContent(/\/|Error|yaml/i);
    expect(screen.getByTestId('row-body-1')).toHaveTextContent('hello');
    expect(screen.queryByTestId('row-knowledge-reset-2')).toBeNull();
    expect(screen.queryByTestId('row-error-1')).toBeNull();
  });

  // A reply that claims a save the Core saw fail carries the failure beside it (#1375).
  // The model's words stay as written; the row says what actually happened, in words built
  // from the Core's counts. The notice must never carry the tool's error text.
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {unsavedNotice ? (
  // Becomes: {false && unsavedNotice ? (
  it('says, under a reply, that nothing was saved to memory when every save failed', () => {
    renderRoom({
      room: room({
        transcript: [
          message({
            seq: 1,
            sender_id: 'scout',
            content: "I've saved your favourite colour.",
            memory_facts_tried: 1,
            memory_facts_unsaved: 1,
          }),
          message({
            seq: 2,
            sender_id: 'scout',
            content: 'and this one did save',
            memory_facts_tried: 1,
            memory_facts_unsaved: 0,
          }),
        ],
      }),
    });

    const notice = screen.getByTestId('row-memory-unsaved-1');
    expect(notice.textContent).toMatch(/tried to save something to memory, and nothing was saved\.$/);
    expect(screen.getByTestId('row-body-1')).toHaveTextContent("I've saved your favourite colour.");
    expect(screen.queryByTestId('row-memory-unsaved-2')).toBeNull();
    expect(screen.queryByTestId('row-error-1')).toBeNull();
  });

  // The reviewer's case on #1400: two different facts, one saved and one not. One success
  // used to silence the row while the reply could claim both.
  // Killed by: frontend/src/lib/turnOutcome.ts :: if (missed === total) {
  // Becomes: if (missed > 0) {
  it('says how many of the facts a reply tried to save were not saved', () => {
    renderRoom({
      room: room({
        transcript: [
          message({
            seq: 1,
            sender_id: 'scout',
            content: "I've saved both.",
            memory_facts_tried: 2,
            memory_facts_unsaved: 1,
          }),
        ],
      }),
    });

    expect(screen.getByTestId('row-memory-unsaved-1').textContent).toMatch(
      /tried to save 2 things to memory, and 1 of them was not saved\.$/,
    );
  });

  it('renders a failed turn in a plain sentence and a way to run it again (#1408)', () => {
    const onRetry = vi.fn();
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: '', error: 'TimeoutError: no response' }),
        ],
      }),
      onRetry,
    });

    const row = screen.getByTestId('row-error-1');
    expect(row).toHaveTextContent(
      "Scout couldn't finish this turn because something went wrong while it was answering.",
    );
    expect(row).not.toHaveTextContent('TimeoutError');
    expect(row).not.toHaveTextContent('no response');
    fireEvent.click(screen.getByTestId('retry-turn'));
    expect(onRetry).toHaveBeenCalled();
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: ) : message.content.trim() === '' ? (
  // Becomes: ) : false ? (
  it('says a turn finished without text rather than drawing an empty bubble (P6)', () => {
    // The room recorded exactly this on 2026-09-21: `completed: true`, `error: null`,
    // `content: ""` -- three of them, while the image tool was writing files to disk. The
    // row drew a bubble with nothing in it, which reads as the head being broken.
    const onRetry = vi.fn();
    renderRoom({
      room: room({
        transcript: [message({ seq: 1, sender_id: 'scout', content: '', completed: true })],
      }),
      onRetry,
    });

    expect(screen.getByTestId('row-silent-1')).toHaveTextContent(
      'Scout finished this turn without sending any text.',
    );
    fireEvent.click(screen.getByTestId('retry-turn'));
    expect(onRetry).toHaveBeenCalled();
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {message.completed
  // Becomes: {true
  it('distinguishes a turn that stopped partway from one that ran out of words', () => {
    renderRoom({
      room: room({
        transcript: [message({ seq: 1, sender_id: 'scout', content: '', completed: false })],
      }),
    });

    expect(screen.getByTestId('row-silent-1')).toHaveTextContent(
      'Scout stopped partway through this turn and sent no text.',
    );
  });

  it('offers no Retry on a refusal a retry would meet again, and says what would help (#969)', () => {
    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {message.refusal ? (
    // Becomes: {false ? (
    // A spent budget refuses every retry: the ledger only grows. The row used to offer
    // Retry anyway, and pressing it appended another identical failure.
    renderRoom({
      room: room({
        transcript: [
          message({
            seq: 1,
            sender_id: 'scout',
            content: '',
            error: 'Session token limit exceeded: 12/10',
            refusal: 'budget_exceeded',
          }),
        ],
      }),
    });

    expect(screen.getByTestId('row-error-1')).toHaveTextContent(
      "Scout couldn't finish this turn: it has used all the tokens this conversation allows.",
    );
    expect(screen.getByTestId('row-error-1')).not.toHaveTextContent('Session token limit');
    expect(screen.queryByTestId('retry-turn')).toBeNull();
    expect(screen.getByTestId('row-remedy-1')).toHaveTextContent(
      'Trying again would be refused too. Start a new conversation to continue.',
    );
  });

  it('names the model remedy when the model cannot use tools, and no internals', () => {
    // Killed by: frontend/src/lib/turnOutcome.ts :: model_without_tools: "its model can't use tools, which clones need",
    // Becomes: model_without_tools: 'the app refused to run it',
    renderRoom({
      room: room({
        transcript: [
          message({
            seq: 1,
            sender_id: 'scout',
            content: '',
            error:
              "The model deepseek-r1:14b can't use tools, which UClone-X clones need. Pick a model that supports tools (for example qwen3:8b) in Settings, or pass --model.",
            refusal: 'model_without_tools',
          }),
        ],
      }),
    });

    const row = screen.getByTestId('row-error-1');
    expect(row).toHaveTextContent("Scout couldn't finish this turn: its model can't use tools, which clones need.");
    expect(screen.queryByTestId('retry-turn')).toBeNull();
    expect(screen.getByTestId('row-remedy-1')).toHaveTextContent(
      'Pick a model that supports tools (for example qwen3:8b) in Settings, then send your message again.',
    );
    for (const internal of ['Traceback', 'status 400', '{', 'LLMProviderError']) {
      expect(row).not.toHaveTextContent(internal);
      expect(screen.getByTestId('row-remedy-1')).not.toHaveTextContent(internal);
    }
  });

  it('keeps Retry on a failure that carries no refusal (#969)', () => {
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: '', error: 'provider unavailable', refusal: null }),
        ],
      }),
    });

    expect(screen.getByTestId('retry-turn')).toBeInTheDocument();
    expect(screen.queryByTestId('row-remedy-1')).toBeNull();
  });

  it('offers only agents who are not already here, as a choice rather than a typed id', () => {
    // The filter used to survive `const invitable = availableAgents`, because the only
    // candidate in the fixture was unseated. `scout` is seated, so it must not be here.
    const onAddAgent = vi.fn();
    renderRoom({
      onAddAgent,
      availableAgents: [
        makeAgentInfo({ id: 'scout', label: 'Scout' }),
        makeAgentInfo({ id: 'dba', label: 'DBA' }),
      ],
    });

    fireEvent.click(screen.getByTestId('add-someone'));

    expect(screen.queryByTestId('invite-scout')).toBeNull();
    fireEvent.click(screen.getByTestId('invite-dba'));
    expect(onAddAgent).toHaveBeenCalledWith('dba');
  });
});

describe('RoomConversation invite list', () => {
  it('distinguishes "nobody is running" from "everybody is already here"', () => {
    // The runtime reports no clones until one has been used, so a fresh install opened
    // the invite list and was told everyone was already in the conversation.
    const { unmount } = renderRoom({ availableAgents: [] });
    fireEvent.click(screen.getByTestId('add-someone'));
    expect(screen.getByTestId('invite-empty-cause')).toHaveTextContent('No clones are running yet');
    unmount();

    renderRoom({ availableAgents: [makeAgentInfo({ id: 'scout' })] });
    fireEvent.click(screen.getByTestId('add-someone'));
    expect(screen.getByTestId('invite-empty-cause')).toHaveTextContent(
      'Everyone available is already in this conversation',
    );
  });
});

/**
 * The three things the retirement of the single-agent surface must not lose (#1208):
 * its history controls, its saturation banner and its model selection.
 */
describe('RoomConversation history controls', () => {
  const soleSeat = (over: Partial<RoomState> = {}): RoomState =>
    room({
      participants: [
        { id: 'user', kind: 'human', display_name: 'Kenny' },
        { id: 'scout', kind: 'agent', display_name: 'Scout' },
      ],
      ...over,
    });

  const full = (over: Partial<RoomContext> = {}): RoomContext => ({
    room_id: 'r1',
    seats: [
      {
        participant_id: 'scout',
        session_id: 's1',
        active_turns: 20,
        is_saturated: true,
        used_tokens: null,
        max_context_tokens: null,
        context_window_source: null,
        live: true,
      },
    ],
    is_saturated: true,
    saturation_threshold: 20,
    ...over,
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {context?.is_saturated ? (
  // Becomes: {context ? (
  it('raises the banner on the Core`s own count, and not merely on having read one', () => {
    // The figure is the Core's per-seat turn count. A head that raised the banner on
    // having an answer at all would show it on every conversation it had read.
    const { unmount } = renderRoom({
      room: soleSeat(),
      context: full({
        seats: [
          {
            participant_id: 'scout',
            session_id: 's1',
            active_turns: 2,
            is_saturated: false,
            used_tokens: null,
            max_context_tokens: null,
            context_window_source: null,
            live: true,
          },
        ],
        is_saturated: false,
      }),
    });
    expect(screen.queryByTestId('saturation-banner')).toBeNull();
    unmount();

    renderRoom({ room: soleSeat(), context: full() });
    const banner = screen.getByTestId('saturation-banner');
    expect(banner).toHaveTextContent(
      'Scout reached the 20-turn limit on context',
    );
    expect(banner.firstElementChild).toHaveClass('max-w-3xl');
    expect(banner.firstElementChild).toHaveClass('mx-auto');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {saturatedSeats.some((seat) => !seat.live) ? (
  // Becomes: {saturatedSeats.some((seat) => seat.live) ? (
  it('says when the figure came from the saved record rather than a running clone', () => {
    // The two copies can disagree: a seat nothing has spoken in since this server started
    // is behind any turn an earlier run did not write (P6).
    const { unmount } = renderRoom({ room: soleSeat(), context: full() });
    expect(screen.queryByTestId('saturation-from-record')).toBeNull();
    unmount();

    renderRoom({
      room: soleSeat(),
      context: full({
        seats: [
          {
            participant_id: 'scout',
            session_id: 's1',
            active_turns: 20,
            is_saturated: true,
            used_tokens: null,
            max_context_tokens: null,
            context_window_source: null,
            live: false,
          },
        ],
      }),
    });
    expect(screen.getByTestId('saturation-from-record')).toHaveTextContent(
      'Counted from the saved record for Scout',
    );
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: void onClearHistory?.();
  // Becomes: // void onClearHistory?.();
  it('clears only after the question naming how much goes has been answered', () => {
    const onClearHistory = vi.fn();
    renderRoom({
      room: soleSeat({ transcript: [message({ seq: 1 }), message({ seq: 2, sender_id: 'scout' })] }),
      onClearHistory,
    });

    fireEvent.click(screen.getByTestId('clear-history'));
    expect(onClearHistory).not.toHaveBeenCalled();
    expect(screen.getByTestId('confirm-history-change')).toHaveTextContent(
      'Its 2 messages are removed for good',
    );

    fireEvent.click(screen.getByTestId('confirm-history-change-yes'));
    expect(onClearHistory).toHaveBeenCalledTimes(1);
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: useEscapeOwner('overlay', Boolean(mention && candidates.length > 0) && pending === null, () =>
  // Becomes: useEscapeOwner('overlay', Boolean(mention && candidates.length > 0), () =>
  it('lets Escape abandon the question even when the mention menu opened after it', () => {
    // Both claim the `overlay` layer, and `escapePrecedence` calls two owners at one layer
    // an undesigned case settled by "whoever registered last". Opening the menu second is
    // exactly that case, so the two are made mutually exclusive rather than left to it.
    renderRoom({
      room: soleSeat({ transcript: [message({ seq: 1 })] }),
      onClearHistory: vi.fn(),
    });

    fireEvent.click(screen.getByTestId('clear-history'));
    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'ask @sc' } });
    expect(screen.getByTestId('mention-completion')).toBeInTheDocument();

    fireEvent.keyDown(screen.getByTestId('room-composer'), { key: 'Escape' });

    expect(screen.queryByTestId('confirm-history-change')).toBeNull();
    expect(screen.getByTestId('mention-completion')).toBeInTheDocument();
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {participantsNotReset.length > 0 ? (
  // Becomes: {participantsNotReset.length >= 0 ? (
  it('names a seat the cut did not reach, rather than reporting a clean result', () => {
    // A seat still holding what the transcript no longer does is the divergence these
    // controls exist to prevent, so it is said.
    const { unmount } = renderRoom({ room: soleSeat() });
    expect(screen.queryByTestId('participants-not-reset')).toBeNull();
    unmount();

    renderRoom({ room: soleSeat(), participantsNotReset: ['scout'] });
    expect(screen.getByTestId('participants-not-reset')).toHaveTextContent(
      'Scout still remembers what was removed here',
    );
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: disabled={running}
  // Becomes: disabled={false}
  it('refuses the cut while a turn is running, and says why instead of going quiet', () => {
    const onClearHistory = vi.fn();
    renderRoom({
      room: soleSeat(),
      onClearHistory,
      live: { turn: { agentId: 'scout', turnId: 't1', text: 'thinking' }, error: null },
    });

    fireEvent.click(screen.getByTestId('clear-history'));
    expect(screen.getByTestId('confirm-blocked')).toHaveTextContent('A turn is running');
    fireEvent.click(screen.getByTestId('confirm-history-change-yes'));
    expect(onClearHistory).not.toHaveBeenCalled();
  });
});

describe('RoomConversation model selection', () => {
  const soleSeat = (): RoomState =>
    room({
      participants: [
        { id: 'user', kind: 'human', display_name: 'Kenny' },
        { id: 'scout', kind: 'agent', display_name: 'Scout' },
      ],
    });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: const A_ROOM_SEAT_CAN_CARRY_A_MODEL: boolean = false;
  // Becomes: const A_ROOM_SEAT_CAN_CARRY_A_MODEL: boolean = true;
  it('offers no model here while the room route carries none (#1235)', () => {
    // Everything the picker needed is supplied -- one seated clone, a model list, a
    // handler -- so this fails the moment the control comes back without the wire behind
    // it. It was hidden rather than deleted because what it should do is #1235's question;
    // what it did was state a choice `POST /api/rooms/{id}/messages` never carried, over a
    // default option naming the workspace model, which does not serve a room turn either.
    renderRoom({
      room: soleSeat(),
      availableModels: ['qwen3:8b', 'hermes3:8b'],
      currentModel: 'hermes3:8b',
      onSelectAgentModel: () => {},
    });

    expect(screen.queryByTestId('room-model-select')).toBeNull();
    expect(screen.queryByText('Model for this conversation')).toBeNull();
  });
});

/**
 * Markdown in the conversation (#1225).
 *
 * The retirement of `PlaygroundTab` (#1234) left both render sites printing their text into a
 * `<p>`. These assert *structure* -- a `<pre>`, a `<table>` -- rather than the absence of
 * backticks: a row that rendered "```" literally and a row that rendered nothing at all both
 * pass a `not.toContain('```')`, and only one of them is the bug.
 *
 * The live-turn site gets its own cases because it is the one a reader watches. A fix applied
 * only to the settled row still shows a reply streaming in as pipes and snapping into a table
 * when it lands.
 */
describe('RoomConversation markdown (#1225)', () => {
  const FENCED = 'Here:\n\n```python\nprint("hi")\n```\n';
  const TABLE = '| column one | column two |\n| --- | --- |\n| a | b |\n';

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: <MessageBody text={message.content} />
  // Becomes: <p>{message.content}</p>
  it('renders a fenced code block in a settled row as a code block, not as its backticks', () => {
    renderRoom({
      room: room({ transcript: [message({ seq: 1, sender_id: 'scout', content: FENCED })] }),
    });

    const body = screen.getByTestId('row-body-1');
    expect(body.querySelector('pre')).not.toBeNull();
    expect(body.querySelector('code')).not.toBeNull();
    // `PreBlock`'s language badge: the block is the renderer's, not a bare <pre>.
    expect(body).toHaveTextContent('python');
    expect(body.textContent).not.toContain('```');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: <MessageBody text={message.content} />
  // Becomes: <p>{message.content}</p>
  it('renders a table in a settled row as a table, inside the scroller that keeps it in the column', () => {
    renderRoom({
      room: room({ transcript: [message({ seq: 1, sender_id: 'scout', content: TABLE })] }),
    });

    const body = screen.getByTestId('row-body-1');
    const table = body.querySelector('table');
    expect(table).not.toBeNull();
    expect(table?.querySelectorAll('th')).toHaveLength(2);
    expect(table?.querySelectorAll('tbody tr')).toHaveLength(1);
    // #1010: a table too wide for the bubble scrolls in its own box rather than widening it.
    expect(table?.closest('[data-testid="table-scroll"]')).not.toBeNull();
    expect(body.textContent).not.toContain('| --- |');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: <MessageBody text={liveTurn.text} />
  // Becomes: <p>{liveTurn.text}</p>
  it('renders a fenced code block in the turn in flight, not only once it has landed', () => {
    renderRoom({
      live: { turn: { agentId: 'scout', turnId: 'turn-1', text: FENCED }, error: null },
    });

    const body = screen.getByTestId('live-turn-body');
    expect(body.querySelector('pre')).not.toBeNull();
    expect(body).toHaveTextContent('python');
    expect(body.textContent).not.toContain('```');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: <MessageBody text={liveTurn.text} />
  // Becomes: <p>{liveTurn.text}</p>
  it('renders a table in the turn in flight, so the transcript does not reflow when it lands', () => {
    renderRoom({
      live: { turn: { agentId: 'scout', turnId: 'turn-1', text: TABLE }, error: null },
    });

    const table = screen.getByTestId('live-turn-body').querySelector('table');
    expect(table).not.toBeNull();
    expect(table?.querySelectorAll('th')).toHaveLength(2);
  });

  it('does not parse raw HTML in a reply into elements, because model output is untrusted', () => {
    // Not a hypothetical: the text a model emits reaches this renderer unfiltered. `RichText`
    // mounts no `rehype-raw`, so an <img onerror> arrives as text. Adding one would undo this
    // silently, and nothing else in the suite would notice.
    renderRoom({
      room: room({
        transcript: [
          message({
            seq: 1,
            sender_id: 'scout',
            content: '<img src=x onerror="alert(1)"> and <b>bold</b>',
          }),
        ],
      }),
    });

    const body = screen.getByTestId('row-body-1');
    expect(body.querySelector('img')).toBeNull();
    expect(body.querySelector('b')).toBeNull();
    // No element carries the handler: it survives as text, which is the escaping working.
    expect(body.querySelector('[onerror]')).toBeNull();
    expect(body.textContent).toContain('<img src=x onerror="alert(1)">');
  });

  it('drops a javascript: href rather than rendering it as a link the reader can click', () => {
    // react-markdown's default `urlTransform` does this before `RichText`'s anchor override
    // is reached. Asserted here because the override takes `href` and passes it on, so a
    // future `urlTransform={(u) => u}` on that component would land silently.
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: '[click me](javascript:alert(1))' }),
        ],
      }),
    });

    const anchor = screen.getByTestId('row-body-1').querySelector('a');
    expect(anchor?.getAttribute('href') ?? '').not.toContain('javascript:');
  });

  it('opens a link in model output with rel="noopener noreferrer"', () => {
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: '[docs](https://example.com/a)' }),
        ],
      }),
    });

    const anchor = screen.getByTestId('row-body-1').querySelector('a');
    expect(anchor).toHaveAttribute('rel', 'noopener noreferrer');
    expect(anchor).toHaveAttribute('target', '_blank');
  });
});

describe('RoomConversation at a width the viewport cannot see (#1227)', () => {
  /**
   * The container query itself is not testable here: jsdom loads no stylesheet and resolves
   * no `@container`, so what a 160px column draws is asserted in the browser, by
   * `tests/e2e/test_room_conversation_layout_e2e.py::test_a_160px_column_between_the_rail_and_the_dock_keeps_its_controls`.
   *
   * What is testable here is the invariant that query depends on. At 320px of column it sets
   * `display: none` on every `.column-icon-only` label, and `display: none` takes the text out
   * of the accessible name computation as well as off the screen. So each button whose label
   * can be hidden must state its name some other way, or it becomes an unnamed icon for
   * exactly the reader least able to guess at it. This walks the rendered tree rather than
   * naming the four buttons, so a fifth added later is held to the same rule.
   */
  const controlsWithAHideableLabel = (container: HTMLElement): string[] =>
    Array.from(container.querySelectorAll('.column-icon-only')).map((label) => {
      const button = label.closest('button');
      if (!button) return 'a hidden label outside any control';
      const name = button.getAttribute('aria-label') ?? '';
      return name.trim() === ''
        ? `${button.dataset.testid ?? '?'}: nothing names it but the label`
        : '';
    });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: aria-label="Clear this conversation"
  // Becomes: aria-label=""
  it('keeps a name on every control whose label the narrow column hides', () => {
    // Both renders, because the five controls never share one: Clear needs a conversation
    // that can be cleared, and Stop stands where Send stands while a turn runs.
    const idle = renderRoom({ onClearHistory: () => {} });
    const running = renderRoom({
      live: { turn: { agentId: 'scout', turnId: 't1', text: 'thinking' }, error: null },
    });

    const seen = [
      ...idle.container.querySelectorAll('.column-icon-only'),
      ...running.container.querySelectorAll('.column-icon-only'),
    ].map((label) => label.closest('button')?.dataset.testid);
    expect(new Set(seen)).toEqual(
      new Set(['add-someone', 'clear-history', 'dictate', 'send-message', 'stop-turn']),
    );

    expect([
      ...controlsWithAHideableLabel(idle.container),
      ...controlsWithAHideableLabel(running.container),
    ]).toEqual(['', '', '', '', '', '', '']);
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: className="column-scope flex flex-col h-full min-h-0"
  // Becomes: className="flex flex-col h-full min-h-0"
  it('makes the conversation its own container, since the rail and dock take its width', () => {
    renderRoom();

    expect(screen.getByTestId('room-conversation')).toHaveClass('column-scope');
  });
});

describe('RoomConversation marks the turn that failed (#1228)', () => {
  /**
   * What the marker *draws* is asserted in the browser, against two rows of one transcript,
   * by `tests/e2e/test_room_failed_turn_row_e2e.py::test_a_failed_row_and_a_good_one_differ_by_more_than_their_words`.
   * jsdom resolves no Tailwind, so here the claim is the narrower one the component owns:
   * the failed row is told apart from the good one at all, and the good one is not mismarked.
   */
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: data-state={message.error ? 'failed' : undefined}
  // Becomes: data-state={undefined}
  it('marks a failed row and leaves a good one unmarked', () => {
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: 'here it is' }),
          message({ seq: 2, sender_id: 'scout', content: '', error: 'provider unavailable' }),
        ],
      }),
    });

    expect(screen.getByTestId('row-body-2')).toHaveAttribute('data-state', 'failed');
    expect(screen.getByTestId('row-body-1')).not.toHaveAttribute('data-state');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: message.error && 'border-l-2 border-rose-800 pl-3',
  // Becomes: message.error && 'pl-3',
  it('draws the mark flat, and only on the row that failed', () => {
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: 'here it is' }),
          message({ seq: 2, sender_id: 'scout', content: '', error: 'provider unavailable' }),
        ],
      }),
    });

    const failed = screen.getByTestId('row-body-2');
    expect(failed).toHaveClass('border-l-2', 'border-rose-800');
    expect(screen.getByTestId('row-body-1').className).not.toMatch(/border/);
    // Visual Restraint (#1061): the marker is a rule, not an alarm.
    expect(failed.className).not.toMatch(/animate-|shadow-|bg-gradient/);
  });
});

describe('RoomConversation auto-scroll (adapted from uclone2)', () => {
  it('scrolls immediately to bottom on initial load with behavior auto', () => {
    const scrollToSpy = vi.fn();
    Element.prototype.scrollTo = scrollToSpy;

    renderRoom({
      room: room({
        transcript: [message({ seq: 1, content: 'hello' })],
      }),
    });

    expect(scrollToSpy).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: 'auto' }),
    );
  });

  it('smoothly scrolls to bottom when a new message is added', () => {
    const scrollToSpy = vi.fn();
    Element.prototype.scrollTo = scrollToSpy;

    const { rerender } = renderRoom({
      room: room({
        transcript: [message({ seq: 1, content: 'first message' })],
      }),
    });

    // Initial load used 'auto'
    expect(scrollToSpy).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: 'auto' }),
    );

    scrollToSpy.mockClear();

    // Re-render with an added message
    rerender(
      <DraftHost
        initial=""
        props={{
          room: room({
            transcript: [
              message({ seq: 1, content: 'first message' }),
              message({ seq: 2, content: 'second message' }),
            ],
          }),
          availableAgents: agents,
          live: EMPTY_LIVE,
          onSend: () => {},
          onStop: () => {},
          onRetry: () => {},
          onAddAgent: () => {},
          onTyping: () => {},
          onOpenTurn: () => {},
        }}
      />,
    );

    expect(scrollToSpy).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: 'smooth' }),
    );
  });

  it('smoothly scrolls to bottom when live turn starts', () => {
    const scrollToSpy = vi.fn();
    Element.prototype.scrollTo = scrollToSpy;

    const { rerender } = renderRoom({
      room: room({
        transcript: [message({ seq: 1, content: 'first message' })],
      }),
      live: EMPTY_LIVE,
    });

    scrollToSpy.mockClear();

    // Turn begins
    rerender(
      <DraftHost
        initial=""
        props={{
          room: room({
            transcript: [message({ seq: 1, content: 'first message' })],
          }),
          availableAgents: agents,
          live: {
            turn: { turnId: 't1', agentId: 'scout', text: 'Thinking...' },
            error: null,
          },
          onSend: () => {},
          onStop: () => {},
          onRetry: () => {},
          onAddAgent: () => {},
          onTyping: () => {},
          onOpenTurn: () => {},
        }}
      />,
    );

    expect(scrollToSpy).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: 'smooth' }),
    );
  });

  it('uses instant auto scroll during live turn streaming to prevent jitter', () => {
    const scrollToSpy = vi.fn();
    Element.prototype.scrollTo = scrollToSpy;

    const { rerender } = renderRoom({
      room: room({
        transcript: [message({ seq: 1, content: 'first message' })],
      }),
      live: {
        turn: { turnId: 't1', agentId: 'scout', text: 'Chunk 1' },
        error: null,
      },
    });

    scrollToSpy.mockClear();

    // Next chunk arrives while turn continues
    rerender(
      <DraftHost
        initial=""
        props={{
          room: room({
            transcript: [message({ seq: 1, content: 'first message' })],
          }),
          availableAgents: agents,
          live: {
            turn: { turnId: 't1', agentId: 'scout', text: 'Chunk 1 and 2' },
            error: null,
          },
          onSend: () => {},
          onStop: () => {},
          onRetry: () => {},
          onAddAgent: () => {},
          onTyping: () => {},
          onOpenTurn: () => {},
        }}
      />,
    );

    expect(scrollToSpy).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: 'auto' }),
    );
  });

  it('resets initial load and scrolls with behavior auto when room changes', () => {
    const scrollToSpy = vi.fn();
    Element.prototype.scrollTo = scrollToSpy;

    const { rerender } = renderRoom({
      room: room({
        room_id: 'r1',
        transcript: [message({ seq: 1, content: 'hello in r1' })],
      }),
    });

    scrollToSpy.mockClear();

    // Switch to r2
    rerender(
      <DraftHost
        initial=""
        props={{
          room: room({
            room_id: 'r2',
            transcript: [message({ seq: 1, content: 'hello in r2' })],
          }),
          availableAgents: agents,
          live: EMPTY_LIVE,
          onSend: () => {},
          onStop: () => {},
          onRetry: () => {},
          onAddAgent: () => {},
          onTyping: () => {},
          onOpenTurn: () => {},
        }}
      />,
    );

    expect(scrollToSpy).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: 'auto' }),
    );
  });
});

describe('RoomConversation auto-scroll leaves a reader who scrolled up where they are', () => {
  // jsdom lays nothing out, so the transcript is given a geometry: 3000px of content in a
  // 600px box. `scrollTo` is replaced with one that moves it the way a browser would, so
  // "at the bottom" and "scrolled up" are positions the component can actually read.
  const HEIGHT = 3000;
  const VIEW = 600;
  const BOTTOM = HEIGHT - VIEW;
  const originalScrollTo = Element.prototype.scrollTo;

  afterEach(() => {
    Element.prototype.scrollTo = originalScrollTo;
  });

  const laidOut = () => {
    const calls: Array<ScrollToOptions> = [];
    Element.prototype.scrollTo = function (this: Element, options?: ScrollToOptions | number) {
      if (typeof options === 'object') {
        calls.push(options);
        (this as HTMLElement).scrollTop = Math.min(options.top ?? 0, BOTTOM);
      }
    } as typeof Element.prototype.scrollTo;
    return calls;
  };

  const giveGeometry = (el: HTMLElement) => {
    let top = 0;
    Object.defineProperty(el, 'scrollHeight', { configurable: true, get: () => HEIGHT });
    Object.defineProperty(el, 'clientHeight', { configurable: true, get: () => VIEW });
    Object.defineProperty(el, 'scrollTop', {
      configurable: true,
      get: () => top,
      set: (v: number) => {
        top = Math.max(0, Math.min(v, BOTTOM));
      },
    });
  };

  const streaming = (text: string) => ({
    turn: { turnId: 't1', agentId: 'scout', text },
    error: null,
  });

  const props = (
    over: Partial<React.ComponentProps<typeof RoomConversation>>,
  ): Omit<React.ComponentProps<typeof RoomConversation>, 'draft' | 'onDraftChange'> => ({
    room: room({ transcript: [message({ seq: 1, content: 'first message' })] }),
    availableAgents: agents,
    live: EMPTY_LIVE,
    onSend: () => {},
    onStop: () => {},
    onRetry: () => {},
    onAddAgent: () => {},
    onTyping: () => {},
    onOpenTurn: () => {},
    ...over,
  });

  /** Open a conversation mid-reply, pinned to the bottom, and hand back its transcript. */
  const openStreaming = () => {
    const calls = laidOut();
    const view = render(<DraftHost initial="" props={props({ live: streaming('Chunk 1') })} />);
    const transcript = screen.getByTestId('transcript');
    giveGeometry(transcript);
    // A delta after the geometry exists, so the component has put the reader at the bottom.
    view.rerender(<DraftHost initial="" props={props({ live: streaming('Chunk 1 2') })} />);
    expect(transcript.scrollTop).toBe(BOTTOM);
    calls.length = 0;
    return { view, transcript, calls };
  };

  it('does not scroll a reader back down while a reply streams', () => {
    // The e2e flake behind this (#1368 made it visible): a delta rendered after the reader
    // scrolled up put them straight back at the bottom -- before the scroll event for their
    // own scroll had even been dispatched. No scroll event is fired here, on purpose.
    const { view, transcript, calls } = openStreaming();

    transcript.scrollTop = 0;
    view.rerender(<DraftHost initial="" props={props({ live: streaming('Chunk 1 2 3') })} />);

    expect(calls).toEqual([]);
    expect(transcript.scrollTop).toBe(0);
  });

  it('does not scroll a reader back down when they scroll up within the bottom slack', () => {
    // A reader who scrolls up only a few pixels (e.g. 10px, inside the 32px bottom slack)
    // has still deliberately moved away from the bottom. Auto-scroll must not fight their
    // upward scroll by snapping them back down on the next streamed delta.
    const { view, transcript, calls } = openStreaming();

    transcript.scrollTop = BOTTOM - 10;
    fireEvent.scroll(transcript);
    view.rerender(<DraftHost initial="" props={props({ live: streaming('Chunk 1 2 3') })} />);

    expect(calls).toEqual([]);
    expect(transcript.scrollTop).toBe(BOTTOM - 10);
  });

  it('does not scroll a reader back down when the reply lands as a new row', () => {
    const { view, transcript, calls } = openStreaming();

    transcript.scrollTop = 100;
    fireEvent.scroll(transcript);
    view.rerender(
      <DraftHost
        initial=""
        props={props({
          room: room({
            transcript: [
              message({ seq: 1, content: 'first message' }),
              message({ seq: 2, content: 'the reply, landed' }),
            ],
          }),
        })}
      />,
    );

    expect(calls).toEqual([]);
    expect(transcript.scrollTop).toBe(100);
  });

  it('follows again once the reader is back at the bottom', () => {
    // The other half: following is suspended by the reader, not switched off for good.
    const { view, transcript, calls } = openStreaming();

    transcript.scrollTop = 0;
    fireEvent.scroll(transcript);
    transcript.scrollTop = BOTTOM;
    fireEvent.scroll(transcript);
    view.rerender(<DraftHost initial="" props={props({ live: streaming('Chunk 1 2 3') })} />);

    expect(calls).toEqual([expect.objectContaining({ top: HEIGHT, behavior: 'auto' })]);
  });

  it('follows again when the reader sends, even from up in the history', async () => {
    // The owner re-reads the conversation while the send is still awaited, so the user's
    // own message lands before `onSend` resolves. It is followed, not left below the fold.
    const calls = laidOut();
    let release: () => void = () => {};
    const onSend = () =>
      new Promise<void>((resolve) => {
        release = resolve;
      });
    const view = render(<DraftHost initial="" props={props({ onSend })} />);
    const transcript = screen.getByTestId('transcript');
    giveGeometry(transcript);
    // At the bottom, and then up into the history: the reader has stopped following.
    transcript.scrollTop = BOTTOM;
    fireEvent.scroll(transcript);
    transcript.scrollTop = 0;
    fireEvent.scroll(transcript);

    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'hello' } });
    await act(async () => {
      fireEvent.click(screen.getByTestId('send-message'));
    });
    calls.length = 0;
    view.rerender(
      <DraftHost
        initial=""
        props={props({
          onSend,
          room: room({
            transcript: [
              message({ seq: 1, content: 'first message' }),
              message({ seq: 2, sender_id: 'user', content: 'hello' }),
            ],
          }),
        })}
      />,
    );

    expect(calls).toEqual([expect.objectContaining({ top: HEIGHT, behavior: 'smooth' })]);
    await act(async () => {
      release();
    });
  });

  describe('saying new text arrived below a reader who scrolled up (#1380)', () => {
    const jump = () => screen.queryByRole('button', { name: /jump to latest/i });

    it('offers nothing while the reader is following the bottom edge', () => {
      const { view } = openStreaming();
      view.rerender(<DraftHost initial="" props={props({ live: streaming('Chunk 1 2 3') })} />);

      expect(jump()).toBeNull();
    });

    it('offers nothing to a reader who scrolled up while nothing new has arrived', () => {
      // "New text below" is a claim; it is made only once there is new text to point at.
      const { transcript } = openStreaming();
      transcript.scrollTop = 0;
      fireEvent.scroll(transcript);

      expect(jump()).toBeNull();
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: setUnseenBelow(true);
    // Becomes: void 0;
    it('offers a way back down once new words arrive below a reader who scrolled up', () => {
      const { view, transcript } = openStreaming();
      transcript.scrollTop = 0;
      fireEvent.scroll(transcript);
      view.rerender(<DraftHost initial="" props={props({ live: streaming('Chunk 1 2 3') })} />);

      const button = jump();
      expect(button).not.toBeNull();
      // A real button, so Tab reaches it and Enter or Space presses it.
      expect(button?.tagName).toBe('BUTTON');
      expect(button).not.toHaveAttribute('tabindex', '-1');
      // Quiet: nothing on it that moves.
      expect(button?.className ?? '').not.toMatch(/animate-/);
      expect(transcript.scrollTop).toBe(0);
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {unseenBelow ? (
    // Becomes: {unseenBelow && false ? (
    it('also offers it when the reply lands as a row rather than as more streamed words', () => {
      const { view, transcript } = openStreaming();
      transcript.scrollTop = 0;
      fireEvent.scroll(transcript);
      view.rerender(
        <DraftHost
          initial=""
          props={props({
            room: room({
              transcript: [
                message({ seq: 1, content: 'first message' }),
                message({ seq: 2, sender_id: 'scout', content: 'the reply, landed' }),
              ],
            }),
          })}
        />,
      );

      expect(jump()).not.toBeNull();
    });

    // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: onClick={jumpToLatest}
    // Becomes: onClick={() => {}}
    it('takes the reader to the bottom, follows again, and goes away', () => {
      const { view, transcript, calls } = openStreaming();
      transcript.scrollTop = 0;
      fireEvent.scroll(transcript);
      view.rerender(<DraftHost initial="" props={props({ live: streaming('Chunk 1 2 3') })} />);

      fireEvent.click(jump() as HTMLElement);

      expect(calls).toEqual([expect.objectContaining({ top: HEIGHT })]);
      expect(transcript.scrollTop).toBe(BOTTOM);
      expect(jump()).toBeNull();
      // Following again: the next delta keeps the reader at the edge.
      calls.length = 0;
      view.rerender(<DraftHost initial="" props={props({ live: streaming('Chunk 1 2 3 4') })} />);
      expect(calls).toEqual([expect.objectContaining({ top: HEIGHT, behavior: 'auto' })]);
    });

    it('goes away when the reader scrolls back down by hand', () => {
      const { view, transcript } = openStreaming();
      transcript.scrollTop = 0;
      fireEvent.scroll(transcript);
      view.rerender(<DraftHost initial="" props={props({ live: streaming('Chunk 1 2 3') })} />);
      expect(jump()).not.toBeNull();

      transcript.scrollTop = BOTTOM;
      fireEvent.scroll(transcript);

      expect(jump()).toBeNull();
    });
  });
});

describe('RoomConversation shows a reply once, however its row and its stream interleave (#1379)', () => {
  // Two channels carry one reply, and they do not arrive in step. The row comes from a read
  // of the room -- the send's own re-read can return it while the stream is still delivering
  // deltas -- and the bubble is cleared by the stream's `final`, which can be well behind.
  // Measured on the committed bundle before this change: 96 consecutive frames with both
  // `row-4` and `live-turn` on screen for one reply, then one frame with neither at the
  // handoff on a slow stream.
  const REPLY = 'The composite index covers the query';
  const exchange = [
    message({ seq: 1, sender_id: 'user', content: 'Is the index used?' }),
  ];
  const landed = [
    ...exchange,
    message({ seq: 2, sender_id: 'scout', content: REPLY, turn_id: 't1' }),
  ];
  const solo = (transcript: RoomTranscriptMessage[]) =>
    room({
      participants: [
        { id: 'user', kind: 'human', display_name: 'Kenny' },
        { id: 'scout', kind: 'agent', display_name: 'Scout' },
      ],
      transcript,
    });
  const streaming = (text: string, turnId = 't1') => ({
    turn: { agentId: 'scout', turnId, text },
    error: null,
  });
  const copies = () =>
    screen.queryAllByText((_content, el) => el?.tagName === 'P' && (el.textContent ?? '').includes('The composite index'));

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: !room.transcript.some((row) => row.turn_id === inFlight.turnId)
  // Becomes: true
  it('draws the reply once when its row lands while its words are still streaming', () => {
    renderRoom({ room: solo(landed), live: streaming('The composite index') });

    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY);
    expect(screen.queryByTestId('live-turn')).toBeNull();
    expect(copies()).toHaveLength(1);
  });

  it('still draws another turn in flight beside a landed row that is not its own', () => {
    // The suppression is by turn, never "a row is newer than the bubble": a second clone's
    // reply streaming under the first one's landed row is a different reply.
    renderRoom({ room: solo(landed), live: streaming('A second opinion', 't2') });

    expect(screen.getByTestId('live-turn')).toHaveTextContent('A second opinion');
    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY);
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: const inFlight = live.turn ?? heldRef.current?.turn ?? null;
  // Becomes: const inFlight = live.turn;
  it('keeps the words on screen between the stream ending and the row arriving', () => {
    // `final` clears the bubble at once; the row comes with the re-read it triggers. Before
    // this, the reply vanished for that round trip and came back as the row.
    const before = solo(exchange);
    const view = renderRoom({ room: before, live: streaming('The composite index covers') });
    const rerender = (r: RoomState, live: typeof EMPTY_LIVE) =>
      view.rerender(
        <DraftHost
          initial=""
          props={{
            room: r,
            availableAgents: agents,
            live,
            onSend: () => {},
            onStop: () => {},
            onRetry: () => {},
            onAddAgent: () => {},
            onTyping: () => {},
            onOpenTurn: () => {},
          }}
        />,
      );

    rerender(before, EMPTY_LIVE);
    expect(copies()).toHaveLength(1);
    expect(screen.getByTestId('live-turn')).toHaveTextContent('The composite index covers');
    // Not running: the handoff is only the words, and the controls are the reader's again.
    expect(screen.getByTestId('send-message')).toBeInTheDocument();

    rerender(solo(landed), EMPTY_LIVE);
    expect(copies()).toHaveLength(1);
    expect(screen.getByTestId('row-2')).toHaveTextContent(REPLY);
    expect(screen.queryByTestId('live-turn')).toBeNull();
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: (heldRef.current.transcript !== room.transcript || live.error !== null)
  // Becomes: live.error !== null
  it('lets the held words go when the record arrives without them', () => {
    // A clear or a rewind answered in that window: the record is the Core's, and a bubble
    // kept past it would show words the conversation no longer holds.
    const before = solo(exchange);
    const view = renderRoom({ room: before, live: streaming('The composite index covers') });
    const props = {
      availableAgents: agents,
      live: EMPTY_LIVE,
      onSend: () => {},
      onStop: () => {},
      onRetry: () => {},
      onAddAgent: () => {},
      onTyping: () => {},
      onOpenTurn: () => {},
    };
    view.rerender(<DraftHost initial="" props={{ ...props, room: before }} />);
    view.rerender(<DraftHost initial="" props={{ ...props, room: solo([]) }} />);

    expect(copies()).toHaveLength(0);
    expect(screen.queryByTestId('live-turn')).toBeNull();
  });
});

describe('RoomConversation failure copy (#1408)', () => {
  /**
   * Rows the Core landed for real failures, written by
   * `tests/unit/test_room_failed_row_copy.py`: a raised `FileNotFoundError`, a spent budget,
   * a provider that failed on a missing file, and a seat session that could not be written.
   * Their `error` / `persist_error` hold class names and paths, which is what makes the
   * "not rendered" checks below able to fail.
   */
  const rows = failedRows.rows as Record<string, Partial<RoomTranscriptMessage>>;
  const renderRow = (name: string) =>
    renderRoom({
      room: room({ transcript: [message(rows[name])] }),
    });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: <p>{turnFailureSentence(label, message.refusal, message.completed)}</p>
  // Becomes: <p>{label} couldn&apos;t finish this turn: {message.error}</p>
  it.each([
    'raised',
    'provider',
  ])('states a %s failure plainly and keeps the cause off the row', (name) => {
    renderRow(name);
    const seq = rows[name].seq;
    const line = screen.getByTestId(`row-error-${seq}`);
    expect(line).toHaveTextContent(
      "Scout couldn't finish this turn because something went wrong while it was answering.",
    );
    expect(line.textContent).not.toContain(rows[name].error as string);
    expectPlain(line.textContent);
    expect(screen.getByTestId('retry-turn')).toBeInTheDocument();
  });

  // Killed by: frontend/src/lib/turnOutcome.ts :: if (refusal) {
  // Becomes: if (false) {
  it("states a budget refusal as what happened, not the budget manager's message", () => {
    renderRow('refused');
    const line = screen.getByTestId(`row-error-${rows.refused.seq}`);
    expect(line).toHaveTextContent(
      "Scout couldn't finish this turn: it has used all the tokens this conversation allows.",
    );
    expect(line.textContent).not.toContain('limit exceeded');
    expectPlain(line.textContent);
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: it could not be saved.
  // Becomes: it could not be saved: {message.persist_error}
  it('says a reply was not saved without printing why', () => {
    renderRow('unsaved');
    const line = screen.getByTestId(`row-unsaved-${rows.unsaved.seq}`);
    expect(line).toHaveTextContent('Scout will not remember this turn after a restart; it could not be saved.');
    expect(line.textContent).not.toContain('NotADirectoryError');
    expectPlain(line.textContent);
  });

  // Killed by: frontend/src/lib/turnOutcome.ts :: if (!completed) return
  // Becomes: if (false) return
  it('says a turn was stopped, apart from one that went wrong', () => {
    // `completed: false` with the Core's own interrupted text, as `_take_turn` lands a
    // cancelled turn.
    renderRoom({
      room: room({
        transcript: [
          message({ seq: 1, sender_id: 'scout', content: '', completed: false, error: 'Turn was interrupted' }),
        ],
      }),
    });
    expect(screen.getByTestId('row-error-1')).toHaveTextContent('Scout was stopped before finishing this turn.');
  });

  // A send that got no answer may have been stored, so it is not called "not sent" (#1441).
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: setSendError(sendFailureNotice(err));
  // Becomes: setSendError(err instanceof Error ? err.message : String(err));
  it('says a send that never got an answer plainly, claims nothing, and keeps the draft', async () => {
    // What `fetch` rejects with when the Core is not there, as `roomsApi` wraps it.
    const noAnswer = new RoomsNoAnswerError(new TypeError('Failed to fetch'));
    renderRoom({ onSend: vi.fn().mockRejectedValue(new RoomSendError('unconfirmed', noAnswer)) });
    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'still here' } });
    fireEvent.click(screen.getByTestId('send-message'));

    const notice = await screen.findByTestId('send-error');
    expect(notice).toHaveTextContent(
      "Could not confirm whether this conversation has your message: The app's background service didn't answer.",
    );
    expect(notice.textContent).not.toMatch(/\bsent\b/i);
    expectPlain(notice.textContent);
    expect(screen.getByTestId('room-composer')).toHaveValue('still here');
  });

  it("does not blame the service for the app's own fault", async () => {
    renderRoom({ onSend: vi.fn().mockRejectedValue(new TypeError('room.transcript is not iterable')) });
    fireEvent.change(screen.getByTestId('room-composer'), { target: { value: 'still here' } });
    fireEvent.click(screen.getByTestId('send-message'));

    const notice = await screen.findByTestId('send-error');
    expect(notice).toHaveTextContent(
      'Could not confirm whether this conversation has your message: Something went wrong in the app.',
    );
    expect(notice.textContent).not.toMatch(/service|not sent/i);
    expectPlain(notice.textContent);
  });
});

describe('RoomConversation autonomous discussion toggle', () => {
  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: <span className="column-icon-only">Auto discuss</span>
  // Becomes: <span className="column-icon-only">자율 토론</span>
  it('renders the autonomous toggle in English with proper accessible labels and invokes callback', () => {
    const onToggleAutonomous = vi.fn();
    const { rerender } = renderRoom({
      onToggleAutonomous,
      room: room({
        policy: {
          max_agent_turns_per_human_message: 3,
          max_span_messages: 40,
          transcript_window: 15,
          hesitation_seconds: 0,
          default_responder_id: '',
          autonomous: false,
        },
      }),
    });

    const toggle = screen.getByTestId('toggle-autonomous');
    expect(toggle).toHaveTextContent('Auto discuss');
    expect(toggle).toHaveTextContent('OFF');
    expect(toggle).toHaveAttribute('aria-label', 'Autonomous discussion off');
    expect(toggle).toHaveAttribute(
      'title',
      'Turn on autonomous discussion (agents converse while you view)',
    );

    fireEvent.click(toggle);
    expect(onToggleAutonomous).toHaveBeenCalledWith(true);

    rerender(
      <DraftHost
        initial=""
        props={{
          room: room({
            policy: {
              max_agent_turns_per_human_message: 3,
              max_span_messages: 40,
              transcript_window: 15,
              hesitation_seconds: 0,
              default_responder_id: '',
              autonomous: true,
            },
          }),
          availableAgents: agents,
          live: EMPTY_LIVE,
          onSend: () => {},
          onStop: () => {},
          onRetry: () => {},
          onAddAgent: () => {},
          onTyping: () => {},
          onOpenTurn: () => {},
          onToggleAutonomous,
        }}
      />,
    );

    const onToggle = screen.getByTestId('toggle-autonomous');
    expect(onToggle).toHaveTextContent('Auto discuss');
    expect(onToggle).toHaveTextContent('ON');
    expect(onToggle).toHaveAttribute('aria-label', 'Autonomous discussion on');
    expect(onToggle).toHaveAttribute(
      'title',
      'Autonomous discussion active (agents converse while you view)',
    );

    fireEvent.click(onToggle);
    expect(onToggleAutonomous).toHaveBeenCalledWith(false);
  });
});

