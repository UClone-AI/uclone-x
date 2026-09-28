import { afterEach, describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { RoomState, RoomTranscriptMessage } from '../../types';
import { EMPTY_LIVE } from '../../lib/rooms';

/**
 * How many times a message body has been drawn.
 *
 * `RichText` is replaced by a counter that is *not* memoized, so a count that stays flat can
 * only be the row above it declining to re-render -- which is what this file is about. The
 * real `RichText` is memoized as well; with it in place a row that did re-render would still
 * leave the count flat, and the test could not fail.
 */
const bodies = vi.hoisted(() => ({ drawn: 0 }));
vi.mock('../RichText', () => ({
  RichText: ({ content }: { content?: string }) => {
    bodies.drawn += 1;
    return <span>{content}</span>;
  },
}));

// After the mock, so the component under test imports the counter.
const { RoomConversation } = await import('./RoomConversation');

afterEach(() => {
  vi.unstubAllGlobals();
});

const message = (seq: number, sender_id: string, content: string): RoomTranscriptMessage => ({
  seq,
  sender_id,
  content,
  kind: 'utterance',
  created_at: '2026-09-26T00:00:00Z',
  completed: true,
});

const room: RoomState = {
  room_id: 'r1',
  title: 'Long conversation',
  participants: [
    { id: 'user', kind: 'human', display_name: 'Kenny' },
    { id: 'scout', kind: 'agent', display_name: 'Scout' },
  ],
  transcript: [
    message(1, 'user', 'first question'),
    message(2, 'scout', 'first answer'),
    message(3, 'user', 'second question'),
    message(4, 'scout', 'second answer'),
  ],
  turn_state: { agent_turns_since_human: 0 },
  policy: {
    max_agent_turns_per_human_message: 3,
    max_span_tokens: 8000,
    transcript_window: 15,
    hesitation_seconds: 0,
    default_responder_id: '',
  },
};

/**
 * Typing in the composer does not redraw the conversation above it.
 *
 * Every keystroke re-rendered every message, so the cost of one key grew with the length of
 * the conversation until typing visibly lagged. The rows are memoized and their callbacks held
 * stable, so a keystroke draws the box and nothing above it.
 */
/** The component as `App` draws it: fresh arrows for every callback, every render. */
const drawn = (onDraftChange: (next: string) => void = () => {}) => (
  <RoomConversation
    room={room}
    availableAgents={[]}
    live={EMPTY_LIVE}
    draft=""
    onDraftChange={onDraftChange}
    onSend={() => {}}
    onStop={() => {}}
    onRetry={() => {}}
    onAddAgent={() => {}}
    onTyping={() => {}}
    onOpenTurn={() => {}}
  />
);

describe('RoomConversation typing cost', () => {
  it('draws no message body again while the reader types', () => {
    // The usage strip reads the Core; nothing here is about it, so it gets a quiet refusal.
    vi.stubGlobal('fetch', () => Promise.resolve(new Response('{}', { status: 404 })));
    const onDraftChange = vi.fn();
    render(drawn(onDraftChange));
    expect(screen.getByText('second answer')).toBeInTheDocument();
    const before = bodies.drawn;

    const box = screen.getByTestId('room-composer');
    for (const value of ['h', 'he', 'hel', 'hell', 'hello']) {
      fireEvent.change(box, { target: { value } });
    }

    expect(box).toHaveValue('hello');
    // Every change still reaches the owner, which keeps the draft across a switch (#1290).
    expect(onDraftChange).toHaveBeenLastCalledWith('hello');
    expect(onDraftChange).toHaveBeenCalledTimes(5);
    expect(bodies.drawn).toBe(before);
  });

  it('draws no message body again when the owner re-renders with new callbacks', () => {
    // The owner re-renders on every stream event and every poll, and passes new arrows each
    // time. The rows are handed callbacks held stable here, or each of those renders would
    // redraw the whole conversation just the same.
    vi.stubGlobal('fetch', () => Promise.resolve(new Response('{}', { status: 404 })));
    const { rerender } = render(drawn());
    const before = bodies.drawn;

    rerender(drawn());
    rerender(drawn());

    expect(screen.getByText('second answer')).toBeInTheDocument();
    expect(bodies.drawn).toBe(before);
  });
});
