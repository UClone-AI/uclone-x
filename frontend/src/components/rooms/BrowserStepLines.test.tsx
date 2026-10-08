import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import type { RoomState, RoomToolUse, RoomTranscriptMessage } from '../../types';
import { EMPTY_LIVE, type RoomLiveState } from '../../lib/rooms';
import type { BrowserStep } from '../../lib/browserLive';
import { RoomConversation } from './RoomConversation';

/**
 * A turn's browser steps in the conversation (`browser-agent.md` §3.4, §6 step 3): live under
 * the turn in flight, from the record under a landed one, each a line that fronts the dock's
 * Browser tab. What was typed and what the tool said on failure never appear.
 */

const message = (over: Partial<RoomTranscriptMessage>): RoomTranscriptMessage => ({
  seq: 1,
  sender_id: 'user',
  content: 'find me a flight',
  kind: 'utterance',
  created_at: '2026-10-01T00:00:00Z',
  completed: true,
  ...over,
});

const use = (over: Partial<RoomToolUse>): RoomToolUse => ({
  turn_id: 't1',
  participant_id: 'scout',
  tool_name: 'browser',
  tool_call_id: null,
  status: 'success',
  error: null,
  duration_ms: 5,
  arguments_preview: '{"action": "click", "ref": "e3"}',
  output_preview: '{"element": "Search", "ok": true}',
  truncated: false,
  written_path: null,
  wrote_unnamed: false,
  subagent_id: null,
  recorded_at: '2026-10-01T00:00:00Z',
  seq: 2,
  ...over,
});

const room = (over: Partial<RoomState> = {}): RoomState => ({
  room_id: 'r1',
  title: 'Flights',
  participants: [
    { id: 'user', kind: 'human', display_name: 'Kenny' },
    { id: 'scout', kind: 'agent', display_name: 'Scout' },
  ],
  transcript: [message({}), message({ seq: 2, sender_id: 'scout', content: 'Found one.', turn_id: 't1' })],
  turn_state: { agent_turns_since_human: 0 },
  policy: {
    max_agent_turns_per_human_message: 3,
    max_span_tokens: 8000,
    transcript_window: 15,
    hesitation_seconds: 0,
    default_responder_id: '',
  },
  ...over,
});

const drawn = (
  r: RoomState,
  extra: { live?: RoomLiveState; browserSteps?: BrowserStep[]; onShowBrowser?: () => void } = {},
) => (
  <RoomConversation
    room={r}
    availableAgents={[]}
    live={extra.live ?? EMPTY_LIVE}
    draft=""
    onDraftChange={() => {}}
    onSend={() => {}}
    onStop={() => {}}
    onRetry={() => {}}
    onAddAgent={() => {}}
    onTyping={() => {}}
    onOpenTurn={() => {}}
    browserSteps={extra.browserSteps}
    onShowBrowser={extra.onShowBrowser}
  />
);

beforeEach(() => {
  vi.stubGlobal('fetch', () => Promise.resolve(new Response('{}', { status: 404 })));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe('browser step lines', () => {
  it('draw a landed turn’s recorded steps, and a line fronts the Browser tab', () => {
    const onShowBrowser = vi.fn();
    render(
      drawn(
        room({
          tool_uses: [
            use({ arguments_preview: '{"action": "open", "url": "https://example.com/"}', output_preview: '{"url": "https://example.com/"}' }),
            use({}),
            use({
              arguments_preview: '{"action": "type", "ref": "e4", "text": "hunter2"}',
              output_preview: '{"element": "Password", "ok": true}',
            }),
            use({ status: 'error', error: 'CdpError: Node is detached (-32000)', output_preview: '' }),
            use({ tool_name: 'web_fetch', arguments_preview: '{"url": "x"}' }),
          ],
        }),
        { onShowBrowser },
      ),
    );
    const lines = within(screen.getByTestId('row-browser-steps-2')).getAllByTestId('browser-step');
    expect(lines.map((line) => line.textContent)).toEqual([
      'Opened a page',
      'Clicked "Search"',
      'Typed into "Password"',
      'Clicked — did not work',
    ]);
    const text = screen.getByTestId('transcript').textContent ?? '';
    expect(text).not.toContain('hunter2');
    expect(text).not.toMatch(/CdpError|-32000|detached/);
    fireEvent.click(lines[1]);
    expect(onShowBrowser).toHaveBeenCalledTimes(1);
  });

  it('draw no lines under a turn that did not browse', () => {
    render(drawn(room({ tool_uses: [use({ turn_id: 'other' })] })));
    expect(screen.queryByTestId('row-browser-steps-2')).toBeNull();
  });

  it('draw the turn in flight’s steps live, its own clone’s only', () => {
    render(
      drawn(room({ transcript: [message({})] }), {
        live: { turn: { agentId: 'scout', turnId: 't9', text: '' }, error: null },
        browserSteps: [
          { clone: 'scout', action: 'click', element: 'Search', ok: true },
          { clone: 'writer', action: 'open', element: '', ok: true },
        ],
      }),
    );
    const lines = within(screen.getByTestId('live-turn-browser-steps')).getAllByTestId('browser-step');
    expect(lines.map((line) => line.textContent)).toEqual(['Clicked "Search"']);
  });
});
