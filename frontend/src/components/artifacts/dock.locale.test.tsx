import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ActivityTimeline } from './ActivityTimeline';
import { LocaleProvider, fmt } from '../../i18n';
import { ko } from '../../i18n/ko';
import { classifyTool } from '../../lib/toolLabels';
import type { EventEnvelope } from '../../types';
import type { RoomToolUse, SeatHistory } from '../../lib/roomDock';

/**
 * The dock's product panels follow the language control (step 4 of `multilingual-ui.md`).
 *
 * `ActivityTimeline.test.tsx` renders without a provider and so reads English; these render
 * inside one, the way `main.tsx` does, and read Korean. Tool names stay as the tool is named.
 */

const HISTORY = '/api/rooms/room-a/seats/scout/history';

const use = (over: Partial<RoomToolUse>): RoomToolUse => ({
  turn_id: 't1',
  participant_id: 'scout',
  tool_name: 'read_file',
  tool_call_id: 'call-1',
  status: 'success',
  error: null,
  duration_ms: 12,
  arguments_preview: '{"path": "notes.md"}',
  output_preview: 'hello',
  truncated: false,
  written_path: null,
  wrote_unnamed: false,
  subagent_id: null,
  recorded_at: '2026-09-22T10:00:00Z',
  seq: 2,
  ...over,
});

const history = (over: Partial<SeatHistory>): SeatHistory => ({
  room_id: 'room-a',
  participant_id: 'scout',
  display_name: 'Scout',
  session_id: 'sess_room__room-a__scout',
  live: true,
  turns: [],
  tool_uses: [],
  tools_note: null,
  unsaved_turns: 0,
  unsaved_note: null,
  reason: null,
  ...over,
} as SeatHistory);

let answer: unknown;

beforeEach(() => {
  answer = history({});
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) =>
      url === HISTORY
        ? new Response(JSON.stringify(answer), { status: 200 })
        : new Response(JSON.stringify({}), { status: 503 }),
    ),
  );
});

afterEach(() => vi.unstubAllGlobals());

const Korean: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <LocaleProvider hints={['ko-KR']}>{children}</LocaleProvider>
);

describe('the dock in the chosen language', () => {
  // Killed by: frontend/src/components/artifacts/ActivityTimeline.tsx :: .map((use, i) => itemFromUse(use, i, steps))
  // Becomes: .map((use, i) => itemFromUse(use, i))
  it('names a recorded step in Korean and keeps the tool name as it is', async () => {
    answer = history({ tool_uses: [use({})] });
    render(
      <Korean>
        <ActivityTimeline roomId="room-a" seatId="scout" events={[]} />
      </Korean>,
    );

    const card = await screen.findByTestId('activity-card-call-1');
    const korean = classifyTool('read_file', { path: 'notes.md' }, ko.toolSteps).summaryTitle;
    expect(korean).not.toBe(classifyTool('read_file', { path: 'notes.md' }).summaryTitle);
    expect(card).toHaveTextContent(korean);
    expect(card).toHaveTextContent('read_file');
    expect(screen.getByText(fmt(ko.dock.activity.heading, { name: 'Scout' }))).toBeInTheDocument();
  });

  // Killed by: frontend/src/components/artifacts/ActivityTimeline.tsx :: liveItems(events, roomId, seatId, history, steps)
  // Becomes: liveItems(events, roomId, seatId, history)
  it('names a live step in Korean too', async () => {
    const envelope = {
      id: 'e1',
      timestamp: '2026-09-22T10:00:00Z',
      type: 'TOOL_CALL',
      event_type: 'TOOL_CALL',
      topic: 'room.room-a.tool',
      payload: {
        room_id: 'room-a',
        participant_id: 'scout',
        turn_id: 't9',
        seq: 7,
        tool_call_id: 'call-live',
        name: 'web_search',
        arguments_preview: '{"query": "tides"}',
      },
    } as unknown as EventEnvelope;
    render(
      <Korean>
        <ActivityTimeline roomId="room-a" seatId="scout" events={[envelope]} />
      </Korean>,
    );

    const card = await screen.findByTestId('activity-card-call-live');
    expect(card).toHaveTextContent(
      classifyTool('web_search', { query: 'tides' }, ko.toolSteps).summaryTitle,
    );
    expect(card).toHaveTextContent('web_search');
  });
});
