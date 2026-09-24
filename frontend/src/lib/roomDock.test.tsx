/**
 * Which seat the dock describes, and what a read shows the moment its scope moves (#1389).
 *
 * `defaultSeat` decides whose Activity, Remembers and Knowledge Graph the dock opens on when
 * the reader has not picked one; `useRoomRead` must never hand the new scope the old one's
 * record, not even for one render. Both escaped the whole vitest suite in #1374's review.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import { defaultSeat, useRoomRead } from './roomDock';
import type { RoomState, RoomTranscriptMessage } from '../types';

const said = (seq: number, sender_id: string, kind: RoomTranscriptMessage['kind'] = 'utterance') =>
  ({ seq, sender_id, kind, content: '', created_at: '2026-09-22T10:00:00Z', completed: true }) as
  RoomTranscriptMessage;

const roomWith = (seats: string[], transcript: RoomTranscriptMessage[] = []): RoomState =>
  ({
    room_id: 'room-a',
    title: 'A',
    participants: [
      { id: 'user', kind: 'human' },
      ...seats.map((id) => ({ id, kind: 'agent' as const })),
    ],
    transcript,
  }) as unknown as RoomState;

describe('defaultSeat', () => {
  it('is null only when no agent is seated', () => {
    expect(defaultSeat(null, 'scout')).toBeNull();
    expect(defaultSeat(roomWith([]), 'scout')).toBeNull();
  });

  // Killed by: frontend/src/lib/roomDock.ts :: if (preferredClone && seats.some((s) => s.id === preferredClone)) return preferredClone;
  // Becomes: if (false) return preferredClone;
  it('follows the clone the reader picked in the rail, when it is seated here', () => {
    const room = roomWith(['scout', 'critic'], [said(1, 'user'), said(2, 'scout')]);
    expect(defaultSeat(room, 'critic')).toBe('critic');
  });

  it('ignores a rail pick that is not seated here', () => {
    const room = roomWith(['scout', 'critic'], [said(1, 'critic')]);
    expect(defaultSeat(room, 'elsewhere')).toBe('critic');
  });

  // Killed by: frontend/src/lib/roomDock.ts :: if (m.kind === 'utterance' && seated.has(m.sender_id)) return m.sender_id;
  // Becomes: if (false) return m.sender_id;
  it('else follows the seated agent that spoke last, skipping people, leavers and roster changes', () => {
    // Critic is seated first, so the fallback would name Critic: only the walk names Scout.
    const room = roomWith(
      ['critic', 'scout'],
      [
        said(1, 'user'),
        said(2, 'critic'),
        said(3, 'scout'),
        // Spoke later, but is no longer seated.
        said(4, 'ghost'),
        // A roster change by a seated agent is not speech.
        said(5, 'critic', 'join'),
        said(6, 'user'),
      ],
    );
    expect(defaultSeat(room, null)).toBe('scout');
  });

  it('else falls back to the first agent seated', () => {
    expect(defaultSeat(roomWith(['scout', 'critic'], [said(1, 'user')]), null)).toBe('scout');
    expect(defaultSeat(roomWith(['critic', 'scout']), undefined)).toBe('critic');
  });
});

describe('useRoomRead', () => {
  let bodies: Record<string, unknown>;

  beforeEach(() => {
    bodies = {};
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async (url: string) =>
          new Response(JSON.stringify(bodies[url] ?? { detail: `no stub for ${url}` }), {
            status: bodies[url] === undefined ? 404 : 200,
          }),
      ),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  /** Every value the hook returned, render by render, with the address it was asked for. */
  const seen: { url: string | null; data: unknown; loading: boolean }[] = [];
  const Probe = ({ url }: { url: string | null }) => {
    const read = useRoomRead<{ who: string }>(url);
    seen.push({ url, data: read.data, loading: read.loading });
    return <span>{read.data?.who ?? ''}</span>;
  };
  const settle = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));

  // Killed by: frontend/src/lib/roomDock.ts :: if (state.url !== url) return { data: null, error: null, fault: null, loading: true };
  // Becomes: if (false) return { data: null, error: null, fault: null, loading: true };
  it("never returns one scope's record for another, not even for the render that moves it", async () => {
    seen.length = 0;
    bodies['/a'] = { who: 'scout' };
    bodies['/b'] = { who: 'critic' };
    const { rerender, container } = render(<Probe url="/a" />);
    await settle();
    expect(container.textContent).toBe('scout');

    rerender(<Probe url="/b" />);
    await settle();
    expect(container.textContent).toBe('critic');

    const forB = seen.filter((s) => s.url === '/b');
    expect(forB[0]).toEqual({ url: '/b', data: null, loading: true });
    expect(forB.some((s) => (s.data as { who: string } | null)?.who === 'scout')).toBe(false);
  });

  it('makes no request and reports nothing loading when there is no scope', async () => {
    seen.length = 0;
    render(<Probe url={null} />);
    await settle();
    expect(fetch).not.toHaveBeenCalled();
    expect(seen[seen.length - 1]).toEqual({ url: null, data: null, loading: false });
  });
});
