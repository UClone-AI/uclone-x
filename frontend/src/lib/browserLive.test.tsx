import { describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { en } from '../i18n/en';
import { ko } from '../i18n/ko';
import type { RoomToolUse } from '../types';
import {
  RECONNECT_MS,
  liveUrl,
  recordedStep,
  stepSentence,
  useBrowserLive,
  type LiveSocketLike,
} from './browserLive';

/** A socket the test drives: it records what the hook sends and delivers what the test says. */
class FakeSocket implements LiveSocketLike {
  binaryType = '';
  readyState = 0;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  sent: Record<string, unknown>[] = [];
  closed = false;
  constructor(readonly url: string) {}
  send(data: string): void {
    this.sent.push(JSON.parse(data) as Record<string, unknown>);
  }
  close(): void {
    this.closed = true;
  }
  open(): void {
    this.readyState = 1;
    this.onopen?.();
  }
  deliver(message: unknown): void {
    this.onmessage?.({ data: typeof message === 'string' ? message : JSON.stringify(message) });
  }
  drop(): void {
    this.readyState = 3;
    this.onclose?.();
  }
}

const harness = () => {
  const sockets: FakeSocket[] = [];
  const factory = (url: string) => {
    const socket = new FakeSocket(url);
    sockets.push(socket);
    return socket;
  };
  return { sockets, factory };
};

const STATE = {
  type: 'state',
  tabs: [{ clone: 'scout', index: 1, url: 'https://example.com/', title: 'Example', current: true }],
  acting: { clone: 'scout', action: 'click' },
  problem: null,
  shown: { clone: 'scout', index: 1 },
  extension: false,
};

const use = (over: Partial<RoomToolUse>): RoomToolUse => ({
  turn_id: 't1',
  participant_id: 'scout',
  tool_name: 'browser',
  tool_call_id: null,
  status: 'success',
  error: null,
  duration_ms: 1,
  arguments_preview: '{"action": "click", "ref": "e3"}',
  output_preview: '{"element": "검색", "ok": true}',
  truncated: false,
  written_path: null,
  wrote_unnamed: false,
  subagent_id: null,
  recorded_at: '2026-10-01T00:00:00Z',
  seq: 2,
  ...over,
});

describe('useBrowserLive', () => {
  it('dials the conversation’s own route on the page’s host', () => {
    const { sockets, factory } = harness();
    renderHook(() => useBrowserLive('room 1', factory));
    expect(sockets[0].url).toBe(liveUrl('room 1'));
    expect(sockets[0].url).toMatch(/^ws:\/\/[^/]+\/api\/browser\/room%201$/);
  });

  it('reads state and steps as data, and keeps the last step past a turn', () => {
    const { sockets, factory } = harness();
    const { result } = renderHook(() => useBrowserLive('r1', factory));
    act(() => {
      sockets[0].open();
      sockets[0].deliver(STATE);
      sockets[0].deliver({ type: 'step', clone: 'scout', action: 'click', element: '검색', ok: true });
    });
    expect(result.current.connection).toBe('open');
    expect(result.current.state?.acting).toEqual({ clone: 'scout', action: 'click' });
    expect(result.current.steps).toEqual([
      { clone: 'scout', action: 'click', element: '검색', ok: true },
    ]);
    act(() => result.current.clearSteps());
    expect(result.current.steps).toEqual([]);
    expect(result.current.lastStep?.element).toBe('검색');
  });

  it('reads an unknown problem code as no problem, and ignores what is not JSON', () => {
    const { sockets, factory } = harness();
    const { result } = renderHook(() => useBrowserLive('r1', factory));
    act(() => {
      sockets[0].open();
      sockets[0].deliver('not json');
      sockets[0].deliver({ ...STATE, acting: null, problem: 'Traceback (most recent call last)' });
    });
    expect(result.current.state?.problem).toBeNull();
  });

  it('asks for frames only once told to, and again after it reconnects', () => {
    vi.useFakeTimers();
    try {
      const { sockets, factory } = harness();
      const { result } = renderHook(() => useBrowserLive('r1', factory));
      act(() => sockets[0].open());
      expect(sockets[0].sent).toEqual([]);
      act(() => result.current.channel.setView(true, 640.4));
      expect(sockets[0].sent).toEqual([{ type: 'view', on: true, width: 640 }]);
      act(() => result.current.channel.setView(true, 640));
      expect(sockets[0].sent).toHaveLength(1);
      act(() => result.current.channel.pickTab('scout', 2));
      expect(sockets[0].sent[sockets[0].sent.length - 1]).toEqual({ type: 'tab', clone: 'scout', index: 2 });

      act(() => sockets[0].drop());
      expect(result.current.connection).toBe('lost');
      act(() => {
        vi.advanceTimersByTime(RECONNECT_MS);
      });
      expect(sockets).toHaveLength(2);
      act(() => sockets[1].open());
      expect(sockets[1].sent).toEqual([{ type: 'view', on: true, width: 640 }]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('hands frames and overlays to listeners, with the frame size', () => {
    const { sockets, factory } = harness();
    const { result } = renderHook(() => useBrowserLive('r1', factory));
    const frames: unknown[] = [];
    const overlays: unknown[] = [];
    result.current.channel.onFrame((jpeg, size) => frames.push([jpeg.size, size]));
    result.current.channel.onOverlay((overlay) => overlays.push(overlay));
    act(() => {
      sockets[0].open();
      sockets[0].deliver({ type: 'frame', width: 800, height: 600 });
      sockets[0].onmessage?.({ data: new Blob([new Uint8Array([0xff, 0xd8, 0xff])]) });
      sockets[0].deliver({ type: 'overlay', clone: 'scout', box: { x: 1, y: 2, width: 3, height: 4 }, label: 'Go', action: 'click' });
      sockets[0].deliver({ type: 'overlay', clone: 'scout', box: { x: 1, y: 'x', width: 3, height: 4 }, label: 'Go', action: 'click' });
    });
    expect(frames).toEqual([[3, { width: 800, height: 600 }]]);
    expect(overlays).toEqual([
      { clone: 'scout', box: { x: 1, y: 2, width: 3, height: 4 }, label: 'Go', action: 'click' },
    ]);
  });

  it('closes the socket when the conversation closes, and starts clean on the next', () => {
    const { sockets, factory } = harness();
    const { result, rerender } = renderHook(({ id }) => useBrowserLive(id, factory), {
      initialProps: { id: 'r1' as string | null },
    });
    act(() => {
      sockets[0].open();
      sockets[0].deliver(STATE);
    });
    rerender({ id: null });
    expect(sockets[0].closed).toBe(true);
    expect(result.current.state).toBeNull();
    rerender({ id: 'r2' });
    expect(sockets).toHaveLength(2);
    expect(sockets[1].url).toMatch(/\/api\/browser\/r2$/);
  });
});

describe('step sentences', () => {
  it('name the element, never the typed text, and say a failure plainly', () => {
    const steps = en.dock.browser.steps;
    expect(stepSentence(steps, { action: 'click', element: 'Search', ok: true })).toBe(
      'Clicked "Search"',
    );
    expect(stepSentence(steps, { action: 'type', element: 'Password', ok: true })).toBe(
      'Typed into "Password"',
    );
    expect(stepSentence(steps, { action: 'open', element: '', ok: false })).toBe(
      'Opened a page — did not work',
    );
    expect(stepSentence(steps, { action: 'clickOn', element: '', ok: true })).toBe(
      'Used the browser',
    );
    expect(stepSentence(steps, { action: 'check', element: 'Agree', ok: true })).toBe(
      'Toggled "Agree"',
    );
    expect(stepSentence(steps, { action: 'check', element: '', ok: true })).toBe(
      'Toggled a checkbox',
    );
    expect(stepSentence(ko.dock.browser.steps, { action: 'click', element: '검색', ok: true })).toBe(
      '"검색"을 클릭했습니다',
    );
    expect(stepSentence(ko.dock.browser.steps, { action: 'check', element: '동의', ok: true })).toBe(
      '"동의" 상태를 전환했습니다',
    );
    expect(stepSentence(ko.dock.browser.steps, { action: 'check', element: '', ok: true })).toBe(
      '확인란 상태를 전환했습니다',
    );
    expect(stepSentence(ko.dock.browser.steps, { action: 'open', element: '', ok: false })).toBe(
      '페이지를 열었습니다 — 실패했습니다',
    );
  });

  it('read a recorded call: the action from its arguments, the element from its result', () => {
    expect(recordedStep(use({}))).toEqual({
      clone: 'scout',
      action: 'click',
      element: '검색',
      ok: true,
    });
    const typed = recordedStep(
      use({
        arguments_preview: '{"action": "type", "ref": "e2", "text": "hunter2"}',
        output_preview: '{"element": "Pass\\"word", "ok": true}',
        status: 'error',
      }),
    );
    expect(typed).toEqual({ clone: 'scout', action: 'type', element: 'Pass"word', ok: false });
    expect(stepSentence(en.dock.browser.steps, typed)).not.toContain('hunter2');
    // Cut short by the preview bound: still a step, with nothing invented.
    expect(recordedStep(use({ arguments_preview: '{"ref": "e', output_preview: '{"elem' }))).toEqual({
      clone: 'scout',
      action: '',
      element: '',
      ok: true,
    });
  });
});
