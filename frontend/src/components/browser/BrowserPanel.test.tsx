import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import type { ReactNode } from 'react';
import { LocaleProvider } from '../../i18n';
import type { RoomState } from '../../types';
import type {
  BrowserChannel,
  BrowserLive,
  BrowserLiveState,
  BrowserOverlay,
  LiveConnection,
} from '../../lib/browserLive';
import { BrowserPanel, OVERLAY_MS, bannerKind } from './BrowserPanel';

const room: RoomState = {
  room_id: 'r1',
  title: 'Flights',
  participants: [
    { id: 'user', kind: 'human', display_name: 'Kenny' },
    { id: 'scout', kind: 'agent', display_name: 'Scout' },
  ],
  transcript: [],
  turn_state: { agent_turns_since_human: 0 },
  policy: {
    max_agent_turns_per_human_message: 3,
    max_span_tokens: 8000,
    transcript_window: 15,
    hesitation_seconds: 0,
    default_responder_id: '',
  },
};

const TAB = {
  clone: 'scout',
  index: 1,
  url: 'https://example.com/flights?from=ICN',
  title: 'Flights',
  current: true,
  controller: 'clone' as const,
};

const state = (over: Partial<BrowserLiveState> = {}): BrowserLiveState => ({
  tabs: [TAB],
  acting: null,
  problem: null,
  shown: { clone: 'scout', index: 1 },
  extension: true,
  ...over,
});

const ACTING = { acting: { clone: 'scout', action: 'click' } };
const USER = { tabs: [{ ...TAB, controller: 'user' as const }] };

const channel = () => {
  const overlays = new Set<(overlay: BrowserOverlay) => void>();
  const frames = new Set<(jpeg: Blob, size: { width: number; height: number } | null) => void>();
  const value: BrowserChannel & {
    views: [boolean, number][];
    picks: [string, number][];
    controls: string[];
    inputs: Record<string, unknown>[];
  } = {
    views: [],
    picks: [],
    controls: [],
    inputs: [],
    setView: (on, width) => value.views.push([on, width]),
    pickTab: (clone, index) => value.picks.push([clone, index]),
    control: (kind) => value.controls.push(kind),
    takeOver: () => value.controls.push('take_over'),
    giveBack: () => value.controls.push('give_back'),
    stop: () => value.controls.push('stop'),
    newTab: (url = 'about:blank') => value.controls.push(`new_tab:${url}`),
    navigate: (url) => value.controls.push(`navigate:${url}`),
    handTo: (target) => value.controls.push(`hand_to:${target}`),
    input: (message) => value.inputs.push(message),
    onFrame: (listener) => {
      frames.add(listener);
      return () => frames.delete(listener);
    },
    onOverlay: (listener) => {
      overlays.add(listener);
      return () => overlays.delete(listener);
    },
  };
  return {
    value,
    overlay: (o: BrowserOverlay) => overlays.forEach((l) => l(o)),
    frame: (size: { width: number; height: number }) =>
      frames.forEach((l) => l(new Blob([new Uint8Array([0xff, 0xd8])]), size)),
  };
};

const live = (
  s: BrowserLiveState | null,
  connection: LiveConnection = 'open',
  over: Partial<BrowserLive> = {},
): BrowserLive => ({
  connection,
  state: s,
  steps: [],
  lastStep: null,
  clearSteps: () => undefined,
  channel: channel().value,
  ...over,
});

const Korean = ({ children }: { children: ReactNode }) => (
  <LocaleProvider hints={['ko-KR']}>{children}</LocaleProvider>
);

/** Words a reader must never be shown: the socket's codes, a URL path of ours, a stack. */
const INTERNALS =
  /chrome_missing|browser_closed|take_over|give_back|\/api\/|Error|Traceback|undefined|null|\{|\}/;

describe('bannerKind', () => {
  it('says why nothing is shown before it says anything else', () => {
    expect(bannerKind(null, 'open', state())).toBe('noRoom');
    expect(bannerKind('r1', 'connecting', null)).toBe('connecting');
    expect(bannerKind('r1', 'lost', null)).toBe('lost');
    expect(bannerKind('r1', 'open', state(ACTING))).toBe('acting');
    expect(bannerKind('r1', 'open', state({ ...ACTING, ...USER }))).toBe('userControl');
    expect(bannerKind('r1', 'open', state({ problem: 'chrome_missing', tabs: [] }))).toBe(
      'chromeMissing',
    );
    expect(bannerKind('r1', 'open', state({ problem: 'browser_closed', tabs: [] }))).toBe('closed');
    expect(bannerKind('r1', 'open', state())).toBe('idle');
    expect(bannerKind('r1', 'open', state({ tabs: [], shown: null }))).toBe('none');
  });
});

describe('BrowserPanel banner', () => {
  const banner = () => screen.getByTestId('browser-banner');

  it('renders the no-room state without opening a socket', () => {
    render(<BrowserPanel roomId={null} />);
    expect(screen.getByTestId('browser-no-room')).toHaveTextContent('No conversation is open.');
  });

  it('names the clone using the browser; Take over and Stop reach the socket', () => {
    const { value } = channel();
    render(
      <BrowserPanel roomId="r1" room={room} live={live(state(ACTING), 'open', { channel: value })} />,
    );
    expect(banner()).toHaveTextContent('Scout is using the browser');
    fireEvent.click(within(banner()).getByRole('button', { name: 'Take over' }));
    fireEvent.click(within(banner()).getByRole('button', { name: 'Stop' }));
    expect(value.controls).toEqual(['take_over', 'stop']);
  });

  it('while U0 drives, says so, gives back to the clone by name, and lets the address be typed', () => {
    const { value } = channel();
    render(
      <BrowserPanel roomId="r1" room={room} live={live(state(USER), 'open', { channel: value })} />,
    );
    expect(banner()).toHaveTextContent('You have control');
    fireEvent.click(within(banner()).getByRole('button', { name: 'Give back to Scout' }));
    expect(value.controls).toEqual(['give_back']);
    expect(screen.getByRole('textbox', { name: 'Page address' })).not.toHaveAttribute('readonly');
  });

  it('says who finished when idle, and shows the address read-only', () => {
    render(<BrowserPanel roomId="r1" room={room} live={live(state())} />);
    expect(banner()).toHaveTextContent('Idle — Scout finished');
    const address = screen.getByRole('textbox', { name: 'Page address' });
    expect(address).toHaveValue(TAB.url);
    expect(address).toHaveAttribute('readonly');
  });

  it('offers Get Chrome when Chrome is missing', () => {
    render(
      <BrowserPanel
        roomId="r1"
        room={room}
        live={live(state({ problem: 'chrome_missing', tabs: [], shown: null }))}
      />,
    );
    expect(banner()).toHaveTextContent("Google Chrome isn't installed.");
    expect(screen.getByRole('link', { name: 'Get Chrome' })).toHaveAttribute(
      'href',
      'https://www.google.com/chrome/',
    );
  });

  it('says the browser was closed without offering a button', () => {
    render(
      <BrowserPanel
        roomId="r1"
        room={room}
        live={live(state({ problem: 'browser_closed', tabs: [], shown: null }))}
      />,
    );
    expect(banner()).toHaveTextContent('The browser was closed.');
    expect(within(banner()).queryByRole('button')).toBeNull();
  });

  it('with nothing open says so, and offers to connect U0’s Chrome only when it is not', () => {
    const onConnect = vi.fn();
    const { rerender } = render(
      <BrowserPanel
        roomId="r1"
        room={room}
        live={live(state({ tabs: [], shown: null, extension: false }))}
        onConnect={onConnect}
      />,
    );
    expect(banner()).toHaveTextContent('No browser is open in this conversation.');
    fireEvent.click(screen.getByRole('button', { name: 'Connect your Chrome' }));
    expect(onConnect).toHaveBeenCalledTimes(1);
    rerender(
      <BrowserPanel
        roomId="r1"
        room={room}
        live={live(state({ tabs: [], shown: null, extension: true }))}
        onConnect={onConnect}
      />,
    );
    expect(screen.queryByRole('button', { name: 'Connect your Chrome' })).toBeNull();
  });

  it('says it is connecting, then that the live view was lost', () => {
    const { rerender } = render(
      <BrowserPanel roomId="r1" room={room} live={live(null, 'connecting')} />,
    );
    expect(banner()).toHaveTextContent('Connecting to the browser…');
    rerender(<BrowserPanel roomId="r1" room={room} live={live(null, 'lost')} />);
    expect(banner()).toHaveTextContent('The live view is not connected. Trying again…');
  });

  it('shows no code, path or error text in any state, in either language', () => {
    const states: [BrowserLiveState | null, LiveConnection][] = [
      [null, 'connecting'],
      [null, 'lost'],
      [state(ACTING), 'open'],
      [state(USER), 'open'],
      [state(), 'lost'],
      [state({ problem: 'chrome_missing', tabs: [], shown: null }), 'open'],
      [state({ problem: 'browser_closed', tabs: [], shown: null }), 'open'],
      [state({ tabs: [], shown: null, extension: false }), 'open'],
    ];
    for (const wrap of [undefined, Korean]) {
      for (const [s, connection] of states) {
        const { container, unmount } = render(
          <BrowserPanel
            roomId="r1"
            room={room}
            live={live(s, connection, {
              lastStep: { clone: 'scout', action: 'click', element: '', ok: false },
            })}
            onConnect={() => undefined}
          />,
          wrap ? { wrapper: wrap } : undefined,
        );
        expect(container.textContent ?? '').not.toMatch(INTERNALS);
        unmount();
      }
    }
  });

  it('speaks Korean in 합니다체', () => {
    render(
      <BrowserPanel
        roomId="r1"
        room={{
          ...room,
          participants: [{ id: 'scout', kind: 'agent', display_name: '스카우트' }],
        }}
        live={live(state(ACTING))}
      />,
      { wrapper: Korean },
    );
    expect(screen.getByTestId('browser-banner')).toHaveTextContent(
      '스카우트가 브라우저를 사용하고 있습니다',
    );
    expect(screen.getByRole('button', { name: '중지' })).toBeInTheDocument();
  });

  it('renders askedUser banner with message, take over button, and calls takeOver', () => {
    // Killed by: frontend/src/components/browser/BrowserPanel.tsx :: if (state.asked_user) return 'askedUser';
    // Becomes: if (false) return 'askedUser';
    const { value } = channel();
    render(
      <BrowserPanel
        roomId="r1"
        room={room}
        live={live(
          state({
            asked_user: {
              clone: 'scout',
              kind: 'sign_in',
              site: 'example.com',
              message: 'Please sign in',
            },
          }),
          'open',
          { channel: value },
        )}
      />,
    );
    expect(screen.getByTestId('browser-banner')).toHaveTextContent(
      'Scout needs you to sign in to example.com',
    );
    const takeOverBtn = screen.getByTestId('browser-btn-take-over');
    expect(takeOverBtn).toBeInTheDocument();
    fireEvent.click(takeOverBtn);
    expect(value.controls).toContain('take_over');
  });

  it('allows clicking + button to open new tab', () => {
    // Killed by: frontend/src/components/browser/BrowserPanel.tsx :: onClick={() => channel.newTab()}
    // Becomes: onClick={() => {}}
    const { value } = channel();
    render(
      <BrowserPanel roomId="r1" room={room} live={live(state(), 'open', { channel: value })} />,
    );
    const newTabBtn = screen.getByTestId('browser-btn-new-tab');
    fireEvent.click(newTabBtn);
    expect(value.controls).toContain('new_tab:about:blank');
  });

  it('allows entering URL in address bar and pressing Enter to navigate when in user control', () => {
    // Killed by: frontend/src/components/browser/BrowserPanel.tsx :: channel.navigate(addressInput.trim());
    // Becomes: channel.navigate('');
    const { value } = channel();
    render(
      <BrowserPanel roomId="r1" room={room} live={live(state(USER), 'open', { channel: value })} />,
    );
    const input = screen.getByTestId('browser-address-bar').querySelector('input')!;
    fireEvent.change(input, { target: { value: 'https://github.com' } });
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
    expect(value.controls).toContain('navigate:https://github.com');
  });

  it('allows handing to another clone when in user control', () => {
    // Killed by: frontend/src/components/browser/BrowserPanel.tsx :: onHandTo(c.id);
    // Becomes: onHandTo('');
    const { value } = channel();
    const twoClonesRoom: RoomState = {
      ...room,
      participants: [
        { id: 'user', kind: 'human', display_name: 'Kenny' },
        { id: 'scout', kind: 'agent', display_name: 'Scout' },
        { id: 'analyst', kind: 'agent', display_name: 'Analyst' },
      ],
    };
    render(
      <BrowserPanel
        roomId="r1"
        room={twoClonesRoom}
        live={live(state(USER), 'open', { channel: value })}
      />,
    );
    const handToBtn = screen.getByTestId('browser-btn-hand-to');
    fireEvent.click(handToBtn);
    const analystBtn = screen.getByTestId('browser-hand-to-analyst');
    fireEvent.click(analystBtn);
    expect(value.controls).toContain('hand_to:analyst');
  });
});

describe('BrowserPanel view', () => {
  it('asks for frames while shown and stops when it goes', () => {
    const { value } = channel();
    const { unmount } = render(
      <BrowserPanel roomId="r1" room={room} live={live(state(), 'open', { channel: value })} />,
    );
    expect(value.views[0][0]).toBe(true);
    unmount();
    expect(value.views[value.views.length - 1]?.[0]).toBe(false);
  });

  it('asks for no frames while the page is hidden, and says why the view is empty', () => {
    const hidden = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
    try {
      const { value } = channel();
      render(
        <BrowserPanel roomId="r1" room={room} live={live(state(), 'open', { channel: value })} />,
      );
      expect(value.views.every(([on]) => !on)).toBe(true);
      expect(screen.getByTestId('browser-view-waiting')).toHaveTextContent(
        'The page is not shown while this tab is hidden.',
      );
    } finally {
      hidden.mockRestore();
    }
  });

  it('lists every tab by clone and title, and picking one watches it', () => {
    const { value } = channel();
    const two = state({
      tabs: [
        TAB,
        { clone: 'scout', index: 2, url: 'https://b.example/', title: '', current: false, controller: 'clone' },
      ],
    });
    render(<BrowserPanel roomId="r1" room={room} live={live(two, 'open', { channel: value })} />);
    expect(screen.getByRole('button', { name: 'Scout · Flights' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    fireEvent.click(screen.getByRole('button', { name: 'Scout · New tab' }));
    expect(value.picks).toEqual([['scout', 2]]);
  });

  it('states the last step in words', () => {
    render(
      <BrowserPanel
        roomId="r1"
        room={room}
        live={live(state(), 'open', {
          lastStep: { clone: 'scout', action: 'click', element: 'Search', ok: true },
        })}
      />,
    );
    expect(screen.getByTestId('browser-last-step')).toHaveTextContent('Last step: Clicked "Search"');
  });

  it('a click on the view while U0 drives reaches the page in its own pixels', () => {
    const { value, frame } = channel();
    render(
      <BrowserPanel roomId="r1" room={room} live={live(state(USER), 'open', { channel: value })} />,
    );
    act(() => frame({ width: 800, height: 600 }));
    const canvas = screen.getByTestId('browser-canvas');
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 400, height: 300, right: 400, bottom: 300, x: 0, y: 0,
      toJSON: () => ({}),
    });
    fireEvent.click(canvas, { clientX: 200, clientY: 150 });
    expect(value.inputs[0]).toEqual({
      event: 'mouse', mouse_type: 'mousePressed', x: 400, y: 300, button: 'left', click_count: 1,
    });
  });

  it('a click while the clone drives sends nothing', () => {
    const { value, frame } = channel();
    render(
      <BrowserPanel roomId="r1" room={room} live={live(state(), 'open', { channel: value })} />,
    );
    act(() => frame({ width: 800, height: 600 }));
    const canvas = screen.getByTestId('browser-canvas');
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 400, height: 300, right: 400, bottom: 300, x: 0, y: 0,
      toJSON: () => ({}),
    });
    fireEvent.click(canvas, { clientX: 10, clientY: 10 });
    expect(value.inputs).toEqual([]);
  });

  it('turns frames off on unmount in the ResizeObserver path', () => {
    const disconnect = vi.fn();
    const observe = vi.fn();
    class MockResizeObserver {
      observe = observe;
      disconnect = disconnect;
      unobserve = vi.fn();
    }
    const originalRO = window.ResizeObserver;
    window.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;
    try {
      const { value } = channel();
      const { unmount } = render(
        <BrowserPanel roomId="r1" room={room} live={live(state(), 'open', { channel: value })} />,
      );
      expect(observe).toHaveBeenCalled();
      unmount();
      expect(disconnect).toHaveBeenCalled();
      expect(value.views[value.views.length - 1]?.[0]).toBe(false);
    } finally {
      window.ResizeObserver = originalRO;
    }
  });
});

describe('BrowserPanel overlay', () => {
  it('outlines the element at the dock’s scale, then fades it out', () => {
    vi.useFakeTimers();
    const width = vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(400);
    try {
      const { value, overlay, frame } = channel();
      render(
        <BrowserPanel roomId="r1" room={room} live={live(state(), 'open', { channel: value })} />,
      );
      act(() => frame({ width: 800, height: 600 }));
      act(() =>
        overlay({
          clone: 'scout',
          box: { x: 100, y: 50, width: 40, height: 20 },
          label: 'Search',
          action: 'click',
        }),
      );
      const outline = screen.getByTestId('browser-overlay');
      expect(outline.style.left).toBe('50px');
      expect(outline.style.top).toBe('25px');
      expect(outline.style.width).toBe('20px');
      expect(outline.style.height).toBe('10px');
      expect(outline).toHaveTextContent('Search');
      act(() => {
        vi.advanceTimersByTime(OVERLAY_MS);
      });
      expect(screen.queryByTestId('browser-overlay')).toBeNull();
    } finally {
      width.mockRestore();
      vi.useRealTimers();
    }
  });
});

describe('BrowserPanel without a shared socket', () => {
  class MockWebSocket {
    static instances: MockWebSocket[] = [];
    url: string;
    binaryType = 'blob';
    readyState = 0;
    onopen: (() => void) | null = null;
    onclose: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onmessage: ((event: { data: unknown }) => void) | null = null;
    sent: string[] = [];
    constructor(url: string) {
      this.url = url;
      MockWebSocket.instances.push(this);
    }
    send(data: string) {
      this.sent.push(data);
    }
    close() {
      this.readyState = 3;
    }
  }
  const original = globalThis.WebSocket;
  beforeEach(() => {
    MockWebSocket.instances = [];
    (globalThis as unknown as { WebSocket: unknown }).WebSocket = MockWebSocket;
  });
  afterEach(() => {
    globalThis.WebSocket = original;
  });

  it('opens its own socket for the conversation and closes it on unmount', async () => {
    const { unmount } = render(<BrowserPanel roomId="room-42" room={room} />);
    await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));
    const ws = MockWebSocket.instances[0];
    expect(ws.url).toContain('/api/browser/room-42');
    act(() => {
      ws.onopen?.();
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'state',
          tabs: [TAB],
          acting: null,
          problem: null,
          shown: { clone: 'scout', index: 1 },
          extension: true,
        }),
      });
    });
    expect(screen.getByTestId('browser-banner')).toHaveTextContent('Idle — Scout finished');
    unmount();
    expect(ws.readyState).toBe(3);
  });
});
