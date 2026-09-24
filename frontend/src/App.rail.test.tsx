/**
 * The rail at the window it is given (#1062): whether it starts open, whether it divides the
 * workspace row or is drawn over it, and whether the user's collapse outlives a reload.
 *
 * jsdom lays nothing out, so these pin the decisions `App` makes and hands down -- the rail's
 * `data-overlay`, and the dock's position, which follows from the width `App` tells it the rail
 * holds. What those decisions look like on screen is pinned in the browser by
 * `tests/e2e/test_rail_responsive_e2e.py`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { App } from './App';
import { RAIL_OPEN_KEY } from './lib/rail';

const answer = (body: unknown, status = 200) =>
  ({
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ 'content-type': 'application/json' }),
    json: () => Promise.resolve(body),
  }) as unknown as Response;

class SilentEventSource {
  onopen: (() => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) {}
  addEventListener() {}
  removeEventListener() {}
  close() {}
}

const JSDOM_WIDTH = window.innerWidth;

const setWindowWidth = (width: number) => {
  Object.defineProperty(window, 'innerWidth', { configurable: true, writable: true, value: width });
};

const resizeTo = (width: number) => {
  act(() => {
    setWindowWidth(width);
    window.dispatchEvent(new Event('resize'));
  });
};

const rail = () => screen.queryByTestId('chat-sidebar');
const toggleRail = () => fireEvent.click(screen.getByTestId('toggle-sidebar'));

beforeEach(() => {
  window.localStorage.clear();
  vi.stubGlobal('EventSource', SilentEventSource);
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => {
      switch (String(input).split('?')[0]) {
        case '/api/agents':
          return Promise.resolve(answer({ agents: [{ id: 'champion', name: 'champion' }] }));
        default:
          return Promise.resolve(answer({}, 404));
      }
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  setWindowWidth(JSDOM_WIDTH);
  window.localStorage.clear();
});

describe('the rail on a first run (#1062)', () => {
  it('starts closed at a 400px window', () => {
    // Killed by: frontend/src/App.tsx :: initialRailOpen(window.innerWidth),
    // Becomes: true,
    setWindowWidth(400);
    render(<App />);

    expect(rail()).toBeNull();
    expect(screen.getByTestId('toggle-sidebar')).toHaveAttribute('title', 'Expand sidebar');
  });

  it('starts open, in the row, at a 1280px window', () => {
    setWindowWidth(1280);
    render(<App />);

    expect(rail()).toHaveAttribute('data-overlay', 'false');
  });
});

describe('the open rail at a narrow window', () => {
  it('is drawn over the conversation when opened at 400px', () => {
    // Killed by: frontend/src/App.tsx :: overlay={railIsOverlaid}
    // Becomes: overlay={false}
    setWindowWidth(400);
    render(<App />);
    toggleRail();

    expect(rail()).toHaveAttribute('data-overlay', 'true');
  });

  it('moves over the conversation when the window narrows, and stays open', () => {
    // Killed by: frontend/src/App.tsx :: const railIsOverlaid = railOverlays(windowWidth);
    // Becomes: const [railIsOverlaid] = useState(() => railOverlays(window.innerWidth));
    setWindowWidth(1280);
    render(<App />);
    expect(rail()).toHaveAttribute('data-overlay', 'false');

    resizeTo(400);
    expect(rail()).toHaveAttribute('data-overlay', 'true');

    resizeTo(1280);
    expect(rail()).toHaveAttribute('data-overlay', 'false');
  });
});

describe("the user's collapse", () => {
  it('survives a reload at a window where the rail would otherwise open', () => {
    // Killed by: frontend/src/App.tsx :: storeRailOpen(open);
    // Becomes:
    setWindowWidth(1280);
    const first = render(<App />);
    toggleRail();
    expect(rail()).toBeNull();
    expect(window.localStorage.getItem(RAIL_OPEN_KEY)).toBe('false');
    first.unmount();

    render(<App />);
    expect(rail()).toBeNull();
  });

  it("is kept when made from the rail's own collapse control", () => {
    setWindowWidth(1280);
    const first = render(<App />);
    const ownControl = within(screen.getByTestId('chat-sidebar')).getByRole('button', {
      name: 'Collapse sidebar',
    });
    fireEvent.click(ownControl);
    expect(rail()).toBeNull();
    first.unmount();

    render(<App />);
    expect(rail()).toBeNull();
  });

  it('keeps a rail opened at 400px open across a reload', () => {
    setWindowWidth(400);
    const first = render(<App />);
    toggleRail();
    first.unmount();

    render(<App />);
    expect(rail()).toHaveAttribute('data-overlay', 'true');
  });
});

describe('the width the dock is told the rail holds', () => {
  it('is none of the row while the rail is drawn over it', () => {
    // 580px: the rail overlays (under 600), and the 520px dock fits the 580px row only if the
    // rail holds none of it. Reserving the rail's 240 there would leave 340 and overlay the
    // dock over a row it fits in.
    // Killed by: frontend/src/App.tsx :: reservedWidth={isSidebarOpen && !railIsOverlaid ? RAIL_WIDTH_PX : 0}
    // Becomes: reservedWidth={isSidebarOpen ? RAIL_WIDTH_PX : 0}
    window.localStorage.setItem(RAIL_OPEN_KEY, 'true');
    setWindowWidth(580);
    render(<App />);
    fireEvent.click(screen.getByTestId('toggle-dock'));

    expect(rail()).toHaveAttribute('data-overlay', 'true');
    const dock = screen.getByTestId('artifacts-dock');
    expect(dock.className).not.toMatch(/\babsolute\b/);
    expect(dock.style.width).toBe('520px');
  });

  it('is the rail while the rail is in the row', () => {
    // 700px, rail in the row: 700 - 240 = 460 cannot seat the 520px dock, so it overlays.
    setWindowWidth(700);
    render(<App />);
    fireEvent.click(screen.getByTestId('toggle-dock'));

    expect(rail()).toHaveAttribute('data-overlay', 'false');
    expect(screen.getByTestId('artifacts-dock').className).toMatch(/\babsolute\b/);
  });
});

describe('navigating from the rail (#1062 review N3)', () => {
  // An overlaid rail covers the left 240px of the conversation it just opened, so it closes the
  // way a phone-width drawer does. A rail in the row covers nothing and stays as it is.
  beforeEach(() => {
    const base = globalThis.fetch;
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input).split('?')[0];
        const method = (init?.method ?? 'GET').toUpperCase();
        if (url === '/api/rooms' && method === 'GET') {
          return Promise.resolve(
            answer({
              rooms: [
                {
                  room_id: 'r1',
                  title: 'Index tuning',
                  agent_ids: ['champion'],
                  human_ids: [],
                  message_count: 0,
                  updated_at: '2026-09-19T00:00:00Z',
                },
              ],
              unreadable: [],
            }),
          );
        }
        return base(input, init);
      }),
    );
  });

  const openRailAt = (width: number) => {
    window.localStorage.setItem(RAIL_OPEN_KEY, 'true');
    setWindowWidth(width);
    render(<App />);
  };

  /** Click a control inside the rail once it has rendered, and let the handler settle. */
  const pickFromRail = async (testId: string) => {
    const target = await within(screen.getByTestId('chat-sidebar')).findByTestId(testId);
    await act(async () => {
      fireEvent.click(target);
    });
  };

  it.each([
    ['New', 'new-conversation-button'],
    ['a conversation', 'conversation-r1'],
  ])(
    'closes the overlaid rail when %s is picked from it at 400px',
    async (_what, testId) => {
    // Killed by: frontend/src/App.tsx :: if (railIsOverlaid) setIsSidebarOpen(false);
    // Becomes:
    openRailAt(400);
    expect(rail()).toHaveAttribute('data-overlay', 'true');

    await pickFromRail(testId);

    expect(rail()).toBeNull();
    // Closing on navigation is the drawer's, not a choice the user made: a reload still
    // opens the rail as they last left it.
    expect(window.localStorage.getItem(RAIL_OPEN_KEY)).toBe('true');
  });

  it.each([
    ['New', 'new-conversation-button'],
    ['a conversation', 'conversation-r1'],
  ])(
    'leaves the rail in the row open when %s is picked from it at 1280px',
    async (_what, testId) => {
    // Killed by: frontend/src/App.tsx :: if (railIsOverlaid) setIsSidebarOpen(false);
    // Becomes: setIsSidebarOpen(false);
    openRailAt(1280);
    expect(rail()).toHaveAttribute('data-overlay', 'false');

    await pickFromRail(testId);

    expect(rail()).toHaveAttribute('data-overlay', 'false');
  });
});

describe('new clone creation from rail', () => {
  it('opens the right dock in create mode when new clone button is clicked', async () => {
    setWindowWidth(1280);
    render(<App />);

    const newCloneBtn = await screen.findByTestId('rail-new-clone');
    fireEvent.click(newCloneBtn);

    expect(screen.getByTestId('dock-surface')).toBeInTheDocument();
    expect(screen.getByTestId('clone-profile-editor')).toBeInTheDocument();
    expect(screen.getByTestId('persona-editor')).toBeInTheDocument();
  });
});

/**
 * Studio mode: the clone editor takes the workspace, in the row (owner report, 2026-09-22).
 *
 * #1332 drew it as a `fixed` panel over the page. The dock's `backdrop-blur` makes the dock the
 * containing block of anything `fixed` inside it, so in the browser that "full screen" was the
 * dock's own box less 32px a side -- 455x791 in a 1400x900 window -- a modal-looking card inside
 * a panel. jsdom lays nothing out, so these pin the decisions that replace it: no `fixed` layer,
 * the conversation column hidden, the dock given the row. The sizes those decisions produce are
 * measured in the browser by `tests/e2e/test_persona_editor_e2e.py`.
 */
describe('Studio mode for the clone editor', () => {
  const dock = () => screen.getByTestId('artifacts-dock');
  const conversationColumn = () => screen.getByRole('main', { hidden: true });

  const openEditorInStudio = async () => {
    setWindowWidth(1280);
    render(<App />);
    fireEvent.click(await screen.findByTestId('rail-new-clone'));
    fireEvent.click(screen.getByTestId('toggle-studio-mode'));
  };

  it('draws the editor in the dock, with no fixed layer over the page', async () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: cloneStudio ? 'relative flex-1 min-w-0' : overlaid
    // Becomes: cloneStudio ? 'fixed inset-8' : overlaid
    await openEditorInStudio();

    expect(screen.queryByTestId('clone-studio-overlay')).toBeNull();
    expect(within(dock()).getByTestId('persona-editor')).toBeInTheDocument();
    expect(dock()).not.toHaveClass('fixed');
    expect(dock()).not.toHaveClass('absolute');
    expect(dock().querySelectorAll('.fixed')).toHaveLength(0);
  });

  it('gives the editor the conversation column and the dock, and leaves the rail', async () => {
    // Killed by: frontend/src/App.tsx :: ${cloneStudio ? 'hidden' : 'flex'} flex-1
    // Becomes: flex flex-1
    await openEditorInStudio();

    expect(conversationColumn()).toHaveClass('hidden');
    // The dock's dragged width no longer applies: it takes what the rail leaves of the row.
    expect(dock().style.width).toBe('');
    expect(dock()).toHaveClass('flex-1');
    expect(screen.queryByTestId('dock-resize-handle')).toBeNull();
    expect(rail()).toHaveAttribute('data-overlay', 'false');
  });

  it('keeps what was typed and opened across entering and leaving Studio mode', async () => {
    // Killed by: frontend/src/components/clones/CloneProfile.tsx :: data-studio={studio ? 'true' : 'false'}
    // Becomes: key={String(studio)} data-studio={studio ? 'true' : 'false'}
    setWindowWidth(1280);
    render(<App />);
    fireEvent.click(await screen.findByTestId('rail-new-clone'));
    fireEvent.change(screen.getByLabelText(/role/i), { target: { value: 'Cartographer' } });
    fireEvent.click(screen.getByTestId('toggle-advanced-settings'));

    fireEvent.click(screen.getByTestId('toggle-studio-mode'));
    expect(conversationColumn()).toHaveClass('hidden');
    expect(screen.getByLabelText(/role/i)).toHaveValue('Cartographer');
    expect(document.getElementById('persona-temperature')).not.toBeNull();

    fireEvent.click(screen.getByTestId('toggle-studio-mode'));
    expect(conversationColumn()).not.toHaveClass('hidden');
    expect(screen.getByLabelText(/role/i)).toHaveValue('Cartographer');
    expect(document.getElementById('persona-temperature')).not.toBeNull();
  });

  it('ends Studio mode on Cancel and gives the conversation back', async () => {
    await openEditorInStudio();
    expect(conversationColumn()).toHaveClass('hidden');

    fireEvent.click(within(dock()).getByRole('button', { name: 'Cancel' }));

    expect(conversationColumn()).not.toHaveClass('hidden');
    expect(screen.queryByTestId('persona-editor')).toBeNull();
    expect(dock().style.width).not.toBe('');
  });

  it('ends Studio mode when the editor is left from the rail, so the next edit opens docked', async () => {
    // Killed by: frontend/src/App.tsx :: if (cloneDockMode === 'view') setCloneStudioRequested(false);
    // Becomes: if (cloneDockMode === 'view') void 0;
    await openEditorInStudio();
    expect(conversationColumn()).toHaveClass('hidden');

    fireEvent.click(await screen.findByTestId('clone-avatar-champion'));
    expect(conversationColumn()).not.toHaveClass('hidden');

    fireEvent.click(screen.getByTestId('rail-new-clone'));
    expect(screen.getByTestId('persona-editor')).toBeInTheDocument();
    expect(conversationColumn()).not.toHaveClass('hidden');
    expect(screen.getByTestId('clone-profile-editor')).toHaveAttribute('data-studio', 'false');
  });
});
