import { describe, expect, it, vi, beforeEach } from 'vitest';
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ArtifactsDock, ArtifactsDockProps } from './ArtifactsDock';
import { DockSurface, RoomState } from '../../types';
import { DEVELOPER_SURFACES } from '../../lib/developerMode';

/**
 * jsdom reports a fixed window size and never resizes itself, so the width the dock reads
 * is stated by each test rather than inherited from whatever ran before it.
 */
const setWindowWidth = (px: number): void => {
  Object.defineProperty(window, 'innerWidth', { value: px, writable: true, configurable: true });
};

const baseProps = {
  activeSurface: 'artifacts' as DockSurface,
  onSelectSurface: vi.fn(),
  onClose: vi.fn(),
  // The rail closed, which is what most of these cases want: the row is the whole window.
  // Since #1042 this is not a default the component supplies — every mount site states it.
  reservedWidth: 0,
  // Off, which is the default a fresh install opens in (owner ruling 2026-09-22). The cases
  // about developer surfaces turn it on themselves.
  developerMode: false,
  // The conversation on screen and the seat in it; every dock read is scoped to these (#1356).
  roomId: 'room-a' as string | null,
  roomPending: null as string | null,
  seatId: 'scout' as string | null,
  selectedArtifactPath: null,
  events: [],
  ontology: null,
  budgetData: null,
  isPaused: false,
  onTogglePause: vi.fn(),
  onClearEvents: vi.fn(),
  lastHeartbeat: '-',
  // No clone picked, which is the state a fresh install opens in.
  selectedClone: '',
  personas: [],
  onStartConversation: vi.fn(),
  onRefresh: vi.fn(),
  isRefreshing: false,
};

describe('ArtifactsDock', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    setWindowWidth(1024);
    // Default fetch mock
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: RequestInfo | URL) => {
      const urlStr = String(url);
      if (urlStr === '/api/rooms/room-a/artifacts') {
        return {
          ok: true,
          json: async () => ({
            room_id: 'room-a',
            artifacts: [
              {
                id: 'docs/test_artifact.md',
                name: 'test_artifact.md',
                path: 'docs/test_artifact.md',
                title: 'test_artifact.md',
                type: 'document',
                participant_id: 'scout',
                tool_name: 'write_file',
                turn_id: 't1',
                tool_call_id: 'c1',
                written_at: '2026-09-22T10:00:00Z',
                write_count: 1,
                writers: ['scout'],
                exists: true,
                size_bytes: 1024,
              },
            ],
            total: 1,
            unattributed_writes: 0,
            unattributed_note: null,
            reason: null,
          }),
        } as Response;
      }
      if (urlStr.includes('/api/artifacts/content?')) {
        return {
          ok: true,
          text: async () => '# Hello Artifact\nThis is a rendered document.',
        } as Response;
      }
      return { ok: false, status: 404 } as Response;
    });
  });

  it('renders the six user surface tabs in the primary row, Remembers among them', () => {
    render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('tab-clone')).toBeDefined();
    expect(screen.getByTestId('tab-turn')).toBeDefined();
    expect(screen.getByTestId('tab-artifacts')).toBeDefined();
    expect(screen.getByTestId('tab-remembers')).toHaveTextContent('Remembers');
    expect(screen.getByTestId('tab-activity')).toBeDefined();
    expect(screen.getByTestId('tab-resource')).toBeDefined();
  });

  // Six tabs do not fit a 520px dock on one line. In a row that scrolls, Activity and
  // Resource sat past the edge with nothing saying they were there (seen in the browser).
  // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: className="flex flex-wrap items-center gap-1 text-xs min-w-0"
  // Becomes: className="flex items-center gap-1 text-xs overflow-x-auto"
  it('wraps the primary row rather than scrolling surfaces out of sight', () => {
    render(<ArtifactsDock {...baseProps} />);
    const row = screen.getByTestId('dock-primary-tabs');
    expect(row.className).toMatch(/\bflex-wrap\b/);
    expect(row.className).not.toMatch(/overflow-x-(auto|scroll)/);
  });

  it('keeps every developer surface out of the primary row, in developer mode too', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: {primaryTabs.map((tab) => (
    // Becomes: {[...primaryTabs, ...devTabs].map((tab) => (
    render(<ArtifactsDock {...baseProps} developerMode />);
    // The drawer is there, so the absences below are the row's and not a missing dock's.
    expect(screen.getByTestId('dev-drawer')).toBeInTheDocument();
    for (const surface of DEVELOPER_SURFACES) {
      expect(screen.queryByTestId(`tab-${surface}`)).toBeNull();
    }
  });

  it('offers no developer tab and no drawer while developer mode is off', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: {developerMode && (
    // Becomes: {true && (
    render(<ArtifactsDock {...baseProps} />);
    // The dock rendered its row, so what is missing below is missing from a live dock.
    expect(screen.getByTestId('tab-artifacts')).toBeInTheDocument();
    expect(screen.queryByTestId('dev-drawer')).toBeNull();
    expect(screen.queryByText(/Developer Tools/i)).toBeNull();
    for (const surface of DEVELOPER_SURFACES) {
      expect(screen.queryByTestId(`dev-tab-${surface}`)).toBeNull();
      expect(screen.queryByTestId(`tab-${surface}`)).toBeNull();
    }
  });

  it('offers the four developer surfaces in the drawer while developer mode is on', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: { id: 'ontology', label: 'Ontology', icon: <Brain className="w-3.5 h-3.5" />, accent: 'violet' },
    // Becomes:
    render(<ArtifactsDock {...baseProps} developerMode />);
    const drawer = screen.getByTestId('dev-drawer');
    expect(drawer).toHaveTextContent('Developer Tools');
    const labels: Record<string, string> = {
      knowledge_graph: 'Knowledge Graph',
      topology: 'DAG',
      ledger: 'EventBus',
      ontology: 'Ontology',
    };
    expect(DEVELOPER_SURFACES).toHaveLength(4);
    for (const surface of DEVELOPER_SURFACES) {
      expect(screen.getByTestId(`dev-tab-${surface}`)).toHaveTextContent(labels[surface]);
    }
    // Skills moved to Settings, and ACP and Evals to Settings' Diagnostics (#1358): the
    // drawer names none of them, by tab or by label.
    for (const gone of ['skills', 'acp', 'evaluations']) {
      expect(screen.queryByTestId(`dev-tab-${gone}`)).toBeNull();
    }
    for (const label of ['Skills', 'ACP', 'Evals']) {
      expect(drawer).not.toHaveTextContent(label);
    }

    fireEvent.click(screen.getByTestId('dev-tab-ledger'));
    expect(baseProps.onSelectSurface).toHaveBeenCalledWith('ledger');
  });

  it('shows Docs & Artifacts when the selected surface is a developer one and the mode is off', () => {
    // Killed by: frontend/src/lib/developerMode.ts :: ? 'artifacts' : active
    // Becomes: ? active : active
    const off = render(<ArtifactsDock {...baseProps} activeSurface="knowledge_graph" />);
    expect(screen.getByTestId('doc-viewer')).toBeInTheDocument();
    expect(screen.queryByTestId('knowledge-graph-viewer')).toBeNull();
    // Docs & Artifacts is marked as the one on screen, not the surface the user cannot see.
    expect(screen.getByTestId('tab-artifacts').className).toContain('font-semibold');
    off.unmount();

    // The selection is kept: the mode back on returns to it.
    render(<ArtifactsDock {...baseProps} activeSurface="knowledge_graph" developerMode />);
    expect(screen.getByTestId('knowledge-graph-viewer')).toBeInTheDocument();
    expect(screen.queryByTestId('doc-viewer')).toBeNull();
  });

  it('renders DocViewer on artifacts surface and displays fetched content', async () => {
    render(<ArtifactsDock {...baseProps} activeSurface="artifacts" />);
    expect(screen.getByTestId('doc-viewer')).toBeDefined();
    await waitFor(() => {
      expect(screen.getByText('Hello Artifact')).toBeDefined();
    });
  });

  it('toggles raw markdown view in DocViewer', async () => {
    render(<ArtifactsDock {...baseProps} activeSurface="artifacts" />);
    await waitFor(() => {
      expect(screen.getByText('Hello Artifact')).toBeDefined();
    });

    const toggleBtn = screen.getByTestId('toggle-raw-btn');
    fireEvent.click(toggleBtn);
    await waitFor(() => {
      const pre = screen.getByTestId('raw-markdown-pre');
      expect(pre).toBeDefined();
      expect(pre.textContent).toContain('# Hello Artifact');
    });
  });

  it('reports surface selection changes', () => {
    render(<ArtifactsDock {...baseProps} />);
    fireEvent.click(screen.getByTestId('tab-activity'));
    expect(baseProps.onSelectSurface).toHaveBeenCalledWith('activity');
  });

  it('closes on request via close button', () => {
    render(<ArtifactsDock {...baseProps} />);
    fireEvent.click(screen.getByTestId('dock-close'));
    expect(baseProps.onClose).toHaveBeenCalled();
  });

  it('restores persisted width and clamps appropriately', () => {
    window.localStorage.setItem('uclone-x.dock.width', '600');
    const { unmount } = render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('600px');
    unmount();

    window.localStorage.setItem('uclone-x.dock.width', '100');
    render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('360px');
  });

  it('falls back to the default width when storage is unreadable', () => {
    const getItem = vi
      .spyOn(Storage.prototype, 'getItem')
      .mockImplementation(() => {
        throw new Error('storage blocked');
      });
    render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('520px');
    getItem.mockRestore();
  });

  it('offers no Budget or Tool Detail surface: Resource and Activity absorbed them', () => {
    // Checked by hand: restoring either retired tab to `primaryTabs` or `devTabs` fails this
    // case. Not a Killed-by declaration, because the line such a mutation adds is absent.
    render(<ArtifactsDock {...baseProps} developerMode />);
    // The drawer is shown and the absorbing surfaces are still offered, so the absences below
    // are the tabs' and not a missing drawer's.
    expect(screen.getByTestId('dev-tab-ledger')).toBeDefined();
    expect(screen.getByTestId('tab-resource')).toBeDefined();
    expect(screen.getByTestId('tab-activity')).toBeDefined();
    expect(screen.queryByTestId('tab-budget')).toBeNull();
    expect(screen.queryByTestId('dev-tab-budget')).toBeNull();
    expect(screen.queryByTestId('dev-tab-tool')).toBeNull();
    expect(screen.queryByText('Tool Detail')).toBeNull();
  });

  it('renders ActivityTimeline on activity surface', () => {
    render(<ArtifactsDock {...baseProps} activeSurface="activity" />);
    expect(screen.getByTestId('activity-timeline')).toBeDefined();
  });

  it('renders ResourceSummary on resource surface', () => {
    render(<ArtifactsDock {...baseProps} activeSurface="resource" />);
    expect(screen.getByTestId('resource-summary')).toBeDefined();
    expect(screen.getByText('Resource & Budget Summary')).toBeDefined();
  });

  it('never renders wider than the window, and hands a wide one the dragged width back', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: overlaid ? windowWidth : width
    // Becomes: width
    window.localStorage.setItem('uclone-x.dock.width', '520');

    setWindowWidth(400);
    const narrow = render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('400px');
    narrow.unmount();

    // The cap is a rendering decision and not a write: the dragged width survives it.
    setWindowWidth(1280);
    render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('520px');
  });

  it('is the container its panels lay themselves out against, seated or overlaid (#1029)', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: } dock-scope bg-slate-950/95
    // Becomes: } bg-slate-950/95
    // `.dock-scope` (index.css) is the query container the panels' grids answer to: without
    // it on the dock, a grid in a 360px dock takes the *viewport's* 3-4 columns.
    setWindowWidth(1280);
    const seated = render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').classList.contains('dock-scope')).toBe(true);
    seated.unmount();

    setWindowWidth(400);
    render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').classList.contains('dock-scope')).toBe(true);
  });

  it('leaves the workspace row only while the window is no wider than the dock', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: windowWidth - reservedWidth <= width
    // Becomes: false
    window.localStorage.setItem('uclone-x.dock.width', '520');

    setWindowWidth(400);
    const narrow = render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').classList.contains('absolute')).toBe(true);
    narrow.unmount();

    setWindowWidth(1280);
    render(<ArtifactsDock {...baseProps} />);
    const dock = screen.getByTestId('artifacts-dock');
    expect(dock.classList.contains('absolute')).toBe(false);
    expect(dock.classList.contains('relative')).toBe(true);
  });

  const dragHandleTo = (clientX: number): void => {
    fireEvent.mouseDown(screen.getByTestId('dock-resize-handle'));
    fireEvent.mouseMove(window, { clientX });
    fireEvent.mouseUp(window);
  };

  it('a drag cannot put the dock into a row that cannot seat it', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: windowWidth - reservedWidth <= width
    // Becomes: windowWidth <= width
    window.localStorage.setItem('uclone-x.dock.width', '520');
    setWindowWidth(400);
    // The rail is open, so 240 of the 400 is not the dock's to take. No width the drag can
    // reach fits what is left -- MIN_WIDTH alone is 360 -- which is why the window on its own
    // was never the quantity that decided this.
    render(<ArtifactsDock {...baseProps} reservedWidth={240} />);
    const dock = screen.getByTestId('artifacts-dock');
    expect(dock.style.width).toBe('400px');

    // 400 - 40 is below MIN_WIDTH, so the drag asks for 360. Taking the overlay off here put
    // the dock in a row 600px wide inside a 400px window, with the close button past its edge.
    dragHandleTo(40);

    expect(dock.style.width).toBe('400px');
    expect(dock.classList.contains('absolute')).toBe(true);
    // The gesture is still honoured -- it is the *layout* that refuses, not the control.
    expect(window.localStorage.getItem('uclone-x.dock.width')).toBe('360');
  });

  it('can always be dragged back out of the overlay it was dragged into', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: setWidth(next);
    // Becomes: setWidth((current) => (window.innerWidth <= current ? current : next));
    window.localStorage.setItem('uclone-x.dock.width', '520');
    setWindowWidth(768);
    render(<ArtifactsDock {...baseProps} reservedWidth={240} />);
    const dock = screen.getByTestId('artifacts-dock');
    expect(dock.classList.contains('relative')).toBe(true);

    // Drag 1: to the window's left edge, which asks for the whole 768 and overlays the dock.
    dragHandleTo(0);
    expect(dock.style.width).toBe('768px');
    expect(dock.classList.contains('absolute')).toBe(true);

    // Drag 2: back to MIN_WIDTH. 360 fits beside a 240px rail in 768, so the dock returns to
    // the row. Freezing the width while overlaid made this second drag inert *permanently* at
    // every window up to MAX_WIDTH -- through further drags, a reload, and a close and reopen.
    dragHandleTo(768 - 360);
    expect(dock.style.width).toBe('360px');
    expect(dock.classList.contains('relative')).toBe(true);
    expect(window.localStorage.getItem('uclone-x.dock.width')).toBe('360');
  });

  it('gives the stored width back when the window widens under one open dock', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: useState<number>(readStoredWidth)
    // Becomes: useState<number>(() => Math.min(readStoredWidth(), window.innerWidth))
    window.localStorage.setItem('uclone-x.dock.width', '520');
    setWindowWidth(400);
    render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('400px');
    expect(window.localStorage.getItem('uclone-x.dock.width')).toBe('520');

    // One mount, narrowed and widened. Mounting fresh at a wide window cannot tell a cap that
    // only *renders* narrow from one that captured the narrow width as the dock's own.
    act(() => {
      setWindowWidth(1280);
      window.dispatchEvent(new Event('resize'));
    });

    expect(screen.getByTestId('artifacts-dock').style.width).toBe('520px');
  });

  it('overlays a window exactly its own width, and not one a pixel wider', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: windowWidth - reservedWidth <= width
    // Becomes: windowWidth - reservedWidth < width
    window.localStorage.setItem('uclone-x.dock.width', '520');

    setWindowWidth(520);
    const equal = render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').classList.contains('absolute')).toBe(true);
    equal.unmount();

    setWindowWidth(521);
    render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').classList.contains('relative')).toBe(true);
  });

  it('overlays a 760px window the open rail leaves exactly its own width of', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: windowWidth - reservedWidth <= width
    // Becomes: windowWidth - reservedWidth * 0.9 <= width
    //
    // The rail-open equality boundary, and the only case here that pins it. Write R for what
    // the predicate effectively subtracts when the prop is 240: every other rail-open case is
    // an *interior* point, so together they pin R only to an interval — 700/240/520 needs
    // R >= 180, 768/240/520 needs R < 248, 400/240/360 needs R >= 40. That 68px band is wide
    // enough for `reservedWidth * 0.9` (R = 216) and `Math.max(0, reservedWidth - 1)`
    // (R = 239) to walk through it, and both escaped the whole suite before this case.
    // 760 requires R >= 240 and closes the band's lower 60px: 760 - 240 is exactly the dock's
    // 520, and the threshold is `<=`, so it overlays. The only other equality boundary in this
    // file is at rail 0 (`overlays a window exactly its own width, and not one a pixel wider`),
    // which any mutation that vanishes at 0 passes untouched.
    //
    // This is the width `docs/ui-dashboard-architecture.md` §3 records as the one that changed
    // side when the row's arithmetic replaced the window-only rule.
    window.localStorage.setItem('uclone-x.dock.width', '520');

    setWindowWidth(760);
    render(<ArtifactsDock {...baseProps} reservedWidth={240} />);

    const dock = screen.getByTestId('artifacts-dock');
    expect(dock.style.width).toBe('760px');
    expect(dock.classList.contains('absolute')).toBe(true);
  });

  it('seats a 761px window the open rail leaves one pixel more than the dock of', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: windowWidth - reservedWidth <= width
    // Becomes: windowWidth - (reservedWidth && 241) <= width
    //
    // The declared mutation vanishes at rail 0 on purpose, so only a rail-open case can catch
    // it. The plainer `reservedWidth + 1` would not: it shifts the rail-*closed* boundary too,
    // and `overlays a window exactly its own width, and not one a pixel wider` already kills
    // it (measured: that case goes red under it), so declaring it here would evidence nothing
    // this file did not already have.
    //
    // The rail-open twin of `overlays a window exactly its own width, and not one a pixel
    // wider`, and the mirror of the case above. Write R for what the predicate effectively
    // subtracts when the prop is 240: the 760 case needs R >= 240, but every case bounding R
    // from above was an *interior* point -- the widest constraint, 768/240/520, needs only
    // R < 248 -- so R was pinned to [240, 248) and a rail over-reserved by 1 to 7px walked
    // through it. Measured against the file as shipped without this case, injecting an exact
    // effective subtraction that still vanishes at rail 0: R = 241 and R = 247 each escaped
    // the whole suite, 315/315; only R = 248 died.
    //
    // 761 - 240 = 521, one pixel more than the dock's stored 520, so the row seats it and the
    // dock stays `relative` at its own width. One more pixel of rail and it would not: at
    // R = 241 the remainder is exactly 520 and the threshold is `<=`, so this case goes red.
    // Together with the 760 case above it, R is pinned to [240, 241) -- 239 overlays a window
    // that must not overlay, 241 seats none in a row that must seat this one -- so 240 is the
    // only integer left, and it is the one the sole mount site passes while the rail is open
    // (`App.tsx` passes `isSidebarOpen ? RAIL_WIDTH_PX : 0`; with the rail closed it passes 0,
    // where the subtraction vanishes and the rail-closed boundary case takes over).
    window.localStorage.setItem('uclone-x.dock.width', '520');

    setWindowWidth(761);
    render(<ArtifactsDock {...baseProps} reservedWidth={240} />);

    const dock = screen.getByTestId('artifacts-dock');
    expect(dock.style.width).toBe('520px');
    expect(dock.classList.contains('relative')).toBe(true);
    expect(dock.classList.contains('absolute')).toBe(false);
  });

  it('follows a window resized under an already open dock', () => {
    // Killed by: frontend/src/lib/rail.ts :: window.addEventListener('resize', handleResize)
    // Becomes:
    window.localStorage.setItem('uclone-x.dock.width', '520');
    setWindowWidth(1280);
    render(<ArtifactsDock {...baseProps} />);
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('520px');

    act(() => {
      setWindowWidth(320);
      window.dispatchEvent(new Event('resize'));
    });

    expect(screen.getByTestId('artifacts-dock').style.width).toBe('320px');
  });

  it('re-seats in the row when the rail closes under an already open dock', () => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: windowWidth - reservedWidth <= width
    // Becomes: windowWidth - useRef(reservedWidth).current <= width
    //
    // Every other case here states one rail width and never changes it, so reading the rail
    // once at mount is indistinguishable from reading it every render — which is how that
    // mutation escaped both suites (#1042 D1, the reviewer's MINE-A on #1032). It is not an
    // inert mutation: with it applied, closing the rail at this window leaves the dock
    // overlaid at `x 0..700` for the rest of the session instead of returning to `x 180..700`.
    // 700px is chosen because it is a width where the two rail states disagree — at 1280 both
    // seat the dock in the row, at 320 neither does, and the mutation survives either.
    window.localStorage.setItem('uclone-x.dock.width', '520');
    setWindowWidth(700);
    const { rerender } = render(<ArtifactsDock {...baseProps} reservedWidth={240} />);

    // 700 - 240 = 460, and nothing the row has left can seat a 520px dock, so it overlays.
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('700px');
    expect(screen.getByTestId('artifacts-dock').classList.contains('absolute')).toBe(true);

    // The rail closes. Same window, same dock, same stored width: the only thing that moved is
    // the 240px the rail was holding, which is now the row's. `x 180..700`.
    rerender(<ArtifactsDock {...baseProps} reservedWidth={0} />);
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('520px');
    expect(screen.getByTestId('artifacts-dock').classList.contains('absolute')).toBe(false);
    expect(screen.getByTestId('artifacts-dock').classList.contains('relative')).toBe(true);

    // And back, so the assertion above cannot pass on a dock that simply never overlays.
    rerender(<ArtifactsDock {...baseProps} reservedWidth={240} />);
    expect(screen.getByTestId('artifacts-dock').style.width).toBe('700px');
    expect(screen.getByTestId('artifacts-dock').classList.contains('absolute')).toBe(true);
  });

  it('will not type a mount site that omits the rail width', () => {
    // No parsing `Killed by:` declaration here, deliberately (#1246). The mutation is
    // `reservedWidth: number;` -> `reservedWidth?: number;` in
    // `frontend/src/components/layout/ArtifactsDock.tsx`, and it is checked by `tsc`, not by
    // the assertions below: types are erased before vitest
    // sees this file, so the run below passes either way. Restoring the `?` leaves the
    // directive below with nothing to suppress, and `tsc` reports
    // `Unused '@ts-expect-error' directive`. `tsc` runs in `npm run build` — the ui-authoring
    // rebuild command, and gate stage 7 under `-fe` — and not on the default `./ucx test check`
    // path, whose stage 6b runs `vite build` alone (measured: that command exits 0 with the
    // prop omitted at the mount site). That is the point of #1042 D2: while the prop was
    // optional and defaulted to 0, a mount site that forgot it silently restored the
    // pre-#1030 window-only rule, with no type error and no test going red.
    const mount = (props: ArtifactsDockProps): ArtifactsDockProps => props;
    const { reservedWidth, ...railWidthForgotten } = baseProps;
    expect(mount({ ...railWidthForgotten, reservedWidth }).reservedWidth).toBe(0);

    // @ts-expect-error - `reservedWidth` is required, so omitting it does not compile.
    const forgetful = mount({ ...railWidthForgotten });
    expect(forgetful.reservedWidth).toBeUndefined();
  });
});

const twoSeats = (roomId: string): RoomState => ({
  room_id: roomId,
  title: 't',
  participants: [
    { id: 'user', kind: 'human', display_name: 'You' },
    { id: 'scout', kind: 'agent', display_name: 'Scout' },
    { id: 'critic', kind: 'agent', display_name: 'Critic' },
  ],
  transcript: [],
  turn_state: { agent_turns_since_human: 0 },
  policy: {
    max_agent_turns_per_human_message: 4,
    max_span_messages: 8,
    transcript_window: 40,
    hesitation_seconds: 0,
    default_responder_id: 'scout',
  },
});

describe('ArtifactsDock: scoped to the conversation on screen (#1356)', () => {
  let requested: string[];
  beforeEach(() => {
    requested = [];
    window.localStorage.clear();
    setWindowWidth(1024);
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: RequestInfo | URL) => {
      requested.push(String(url));
      return { ok: false, status: 404, statusText: 'Not Found', json: async () => ({}) } as Response;
    });
  });

  it('never reads a session: Activity and Docs ask for the room on screen', async () => {
    const { rerender } = render(<ArtifactsDock {...baseProps} activeSurface="activity" />);
    await waitFor(() => expect(requested).toContain('/api/rooms/room-a/seats/scout/history'));
    rerender(<ArtifactsDock {...baseProps} activeSurface="artifacts" />);
    await waitFor(() => expect(requested).toContain('/api/rooms/room-a/artifacts'));
    expect(requested.some((u) => /session/.test(u))).toBe(false);
  });

  it('re-scopes to another conversation without a reload', async () => {
    const { rerender } = render(<ArtifactsDock {...baseProps} activeSurface="artifacts" />);
    await waitFor(() => expect(requested).toContain('/api/rooms/room-a/artifacts'));
    rerender(<ArtifactsDock {...baseProps} roomId="room-b" activeSurface="artifacts" />);
    await waitFor(() => expect(requested).toContain('/api/rooms/room-b/artifacts'));
  });

  it('reads what the seat remembers on the Remembers surface', async () => {
    render(<ArtifactsDock {...baseProps} activeSurface="remembers" />);
    expect(screen.getByTestId('remembers-panel')).toBeInTheDocument();
    await waitFor(() =>
      expect(requested).toContain('/api/rooms/room-a/knowledge?agent_id=scout'),
    );
  });

  // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: {seatScoped && seats.length > 1 && (
  // Becomes: {seatScoped && seats.length > 0 && (
  it('offers a seat picker only when there is more than one clone to choose', () => {
    const one: RoomState = { ...twoSeats('room-a'), participants: twoSeats('room-a').participants.slice(0, 2) };
    const { rerender } = render(
      <ArtifactsDock {...baseProps} activeSurface="activity" room={one} />,
    );
    expect(screen.queryByTestId('dock-seat-picker')).toBeNull();

    const onSelectSeat = vi.fn();
    rerender(
      <ArtifactsDock
        {...baseProps}
        activeSurface="activity"
        room={twoSeats('room-a')}
        onSelectSeat={onSelectSeat}
      />,
    );
    const picker = screen.getByTestId('dock-seat-picker') as HTMLSelectElement;
    expect(Array.from(picker.options).map((o) => o.textContent)).toEqual(['Scout', 'Critic']);
    fireEvent.change(picker, { target: { value: 'critic' } });
    expect(onSelectSeat).toHaveBeenCalledWith('critic');

    // Docs is the room's, not a seat's: no picker there.
    rerender(<ArtifactsDock {...baseProps} activeSurface="artifacts" room={twoSeats('room-a')} />);
    expect(screen.queryByTestId('dock-seat-picker')).toBeNull();
  });
});

/**
 * A conversation that is open but not yet on screen (#1389).
 *
 * `roomId` is `null` in two different situations: nothing is open, or the open one's record
 * has not landed (or could not be read). Every conversation-scoped surface used to say "No
 * conversation is open" for both; the dock now says which.
 */
describe('ArtifactsDock: a conversation still opening, or unreadable (#1389)', () => {
  it.each([
    'artifacts',
    'activity',
    'remembers',
    'resource',
    'topology',
    'knowledge_graph',
  ])('%s says the conversation is opening, not that none is open', (surface) => {
    // Killed by: frontend/src/components/layout/ArtifactsDock.tsx :: const pending = roomId === null && roomScoped ? roomPending : null;
    // Becomes: const pending = null as string | null;
    render(
      <ArtifactsDock
        {...baseProps}
        developerMode
        activeSurface={surface as DockSurface}
        roomId={null}
        roomPending="Opening this conversation…"
      />,
    );
    expect(screen.getByTestId('dock-room-pending')).toHaveTextContent('Opening this conversation…');
    expect(screen.getByTestId('artifacts-dock')).not.toHaveTextContent('No conversation is open');
  });

  it.each([
    'artifacts',
    'activity',
    'remembers',
    'resource',
    'topology',
    'knowledge_graph',
  ])('%s still says none is open when none is', (surface) => {
    render(
      <ArtifactsDock
        {...baseProps}
        developerMode
        activeSurface={surface as DockSurface}
        roomId={null}
        roomPending={null}
      />,
    );
    expect(screen.queryByTestId('dock-room-pending')).toBeNull();
    expect(screen.getByTestId('artifacts-dock')).toHaveTextContent('No conversation is open');
  });

  it('leaves the surfaces that do not read the conversation alone', () => {
    render(
      <ArtifactsDock
        {...baseProps}
        activeSurface="clone"
        roomId={null}
        roomPending="Opening this conversation…"
      />,
    );
    expect(screen.queryByTestId('dock-room-pending')).toBeNull();
  });
});
