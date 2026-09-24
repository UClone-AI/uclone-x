import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  MIN_CONVERSATION_PX,
  RAIL_OPEN_KEY,
  RAIL_OVERLAY_BELOW_PX,
  initialRailOpen,
  railOverlays,
  storeRailOpen,
} from './rail';
import { RAIL_WIDTH_PX } from '../components/layout/WorkspaceSidebar';

afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
});

describe('where the rail stops dividing the row (#1062)', () => {
  it('is the rail plus the narrowest conversation it may leave: 600px', () => {
    expect(RAIL_OVERLAY_BELOW_PX).toBe(RAIL_WIDTH_PX + MIN_CONVERSATION_PX);
    expect(RAIL_OVERLAY_BELOW_PX).toBe(600);
  });

  it('overlays a window one pixel narrower than that, and not one exactly that wide', () => {
    // The equality boundary: at 600 the conversation beside the rail is exactly 360px, which
    // is the minimum, so the rail stays in the row.
    // Killed by: frontend/src/lib/rail.ts :: windowWidth < RAIL_OVERLAY_BELOW_PX
    // Becomes: windowWidth <= RAIL_OVERLAY_BELOW_PX
    expect(railOverlays(599)).toBe(true);
    expect(railOverlays(600)).toBe(false);
  });

  it('overlays at the phone widths the issue names, and not at desktop widths', () => {
    for (const width of [320, 375, 400]) expect(railOverlays(width)).toBe(true);
    for (const width of [768, 1024, 1280]) expect(railOverlays(width)).toBe(false);
  });
});

describe('the rail on a first run', () => {
  it('starts closed at a phone width, where it would overlay', () => {
    // Killed by: frontend/src/lib/rail.ts :: return !railOverlays(windowWidth);
    // Becomes: return true;
    expect(initialRailOpen(400)).toBe(false);
  });

  it('starts open at a desktop width, where it has the room', () => {
    // Killed by: frontend/src/lib/rail.ts :: return !railOverlays(windowWidth);
    // Becomes: return false;
    expect(initialRailOpen(1280)).toBe(true);
  });

  it('decides from the width when the stored value is not a choice anyone made', () => {
    window.localStorage.setItem(RAIL_OPEN_KEY, 'maybe');
    expect(initialRailOpen(400)).toBe(false);
    expect(initialRailOpen(1280)).toBe(true);
  });

  it('decides from the width when storage refuses to be read', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('SecurityError');
    });
    expect(initialRailOpen(400)).toBe(false);
    expect(initialRailOpen(1280)).toBe(true);
  });
});

describe("the user's choice", () => {
  it('keeps a collapse made at a wide window across a reload', () => {
    // Killed by: frontend/src/lib/rail.ts :: if (stored === 'false') return false;
    // Becomes:
    storeRailOpen(false);
    expect(window.localStorage.getItem(RAIL_OPEN_KEY)).toBe('false');
    expect(initialRailOpen(1280)).toBe(false);
  });

  it('keeps an open rail opened at a narrow window across a reload', () => {
    // Killed by: frontend/src/lib/rail.ts :: if (stored === 'true') return true;
    // Becomes:
    storeRailOpen(true);
    expect(initialRailOpen(400)).toBe(true);
  });

  it('still applies to this page when storage refuses to keep it', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError');
    });
    expect(() => storeRailOpen(false)).not.toThrow();
  });
});
