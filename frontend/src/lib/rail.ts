import { useEffect, useState } from 'react';
import { RAIL_WIDTH_PX } from '../components/layout/WorkspaceSidebar';

/**
 * The narrowest conversation the rail may leave beside itself: the dock's own minimum width,
 * which is also the narrowest phone width (320–400px) the conversation is laid out for with the
 * rail closed.
 *
 * Below it the rail stops taking a share of the workspace row and is drawn *over* the
 * conversation instead (#1062). At a 400px window it used to take 240 of the 400 and leave the
 * conversation -- U0's default state, the one surface with no condition on it -- 160px.
 */
export const MIN_CONVERSATION_PX = 360;

/**
 * The window width below which the rail overlays rather than divides the row: 600px.
 *
 * The rail is the row's leftmost region and its parent spans the window, so the window is the
 * quantity the rail is sized against, the same argument `useWindowWidth` makes for the dock.
 *
 * **The case this rule does not cover.** It reads the window and the rail, not the dock. With
 * the dock seated in the row the conversation gets the window minus the rail minus the dock,
 * and the dock seats itself whenever that is more than 0 (`ArtifactsDock`'s `overlaid`). So at
 * 761px with the rail open and a 520px dock the conversation is 1px wide, and the rail stays in
 * the row because the window is wider than 600. The rail does not give way to the dock; the two
 * rules each keep their own region inside the window, and neither guarantees the conversation a
 * width once both are open. That was true before #1062 and is unchanged by it.
 */
export const RAIL_OVERLAY_BELOW_PX = RAIL_WIDTH_PX + MIN_CONVERSATION_PX;

/** Whether, at this window width, the open rail is drawn over the conversation. */
export const railOverlays = (windowWidth: number): boolean => windowWidth < RAIL_OVERLAY_BELOW_PX;

/**
 * Where the user's own open/closed choice is kept.
 *
 * In the browser and not in the Core: collapse state is the screen's, and a second head
 * attached to the same Core must not inherit this one's (ui-authoring §3).
 */
export const RAIL_OPEN_KEY = 'uclone-x.rail.open';

/**
 * Whether the rail starts open: the user's last choice, or on a first run whether the window
 * can seat the rail beside the conversation.
 *
 * A first run at a phone width therefore starts with the rail closed, and a first run at a
 * desktop width with it open. A stored value that is neither `'true'` nor `'false'` is not a
 * choice anyone made, so it falls back to the width as well.
 */
export const initialRailOpen = (windowWidth: number): boolean => {
  try {
    const stored = window.localStorage.getItem(RAIL_OPEN_KEY);
    if (stored === 'true') return true;
    if (stored === 'false') return false;
  } catch {
    /* Storage refused (private mode, a blocked origin): decide from the width. */
  }
  return !railOverlays(windowWidth);
};

/** Keep the user's choice for the next load. Only an explicit toggle calls this. */
export const storeRailOpen = (open: boolean): void => {
  try {
    window.localStorage.setItem(RAIL_OPEN_KEY, String(open));
  } catch {
    /* A choice that cannot be kept still applies to this page. */
  }
};

/**
 * The window's width, as state, so a region sized against it re-renders on resize.
 *
 * Shared by the rail (here) and the dock (`ArtifactsDock`), which are the workspace row's two
 * side regions; both are sized against the window because their parent spans it.
 */
export const useWindowWidth = (): number => {
  const [windowWidth, setWindowWidth] = useState<number>(() => window.innerWidth);
  useEffect(() => {
    const handleResize = (): void => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  return windowWidth;
};

/**
 * Where the user's pinned clones are kept in the browser.
 */
export const PINNED_CLONES_KEY = 'uclone-x.rail.pinned_clones';

/**
 * Load pinned clone IDs from localStorage.
 * If unset, defaults to [defaultCloneId] (e.g. ['clone']) if provided.
 */
export const initialPinnedClones = (defaultCloneId = 'clone'): string[] => {
  try {
    const raw = window.localStorage.getItem(PINNED_CLONES_KEY);
    if (raw !== null) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed.filter((id): id is string => typeof id === 'string');
      }
    }
  } catch {
    /* Storage unavailable or malformed JSON */
  }
  return defaultCloneId ? [defaultCloneId] : [];
};

/** Keep the user's pinned clones for subsequent visits. */
export const storePinnedClones = (pinnedIds: string[]): void => {
  try {
    window.localStorage.setItem(PINNED_CLONES_KEY, JSON.stringify(pinnedIds));
  } catch {
    /* Storage refused */
  }
};

/**
 * Where rooms created explicitly as group chats are recorded in the browser.
 *
 * Keeps group chats in the top Group Chats section even while they have 0 or 1 agent seated,
 * so inviting a first clone does not prematurely shift the room into a 1:1 clone session.
 */
export const GROUP_ROOMS_KEY = 'uclone-x.rail.group_rooms';

/** Load group room IDs from localStorage. */
export const initialGroupRooms = (): string[] => {
  try {
    const raw = window.localStorage.getItem(GROUP_ROOMS_KEY);
    if (raw !== null) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed.filter((id): id is string => typeof id === 'string');
      }
    }
  } catch {
    /* Storage unavailable or malformed JSON */
  }
  return [];
};

/** Keep the group room IDs for subsequent visits. */
export const storeGroupRooms = (roomIds: string[]): void => {
  try {
    window.localStorage.setItem(GROUP_ROOMS_KEY, JSON.stringify(roomIds));
  } catch {
    /* Storage refused */
  }
};
