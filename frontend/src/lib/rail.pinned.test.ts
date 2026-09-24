import { describe, it, expect, beforeEach } from 'vitest';
import {
  PINNED_CLONES_KEY,
  initialPinnedClones,
  storePinnedClones,
} from './rail';

describe('rail clone pinning persistence (#1350)', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('defaults to [defaultCloneId] when localStorage has no entry', () => {
    expect(initialPinnedClones('clone')).toEqual(['clone']);
    expect(initialPinnedClones()).toEqual(['clone']);
  });

  it('loads pinned clones from localStorage when set', () => {
    window.localStorage.setItem(PINNED_CLONES_KEY, JSON.stringify(['agent-a', 'agent-b']));
    expect(initialPinnedClones('clone')).toEqual(['agent-a', 'agent-b']);
  });

  it('stores pinned clones into localStorage', () => {
    storePinnedClones(['scout', 'critic']);
    expect(window.localStorage.getItem(PINNED_CLONES_KEY)).toBe(
      JSON.stringify(['scout', 'critic']),
    );
  });

  it('falls back to default if stored JSON is corrupt or non-array', () => {
    window.localStorage.setItem(PINNED_CLONES_KEY, 'not-json{');
    expect(initialPinnedClones('clone')).toEqual(['clone']);

    window.localStorage.setItem(PINNED_CLONES_KEY, '123');
    expect(initialPinnedClones('clone')).toEqual(['clone']);
  });
});
