import { describe, it, expect } from 'vitest';

/**
 * The rail does not pulse and does not glow (#1061, ui-authoring §3 "Visual Restraint").
 *
 * A dot that throbs and a card that haloes are decoration on a working instrument: they draw
 * the eye to a thing that is not changing, and they leave nothing left to draw it with when
 * something is. #1061 asked for a check that fails if a pulse comes back, and the design's
 * §3.2.6 lists it among the issues this redesign absorbs. Before this there was no such check,
 * so the two `shadow-[0_0_6px_...]` halos in the kit's `StatusDot` survived every review of
 * the rail by not being looked at.
 *
 * ## Scope, and why it is this one
 *
 * This reads the files the redesign owns, and says so rather than leaving it to be inferred:
 *
 *   - everything under `ui-kit/` -- the rail, its list, its primitives. The whole kit, not
 *     only the rail's own folder: `StatusDot` and `Button` live a directory up and are what
 *     the rail draws with, so a glow re-introduced there reaches the rail (it is exactly where
 *     the two removed ones were).
 *   - `components/layout/WorkspaceSidebar.tsx` and `components/rooms/` -- this head's side of
 *     the rail: the words, the glyphs, and the props that bind them.
 *
 * Out of scope, deliberately, and each for a stated reason:
 *
 *   - `components/artifacts/ResourceSummary.tsx:294` pulses a dot. §3.2.6 places it outside
 *     this redesign by name, so it stays and this check does not read it.
 *   - `components/ui/StatusDot.tsx` is the pre-kit copy, still drawn by `Header`,
 *     `PlaygroundTab` and `ResourceSummary`. The rail reads the kit's copy instead, so
 *     flattening this one is a change to three surfaces this branch is not about.
 *   - `App.tsx`'s dock opener carries a cyan halo, which §3.2.6 likewise leaves alone.
 *
 * Widening the scope is a change to those surfaces, not to this test: add the file to
 * `SCANNED` once it has actually been quieted.
 */

/** Every script in the kit, as text. The raw glob is Vite's; `?raw` needs no `@types/node`. */
const KIT_SOURCES: Record<string, string> = import.meta.glob(
  './ui-kit/**/*.{ts,tsx}',
  { query: '?raw', import: 'default', eager: true },
);

/** This head's side of the rail: the words, the glyphs, the binding. */
const HEAD_SOURCES: Record<string, string> = import.meta.glob(
  ['./components/layout/WorkspaceSidebar.tsx', './components/rooms/*.tsx'],
  { query: '?raw', import: 'default', eager: true },
);

const SCANNED: Record<string, string> = { ...KIT_SOURCES, ...HEAD_SOURCES };

/**
 * The classes that make a surface move or glow.
 *
 * `shadow-[0_0_` is the arbitrary-value spelling of a halo -- a shadow with no offset is a
 * glow and nothing else. An ordinary `shadow-lg` is depth, not decoration, and is not matched.
 */
const RESTLESS = [
  { pattern: /animate-pulse/g, why: 'a pulsing element (#1061)' },
  { pattern: /animate-ping/g, why: 'a pinging element (#1061)' },
  { pattern: /animate-bounce/g, why: 'a bouncing element (#1061)' },
  { pattern: /shadow-\[0_0_/g, why: 'a neon glow (ui-authoring §3, Visual Restraint)' },
];

/** Every offence in `sources`, as `path: why`. Exported shape is the assertion's message. */
const restlessOffences = (sources: Record<string, string>): string[] =>
  Object.entries(sources).flatMap(([path, text]) =>
    RESTLESS.filter(({ pattern }) => new RegExp(pattern.source).test(text)).map(
      ({ why }) => `${path}: ${why}`,
    ),
  );

describe('the rail stays still (#1061)', () => {
  // Killed by: frontend/src/ui-kit/primitives/StatusDot.tsx :: success: 'bg-emerald-400',
  // Becomes: success: 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]',
  it('holds: nothing the rail draws pulses or glows', () => {
    // A glob that matched nothing would pass having read nothing. The kit has eleven modules
    // and this head binds three files to it.
    expect(Object.keys(KIT_SOURCES).length).toBeGreaterThanOrEqual(11);
    expect(Object.keys(HEAD_SOURCES).length).toBeGreaterThanOrEqual(1);

    expect(restlessOffences(SCANNED)).toEqual([]);
  });

  it('would notice each of them, and does not mistake depth for decoration', () => {
    // The check is only worth its green if it goes red on the real spellings. These are the
    // exact strings that were in the tree: `ResourceSummary`'s pulse and `StatusDot`'s halo.
    const planted = {
      './ui-kit/rail/Planted.tsx': '<StatusDot tone="success" className="animate-pulse" />',
      './ui-kit/primitives/Planted.tsx':
        "success: 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]',",
      './ui-kit/rail/Quiet.tsx': "<div className='shadow-lg rounded-md transition-colors' />",
    };

    expect(restlessOffences(planted)).toEqual([
      './ui-kit/rail/Planted.tsx: a pulsing element (#1061)',
      './ui-kit/primitives/Planted.tsx: a neon glow (ui-authoring §3, Visual Restraint)',
    ]);
  });
});
