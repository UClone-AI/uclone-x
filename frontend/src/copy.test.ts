import { describe, it, expect } from 'vitest';
import {
  PRINCIPLE_NUMBER,
  copyLines,
  isScanned,
  principleNumberOffences,
  staleExemptions,
  type Exemptions,
} from './lib/copyGuard';

/**
 * Principle numbers are not user-facing copy (#1018, item 4 of PR #1017's review).
 *
 * The dock's activity timeline shipped `P6 Provenance:` and eighteen further sites carried
 * the same habit, because no check read the copy -- `App.test.tsx`'s welcome case reads the
 * one screen it renders, and a panel nobody renders in a test is unread.
 *
 * This reads the sources instead, so a panel with no test of its own is still covered. The
 * sources arrive through Vite's own raw glob rather than `node:fs`: `frontend` declares no
 * `@types/node`, and `npm run build` runs `tsc` over `src`, so a filesystem import here would
 * fail the build and with it the committed-bundle stage of the gate (#878).
 *
 * The rule itself is `lib/copyGuard.ts`. The exemptions are here, in a file the scan already
 * skips, so that quoting an exempt line does not oblige the guard to skip a second file.
 */

//: Every source under `src`, as text. Relative to this file, which sits at the root of it.
const SOURCES: Record<string, string> = import.meta.glob('./**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
});

/**
 * The one legitimate `P<digit>` on a screen: an EventBus event's priority.
 *
 * `src/uclone_x/ui/app.py` stamps the priority on the SSE envelope by envelope type --
 * `"P1"` on `SYSTEM_CONNECTED`, `"P2"` on `AGENT_EVENT`, `"P3"` on `HEARTBEAT` -- and reads
 * `AgentEvent.priority` for none of them. So the ledger's filter and badge must spell those
 * three as the Core does: they are data, not principle numbers, and nothing in the source can
 * tell the two apart. `EventPriority.CRITICAL` is the int `0` and is never serialised, so no
 * `P0` can reach the stream; the filter option and badge branch that named one were copy, and
 * are gone (#1027).
 *
 * **Keyed by file, because an exemption is a site and not a string.** A new line under an
 * existing key still needs its own reason; a copy of one under any other key is an offence.
 */
const EXEMPT_LINES: Exemptions = new Map([
  [
    './components/LedgerTab.tsx',
    new Set([
      '<option value="P1">P1 - High</option>',
      '<option value="P2">P2 - Normal</option>',
      '<option value="P3">P3 - Low</option>',
      "ev.priority === 'P1'",
    ]),
  ],
]);

describe('user-facing copy (#1018)', () => {
  it('reads the copy and not the comments', () => {
    const lines = copyLines(
      [
        'const note = 1; // P6 in a trailing comment',
        '/* P7 in a block */',
        '<span>P6 Provenance:</span>',
        "href={'https://example.test/p'}",
      ].join('\n'),
    );

    expect(lines.filter((line) => PRINCIPLE_NUMBER.test(line))).toEqual([
      '<span>P6 Provenance:</span>',
    ]);
  });

  // Killed by: frontend/src/components/artifacts/ActivityTimeline.tsx :: <span>Input Parameters</span>
  // Becomes: <span>P6 Input Parameters</span>
  it('names no principle number anywhere a user can read one', () => {
    const scanned = Object.keys(SOURCES).filter(isScanned);
    expect(scanned.length).toBeGreaterThan(20);

    expect(principleNumberOffences(SOURCES, EXEMPT_LINES)).toEqual([]);
  });

  // Killed by: frontend/src/lib/copyGuard.ts :: exempt.get(path)?.has(line.trim()) === true
  // Becomes: [...exempt.values()].some((lines) => lines.has(line.trim()))
  it('exempts a line only in the file it was exempted for (#1027)', () => {
    const exempt = '            <option value="P1">P1 - High</option>';

    expect(
      principleNumberOffences(
        {
          './components/LedgerTab.tsx': exempt,
          './components/SkillsTab.tsx': exempt,
        },
        EXEMPT_LINES,
      ),
    ).toEqual(['./components/SkillsTab.tsx:1: <option value="P1">P1 - High</option>']);
  });

  // Killed by: frontend/src/components/LedgerTab.tsx :: <option value="P2">P2 - Normal</option>
  // Becomes: <option value="P2">P2 - Standard</option>
  it('keeps no exemption a screen no longer holds (#1027)', () => {
    expect(EXEMPT_LINES.size).toBeGreaterThan(0);
    expect(staleExemptions(SOURCES, EXEMPT_LINES)).toEqual([]);
  });

  // A direct case for `staleExemptions` itself (#1052 finding 3): the two tests above pin
  // it only through the real tree, where a helper neutered to always return `[]` would
  // still read as passing. This plants a file that no longer holds its exempt line and
  // asserts the helper reports it.
  //
  // Killed by: frontend/src/lib/copyGuard.ts :: !(sources[path] ?? '').includes(line)
  // Becomes: false
  it('reports an exemption its file no longer holds', () => {
    const exempt: Exemptions = new Map([
      ['./components/Fixture.tsx', new Set(['<option value="P9">P9</option>'])],
    ]);

    expect(staleExemptions({ './components/Fixture.tsx': '<div>nothing here</div>' }, exempt)).toEqual([
      './components/Fixture.tsx: <option value="P9">P9</option>',
    ]);
  });
});
