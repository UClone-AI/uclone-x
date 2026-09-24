/**
 * The rule that keeps principle numbers off the screen (#1027).
 *
 * `swarm/skills/ui-authoring/SKILL.md` forbids `P6`, `P9` and their neighbours in copy a user
 * reads: they are this repository's internal grammar, and to the person who installs the
 * product and does not read its code they are noise. `../copy.test.ts` applies this to every
 * source under `src` and holds the exemptions; a fitness check in the Python suite enforces
 * the same rule over the strings `src/uclone_x/ui/app.py` puts on the screen.
 *
 * The rule lives here and not in the test file so that a `Killed by:` declaration can name one
 * of these lines: an anchor in the declaring file is quoted by its own declaration, so it never
 * occurs exactly once, which the suite's declaration checker rejects.
 *
 * **This module holds no exempt copy, deliberately.** The exemption table quotes the very lines
 * it exempts, so a module holding it would have to be skipped by the scan -- a blind spot in
 * the guard itself. Keeping the table in `copy.test.ts`, which the scan already skips along
 * with every other test, leaves the scanned set exactly what it was. Nothing imports this
 * outside the test, so it is not in the shipped bundle either.
 */

const BLOCK_COMMENT = /\/\*[\s\S]*?\*\//g;
const LINE_COMMENT = /(?<!:)\/\//;

export const PRINCIPLE_NUMBER = /\bP[0-9]\b/;
export const VITEST_FILE = /\.test\.tsx?$/;

/** An exemption is a site, not a string: each file maps to the lines allowed *in it*. */
export type Exemptions = ReadonlyMap<string, ReadonlySet<string>>;

/**
 * A source's lines with its comments removed, so only what can reach a screen is left.
 *
 * A `//` preceded by `:` is a URL, not a comment. Erring here can only make the scan read
 * less than it should, never more, so a miss is a weaker check and not a false accusation.
 */
export const copyLines = (source: string): string[] =>
  source
    .replace(BLOCK_COMMENT, '')
    .split('\n')
    .map((line) => {
      const comment = line.search(LINE_COMMENT);
      return comment === -1 ? line : line.slice(0, comment);
    });

/**
 * Whether `line` is exempt **in `path`** -- nowhere else.
 *
 * `exempt` was a bare set of lines, which made every exempt line exempt everywhere: PR #1025's
 * reviewer pasted a ledger `<option>` verbatim into `SkillsTab.tsx` and the guard passed.
 */
export const isExempt = (exempt: Exemptions, path: string, line: string): boolean =>
  exempt.get(path)?.has(line.trim()) === true;

/** Whether the scan reads `path` at all. A test renders nothing, so it is not a screen. */
export const isScanned = (path: string): boolean => !VITEST_FILE.test(path);

/**
 * Every `<path>:<line>: <text>` in `sources` a user could read a principle number on.
 *
 * Taking the sources and the exemptions as arguments is what makes the file-keying testable:
 * the live scan passes Vite's glob of the real tree, and the case in `copy.test.ts` passes two
 * files holding the same exempt line.
 */
export const principleNumberOffences = (
  sources: Record<string, string>,
  exempt: Exemptions,
): string[] =>
  Object.entries(sources)
    .filter(([path]) => isScanned(path))
    .flatMap(([path, source]) =>
      copyLines(source).flatMap((line, index) =>
        PRINCIPLE_NUMBER.test(line) && !isExempt(exempt, path, line)
          ? [`${path}:${index + 1}: ${line.trim()}`]
          : [],
      ),
    );

/** Every exempt line that its own file no longer holds -- an exemption outliving its site. */
export const staleExemptions = (sources: Record<string, string>, exempt: Exemptions): string[] =>
  [...exempt].flatMap(([path, lines]) =>
    [...lines]
      .filter((line) => !(sources[path] ?? '').includes(line))
      .map((line) => `${path}: ${line}`),
  );
