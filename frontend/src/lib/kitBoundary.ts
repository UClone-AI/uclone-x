/**
 * The rule that keeps `ui-kit/` portable (#1063 D, #1158).
 *
 * The kit is meant to be lifted into another head (uclone2 first) as it stands. It can be only
 * if nothing in it reaches back into this one: no store, no API client, no copy module, and no
 * icon library -- this repo is on lucide-react 1.x and uclone2 on 0.378, so even a shared
 * dependency is not shared. So the rule is an allow-list, not a deny-list: a file under
 * `ui-kit/` may import `react`, and files that are themselves under `ui-kit/`. Nothing else.
 *
 * `../ui-kit.test.ts` applies it to every source under `ui-kit/`. The test sits outside the kit
 * on purpose: a test inside it would import `vitest`, which is exactly what the rule refuses.
 *
 * The rule lives here and not in the test file for the reason `copyGuard.ts` gives: a
 * `Killed by:` declaration must be able to name one of these lines exactly once.
 */

/** The kit's root, as Vite's glob keys paths relative to `src`. */
export const KIT_ROOT = './ui-kit/';

/** The one package the kit may import. Exact: `react-dom` or `react/jsx-runtime` is not it. */
export const KIT_ALLOWED_PACKAGES: ReadonlySet<string> = new Set(['react']);

const BLOCK_COMMENT = /\/\*[\s\S]*?\*\//g;
const LINE_COMMENT = /(?<!:)\/\/.*$/gm;

/**
 * Every way a module can name another. `from '...'` covers `import ... from` and
 * `export ... from`, over as many lines as the braces take.
 */
const SPECIFIER_FORMS: readonly RegExp[] = [
  /\bfrom\s*['"]([^'"]+)['"]/g,
  /\bimport\s*['"]([^'"]+)['"]/g,
  /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
  /\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
];

/**
 * Forms whose target cannot be read from the source, so cannot be shown to be inside the kit:
 * a dynamic `import()` of anything but a string literal, and `import.meta` (whose `glob` reads
 * any file in the tree). Each is refused outright rather than guessed at.
 */
const UNREADABLE_FORMS: readonly RegExp[] = [/\bimport\s*\(\s*(?!['"])/, /\bimport\.meta\b/];

/** `source` with its comments removed, so a comment that mentions an import is not one. */
export const withoutComments = (source: string): string =>
  source.replace(BLOCK_COMMENT, '').replace(LINE_COMMENT, '');

/** Every module specifier `source` names, in the order the forms find them. */
export const importSpecifiers = (source: string): string[] => {
  const code = withoutComments(source);
  return SPECIFIER_FORMS.flatMap((form) => [...code.matchAll(form)].map((match) => match[1]));
};

/** `spec`, relative to the file at `path`, as a normalised `./`-rooted path. */
export const resolveRelative = (path: string, spec: string): string => {
  const parts = path.split('/').slice(0, -1);
  for (const segment of spec.split('/')) {
    if (segment === '..') parts.pop();
    else if (segment !== '.' && segment !== '') parts.push(segment);
  }
  return `${parts.join('/')}/`.replace(/^(?!\.\/)/, './');
};

/** Whether the file at `path` may import `spec`. */
export const isAllowedImport = (path: string, spec: string): boolean => {
  if (KIT_ALLOWED_PACKAGES.has(spec)) return true;
  if (!spec.startsWith('.')) return false;
  return resolveRelative(path, spec).startsWith(KIT_ROOT);
};

/**
 * Every `<path>: <specifier>` under `ui-kit/` that reaches outside it.
 *
 * Taking the sources as an argument is what lets the rule be tested on planted files: the live
 * scan passes Vite's glob of the real kit, and the cases in `ui-kit.test.ts` pass fixtures.
 */
export const kitBoundaryOffences = (sources: Record<string, string>): string[] =>
  Object.entries(sources)
    .filter(([path]) => path.startsWith(KIT_ROOT))
    .flatMap(([path, source]) => [
      ...importSpecifiers(source)
        .filter((spec) => !isAllowedImport(path, spec))
        .map((spec) => `${path}: ${spec}`),
      ...UNREADABLE_FORMS.filter((form) => form.test(withoutComments(source))).map(
        (form) => `${path}: an import the rule cannot resolve (${form.source})`,
      ),
    ]);
