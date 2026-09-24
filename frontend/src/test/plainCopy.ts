import { expect } from 'vitest';

/**
 * Asserts that `text` is copy a reader who does not read code can be shown (#1408, #1411).
 *
 * The same rules `_assert_plain` holds the Core's sentences to, plus the transport text a
 * browser and a bare HTTP answer produce: no path, no exception class name, no parser
 * words, no "Failed to fetch" / "fetch failed" / "Load failed", no status line such as "500 Internal".
 * Each check names what it caught, so a failure says which rule the copy broke.
 */
export function expectPlain(text: string | null | undefined): void {
  const copy = text ?? '';
  expect(copy.trim(), 'the copy is empty').not.toBe('');
  expect(copy, 'a path').not.toMatch(/[/\\]/);
  expect(copy, 'an exception class name').not.toMatch(/[A-Za-z]*(Error|Exception)\b/);
  // "fetch failed" is Node's wording for the refused connection a browser calls "Failed to fetch".
  expect(copy, 'transport text').not.toMatch(/Failed to fetch|fetch failed|Load failed|NetworkError/i);
  expect(copy, 'a status line').not.toMatch(/\b\d{3} [A-Z]/);
  expect(copy, 'an HTTP status').not.toMatch(/\bHTTP\b/);
  expect(copy, 'a stringified object').not.toMatch(/\[object /);
  expect(copy, 'parser words').not.toMatch(/\bline \d|\bcolumn \d|\bunterminated\b|\bvalidation error/i);
}
