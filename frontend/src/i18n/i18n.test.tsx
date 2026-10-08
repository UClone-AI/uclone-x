import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import { en } from './en';
import { fmt, placeholders, plural } from './format';
import { ko } from './ko';
import { LocaleProvider, useCopy, useLocale } from './LocaleProvider';
import { PAINT_HINT_KEY, resolveLanguage } from './language';

/**
 * The head's languages.
 *
 * Each catalog is JSON, and `ko` is typed as `Messages`, so a key Korean lacks already fails
 * `tsc`. What the type cannot see: an extra Korean key (a nested JSON object is not checked for
 * excess properties), and a placeholder, because every sentence is a `string`. A Korean
 * sentence that drops `{name}` would show a screen without the name, and one that renames it
 * would throw from `fmt`. The parity case compares each sentence's placeholder set for that
 * reason.
 */

type Tree = { [key: string]: unknown };

const leaves = (tree: Tree, prefix = ''): Array<[string, unknown]> =>
  Object.entries(tree).flatMap(([key, value]) =>
    value !== null && typeof value === 'object'
      ? leaves(value as Tree, `${prefix}${key}.`)
      : [[`${prefix}${key}`, value] as [string, unknown]],
  );

const shape = (tree: Tree) =>
  Object.fromEntries(
    leaves(tree).map(([path, value]) => [
      path,
      typeof value === 'string' ? `string{${[...placeholders(value)].sort().join(',')}}` : typeof value,
    ]),
  );

describe('catalogs', () => {
  // Killed by: frontend/src/i18n/locales/ko/settings.json :: "workspace": "📁 작업 폴더: {dir}",
  // Becomes: "workspace": "📁 작업 폴더",
  it('give Korean every key English has, with the same placeholders', () => {
    expect(shape(ko as unknown as Tree)).toEqual(shape(en as unknown as Tree));
  });

  it('write no empty string in either language', () => {
    for (const catalog of [en, ko]) {
      const empty = leaves(catalog as unknown as Tree)
        .filter(([, value]) => typeof value !== 'string' || value.trim() === '')
        .map(([path]) => path);
      expect(empty).toEqual([]);
    }
  });
});

describe('fmt', () => {
  it('puts each value where its placeholder is, in any order', () => {
    expect(fmt('{b} then {a}, {b} again', { a: 1, b: 'x' })).toBe('x then 1, x again');
  });

  it('inserts a value as it is, braces and all', () => {
    expect(fmt('starts with {snippet}', { snippet: '{"mcpServers": {name}}' })).toBe(
      'starts with {"mcpServers": {name}}',
    );
  });

  // Killed by: frontend/src/i18n/format.ts ::       throw new Error(`No value for ${marker} in "${template}"`);
  // Becomes:       return marker;
  it('refuses, under test, a placeholder the call site gave no value for', () => {
    expect(() => fmt('Saved {name}.', { nmae: 'x' })).toThrow('No value for {name}');
  });
});

describe('plural', () => {
  const forms = { one: '{count} tool from {name}', other: '{count} tools from {name}' };

  // Killed by: frontend/src/i18n/format.ts ::   fmt(count === 0 && forms.zero ? forms.zero : count === 1 ? forms.one : forms.other, { ...values, count });
  // Becomes:   fmt(forms.other, { ...values, count });
  it('uses `one` for exactly one and `other` for every other count', () => {
    expect(plural(forms, 1, { name: 'git' })).toBe('1 tool from git');
    expect(plural(forms, 0, { name: 'git' })).toBe('0 tools from git');
    expect(plural(forms, 2, { name: 'git' })).toBe('2 tools from git');
  });

  it('uses `zero` when provided and count is 0', () => {
    const withZero = { ...forms, zero: 'no tools from {name}' };
    expect(plural(withZero, 0, { name: 'git' })).toBe('no tools from git');
    expect(plural(withZero, 1, { name: 'git' })).toBe('1 tool from git');
    expect(plural(withZero, 2, { name: 'git' })).toBe('2 tools from git');
  });
});

/**
 * Korean appears only in the Korean catalog (§3.3, step 2's head guard).
 *
 * Read through Vite's raw glob for the reason `copy.test.ts` gives: `frontend` declares no
 * `@types/node`. Tests are exempt, because a locale-switch case has to name the Korean it
 * expects. The English JSON is scanned too: a Korean sentence pasted into it would put Korean
 * on an English screen.
 */
const SOURCES: Record<string, string> = import.meta.glob(['../**/*.{ts,tsx}', './locales/en/*.json'], {
  query: '?raw',
  import: 'default',
  eager: true,
});

const HANGUL = /[ᄀ-ᇿ㄰-㆏가-힯]/;

// Vite keys a glob by the path relative to this file, so `src/i18n/ko/x.ts` is `./ko/x.ts`.
const mayHoldHangul = (path: string) =>
  path.startsWith('./ko/') || /\.test\.tsx?$/.test(path) || path.startsWith('../test/');

describe('head Hangul guard', () => {
  // Killed by: frontend/src/i18n/locales/en/settings.json :: "title": "Language",
  // Becomes: "title": "언어",
  it('finds Korean only in the Korean catalog and in tests', () => {
    const scanned = Object.keys(SOURCES).filter((path) => !mayHoldHangul(path));
    expect(scanned.length).toBeGreaterThan(50);
    expect(scanned).toContain('./locales/en/settings.json');
    const offenders = scanned.filter((path) => HANGUL.test(SOURCES[path]));
    expect(offenders).toEqual([]);
  });

  it('does scan the Korean source it exempts, so the exemption is not vacuous', () => {
    expect(HANGUL.test(SOURCES['./ko/autonym.ts'])).toBe(true);
  });
});

describe('resolveLanguage', () => {
  it.each([
    ['en', ['ko-KR'], 'en'],
    ['ko', ['en-US'], 'ko'],
    ['system', ['ko-KR', 'en-US'], 'ko'],
    ['system', ['fr-FR', 'en-GB', 'ko'], 'en'],
    ['system', ['ko_KR.UTF-8'], 'ko'],
    ['system', ['fr-FR'], 'en'],
    ['system', [], 'en'],
  ] as const)('%s with %j is %s', (choice, hints, expected) => {
    expect(resolveLanguage(choice, hints)).toBe(expected);
  });
});

const Probe = () => {
  const copy = useCopy();
  const locale = useLocale();
  return (
    <div>
      <span data-testid="title">{copy.settings.language.title}</span>
      <span data-testid="choice">{locale.choice}</span>
      <button onClick={() => locale.setChoice('ko').catch(() => {})}>ko</button>
    </div>
  );
};

type Answer = { ok: boolean; status: number; json: () => Promise<unknown> };
const answer = (body: unknown, ok = true, status = 200): Answer => ({
  ok,
  status,
  json: async () => body,
});

describe('LocaleProvider', () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(() => {
    vi.unstubAllGlobals();
    document.documentElement.lang = '';
  });

  it('renders English outside a provider, so component tests need no wrapper', () => {
    render(<Probe />);
    expect(screen.getByTestId('title').textContent).toBe('Language');
  });

  it('follows the browser for "system" and marks the document', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => answer({ ui_language: 'system' })));
    render(
      <LocaleProvider hints={['ko-KR']}>
        <Probe />
      </LocaleProvider>,
    );
    expect(screen.getByTestId('title').textContent).toBe('언어');
    await waitFor(() => expect(document.documentElement.lang).toBe('ko'));
  });

  // Killed by: frontend/src/i18n/LocaleProvider.tsx :: setChoiceState(data.ui_language);
  // Becomes: void data.ui_language;
  it("adopts the Core's saved choice over the browser", async () => {
    vi.stubGlobal('fetch', vi.fn(async () => answer({ ui_language: 'ko' })));
    render(
      <LocaleProvider hints={['en-US']}>
        <Probe />
      </LocaleProvider>,
    );
    await waitFor(() => expect(screen.getByTestId('title').textContent).toBe('언어'));
    expect(window.localStorage.getItem(PAINT_HINT_KEY)).toBe('ko');
  });

  it('paints the stored hint before the Core answers', () => {
    window.localStorage.setItem(PAINT_HINT_KEY, 'ko');
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    render(
      <LocaleProvider hints={['en-US']}>
        <Probe />
      </LocaleProvider>,
    );
    expect(screen.getByTestId('title').textContent).toBe('언어');
  });

  it('saves a choice to the Core', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => answer({ ui_language: 'en' }));
    vi.stubGlobal('fetch', fetchMock);
    render(
      <LocaleProvider hints={['en-US']}>
        <Probe />
      </LocaleProvider>,
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => screen.getByText('ko').click());

    expect(screen.getByTestId('title').textContent).toBe('언어');
    const post = fetchMock.mock.calls.find(([, init]) => init?.method === 'POST');
    expect(post?.[0]).toBe('/api/settings');
    expect(JSON.parse(post?.[1]?.body as string)).toEqual({ ui_language: 'ko' });
  });

  // Killed by: frontend/src/i18n/LocaleProvider.tsx :: if (live && !chosenLocally.current && isUiLanguage(data.ui_language)) {
  // Becomes: if (live && isUiLanguage(data.ui_language)) {
  it('keeps a choice made before the first read returns', async () => {
    let answerRead: (value: Answer) => void = () => {};
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init?: RequestInit) =>
        init?.method === 'POST'
          ? Promise.resolve(answer({ ui_language: 'ko' }))
          : new Promise<Answer>((resolve) => {
              answerRead = resolve;
            }),
      ),
    );
    render(
      <LocaleProvider hints={['en-US']}>
        <Probe />
      </LocaleProvider>,
    );
    await act(async () => screen.getByText('ko').click());
    // The read was sent before the choice and answers with what the Core held then.
    await act(async () => answerRead(answer({ ui_language: 'en' })));

    expect(screen.getByTestId('choice').textContent).toBe('ko');
    expect(screen.getByTestId('title').textContent).toBe('언어');
  });

  // Killed by: frontend/src/i18n/LocaleProvider.tsx :: setChoiceState(previous);
  // Becomes: void previous;
  it('puts the previous language back when the save fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init?: RequestInit) =>
        init?.method === 'POST' ? answer({}, false, 500) : answer({ ui_language: 'en' }),
      ),
    );
    render(
      <LocaleProvider hints={['en-US']}>
        <Probe />
      </LocaleProvider>,
    );
    await waitFor(() => expect(screen.getByTestId('choice').textContent).toBe('en'));
    await act(async () => screen.getByText('ko').click());

    await waitFor(() => expect(screen.getByTestId('title').textContent).toBe('Language'));
    expect(screen.getByTestId('choice').textContent).toBe('en');
  });
});
