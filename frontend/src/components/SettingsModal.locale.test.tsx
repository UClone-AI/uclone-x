import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SettingsModal } from './SettingsModal';
import { LocaleProvider } from '../i18n';
import { en } from '../i18n/en';
import { ko } from '../i18n/ko';
import type { RuntimeSettings } from '../types';

/**
 * Settings follows the language control at once and leaves no English behind.
 *
 * Every other Settings case renders without a provider and so reads English; this is the one
 * that renders the window the way `main.tsx` does and switches it.
 */

const settings: RuntimeSettings = {
  ui_language: 'en',
};

/** One connection that lists models and one that cannot, so both kinds of line are drawn. */
const connections = {
  connections: [
    {
      id: 'ollama', kind: 'ollama', label: 'Ollama', base_url: 'http://127.0.0.1:11434', key_set: false,
      key_masked: null, source: 'settings', paid: false, status: 'connected', detail: null, model_count: 1,
    },
    {
      id: 'gemini', kind: 'gemini', label: 'Google Gemini', base_url: null, key_set: false, key_masked: null,
      source: 'env', env_var: 'GEMINI_API_KEY', paid: true, status: 'no_key', detail: 'No API key is set', model_count: null,
    },
  ],
  kinds: [],
};
const modelSet = {
  groups: [
    {
      connection_id: 'ollama', label: 'Ollama', kind: 'ollama', status: 'connected', detail: null,
      models: [{ ref: 'ollama/qwen3:8b', id: 'qwen3:8b', display_name: 'qwen3:8b', capabilities: ['chat'], context_window: null }],
    },
    { connection_id: 'gemini', label: 'Google Gemini', kind: 'gemini', status: 'no_key', detail: 'No API key is set', models: [] },
  ],
  defaults: { deep: null, fast: null, image: 'auto' },
  recommended: { deep: 'ollama/qwen3:8b', fast: null },
};

const answer = (body: unknown) => ({ ok: true, status: 200, json: async () => body });

const routes = (url: string) => {
  if (url.includes('/api/settings')) return settings;
  if (url.includes('/api/connections')) return connections;
  if (url.includes('/api/models')) return modelSet;
  if (url.includes('/api/clones')) return { clones: [], available_tools: [], personas_dir: null };
  if (url.includes('/api/diagnostics/consent')) return { state: 'unasked', error: null, journal: '' };
  if (url.includes('/api/diagnostics/report'))
    return { state: 'unasked', available: true, error: null, unreadable_lines: 0, count: 0, distinct: 0 };
  if (url.includes('/api/mcp')) return { servers: [] };
  if (url.includes('/api/skills')) return { skills: [] };
  return {};
};

type Tree = { [key: string]: unknown };
const strings = (tree: Tree): string[] =>
  Object.values(tree).flatMap((value) =>
    typeof value === 'string' ? [value] : value && typeof value === 'object' ? strings(value as Tree) : [],
  );

/** English sentences Korean writes differently; the ones spelled alike (product names) are not leftovers. */
const englishOnly = (() => {
  const korean = new Set(strings(ko as unknown as Tree));
  return strings(en as unknown as Tree)
    .map((s) => s.trim())
    .filter((s) => s.length > 3 && !korean.has(s) && /[a-z]{3}/.test(s));
})();

const shownText = (root: HTMLElement) => root.textContent ?? '';

const visibleWords = (root: HTMLElement) => {
  const attributes = Array.from(root.querySelectorAll('[placeholder],[aria-label],[title]')).flatMap((el) =>
    ['placeholder', 'aria-label', 'title'].map((name) => el.getAttribute(name) ?? ''),
  );
  return [root.textContent ?? '', ...attributes].join('\n');
};

describe('Settings in the chosen language', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('fetch', vi.fn(async (url: string) => answer(routes(url))));
  });
  afterEach(() => vi.unstubAllGlobals());

  // Killed by: frontend/src/components/SettingsModal.tsx :: <span>{t.tabs[tab.id]}</span>
  // Becomes: <span>{tab.id}</span>
  it('switches to Korean from the language control with no English left', async () => {
    const { baseElement } = render(
      <LocaleProvider hints={['en-US']}>
        <SettingsModal developerMode={false} onDeveloperModeChange={() => {}} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );
    const select = await screen.findByTestId('settings-language-select');
    expect(screen.getByText(en.settings.header.title)).toBeTruthy();
    // The leftover check below must be able to see English at all.
    expect(englishOnly.filter((sentence) => visibleWords(baseElement).includes(sentence)).length).toBeGreaterThan(20);

    fireEvent.change(select, { target: { value: 'ko' } });

    await waitFor(() => expect(screen.getByText(ko.settings.header.title)).toBeTruthy());
    expect(screen.getByText(ko.settings.tabs.llm)).toBeTruthy();
    expect(screen.getByText(ko.settings.footer.close)).toBeTruthy();
    // The model sections are drawn in Korean too, and never with the Core's English detail.
    expect(await screen.findByText(ko.gateway.connections.title)).toBeTruthy();
    expect(screen.getByTestId('connection-status-gemini')).toHaveTextContent(ko.gateway.status.no_key);
    expect(shownText(baseElement)).not.toContain('No API key is set');
    expect(document.documentElement.lang).toBe('ko');

    const shown = visibleWords(baseElement);
    // A whole-word match: the dock's tab label `Clone` sits inside the product name `UClone-X`.
    const standsAlone = (sentence: string) =>
      new RegExp(`(?<![A-Za-z])${sentence.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![A-Za-z])`).test(shown);
    expect(englishOnly.filter((sentence) => shown.includes(sentence)).filter(standsAlone)).toEqual([]);
    // Renders the whole window twice and searches it for every English sentence: under 2 s on
    // a quiet machine, past the 5 s default under a parallel gate, where it timed out.
  }, 20_000);

  it('keeps what the user typed when the language changes', async () => {
    render(
      <LocaleProvider hints={['en-US']}>
        <SettingsModal developerMode={false} onDeveloperModeChange={() => {}} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );
    const folder = (await screen.findByLabelText(en.settings.folders.addLabel)) as HTMLInputElement;
    fireEvent.change(folder, { target: { value: '~/notes' } });

    fireEvent.change(screen.getByTestId('settings-language-select'), { target: { value: 'ko' } });

    await waitFor(() => expect(screen.getByText(ko.settings.header.title)).toBeTruthy());
    expect(folder.value).toBe('~/notes');
  }, 20_000);
});
