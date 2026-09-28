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
  llm_provider: 'ollama',
  llm_base_url: 'http://localhost:11434',
  llm_model: 'qwen3:8b',
  llm_api_key_set: false,
  llm_api_key_masked: '',
  comfyui_base_url: 'http://127.0.0.1:8188',
  providers_available: ['ollama'],
  available_models: ['qwen3:8b'],
  ui_language: 'en',
};

const answer = (body: unknown) => ({ ok: true, status: 200, json: async () => body });

const routes = (url: string) => {
  if (url.includes('/api/settings')) return settings;
  if (url.includes('/api/personas')) return { personas: [], available_tools: [], personas_dir: null };
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
    expect(screen.getByText(ko.settings.footer.save)).toBeTruthy();
    expect(document.documentElement.lang).toBe('ko');

    const shown = visibleWords(baseElement);
    // A whole-word match: the dock's tab label `Clone` sits inside the product name `UClone-X`.
    const standsAlone = (sentence: string) =>
      new RegExp(`(?<![A-Za-z])${sentence.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![A-Za-z])`).test(shown);
    expect(englishOnly.filter((sentence) => shown.includes(sentence)).filter(standsAlone)).toEqual([]);
  });

  it('keeps what the user typed when the language changes', async () => {
    render(
      <LocaleProvider hints={['en-US']}>
        <SettingsModal developerMode={false} onDeveloperModeChange={() => {}} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );
    const endpoint = (await screen.findByDisplayValue('http://localhost:11434')) as HTMLInputElement;
    fireEvent.change(endpoint, { target: { value: 'http://gpu.lan:11434' } });

    fireEvent.change(screen.getByTestId('settings-language-select'), { target: { value: 'ko' } });

    await waitFor(() => expect(screen.getByText(ko.settings.header.title)).toBeTruthy());
    expect(endpoint.value).toBe('http://gpu.lan:11434');
  });
});
