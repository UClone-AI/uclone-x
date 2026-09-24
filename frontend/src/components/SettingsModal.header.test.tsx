import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { SettingsModal } from './SettingsModal';

/**
 * The Settings header at phone width (#1416).
 *
 * At 375 px the header overflowed to ~990 px and the modal's `overflow-hidden` cut it off,
 * taking the close button with it. The cause was one flex minimum: the workspace path is a
 * `truncate` (nowrap) line, and a flex item's default `min-width: auto` is its min-content
 * width, so the text column could not be narrower than the whole path. Every ancestor
 * inherited that width and the close button was pushed out past the right edge.
 *
 * jsdom does no layout, so these cases cannot measure a width. They pin the structural
 * properties the fix relies on instead. The fix as a whole was measured in a real browser at
 * 375x812 (Playwright against a built bundle): header scrollWidth 992 before, 341 =
 * clientWidth after, with the close button inside the modal. The `min-w-0` chain is the
 * load-bearing one; `flex-wrap` and `shrink-0` keep the title badge and the close button
 * from being squeezed once the row is allowed to shrink.
 */

const settings = {
  llm_provider: 'ollama',
  llm_base_url: 'http://localhost:11434',
  llm_model: 'qwen3:8b',
  llm_api_key_set: false,
  llm_api_key_masked: '',
  comfyui_base_url: 'http://127.0.0.1:8188',
  providers_available: ['ollama'],
  available_models: ['qwen3:8b'],
  workspace_dir: '/Users/someone/a/long/workspace/path/that/cannot/fit/at/phone/width',
};

const classesOf = (el: Element) => el.className.split(/\s+/);

describe('SettingsModal header at narrow widths (#1416)', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (url.includes('/api/settings')) return { ok: true, json: async () => settings } as Response;
        if (url.includes('/api/personas')) {
          return { ok: true, json: async () => ({ personas: [], personas_dir: '' }) } as Response;
        }
        return { ok: true, json: async () => ({}) } as Response;
      }),
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it('lets the text column shrink below the width of the workspace path', async () => {
    render(<SettingsModal isOpen onClose={() => {}} developerMode={false} onDeveloperModeChange={() => {}} />);
    const path = await screen.findByText(/Workspace:/);
    const header = screen.getByTestId('settings-header');

    // Every flex item between the header and the truncated path must be allowed to be
    // narrower than its content; one `min-width: auto` on the way up is enough to overflow.
    const chain: Element[] = [];
    for (let el = path.parentElement; el && el !== header; el = el.parentElement) chain.push(el);
    const flexItems = chain.filter((el) => el.parentElement && classesOf(el.parentElement).includes('flex'));
    expect(flexItems.length).toBeGreaterThan(0);
    for (const el of flexItems) expect(classesOf(el)).toContain('min-w-0');
  });

  it('keeps the close button in the header at full size', async () => {
    render(<SettingsModal isOpen onClose={() => {}} developerMode={false} onDeveloperModeChange={() => {}} />);
    await screen.findByText(/Workspace:/);
    const header = screen.getByTestId('settings-header');

    const close = within(header).getByTitle('Close');
    expect(close.parentElement).toBe(header);
    expect(classesOf(close)).toContain('shrink-0');
  });

  it('wraps the badge under the title rather than widening the row', async () => {
    render(<SettingsModal isOpen onClose={() => {}} developerMode={false} onDeveloperModeChange={() => {}} />);
    const title = await screen.findByRole('heading', { name: /Runtime Settings/ });

    expect(classesOf(title)).toContain('flex-wrap');
  });
});
