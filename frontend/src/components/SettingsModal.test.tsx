import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SettingsModal } from './SettingsModal';
import type { RuntimeSettings } from '../types';
import { expectPlain } from '../test/plainCopy';
import { en } from '../i18n/en';
import { fmt, plural } from '../i18n/format';
import { ko } from '../i18n/ko';
import { LocaleProvider } from '../i18n';

const SETTINGS_FAILURE = en.settings.failure;
const PERSONA_EDITOR_COPY = en.personaEditor;

const baseSettings: RuntimeSettings = {
};

/** `GET /api/models`: no connection lists anything, and nothing is chosen. */
const emptyModelSet = {
  groups: [],
  defaults: { deep: null, fast: null, image: 'auto' },
  recommended: { deep: null, fast: null },
};

const consentPayload = {
  state: 'unasked',
  error: null,
  journal: '/home/u/.uclone/diagnostics/failures.jsonl',
};

const reportPayload = {
  state: 'unasked',
  available: true,
  error: null,
  unreadable_lines: 0,
  recording_blocked: null,
  count: 0,
  distinct: 0,
  title: null,
  body: null,
  issue_url: null,
  search_url: null,
};

const personasPayload = { clones: [], available_tools: [], personas_dir: null };

/** Developer mode is a required prop; the cases that are not about it render it off, its default. */
const devModeOff = { developerMode: false, onDeveloperModeChange: () => {} };

type RouteResponse = { ok: boolean; status: number; statusText?: string; json: () => Promise<unknown> };
type Handler = (url: string, init?: RequestInit) => RouteResponse | Promise<RouteResponse>;

const jsonResponse = (body: unknown, ok = true, status = 200): RouteResponse => ({
  ok,
  status,
  json: async () => body,
});

/** Routes the common GETs every render needs, plus whatever overrides the test cares about. */
const mockFetch = (overrides: Record<string, Handler> = {}) => {
  const calls: Array<{ url: string; method: string; body?: unknown }> = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const body = init?.body ? JSON.parse(init.body as string) : undefined;
    calls.push({ url, method, body });

    for (const [pattern, handler] of Object.entries(overrides)) {
      if (url.includes(pattern)) return handler(url, init);
    }
    if (url.includes('/api/settings/remote-gpu/status')) return jsonResponse({ host: '', connected: false });
    if (url.includes('/api/settings/remote-gpu/hosts')) return jsonResponse({ hosts: ['macmini', 'dell'] });
    if (url.includes('/api/settings')) return jsonResponse(baseSettings);
    if (url.includes('/api/connections')) return jsonResponse({ connections: [], kinds: [] });
    if (url.includes('/api/models')) return jsonResponse(emptyModelSet);
    if (url.includes('/api/clones')) return jsonResponse(personasPayload);
    if (url.includes('/api/diagnostics/consent')) return jsonResponse(consentPayload);
    if (url.includes('/api/diagnostics/report')) return jsonResponse(reportPayload);
    return jsonResponse({});
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
};

describe('SettingsModal Escape (#1036)', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (url.includes('/api/clones')) {
          return { ok: true, json: async () => ({ clones: [], personas_dir: '' }) } as Response;
        }
        if (url.includes('/api/diagnostics/consent')) {
          return {
            ok: true,
            json: async () => ({ state: 'undecided', error: null, journal: '' }),
          } as Response;
        }
        return { ok: true, json: async () => ({}) } as Response;
      }),
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it('closes on Escape while open', () => {
    const onClose = vi.fn();
    render(<SettingsModal {...devModeOff} isOpen onClose={onClose} />);

    fireEvent.keyDown(window, { key: 'Escape' });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('does not respond to Escape once closed', () => {
    const onClose = vi.fn();
    render(<SettingsModal {...devModeOff} isOpen={false} onClose={onClose} />);

    // Killed by: frontend/src/components/SettingsModal.tsx :: useEscapeOwner('dialog', isOpen, handleClose);
    // Becomes: useEscapeOwner('dialog', true, handleClose);
    fireEvent.keyDown(window, { key: 'Escape' });

    expect(onClose).not.toHaveBeenCalled();
  });

  it('does not render dialog content while closed', () => {
    render(<SettingsModal {...devModeOff} isOpen={false} onClose={vi.fn()} />);

    expect(screen.queryByText('Settings')).toBeNull();
  });
});

/**
 * Cancelling a model install (#1233).
 *
 * An Ollama pull is minutes to tens of minutes of multi-gigabyte transfer, and
 * before this the browser had no way to stop waiting for one: the `fetch` carried
 * no signal, so closing Settings or navigating away left it running with nobody to
 * receive it. These tests drive the abort themselves rather than waiting on a
 * clock — the fetch mock here holds its promise open until the test's own
 * `signal.addEventListener('abort', ...)` fires, so a component that never wires a
 * signal leaves the promise unresolved and the assertion fails within the
 * `waitFor` budget instead of passing because the request happened to be quick.
 */

describe('SettingsModal developer mode (owner ruling 2026-09-22)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('shows the switch off when developer mode is off, and asks to turn it on', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: onClick={() => onDeveloperModeChange(!developerMode)}
    // Becomes: onClick={() => onDeveloperModeChange(developerMode)}
    const calls = mockFetch();
    const onChange = vi.fn();
    render(<SettingsModal isOpen onClose={() => {}} developerMode={false} onDeveloperModeChange={onChange} />);

    const toggle = screen.getByRole('switch', { name: /developer mode/i });
    expect(toggle).toHaveAttribute('aria-checked', 'false');
    fireEvent.click(toggle);
    expect(onChange).toHaveBeenCalledWith(true);
    // A head preference: switching it writes nothing to the runtime.
    await waitFor(() => expect(calls.some((c) => c.url.includes('/api/settings'))).toBe(true));
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([]);
  });

  it('shows the switch on when developer mode is on, and asks to turn it off', () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: aria-checked={developerMode}
    // Becomes: aria-checked={false}
    mockFetch();
    const onChange = vi.fn();
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={onChange} />);

    const toggle = screen.getByRole('switch', { name: /developer mode/i });
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(toggle);
    expect(onChange).toHaveBeenCalledWith(false);
  });
});

/** One registered skill, the shape `GET /api/skills` answers with. */
const skillsPayload = {
  skills: [
    {
      name: 'csv_summariser',
      description: 'Summarises a CSV file into a short table.',
      version: '1.0.0',
      author: 'local',
      origin: 'human',
      status: 'active',
      isolation_level: 'sandbox',
      content_sha256: 'abc123',
      scripts: ['summarise.py'],
      tags: ['data'],
      approved_by: 'owner',
      approved_at: '2026-09-20T10:00:00Z',
      audit_report: {
        skill_name: 'csv_summariser',
        is_safe: true,
        recommendation: 'approve',
        risk_score: 0,
        detected_risks: [],
        auditor_version: '1',
        content_sha256: 'abc123',
      },
    },
  ],
  summary: { total_skills: 1, active_count: 1, pending_count: 0, quarantined_count: 0 },
};

const acpPayload = {
  transport: 'stdio',
  sdk_version_specified: '0.12.1',
  serving: false,
  presence: {
    shell_module_present: false,
    sdk_installed: false,
    transport: 'stdio',
    sdk_version_specified: '0.12.1',
    reason: 'No ACP shell is installed in this build.',
  },
  methods: [],
  mcp_descriptors: [],
  mcp_loader_warning: '',
  counts: {
    agent: { total: 0, implemented: 0, not_implemented: 0, not_implementable: 0, out_of_scope: 0 },
    client: { total: 0, implemented: 0, not_implemented: 0, not_implementable: 0, out_of_scope: 0 },
  },
};

const evalMetrics = {
  total_suites: 1,
  total_probes: 4,
  passed_probes: 4,
  failed_probes: 0,
  pass_rate: 1,
};
const evalsPayload = { status: 'ok', error: null, suites: [], scorecard: {}, metrics: evalMetrics };

describe('SettingsModal Skills section (#1358)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('lists a registered skill read from /api/skills', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: <SkillsSection />
    // Becomes:
    mockFetch({ '/api/skills': () => jsonResponse(skillsPayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-skills');
    expect(section).toHaveTextContent('Skills');
    await waitFor(() => expect(section).toHaveTextContent('csv_summariser'));
  });

  it("states why the skills could not be read in the runtime's own words, not as a status line", async () => {
    // Killed by: frontend/src/lib/useApiRead.ts :: if (!res.ok) {
    // Becomes: if (false) {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    mockFetch({
      '/api/skills': () => jsonResponse({ detail: 'The skill folder could not be opened.' }, false, 500),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const failure = await screen.findByTestId('settings-skills-error');
    expect(failure).toHaveTextContent('Your skills could not be loaded');
    expect(failure).toHaveTextContent('The skill folder could not be opened.');
    expect(failure).not.toHaveTextContent('HTTP');
    expect(failure).not.toHaveTextContent('500');
  });

  it("says in plain words that the runtime could not be reached, never the browser's transport message", async () => {
    // Killed by: frontend/src/i18n/locales/en/skills.json :: UClone-X could not be reached. If
    // Becomes: Failed to fetch. If
    vi.spyOn(console, 'error').mockImplementation(() => {});
    mockFetch({ '/api/skills': () => Promise.reject(new TypeError('Failed to fetch')) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const failure = await screen.findByTestId('settings-skills-error');
    expect(failure).toHaveTextContent('Your skills could not be loaded');
    expect(failure).toHaveTextContent('UClone-X could not be reached');
    expect(failure).not.toHaveTextContent('Failed to fetch');
    expect(failure).not.toHaveTextContent('TypeError');
  });

  it('says in plain words that the runtime gave no reason, never its status line, when a 500 has no JSON body', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: fault.detail ?? copy.plainCause[fault.kind]
    // Becomes: fault.detail ?? fault.kind
    vi.spyOn(console, 'error').mockImplementation(() => {});
    mockFetch({
      '/api/skills': () => ({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => {
          throw new SyntaxError('Unexpected token \'I\', "Internal S"... is not valid JSON');
        },
      }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const failure = await screen.findByTestId('settings-skills-error');
    expect(failure).toHaveTextContent('Your skills could not be loaded');
    expect(failure).toHaveTextContent('UClone-X answered with an error but gave no reason');
    for (const raw of ['500', 'HTTP', 'Internal Server Error', 'Unexpected token', 'JSON']) {
      expect(failure).not.toHaveTextContent(raw);
    }
  });

  it('says it is loading the skills while the read is out, not nothing and not an empty list', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: {fault === null && data === null && (
    // Becomes: {false && (
    mockFetch({ '/api/skills': () => new Promise<RouteResponse>(() => {}) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-skills');
    expect(within(section).getByRole('status')).toHaveTextContent('Loading your skills');
    expect(screen.queryByTestId('settings-skills-empty')).toBeNull();
    expect(screen.queryByTestId('settings-skills-error')).toBeNull();
  });

  it('offers to try again after a failed read, and shows the skills once it succeeds', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: onRetry={reload}
    // Becomes: onRetry={() => {}}
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let reads = 0;
    mockFetch({
      '/api/skills': () => {
        reads += 1;
        return reads === 1 ? jsonResponse({ detail: 'boom' }, false, 500) : jsonResponse(skillsPayload);
      },
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const failure = await screen.findByTestId('settings-skills-error');
    fireEvent.click(within(failure).getByRole('button', { name: 'Try again' }));

    const section = screen.getByTestId('settings-skills');
    await waitFor(() => expect(section).toHaveTextContent('csv_summariser'));
    expect(screen.queryByTestId('settings-skills-error')).toBeNull();
  });

  it("describes the skills in plain words, with none of the checker's internal terms", async () => {
    // Killed by: frontend/src/i18n/locales/en/skills.json :: "hint": "Passed the safety check"
    // Becomes: "hint": "AST Verification Passed"
    mockFetch({ '/api/skills': () => jsonResponse(skillsPayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-skills');
    await waitFor(() => expect(section).toHaveTextContent('csv_summariser'));
    const text = section.textContent ?? '';
    const shown = Array.from(section.querySelectorAll('option')).map((o) => o.textContent ?? '');
    for (const term of [
      // Letters only on either side: textContent runs a count into the next word ("1AST").
      /(?<![A-Za-z])AST(?![A-Za-z])/,
      /quarantin/i,
      /governed/i,
      /audit/i,
      /verdict/i,
      /taint/i,
      /registry/i,
      /synthesi[sz]ed/i,
      /SHA-256/,
      /(?<![A-Za-z])LLM(?![A-Za-z])/,
    ]) {
      expect(text, String(term)).not.toMatch(term);
      for (const option of shown) expect(option, String(term)).not.toMatch(term);
    }
  });

  it('says why a skill changed after its approval is not used (#1720)', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: <strong>{notLoadedCopy.label}</strong> {notLoaded}
    // Becomes: <strong>{notLoadedCopy.label}</strong>
    const reason =
      'This skill was changed after it was approved, so it is not used. To use it, check what ' +
      'changed, then approve it again in a terminal window: ucx skill approve csv_summariser';
    const [skill] = skillsPayload.skills;
    mockFetch({
      '/api/skills': () =>
        jsonResponse({
          skills: [{ ...skill, status: 'quarantined', not_loaded_reason: reason }],
          summary: { total_skills: 1, active_count: 0, pending_count: 0, quarantined_count: 1 },
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const shown = await screen.findByTestId('skill-not-loaded-reason');
    expect(shown).toHaveTextContent(`Why it is not used: ${reason}`);
    expect(screen.getByTestId('settings-skills')).toHaveTextContent('Blocked');
  });

  /** A skill the Core refused, with the reason as a code (#1777). */
  const refusedSkill = (code: string, reason: string, name = 'csv_summariser') => ({
    ...skillsPayload.skills[0],
    name,
    status: 'quarantined',
    not_loaded_reason: reason,
    not_loaded_code: code,
    not_loaded_params: { name },
  });
  const refusedPayload = (...skills: ReturnType<typeof refusedSkill>[]) => ({
    skills,
    summary: {
      total_skills: skills.length,
      active_count: 0,
      pending_count: 0,
      quarantined_count: skills.length,
    },
  });
  const englishChanged = fmt(en.skills.notLoaded.codes.changed_after_approval, {
    name: 'csv_summariser',
  });

  it('words the reason a skill is not used in Korean when the screens are in Korean (#1777)', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: if (!code || !Object.prototype.hasOwnProperty.call(t.codes, code)) return fallback;
    // Becomes: return fallback;
    window.localStorage.clear();
    mockFetch({
      '/api/settings': () => jsonResponse({ ...baseSettings, ui_language: 'ko' }),
      '/api/skills': () =>
        jsonResponse(refusedPayload(refusedSkill('changed_after_approval', englishChanged))),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    const korean = fmt(ko.skills.notLoaded.codes.changed_after_approval, { name: 'csv_summariser' });
    const shown = await screen.findByTestId('skill-not-loaded-reason');
    await waitFor(() => expect(shown.textContent).toBe(`${ko.skills.notLoaded.label} ${korean}`));
    expect(shown.textContent).not.toContain(englishChanged);
  });

  it('words the reason from the English catalog, and keeps the sent words for an unknown code (#1777)', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: if (!code || !Object.prototype.hasOwnProperty.call(t.codes, code)) return fallback;
    // Becomes: if (!code) return fallback;
    const unknown = 'This skill is not used for a reason this screen does not know yet.';
    mockFetch({
      '/api/skills': () =>
        jsonResponse(
          refusedPayload(
            refusedSkill('renamed_later', unknown, 'alpha_skill'),
            refusedSkill('failed_safety_check', 'sent words', 'beta_skill'),
          ),
        ),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const shown = await screen.findByTestId('skill-not-loaded-reason');
    expect(shown.textContent).toBe(`${en.skills.notLoaded.label} ${unknown}`);
    fireEvent.click(within(screen.getByTestId('settings-skills')).getByText('beta_skill'));
    const safety = `${en.skills.notLoaded.label} ${en.skills.notLoaded.codes.failed_safety_check}`;
    await waitFor(() =>
      expect(screen.getByTestId('skill-not-loaded-reason').textContent).toBe(safety),
    );
    expectPlain(screen.getByTestId('skill-not-loaded-reason').textContent);
  });

  it('asks in one line for skills approved before approvals were recorded to be approved again (#1777)', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: .filter((skill) => skill.not_loaded_code === 'approved_before_pins')
    // Becomes: .filter((skill) => skill.not_loaded_code === 'approved_before_pins_')
    const before = (name: string) =>
      refusedSkill(
        'approved_before_pins',
        fmt(en.skills.notLoaded.codes.approved_before_pins, { name }),
        name,
      );
    mockFetch({
      '/api/skills': () =>
        jsonResponse(
          refusedPayload(
            before('alpha_skill'),
            refusedSkill('changed_after_approval', englishChanged),
            before('beta_skill'),
          ),
        ),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const notice = await screen.findByTestId('settings-skills-reapprove');
    expect(notice.textContent).toBe(
      plural(en.skills.reapprove, 2, { names: 'alpha_skill, beta_skill' }),
    );
    expect(notice).toHaveTextContent('ucx skill approve');
    expect(notice).not.toHaveTextContent('csv_summariser');
    expectPlain(notice.textContent);
  });

  it('shows no re-approval notice when every skill was approved with a record (#1777)', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: {fault === null && reapprove.length > 0 && (
    // Becomes: {fault === null && (
    mockFetch({ '/api/skills': () => jsonResponse(skillsPayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await waitFor(() =>
      expect(screen.getByTestId('settings-skills')).toHaveTextContent('csv_summariser'),
    );
    expect(screen.queryByTestId('settings-skills-reapprove')).toBeNull();
  });

  /** Every sentence of the tab's English copy, for checking none of it shows on a Korean screen. */
  const englishTabCopy = (): string[] => {
    const out: string[] = [];
    const walk = (value: unknown) => {
      if (typeof value === 'string') out.push(value.replace(/\s*\(\{count\}\)$/, ''));
      else if (value && typeof value === 'object') Object.values(value).forEach(walk);
    };
    walk(en.skills.tab);
    return out;
  };

  it("words the whole Skills tab in Korean when the screens are in Korean, with none of it left in English (#1782)", async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: <h3 className="text-sm font-semibold text-white">{copy.checkHeading}</h3>
    // Becomes: <h3 className="text-sm font-semibold text-white">Safety check</h3>
    window.localStorage.clear();
    mockFetch({
      '/api/settings': () => jsonResponse({ ...baseSettings, ui_language: 'ko' }),
      '/api/skills': () => jsonResponse(skillsPayload),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    const section = await screen.findByTestId('settings-skills');
    await waitFor(() => expect(section).toHaveTextContent(ko.skills.tab.checkHeading));
    const tab = ko.skills.tab;
    for (const korean of [
      tab.cards.active.hint,
      tab.status.active,
      tab.recommendation.approve,
      tab.noFindings,
      fmt(tab.listHeading, { count: 1 }),
    ]) {
      expect(section).toHaveTextContent(korean);
    }
    expect(within(section).getByRole('button', { name: tab.refresh })).toBeInTheDocument();
    expect(within(section).getByPlaceholderText(tab.search)).toBeInTheDocument();
    const options = Array.from(section.querySelectorAll('option')).map((o) => o.textContent ?? '');
    expect(options).toContain(tab.anyStatus);
    expect(options).toContain(tab.origin.human);

    const shown = [section.textContent ?? '', ...options, tab.search].join('\n');
    for (const english of englishTabCopy()) {
      expect(shown, english).not.toContain(english);
    }
    // Plain words in Korean too: none of the checker's own terms.
    expect(shown).not.toMatch(/(?<![A-Za-z])AST(?![A-Za-z])|quarantin|audit|verdict|synthesi[sz]ed|SHA-256/i);
  });

  it('counts the skills the search leaves in the list heading, in words (#1782)', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: {fmt(copy.listHeading, { count: filteredSkills.length })}
    // Becomes: {copy.listHeading}
    mockFetch({ '/api/skills': () => jsonResponse(skillsPayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-skills');
    await waitFor(() => expect(section).toHaveTextContent('Skills (1)'));
    fireEvent.change(within(section).getByPlaceholderText(en.skills.tab.search), {
      target: { value: 'no such skill' },
    });
    await waitFor(() => expect(section).toHaveTextContent('Skills (0)'));
    expect(section.textContent).not.toContain('{count}');
  });

  it('says in words that no skill is registered', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: skills.length === 0
    // Becomes: skills.length === -1
    mockFetch({
      '/api/skills': () =>
        jsonResponse({
          skills: [],
          summary: { total_skills: 0, active_count: 0, pending_count: 0, quarantined_count: 0 },
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect(await screen.findByTestId('settings-skills-empty')).toHaveTextContent(
      'No skills are registered',
    );
  });

  const noStorePayload = {
    skills: [],
    store_missing: true,
    summary: { total_skills: 0, active_count: 0, pending_count: 0, quarantined_count: 0 },
  };

  it('says no skill folder was found, and where to look, rather than that none is registered (#1721)', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: {data.store_missing ? copy.noStore : copy.empty}
    // Becomes: {copy.empty}
    mockFetch({ '/api/skills': () => jsonResponse(noStorePayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const empty = await screen.findByTestId('settings-skills-empty');
    expect(empty.textContent).toBe(en.skills.noStore);
    expect(empty).toHaveTextContent('ucx skill --help');
    expect(empty).not.toHaveTextContent(en.skills.empty);
    expectPlain(empty.textContent);
    for (const internal of ['git', 'store_missing', 'rev-parse', '/']) {
      expect(empty.textContent, internal).not.toContain(internal);
    }
  });

  it('says in Korean that no skill folder was found when the screens are in Korean (#1721)', async () => {
    // Killed by: frontend/src/i18n/locales/ko/skills.json :: "noStore": "UClone-X를 시작한 프로젝트에
    // Becomes: "noStore": "No skills are loaded. UClone-X를 시작한 프로젝트에
    window.localStorage.clear();
    mockFetch({
      '/api/settings': () => jsonResponse({ ...baseSettings, ui_language: 'ko' }),
      '/api/skills': () => jsonResponse(noStorePayload),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    const empty = await screen.findByTestId('settings-skills-empty');
    await waitFor(() => expect(empty.textContent).toBe(ko.skills.noStore));
    expect(empty.textContent).not.toMatch(/No skills|loaded/);
  });
});

describe('SettingsModal Diagnostics (#1358)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  const diagnosticsRoutes = {
    '/api/acp/status': () => jsonResponse(acpPayload),
    '/api/evaluations/latest': () => jsonResponse(evalsPayload),
  };

  it('shows the ACP report and the Evals scorecard while developer mode is on', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: {developerMode && <DiagnosticsSection />}
    // Becomes:
    mockFetch(diagnosticsRoutes);
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const section = await screen.findByTestId('settings-diagnostics');
    expect(section).toHaveTextContent('Diagnostics');
    await waitFor(() => expect(screen.getByTestId('acp-panel')).toBeInTheDocument());
    expect(screen.getByTestId('acp-presence')).toHaveTextContent('No ACP shell is installed in this build.');
    await waitFor(() => expect(screen.getAllByTestId('eval-metric-card')).toHaveLength(4));
  });

  it('offers neither, and reads neither, while developer mode is off', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: {developerMode && <DiagnosticsSection />}
    // Becomes: {<DiagnosticsSection />}
    const calls = mockFetch(diagnosticsRoutes);
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    // The modal rendered and read its data, so what is missing is missing from a live modal.
    await screen.findByTestId('settings-skills');
    await waitFor(() => expect(calls.some((c) => c.url.includes('/api/skills'))).toBe(true));
    expect(screen.queryByTestId('settings-diagnostics')).toBeNull();
    expect(screen.queryByTestId('acp-panel')).toBeNull();
    expect(screen.queryByTestId('acp-unavailable')).toBeNull();
    expect(screen.queryAllByTestId('eval-metric-card')).toHaveLength(0);
    expect(calls.filter((c) => c.url.includes('/api/acp/status'))).toEqual([]);
    expect(calls.filter((c) => c.url.includes('/api/evaluations/latest'))).toEqual([]);
  });

  it("keeps the Evals read failure's cause in words (#1344)", async () => {
    // Killed by: frontend/src/components/EvaluationsTab.tsx :: evaluationsData?.status === 'error'
    // Becomes: evaluationsData?.status === 'never'
    const readError = 'Could not read evaluation reports from /srv/evals/reports: NotADirectoryError';
    mockFetch({
      ...diagnosticsRoutes,
      '/api/evaluations/latest': () =>
        jsonResponse({ status: 'error', error: readError, suites: [], scorecard: {}, metrics: evalMetrics }),
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const alert = await screen.findByTestId('eval-read-failure');
    expect(alert).toHaveTextContent('Evaluation results could not be read');
    expect(alert).toHaveTextContent(readError);
    expect(screen.queryAllByTestId('eval-metric-card')).toHaveLength(0);
  });

  it('states why the ACP report could not be read, rather than drawing it as unread', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: {acp.error !== null ? (
    // Becomes: {false ? (
    mockFetch({
      ...diagnosticsRoutes,
      '/api/acp/status': () => jsonResponse({ detail: 'no' }, false, 503),
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const failure = await screen.findByTestId('diagnostics-acp-error');
    expect(failure).toHaveTextContent('ACP status could not be read');
    expect(failure).toHaveTextContent('HTTP 503');
  });

  it('states why the Evals results could not be read, rather than a scorecard of zeros', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: {evals.error !== null ? (
    // Becomes: {false ? (
    vi.spyOn(console, 'error').mockImplementation(() => {});
    mockFetch({
      ...diagnosticsRoutes,
      '/api/evaluations/latest': () => jsonResponse({ detail: 'no' }, false, 502),
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const failure = await screen.findByTestId('diagnostics-evals-error');
    expect(failure).toHaveTextContent('Evaluation results could not be read');
    expect(failure).toHaveTextContent('HTTP 502');
    expect(screen.queryAllByTestId('eval-metric-card')).toHaveLength(0);
  });

  it('says each read is loading while it is out, never "not read" and never zeros', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: ) : acp.data === null ? (
    // Becomes: ) : false ? (
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: ) : evals.data === null ? (
    // Becomes: ) : false ? (
    const hold = () => new Promise<RouteResponse>(() => {});
    mockFetch({ '/api/acp/status': hold, '/api/evaluations/latest': hold });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    expect(await screen.findByTestId('diagnostics-acp-loading')).toHaveTextContent('Reading ACP status');
    expect(screen.getByTestId('diagnostics-evals-loading')).toHaveTextContent('Reading evaluation results');
    expect(screen.queryByTestId('acp-unavailable')).toBeNull();
    expect(screen.queryAllByTestId('eval-metric-card')).toHaveLength(0);
  });

  it('retries a failed ACP read from its error, and shows the report once it succeeds', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: onRetry={acp.reload}
    // Becomes: onRetry={() => {}}
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let reads = 0;
    mockFetch({
      ...diagnosticsRoutes,
      '/api/acp/status': () => {
        reads += 1;
        return reads === 1 ? jsonResponse({ detail: 'no' }, false, 503) : jsonResponse(acpPayload);
      },
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const failure = await screen.findByTestId('diagnostics-acp-error');
    fireEvent.click(within(failure).getByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(screen.getByTestId('acp-panel')).toBeInTheDocument());
    expect(screen.queryByTestId('diagnostics-acp-error')).toBeNull();
  });

  it('retries a failed Evals read from its error, and shows the scorecard once it succeeds', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: onRetry={evals.reload}
    // Becomes: onRetry={() => {}}
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let reads = 0;
    mockFetch({
      ...diagnosticsRoutes,
      '/api/evaluations/latest': () => {
        reads += 1;
        return reads === 1 ? jsonResponse({ detail: 'no' }, false, 502) : jsonResponse(evalsPayload);
      },
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const failure = await screen.findByTestId('diagnostics-evals-error');
    fireEvent.click(within(failure).getByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(screen.getAllByTestId('eval-metric-card')).toHaveLength(4));
    expect(screen.queryByTestId('diagnostics-evals-error')).toBeNull();
  });
});

describe('SettingsModal read-only folders', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  const withRoots: RuntimeSettings = {
    ...baseSettings,
    workspace_dir: '/home/u/.uclone/workspace',
    read_roots: ['/data/papers', '~/gone'],
    read_roots_missing: ['~/gone'],
  };

  it('shows the workspace folder and each read-only folder, marking the missing one', async () => {
    mockFetch({ '/api/settings': () => jsonResponse(withRoots) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect((await screen.findByTestId('settings-workspace-dir')).textContent).toBe(
      '/home/u/.uclone/workspace',
    );
    const list = await screen.findByRole('list', { name: 'Read-only folders' });
    const rows = Array.from(list.querySelectorAll('li'));
    expect(rows.map((r) => r.textContent)).toEqual(['/data/papers', '~/gonenot usable']);
  });

  it('shows the folders set by UCLONE_READ_ROOTS and the entries it ignored', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse({
          ...withRoots,
          read_roots_env: ['/srv/shared'],
          read_roots_env_ignored: ["'rel' is not a full path; start it with / or ~ (from UCLONE_READ_ROOTS)"],
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const envList = await screen.findByRole('list', { name: 'Read-only folders from the environment' });
    expect(Array.from(envList.querySelectorAll('li')).map((r) => r.textContent)).toEqual(['/srv/shared']);
    expect(screen.getByTestId('settings-read-roots-env-ignored').textContent).toContain("'rel' is not a full path");
  });

  it('says there are no other folders instead of showing an empty list', async () => {
    mockFetch({ '/api/settings': () => jsonResponse({ ...withRoots, read_roots: [], read_roots_missing: [] }) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect(await screen.findByTestId('settings-read-roots-empty')).toBeTruthy();
    expect(screen.queryByRole('list', { name: 'Read-only folders' })).toBeNull();
  });

  const saveCalls = (calls: ReturnType<typeof mockFetch>) =>
    calls.filter((c) => c.method === 'POST' && c.url.endsWith('/api/settings'));

  // Killed by: frontend/src/components/SettingsModal.tsx :: const next = readRoots.filter((r) => r !== root);
  // Becomes: const next = readRoots.filter((r) => r === root);
  it('saves each folder change as it is made, and sends only the folders', async () => {
    const calls = mockFetch({
      '/api/settings': (_url, init) => {
        if (init?.method !== 'POST') return jsonResponse(withRoots);
        const sent = JSON.parse(init.body as string) as { read_roots: string[] };
        return jsonResponse({ ...withRoots, read_roots: sent.read_roots, read_roots_missing: [] });
      },
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    fireEvent.click(await screen.findByRole('button', { name: 'Remove ~/gone' }));
    await waitFor(() => expect(saveCalls(calls)).toHaveLength(1));
    expect(saveCalls(calls)[0].body).toEqual({ read_roots: ['/data/papers'] });

    fireEvent.change(screen.getByLabelText(/folder to add/i), { target: { value: '  ~/notes ' } });
    fireEvent.click(screen.getByRole('button', { name: /add folder/i }));
    await waitFor(() => expect(saveCalls(calls)).toHaveLength(2));
    expect(saveCalls(calls)[1].body).toEqual({ read_roots: ['/data/papers', '~/notes'] });
    expect(await screen.findByTestId('settings-read-roots-status')).toHaveAttribute('data-state', 'saved');
  });

  it('saves nothing when the folders were not touched and the window is closed', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(withRoots) });
    const onClose = vi.fn();
    render(<SettingsModal {...devModeOff} isOpen onClose={onClose} />);

    await screen.findByRole('list', { name: 'Read-only folders' });
    fireEvent.click(screen.getByTestId('settings-close'));

    expect(onClose).toHaveBeenCalled();
    expect(saveCalls(calls)).toEqual([]);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: restore: setReadRoots,
  // Becomes: restore: () => {},
  it("shows the server's reason when it refuses a folder, and takes the folder back off the list", async () => {
    const detail = "read_roots: '/nope' is not an existing folder";
    mockFetch({
      '/api/settings': (_url, init) =>
        init?.method === 'POST' ? jsonResponse({ detail }, false, 400) : jsonResponse(withRoots),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    fireEvent.change(await screen.findByLabelText(/folder to add/i), { target: { value: '/nope' } });
    fireEvent.click(screen.getByRole('button', { name: /add folder/i }));

    // Said as a sentence: the Core's words, with the full stop it left off (#1436).
    const status = await screen.findByTestId('settings-read-roots-status');
    await waitFor(() => expect(status).toHaveAttribute('data-state', 'error'));
    expect(status).toHaveTextContent(`${detail}. ${en.settings.autosave.restored}`);
    const list = screen.getByRole('list', { name: 'Read-only folders' });
    expect(Array.from(list.querySelectorAll('li')).map((r) => r.textContent)).toEqual([
      '/data/papers',
      '~/gonenot usable',
    ]);
  });
});

describe('SettingsModal saves as you go', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('has no Save button to press: the footer only closes', async () => {
    mockFetch({});
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('settings-close');

    expect(screen.queryByRole('button', { name: /save/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /check connection/i })).toBeNull();
    expect(screen.getByTestId('settings-close')).toHaveTextContent(en.settings.footer.close);
  });

});

describe('SettingsModal external tool servers', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('shows the tool server section, read from /api/mcp/servers', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: <McpServersSection />
    // Becomes:
    mockFetch({ '/api/mcp/servers': () => jsonResponse({ config_path: '/c/mcp.json', servers: [] }) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-mcp');
    expect(section).toHaveTextContent('External tools (MCP)');
    expect(await screen.findByTestId('settings-mcp-empty')).toHaveTextContent(
      'No tool servers are connected yet.',
    );
  });
});

describe('SettingsModal failures in plain words (#1436)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('confirm', vi.fn(() => true));
  });
  afterEach(() => vi.unstubAllGlobals());

  /** The two answers a reader must never see as text: no answer at all, and a 500 with no JSON. */
  const faults: Array<[string, Handler]> = [
    ['a rejected fetch', () => Promise.reject(new TypeError('Failed to fetch'))],
    [
      'a bodyless 500',
      () => ({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => {
          throw new SyntaxError('Unexpected token \'I\', "Internal S"... is not valid JSON');
        },
      }),
    ],
  ];

  /** Only the named request fails, and only for `method`; every other request answers as usual. */
  const failOnly = (path: string, method: string, fault: Handler): Handler => (url, init) => {
    if ((init?.method ?? 'GET') === method) return fault(url, init);
    if (path === '/api/settings') return jsonResponse(baseSettings);
    return jsonResponse({});
  };

  it.each([
    ...faults,
  ])('says the settings could not be loaded after %s', async (_label, fault) => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, copyRef.current.settings.failure.load)
    // Becomes: String(err)
    mockFetch({ '/api/settings': failOnly('/api/settings', 'GET', fault) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const feedback = await screen.findByTestId('settings-feedback');
    expect(feedback).toHaveTextContent(SETTINGS_FAILURE.load);
    expectPlain(feedback.textContent);
  });

  it.each([
    ...faults,
  ])('says saving went wrong after %s, under the field, and puts the saved value back', async (_label, fault) => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: `${plainFailure(err, copyRef.current.settings.failure.save)} ${copyRef.current.settings.autosave.restored}`
    // Becomes: `${String(err)} ${copyRef.current.settings.autosave.restored}`
    // Killed by: frontend/src/lib/useAutoSave.ts :: opts.current.restore(savedNow());
    // Becomes:
    mockFetch({ '/api/settings': failOnly('/api/settings', 'POST', fault) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const folder = await screen.findByLabelText(en.settings.folders.addLabel);
    fireEvent.change(folder, { target: { value: '/tmp/notes' } });
    fireEvent.keyDown(folder, { key: 'Enter' });

    const status = await screen.findByTestId('settings-read-roots-status');
    await waitFor(() => expect(status).toHaveAttribute('data-state', 'error'));
    expect(status).toHaveTextContent(SETTINGS_FAILURE.save);
    expect(status).toHaveTextContent(en.settings.autosave.restored);
    expectPlain(status.textContent);
    // The list shows what the app is still using, not the folder that was refused.
    expect(screen.getByTestId('settings-read-roots')).not.toHaveTextContent('/tmp/notes');
    // Said under the field, not in the banner at the top.
    expect(screen.queryByTestId('settings-feedback')).toBeNull();
  });

  it.each([
    ...faults,
  ])('says the clones could not be loaded after %s', async (_label, fault) => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: setPersonaLoadError(coreReason(err) ?? '');
    // Becomes: setPersonaLoadError(String(err));
    mockFetch({ '/api/clones': fault });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const alert = await screen.findByText(/could not load clones/i);
    expect(alert).toHaveTextContent(PERSONA_EDITOR_COPY.loadFailed);
    expectPlain(alert.textContent);
  });

  it("shows the Core's own reason when it gives one, in place of the fixed sentence", async () => {
    // Killed by: frontend/src/lib/coreFailure.ts :: return coreReason(err) ?? asSentence(fallback);
    // Becomes: return asSentence(fallback);
    mockFetch({
      '/api/settings': failOnly('/api/settings', 'GET', () =>
        jsonResponse({ detail: 'The settings file is not readable' }, false, 500),
      ),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const feedback = await screen.findByTestId('settings-feedback');
    expect(feedback).toHaveTextContent('The settings file is not readable.');
    expect(feedback).not.toHaveTextContent(SETTINGS_FAILURE.load);
  });

  it("keeps the Core's reason for the clones when it gives one", async () => {
    // Killed by: frontend/src/lib/personasApi.ts :: if (!res.ok) throw await failureOf(res);
    // Becomes: if (!res.ok) throw new Error(`HTTP ${res.status}`);
    mockFetch({ '/api/clones': () => jsonResponse({ detail: 'The clones folder is missing.' }, false, 500) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const alert = await screen.findByText(/could not load clones/i);
    expect(alert).toHaveTextContent(fmt(PERSONA_EDITOR_COPY.loadFailedBecause, { reason: 'The clones folder is missing.' }));
  });
});

describe('SettingsModal tabs', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('switches visible sections when tabs are clicked', async () => {
    mockFetch({});
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByTestId('settings-tabs-bar');
    expect(screen.getByTestId('settings-tab-llm')).toBeTruthy();

    fireEvent.click(screen.getByTestId('settings-tab-llm'));
    expect(await screen.findByTestId('settings-connections')).toBeTruthy();
    expect(screen.getByTestId('settings-default-models')).toBeTruthy();
    expect(screen.queryByTestId('settings-read-roots')).toBeNull();
  });

  it('opens on the tab a link asks for, each time it opens', async () => {
    mockFetch({});
    const props = { ...devModeOff, onClose: () => {}, initialTab: 'usage' as const };
    const { rerender } = render(<SettingsModal {...props} isOpen />);

    expect(await screen.findByTestId('settings-usage')).toBeTruthy();
    expect(screen.queryByTestId('settings-connections')).toBeNull();

    // The user moves away, closes, and follows the link again: Usage, not where they left it.
    fireEvent.click(screen.getByTestId('settings-tab-all'));
    rerender(<SettingsModal {...props} isOpen={false} />);
    rerender(<SettingsModal {...props} isOpen />);
    await waitFor(() => expect(screen.queryByTestId('settings-connections')).toBeNull());
    expect(screen.getByTestId('settings-usage')).toBeTruthy();
  });

  it('prevents tabs bar and header from shrinking and hides scrollbars (#1644)', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: scrollbar-hide shrink-0
    // Becomes: scrollbar-none
    mockFetch({});
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const header = await screen.findByTestId('settings-header');
    expect(header.className).toContain('shrink-0');

    const tabsBar = await screen.findByTestId('settings-tabs-bar');
    expect(tabsBar.className).toContain('shrink-0');
    expect(tabsBar.className).toContain('scrollbar-hide');

    const allTab = screen.getByTestId('settings-tab-all');
    expect(allTab.className).toContain('shrink-0');
  });
});

describe('SettingsModal Remote GPU Worker', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('renders remote GPU card and connects successfully', async () => {
    const mockTunnel = {
      host: 'dell',
      connected: true,
      mappings: [
        { service_name: 'comfyui', remote_port: 8188, local_port: 8188 },
        { service_name: 'ollama', remote_port: 11434, local_port: 11434 },
      ],
      gpu: {
        name: 'NVIDIA GeForce RTX 5070 Ti',
        total_mb: 16303,
        used_mb: 1024,
        driver: '610.88',
      },
    };

    const calls = mockFetch({
      '/api/settings/remote-gpu/connect': () =>
        jsonResponse({
          status: 'ok',
          connected: true,
          tunnel: mockTunnel,
          applied_changes: { picture_connection: 'remote-gpu-pictures' },
        }),
    });

    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);

    const card = await screen.findByTestId('settings-remote-gpu-card');
    expect(card).toBeTruthy();

    // No machine name is assumed: the field starts empty and says what it wants.
    const hostInput = screen.getByTestId('remote-gpu-host-input') as HTMLInputElement;
    expect(hostInput.value).toBe('');
    expect(hostInput.placeholder).toBe(en.settings.remoteGpu.hostPlaceholder);
    fireEvent.change(hostInput, { target: { value: 'gpu-box' } });

    const connectBtn = screen.getByTestId('remote-gpu-connect-button');
    fireEvent.click(connectBtn);

    await waitFor(() => {
      expect(
        calls.some((c) => c.method === 'POST' && c.url.includes('/api/settings/remote-gpu/connect')),
      ).toBe(true);
    });

    await waitFor(() => {
      expect(card.innerHTML).toContain('NVIDIA GeForce RTX 5070 Ti');
    });
    expect(screen.getByTestId('remote-gpu-disconnect-button')).toBeTruthy();
    expect(screen.getByTestId('remote-gpu-status-info')).toHaveTextContent(
      'Reached: comfyui (port 8188), ollama (port 11434)',
    );
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: connectionsChangedHere(); // what the connect added
  // Becomes:
  it('reads the connections again after a connect, which adds the GPU computer as connections', async () => {
    const calls = mockFetch({
      '/api/settings/remote-gpu/connect': () =>
        jsonResponse({
          status: 'ok',
          connected: true,
          tunnel: { host: 'gpu-box', connected: true, mappings: [] },
          applied_changes: { picture_connection: 'remote-gpu-pictures' },
        }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);
    const reads = () => calls.filter((c) => c.method === 'GET' && c.url === '/api/connections').length;
    await waitFor(() => expect(reads()).toBeGreaterThan(0));
    const before = reads();

    fireEvent.change(await screen.findByTestId('remote-gpu-host-input'), { target: { value: 'gpu-box' } });
    fireEvent.click(screen.getByTestId('remote-gpu-connect-button'));

    await waitFor(() => expect(reads()).toBeGreaterThan(before));
    // The connection saved its own rows; the form sent no setting for it.
    expect(calls.filter((c) => c.method === 'POST' && c.url === '/api/settings')).toEqual([]);
  });

  it('keeps the LLM local on connect unless the box is ticked', async () => {
    const tunnel = { host: 'dell', connected: true, mappings: [] };
    const calls = mockFetch({
      '/api/settings/remote-gpu/connect': () =>
        jsonResponse({ status: 'ok', connected: true, tunnel, applied_changes: {} }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);
    const box = (await screen.findByTestId('remote-gpu-sync-llm')) as HTMLInputElement;
    expect(box.checked).toBe(false);
    fireEvent.change(screen.getByTestId('remote-gpu-host-input'), { target: { value: 'gpu-box' } });
    fireEvent.click(screen.getByTestId('remote-gpu-connect-button'));
    await waitFor(() => expect(calls.some((c) => c.url.includes('/remote-gpu/connect'))).toBe(true));
    const first = calls.find((c) => c.url.includes('/remote-gpu/connect'));
    expect((first?.body as { sync_llm: boolean }).sync_llm).toBe(false);
  });

  it('asks for the LLM too when the box is ticked, and says when the model is missing', async () => {
    const tunnel = { host: 'dell', connected: true, mappings: [] };
    const calls = mockFetch({
      '/api/settings/remote-gpu/connect': () =>
        jsonResponse({
          status: 'ok',
          connected: true,
          tunnel,
          applied_changes: {},
          llm_skipped: 'model_not_on_remote',
        }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);
    fireEvent.click(await screen.findByTestId('remote-gpu-sync-llm'));
    fireEvent.change(screen.getByTestId('remote-gpu-host-input'), { target: { value: 'gpu-box' } });
    fireEvent.click(screen.getByTestId('remote-gpu-connect-button'));
    await waitFor(() => expect(calls.some((c) => c.url.includes('/remote-gpu/connect'))).toBe(true));
    const first = calls.find((c) => c.url.includes('/remote-gpu/connect'));
    expect((first?.body as { sync_llm: boolean }).sync_llm).toBe(true);
    expect(await screen.findByText(new RegExp(en.settings.remoteGpu.llmModelMissing))).toBeTruthy();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: if (gpuData.restored_settings) connectionsChangedHere();
  // Becomes:
  it('reads the connections again when a dead tunnel removed the ones it had added', async () => {
    // The status route first: '/api/settings' also matches its URL.
    const calls = mockFetch({
      '/api/settings/remote-gpu/status': () =>
        jsonResponse({
          host: 'dell',
          connected: false,
          restored_settings: { picture_connection: 'remote-gpu-pictures' },
        }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);
    await screen.findByDisplayValue('dell');
    await waitFor(() =>
      expect(calls.filter((c) => c.method === 'GET' && c.url === '/api/connections').length).toBeGreaterThan(1),
    );
  });

  it('disconnects remote GPU tunnel and restores state', async () => {
    const calls = mockFetch({
      '/api/settings/remote-gpu/status': () =>
        jsonResponse({
          host: 'dell',
          connected: true,
          gpu: { name: 'NVIDIA GeForce RTX 5070 Ti', total_mb: 16303, used_mb: 1024, driver: '610.88' },
        }),
      '/api/settings/remote-gpu/disconnect': () =>
        jsonResponse({
          status: 'ok',
          connected: false,
          restored_settings: { picture_connection: 'remote-gpu-pictures' },
        }),
    });

    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);

    const disconnectBtn = await screen.findByTestId('remote-gpu-disconnect-button');
    fireEvent.click(disconnectBtn);

    await waitFor(() => {
      expect(
        calls.some((c) => c.method === 'POST' && c.url.includes('/api/settings/remote-gpu/disconnect')),
      ).toBe(true);
    });

    expect(await screen.findByTestId('remote-gpu-connect-button')).toBeTruthy();
  });

  it('provides quick preset buttons for 1-click filling remote host', async () => {
    mockFetch({
      '/api/settings/remote-gpu/hosts': () => jsonResponse({ hosts: ['box-a', 'box-b'] }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);

    const hostInput = (await screen.findByTestId('remote-gpu-host-input')) as HTMLInputElement;
    expect(hostInput.value).toBe('');

    const boxAPreset = await screen.findByTestId('remote-gpu-preset-box-a');
    const boxBPreset = await screen.findByTestId('remote-gpu-preset-box-b');

    fireEvent.click(boxAPreset);
    expect(hostInput.value).toBe('box-a');

    fireEvent.click(boxBPreset);
    expect(hostInput.value).toBe('box-b');
  });

  it('displays honest routing badges when connected with remote flags', async () => {
    mockFetch({
      '/api/settings/remote-gpu/status': () =>
        jsonResponse({
          host: 'dell',
          connected: true,
          llm_on_remote: true,
          images_on_remote: true,
          gpu: { name: 'RTX 4090', total_mb: 24576, used_mb: 2048, driver: '550.0' },
          mappings: [{ service_name: 'ollama', remote_port: 11434, local_port: 11435 }],
        }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);

    const routingStatus = await screen.findByTestId('remote-gpu-routing-status');
    expect(routingStatus).toBeTruthy();
    expect(routingStatus.className).not.toContain('bg-gradient');
    expect(routingStatus.textContent).not.toContain('──(부재 시 자동 백업)──▶');

    const routingLlm = screen.getByTestId('remote-gpu-routing-llm');
    expect(routingLlm.textContent).toContain('dell');

    const routingImages = screen.getByTestId('remote-gpu-routing-images');
    expect(routingImages.textContent).toContain('dell');
  });

  it('reports connection error cleanly on card with raw error in disclosure', async () => {
    mockFetch({
      '/api/settings/remote-gpu/connect': () =>
        jsonResponse(
          {
            status: 'error',
            error: 'ssh: connect to host badhost port 22: Connection refused',
          },
          false,
          500,
        ),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);

    const hostInput = await screen.findByTestId('remote-gpu-host-input');
    fireEvent.change(hostInput, { target: { value: 'badhost' } });
    fireEvent.click(screen.getByTestId('remote-gpu-connect-button'));

    const errorEl = await screen.findByTestId('remote-gpu-connect-error');
    expect(errorEl.textContent).toContain('badhost');
    expect(errorEl.textContent).toContain('Connection refused');

    const details = errorEl.querySelector('details');
    expect(details).toBeTruthy();
    expect(details?.textContent).toContain('ssh: connect to host badhost port 22: Connection refused');
  });
});

/** The usage offer in "All" reads its own report; a save in Settings → Usage must still reach it. */

describe('SettingsModal usage offer', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });
  afterEach(() => vi.unstubAllGlobals());

  const usageReport = (limits: Record<string, number | null>) => ({
    checked_at: '2026-09-26T10:00:00Z',
    windows: Object.entries(limits).map(([window, limit]) => ({ window, used: 0, limit, available_again_at: null })),
    limits,
    env_overrides: [],
    providers: [],
  });
  const noLimits = { per_10_minutes: null, per_5_hours: null, per_week: null };
  const light = { per_10_minutes: 150_000, per_5_hours: 500_000, per_week: 3_000_000 };

  it('hides the offer as soon as a limit is saved, without reopening Settings', async () => {
    const calls = mockFetch({
      '/api/connections': () =>
        jsonResponse({
          connections: [
            {
              id: 'openai', kind: 'openai', label: 'OpenAI', base_url: null, key_set: true, key_masked: 'sk-…abcd',
              source: 'settings', paid: true, status: 'connected', detail: null, model_count: 3,
            },
          ],
          kinds: [],
        }),
      '/api/usage/limits': () => jsonResponse(usageReport(light)),
      '/api/usage': () => jsonResponse(usageReport(noLimits)),
    });

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect(await screen.findByTestId('settings-usage-offer')).toBeTruthy();
    fireEvent.click(await screen.findByTestId('settings-usage-preset-light'));

    await waitFor(() =>
      expect(screen.getByTestId('settings-usage-save-status')).toHaveAttribute('data-state', 'saved'),
    );
    expect(calls.some((c) => c.method === 'PUT' && c.url === '/api/usage/limits')).toBe(true);
    expect(screen.queryByTestId('settings-usage-offer')).toBeNull();
  });

  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx :: data.connections.some((c) => c.paid) && paidOffer
  // Becomes: paidOffer
  it('offers no limit while every connection is free', async () => {
    mockFetch({
      '/api/connections': () =>
        jsonResponse({
          connections: [
            {
              id: 'ollama', kind: 'ollama', label: 'Ollama', base_url: 'http://127.0.0.1:11434', key_set: false,
              key_masked: null, source: 'settings', paid: false, status: 'connected', detail: null, model_count: 2,
            },
          ],
          kinds: [],
        }),
      '/api/usage': () => jsonResponse(usageReport(noLimits)),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByTestId('connection-row-ollama');
    // The Usage section's own read is out too; give the offer's read the same chance.
    await screen.findByTestId('settings-usage');
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByTestId('settings-usage-offer')).toBeNull();
  });
});

describe('SettingsModal unreadable settings file (#1860)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it('says in plain English that the unreadable settings file was kept beside the new one', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: {currentSettings?.settings_set_aside && (
    // Becomes: {false && (
    mockFetch({ '/api/settings': () => jsonResponse({ ...baseSettings, settings_set_aside: true }) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const notice = await screen.findByTestId('settings-set-aside');
    expect(notice.textContent).toBe(en.settings.setAside.notice);
    expect(notice.textContent).toMatch(/kept that file unchanged beside the new one/);
    expectPlain(notice.textContent);
  });

  it('says it in Korean, in 합니다체, when the screens are in Korean', async () => {
    // Killed by: frontend/src/i18n/locales/ko/settings.json :: 표시되지 않습니다.
    // Becomes: 표시되지 않음.
    mockFetch({
      '/api/settings': () =>
        jsonResponse({ ...baseSettings, ui_language: 'ko', settings_set_aside: true }),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    const notice = await screen.findByTestId('settings-set-aside');
    await waitFor(() => expect(notice.textContent).toBe(ko.settings.setAside.notice));
    const copy = notice.textContent ?? '';
    expectPlain(copy);
    const sentences = copy.split(/(?<=\.)\s+/).filter(Boolean);
    expect(sentences.length).toBeGreaterThan(0);
    for (const sentence of sentences) expect(sentence).toMatch(/니다\.$/);
    // Only "API" is allowed through untranslated; everything else is Korean.
    expect(copy.replace(/API/g, '')).not.toMatch(/[A-Za-z]/);
  });

  it('shows no notice when the settings file was read', async () => {
    mockFetch({ '/api/settings': () => jsonResponse({ ...baseSettings, settings_set_aside: false }) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    // The language control sits right below the notice's place, so the section has rendered.
    await screen.findByText(en.settings.language.title);
    expect(screen.queryByTestId('settings-set-aside')).toBeNull();
  });
});

/** The pre-gateway single-provider form is gone, and so are its routes (model-gateway.md §3.7.1). */
describe('SettingsModal models', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('reads the connections and the model sets, and calls none of the deleted routes', async () => {
    const calls = mockFetch({});
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} initialTab="llm" />);

    expect(await screen.findByTestId('settings-connections-empty')).toBeTruthy();
    await screen.findByTestId('settings-default-models-no-connection');
    expect(calls.some((c) => c.url === '/api/connections')).toBe(true);
    expect(calls.some((c) => c.url.startsWith('/api/models?capability=chat'))).toBe(true);
    expect(calls.some((c) => c.url.startsWith('/api/models?capability=image'))).toBe(true);
    for (const gone of ['/api/settings/api-keys', '/api/models/catalog', '/api/settings/test', '/api/models/pull']) {
      expect(calls.some((c) => c.url.startsWith(gone))).toBe(false);
    }
    // No provider cards, key field or endpoint field from the old form.
    expect(screen.queryByTestId('api-key-input')).toBeNull();
    expect(screen.queryByTestId('settings-model-select')).toBeNull();
    expect(screen.queryByRole('button', { name: /^Gemini/ })).toBeNull();
  });
});
