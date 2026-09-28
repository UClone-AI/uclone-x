import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SkillsSection } from './SkillsSection';
import type { SkillManifest, SkillProposal } from '../../types';
import { expectPlain } from '../../test/plainCopy';
import { en } from '../../i18n/en';
import { ko } from '../../i18n/ko';
import { fmt } from '../../i18n/format';
import { LocaleProvider } from '../../i18n';

/**
 * Settings › Skills: a clone's skill proposals, and a person's approve, turn down and revoke
 * (#1827). Each decision is refused from a window the Core did not open (#1589); the panel
 * then offers to open one. Every refusal shown is plain words.
 */

type RouteResponse = { ok: boolean; status: number; json: () => Promise<unknown> };

const jsonResponse = (body: unknown, ok = true, status = 200): RouteResponse => ({
  ok,
  status,
  json: async () => body,
});

const proposal = (overrides: Partial<SkillProposal> = {}): SkillProposal => ({
  name: 'tidy-notes',
  version: '0.1.0',
  description: 'When the notes are a mess.',
  requires_tools: ['file_read'],
  agent_id: 'clone-1',
  session_id: 's-1',
  proposed_at: '2026-09-28T10:00:00Z',
  instructions: '# Tidy Notes\n\n1. Read the notes.\n',
  current_version: null,
  diff: '',
  digest: 'd1gest',
  ...overrides,
});

const skill = (overrides: Partial<SkillManifest> = {}): SkillManifest => ({
  name: 'tidy-notes',
  description: 'When the notes are a mess.',
  version: '0.1.0',
  author: 'agent:clone-1',
  origin: 'synthesized',
  status: 'active',
  isolation_level: 'none',
  content_sha256: 'abc123abc123abc123',
  scripts: [],
  tags: ['proposed'],
  approved_by: 'human:settings',
  approved_at: '2026-09-28T10:00:00Z',
  audit_report: {
    skill_name: 'tidy-notes',
    is_safe: true,
    recommendation: 'approve',
    risk_score: 0,
    detected_risks: [],
    auditor_version: '1',
    content_sha256: 'abc123abc123abc123',
  },
  shipped: false,
  ...overrides,
});

const payload = (skills: SkillManifest[], proposals: SkillProposal[]) => ({
  skills,
  proposals,
  summary: {
    total_skills: skills.length,
    active_count: skills.length,
    pending_count: 0,
    quarantined_count: 0,
  },
});

/** Routes by "METHOD path"; GET /api/skills answers `listing`, an unrouted call 599. */
const mockFetch = (
  listing: () => unknown,
  routes: Record<string, () => RouteResponse> = {},
) => {
  const calls: Array<{ url: string; method: string; body?: unknown }> = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    calls.push({ url, method, body: init?.body ? JSON.parse(init.body as string) : undefined });
    const handler = routes[`${method} ${url}`];
    if (handler) return handler();
    if (method === 'GET' && url === '/api/skills') return jsonResponse(listing());
    return jsonResponse({ detail: `unrouted ${method} ${url}` }, false, 599);
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
};

const PERSON_REFUSAL = 'Only a window UClone-X opened can make this decision.';

describe('Settings › Skills proposals (#1827)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('shows a new proposal with its clone, tools and full instructions', async () => {
    mockFetch(() => payload([], [proposal()]));
    render(<SkillsSection />);

    const card = await screen.findByTestId('skill-proposal');
    expect(card).toHaveTextContent('tidy-notes');
    expect(card).toHaveTextContent(fmt(en.skills.proposals.by, { clone: 'clone-1' }));
    expect(card).toHaveTextContent(fmt(en.skills.proposals.tools, { tools: 'file_read' }));
    expect(within(card).getByTestId('skill-proposal-instructions')).toHaveTextContent(
      '1. Read the notes.',
    );
    expect(within(card).queryByTestId('skill-proposal-diff')).toBeNull();
  });

  it('counts a proposal as waiting for review', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: {summary.pending_count + (skillsData?.proposals?.length ?? 0)}
    // Becomes: {summary.pending_count}
    mockFetch(() => payload([skill()], [proposal({ version: '0.1.1' })]));
    render(<SkillsSection />);

    expect(await screen.findByTestId('skills-waiting-count')).toHaveTextContent('1');
  });

  it('shows the changes from the version in use when a proposal replaces one', async () => {
    mockFetch(() =>
      payload(
        [skill()],
        [proposal({ version: '0.1.1', current_version: '0.1.0', diff: '-1. Old.\n+1. New.' })],
      ),
    );
    render(<SkillsSection />);

    const card = await screen.findByTestId('skill-proposal');
    expect(card).toHaveTextContent(fmt(en.skills.proposals.replaces, { version: '0.1.0' }));
    const diff = within(card).getByTestId('skill-proposal-diff');
    expect(diff).toHaveTextContent('-1. Old.');
    expect(diff).toHaveTextContent('+1. New.');
  });

  it('approves the version shown, says so and reads the catalogue again', async () => {
    let approved = false;
    const calls = mockFetch(() => (approved ? payload([skill()], []) : payload([], [proposal()])), {
      'POST /api/skills/tidy-notes/approve': () => {
        approved = true;
        return jsonResponse({ ok: true });
      },
    });
    render(<SkillsSection />);

    fireEvent.click(await screen.findByTestId('skill-proposal-approve'));

    const notice = await screen.findByTestId('skill-decision-notice');
    expect(notice).toHaveTextContent(fmt(en.skills.proposals.approved, { name: 'tidy-notes' }));
    expect(calls.find((c) => c.method === 'POST')?.body).toEqual({
      version: '0.1.0',
      seen_digest: 'd1gest',
    });
    await waitFor(() => expect(screen.queryByTestId('skill-proposal')).toBeNull());
  });

  it('turns a proposal down with its version', async () => {
    const calls = mockFetch(() => payload([], [proposal()]), {
      'POST /api/skills/tidy-notes/reject': () => jsonResponse({ ok: true }),
    });
    render(<SkillsSection />);

    fireEvent.click(await screen.findByTestId('skill-proposal-turn-down'));

    const notice = await screen.findByTestId('skill-decision-notice');
    expect(notice).toHaveTextContent(fmt(en.skills.proposals.turnedDown, { name: 'tidy-notes' }));
    expect(calls.find((c) => c.method === 'POST')?.body).toEqual({ version: '0.1.0' });
  });

  it('offers a confirmed window when this window may not decide, without the refusal text', async () => {
    // Killed by: frontend/src/components/settings/SkillProposals.tsx :: setUnconfirmed(notThisWindow);
    // Becomes: setUnconfirmed(false);
    const calls = mockFetch(() => payload([], [proposal()]), {
      'POST /api/skills/tidy-notes/approve': () =>
        jsonResponse({ detail: PERSON_REFUSAL }, false, 403),
      'POST /api/person/window': () => jsonResponse({ ok: true }),
    });
    render(<SkillsSection />);

    fireEvent.click(await screen.findByTestId('skill-proposal-approve'));

    const notice = await screen.findByTestId('skill-decision-notice');
    expect(notice).toHaveTextContent(en.skills.confirmWindow.unconfirmed);
    expectPlain(notice.textContent);
    fireEvent.click(within(notice).getByTestId('skill-open-confirmed-window'));
    await waitFor(() =>
      expect(screen.getByTestId('skill-decision-notice')).toHaveTextContent(
        en.skills.confirmWindow.opened,
      ),
    );
    expect(calls.some((c) => c.method === 'POST' && c.url === '/api/person/window')).toBe(true);
  });

  it('says the proposal changed since it was shown, and shows it again', async () => {
    // Killed by: frontend/src/components/settings/SkillProposals.tsx :: const changedSinceShown = err instanceof CoreFailure && err.status === 412;
    // Becomes: const changedSinceShown = false;
    let swapped = false;
    mockFetch(
      () =>
        payload(
          [],
          [
            swapped
              ? proposal({ instructions: '# Tidy Notes\n\n1. Send them away.\n', digest: 'other' })
              : proposal(),
          ],
        ),
      {
        'POST /api/skills/tidy-notes/approve': () => {
          swapped = true;
          return jsonResponse({ detail: 'This proposal changed after it was shown to you.' }, false, 412);
        },
      },
    );
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SkillsSection />
      </LocaleProvider>,
    );

    fireEvent.click(await screen.findByTestId('skill-proposal-approve'));

    const notice = await screen.findByTestId('skill-decision-notice');
    expect(notice).toHaveTextContent(ko.skills.proposals.changed);
    expectPlain(notice.textContent);
    expect(notice).not.toHaveTextContent('412');
    expect(screen.queryByTestId('skill-open-confirmed-window')).toBeNull();
    await waitFor(() =>
      expect(screen.getByTestId('skill-proposal-instructions')).toHaveTextContent('1. Send them away.'),
    );
  });

  it("shows the Core's plain refusal, and a fixed sentence when it gave none", async () => {
    let answer = jsonResponse({ detail: 'The safety check did not pass this skill.' }, false, 409);
    mockFetch(() => payload([], [proposal()]), {
      'POST /api/skills/tidy-notes/approve': () => answer,
    });
    render(<SkillsSection />);

    fireEvent.click(await screen.findByTestId('skill-proposal-approve'));
    const notice = await screen.findByTestId('skill-decision-notice');
    await waitFor(() => expect(notice).toHaveTextContent('The safety check did not pass this skill.'));
    expect(screen.queryByTestId('skill-open-confirmed-window')).toBeNull();

    answer = { ok: false, status: 500, json: async () => { throw new SyntaxError('Unexpected token <'); } };
    fireEvent.click(await screen.findByTestId('skill-proposal-approve'));
    await waitFor(() =>
      expect(screen.getByTestId('skill-decision-notice')).toHaveTextContent(
        en.skills.proposals.approveFailed,
      ),
    );
    const shown = screen.getByTestId('skill-decision-notice').textContent;
    expectPlain(shown);
    expect(shown).not.toMatch(/SyntaxError|Unexpected token|500/);
  });

  it('offers Revoke only for an approved skill that did not ship', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: {onRevoke && activeSkill.status === 'active' && !activeSkill.shipped && (
    // Becomes: {onRevoke && activeSkill.status === 'active' && (
    const calls = mockFetch(() => payload([skill({ shipped: true, name: 'avatar' })], []));
    const { unmount } = render(<SkillsSection />);
    await screen.findByText('avatar', { selector: 'h4' });
    expect(screen.queryByTestId('skill-revoke')).toBeNull();
    unmount();

    vi.unstubAllGlobals();
    const revokeCalls = mockFetch(() => payload([skill()], []), {
      'POST /api/skills/tidy-notes/revoke': () => jsonResponse({ ok: true }),
    });
    render(<SkillsSection />);
    fireEvent.click(await screen.findByTestId('skill-revoke'));
    const notice = await screen.findByTestId('skill-decision-notice');
    expect(notice).toHaveTextContent(fmt(en.skills.revoke.done, { name: 'tidy-notes' }));
    expect(revokeCalls.some((c) => c.method === 'POST' && c.url === '/api/skills/tidy-notes/revoke')).toBe(true);
    expect(calls.every((c) => c.method === 'GET')).toBe(true);
  });

  it('speaks Korean in a Korean window', async () => {
    mockFetch(() => payload([], [proposal()]), {
      'POST /api/skills/tidy-notes/approve': () =>
        jsonResponse({ detail: PERSON_REFUSAL }, false, 403),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SkillsSection />
      </LocaleProvider>,
    );

    const card = await screen.findByTestId('skill-proposal');
    expect(card).toHaveTextContent(ko.skills.proposals.approve);
    fireEvent.click(within(card).getByTestId('skill-proposal-approve'));
    const notice = await screen.findByTestId('skill-decision-notice');
    expect(notice).toHaveTextContent(ko.skills.confirmWindow.unconfirmed);
    expect(notice).toHaveTextContent(ko.skills.confirmWindow.open);
    expect(notice).not.toHaveTextContent(PERSON_REFUSAL);
  });
});
