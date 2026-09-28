import type React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { CloneSkillNotice } from './CloneSkillNotice';
import { SkillHiddenFromNotice, SkillsTab } from '../SkillsTab';
import { CloneProfile } from './CloneProfile';
import { makePersonaInfo } from '../../test/fixtures';
import { LocaleProvider } from '../../i18n';
import { en } from '../../i18n/en';
import { ko } from '../../i18n/ko';
import { leftoverEnglish } from '../../test/leftoverEnglish';
import type { SkillManifest, SkillsData } from '../../types';

/**
 * A skill a clone cannot use is not offered to it, and the screen says so (#1826).
 *
 * The runtime leaves a skill out of a clone's skill list when the clone lacks a tool the skill
 * requires. Without these notices the skill would vanish from that clone in silence. Both the
 * Settings Skills panel (per skill) and the Clone surface (per clone) name what is missing, in
 * the reader's language, with the remedy.
 */

const avatar = {
  name: 'avatar',
  description: 'Make a profile picture',
  version: '0.2.0',
  author: 'unknown',
  origin: 'synthesized',
  status: 'active',
  isolation_level: 'workspace',
  content_sha256: 'x',
  scripts: [],
  tags: [],
  approved_by: null,
  approved_at: null,
  requires_tools: ['generate_image', 'set_avatar'],
  hidden_from: [
    { persona: 'scout', missing_tools: ['generate_image'] },
    { persona: 'writer', missing_tools: ['generate_image', 'set_avatar'] },
  ],
  audit_report: {
    skill_name: 'avatar',
    is_safe: true,
    recommendation: 'approve',
    risk_score: 0,
    detected_risks: [],
    auditor_version: '0.1.0',
    content_sha256: 'x',
  },
} satisfies SkillManifest;

const skills: SkillsData = {
  skills: [avatar, { ...avatar, name: 'notes', requires_tools: [], hidden_from: [] }],
  summary: { total_skills: 2, active_count: 2, pending_count: 0, quarantined_count: 0 },
};

const inLocale = (hint: string, node: React.ReactNode) =>
  render(<LocaleProvider hints={[hint]}>{node}</LocaleProvider>);

describe('the notice that a skill is not offered to a clone', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => ({
        ok: true,
        status: 200,
        json: async () => (url === '/api/skills' ? skills : { ui_language: 'system' }),
      })),
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  // Killed by: frontend/src/components/clones/CloneSkillNotice.tsx :: const entry = (skill.hidden_from ?? []).find((h) => h.persona === cloneId);
  // Becomes: const entry = (skill.hidden_from ?? [])[0];
  it('lists, on the Clone surface, the skills that clone is not offered and the tools it lacks', async () => {
    inLocale('en-US', <CloneSkillNotice cloneId="writer" />);

    const notice = await screen.findByTestId('clone-skill-notice');
    expect(notice).toHaveTextContent(en.skills.requiresTools.cloneHeading);
    expect(notice).toHaveTextContent('avatar: needs generate_image, set_avatar');
    expect(notice).not.toHaveTextContent('notes');
    expect(notice).toHaveTextContent(en.skills.requiresTools.cloneRemedy);
  });

  // Killed by: frontend/src/components/clones/CloneSkillNotice.tsx :: if (hidden.length === 0) return null;
  // Becomes: (removed)
  it('says nothing for a clone that is offered every skill', async () => {
    inLocale('en-US', <CloneSkillNotice cloneId="artist" />);

    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/skills'));
    expect(screen.queryByTestId('clone-skill-notice')).toBeNull();
  });

  // Killed by: frontend/src/components/clones/CloneSkillNotice.tsx :: <strong className="text-slate-200">{t.cloneHeading}</strong>
  // Becomes: <strong className="text-slate-200">Skills this clone is not offered</strong>
  it('writes the Clone surface notice in Korean', async () => {
    const { baseElement } = inLocale('ko-KR', <CloneSkillNotice cloneId="scout" />);

    const notice = await screen.findByTestId('clone-skill-notice');
    expect(notice).toHaveTextContent(ko.skills.requiresTools.cloneHeading);
    expect(notice).toHaveTextContent('avatar: generate_image 도구가 필요합니다');
    expect(leftoverEnglish(baseElement, ['skills'])).toEqual([]);
  });

  // Killed by: frontend/src/components/SkillsTab.tsx :: {fmt(t.row, { clone: h.persona, tools: h.missing_tools.join(', ') })}
  // Becomes: {h.persona}
  it('lists, in Settings, each clone the skill is not offered to and what it lacks', () => {
    inLocale('en-US', <SkillHiddenFromNotice skill={avatar} />);

    const notice = screen.getByTestId('skill-hidden-from');
    expect(notice).toHaveTextContent(en.skills.requiresTools.heading);
    expect(notice).toHaveTextContent('scout: needs generate_image');
    expect(notice).toHaveTextContent('writer: needs generate_image, set_avatar');
    expect(notice).toHaveTextContent(en.skills.requiresTools.remedy);
  });

  // Killed by: frontend/src/components/SkillsTab.tsx :: <p className="text-slate-400">{t.remedy}</p>
  // Becomes: <p className="text-slate-400">A clone is offered a skill only when it has every tool the skill needs.</p>
  it('writes the Settings notice in Korean, and nothing for a skill every clone can use', () => {
    const { baseElement } = inLocale('ko-KR', <SkillHiddenFromNotice skill={avatar} />);

    const notice = screen.getByTestId('skill-hidden-from');
    expect(notice).toHaveTextContent(ko.skills.requiresTools.heading);
    expect(notice).toHaveTextContent('writer: generate_image, set_avatar 도구가 필요합니다');
    expect(notice).toHaveTextContent(ko.skills.requiresTools.remedy);
    expect(leftoverEnglish(baseElement, ['skills'])).toEqual([]);

    inLocale('ko-KR', <SkillHiddenFromNotice skill={skills.skills[1]} />);
    expect(screen.getAllByTestId('skill-hidden-from')).toHaveLength(1);
  });

  // Killed by: frontend/src/components/SkillsTab.tsx ::                 <SkillHiddenFromNotice skill={activeSkill} />
  // Becomes: (removed)
  it('shows the Settings notice in the details of the chosen skill', () => {
    inLocale('en-US', <SkillsTab skillsData={skills} onRefresh={vi.fn()} isLoading={false} />);

    expect(screen.getByTestId('skill-hidden-from')).toHaveTextContent('scout: needs generate_image');
  });

  // Killed by: frontend/src/components/clones/CloneProfile.tsx ::           <CloneSkillNotice cloneId={cloneId} />
  // Becomes: (removed)
  it('shows the clone notice on the Clone surface, under its settings', async () => {
    inLocale(
      'en-US',
      <CloneProfile
        cloneId="scout"
        persona={makePersonaInfo({ name: 'scout' })}
        onStartConversation={vi.fn()}
      />,
    );

    expect(await screen.findByTestId('clone-skill-notice')).toHaveTextContent(
      'avatar: needs generate_image',
    );
  });
});
