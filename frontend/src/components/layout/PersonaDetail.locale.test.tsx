import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PersonaDetail } from './PersonaDetail';
import { LocaleProvider } from '../../i18n';
import { ko } from '../../i18n/ko';
import { leftoverEnglish } from '../../test/leftoverEnglish';
import { makePersonaInfo } from '../../test/fixtures';

/** The persona card in Korean (multilingual-ui.md step 3). Temperature and Max tokens stay English (§5). */

const renderIn = (hint: string, persona = makePersonaInfo({ role: '정찰', description: '자료를 찾습니다.' })) =>
  render(
    <LocaleProvider hints={[hint]}>
      <PersonaDetail persona={persona} />
    </LocaleProvider>,
  );

describe('the persona card in the chosen language', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, json: async () => ({ ui_language: 'system' }) })));
  });
  afterEach(() => vi.unstubAllGlobals());

  // Killed by: frontend/src/components/layout/PersonaDetail.tsx :: const copy = useMemo(() => personaDetailCopy(t.personaDetail), [t]);
  // Becomes: const copy = useMemo(() => personaDetailCopy(en.personaDetail), [t]);
  it('writes every field and capability in Korean, keeping the English technical terms', () => {
    const english = renderIn('en-US');
    expect(leftoverEnglish(english.baseElement, ['personaDetail']).length).toBeGreaterThan(3);
    english.unmount();

    const { baseElement } = renderIn('ko-KR');
    expect(screen.getByText(ko.personaDetail.writeAccess.off)).toBeInTheDocument();
    expect(screen.getByText(ko.personaDetail.subagentAccess.off)).toBeInTheDocument();
    expect(screen.getByText('Temperature')).toBeInTheDocument();
    expect(leftoverEnglish(baseElement, ['personaDetail'])).toEqual([]);
  });

  // Killed by: frontend/src/components/layout/PersonaDetail.tsx :: noTools: copy.noTools,
  // Becomes: noTools: en.personaDetail.noTools,
  it('says in Korean that an empty tool list is no restriction', () => {
    const { baseElement } = renderIn(
      'ko-KR',
      makePersonaInfo({ allowed_tools: [], model_name: undefined, enable_write_tools: true, enable_subagent_tools: true }),
    );
    expect(screen.getByText(ko.personaDetail.noTools)).toBeInTheDocument();
    expect(screen.getByText(ko.personaDetail.writeAccess.on)).toBeInTheDocument();
    expect(screen.getByText(ko.personaDetail.subagentAccess.on)).toBeInTheDocument();
    expect(leftoverEnglish(baseElement, ['personaDetail'])).toEqual([]);
  });
});
