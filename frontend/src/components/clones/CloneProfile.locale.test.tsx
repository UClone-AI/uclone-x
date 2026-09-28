import type React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CloneProfile } from './CloneProfile';
import { LocaleProvider } from '../../i18n';
import { ko } from '../../i18n/ko';
import { leftoverEnglish } from '../../test/leftoverEnglish';
import { makePersonaInfo } from '../../test/fixtures';

/**
 * The dock's Clone surface in Korean (multilingual-ui.md §10). Its persona card moved in step 3,
 * so the header and buttons around it have to move too, or the one surface is half in each language.
 */

const renderIn = (hint: string, props: Partial<React.ComponentProps<typeof CloneProfile>> = {}) =>
  render(
    <LocaleProvider hints={[hint]}>
      <CloneProfile
        cloneId="reader"
        persona={makePersonaInfo({ role: '', description: '자료를 찾습니다.' })}
        onStartConversation={vi.fn()}
        onModeChange={vi.fn()}
        canWrite
        {...props}
      />
    </LocaleProvider>,
  );

describe('the Clone surface in the chosen language', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, json: async () => ({ ui_language: 'system' }) })));
  });
  afterEach(() => vi.unstubAllGlobals());

  // Killed by: frontend/src/components/clones/CloneProfile.tsx ::           {t.start}
  // Becomes:           Start a conversation
  it('writes the header, the buttons and the persona card in Korean', () => {
    const english = renderIn('en-US');
    expect(leftoverEnglish(english.baseElement, ['cloneProfile', 'personaDetail']).length).toBeGreaterThan(4);
    english.unmount();

    const { baseElement } = renderIn('ko-KR');
    expect(screen.getByTestId('clone-profile-role')).toHaveTextContent(ko.cloneProfile.noRole);
    expect(screen.getByTestId('clone-profile-start')).toHaveTextContent(ko.cloneProfile.start);
    expect(screen.getByTestId('clone-profile-start')).toHaveAttribute('aria-label', 'reader와(과) 대화 시작');
    expect(screen.getByTestId('clone-profile-edit')).toHaveTextContent(ko.cloneProfile.edit);
    expect(leftoverEnglish(baseElement, ['cloneProfile', 'personaDetail'])).toEqual([]);
  });

  // Killed by: frontend/src/components/clones/CloneProfile.tsx :: <span className="text-slate-300">{copy.rail.clones}</span>
  // Becomes: <span className="text-slate-300">Clones</span>
  it('points to the rail by the name the Korean rail shows', () => {
    const { baseElement } = renderIn('ko-KR', { cloneId: '', persona: undefined });
    const none = screen.getByTestId('clone-profile-none');
    expect(none).toHaveTextContent(`왼쪽 사이드바의 ${ko.rail.clones}에서`);
    expect(none.textContent).not.toContain('{clones}');
    expect(leftoverEnglish(baseElement, ['cloneProfile'])).toEqual([]);
  });

  it('says in Korean why a clone with no file shows no settings', () => {
    const { baseElement } = renderIn('ko-KR', { cloneId: 'ephemeral', persona: undefined });
    expect(screen.getByTestId('clone-profile-no-definition')).toHaveTextContent(ko.cloneProfile.noDefinition);
    expect(leftoverEnglish(baseElement, ['cloneProfile'])).toEqual([]);
  });
});
