import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { WorkspaceSidebar } from './WorkspaceSidebar';
import { LocaleProvider } from '../../i18n';
import { ko } from '../../i18n/ko';
import { leftoverEnglish } from '../../test/leftoverEnglish';
import { makePersonaInfo } from '../../test/fixtures';
import type { RoomSummary } from '../../types';

/**
 * The rail in Korean, with no English word of its own left on it (multilingual-ui.md step 3).
 *
 * Every other rail case renders without a provider and so reads English. These render the rail
 * the way `main.tsx` does, once in each language, so the leftover check is shown to see the
 * English before it is trusted to find none.
 */

const NAMESPACES = ['rail', 'conversationList', 'emptyStates', 'time'] as const;

const HOUR = 3_600_000;
const rooms = (): RoomSummary[] => [
  {
    room_id: 'r-group',
    title: '분기 계획',
    agent_ids: ['scout', 'critic'],
    human_ids: ['user'],
    message_count: 5,
    updated_at: new Date(Date.now() - 3 * HOUR).toISOString(),
  },
  {
    room_id: 'r-solo',
    title: '정찰 메모',
    agent_ids: ['scout'],
    human_ids: ['user'],
    message_count: 2,
    updated_at: new Date(Date.now() - 50 * HOUR).toISOString(),
  },
];

const renderIn = (hint: string, over: Partial<React.ComponentProps<typeof WorkspaceSidebar>> = {}) =>
  render(
    <LocaleProvider hints={[hint]}>
      <WorkspaceSidebar
        personas={[
          makePersonaInfo({ name: 'scout', role: '정찰', description: '자료를 찾습니다.' }),
          makePersonaInfo({ name: 'critic', role: '비평', description: '검토합니다.' }),
        ]}
        pinnedCloneIds={['scout']}
        selectedAgent="scout"
        onSelectAgent={() => {}}
        onInspectAgent={() => {}}
        onEditAgent={() => {}}
        onTogglePinClone={() => {}}
        onNewClone={() => {}}
        rooms={rooms()}
        unreadableRoomIds={['r-broken']}
        currentRoomId={null}
        onSelectRoom={() => {}}
        onNewRoom={() => {}}
        onRenameRoom={async () => {}}
        onDeleteRoom={async () => {}}
        modelConfigured
        overlay={false}
        {...over}
      />
    </LocaleProvider>,
  );

describe('the rail in the chosen language', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, json: async () => ({ ui_language: 'system' }) })));
  });
  afterEach(() => vi.unstubAllGlobals());

  // Killed by: frontend/src/components/layout/WorkspaceSidebar.tsx :: const copy = useMemo(() => railCopy(t, language), [t, language]);
  // Becomes: const copy = useMemo(() => railCopy(en, language), [t, language]);
  it('writes the clones, the group chats and a clone menu in Korean, with no English left', () => {
    const english = renderIn('en-US');
    fireEvent.click(screen.getByTestId('clone-menu-scout'));
    // The check below must be able to see the rail's English at all.
    expect(leftoverEnglish(english.baseElement, NAMESPACES).length).toBeGreaterThan(15);
    english.unmount();

    const { baseElement } = renderIn('ko-KR');
    fireEvent.click(screen.getByTestId('clone-menu-scout'));

    expect(screen.getByText(ko.rail.clones)).toBeInTheDocument();
    expect(screen.getByText(ko.rail.groupChats.heading)).toBeInTheDocument();
    expect(screen.getByText(ko.rail.pinnedHeading)).toBeInTheDocument();
    expect(screen.getByRole('menu', { name: 'scout에 대한 추가 작업' })).toBeInTheDocument();
    expect(leftoverEnglish(baseElement, NAMESPACES)).toEqual([]);
  });

  // Killed by: frontend/src/components/rooms/ConversationList.tsx :: emptyCause: (modelConfigured, agentCount) => emptyCause(modelConfigured, agentCount, t.emptyStates),
  // Becomes: emptyCause: (modelConfigured, agentCount) => emptyCause(modelConfigured, agentCount),
  it('states an absence in Korean, the same sentence in both of its regions', () => {
    const { baseElement } = renderIn('ko-KR', {
      personas: [],
      rooms: [],
      unreadableRoomIds: [],
      modelConfigured: false,
    });

    expect(screen.getAllByText(ko.emptyStates.noModel)).toHaveLength(2);
    expect(leftoverEnglish(baseElement, NAMESPACES)).toEqual([]);
  });

  // Killed by: frontend/src/components/rooms/ConversationList.tsx :: lastActive: (timestamp, now) => relativeTimeLabel(timestamp, now, language),
  // Becomes: lastActive: (timestamp, now) => relativeTimeLabel(timestamp, now),
  it('says when a conversation was last active in Korean', () => {
    renderIn('ko-KR');
    const label = within(screen.getByTestId('conversation-r-group')).getByTestId('conversation-last-active-r-group');
    expect(label.textContent).toBe('3시간');
    expect(label.getAttribute('title')).toMatch(/^마지막 활동: \d{4}\. /);
  });
});
