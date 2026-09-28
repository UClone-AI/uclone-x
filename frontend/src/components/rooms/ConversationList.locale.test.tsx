import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { ConversationList } from './ConversationList';
import { LocaleProvider } from '../../i18n';
import { ko } from '../../i18n/ko';
import { leftoverEnglish } from '../../test/leftoverEnglish';
import type { RoomSummary } from '../../types';

/** The conversation list's own controls and its delete dialog, in Korean (multilingual-ui.md step 3). */

const NAMESPACES = ['conversationList', 'emptyStates', 'time'] as const;

const room = (over: Partial<RoomSummary> = {}): RoomSummary => ({
  room_id: 'r1',
  title: '구조 검토',
  agent_ids: ['scout', 'critic'],
  human_ids: ['user'],
  message_count: 4,
  updated_at: new Date(Date.now() - 5 * 60_000).toISOString(),
  ...over,
});

const renderIn = (hint: string, rooms: RoomSummary[] = [room()]) =>
  render(
    <LocaleProvider hints={[hint]}>
      <ConversationList
        rooms={rooms}
        unreadableRoomIds={[]}
        currentRoomId={null}
        onSelectRoom={() => {}}
        onNewConversation={() => {}}
        onRenameRoom={async () => {}}
        onDeleteRoom={async () => {}}
        agentCount={2}
        modelConfigured
        onCollapse={() => {}}
      />
    </LocaleProvider>,
  );

describe('the conversation list in the chosen language', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, json: async () => ({ ui_language: 'system' }) })));
  });
  afterEach(() => vi.unstubAllGlobals());

  // Killed by: frontend/src/components/rooms/ConversationList.tsx :: body: (entries, who) => plural(c.deleteDialog.body, entries, { who }),
  // Becomes: body: (entries, who) => plural(en.conversationList.deleteDialog.body, entries, { who }),
  it('asks before a delete in Korean, naming what goes', () => {
    const english = renderIn('en-US');
    fireEvent.click(screen.getByRole('button', { name: 'Delete “구조 검토”' }));
    expect(leftoverEnglish(english.baseElement, NAMESPACES).length).toBeGreaterThan(8);
    english.unmount();

    const { baseElement } = renderIn('ko-KR');
    fireEvent.click(screen.getByRole('button', { name: '“구조 검토” 삭제' }));

    const dialog = screen.getByRole('alertdialog');
    expect(dialog).toHaveAccessibleName('“구조 검토”를 삭제하시겠습니까?');
    expect(dialog).toHaveTextContent('항목 4개');
    expect(dialog).toHaveTextContent('scout, critic');
    expect(screen.getByRole('button', { name: ko.conversationList.deleteDialog.confirm })).toBeInTheDocument();
    expect(leftoverEnglish(baseElement, NAMESPACES)).toEqual([]);
  });

  // Killed by: frontend/src/components/rooms/ConversationList.tsx :: renameLabel: (name) => fmt(c.renameLabel, { name }),
  // Becomes: renameLabel: (name) => fmt(en.conversationList.renameLabel, { name }),
  it('renames in Korean', () => {
    const { baseElement } = renderIn('ko-KR');
    fireEvent.click(screen.getByRole('button', { name: '“구조 검토” 이름 바꾸기' }));

    expect(screen.getByRole('textbox', { name: ko.conversationList.editor.titleLabel })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: ko.conversationList.editor.save })).toBeInTheDocument();
    expect(leftoverEnglish(baseElement, NAMESPACES)).toEqual([]);
  });

  // Killed by: frontend/src/i18n/locales/en/conversationList.json :: "one": "Its whole transcript ({count} entry)
  // Becomes: "one": "Its whole transcript ({count} entries)
  it('keeps the English delete sentence singular for one entry', () => {
    renderIn('en-US', [room({ message_count: 1, agent_ids: [] })]);
    fireEvent.click(screen.getByRole('button', { name: 'Delete “구조 검토”' }));
    expect(screen.getByRole('alertdialog')).toHaveTextContent('Its whole transcript (1 entry) and its record of who was in it (no clones)');
  });
});
