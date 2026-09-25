import { afterEach, describe, it, expect, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ConversationList } from './ConversationList';
import type { RoomSummary } from '../../types';

const renderList = (over: Partial<React.ComponentProps<typeof ConversationList>> = {}) =>
  render(
    <ConversationList
      rooms={[]}
      unreadableRoomIds={[]}
      currentRoomId={null}
      onSelectRoom={() => {}}
      onNewConversation={() => {}}
      onRenameRoom={async () => {}}
      onDeleteRoom={async () => {}}
      agentCount={2}
      modelConfigured
      {...over}
    />,
  );

const summary = (over: Partial<RoomSummary> = {}): RoomSummary => ({
  room_id: 'r1',
  title: 'Architecture triage',
  agent_ids: ['scout', 'critic'],
  human_ids: ['user'],
  message_count: 4,
  updated_at: '2026-09-14T00:00:00Z',
  ...over,
});

describe('ConversationList absences', () => {
  it('distinguishes no model, no clones, and no conversations', () => {
    const { unmount } = renderList({ modelConfigured: false, agentCount: 0 });
    expect(screen.getByTestId('conversations-empty-cause')).toHaveTextContent('No model');
    unmount();

    const second = renderList({ agentCount: 0 });
    expect(screen.getByTestId('conversations-empty-cause')).toHaveTextContent('No clones');
    second.unmount();

    renderList();
    expect(screen.getByTestId('conversations-empty-cause')).toHaveTextContent(
      'No conversations yet',
    );
  });

  it('does not accuse the runtime of having no model before it has answered', () => {
    // `null` is "not known yet". Collapsed onto `false` it puts "No model is configured"
    // on screen for the moment before the first `/api/models` reply -- a wrong cause,
    // which is worse than none.
    renderList({ modelConfigured: null, agentCount: 2 });

    expect(screen.getByTestId('conversations-empty-cause')).toHaveTextContent(
      'No conversations yet',
    );
  });
});

describe('ConversationList', () => {
  it('lists a conversation by its title and who is in it', () => {
    renderList({ rooms: [summary()] });

    expect(screen.getByTestId('conversation-r1')).toHaveTextContent('Architecture triage');
    expect(screen.getByTestId('conversation-r1')).toHaveTextContent('scout, critic');
  });

  it('offers a way to start the first conversation even with nothing in the list', () => {
    const onNewConversation = vi.fn();
    renderList({ onNewConversation });

    expect(screen.getByTestId('conversations-empty-cause')).toBeInTheDocument();
    // Visible, not merely mounted: the affordance that creates the first conversation
    // cannot be conditional on a conversation already existing.
    expect(screen.getByTestId('new-conversation-button')).toBeVisible();
    screen.getByTestId('new-conversation-button').click();

    expect(onNewConversation).toHaveBeenCalledTimes(1);
  });

  it('carries the list testid the end-to-end suite selects the rail by', () => {
    renderList({ listTestId: 'sessions-list' });

    expect(screen.getByTestId('sessions-list')).toBeInTheDocument();
  });
});

describe('ConversationList last-active wording (#1053)', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('says when each conversation was last active, in words', () => {
    vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] });
    const now = new Date(2026, 8, 19, 12, 0, 0);
    vi.setSystemTime(now);
    const yesterday = new Date(2026, 8, 18, 9, 0, 0);

    renderList({
      rooms: [
        summary({ room_id: 'recent', updated_at: new Date(now.getTime() - 5 * 60_000).toISOString() }),
        summary({ room_id: 'older', updated_at: yesterday.toISOString() }),
      ],
    });

    expect(screen.getByTestId('conversation-recent')).toHaveTextContent('5m');
    expect(screen.getByTestId('conversation-older')).toHaveTextContent('1d');
    // The exact moment is one hover away, for the reader who needs it.
    expect(screen.getByTestId('conversation-last-active-recent')).toHaveAttribute(
      'dateTime',
      new Date(now.getTime() - 5 * 60_000).toISOString(),
    );
  });

  it('re-reads the wording as time passes, rather than freezing it at first render', () => {
    vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] });
    const now = new Date(2026, 8, 19, 12, 0, 0);
    vi.setSystemTime(now);

    renderList({ rooms: [summary({ room_id: 'r1', updated_at: now.toISOString() })] });
    expect(screen.getByTestId('conversation-r1')).toHaveTextContent('now');

    act(() => {
      vi.advanceTimersByTime(3 * 60_000);
    });
    expect(screen.getByTestId('conversation-r1')).toHaveTextContent('3m');
  });

  it('renders the rows in the order the Core sent them', () => {
    // The Core orders `list_rooms` by recency; the head does not second-guess it, so a
    // second head on the same Core cannot disagree with this one about what came last.
    renderList({
      rooms: [
        summary({ room_id: 'b', updated_at: '2026-09-01T00:00:00Z' }),
        summary({ room_id: 'a', updated_at: '2026-09-19T00:00:00Z' }),
      ],
    });
    const ids = screen
      .getAllByRole('button')
      .map((el) => el.getAttribute('data-testid'))
      .filter((id) => id === 'conversation-a' || id === 'conversation-b');
    expect(ids).toEqual(['conversation-b', 'conversation-a']);
  });

  it('shows no wording at all for a stamp that is not a date', () => {
    renderList({ rooms: [summary({ room_id: 'r1', updated_at: 'garbled' })] });

    expect(screen.getByTestId('conversation-r1')).toHaveTextContent('Architecture triage');
    expect(screen.queryByTestId('conversation-last-active-r1')).toBeNull();
    expect(screen.getByTestId('conversation-r1')).not.toHaveTextContent('Invalid');
  });
});

/** The Core's own refusals, as `PATCH`/`DELETE /api/rooms/{room_id}` send them. */
const BLANK_TITLE_REFUSAL =
  'A room needs a title: it is how a person finds this conversation again, and an untitled room is a hex id in a list';
const MISSING_ROOM_REFUSAL =
  "No room 'r1' in the store: it has been deleted, or was never created. List the rooms to see the ones that exist.";

describe('ConversationList row actions (#1058)', () => {
  it('offers rename and delete on every row, revealed on hover and focus', () => {
    renderList({ rooms: [summary()] });

    // Real buttons, named for the row they act on, so a keyboard and a screen reader reach
    // them. Revealed on hover and focus to preserve space for conversation titles.
    for (const name of ['Rename “Architecture triage”', 'Delete “Architecture triage”']) {
      const control = screen.getByRole('button', { name });
      expect(control).not.toBeDisabled();
    }

    const rename = screen.getByRole('button', { name: 'Rename “Architecture triage”' });
    const container = rename.parentElement;
    expect(container?.className).toMatch(/(^|\s)opacity-0(\s|$)/);
    expect(container?.className).toMatch(/group-hover:opacity-100/);
    expect(container?.className).toMatch(/focus-within:opacity-100/);
  });

  it('renames in place and hands the Core the new title', async () => {
    const onRenameRoom = vi.fn(async () => {});
    renderList({ rooms: [summary()], onRenameRoom });

    fireEvent.click(screen.getByRole('button', { name: 'Rename “Architecture triage”' }));
    const input = screen.getByRole('textbox', { name: 'Conversation title' });
    expect(input).toHaveValue('Architecture triage');
    expect(input).toHaveFocus();

    fireEvent.change(input, { target: { value: 'Index tuning' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(onRenameRoom).toHaveBeenCalledWith('r1', 'Index tuning'));
    await waitFor(() => expect(screen.queryByRole('textbox')).toBeNull());
  });

  it('shows a refused rename in the Core`s own words and keeps the draft', async () => {
    const onRenameRoom = vi.fn(async () => {
      throw new Error(BLANK_TITLE_REFUSAL);
    });
    renderList({ rooms: [summary()], onRenameRoom });

    fireEvent.click(screen.getByRole('button', { name: 'Rename “Architecture triage”' }));
    const input = screen.getByRole('textbox', { name: 'Conversation title' });
    fireEvent.change(input, { target: { value: '   ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save title' }));

    // The head does not pre-empt the Core with a refusal of its own: the Core's says what a
    // title is *for*, which is the remedy.
    expect(await screen.findByRole('alert')).toHaveTextContent(BLANK_TITLE_REFUSAL);
    expect(onRenameRoom).toHaveBeenCalledWith('r1', '   ');
    expect(screen.getByRole('textbox', { name: 'Conversation title' })).toHaveValue('   ');
  });

  it('abandons a rename on Escape without asking the Core', () => {
    const onRenameRoom = vi.fn(async () => {});
    renderList({ rooms: [summary()], onRenameRoom });

    fireEvent.click(screen.getByRole('button', { name: 'Rename “Architecture triage”' }));
    fireEvent.keyDown(screen.getByRole('textbox', { name: 'Conversation title' }), {
      key: 'Escape',
    });

    expect(screen.queryByRole('textbox')).toBeNull();
    expect(onRenameRoom).not.toHaveBeenCalled();
    expect(screen.getByTestId('conversation-r1')).toHaveTextContent('Architecture triage');
  });

  it('confirms a delete in a dialog naming what will be lost, before deleting anything', () => {
    const onDeleteRoom = vi.fn(async () => {});
    renderList({ rooms: [summary()], onDeleteRoom });

    fireEvent.click(screen.getByRole('button', { name: 'Delete “Architecture triage”' }));

    const dialog = screen.getByRole('alertdialog');
    expect(dialog).toHaveAccessibleName('Delete “Architecture triage”?');
    expect(dialog).toHaveTextContent('4 entries');
    expect(dialog).toHaveTextContent('scout, critic');
    expect(dialog).toHaveTextContent('cannot be undone');
    // The safe choice holds the focus, so a stray Enter does not delete.
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus();
    expect(onDeleteRoom).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('alertdialog')).toBeNull();
    expect(onDeleteRoom).not.toHaveBeenCalled();
  });

  it('closes the dialog on Escape without deleting', () => {
    const onDeleteRoom = vi.fn(async () => {});
    renderList({ rooms: [summary()], onDeleteRoom });

    fireEvent.click(screen.getByRole('button', { name: 'Delete “Architecture triage”' }));
    fireEvent.keyDown(screen.getByRole('alertdialog'), { key: 'Escape' });

    expect(screen.queryByRole('alertdialog')).toBeNull();
    expect(onDeleteRoom).not.toHaveBeenCalled();
  });

  it('deletes on confirmation and closes the dialog', async () => {
    const onDeleteRoom = vi.fn(async () => {});
    renderList({ rooms: [summary()], onDeleteRoom });

    fireEvent.click(screen.getByRole('button', { name: 'Delete “Architecture triage”' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete conversation' }));

    await waitFor(() => expect(onDeleteRoom).toHaveBeenCalledWith('r1'));
    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull());
  });

  it('keeps the dialog open on a refusal and says the Core`s cause and remedy', async () => {
    const onDeleteRoom = vi.fn(async () => {
      throw new Error(MISSING_ROOM_REFUSAL);
    });
    const { rerender } = renderList({ rooms: [summary()], onDeleteRoom });

    fireEvent.click(screen.getByRole('button', { name: 'Delete “Architecture triage”' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete conversation' }));

    const dialog = screen.getByRole('alertdialog');
    expect(await screen.findByRole('alert')).toHaveTextContent(MISSING_ROOM_REFUSAL);
    expect(dialog).toBeInTheDocument();

    // The refusal is why the list is re-read, and the row it was opened from may be gone
    // by the time it is. The dialog belongs to the list, so it outlives the row and the
    // reason stays on screen until the reader dismisses it.
    rerender(
      <ConversationList
        rooms={[]}
        unreadableRoomIds={[]}
        currentRoomId={null}
        onSelectRoom={() => {}}
        onNewConversation={() => {}}
        onRenameRoom={async () => {}}
        onDeleteRoom={onDeleteRoom}
        agentCount={2}
        modelConfigured
      />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent(MISSING_ROOM_REFUSAL);
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.queryByRole('alertdialog')).toBeNull();
  });
});

describe('ConversationList: a conversation that could not be read (#1440)', () => {
  // "No conversations yet" would be false beside a row for one that exists.
  // Killed by: frontend/src/ui-kit/rail/ConversationList.tsx :: {rooms.length === 0 && unreadableRoomIds.length === 0 ? (
  // Becomes: {rooms.length === 0 ? (
  it('is a row with a Delete, and the list is not said to be empty', async () => {
    const onDeleteRoom = vi.fn().mockResolvedValue(undefined);
    renderList({ unreadableRoomIds: ['r_bad'], onDeleteRoom });

    expect(screen.queryByTestId('conversations-empty-cause')).toBeNull();
    expect(screen.getByTestId('unreadable-conversation-r_bad')).toHaveTextContent(
      'A conversation that could not be read',
    );
    // Not a control that opens it: there is nothing the Core will read.
    expect(screen.queryByTestId('conversation-r_bad')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Delete the conversation that could not be read' }));
    expect(screen.getByRole('alertdialog')).toHaveTextContent(
      'Delete the conversation that could not be read?Its saved copy will be removed. This cannot be undone.',
    );
    fireEvent.click(screen.getByTestId('confirm-delete-conversation'));
    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull());
    expect(onDeleteRoom).toHaveBeenCalledWith('r_bad');
  });
});
