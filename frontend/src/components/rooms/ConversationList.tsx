import React from 'react';
import { Bot, Check, ChevronLeft, MessageSquare, Pencil, Plus, Trash2, X } from 'lucide-react';
import type { RoomSummary } from '../../types';
import { emptyCause } from '../../lib/emptyStates';
import { relativeTimeLabel } from '../../lib/relativeTime';
import { useEscapeOwner } from '../../lib/escapePrecedence';
import { ConversationList as KitConversationList } from '../../ui-kit';
import type { ConversationListCopy, ConversationListIcons, UseKitEscape } from '../../ui-kit';

/**
 * Every word the conversation list shows, as this head words it.
 *
 * The list itself is `ui-kit/rail/ConversationList.tsx` (#1158), which holds no words. The
 * empty-state sentences stay in `lib/emptyStates.ts`, because the Agents section of the rail
 * states the same fact and used to state it differently (#1060).
 */
export const CONVERSATION_LIST_COPY: ConversationListCopy = {
  heading: 'Conversations',
  newLabel: 'New',
  newTitle: 'Start a new conversation',
  collapse: 'Collapse sidebar',
  rename: 'Rename',
  renameLabel: (name) => `Rename “${name}”`,
  delete: 'Delete',
  deleteLabel: (name) => `Delete “${name}”`,
  emptyCause,
  lastActive: relativeTimeLabel,
  lastActiveTitle: (timestamp) => `Last active ${new Date(timestamp).toLocaleString()}`,
  editor: {
    titleLabel: 'Conversation title',
    save: 'Save title',
    cancel: 'Cancel rename',
  },
  deleteDialog: {
    heading: (name) => `Delete “${name}”?`,
    body: (entries, who) =>
      `Its whole transcript (${entries} ${entries === 1 ? 'entry' : 'entries'}) and its record ` +
      `of who was in it (${who}) will be removed, and any reply still being written in it is ` +
      'stopped. This cannot be undone.',
    // "clones", not "agents": §3.2.2 R3 gives the product one word for the thing the user
    // talks to, and this sentence is one a user reads.
    noAgents: 'no clones',
    cancel: 'Cancel',
    close: 'Close',
    confirm: 'Delete conversation',
    confirming: 'Deleting…',
    unreadableHeading: 'Delete the conversation that could not be read?',
    unreadableBody: 'Its saved copy will be removed. This cannot be undone.',
  },
  unreadableTitle: 'A conversation that could not be read',
  unreadableDeleteLabel: 'Delete the conversation that could not be read',
};

/**
 * Escape's registry, as this head keeps it (#1036).
 *
 * The kit's title editor and delete dialog are two of Escape's five owners but may not import
 * `lib/escapePrecedence.ts` themselves (#1158), so the binding happens here, beside the words
 * and the glyphs. `useEscapeOwner` accepts `'turn'` as well; the kit never asks for it.
 */
export const CONVERSATION_LIST_ESCAPE: UseKitEscape = useEscapeOwner;

/** Every glyph the conversation list draws, from this head's icon library. */
export const CONVERSATION_LIST_ICONS: ConversationListIcons = {
  heading: MessageSquare,
  newConversation: Plus,
  collapseRail: ChevronLeft,
  rename: Pencil,
  delete: Trash2,
  agent: Bot,
  saveTitle: Check,
  cancelRename: X,
};

interface ConversationListProps {
  rooms: RoomSummary[];
  /** Conversations the Core has and cannot read (#1440). */
  unreadableRoomIds: string[];
  currentRoomId: string | null;
  onSelectRoom: (roomId: string) => void;
  onNewConversation: () => void;
  /** Rename a conversation. Rejects with the Core's refusal, which the row shows. */
  onRenameRoom: (roomId: string, title: string) => Promise<void>;
  /** Delete a conversation. Rejects with the Core's refusal, which the dialog shows. */
  onDeleteRoom: (roomId: string) => Promise<void>;
  /** How many agents the runtime can offer. Zero is a cause, not an empty list. */
  agentCount: number;
  /**
   * Whether a model is configured at all. A different cause, with a different remedy.
   *
   * Three-valued, and it has to be: `null` is "the runtime has not answered yet", and
   * collapsing it onto `false` puts "No model is configured" on screen for the moment
   * before the first `/api/models` reply — a wrong cause is worse than no cause. The
   * signal is `/api/models`'s `current_model`, which is `settings["llm_model"]`. It used
   * to be `agents.length > 0`, which made this condition and `agentCount === 0` the same
   * one, so the no-agents sentence below could never be reached.
   */
  modelConfigured: boolean | null;
  /** The testid the E2E suite selects the rail's list by. */
  listTestId?: string;
  /** Collapse the rail this list sits in, when it sits in one. */
  onCollapse?: () => void;
}

/** The kit's conversation list, bound to this head's words and icons. */
export const ConversationList: React.FC<ConversationListProps> = (props) => (
  <KitConversationList
    {...props}
    copy={CONVERSATION_LIST_COPY}
    icons={CONVERSATION_LIST_ICONS}
    useEscape={CONVERSATION_LIST_ESCAPE}
  />
);
