import React, { useMemo } from 'react';
import { Bot, Check, ChevronLeft, MessageSquare, Pencil, Plus, Trash2, X } from 'lucide-react';
import type { RoomSummary } from '../../types';
import { emptyCause } from '../../lib/emptyStates';
import { exactTimeLabel, relativeTimeLabel } from '../../lib/relativeTime';
import { fmt, plural, useCopy, useLocale, type Language, type Messages } from '../../i18n';
import { en } from '../../i18n/en';
import { useEscapeOwner } from '../../lib/escapePrecedence';
import { ConversationList as KitConversationList } from '../../ui-kit';
import type { ConversationListCopy, ConversationListIcons, UseKitEscape } from '../../ui-kit';

/**
 * Every word the conversation list shows, as this head words it, in `language`.
 *
 * The list itself is `ui-kit/rail/ConversationList.tsx` (#1158), which holds no words. The
 * sentences are the `conversationList` catalog's; the empty-state ones stay in `emptyStates`
 * (`lib/emptyStates.ts`), because the Agents section of the rail states the same fact and used
 * to state it differently (#1060). The kit asks for functions where a sentence carries a
 * value; they are filled here, from the catalog's templates.
 */
export const conversationListCopy = (t: Messages, language: Language): ConversationListCopy => {
  const c = t.conversationList;
  return {
    heading: c.heading,
    newLabel: c.newLabel,
    newTitle: c.newTitle,
    collapse: c.collapse,
    rename: c.rename,
    renameLabel: (name) => fmt(c.renameLabel, { name }),
    delete: c.delete,
    deleteLabel: (name) => fmt(c.deleteLabel, { name }),
    emptyCause: (modelConfigured, agentCount) => emptyCause(modelConfigured, agentCount, t.emptyStates),
    lastActive: (timestamp, now) => relativeTimeLabel(timestamp, now, language),
    lastActiveTitle: (timestamp) => exactTimeLabel(timestamp, language),
    editor: { ...c.editor },
    deleteDialog: {
      ...c.deleteDialog,
      heading: (name) => fmt(c.deleteDialog.heading, { name }),
      // "clones", not "agents", in `noAgents`: §3.2.2 R3 gives the product one word for the
      // thing the user talks to, and this sentence is one a user reads.
      body: (entries, who) => plural(c.deleteDialog.body, entries, { who }),
    },
    unreadableTitle: c.unreadableTitle,
    unreadableDeleteLabel: c.unreadableDeleteLabel,
  };
};

/** The conversation list's words in English, for callers outside a `LocaleProvider`. */
export const CONVERSATION_LIST_COPY: ConversationListCopy = conversationListCopy(en, 'en');

/** The conversation list's words in the screen's language, rebuilt only when it changes. */
export const useConversationListCopy = (): ConversationListCopy => {
  const t = useCopy();
  const { language } = useLocale();
  return useMemo(() => conversationListCopy(t, language), [t, language]);
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
   * signal is `/api/models`'s `defaults.deep`, the default conversation model. It used
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
export const ConversationList: React.FC<ConversationListProps> = (props) => {
  const copy = useConversationListCopy();
  return (
    <KitConversationList
      {...props}
      copy={copy}
      icons={CONVERSATION_LIST_ICONS}
      useEscape={CONVERSATION_LIST_ESCAPE}
    />
  );
};
