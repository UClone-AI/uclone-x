import {
  Bot,
  ChevronDown,
  ChevronRight,
  MoreHorizontal,
  Pin,
  Plus,
  Settings,
  SlidersHorizontal,
} from 'lucide-react';
import { AgentInfo, PersonaInfo, RoomSummary } from '../../types';
import {
  CONVERSATION_LIST_COPY,
  CONVERSATION_LIST_ESCAPE,
  CONVERSATION_LIST_ICONS,
} from '../rooms/ConversationList';
import { personaAvatarUrl } from '../../lib/personaAvatar';
import { Rail } from '../../ui-kit';
import type { RailCopy, RailIcons, RailProps } from '../../ui-kit';

/**
 * The workspace rail, as this head shows it.
 *
 * The rail itself is `ui-kit/rail/Rail.tsx` (#1063 D, #1158): props-only, with no store, no
 * API, no words and no icon import, so another head can mount it. This file is this head's
 * side of it -- the words and the lucide glyphs -- bound to the kit component under the name
 * and props `App.tsx` has always mounted.
 *
 * Since #1059 it words no readouts: the step budget, the turn counter and the token total are
 * the dock's Resource surface, so the words for them are in `ResourceSummary` beside the
 * figures rather than here beside a rail that no longer draws them.
 */

/** The rail's rendered width in pixels. Stated beside its class in the kit; see there. */
export { RAIL_WIDTH_CLASS, RAIL_WIDTH_PX } from '../../ui-kit';

/** Every word the rail shows, as this head words it. */
export const RAIL_COPY: RailCopy = {
  clones: 'Clones',
  newClone: 'New clone',
  activeClones: (count) => `${count} active`,
  cloneLiveness: {
    // Idle and offline get no word: "running, nothing in progress" and "not running" are the
    // two ordinary states, and a rail that labels every row says nothing by labelling any.
    // The four that a glance must not miss are spelled out beside the name.
    word: (liveness) =>
      liveness === 'busy'
        ? 'working'
        : liveness === 'error'
          ? 'error'
          : liveness === 'terminated'
            ? 'stopped'
            : liveness === 'unknown'
              ? 'unknown'
              : null,
    title: (liveness, status) => {
      switch (liveness) {
        case 'offline':
          return 'Not running. It starts when you message it.';
        case 'idle':
          return 'Running, with nothing in progress.';
        case 'busy':
          return 'Working on something right now.';
        case 'error':
          return 'Its last run ended in an error.';
        case 'terminated':
          return 'Stopped. It starts again when you message it.';
        default:
          // Named, not hidden: a state this build has not been taught is a fact about the
          // screen, and reporting it as nothing would read as health.
          return `Running, in a state this screen does not know: ${status}.`;
      }
    },
  },
  expandLabel: (name) => `Show ${name}'s conversations`,
  collapseLabel: (name) => `Hide ${name}'s conversations`,
  expandTitle: 'Show conversations',
  collapseTitle: 'Hide conversations',
  inspectLabel: (name) => `View ${name}'s profile`,
  inspectTitle: 'View profile in dock',
  editLabel: (name) => `Edit ${name}'s settings`,
  editTitle: 'Edit settings in dock',
  pinLabel: (name) => `Pin ${name} to top`,
  unpinLabel: (name) => `Unpin ${name}`,
  pinTitle: 'Pin clone to top',
  unpinTitle: 'Unpin clone',
  moreLabel: (name) => `More actions for ${name}`,
  moreTitle: 'More actions',
  pinnedHeading: 'Pinned',
  cloneConversations: {
    heading: 'Conversations',
    none: 'In no conversation yet.',
    start: 'Start one',
    openLabel: (clone, title) => `Open ${title}, a conversation ${clone} is in`,
    startLabel: (clone) => `Start a conversation with ${clone}`,
    thread: 'Start thread',
    threadLabel: (clone) => `Start thread with ${clone}`,
  },
  conversations: {
    ...CONVERSATION_LIST_COPY,
    heading: 'Group Chats',
    newTitle: 'Start a new group chat',
  },
};

/** Every glyph the rail draws, from this head's icon library. */
export const RAIL_ICONS: RailIcons = {
  ...CONVERSATION_LIST_ICONS,
  clones: Bot,
  newClone: Plus,
  showDetails: ChevronRight,
  hideDetails: ChevronDown,
  inspectAgent: SlidersHorizontal,
  editAgent: Settings,
  pin: Pin,
  newThread: Plus,
  more: MoreHorizontal,
};

/** The kit rail's props in this head's types; `copy`, `icons` and `useEscape` are bound here. */
interface WorkspaceSidebarProps
  extends Omit<
    RailProps,
    'copy' | 'icons' | 'useEscape' | 'avatarSrc' | 'agents' | 'personas' | 'rooms'
  > {
  agents: AgentInfo[];
  personas?: PersonaInfo[];
  rooms: RoomSummary[];
}

export const WorkspaceSidebar: React.FC<WorkspaceSidebarProps> = (props) => (
  <Rail
    {...props}
    copy={RAIL_COPY}
    icons={RAIL_ICONS}
    // Bound here for the reason `copy` and `icons` are: where this head keeps a clone's
    // picture is this head's knowledge, and the kit reaches no API of its own.
    avatarSrc={personaAvatarUrl}
    useEscape={CONVERSATION_LIST_ESCAPE}
  />
);
