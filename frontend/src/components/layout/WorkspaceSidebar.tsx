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
import { PersonaInfo, RoomSummary } from '../../types';
import { useMemo } from 'react';
import {
  CONVERSATION_LIST_ESCAPE,
  CONVERSATION_LIST_ICONS,
  conversationListCopy,
} from '../rooms/ConversationList';
import { fmt, useCopy, useLocale, type Language, type Messages } from '../../i18n';
import { cloneLabel } from '../../lib/cloneLabel';
import { en } from '../../i18n/en';
import { avatarUrlsOf, pictureOf } from '../../lib/avatarChoice';
import { Rail } from '../../ui-kit';
import type { RailCopy, RailIcons, RailProps } from '../../ui-kit';

/**
 * The workspace rail, as this head shows it.
 *
 * The rail itself is `ui-kit/rail/Rail.tsx` (#1063 D, #1158): props-only, with no store, no
 * API, no words and no icon import, so another head can mount it. This file is this head's
 * side of it -- the words (from the catalog, in the screen's language) and the lucide glyphs --
 * bound to the kit component under the name and props `App.tsx` has always mounted.
 *
 * Since #1059 it words no readouts: the step budget, the turn counter and the token total are
 * the dock's Resource surface, so the words for them are in `ResourceSummary` beside the
 * figures rather than here beside a rail that no longer draws them.
 */

/** The rail's rendered width in pixels. Stated beside its class in the kit; see there. */
export { RAIL_WIDTH_CLASS, RAIL_WIDTH_PX } from '../../ui-kit';

/**
 * Every word the rail shows, as this head words it, in `language`.
 *
 * The sentences are the `rail` catalog's; the conversation list's are its own adapter's
 * (`conversationListCopy`), which this nests. Where the kit asks for a function, the function
 * is filled here from the catalog's template.
 */
export const railCopy = (t: Messages, language: Language): RailCopy => {
  const r = t.rail;
  const conversations = conversationListCopy(t, language);
  return {
    clones: r.clones,
    newClone: r.newClone,
    expandLabel: (name) => fmt(r.expandLabel, { name }),
    collapseLabel: (name) => fmt(r.collapseLabel, { name }),
    expandTitle: r.expandTitle,
    collapseTitle: r.collapseTitle,
    inspectLabel: (name) => fmt(r.inspectLabel, { name }),
    inspectTitle: r.inspectTitle,
    editLabel: (name) => fmt(r.editLabel, { name }),
    editTitle: r.editTitle,
    pinLabel: (name) => fmt(r.pinLabel, { name }),
    unpinLabel: (name) => fmt(r.unpinLabel, { name }),
    pinTitle: r.pinTitle,
    unpinTitle: r.unpinTitle,
    moreLabel: (name) => fmt(r.moreLabel, { name }),
    moreTitle: r.moreTitle,
    pinnedHeading: r.pinnedHeading,
    cloneConversations: {
      heading: r.cloneConversations.heading,
      none: r.cloneConversations.none,
      start: r.cloneConversations.start,
      openLabel: (clone, title) => fmt(r.cloneConversations.openLabel, { clone, title }),
      startLabel: (clone) => fmt(r.cloneConversations.startLabel, { clone }),
      thread: r.cloneConversations.thread,
      threadLabel: (clone) => fmt(r.cloneConversations.threadLabel, { clone }),
    },
    conversations: {
      ...conversations,
      heading: r.groupChats.heading,
      newTitle: r.groupChats.newTitle,
    },
  };
};

/** The rail's words in English, for callers outside a `LocaleProvider`. */
export const RAIL_COPY: RailCopy = railCopy(en, 'en');

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
    'copy' | 'icons' | 'useEscape' | 'avatarSrc' | 'personas' | 'rooms'
  > {
  personas?: PersonaInfo[];
  rooms: RoomSummary[];
}

export const WorkspaceSidebar: React.FC<WorkspaceSidebarProps> = (props) => {
  const t = useCopy();
  const { language } = useLocale();
  const copy = useMemo(() => railCopy(t, language), [t, language]);
  const { personas } = props;
  // Each clone's listed `avatar_url`, which changes with the picture, so a picture chosen
  // anywhere shows here at the next read of the clone list.
  const avatarSrc = useMemo(() => {
    const urls = avatarUrlsOf(personas);
    return (cloneId: string) => pictureOf(cloneId, urls);
  }, [personas]);
  // Each clone labelled by its display name for this screen's language; the id stays `name`.
  const labelled = useMemo(
    () => personas?.map((p) => ({ ...p, label: cloneLabel(p, language) })),
    [personas, language],
  );
  return (
    <Rail
      {...props}
      personas={labelled}
      copy={copy}
      icons={RAIL_ICONS}
      // Bound here for the reason `copy` and `icons` are: where this head keeps a clone's
      // picture is this head's knowledge, and the kit reaches no API of its own.
      avatarSrc={avatarSrc}
      useEscape={CONVERSATION_LIST_ESCAPE}
    />
  );
};
