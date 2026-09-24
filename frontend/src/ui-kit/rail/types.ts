import type { KitIcon } from '../kit';

/**
 * The shapes the rail reads, and only the fields it reads.
 *
 * The kit may not import the head's `types.ts` (the boundary in `ui-kit.test.ts`), so these are
 * declared here. They are structural: the head's `RoomSummary`, `SessionSummary`, `AgentInfo`
 * and `PersonaInfo` carry more fields and are accepted as they are. The field names are the
 * wire's, so a head passes what its API returned without mapping it.
 */

export interface RailRoom {
  room_id: string;
  title: string;
  agent_ids: string[];
  message_count: number;
  updated_at: string;
}

export interface RailAgent {
  id: string;
  label: string;
  role: string;
  /**
   * The running instance's state, as `/api/agents` sends it.
   *
   * It is `AgentState` (`src/uclone_x/agent/models.py`) stringified -- `IDLE`, `INGESTING`,
   * `REASONING`, `CALLING_TOOL`, `AWAITING_INPUT`, `EMITTING_RESPONSE`, `ERROR`,
   * `TERMINATED`. Typed `string` rather than that union on purpose: it is a value off the
   * wire, and a kit that *declares* the set cannot also have a branch for a value outside
   * it. `livenessOf` in `Rail.tsx` folds the eight it knows and reports anything else as
   * unknown, which is the branch the type would have made unreachable.
   */
  status: string;
}

/**
 * What a clone's row says about the instance behind it.
 *
 * `offline` is a clone with no running instance at all -- the ordinary state of an installed
 * persona nobody has messaged yet, and not a fault. `unknown` is a `status` this kit has not
 * been taught: it is reported as such rather than folded onto `idle`, because a state the
 * screen cannot read is not evidence that the clone is healthy.
 */
export type CloneLiveness = 'offline' | 'idle' | 'busy' | 'error' | 'terminated' | 'unknown';

export interface RailPersona {
  name: string;
  role: string;
  description: string;
  allowed_tools: string[];
  model_name?: string;
  temperature?: number;
  max_tokens?: number;
  enable_write_tools?: boolean;
  enable_subagent_tools?: boolean;
}

/**
 * Why a region is empty, in one sentence naming a cause and a remedy.
 *
 * `modelConfigured` is three-valued: `null` is "not answered yet" and must not be read as
 * `false`. The rail calls this one function for both of its regions, so the list and the
 * Clones section can never state the same absence in two different ways (#1060).
 */
export type EmptyCause = (modelConfigured: boolean | null, agentCount: number) => string;

// ---------------------------------------------------------------------------------------------
// Copy. Every word the rail puts on a screen arrives through one of these; the kit holds none.
// ---------------------------------------------------------------------------------------------

export interface PersonaDetailCopy {
  model: string;
  temperature: string;
  maxTokens: string;
  tools: string;
  /** A field the persona leaves unset. */
  notSet: string;
  /** An empty `allowed_tools`, as a sentence rather than a blank list. */
  noTools: string;
  /** `enable_write_tools` as a capability sentence, never a bare boolean. */
  writeAccess: (enabled: boolean | undefined) => string;
  /** `enable_subagent_tools` as a capability sentence, never a bare boolean. */
  subagentAccess: (enabled: boolean | undefined) => string;
}

export interface ConversationTitleEditorCopy {
  /** The input's accessible name. */
  titleLabel: string;
  save: string;
  cancel: string;
}

export interface DeleteConversationCopy {
  /** The dialog's heading, for the conversation called `name`. */
  heading: (name: string) => string;
  /** What goes: `entries` transcript entries and the record of `who` was in it. */
  body: (entries: number, who: string) => string;
  /** `who`, for a conversation no agent was ever in. */
  noAgents: string;
  cancel: string;
  /** Cancel's label once a delete has been refused: nothing is pending any more. */
  close: string;
  confirm: string;
  confirming: string;
  /** The heading, for a conversation whose saved copy could not be read (#1440). */
  unreadableHeading: string;
  /** The body, for the same: nothing of it can be read, so nothing of it can be named. */
  unreadableBody: string;
}

export interface ConversationListCopy {
  heading: string;
  newLabel: string;
  newTitle: string;
  collapse: string;
  rename: string;
  renameLabel: (name: string) => string;
  delete: string;
  deleteLabel: (name: string) => string;
  emptyCause: EmptyCause;
  /** When a row was last active, relative to `now`; `null` for a stamp that is not a date. */
  lastActive: (timestamp: string, now: number) => string | null;
  /** The exact time, as the tooltip on `lastActive`. */
  lastActiveTitle: (timestamp: string) => string;
  editor: ConversationTitleEditorCopy;
  deleteDialog: DeleteConversationCopy;
  /**
   * The row for a conversation the Core has and cannot read (#1440). It has no title, no
   * roster and no time to show, so the row says only that.
   */
  unreadableTitle: string;
  /** Deleting that row, as its accessible name. */
  unreadableDeleteLabel: string;
}

/**
 * The words inside an expanded clone row: the conversations that clone is in.
 *
 * Separate from the list above it because it answers a different question -- *who can I talk
 * to?* rather than *which conversation was I in?* (§3.2.2 R1). Both routes open the same
 * conversation object; nothing here creates a second one.
 */
export interface CloneConversationsCopy {
  /** The sub-heading above the clone's conversations. */
  heading: string;
  /**
   * A clone that is in no conversation yet.
   *
   * A sentence, not an empty list: "this clone has no conversations" and "the rail has not
   * loaded them" render identically as a blank block, and P6 forbids that.
   */
  none: string;
  /** The control that starts the first one, for a clone with none. */
  start: string;
  /** Opening one of `name`'s conversations, as its accessible name. */
  openLabel: (clone: string, title: string) => string;
  /** Starting a conversation with `clone`, as its accessible name. */
  startLabel: (clone: string) => string;
  /** Label/title for starting a thread with a clone. */
  thread?: string;
  threadLabel?: (clone: string) => string;
}

/**
 * The words a clone's liveness is said in.
 *
 * Two of them, because a dot cannot carry this alone. `busy` and `idle` are amber and
 * emerald, which is the pairing red-green colour blindness collapses, and the design's #1061
 * row asks that busy stay distinguishable from idle *by colour or a word*. So the row shows
 * the word for every state a glance must not miss, and the dot carries the sentence.
 */
export interface CloneLivenessCopy {
  /**
   * A word beside the clone's name, or `null` to show none.
   *
   * `null` is for the two states that need no word: a clone that is running with nothing in
   * progress, and one that is not running, which is the ordinary state of an installed clone
   * and is already what the rest of the row is about.
   */
  word: (liveness: CloneLiveness) => string | null;
  /**
   * The dot's accessible name and its tooltip: the state in a sentence.
   *
   * `status` is the raw value off the wire, so the `unknown` sentence can name what arrived
   * instead of reporting an unreadable state as nothing at all.
   */
  title: (liveness: CloneLiveness, status: string) => string;
}

export interface RailCopy {
  /**
   * The section heading. `Clones` since §3.2.2 R3 -- the product's one word for the thing
   * `AgentInfo` and `PersonaInfo` both describe. The types keep their names (P8); this is
   * what a user reads.
   */
  clones: string;
  /** Label/title for creating a new clone. */
  newClone: string;
  /** How many of the clones the section lists are running. Counted over the same array. */
  activeClones: (count: number) => string;
  cloneLiveness: CloneLivenessCopy;
  /**
   * The chevron's words.
   *
   * They name conversations and nothing else. The expansion used to carry a clone's
   * configuration card as well, which is why these took a `hasDetails` flag and said "and
   * details" when it was set. That card is the dock's Clone surface now, so there is one
   * thing behind the chevron and one sentence for it.
   */
  expandLabel: (name: string) => string;
  collapseLabel: (name: string) => string;
  expandTitle: string;
  collapseTitle: string;
  inspectLabel: (name: string) => string;
  inspectTitle: string;
  editLabel?: (name: string) => string;
  editTitle?: string;
  pinLabel?: (name: string) => string;
  unpinLabel?: (name: string) => string;
  pinTitle?: string;
  unpinTitle?: string;
  /**
   * The row's `⋯`, which holds pin, edit and profile. They were a button each beside the
   * name, and four buttons left a 240px rail about 70px for it.
   */
  moreLabel: (name: string) => string;
  moreTitle: string;
  /** Heads the pinned clones, which the list sorts first, so a row needs no mark of its own. */
  pinnedHeading: string;
  cloneConversations: CloneConversationsCopy;
  /*
   * `stepBudget`, `turns`, `saturated`, `tokens`, `estimated` and `estimatedTitle` were keys
   * here until #1059. The rail no longer draws those readouts, so it no longer asks a head
   * for words for them; the words now live beside the figures in the dock's Resource surface.
   */
  conversations: ConversationListCopy;
  /*
   * `persona` was here while the rail drew a clone's configuration card inside an expanded
   * row. The card is the dock's Clone surface now, so the words for it are asked of the head
   * by whatever mounts that surface -- `PersonaDetailCopy` is still exported, and the rail no
   * longer takes words for something it does not draw.
   */
}

// ---------------------------------------------------------------------------------------------
// Icons. Injected for the reason `KitIcon` gives.
// ---------------------------------------------------------------------------------------------

export interface ConversationListIcons {
  heading: KitIcon;
  newConversation: KitIcon;
  collapseRail: KitIcon;
  rename: KitIcon;
  delete: KitIcon;
  /** The glyph an agent's avatar is drawn with. */
  agent: KitIcon;
  saveTitle: KitIcon;
  cancelRename: KitIcon;
}

export interface RailIcons extends ConversationListIcons {
  clones: KitIcon;
  newClone: KitIcon;
  showDetails?: KitIcon;
  hideDetails?: KitIcon;
  inspectAgent: KitIcon;
  editAgent?: KitIcon;
  pin?: KitIcon;
  newThread?: KitIcon;
  /** The row's `⋯`. */
  more: KitIcon;
}

// ---------------------------------------------------------------------------------------------
// Behaviour the kit cannot own. Injected for the reason `KitIcon` gives.
// ---------------------------------------------------------------------------------------------

/**
 * Which of Escape's layers a kit surface claims (#1036). The head owns the third, `'turn'`;
 * nothing in the kit runs a turn, so the kit cannot claim it.
 */
export type KitEscapeLayer = 'dialog' | 'overlay';

/**
 * Escape's registry, as a hook the head passes in.
 *
 * The kit's delete dialog and title editor are two of Escape's five owners, but `ui-kit/`
 * imports nothing from outside itself but `react` (#1158), so they cannot reach
 * `lib/escapePrecedence.ts`. They take it the way they take their words and their glyphs.
 * A head that has no such registry passes a hook that does nothing, and Escape is then
 * simply unclaimed -- it is never handled twice.
 */
export type UseKitEscape = (layer: KitEscapeLayer, active: boolean, onEscape: () => void) => void;
