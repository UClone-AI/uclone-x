import React from 'react';
import { Avatar } from '../primitives/Avatar';
import { StatusDot } from '../primitives/StatusDot';
import type { KitTone } from '../primitives/StatusDot';
import { ConversationList, LastActive } from './ConversationList';
import { CloneRowMenu } from './CloneRowMenu';
import type { CloneRowMenuItem } from './CloneRowMenu';
import { ConversationTitleEditor } from './ConversationTitleEditor';
import { DeleteConversationDialog } from './DeleteConversationDialog';
import type {
  CloneLiveness,
  RailAgent,
  RailCopy,
  RailIcons,
  RailPersona,
  RailRoom,
  UseKitEscape,
} from './types';

/**
 * `AgentState` (`src/uclone_x/agent/models.py`) folded onto what a 240px row can draw.
 *
 * `/api/agents` sends that enum's value verbatim as `status`. The row used to compare it
 * against `'busy'`, a value the endpoint has never sent, so the comparison was never true and
 * a clone mid-tool-call, in `ERROR` or `TERMINATED` drew the same emerald dot as an idle one.
 * The eight states the runtime has are named here, once, against the file that defines them.
 */
const LIVENESS_BY_STATE: Record<string, CloneLiveness> = {
  IDLE: 'idle',
  AWAITING_INPUT: 'idle',
  INGESTING: 'busy',
  REASONING: 'busy',
  CALLING_TOOL: 'busy',
  EMITTING_RESPONSE: 'busy',
  ERROR: 'error',
  TERMINATED: 'terminated',
};

/**
 * What the row may say about the instance behind a clone.
 *
 * A `status` this table does not hold is `unknown`, never `idle`: the runtime is free to grow
 * a state, and a screen that folds one it cannot read onto "running, nothing in progress"
 * reports health it has no evidence for.
 */
export const livenessOf = (status: string | undefined): CloneLiveness =>
  status === undefined ? 'offline' : (LIVENESS_BY_STATE[status] ?? 'unknown');

/**
 * The dot's colour per liveness.
 *
 * Amber and emerald are the pairing red-green colour blindness collapses, so the colour is
 * never the only carrier: `copy.cloneLiveness.word` puts busy, error, stopped and unknown
 * beside the name in words, and the dot's own `title` says all six in a sentence.
 */
const TONE_BY_LIVENESS: Record<CloneLiveness, KitTone> = {
  offline: 'neutral',
  idle: 'success',
  busy: 'warning',
  error: 'danger',
  terminated: 'neutral',
  unknown: 'neutral',
};

/**
 * The rail's rendered width in pixels -- the `w-60` on its `<aside>` below, as a number.
 *
 * Exported because the dock needs the quantity, not the class: it decides whether what is
 * left of the workspace row can seat it (`ArtifactsDock`'s `reservedWidth`). A Tailwind class
 * cannot be read from another component, and jsdom resolves no stylesheet, so the number is
 * stated here beside the class it mirrors and pinned against it by a test in
 * `WorkspaceSidebar.test.tsx` that fails if the two ever part.
 */
export const RAIL_WIDTH_PX = 240;

/** The Tailwind class `RAIL_WIDTH_PX` mirrors. `w-60` is 15rem, which is 240px. */
export const RAIL_WIDTH_CLASS = 'w-60';

export interface RailProps {
  agents: RailAgent[];
  personas?: RailPersona[];
  selectedAgent: string;
  onSelectAgent: (id: string) => void;
  /** Inspect a clone's settings and profile in the dock. */
  onInspectAgent?: (id: string) => void;
  /** Edit a clone's settings directly in the dock. */
  onEditAgent?: (id: string) => void;
  /** Start creating a new clone. */
  onNewClone?: () => void;
  /*
   * The step budget, the turn counter and the token total are **not** props here (#1059).
   * They are U1/U3 instruments and the rail is U0's default screen, so they live in the
   * dock's Resource surface -- `components/artifacts/ResourceSummary.tsx`, which already
   * held two of the three. The kit does not take them at all, rather than taking them and
   * declining to draw them, so a head cannot pass them here and wonder where they went.
   */
  onClose?: () => void;
  /**
   * Draw the rail over the conversation instead of beside it.
   *
   * **Required, with no default.** The caller decides it from the window (`railOverlays` in
   * the head's `lib/rail.ts`) because the caller is also what tells the dock how much of the
   * row the rail holds: an overlaid rail holds none of it. A default of `false` would let a
   * mount site that forgot the prop divide a 400px row again with nothing going red.
   */
  overlay: boolean;
  /**
   * Conversations that can seat more than one agent.
   *
   * **Required, and rendered with no condition on it.** The list used to be gated on
   * `rooms.length > 0` -- and the only control that creates the first conversation lives
   * inside the list, so on a fresh install the feature could not be reached at all. A
   * list whose presence depends on its own contents can never be filled (ui-authoring
   * §2: the conversation carries no condition).
   *
   * These are the only chats the rail lists. The pre-D1 one-agent sessions used to sit
   * below them under a `Past chats` heading; the owner ruled on 2026-09-19 that the legacy
   * `/api/chat` path may be ignored, so the group is gone and the rail no longer takes
   * sessions at all. The container still carries the `sessions-list` testid two end-to-end
   * tests select.
   */
  rooms: RailRoom[];
  /** Conversations the Core has and cannot read. See `ConversationList` (#1440). */
  unreadableRoomIds: string[];
  currentRoomId: string | null;
  onSelectRoom: (roomId: string) => void;
  /**
   * Start a conversation.
   *
   * The optional `cloneId` is what an expanded clone row passes when that clone is in no
   * conversation yet (§3.2.2 R1): the new conversation seats *that* clone, not whichever one
   * happens to be selected. It is an argument rather than a `onSelectAgent` call followed by
   * `onNewRoom()`, because the head reads the selection from state the second call would not
   * yet see. Omitted, the head uses its selection, which is the list's New.
   */
  onNewRoom: (cloneId?: string, isGroup?: boolean) => void;
  /** See `ConversationList`: each rejects with the Core's refusal, shown where it was asked. */
  onRenameRoom: (roomId: string, title: string) => Promise<void>;
  onDeleteRoom: (roomId: string) => Promise<void>;
  /** `null` until the runtime has said. See `ConversationList`'s own note. */
  modelConfigured: boolean | null;
  /** Pinned clone IDs to display at the top of the clones section. */
  pinnedCloneIds?: string[];
  /** Callback when user clicks the pin toggle for a clone. */
  onTogglePinClone?: (id: string) => void;
  /** Explicit group room IDs, retained in Group Chats even if seating 0 or 1 agent. */
  groupRoomIds?: string[];
  /** Every word the rail shows. The kit holds none (#1063 D). */
  copy: RailCopy;
  /**
   * Where a clone's picture is, asked of the head by id.
   *
   * A URL or nothing. The kit fetches no pictures and derives none from a name: a head that
   * knows where its clones' pictures are answers with one, and a clone the head has no
   * picture for draws `Avatar`'s single default -- the same default for every such clone, so
   * the drawing is never read as a likeness. Left off entirely, every clone draws it.
   */
  avatarSrc?: (cloneId: string) => string | undefined;
  /** Every glyph the rail draws, injected so the kit depends on no icon library. */
  icons: RailIcons;
  /** Escape's registry, injected: the kit may not import the head's (#1036, #1158). */
  useEscape: UseKitEscape;
}

/**
 * The workspace rail: conversations, and the clones whose conversations they are.
 *
 * **No readouts (#1059).** The rail used to end in a step-budget meter, a turn counter and a
 * token total, which put token accounting on U0's default screen -- the thing ui-authoring §2
 * step 4 calls an instrument and `ConversationList`'s own docstring, inches above, calls a
 * second job. They are not removed; they are one keystroke away, in the dock's **Resource**
 * surface, with the `estimated` qualifier (#939) and the saturation badge intact.
 *
 * The clones section stays. Since #1187 a clone's row expands to *the conversations that
 * clone is in*, so it is an index into the list above it rather than the active
 * conversation's roster -- the second job §3.2.4 moved to the conversation header.
 *
 * Props-only by construction (#1063 D, #1158): no store, no fetch, no copy, no icon import.
 * The boundary test in `frontend/src/ui-kit.test.ts` fails if anything under `ui-kit/`
 * imports from outside it, other than `react`.
 */
export const Rail: React.FC<RailProps> = ({
  agents,
  personas,
  selectedAgent,
  onSelectAgent,
  onInspectAgent,
  onEditAgent,
  onNewClone,
  onClose,
  overlay,
  rooms,
  unreadableRoomIds,
  currentRoomId,
  onSelectRoom,
  onNewRoom,
  onRenameRoom,
  onDeleteRoom,
  modelConfigured,
  pinnedCloneIds,
  onTogglePinClone,
  groupRoomIds,
  avatarSrc,
  copy,
  icons,
  useEscape,
}) => {
  const ClonesIcon = icons.clones;
  const NewCloneIcon = icons.newClone;
  const EditAgentIcon = icons.editAgent || icons.inspectAgent;
  const PinIcon = icons.pin;
  const MoreIcon = icons.more;
  const NewThreadIcon = icons.newThread || icons.newConversation;
  const ChevronDownIcon = icons.hideDetails;
  const ChevronRightIcon = icons.showDetails;
  const DeleteIcon = icons.delete;
  const RenameIcon = icons.rename;

  /** Multi-agent rooms and rooms created explicitly as group chats belong in Group Chats. */
  const groupRooms = React.useMemo(() => {
    const groupSet = new Set(groupRoomIds ?? []);
    return rooms.filter((r) => r.agent_ids.length !== 1 || groupSet.has(r.room_id));
  }, [rooms, groupRoomIds]);

  /** Group 1:1 sessions strictly by clone ID, excluding group chats. */
  const sessionsByClone = React.useMemo(() => {
    const groupSet = new Set(groupRoomIds ?? []);
    const map = new Map<string, RailRoom[]>();
    for (const room of rooms) {
      if (room.agent_ids.length === 1 && !groupSet.has(room.room_id)) {
        const agentId = room.agent_ids[0];
        const list = map.get(agentId) || [];
        list.push(room);
        map.set(agentId, list);
      }
    }
    for (const list of map.values()) {
      list.sort((a, b) => {
        const tA = new Date(a.updated_at).getTime() || 0;
        const tB = new Date(b.updated_at).getTime() || 0;
        return tB - tA;
      });
    }
    return map;
  }, [rooms, groupRoomIds]);

  const [expandedClones, setExpandedClones] = React.useState<Record<string, boolean>>({});

  const toggleCloneExpanded = (cloneId: string, currentlyExpanded: boolean) => {
    setExpandedClones((prev) => ({ ...prev, [cloneId]: !currentlyExpanded }));
  };

  const LAST_ACTIVE_TICK_MS = 60_000;
  const [now, setNow] = React.useState(() => Date.now());
  React.useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), LAST_ACTIVE_TICK_MS);
    return () => clearInterval(tick);
  }, []);

  /** The clone whose `⋯` menu is open. One at a time: opening another closes this one. */
  const [menuCloneId, setMenuCloneId] = React.useState<string | null>(null);
  const closeMenu = React.useCallback(() => setMenuCloneId(null), []);

  const [deletingSession, setDeletingSession] = React.useState<RailRoom | null>(null);
  const [renamingId, setRenamingId] = React.useState<string | null>(null);

  /**
   * The clones the section lists, from whichever source the head has.
   *
   * Ordered by hybrid Pin + MRU:
   * 1. Pinned clones first.
   * 2. Within pinned / unpinned tiers, by most recent conversation activity.
   * 3. Clones without conversations fall to the bottom of their tier, sorted alphabetically.
   */
  const clones = React.useMemo(() => {
    const liveOf = (id: string) => agents.find((a) => a.id === id);
    const base =
      personas && personas.length > 0
        ? personas.map((p) => ({ id: p.name, label: p.name, role: p.role, live: liveOf(p.name) }))
        : agents.map((a) => ({ id: a.id, label: a.label || a.id, role: a.role, live: a }));

    const latestActive = new Map<string, number>();
    for (const room of rooms) {
      const roomTime = new Date(room.updated_at).getTime() || 0;
      for (const agentId of room.agent_ids) {
        const prev = latestActive.get(agentId) ?? 0;
        if (roomTime > prev) {
          latestActive.set(agentId, roomTime);
        }
      }
    }

    const pinnedSet = new Set(pinnedCloneIds ?? []);

    return [...base].sort((a, b) => {
      const aPinned = pinnedSet.has(a.id);
      const bPinned = pinnedSet.has(b.id);
      if (aPinned !== bPinned) return aPinned ? -1 : 1;

      const aTime = latestActive.get(a.id) ?? 0;
      const bTime = latestActive.get(b.id) ?? 0;

      if (aTime !== bTime) {
        return bTime - aTime;
      }

      const cmp = a.label.localeCompare(b.label);
      return cmp !== 0 ? cmp : a.id.localeCompare(b.id);
    });
  }, [agents, personas, rooms, pinnedCloneIds]);

  return (
    <aside
      data-testid="chat-sidebar"
      data-overlay={overlay ? 'true' : 'false'}
      // Overlaid, the rail is drawn over the left of the workspace and takes none of the row, so
      // the conversation keeps the whole window (#1062). It sits above the dock (z-20) because it
      // is only open at that width when the user has just asked for it; the dock's close control
      // is at its right edge and stays reachable beside it. The background is opaque here, since
      // the conversation is underneath rather than beside it.
      className={`${
        overlay ? 'absolute inset-y-0 left-0 z-30 bg-slate-950 shadow-2xl' : 'relative bg-slate-950/80'
      } w-60 border-r border-slate-800/60 flex flex-col h-full min-h-0 shrink-0 select-none`}
    >
      {/* One scroller for the whole rail body, not one per section.

          The two lists used to scroll independently -- conversations took the leftover
          height, clones were capped at `max-h-56` -- so the rail could show two
          scrollbars at once and neither section could borrow the other's unused room.
          Both now size to their rows inside this one scroller, which is the shape the
          sibling product's side panel has (a fixed head, one flexible scrolling body).

          `min-h-0` is what lets it scroll at all: a flex child defaults to a floor of its
          own content's height, which would push the rail past the window instead. */}
      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-4">
        {/* The rail's top list: Group Chats. It sizes to its rows and scrolls with the
            section below it, rather than holding a scroller of its own. */}
        <div className="flex flex-col">
          <ConversationList
            rooms={groupRooms}
            unreadableRoomIds={unreadableRoomIds}
            currentRoomId={currentRoomId}
            onSelectRoom={onSelectRoom}
            onNewConversation={() => onNewRoom(undefined, true)}
            onRenameRoom={onRenameRoom}
            onDeleteRoom={onDeleteRoom}
            // `clones`, not `agents`: the empty list's cause says whether any clone is set up,
            // and `clones` is what the section below actually lists. Counted from `agents`, an
            // install with personas but no running instance read "No clones are set up yet"
            // directly above the clones it had (#1088).
            agentCount={clones.length}
            modelConfigured={modelConfigured}
            listTestId="sessions-list"
            onCollapse={onClose}
            copy={copy.conversations}
            icons={icons}
            useEscape={useEscape}
          />
        </div>

        {/* Clones -- who the user can talk to, beneath the conversations they have had. */}
        <div className="space-y-1.5 px-1 pt-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-slate-400">
            <div className="flex items-center gap-1.5">
              <ClonesIcon className="w-3.5 h-3.5 text-indigo-400" />
              <span>{copy.clones}</span>
            </div>
            <div className="flex items-center gap-1.5">
              {/* The number and the list come from one expression. The badge counted `agents`
                  while the rows came from `clones`, so an install with three personas and one
                  running instance read "1 active" over three rows -- and one with a live agent
                  that has no persona read "2 active" over rows that did not include it. It
                  counts the rows it sits above, and only the ones with an instance behind
                  them, which is what "active" claims. */}
              {clones.length > 1 && (
                <span className="text-[10px] text-slate-500 font-normal">
                  {copy.activeClones(clones.filter((clone) => clone.live).length)}
                </span>
              )}
              {onNewClone && (
                <button
                  type="button"
                  data-testid="rail-new-clone"
                  aria-label={copy.newClone}
                  title={copy.newClone}
                  onClick={onNewClone}
                  className="p-0.5 rounded text-slate-400 hover:text-white hover:bg-slate-800/80 transition-colors"
                >
                  <NewCloneIcon className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
          {/* No ceiling on this list, and no scroller of its own. `max-h-56` capped it at
              224px whatever room the rail had: on an install with one clone and no
              conversations, an expanded row's detail card was cut off mid-field inside a
              short scrolling box while two thirds of the rail sat empty above it. The
              cap was load-bearing while this section was the second of two scrollers;
              with one scroller for the whole body, the section sizes to its rows. */}
          <div className="space-y-0.5">
            {clones.length > 0 ? (
              clones.map((clone, index) => {
                const cloneSessions = sessionsByClone.get(clone.id) || [];
                const hasSessions = cloneSessions.length > 0;
                const isExpanded = expandedClones[clone.id] ?? true;
                const isSelected = clone.id === selectedAgent || cloneSessions.some((s) => s.room_id === currentRoomId);
                const isPinned = (pinnedCloneIds ?? []).includes(clone.id);
                // The list sorts pinned clones first, so the tier is marked once, at its
                // boundaries, rather than on every row: a heading over the first pinned row,
                // a rule over the first unpinned one.
                const startsPinned = isPinned && index === 0;
                const startsUnpinned =
                  !isPinned && index > 0 && (pinnedCloneIds ?? []).includes(clones[index - 1].id);
                const isMenuOpen = menuCloneId === clone.id;
                const menuItems: CloneRowMenuItem[] = [];
                if (onTogglePinClone) {
                  menuItems.push({
                    testId: `clone-pin-${clone.id}`,
                    label: isPinned
                      ? copy.unpinTitle || `Unpin ${clone.label}`
                      : copy.pinTitle || `Pin ${clone.label} to top`,
                    icon: PinIcon,
                    onSelect: () => onTogglePinClone(clone.id),
                  });
                }
                if (onEditAgent) {
                  menuItems.push({
                    testId: `persona-edit-${clone.id}`,
                    label: copy.editTitle || `Edit ${clone.label} settings`,
                    icon: EditAgentIcon,
                    onSelect: () => onEditAgent(clone.id),
                  });
                }
                if (onInspectAgent) {
                  menuItems.push({
                    testId: `clone-profile-item-${clone.id}`,
                    label: copy.inspectTitle,
                    icon: icons.inspectAgent,
                    onSelect: () => onInspectAgent(clone.id),
                  });
                }
                const liveness = livenessOf(clone.live?.status);
                const livenessWord = copy.cloneLiveness.word(liveness);

                const handleCloneClick = () => {
                  onSelectAgent(clone.id);
                  if (hasSessions) {
                    toggleCloneExpanded(clone.id, isExpanded);
                    if (!cloneSessions.some((s) => s.room_id === currentRoomId)) {
                      onSelectRoom(cloneSessions[0].room_id);
                    }
                  } else {
                    onNewRoom(clone.id);
                  }
                };

                return (
                  <React.Fragment key={clone.id}>
                  {startsPinned && (
                    <p
                      data-testid="clones-pinned-heading"
                      className="px-1.5 pt-0.5 pb-1 text-[10px] font-medium text-slate-500"
                    >
                      {copy.pinnedHeading}
                    </p>
                  )}
                  {startsUnpinned && (
                    <div
                      role="separator"
                      data-testid="clones-pinned-separator"
                      className="mx-1.5 my-1.5 border-t border-slate-800/80"
                    />
                  )}
                  <div>
                    <div
                      onContextMenu={
                        menuItems.length > 0
                          ? (e) => {
                              e.preventDefault();
                              setMenuCloneId(clone.id);
                            }
                          : undefined
                      }
                      className={`group relative w-full flex items-center justify-between gap-1 rounded-lg text-xs transition-all ${
                        isSelected
                          ? 'bg-slate-800/80 text-slate-100 font-medium'
                          : 'text-slate-400 hover:bg-slate-900/60 hover:text-slate-200'
                      }`}
                    >
                      {/* The clone's picture: clicking opens profile view in dock */}
                      <span className="relative shrink-0 pl-1.5 py-1.5">
                        <Avatar
                          label={clone.label}
                          kind="agent"
                          agentIcon={icons.agent}
                          imageSrc={avatarSrc?.(clone.id)}
                          size="sm"
                          className="border border-slate-700/60 cursor-pointer hover:ring-2 hover:ring-cyan-500/80 transition-all"
                          data-testid={`clone-avatar-${clone.id}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            onInspectAgent?.(clone.id);
                          }}
                          interactiveLabel={copy.inspectLabel(clone.label)}
                        />
                        {/* A dark outline, so the dot stays a dot against whatever the
                            picture happens to be behind it. */}
                        <span className="absolute bottom-1 right-0 flex items-center justify-center leading-none rounded-full ring-2 ring-slate-950 bg-slate-950 pointer-events-none">
                          <StatusDot
                            tone={TONE_BY_LIVENESS[liveness]}
                            data-testid={`clone-liveness-${clone.id}`}
                            title={copy.cloneLiveness.title(liveness, clone.live?.status ?? '')}
                          />
                        </span>
                      </span>

                      {/* Main card body: clicking opens recent conversation. The name owns
                          this width; the row's actions float over its end only while the row
                          is hovered or holds focus, so they cost the name nothing at rest. A
                          button that is merely transparent still holds its place in the row,
                          which is how a hidden pin used to cut the name short. */}
                      <div className="relative flex-1 min-w-0 flex items-center">
                        <button
                          type="button"
                          data-testid={`persona-item-${clone.id}`}
                          aria-current={isSelected ? 'true' : undefined}
                          onClick={handleCloneClick}
                          className="flex-1 min-w-0 text-left py-2 px-2.5 rounded-lg flex flex-col justify-center overflow-hidden"
                        >
                          <div className="flex items-center gap-1.5 truncate">
                            <span className="truncate text-sm font-medium text-slate-200">{clone.label}</span>
                            {livenessWord !== null && (
                              <span
                                data-testid={`clone-liveness-word-${clone.id}`}
                                className="shrink-0 text-[11px] text-slate-500 font-normal"
                              >
                                {livenessWord}
                              </span>
                            )}
                          </div>
                          {clone.role && (
                            <span
                              className="text-[12px] font-mono text-slate-500 truncate flex items-center gap-0.5"
                              title={clone.role}
                            >
                              <span className="text-slate-600 font-mono">@</span>
                              <span className="truncate">{clone.role}</span>
                            </span>
                          )}
                        </button>
                        <div
                          data-testid={`clone-actions-${clone.id}`}
                          className={`absolute right-0 inset-y-0 my-auto h-fit flex items-center gap-0.5 p-0.5 rounded-md bg-slate-900 ring-1 ring-slate-800 transition-opacity [@media(hover:none)]:static [@media(hover:none)]:shrink-0 [@media(hover:none)]:ring-0 [@media(hover:none)]:bg-transparent [@media(hover:none)]:opacity-100 [@media(hover:none)]:pointer-events-auto ${
                            isMenuOpen
                              ? 'opacity-100'
                              : 'opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto group-focus-within:opacity-100 group-focus-within:pointer-events-auto'
                          }`}
                        >
                          <button
                            type="button"
                            data-testid={`clone-new-thread-${clone.id}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              onSelectAgent(clone.id);
                              onNewRoom(clone.id);
                            }}
                            aria-label={
                              copy.cloneConversations?.threadLabel
                                ? copy.cloneConversations.threadLabel(clone.label)
                                : copy.cloneConversations.startLabel(clone.label)
                            }
                            title={
                              copy.cloneConversations?.thread || copy.cloneConversations.start
                            }
                            className="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
                          >
                            <NewThreadIcon className="w-3.5 h-3.5" />
                          </button>
                          {menuItems.length > 0 && (
                            <button
                              type="button"
                              data-testid={`clone-menu-${clone.id}`}
                              aria-label={copy.moreLabel(clone.label)}
                              aria-haspopup="menu"
                              aria-expanded={isMenuOpen}
                              title={copy.moreTitle}
                              onClick={(e) => {
                                e.stopPropagation();
                                setMenuCloneId(isMenuOpen ? null : clone.id);
                              }}
                              className="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
                            >
                              <MoreIcon className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                        {isMenuOpen && (
                          <CloneRowMenu
                            label={copy.moreLabel(clone.label)}
                            items={menuItems}
                            onClose={closeMenu}
                            useEscape={useEscape}
                            testId={`clone-menu-list-${clone.id}`}
                          />
                        )}
                      </div>

                      {/* The chevron stays at rest: it says the row has conversations under it,
                          which is a fact about the row rather than an action on it. */}
                      <div className="flex items-center pr-1.5 shrink-0">
                        {hasSessions && ChevronDownIcon && ChevronRightIcon && (
                          <button
                            type="button"
                            data-testid={`clone-chevron-${clone.id}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleCloneExpanded(clone.id, isExpanded);
                            }}
                            aria-label={
                              isExpanded
                                ? copy.collapseLabel(clone.label)
                                : copy.expandLabel(clone.label)
                            }
                            title={isExpanded ? copy.collapseTitle : copy.expandTitle}
                            className="p-1 rounded text-slate-500 hover:text-slate-200 hover:bg-slate-800/80 transition-colors"
                          >
                            {isExpanded ? (
                              <ChevronDownIcon className="w-3.5 h-3.5" />
                            ) : (
                              <ChevronRightIcon className="w-3.5 h-3.5" />
                            )}
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Nested 1:1 sessions - shown when sessions exist and clone is expanded */}
                    {hasSessions && isExpanded && (
                      <div
                        data-testid={`clone-expansion-${clone.id}`}
                        className="pl-8 pr-1 space-y-0.5 mt-0.5"
                      >
                        <div data-testid="sessions-list" className="space-y-0.5">
                          {cloneSessions.map((session) => {
                            const isSessionActive = session.room_id === currentRoomId;
                            return renamingId === session.room_id ? (
                              <ConversationTitleEditor
                                useEscape={useEscape}
                                key={session.room_id}
                                title={session.title}
                                onSave={async (title) => {
                                  await onRenameRoom(session.room_id, title);
                                  setRenamingId(null);
                                }}
                                onCancel={() => setRenamingId(null)}
                                copy={copy.conversations.editor}
                                saveIcon={icons.saveTitle}
                                cancelIcon={icons.cancelRename}
                              />
                            ) : (
                              <div
                                key={session.room_id}
                                data-testid={`clone-session-item-${session.room_id}`}
                                className={`group relative flex items-center justify-between gap-1 px-2 py-1.5 rounded-lg text-xs cursor-pointer transition-colors border ${
                                  isSessionActive
                                    ? 'bg-blue-950/40 text-blue-200 border-blue-500/30 font-medium'
                                    : 'text-slate-400 hover:bg-slate-900/60 hover:text-slate-200 border-transparent'
                                }`}
                                onClick={() => {
                                  onSelectAgent(clone.id);
                                  onSelectRoom(session.room_id);
                                }}
                              >
                                <button
                                  type="button"
                                  data-testid={`conversation-${session.room_id}`}
                                  aria-label={copy.cloneConversations.openLabel(clone.label, session.title || session.room_id)}
                                  title={session.title || session.room_id}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onSelectAgent(clone.id);
                                    onSelectRoom(session.room_id);
                                  }}
                                  className="flex-1 min-w-0 text-left overflow-hidden flex items-baseline justify-between gap-1.5"
                                >
                                  <span className="truncate">{session.title || session.room_id}</span>
                                  <LastActive
                                    updatedAt={session.updated_at}
                                    now={now}
                                    roomId={session.room_id}
                                    copy={copy.conversations}
                                  />
                                </button>
                                <span
                                  className={`absolute right-1 top-1/2 -translate-y-1/2 z-10 flex items-center gap-0.5 rounded-md px-0.5 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-within:opacity-100 transition-opacity ${
                                    isSessionActive
                                      ? 'bg-blue-950 text-blue-200'
                                      : 'bg-slate-900 text-slate-400'
                                  }`}
                                >
                                  {RenameIcon && (
                                    <button
                                      type="button"
                                      data-testid={`rename-conversation-${session.room_id}`}
                                      aria-label={copy.conversations.renameLabel(session.title || session.room_id)}
                                      title={copy.conversations.rename}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setRenamingId(session.room_id);
                                      }}
                                      className="p-1 rounded text-slate-500 hover:text-slate-200 hover:bg-slate-800 transition-colors shrink-0"
                                    >
                                      <RenameIcon className="w-3 h-3" />
                                    </button>
                                  )}
                                  <button
                                    type="button"
                                    data-testid={`delete-conversation-${session.room_id}`}
                                    aria-label={copy.conversations.deleteLabel(session.title || session.room_id)}
                                    title={copy.conversations.delete}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setDeletingSession(session);
                                    }}
                                    className="p-1 rounded text-slate-500 hover:text-slate-200 hover:bg-slate-800 transition-colors shrink-0"
                                  >
                                    <DeleteIcon className="w-3 h-3" />
                                  </button>
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                  </React.Fragment>
                );
              })
            ) : (
              /* No clones, so the section says so, in the words the list above it uses for
                 the same fact (#1060). It used to render a row reading "General Assistant" --
                 a name no clone here has ever had, contradicting the list inches above and
                 naming no remedy. */
              <p
                data-testid="agents-empty-cause"
                className="text-[11px] text-slate-500 leading-relaxed py-1 px-1"
              >
                {copy.conversations.emptyCause(modelConfigured, clones.length)}
              </p>
            )}
          </div>
        </div>
      </div>

      {deletingSession ? (
        <DeleteConversationDialog
          useEscape={useEscape}
          room={deletingSession}
          onConfirm={async () => {
            await onDeleteRoom(deletingSession.room_id);
            setDeletingSession(null);
          }}
          onClose={() => setDeletingSession(null)}
          copy={copy.conversations.deleteDialog}
        />
      ) : null}
    </aside>
  );
};
