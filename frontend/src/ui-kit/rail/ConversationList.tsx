import React, { useEffect, useState } from 'react';
import { Avatar } from '../primitives/Avatar';
import { Button } from '../primitives/Button';
import { ConversationTitleEditor } from './ConversationTitleEditor';
import { DeleteConversationDialog } from './DeleteConversationDialog';
import type {
  ConversationListCopy,
  ConversationListIcons,
  RailRoom,
  UseKitEscape,
} from './types';

/** How many of a room's agents get a face in the rail before the rest collapse to a count. */
const ROSTER_PREVIEW_MAX = 3;

/** How often the rows' "5 minutes ago" is re-read against the clock. */
const LAST_ACTIVE_TICK_MS = 60_000;

export interface ConversationListProps {
  rooms: RailRoom[];
  /**
   * Ids of conversations the Core has and cannot read (#1440).
   *
   * Required, because leaving it off is how the rail used to behave: such a conversation
   * was dropped from the listing, so it could be neither seen nor deleted. Each gets a row
   * that says only that it could not be read, and a Delete.
   */
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
   * collapsing it onto `false` puts "no model is configured" on screen for the moment
   * before the runtime's first reply -- a wrong cause is worse than no cause.
   */
  modelConfigured: boolean | null;
  /** The testid the E2E suite selects the rail's list by. */
  listTestId?: string;
  /** Collapse the rail this list sits in, when it sits in one. */
  onCollapse?: () => void;
  copy: ConversationListCopy;
  icons: ConversationListIcons;
  /** Escape's registry, injected: the kit may not import the head's (#1036, #1158). */
  useEscape: UseKitEscape;
}

/**
 * The rail's one job: list conversations.
 *
 * Participants are not here. They belong to the conversation they are in; a rail that also
 * carries the active conversation's roster is a rail doing two jobs.
 *
 * **There is no `Past chats` group.** The list used to end with the pre-D1 one-agent chats
 * from `/api/chat`, under their own heading. The owner ruled on 2026-09-19 that the legacy
 * session path may be ignored and is removed outright with the single-agent surface, and
 * §3.2.2 **[Rev 20]** had already said *"One list means one list, so the separation goes."*
 * So the group is gone rather than taught to feed the rest of the rail.
 *
 * The three empty states are three different sentences on purpose (`copy.emptyCause`). "No
 * model", "no agents" and "no conversations" have three different remedies, and a single empty
 * list renders them identically -- which tells a stuck user nothing about which of the three
 * they are looking at.
 */
export const ConversationList: React.FC<ConversationListProps> = ({
  rooms,
  unreadableRoomIds,
  currentRoomId,
  onSelectRoom,
  onNewConversation,
  onRenameRoom,
  onDeleteRoom,
  agentCount,
  modelConfigured,
  listTestId,
  onCollapse,
  copy,
  icons,
  useEscape,
}) => {
  const cause = copy.emptyCause(modelConfigured, agentCount);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  // The row as it was when Delete was pressed, not its id: a refused delete re-reads the
  // list, and the dialog must still be able to name what it was asked about.
  const [deleting, setDeleting] = useState<RailRoom | null>(null);
  const [deletingUnreadable, setDeletingUnreadable] = useState<string | null>(null);

  // "5 minutes ago" is a claim about now, so it is re-read as now moves; otherwise a rail
  // left open all afternoon keeps saying "just now" about the morning.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), LAST_ACTIVE_TICK_MS);
    return () => clearInterval(tick);
  }, []);

  const HeadingIcon = icons.heading;
  const NewIcon = icons.newConversation;
  const CollapseIcon = icons.collapseRail;
  const RenameIcon = icons.rename;
  const DeleteIcon = icons.delete;

  return (
    <div data-testid="conversation-list" className="flex flex-col min-h-0 gap-3">
      <div className="flex items-center justify-between gap-2 px-1">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
          <HeadingIcon className="w-4 h-4 text-slate-400" />
          <span>{copy.heading}</span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            data-testid="new-conversation-button"
            title={copy.newTitle}
            onClick={onNewConversation}
          >
            <NewIcon className="w-3.5 h-3.5" />
            <span>{copy.newLabel}</span>
          </Button>
          {onCollapse ? (
            <Button
              variant="quiet"
              size="compact"
              onClick={onCollapse}
              title={copy.collapse}
              aria-label={copy.collapse}
            >
              <CollapseIcon className="w-4 h-4" />
            </Button>
          ) : null}
        </div>
      </div>

      <div
        data-testid={listTestId}
        className="flex-1 overflow-y-auto min-h-0 space-y-0.5 text-xs pr-1"
      >
        {rooms.length === 0 && unreadableRoomIds.length === 0 ? (
          <p data-testid="conversations-empty-cause" className="px-2 py-3 text-slate-500 leading-relaxed">
            {cause}
          </p>
        ) : null}

        {rooms.map((room) =>
          renamingId === room.room_id ? (
            <ConversationTitleEditor
              useEscape={useEscape}
              key={room.room_id}
              title={room.title}
              onSave={async (title) => {
                await onRenameRoom(room.room_id, title);
                setRenamingId(null);
              }}
              onCancel={() => setRenamingId(null)}
              copy={copy.editor}
              saveIcon={icons.saveTitle}
              cancelIcon={icons.cancelRename}
            />
          ) : (
            // Rename and Delete sit beside the row, revealed on hover and focus to
            // preserve title space. Siblings, not children -- a button may not contain another.
            <div
              key={room.room_id}
              className={`group relative flex items-center rounded-lg transition-colors border ${
                room.room_id === currentRoomId
                  ? 'bg-blue-950/40 text-blue-200 border-blue-500/30 font-medium'
                  : 'text-slate-400 hover:bg-slate-900/60 hover:text-slate-200 border-transparent'
              }`}
            >
              <button
                type="button"
                data-testid={`conversation-${room.room_id}`}
                onClick={() => onSelectRoom(room.room_id)}
                className="flex-1 min-w-0 text-left px-2 py-1.5"
              >
                {/* The order is the caller's -- the Core sends the most recently active first
                    (#1053) -- so it is rendered as it arrives and not re-sorted here. */}
                <span className="flex items-baseline justify-between gap-1.5">
                  <span className="truncate">{room.title || room.room_id}</span>
                  <LastActive
                    updatedAt={room.updated_at}
                    now={now}
                    roomId={room.room_id}
                    copy={copy}
                  />
                </span>
                {/* A count that is always "1" is noise, so a single agent shows its name. Faces
                    before names: who's in the room is the thing a glance should answer first. */}
                {room.agent_ids.length > 0 ? (
                  <span className="flex items-center gap-1 mt-0.5">
                    <span className="flex items-center -space-x-1 shrink-0">
                      {room.agent_ids.slice(0, ROSTER_PREVIEW_MAX).map((agentId) => (
                        <Avatar
                          key={agentId}
                          label={agentId}
                          kind="agent"
                          agentIcon={icons.agent}
                          size="2xs"
                          className="ring-2 ring-slate-950"
                        />
                      ))}
                    </span>
                    <span className="truncate text-[11px] text-slate-500">
                      {room.agent_ids.join(', ')}
                      {room.agent_ids.length > ROSTER_PREVIEW_MAX
                        ? ` +${room.agent_ids.length - ROSTER_PREVIEW_MAX}`
                        : ''}
                    </span>
                  </span>
                ) : null}
              </button>
              <span
                className={`absolute right-1 top-1/2 -translate-y-1/2 z-10 flex items-center rounded-md px-0.5 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-within:opacity-100 transition-opacity ${
                  room.room_id === currentRoomId
                    ? 'bg-blue-950 text-blue-200'
                    : 'bg-slate-900 text-slate-400'
                }`}
              >
                <Button
                  variant="quiet"
                  size="compact"
                  data-testid={`rename-conversation-${room.room_id}`}
                  aria-label={copy.renameLabel(room.title || room.room_id)}
                  title={copy.rename}
                  onClick={() => setRenamingId(room.room_id)}
                >
                  <RenameIcon className="w-3 h-3" />
                </Button>
                <Button
                  variant="quiet"
                  size="compact"
                  data-testid={`delete-conversation-${room.room_id}`}
                  aria-label={copy.deleteLabel(room.title || room.room_id)}
                  title={copy.delete}
                  onClick={() => setDeleting(room)}
                >
                  <DeleteIcon className="w-3 h-3" />
                </Button>
              </span>
            </div>
          ),
        )}

        {/* Not a button to open: the Core refuses to read it, so there is nothing to show. */}
        {unreadableRoomIds.map((roomId) => (
          <div
            key={`unreadable-${roomId}`}
            data-testid={`unreadable-conversation-${roomId}`}
            className="group relative flex items-center rounded-lg border border-transparent text-slate-500 hover:bg-slate-900/60"
          >
            <span className="flex-1 min-w-0 px-2 py-1.5 italic">{copy.unreadableTitle}</span>
            <span className="absolute right-1 top-1/2 -translate-y-1/2 z-10 flex items-center rounded-md px-0.5 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-within:opacity-100 transition-opacity bg-slate-900 text-slate-400">
              <Button
                variant="quiet"
                size="compact"
                data-testid={`delete-unreadable-conversation-${roomId}`}
                aria-label={copy.unreadableDeleteLabel}
                title={copy.delete}
                onClick={() => setDeletingUnreadable(roomId)}
              >
                <DeleteIcon className="w-3 h-3" />
              </Button>
            </span>
          </div>
        ))}
      </div>

      {deleting ? (
        <DeleteConversationDialog
          useEscape={useEscape}
          room={deleting}
          onConfirm={async () => {
            await onDeleteRoom(deleting.room_id);
            setDeleting(null);
          }}
          onClose={() => setDeleting(null)}
          copy={copy.deleteDialog}
        />
      ) : null}
      {deletingUnreadable !== null ? (
        <DeleteConversationDialog
          useEscape={useEscape}
          room={null}
          onConfirm={async () => {
            await onDeleteRoom(deletingUnreadable);
            setDeletingUnreadable(null);
          }}
          onClose={() => setDeletingUnreadable(null)}
          copy={copy.deleteDialog}
        />
      ) : null}
    </div>
  );
};

/**
 * When a conversation was last active, in words ("5 minutes ago", "yesterday").
 *
 * A stamp that will not parse renders nothing rather than "Invalid Date": the Core always
 * writes one, so this is a hand-edited record, and the row is still findable by its title.
 */
export const LastActive: React.FC<{
  updatedAt: string;
  now: number;
  roomId: string;
  copy: ConversationListCopy;
}> = ({ updatedAt, now, roomId, copy }) => {
  const label = copy.lastActive(updatedAt, now);
  if (label === null) return null;
  return (
    <time
      dateTime={updatedAt}
      data-testid={`conversation-last-active-${roomId}`}
      title={copy.lastActiveTitle(updatedAt)}
      className="shrink-0 text-[10px] text-slate-500"
    >
      {label}
    </time>
  );
};
