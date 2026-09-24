import React, { useEffect, useId, useRef, useState } from 'react';
import { Button } from '../primitives/Button';
import type { DeleteConversationCopy, RailRoom, UseKitEscape } from './types';

interface DeleteConversationDialogProps {
  /**
   * The row as it was when Delete was pressed. Held here so the dialog outlives the row.
   *
   * `null` for a conversation whose saved copy could not be read (#1440): there is no title,
   * count or roster to name, and the dialog says so in `copy.unreadable*` instead.
   */
  room: RailRoom | null;
  /** Rejects with the Core's refusal, which the dialog shows and stays open for. */
  onConfirm: () => Promise<void>;
  onClose: () => void;
  copy: DeleteConversationCopy;
  /** Escape's registry, injected: the kit may not import the head's (#1036, #1158). */
  useEscape: UseKitEscape;
}

/**
 * The confirmation a conversation's deletion needs, because the Core keeps no undo.
 *
 * A delete guarded only by being hover-only cannot be reached from a phone, and once it can be
 * reached it sits right beside the row a thumb taps to open the conversation. So the dialog
 * says what goes -- the transcript, and who was in it -- and focuses Cancel, so a stray Enter
 * keeps everything.
 */
export const DeleteConversationDialog: React.FC<DeleteConversationDialogProps> = ({
  room,
  onConfirm,
  onClose,
  copy,
  useEscape,
}) => {
  const [refusal, setRefusal] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const bodyId = useId();

  useEffect(() => {
    cancelRef.current?.focus();
  }, []);

  const confirm = async () => {
    setDeleting(true);
    setRefusal(null);
    try {
      await onConfirm();
    } catch (err) {
      setRefusal(err instanceof Error ? err.message : String(err));
      setDeleting(false);
      cancelRef.current?.focus();
    }
  };

  // Escape closes the dialog, but only when no higher-precedence layer claims it first
  // (#1036) -- this dialog is itself top-precedence, so in practice it always wins.
  useEscape('dialog', true, () => {
    if (!deleting) onClose();
  });

  // Tab stays inside while the dialog is up: `aria-modal` tells a screen reader so, and
  // a keyboard must not wander into the rail behind the overlay either.
  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Tab' || !dialogRef.current) return;
    const focusable = Array.from(
      dialogRef.current.querySelectorAll<HTMLButtonElement>('button:not([disabled])'),
    );
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const heading =
    room === null ? copy.unreadableHeading : copy.heading(room.title || room.room_id);
  const body =
    room === null
      ? copy.unreadableBody
      : copy.body(room.message_count, room.agent_ids.length > 0 ? room.agent_ids.join(', ') : copy.noAgents);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70">
      <div
        ref={dialogRef}
        data-testid="delete-conversation-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={bodyId}
        onKeyDown={onKeyDown}
        className="w-full max-w-sm rounded-xl border border-slate-800 bg-slate-900 p-4 text-sm text-slate-200"
      >
        <h2 id={titleId} className="font-semibold text-slate-100 break-words">
          {heading}
        </h2>
        <p id={bodyId} className="mt-2 text-xs leading-relaxed text-slate-400">
          {body}
        </p>
        {refusal ? (
          <p role="alert" className="mt-3 text-xs leading-relaxed text-rose-300">
            {refusal}
          </p>
        ) : null}
        <div className="mt-4 flex justify-end gap-2">
          <Button ref={cancelRef} onClick={onClose} disabled={deleting}>
            {refusal ? copy.close : copy.cancel}
          </Button>
          <Button
            variant="danger"
            data-testid="confirm-delete-conversation"
            onClick={() => void confirm()}
            disabled={deleting}
          >
            {deleting ? copy.confirming : copy.confirm}
          </Button>
        </div>
      </div>
    </div>
  );
};
