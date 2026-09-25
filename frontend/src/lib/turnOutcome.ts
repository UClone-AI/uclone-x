/**
 * What a turn's bubble says about a refusal (#998, #1000, #969).
 *
 * This module used to hold the rest of "what a bubble says once its turn is over" as
 * well -- `finishedTurnContent`, `turnFailed`, `resultOutcome`, `stoppedExecutions`,
 * `withNotice`, `INTERRUPTED_NOTICE`, `turnChip`, `TURN_CHIP_LABEL`. All of it read the
 * shape `POST /api/chat/stream` sent and was rendered by `PlaygroundTab`; #1208 deleted
 * both. The conversation states a turn's end from `RoomState`, which the Core writes,
 * rather than from a stream result this head has to interpret -- so only the remedy
 * below, which is a sentence and not an interpretation, is shared by the two.
 */

import type { RoomTurnRefusal } from '../types';

/**
 * What to tell the user about a turn the Core refused, beside the reason it gave.
 *
 * Such a row offers no Retry: a spent budget refuses every retry, and the button used to
 * append one identical failure per press. A refusal kind this head does not know yet still
 * withholds Retry, since the Core says a retry fails the same way.
 */
const REFUSAL_REMEDY: Record<RoomTurnRefusal, string> = {
  budget_exceeded: 'Trying again would be refused too. Start a new conversation to continue.',
  model_without_tools:
    'Pick a model that supports tools (for example qwen3:8b) in Settings, then send your message again.',
};

export const refusalRemedy = (refusal: RoomTurnRefusal): string =>
  REFUSAL_REMEDY[refusal] ?? 'Trying again would be refused for the same reason.';

/** Why the Core refused a turn, as the end of a sentence about the seat. */
const REFUSAL_REASON: Record<RoomTurnRefusal, string> = {
  budget_exceeded: 'it has used all the tokens this conversation allows',
  model_without_tools: "its model can't use tools, which clones need",
};

/**
 * What a failed turn's row says happened (#1408).
 *
 * Never the row's `error` field: the Core writes it as `"<ExceptionClass>: <message>"`
 * (or the agent's own failure text), which is the cause for the log's reader and can hold
 * class names and file paths. The row says which of three things happened, from the
 * fields the Core states rather than from `error`'s wording (#969): the turn was refused,
 * it was stopped before it finished (`completed: false`), or something went wrong while
 * it answered.
 */
export const turnFailureSentence = (
  label: string,
  refusal: RoomTurnRefusal | null | undefined,
  completed: boolean,
): string => {
  if (refusal) {
    const reason = REFUSAL_REASON[refusal] ?? 'the app refused to run it';
    return `${label} couldn't finish this turn: ${reason}.`;
  }
  if (!completed) return `${label} was stopped before finishing this turn.`;
  return `${label} couldn't finish this turn because something went wrong while it was answering.`;
};

/**
 * What to tell the user when a turn asked to save facts to memory and some were never saved
 * (#1375), or `null` when there is nothing to say. Built from the Core's counts alone: the
 * tool's error is written for the model -- argument dumps, class names -- and is not copy.
 */
export const memoryUnsavedNotice = (
  label: string,
  tried: number | undefined,
  unsaved: number | undefined,
): string | null => {
  const missed = unsaved ?? 0;
  if (missed <= 0) return null;
  const total = Math.max(tried ?? 0, missed);
  if (missed === total) {
    return total === 1
      ? `${label} tried to save something to memory, and nothing was saved.`
      : `${label} tried to save ${total} things to memory, and none of them were saved.`;
  }
  return `${label} tried to save ${total} things to memory, and ${missed} of them ${
    missed === 1 ? 'was' : 'were'
  } not saved.`;
};
