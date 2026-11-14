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

import type { ProviderFailureKind, RoomProviderFailure, RoomTurnRefusal } from '../types';
import { en, type Messages } from '../i18n/en';
import { fmt, plural } from '../i18n/format';

/**
 * The sentences below, in the reader's language (`conversation.outcome`).
 *
 * Every function takes them last and defaults to English, so the developer-mode Turn surface,
 * which stays English until step 7 of `multilingual-ui.md`, calls them unchanged.
 */
export type OutcomeCopy = Messages['conversation']['outcome'];

const ENGLISH: OutcomeCopy = en.conversation.outcome;

/**
 * What to tell the user about a turn the Core refused, beside the reason it gave
 * (`outcome.refusalRemedy`).
 *
 * Such a row offers no Retry: a spent budget refuses every retry, and the button used to
 * append one identical failure per press. A refusal kind this head does not know yet still
 * withholds Retry, since the Core says a retry fails the same way.
 */
export const refusalRemedy = (refusal: RoomTurnRefusal, copy: OutcomeCopy = ENGLISH): string =>
  (copy.refusalRemedy as Partial<Record<string, string>>)[refusal] ?? copy.refusalRemedyOther;

/**
 * Where to act on a provider failure a retry can cure, when there is somewhere (#1630).
 *
 * The Core's sentence says what stopped and whose side it is on, and leaves *where* to the
 * head (P8). A failure that is not a refusal still offers Retry; this is said beside it.
 * The two refusals take their remedy from `outcome.refusalRemedy`, as every refusal does.
 */
export const providerFailureRemedy = (
  kind: ProviderFailureKind,
  copy: OutcomeCopy = ENGLISH,
): string | null => (copy.providerRemedy as Partial<Record<string, string>>)[kind] ?? null;

/**
 * The clone's own model ref, split at its first `/`, when that model is what failed; else
 * `null` (model-gateway.md §3.6).
 *
 * Decided by the Core's `clone` and `model_ref`, never by `action`: the Core leaves `action`
 * out when no system default can serve the turn (#2424), and the failure is still the
 * clone's own model's.
 */
const ownModel = (
  failure: RoomProviderFailure | null | undefined,
): { connection: string; model: string } | null => {
  const ref = failure?.model_ref ?? null;
  if (!failure?.clone || !ref) return null;
  const slash = ref.indexOf('/');
  return slash > 0 ? { connection: ref.slice(0, slash), model: ref.slice(slash + 1) } : null;
};

const HOSTED_CLOUD_CONNECTIONS = new Set(['gemini', 'openai', 'anthropic']);

/**
 * Where to act when the clone's own model is what failed, or `null` when it is not (#2418).
 *
 * A clone's own model is chosen in the clone editor, not in Settings, so the remedy for a
 * system default ("Choose another model in Settings") sends the person to the wrong place.
 * A hosted cloud API (`gemini`, `openai`, `anthropic`) is not a user-run daemon, so
 * `provider_unreachable` there points to choosing another model in the clone editor rather
 * than checking that the connection is running.
 */
export const ownModelRemedy = (
  label: string,
  copy: OutcomeCopy,
  failure: RoomProviderFailure | null | undefined,
): string | null => {
  const own = ownModel(failure);
  if (!failure || !own) return null;
  const remedyKind =
    failure.kind === 'provider_unreachable' &&
    HOSTED_CLOUD_CONNECTIONS.has(own.connection.toLowerCase())
      ? 'model_unavailable'
      : failure.kind;
  const template = (copy.ownModelRemedy as Partial<Record<string, string>>)[remedyKind];
  return template ? fmt(template, { label, connection: own.connection }) : null;
};

/**
 * The sentence for a turn that failed at the model's provider, from the Core's structured
 * fields alone (#2167): the failure `kind`, the `provider`'s display name, and, when the
 * clone's own model was the cause (model-gateway.md §3.6), the `model_ref` it names.
 *
 * Never `message`: that is the Core's English sentence, and on an own-model failure it is
 * the clone's name and ref spliced into the provider's text. A ref is split at its first
 * `/` (model-gateway.md §3.1): the model is named, and its connection stands in for the
 * provider when the Core names none.
 */
export const providerFailureSentence = (
  label: string,
  failure: RoomProviderFailure,
  copy: OutcomeCopy = ENGLISH,
): string => {
  const own = ownModel(failure);
  const provider = failure.provider?.trim() || own?.connection || copy.providerUnknown;
  const template =
    (copy.providerCause as Partial<Record<string, string>>)[failure.kind] ??
    copy.providerCauseOther;
  const cause = fmt(template, { provider });
  if (own) return fmt(copy.ownModelFailed, { label, model: own.model, cause });
  return fmt(copy.providerFailed, { label, cause });
};

/**
 * The sentence for a turn that refused a step because its tools' results did not fit the
 * model's window (#2264), or that ended with nothing to show or thinking past its budget
 * (#2264, #2202), from the Core's `step_refusal` code alone, or `null` for a code
 * this head has no sentence for. Never the row's `error`, which is the Core's English.
 */
export const stepRefusalSentence = (
  label: string,
  code: string | null | undefined,
  copy: OutcomeCopy = ENGLISH,
): string | null => {
  if (!code) return null;
  const cause = (copy.stepCause as Partial<Record<string, string>>)[code];
  return cause ? fmt(copy.stepFailed, { label, cause }) : null;
};

/**
 * What a failed turn's row says happened (#1408).
 *
 * Never the row's `error` field: the Core writes it as `"<ExceptionClass>: <message>"`
 * (or the agent's own failure text), which is the cause for the log's reader and can hold
 * class names and file paths. The row says which of three things happened, from the
 * fields the Core states rather than from `error`'s wording (#969): the turn failed at the
 * model's provider, it was refused, it was stopped before it finished (`completed: false`),
 * or something went wrong while it answered.
 *
 * A provider failure says what stopped and whose side it is on (#1630): "something went
 * wrong" over a retired model or a rejected key reads as a fault in the app. It is built
 * here, in the reader's language, from the fields the Core states (`providerFailureSentence`),
 * never from the Core's English `message` (#2167).
 */
export const turnFailureSentence = (
  label: string,
  refusal: RoomTurnRefusal | null | undefined,
  completed: boolean,
  providerFailure?: RoomProviderFailure | null,
  copy: OutcomeCopy = ENGLISH,
): string => {
  if (providerFailure) return providerFailureSentence(label, providerFailure, copy);
  if (refusal) {
    const reason =
      (copy.refusalReason as Partial<Record<string, string>>)[refusal] ?? copy.refusalReasonOther;
    return fmt(copy.refused, { label, reason });
  }
  if (!completed) return fmt(copy.stopped, { label });
  return fmt(copy.wentWrong, { label });
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
  copy: OutcomeCopy = ENGLISH,
): string | null => {
  const missed = unsaved ?? 0;
  if (missed <= 0) return null;
  const total = Math.max(tried ?? 0, missed);
  if (missed === total) return plural(copy.memoryNoneSaved, total, { label });
  return plural(copy.memorySomeUnsaved, missed, { label, total });
};
