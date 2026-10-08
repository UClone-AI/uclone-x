/**
 * The rail's statements of absence, in one place because two regions make them.
 *
 * The two regions are the conversation list at the top of the rail and the Clones
 * section below it. While each owned its own words, they disagreed -- the list said no
 * clones were set up, and the Clones section rendered a row reading "General Assistant",
 * naming a clone that nothing in the runtime had created (#1060). An absence rendered as
 * an entity is what `docs/principles/core-principles.md` P6 forbids, and the fabricated
 * name was the more reassuring of the two statements, so it was the one likelier to be
 * believed.
 *
 * `emptyCause` returns one of the three sentences in the `emptyStates` catalog. A region that shows it shows
 * the same cause and the same remedy as every other region calling this function with
 * the same arguments.
 *
 * Each sentence names a cause and a remedy, because these are three different situations
 * with three different remedies and a bare empty container renders them identically.
 */

import { en, type Messages } from '../i18n/en';

/**
 * Internal constants capturing the default English causes.
 * Consumers read causes via `emptyCause(modelConfigured, agentCount, copy)` rather than
 * importing English strings directly, ensuring caller localization is respected (#1295).
 */
const NO_MODEL_CAUSE = en.emptyStates.noModel;
const NO_AGENTS_CAUSE = en.emptyStates.noAgents;
const NO_CONVERSATIONS_CAUSE = en.emptyStates.noConversations;

const DEFAULT_EMPTY_STATES: Messages['emptyStates'] = {
  noModel: NO_MODEL_CAUSE,
  noAgents: NO_AGENTS_CAUSE,
  noConversations: NO_CONVERSATIONS_CAUSE,
};

/**
 * Why this region is empty, in one sentence naming a cause and a remedy.
 *
 * @param modelConfigured Whether a model is configured. `null` is "the runtime has not
 *   answered yet" and is deliberately not collapsed onto `false`.
 * @param agentCount How many clones the runtime can offer: the rail's clone list, which the
 *   persona catalog fills.
 * @param copy The sentences, in the screen's language (`useCopy().emptyStates`). English when
 *   omitted, which is what a caller outside a `LocaleProvider` reads anyway.
 */
export const emptyCause = (
  modelConfigured: boolean | null,
  agentCount: number,
  copy: Messages['emptyStates'] = DEFAULT_EMPTY_STATES,
): string =>
  modelConfigured === false ? copy.noModel : agentCount === 0 ? copy.noAgents : copy.noConversations;
