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
 * `emptyCause` returns one of the three sentences below. A region that shows it shows
 * the same cause and the same remedy as every other region calling this function with
 * the same arguments.
 *
 * Each sentence names a cause and a remedy, because these are three different situations
 * with three different remedies and a bare empty container renders them identically.
 */

/** No model is configured: the setting exists and holds nothing. */
export const NO_MODEL_CAUSE = 'No model is configured. Pick one in Settings.';

/** A model is configured, but the runtime offers no clone to send a message to. */
export const NO_AGENTS_CAUSE = 'No clones are set up yet — add one in Settings.';

/** Model and clones are both present; the user simply has not started anything. */
export const NO_CONVERSATIONS_CAUSE = 'No conversations yet. Press New to start one.';

/**
 * Why this region is empty, in one sentence naming a cause and a remedy.
 *
 * @param modelConfigured Whether a model is configured. `null` is "the runtime has not
 *   answered yet" and is deliberately not collapsed onto `false`.
 * @param agentCount How many clones the runtime can offer. The parameter keeps the wire's
 *   word: it is fed from `AgentInfo[]`, which is Core contract and does not get renamed.
 */
export const emptyCause = (modelConfigured: boolean | null, agentCount: number): string =>
  modelConfigured === false
    ? NO_MODEL_CAUSE
    : agentCount === 0
      ? NO_AGENTS_CAUSE
      : NO_CONVERSATIONS_CAUSE;
