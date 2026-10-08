import React, { useMemo } from 'react';
import { PersonaInfo } from '../../types';
import { PersonaDetail as KitPersonaDetail } from '../../ui-kit';
import type { PersonaDetailCopy } from '../../ui-kit';
import { useCopy, type Messages } from '../../i18n';
import { en } from '../../i18n/en';

/**
 * Read-only detail for one persona — issue #1056, scope (a).
 *
 * The clone listing (`GET /api/clones`, `src/uclone_x/ui/clones.py`; `GET /api/personas`
 * until #1814) has always returned nine fields per persona. `WorkspaceSidebar`'s Agents list rendered two of them (`name`, `role`)
 * and dropped the rest, including the two fields that are safety boundaries rather than
 * cosmetics: `enable_write_tools` and `enable_subagent_tools` say whether a persona may touch
 * the filesystem or spawn sub-agents, and neither was visible anywhere in the head.
 *
 * This is **inspection only**. Scope (b) — create and edit — shipped in #892 as a separate
 * editor: `POST /api/clones` and `PUT /api/clones/{clone}`, reached from Settings
 * (`components/personas/PersonaManager.tsx`), after the owner overrode the design doc's
 * deferral of the write path on 2026-09-19. This card stays read-only; the rail it sits in is
 * not where personas are edited.
 *
 * No component in this tree matched this shape closely enough to reuse: `ToolDetail.tsx` (since
 * retired into Activity & Tools) was the nearest precedent (a read-only drawer for one selected item, in this same directory)
 * and this followed its layout conventions -- label above value, `text-[10px] uppercase
 * font-sans` section headers -- rather than introducing a new visual language.
 *
 * A sibling repository (`uclone2`) has a `PersonaPreviewCard.tsx` the originating issue named
 * as a possible reference. It was not read or copied: UClone-X is Apache-2.0 and headed for an
 * OSS split, so copying source across that boundary is a licensing question only
 * the maintainers can answer, and it imports state (`zustand` stores, a `botsApi`) and a
 * component kit that do not exist in this tree regardless. This component reimplements the
 * *idea* -- a compact read-only persona card -- from the fields the listing already
 * returns, using only what this file itself defines.
 *
 * The markup now lives in the kit (`ui-kit/rail/PersonaDetail.tsx`, #1158), which holds no
 * words. This file is the head's side of it: the `personaDetail` catalog's sentences, in the
 * screen's language, bound to the kit component.
 */

interface PersonaDetailProps {
  persona: PersonaInfo;
}

/**
 * `allowed_tools` empty-state sentence, following the naming and shape of
 * `frontend/src/lib/emptyStates.ts` (a plain sentence naming what is absent, not a blank
 * container) without adding a fourth entry to that file's own rail-region set — a persona
 * with no tools is a fact about the persona, not a "the runtime has not told us yet" cause,
 * so it does not share that file's three-way `modelConfigured`/`agentCount` decision.
 *
 * The sentence used to read "this persona can only converse", which was the opposite of the
 * runtime: an empty `allowed_tools` is no restriction (`ScopedToolRegistry` and
 * `BaseAgent._apply_persona_tool_scope` both treat it so), not a restriction to nothing.
 */
export const NO_TOOLS_CAUSE = en.personaDetail.noTools;

/** `enable_write_tools` as a plain-language capability statement, never a bare boolean. */
export const describeWriteAccess = (
  enabled: boolean | undefined,
  copy: Messages['personaDetail'] = en.personaDetail,
): string => (enabled ? copy.writeAccess.on : copy.writeAccess.off);

/** `enable_subagent_tools` as a plain-language capability statement, never a bare boolean. */
export const describeSubagentAccess = (
  enabled: boolean | undefined,
  copy: Messages['personaDetail'] = en.personaDetail,
): string => (enabled ? copy.subagentAccess.on : copy.subagentAccess.off);

/** Every word the persona card shows, as this head words it, from the `personaDetail` catalog. */
export const personaDetailCopy = (copy: Messages['personaDetail']): PersonaDetailCopy => ({
  model: copy.model,
  temperature: copy.temperature,
  maxTokens: copy.maxTokens,
  tools: copy.tools,
  notSet: copy.notSet,
  systemDefault: copy.systemDefault,
  noTools: copy.noTools,
  writeAccess: (enabled) => describeWriteAccess(enabled, copy),
  subagentAccess: (enabled) => describeSubagentAccess(enabled, copy),
});

/** The persona card's words in English, for callers outside a `LocaleProvider`. */
export const PERSONA_DETAIL_COPY: PersonaDetailCopy = personaDetailCopy(en.personaDetail);

export const PersonaDetail: React.FC<PersonaDetailProps> = ({ persona }) => {
  const t = useCopy();
  const copy = useMemo(() => personaDetailCopy(t.personaDetail), [t]);
  return <KitPersonaDetail persona={persona} copy={copy} />;
};
