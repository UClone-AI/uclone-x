import React from 'react';
import { PersonaInfo } from '../../types';
import { PersonaDetail as KitPersonaDetail } from '../../ui-kit';
import type { PersonaDetailCopy } from '../../ui-kit';

/**
 * Read-only detail for one persona — issue #1056, scope (a).
 *
 * `GET /api/personas` (`src/uclone_x/ui/app.py`, `list_personas`) has always returned nine
 * fields per persona. `WorkspaceSidebar`'s Agents list rendered two of them (`name`, `role`)
 * and dropped the rest, including the two fields that are safety boundaries rather than
 * cosmetics: `enable_write_tools` and `enable_subagent_tools` say whether a persona may touch
 * the filesystem or spawn sub-agents, and neither was visible anywhere in the head.
 *
 * This is **inspection only**. Scope (b) — create and edit — shipped in #892 as a separate
 * editor: `POST /api/personas` and `PUT /api/personas/{name}`, reached from Settings
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
 * *idea* -- a compact read-only persona card -- from the fields `/api/personas` already
 * returns, using only what this file itself defines.
 *
 * The markup now lives in the kit (`ui-kit/rail/PersonaDetail.tsx`, #1158), which holds no
 * words. This file is the head's side of it: the sentences below, bound to the kit component.
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
export const NO_TOOLS_CAUSE = 'No tool list — this clone can use every tool the runtime has.';

/** `enable_write_tools` as a plain-language capability statement, never a bare boolean. */
export const describeWriteAccess = (enabled: boolean | undefined): string =>
  enabled ? 'Can write files' : 'Cannot write files';

/** `enable_subagent_tools` as a plain-language capability statement, never a bare boolean. */
export const describeSubagentAccess = (enabled: boolean | undefined): string =>
  enabled ? 'Can spawn sub-agents' : 'Cannot spawn sub-agents';

/** Every word the persona card shows, as this head words it. */
export const PERSONA_DETAIL_COPY: PersonaDetailCopy = {
  model: 'Model',
  temperature: 'Temperature',
  maxTokens: 'Max tokens',
  tools: 'Tools',
  notSet: 'not set',
  noTools: NO_TOOLS_CAUSE,
  writeAccess: describeWriteAccess,
  subagentAccess: describeSubagentAccess,
};

export const PersonaDetail: React.FC<PersonaDetailProps> = ({ persona }) => (
  <KitPersonaDetail persona={persona} copy={PERSONA_DETAIL_COPY} />
);
