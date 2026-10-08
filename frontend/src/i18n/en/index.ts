/**
 * The English catalog, and the shape every other catalog is typed against.
 *
 * `Messages` is the type of the English JSON, so English is the source of truth for which keys
 * exist: a key added here and missing from `ko` is a type error, not an English string on a
 * Korean screen. What the type cannot see is a placeholder: every JSON value is a `string`, so
 * `i18n.test.tsx` compares each Korean sentence's placeholders with the English one's.
 *
 * Wording notes that used to sit beside the sentences, and still bind them:
 * - `personaEditor.fields.enableWriteTools` / `enableSubagentTools` read exactly as
 *   `BaseAgent._capability_refusal` quotes them (#1167); rename all three together.
 * - `settings.failure.*` say only what the reader can see did not happen, never why (#1436).
 * - `skills.plainCause.*` never quote the transport ("Failed to fetch") or a status line (#1369).
 * - `diagnostics.panel.intro*` is one sentence split around its emphasised phrase.
 * - `mcp.pairs.unnamed` / `duplicate` take `what` from `mcp.pairs.header` / `envVar`, and
 *   `mcp.row.stillSaved` takes it from `mcp.row.checkHttp` / `checkCommand`.
 * - `gateway.status.*` and `gateway.picker.*` are chosen by a connection's `status`, never the
 *   Core's English `detail` (#1631). No sentence names a model: which models exist is each
 *   connection's listing's to say (model-gateway.md §3.5).
 * - `notices.codes.*` are the Core's stored English fallback word for word (`room/notices.py`'s
 *   `NOTICE_FALLBACK`), so an export reads as an English screen; `test_notice_contract.py`
 *   holds the two equal. Change both together.
 */
import cloneProfile from '../locales/en/cloneProfile.json';
import composer from '../locales/en/composer.json';
import conversation from '../locales/en/conversation.json';
import diagnostics from '../locales/en/diagnostics.json';
import dock from '../locales/en/dock.json';
import mcp from '../locales/en/mcp.json';
import notices from '../locales/en/notices.json';
import personaEditor from '../locales/en/personaEditor.json';
import settings from '../locales/en/settings.json';
import skills from '../locales/en/skills.json';
import conversationList from '../locales/en/conversationList.json';
import emptyStates from '../locales/en/emptyStates.json';
import personaDetail from '../locales/en/personaDetail.json';
import rail from '../locales/en/rail.json';
import time from '../locales/en/time.json';
import toolSteps from '../locales/en/toolSteps.json';
import usage from '../locales/en/usage.json';
import avatar from '../locales/en/avatar.json';
import imageCard from '../locales/en/imageCard.json';
import links from '../locales/en/links.json';
import browser from '../locales/en/browser.json';
import gateway from '../locales/en/gateway.json';

export const en = {
  settings,
  personaEditor,
  skills,
  mcp,
  diagnostics,
  conversation,
  composer,
  toolSteps,
  dock,
  rail,
  conversationList,
  personaDetail,
  emptyStates,
  time,
  cloneProfile,
  notices,
  usage,
  avatar,
  imageCard,
  links,
  browser,
  gateway,
};

export type Messages = typeof en;
