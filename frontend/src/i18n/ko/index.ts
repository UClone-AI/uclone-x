/**
 * The Korean catalog. Typed as `Messages`, so it cannot miss a key English has; the parity
 * cases in `i18n.test.tsx` also refuse an extra key and a sentence that drops or renames a
 * placeholder.
 */
import type { Messages } from '../en';
import cloneProfile from '../locales/ko/cloneProfile.json';
import composer from '../locales/ko/composer.json';
import conversation from '../locales/ko/conversation.json';
import diagnostics from '../locales/ko/diagnostics.json';
import dock from '../locales/ko/dock.json';
import mcp from '../locales/ko/mcp.json';
import notices from '../locales/ko/notices.json';
import personaEditor from '../locales/ko/personaEditor.json';
import settings from '../locales/ko/settings.json';
import skills from '../locales/ko/skills.json';
import conversationList from '../locales/ko/conversationList.json';
import emptyStates from '../locales/ko/emptyStates.json';
import personaDetail from '../locales/ko/personaDetail.json';
import rail from '../locales/ko/rail.json';
import time from '../locales/ko/time.json';
import toolSteps from '../locales/ko/toolSteps.json';
import usage from '../locales/ko/usage.json';
import avatar from '../locales/ko/avatar.json';
import images from '../locales/ko/images.json';
import links from '../locales/ko/links.json';

export const ko: Messages = {
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
  images,
  links,
};
