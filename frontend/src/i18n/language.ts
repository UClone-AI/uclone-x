/**
 * Which language the head writes in.
 *
 * The choice is the Core's (`ui_language` in `/api/settings`), because the CLI and the
 * Core's own conversation notices read it too. `'system'` stays the stored choice and is
 * resolved here from the browser's languages, with the same rule as the Core's
 * `uclone_x.i18n.resolve_language`, so a machine resolves to one language in every head.
 */

import { KO_AUTONYM } from './ko/autonym';

export type Language = 'en' | 'ko';
export type UiLanguage = 'system' | Language;

export const UI_LANGUAGES: readonly UiLanguage[] = ['system', 'en', 'ko'];
export const FALLBACK_LANGUAGE: Language = 'en';

export const isUiLanguage = (value: unknown): value is UiLanguage =>
  typeof value === 'string' && (UI_LANGUAGES as readonly string[]).includes(value);

const languageOf = (hint: string): Language | null => {
  const primary = hint.trim().toLowerCase().replace(/_/g, '-').split('.')[0].split('-')[0];
  return primary === 'ko' || primary === 'en' ? primary : null;
};

/** The choice itself, or for `'system'` the first hint this build can write in. */
export const resolveLanguage = (choice: UiLanguage, hints: readonly string[]): Language => {
  if (choice !== 'system') return choice;
  for (const hint of hints) {
    const found = languageOf(hint);
    if (found !== null) return found;
  }
  return FALLBACK_LANGUAGE;
};

/** The browser's languages in preference order; empty where the browser names none. */
export const browserLanguageHints = (): readonly string[] => {
  if (typeof navigator === 'undefined') return [];
  if (navigator.languages && navigator.languages.length > 0) return navigator.languages;
  return navigator.language ? [navigator.language] : [];
};

/**
 * The last choice this browser saw, so the first paint does not flash English at a user who
 * chose Korean before the settings read returns. A paint hint only: the Core's value wins as
 * soon as it arrives, and a browser with no storage paints `'system'`.
 */
export const PAINT_HINT_KEY = 'uclone-x.ui-language-hint';

export const readPaintHint = (): UiLanguage => {
  try {
    const stored = window.localStorage.getItem(PAINT_HINT_KEY);
    return isUiLanguage(stored) ? stored : 'system';
  } catch {
    return 'system';
  }
};

export const writePaintHint = (choice: UiLanguage): void => {
  try {
    window.localStorage.setItem(PAINT_HINT_KEY, choice);
  } catch {
    /* Storage refused: the next load paints 'system' until the Core answers. */
  }
};

/** Each language by its own name, the same on every screen, for the language control. */
export const LANGUAGE_AUTONYMS: Readonly<Record<Language, string>> = {
  en: 'English',
  ko: KO_AUTONYM,
};
