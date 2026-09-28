/**
 * The head's language, and the catalog every component reads its words from.
 *
 * The choice is read from and saved to the Core (`ui_language` in `/api/settings`), because
 * the CLI and the Core's conversation notices follow it too (ui-authoring §3). The browser
 * keeps only a paint hint, so a reload does not flash English at a user who chose Korean
 * before the settings read returns.
 *
 * Outside a provider, `useCopy()` is the English catalog and `useLocale()` a fixed English
 * locale. A component test therefore renders English with no wrapper, which is what every
 * existing test expects.
 */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { en, type Messages } from './en';
import { ko } from './ko';
import {
  browserLanguageHints,
  isUiLanguage,
  readPaintHint,
  resolveLanguage,
  writePaintHint,
  type Language,
  type UiLanguage,
} from './language';

export const CATALOGS: Readonly<Record<Language, Messages>> = { en, ko };

export interface Locale {
  /** What the user chose, `'system'` included. */
  choice: UiLanguage;
  /** What the screens are written in: the choice, or for `'system'` the browser's language. */
  language: Language;
  /**
   * Apply `choice` at once and save it to the Core. Rejects when the save fails, after
   * putting the previous choice back, so the screen never shows a language that was not kept.
   */
  setChoice: (choice: UiLanguage) => Promise<void>;
  /** What `'system'` resolves to on this browser, so the control can say it. */
  systemLanguage: Language;
}

const FIXED_ENGLISH: Locale = {
  choice: 'en',
  language: 'en',
  setChoice: async () => {},
  systemLanguage: 'en',
};

const LocaleContext = createContext<Locale>(FIXED_ENGLISH);
const CopyContext = createContext<Messages>(en);

export const useLocale = (): Locale => useContext(LocaleContext);
export const useCopy = (): Messages => useContext(CopyContext);

interface LocaleProviderProps {
  children: React.ReactNode;
  /** The browser's languages; a test passes its own instead of the runtime's. */
  hints?: readonly string[];
}

export const LocaleProvider: React.FC<LocaleProviderProps> = ({ children, hints }) => {
  const [choice, setChoiceState] = useState<UiLanguage>(readPaintHint);
  const [browserHints, setBrowserHints] = useState<readonly string[]>(() => hints ?? browserLanguageHints());
  // A choice made before the first read returns must not be overwritten by that read.
  const chosenLocally = useRef(false);

  useEffect(() => {
    if (hints !== undefined) {
      setBrowserHints(hints);
      return undefined;
    }
    const onChange = () => setBrowserHints(browserLanguageHints());
    window.addEventListener('languagechange', onChange);
    return () => window.removeEventListener('languagechange', onChange);
  }, [hints]);

  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const res = await fetch('/api/settings');
        if (!res.ok) return;
        const data: { ui_language?: unknown } = await res.json();
        if (live && !chosenLocally.current && isUiLanguage(data.ui_language)) {
          setChoiceState(data.ui_language);
        }
      } catch {
        // No answer: keep painting the hint. Settings says so when it cannot load.
      }
    })();
    return () => {
      live = false;
    };
  }, []);

  const language = resolveLanguage(choice, browserHints);
  const systemLanguage = resolveLanguage('system', browserHints);

  useEffect(() => {
    document.documentElement.lang = language;
    writePaintHint(choice);
  }, [choice, language]);

  const setChoice = useCallback(
    async (next: UiLanguage) => {
      const previous = choice;
      chosenLocally.current = true;
      setChoiceState(next);
      try {
        const res = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ui_language: next }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
      } catch (err) {
        setChoiceState(previous);
        throw err;
      }
    },
    [choice],
  );

  const locale = useMemo<Locale>(
    () => ({ choice, language, setChoice, systemLanguage }),
    [choice, language, setChoice, systemLanguage],
  );

  return (
    <LocaleContext.Provider value={locale}>
      <CopyContext.Provider value={CATALOGS[language]}>{children}</CopyContext.Provider>
    </LocaleContext.Provider>
  );
};
