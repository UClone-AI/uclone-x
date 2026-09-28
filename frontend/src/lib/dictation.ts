/**
 * Speaking into the composer, over the browser's own speech recognition.
 *
 * The sibling product does this with the same Web Speech API and no server (#1301 refers).
 * Three of its behaviours are deliberately not carried over, because each one renders a
 * cause the reader cannot see:
 *
 * * an unsupported browser gets a greyed mic and a `title` tooltip, which on a touch
 *   device is unreachable and on a pointer device has to be hunted for. An absence states
 *   its cause in words on the surface (P6), so `unsupported` carries the sentence rather
 *   than a hover.
 * * a recognition error is written to the console and nowhere else. The reader presses
 *   the mic, nothing happens, and the surface says nothing at all.
 * * a refused microphone is not distinguished from any other failure, so the one error
 *   with an obvious remedy -- allow the site -- does not state it.
 *
 * The listening state is also plain rather than pulsing: this surface forbids
 * `animate-pulse` and the ping halo the sibling draws around its mic.
 */

import { en, type Messages } from '../i18n/en';
import { fmt } from '../i18n/format';

/** What the mic control is currently able to do, and what to say when it cannot. */
export type DictationState =
  /**
   * This browser has no speech recognition at all. The surface shows the reason
   * (`composer.dictation.unsupported`) in words, not in a hover.
   */
  | { kind: 'unsupported' }
  | { kind: 'idle' }
  | { kind: 'listening' }
  /**
   * The last attempt stopped, with the browser's code for why. The surface says it with
   * `dictationSentence`, in the reader's language, which names the cause and the remedy.
   */
  | { kind: 'error'; code: string };

/**
 * A recognition session, narrowed to what this module uses.
 *
 * Declared here rather than pulled from a types package: `SpeechRecognition` is not in
 * TypeScript's DOM library, and the shipping implementations are still behind
 * `webkitSpeechRecognition`.
 */
export interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
}

export interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }>;
}

type RecognitionConstructor = new () => SpeechRecognitionLike;

/**
 * The constructor this browser offers, or `null`.
 *
 * Both names are read because the prefixed one is what Chrome, Edge and Safari actually
 * ship; a check for the unprefixed name alone reports "unsupported" on every browser that
 * supports it.
 */
export function speechRecognitionConstructor(scope: unknown = globalThis): RecognitionConstructor | null {
  const w = scope as Record<string, unknown>;
  const ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition;
  return typeof ctor === 'function' ? (ctor as RecognitionConstructor) : null;
}

/** The dictation sentences, in the reader's language (`composer.dictation`). */
export type DictationCopy = Messages['composer']['dictation'];

/** The sentence shown where the mic would be, when the browser has no recognition. */
export const DICTATION_UNSUPPORTED = en.composer.dictation.unsupported;

/**
 * What to say about a recognition error, by the code the browser reports.
 *
 * Every branch names the cause and what would let it through, because a refusal that says
 * only that it failed leaves the reader with nothing to do about it. `aborted` is absent:
 * it is what the browser reports when the reader themselves pressed stop, which is not a
 * failure and is handled before this is called.
 */
export function dictationSentence(code: string, copy: DictationCopy): string {
  switch (code) {
    case 'not-allowed':
    case 'service-not-allowed':
      return copy.notAllowed;
    case 'no-speech':
      return copy.noSpeech;
    case 'audio-capture':
      return copy.audioCapture;
    case 'network':
      return copy.network;
    default:
      return fmt(copy.other, { code });
  }
}

/**
 * `dictationSentence` in English. One argument only, so it can be handed to `map` without the
 * index arriving as a catalog.
 */
export function dictationErrorMessage(code: string): string {
  return dictationSentence(code, en.composer.dictation);
}

/**
 * The language to listen in: the head's own, so a reader who chose Korean is heard in
 * Korean whatever the browser is set to (`multilingual-ui.md` §3.6).
 *
 * The browser's own tag is kept when it is the same language, because it carries the region
 * (`en-GB`, `ko-KR`) the recogniser is tuned by; otherwise the language's usual region.
 */
export function dictationLang(language: string, browser: string | undefined): string {
  if (browser && browser.split('-')[0].toLowerCase() === language) return browser;
  return language === 'ko' ? 'ko-KR' : 'en-US';
}

/**
 * The draft with a freshly heard phrase added to it.
 *
 * Separated by a single space and never by none: two dictated phrases run together into
 * one word otherwise. Trimmed on both sides so that starting from an empty box does not
 * leave the message with a leading space the reader has to notice and delete.
 */
export function appendPhrase(draft: string, phrase: string): string {
  const heard = phrase.trim();
  if (heard === '') return draft;
  return draft.trim() === '' ? heard : `${draft.replace(/\s+$/, '')} ${heard}`;
}
