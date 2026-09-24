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

/** What the mic control is currently able to do, and what to say when it cannot. */
export type DictationState =
  /** This browser has no speech recognition at all. `reason` is shown, not hovered. */
  | { kind: 'unsupported'; reason: string }
  | { kind: 'idle' }
  | { kind: 'listening' }
  /** The last attempt stopped. `message` names the cause and the remedy. */
  | { kind: 'error'; message: string };

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

/** The sentence shown where the mic would be, when the browser has no recognition. */
export const DICTATION_UNSUPPORTED =
  'This browser cannot listen. Type the message, or open the conversation in Chrome, Edge or Safari to speak it.';

/**
 * What to say about a recognition error, by the code the browser reports.
 *
 * Every branch names the cause and what would let it through, because a refusal that says
 * only that it failed leaves the reader with nothing to do about it. `aborted` is absent:
 * it is what the browser reports when the reader themselves pressed stop, which is not a
 * failure and is handled before this is called.
 */
export function dictationErrorMessage(code: string): string {
  switch (code) {
    case 'not-allowed':
    case 'service-not-allowed':
      return 'The browser is not letting this page use the microphone. Allow it for this site in the browser, then press the mic again.';
    case 'no-speech':
      return 'Nothing was heard. Press the mic and speak, or type the message instead.';
    case 'audio-capture':
      return 'No microphone was found. Connect one and press the mic again, or type the message instead.';
    case 'network':
      return 'The browser could not reach its speech service. Type the message instead, or press the mic again once the connection is back.';
    default:
      return `Listening stopped (${code}). Press the mic to try again, or type the message instead.`;
  }
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
