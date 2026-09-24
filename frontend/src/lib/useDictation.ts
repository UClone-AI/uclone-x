import { useCallback, useEffect, useRef, useState } from 'react';
import {
  DICTATION_UNSUPPORTED,
  appendPhrase,
  dictationErrorMessage,
  speechRecognitionConstructor,
  type DictationState,
  type SpeechRecognitionEventLike,
  type SpeechRecognitionLike,
} from './dictation';

/**
 * Drive the browser's speech recognition for one composer.
 *
 * `onPhrase` is handed only final results. Interim ones are off: a box that rewrites
 * itself while the reader is still speaking cannot be edited, and the half-heard words it
 * shows are not what will be sent.
 *
 * The recognition object is built on the first press rather than on mount, because
 * constructing one is what some browsers treat as the moment to ask for the microphone --
 * and a conversation that asks for the microphone merely by being opened is asking for
 * something the reader has not reached for.
 */
export function useDictation(onPhrase: (phrase: string) => void): {
  state: DictationState;
  /** Start, or stop if already listening. Safe to call in any state. */
  toggle: () => void;
  /** Put the control back to `idle` from an error, without starting. */
  dismissError: () => void;
} {
  const [state, setState] = useState<DictationState>({ kind: 'idle' });
  const recognition = useRef<SpeechRecognitionLike | null>(null);
  // Read inside the browser's callbacks, which close over the render that made them.
  const phraseRef = useRef(onPhrase);
  phraseRef.current = onPhrase;
  const stoppedByReader = useRef(false);

  // Leaving the conversation while it is listening must release the microphone. Without
  // this the browser keeps the tab's recording indicator lit after the surface is gone.
  useEffect(
    () => () => {
      recognition.current?.abort();
      recognition.current = null;
    },
    [],
  );

  const toggle = useCallback(() => {
    if (recognition.current) {
      stoppedByReader.current = true;
      recognition.current.stop();
      return;
    }

    const Recognition = speechRecognitionConstructor();
    if (Recognition === null) {
      setState({ kind: 'unsupported', reason: DICTATION_UNSUPPORTED });
      return;
    }

    const session = new Recognition();
    // Set on the instance rather than passed to the constructor: the prefixed
    // implementations read `lang` at `start()`, so a language set any earlier is ignored.
    session.lang = navigator.language || 'en-US';
    session.continuous = true;
    session.interimResults = false;
    session.onresult = (event: SpeechRecognitionEventLike) => {
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (result.isFinal) phraseRef.current(result[0].transcript);
      }
    };
    session.onerror = (event: { error: string }) => {
      if (event.error === 'aborted') return;
      setState({ kind: 'error', message: dictationErrorMessage(event.error) });
    };
    session.onend = () => {
      recognition.current = null;
      // An error has already put its own sentence on the screen, and `onend` fires after
      // it. Overwriting that with `idle` is how the reader comes to press a mic that
      // silently does nothing and is told nothing about why.
      setState((current) => (current.kind === 'error' ? current : { kind: 'idle' }));
      stoppedByReader.current = false;
    };

    try {
      session.start();
    } catch (err) {
      setState({ kind: 'error', message: dictationErrorMessage(String(err)) });
      return;
    }
    recognition.current = session;
    setState({ kind: 'listening' });
  }, []);

  const dismissError = useCallback(() => setState({ kind: 'idle' }), []);

  return { state, toggle, dismissError };
}

export { appendPhrase };
