import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  DICTATION_UNSUPPORTED,
  appendPhrase,
  dictationErrorMessage,
  speechRecognitionConstructor,
} from './dictation';
import { dictationLang } from './dictation';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('speechRecognitionConstructor', () => {
  // Killed by: frontend/src/lib/dictation.ts :: const ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition;
  // Becomes: const ctor = w.SpeechRecognition;
  it('finds the prefixed constructor, which is the one browsers actually ship', () => {
    class Fake {}
    // Chrome, Edge and Safari all expose it under this name and not the other. A check
    // for the unprefixed name alone reports "this browser cannot listen" on every
    // browser that can.
    expect(speechRecognitionConstructor({ webkitSpeechRecognition: Fake })).toBe(Fake);
    expect(speechRecognitionConstructor({ SpeechRecognition: Fake })).toBe(Fake);
  });

  it('reports none where there is none, rather than a falsy value to call', () => {
    expect(speechRecognitionConstructor({})).toBeNull();
    expect(speechRecognitionConstructor({ SpeechRecognition: undefined })).toBeNull();
  });
});

describe('dictationErrorMessage', () => {
  // Killed by: frontend/src/lib/dictation.ts :: return copy.notAllowed;
  // Becomes: return copy.noSpeech;
  it('names the remedy for a refused microphone, not just the refusal', () => {
    // The one error with an obvious way out. A sentence that says only that it failed
    // leaves the reader nothing to do, which is what this surface calls a defect.
    const said = dictationErrorMessage('not-allowed');
    expect(said).toContain('microphone');
    expect(said).toContain('Allow it for this site');
  });

  it('still names a cause and a remedy for a code it has never seen', () => {
    const said = dictationErrorMessage('some-future-code');
    expect(said).toContain('some-future-code');
    expect(said).toContain('type the message instead');
  });

  it('gives each cause its own sentence', () => {
    const codes = ['not-allowed', 'no-speech', 'audio-capture', 'network'];
    expect(new Set(codes.map(dictationErrorMessage)).size).toBe(codes.length);
  });

  it('offers a way to keep going where the browser cannot listen at all', () => {
    expect(DICTATION_UNSUPPORTED).toContain('Type the message');
  });
});

describe('appendPhrase', () => {
  // Killed by: frontend/src/lib/dictation.ts :: return draft.trim() === '' ? heard : `${draft.replace(/\s+$/, '')} ${heard}`;
  // Becomes: return draft.trim() === '' ? heard : `${draft.replace(/\s+$/, '')}${heard}`;
  it('separates two heard phrases, which otherwise run into one word', () => {
    expect(appendPhrase('check the', 'index')).toBe('check the index');
    expect(appendPhrase('check the ', 'index')).toBe('check the index');
  });

  it('starts an empty box without a leading space to delete', () => {
    expect(appendPhrase('', 'check the index')).toBe('check the index');
    expect(appendPhrase('   ', 'check the index')).toBe('check the index');
  });

  it('leaves the draft alone when nothing was heard', () => {
    expect(appendPhrase('check the index', '   ')).toBe('check the index');
  });
});

describe('dictationLang', () => {
  // Killed by: frontend/src/lib/dictation.ts :: if (browser && browser.split('-')[0].toLowerCase() === language) return browser;
  // Becomes: if (browser) return browser;
  it('listens in the language on screen, not the browser language, when the two differ', () => {
    // A reader who chose Korean on an English browser speaks Korean to the composer.
    expect(dictationLang('ko', 'en-US')).toBe('ko-KR');
    expect(dictationLang('en', 'ko-KR')).toBe('en-US');
  });

  it('keeps the regional spelling of the browser when it is the same language', () => {
    expect(dictationLang('en', 'en-GB')).toBe('en-GB');
    expect(dictationLang('ko', 'ko-KR')).toBe('ko-KR');
    expect(dictationLang('ko', undefined)).toBe('ko-KR');
  });
});
