import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// Without this, a component left mounted by one test is still in the document for the
// next one, and a `getByTestId` that should be unique starts throwing on duplicates —
// a failure mode that looks like a bug in the assertion rather than in the harness.
afterEach(() => {
  cleanup();
});

// No test opens a real socket. jsdom's own `WebSocket` dials the address it is given, so a
// component that keeps one open (the dock's Browser tab, `lib/browserLive.ts`) would reach
// for a server that is not there and retry every few seconds through the rest of the file.
// This one never connects; a test about the socket hands the hook its own fake.
class InertWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  readonly readyState = 0;
  binaryType = 'blob';
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  constructor(readonly url: string) {}
  send(): void {}
  close(): void {}
}
globalThis.WebSocket = InertWebSocket as unknown as typeof WebSocket;
