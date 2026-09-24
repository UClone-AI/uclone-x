import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// Without this, a component left mounted by one test is still in the document for the
// next one, and a `getByTestId` that should be unique starts throwing on duplicates —
// a failure mode that looks like a bug in the assertion rather than in the harness.
afterEach(() => {
  cleanup();
});
