/**
 * What Settings says when a request fails (#1436).
 *
 * Each sentence says only what the reader can see did not happen, and nothing about why or
 * about what the app still holds: the same sentence is shown for a request that never got an
 * answer and for one answered with a bare failure, and the code cannot tell from either
 * whether a write took effect before the failure.
 */
export const SETTINGS_FAILURE = {
  load: 'The settings could not be loaded.',
  test: 'The connection could not be tested.',
  save: 'Something went wrong saving the settings.',
  install: 'Something went wrong installing the model.',
  remove: 'Something went wrong deleting the model.',
  diagnosticsRead: 'Problem reporting could not be loaded.',
  diagnosticsChoice: 'Something went wrong saving your choice.',
  diagnosticsClear: 'Something went wrong deleting the recorded failures.',
} as const;
