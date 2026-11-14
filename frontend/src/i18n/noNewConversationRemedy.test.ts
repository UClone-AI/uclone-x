import { describe, expect, it } from 'vitest';
import { en } from './en';
import { ko } from './ko';

/**
 * Owner ruling 2026-10-29 (#2424): the product never suggests or pushes a new conversation
 * as the way past an error or a limit. A failed turn and a full context are answered in
 * place (shorten the conversation, change a setting, try again); the rail's New button is
 * the person's own choice and is not a notice.
 *
 * Every string under a conversation's failure notices (`outcome`) and its context notices
 * (`reset`) is read, in both languages, so a remedy added later is held to the same rule.
 */
const strings = (node: unknown, path: string): [string, string][] => {
  if (typeof node === 'string') return [[path, node]];
  if (node && typeof node === 'object') {
    return Object.entries(node as Record<string, unknown>).flatMap(([key, value]) =>
      strings(value, `${path}.${key}`),
    );
  }
  return [];
};

const PUSHES_A_NEW_CONVERSATION = /new conversation|new one|새 대화|새로운 대화/i;

describe('no notice sends the person to a new conversation (#2424)', () => {
  it.each([
    ['en', en],
    ['ko', ko],
  ] as const)('%s: failure and context notices', (_lang, catalog) => {
    const notices = [
      ...strings(catalog.conversation.outcome, 'conversation.outcome'),
      ...strings(catalog.conversation.reset, 'conversation.reset'),
    ];
    expect(notices.length).toBeGreaterThan(10);
    expect(notices.filter(([, text]) => PUSHES_A_NEW_CONVERSATION.test(text))).toEqual([]);
  });
});
