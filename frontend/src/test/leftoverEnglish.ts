import { en, type Messages } from '../i18n/en';
import { ko } from '../i18n/ko';

/**
 * What a Korean screen must not show: the fixed English words of the namespaces it reads.
 *
 * A sentence with a value in it is split around its placeholders, so "Show {name}'s
 * conversations" leaves "Show" and "'s conversations" to look for, and a component that
 * fills the English template with a Korean-screen value is still caught. A fragment Korean
 * also writes (a product or technical term kept in English, such as "Temperature") is not a
 * leftover, and neither is a fragment too short to be a word.
 */

type Tree = { [key: string]: unknown };

const strings = (tree: unknown): string[] =>
  typeof tree === 'string'
    ? [tree]
    : tree && typeof tree === 'object'
      ? Object.values(tree as Tree).flatMap(strings)
      : [];

export const englishFragments = (namespaces: ReadonlyArray<keyof Messages>): string[] => {
  const korean = namespaces.flatMap((ns) => strings(ko[ns])).join('\n');
  const fragments = namespaces
    .flatMap((ns) => strings(en[ns]))
    .flatMap((sentence) => sentence.split(/\{\w+\}/))
    .map((fragment) => fragment.trim())
    .filter((fragment) => fragment.length > 3 && /[a-z]{3}/i.test(fragment) && !korean.includes(fragment));
  return [...new Set(fragments)];
};

/** The words a reader meets under `root`: its text, and the names and tips of its elements. */
export const visibleWords = (root: HTMLElement): string => {
  const attributes = Array.from(root.querySelectorAll('[placeholder],[aria-label],[title]')).flatMap((el) =>
    ['placeholder', 'aria-label', 'title'].map((name) => el.getAttribute(name) ?? ''),
  );
  return [root.textContent ?? '', ...attributes].join('\n');
};

/** The fragments of `namespaces` that `root` shows. */
export const leftoverEnglish = (root: HTMLElement, namespaces: ReadonlyArray<keyof Messages>): string[] => {
  const shown = visibleWords(root);
  return englishFragments(namespaces).filter((fragment) => shown.includes(fragment));
};
