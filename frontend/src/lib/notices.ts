import { fmt, plural, type Messages, type Values } from '../i18n';
import { placeholders } from '../i18n/format';

/**
 * The words of a note the Core wrote, in the reader's language (multilingual-ui.md §3.3).
 *
 * A `/loop` note is stored with a `code` and its `params`, and worded here each time it is
 * drawn, so switching languages re-words the whole history. The stored `content` is the
 * English fallback, and it is what shows when there is no code (a note stored before codes
 * existed), when the code is one this head does not know (a newer Core), or when the params
 * lack a value the sentence needs. A note is never left blank or half-filled.
 */

export type NoticeParams = Readonly<Record<string, string | number>>;

type NoticeCatalog = Messages['notices'];

/** An interval of `seconds`, the way the Core's English fallback writes it. */
export const intervalText = (seconds: number, t: NoticeCatalog): string =>
  seconds < 60 || seconds % 60 !== 0
    ? plural(t.interval.seconds, seconds)
    : plural(t.interval.minutes, seconds / 60);

const isKnownCode = (code: string, t: NoticeCatalog): code is keyof NoticeCatalog['codes'] =>
  Object.prototype.hasOwnProperty.call(t.codes, code);

export const noticeText = (
  note: { content: string; code?: string | null; params?: NoticeParams | null },
  t: NoticeCatalog,
): string => {
  const { code, content } = note;
  if (!code || !isKnownCode(code, t)) return content;
  const template = t.codes[code];
  const values: Record<string, string | number> = { ...(note.params ?? {}) };
  const seconds = values.interval_seconds;
  if (typeof seconds === 'number') values.interval = intervalText(seconds, t);
  if (!placeholders(template).every((name) => Object.prototype.hasOwnProperty.call(values, name))) {
    return content;
  }
  return fmt(template, values as Values);
};
