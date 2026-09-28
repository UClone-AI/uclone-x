/**
 * Korean particles that depend on the word before them (#1900).
 *
 * A Korean template cannot know the name it will hold, so the catalog writes both forms of
 * a particle, `{name}이(가)`, and `fmt` settles it here once the name is known: `이` after a
 * final syllable with a 받침, `가` after one without. Only a name that ends in a Hangul
 * syllable is decided. After anything else -- a Latin name, a digit, a symbol -- the paired
 * form is kept as written (author's choice): reading `Qwen3` aloud is the reader's call, and
 * the paired form is what the catalog has always shown.
 *
 * It lives under `ko/` because it is Korean text (the head Hangul guard, `i18n.test.tsx`).
 */

/** The paired form as the catalog writes it, and its two forms: [after a 받침, after none]. */
const PAIRS: Readonly<Record<string, readonly [string, string]>> = {
  '이(가)': ['이', '가'],
  '을(를)': ['을', '를'],
  '은(는)': ['은', '는'],
  '와(과)': ['과', '와'],
  '(으)로': ['으로', '로'],
};

/** A regular-expression alternation matching any paired form above. */
export const KO_PARTICLE_PATTERN = Object.keys(PAIRS)
  .map((pair) => pair.replace(/[()]/g, '\\$&'))
  .join('|');

const FIRST_SYLLABLE = 0xac00;
const LAST_SYLLABLE = 0xd7a3;
/** The final-consonant index of ㄹ in a precomposed syllable. */
const RIEUL = 8;

/**
 * The final consonant (받침) index of `word`'s last character: 0 for none, 1-27 for one, or
 * `null` when that character is not a Hangul syllable.
 */
export const finalConsonant = (word: string): number | null => {
  const last = word.trimEnd().slice(-1);
  if (last === '') return null;
  const code = last.charCodeAt(0);
  if (code < FIRST_SYLLABLE || code > LAST_SYLLABLE) return null;
  return (code - FIRST_SYLLABLE) % 28;
};

/**
 * The form of `pair` (as the catalog writes it, `이(가)`) that follows `word`, or `pair`
 * unchanged when `word` does not end in a Hangul syllable or `pair` is not one this knows.
 * `(으)로` takes `로` after ㄹ as well as after no 받침.
 */
export const koParticle = (word: string, pair: string): string => {
  const forms = PAIRS[pair];
  const jong = finalConsonant(word);
  if (!forms || jong === null) return pair;
  if (pair === '(으)로' && jong === RIEUL) return forms[1];
  return jong === 0 ? forms[1] : forms[0];
};

/**
 * The pronouns whose subject form is not the word plus `가`: `나` is `내가`, not `나가`, and
 * likewise `저` and `너` (#1903). Only the whole value is one of these; a name that merely ends
 * in `나` (`하나`) takes `가` as usual.
 */
const SUBJECT_PRONOUNS: Readonly<Record<string, string>> = {
  나: '내가',
  저: '제가',
  너: '네가',
};

/**
 * `word` followed by the form of `pair` that fits it (`koParticle`), except that a pronoun
 * before the subject particle takes its own subject form: `나` + `이(가)` is `내가`.
 */
export const koWithParticle = (word: string, pair: string): string =>
  (pair === '이(가)' ? SUBJECT_PRONOUNS[word] : undefined) ?? `${word}${koParticle(word, pair)}`;
