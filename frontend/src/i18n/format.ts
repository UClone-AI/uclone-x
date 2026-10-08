import { KO_PARTICLE_PATTERN, koParticle, koWithParticle } from './ko/particles';

/**
 * Filling a catalog sentence. Every catalog is data (`locales/<language>/<namespace>.json`), so a
 * sentence with a value in it is a template: `{name}` marks where the value goes, and each
 * language puts it wherever its grammar wants it.
 */

export type Values = Readonly<Record<string, string | number>>;

/**
 * A sentence whose wording depends on a count. `one` is used for exactly 1 and `other` for
 * every other count; both receive the count as `{count}`. That is the whole plural rule of the
 * two languages this head ships (Korean writes both forms the same); a language with more forms
 * needs `Intl.PluralRules` here before its catalog is added.
 */
export interface Plural {
  one: string;
  other: string;
  zero?: string;
}

const PLACEHOLDER = /\{(\w+)\}/g;

/**
 * A placeholder, a closing quote the sentence may wrap it in, and a Korean paired particle
 * after it (`ko/particles.ts` lists the pairs). Only the Korean catalog writes them.
 */
const FILLED = new RegExp(`\\{(\\w+)\\}(["'”’]?)(${KO_PARTICLE_PATTERN})?`, 'g');

/**
 * `template` with each `{name}` replaced by `values[name]`. Values are inserted as they are and
 * never read as templates themselves, so a value may contain braces.
 *
 * A Korean paired particle right after a placeholder is settled by the value it follows
 * (`koParticle`, `ko/particles.ts`): the subject particle after a name follows its last
 * syllable, and stays paired after a name that does not end in Hangul (#1900). The first- and
 * second-person pronouns take their own subject forms (`koWithParticle`, #1903).
 *
 * A placeholder with no value is a call site that names the wrong value. Under test that
 * throws, so the component test that renders the sentence fails; in a running app the marker is
 * left visible rather than breaking the screen.
 */
export const fmt = (template: string, values: Values = {}): string =>
  template.replace(FILLED, (marker, name: string, quote: string, particle: string | undefined) => {
    if (Object.prototype.hasOwnProperty.call(values, name)) {
      const value = String(values[name]);
      // A quoted value is the word as written, so a quoted pronoun keeps its plain particle;
      // bare, a pronoun takes its own subject form (`koWithParticle`).
      if (particle && !quote) return koWithParticle(value, particle);
      return `${value}${quote}${particle ? koParticle(value, particle) : ''}`;
    }
    if (import.meta.env.MODE === 'test') {
      throw new Error(`No value for ${marker} in "${template}"`);
    }
    return marker;
  });

/** The form of `forms` for `count`, filled with `count` and `values`. */
export const plural = (forms: Plural, count: number, values: Values = {}): string =>
  fmt(count === 0 && forms.zero ? forms.zero : count === 1 ? forms.one : forms.other, { ...values, count });

/** The placeholder names a template uses, in order of first appearance. */
export const placeholders = (template: string): string[] => [
  ...new Set(Array.from(template.matchAll(PLACEHOLDER), (match) => match[1])),
];
