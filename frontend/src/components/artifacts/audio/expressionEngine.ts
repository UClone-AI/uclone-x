/**
 * Musical Expression, Emotional Phrasing Arch, CC11 Swell, and Rubato/Swing Humanizer.
 *
 * Shapes raw score note events into expressive human-like performances by modeling:
 * 1. **Section Emotion Curve**:
 *    - `intro`: intimate start (`pp`–`mp`, `0.76x`) blooming gently into the verse (`0.90x`).
 *    - `verse`: warm narrative storytelling (`0.88x` → `0.96x`).
 *    - `pre_chorus`: dramatic crescendo build-up (`0.90x` → `1.16x`).
 *    - `chorus` / `solo` / `movement`: full emotional climax (`1.12x` → `1.24x`) with harmonic bloom
 *      and intensified vibrato.
 *    - `bridge`: reflective contrast (`0.84x`) rising into a climactic return (`1.12x`).
 *    - `outro`: expressive decrescendo (`0.92x` → `0.64x`) with gentle cadence softening.
 * 2. **4-Bar Phrase Arch Breathing**:
 *    - Natural musical phrasing rises from Bar 1 (`0.96x`) through Bar 2 (`1.03x`) to Bar 3 (`1.08x` peak)
 *      and resolves gracefully on Bar 4 (`0.95x`).
 * 3. **CC11 Sustained Note Expression Swell**:
 *    - Bowed strings, woodwinds, brass, and vocals on sustained notes (`>= 0.38s`) swell in both
 *      volume (`swellPeakRatio` up to `1.30x`) and harmonic brightness mid-note, mimicking real bow
 *      pressure and breath control.
 * 4. **Groove Swing, Rubato & Chord Roll Humanization**:
 *    - Applies track/genre `groove_swing` to 8th-note offbeats, subtle rubato timing offsets,
 *      and gentle upward arpeggiation roll (`5ms` per voice) on piano/guitar/harp chords.
 */

import type { InstrumentCategory } from '../MusicAudioEngine';
import type { ExpressionContour, GenreStyleProfile } from './types';

export type SectionEmotionKind =
  | 'intro'
  | 'verse'
  | 'pre_chorus'
  | 'chorus'
  | 'bridge'
  | 'solo'
  | 'outro'
  | 'movement';

export interface NoteExpressionInput {
  sectionId: string;
  /** The section's structured kind (`ScoreSection.kind`). */
  sectionKind?: string;
  /** 1-indexed bar number within the section. */
  sectionBar: number;
  /** Total bars in this section (`>= 1`). */
  sectionTotalBars: number;
  /** 1-indexed beat within the bar (`1.0 .. 4.99`). */
  beat: number;
  /** Note duration in seconds. */
  durationSeconds: number;
  /** Beat duration in seconds (`60 / bpm`). */
  secPerBeat: number;
  /** MIDI pitch (`0..127`). */
  midi: number;
  /** Note velocity (`1..127`). */
  velocity: number;
  /** Note articulation (`normal`, `legato`, `staccato`, `tenuto`, `accent`, `vibrato`, `pizzicato`). */
  articulation: string;
  /** Instrument category. */
  category: InstrumentCategory;
  /** Instrument role (`lead_melody`, `harmonic_comping`, etc.). */
  role?: string;
  /** Genre style profile. */
  genre: GenreStyleProfile;
  /** Track-level swing ratio (`0.5` = straight, `0.58..0.67` = swing). */
  grooveSwing?: number;
  /** 0-indexed voice order when multiple notes on the same instrument strike on the same beat. */
  chordVoiceIndex?: number;
}

const SECTION_EMOTION_KINDS: readonly SectionEmotionKind[] = [
  'intro',
  'verse',
  'pre_chorus',
  'chorus',
  'bridge',
  'solo',
  'outro',
  'movement',
];

/**
 * The section's `SectionEmotionKind`, read from its structured `kind` (the score's
 * `ScoreSection.kind`). Titles and ids are names, not data: they are never searched for words.
 */
export function inferSectionEmotionKind(explicitKind?: string): SectionEmotionKind {
  return SECTION_EMOTION_KINDS.find((kind) => kind === explicitKind) ?? 'verse';
}

/**
 * Compute the macro dynamic multiplier, timbral bloom, and vibrato boost from the section's emotional role
 * and the current bar's progress (`0.0 .. 1.0`) through that section.
 */
export function computeSectionEmotionEnvelope(
  kind: SectionEmotionKind,
  sectionBar: number,
  sectionTotalBars: number,
): { sectionDynamic: number; timbralBloom: number; vibratoScale: number } {
  const total = Math.max(1, sectionTotalBars);
  const progress = total <= 1 ? 0.5 : Math.max(0, Math.min(1, (sectionBar - 1) / (total - 1)));

  switch (kind) {
    case 'intro': {
      // Intimate opening (pp/mp) that gently warms up toward the end of the intro
      const sectionDynamic = 0.76 + 0.14 * progress;
      return {
        sectionDynamic,
        timbralBloom: 0.88 + 0.12 * progress,
        vibratoScale: 0.9,
      };
    }
    case 'pre_chorus': {
      // Dramatic crescendo build-up driving into the chorus climax
      const sectionDynamic = 0.9 + 0.26 * Math.pow(progress, 1.15);
      return {
        sectionDynamic,
        timbralBloom: 0.96 + 0.24 * progress,
        vibratoScale: 1.05 + 0.2 * progress,
      };
    }
    case 'chorus':
    case 'solo': {
      // Full emotional climax (f/ff) with rich harmonic bloom and passionate vibrato
      const arch = Math.sin(progress * Math.PI);
      const sectionDynamic = 1.12 + 0.12 * arch;
      return {
        sectionDynamic,
        timbralBloom: 1.16 + 0.1 * arch,
        vibratoScale: 1.28,
      };
    }
    case 'movement': {
      // Symphonic movement arch: builds from noble statement to grand orchestral peak
      const arch = Math.sin(progress * Math.PI);
      const sectionDynamic = 0.94 + 0.26 * arch;
      return {
        sectionDynamic,
        timbralBloom: 1.02 + 0.18 * arch,
        vibratoScale: 1.18 + 0.15 * arch,
      };
    }
    case 'bridge': {
      // Reflective contrast in the first half, rising into a dramatic return in the second half
      const sectionDynamic = progress < 0.5 ? 0.84 + 0.08 * (progress * 2) : 0.92 + 0.22 * ((progress - 0.5) * 2);
      return {
        sectionDynamic,
        timbralBloom: 0.92 + 0.2 * progress,
        vibratoScale: 1.12,
      };
    }
    case 'outro': {
      // Expressive decrescendo (diminuendo al fine)
      const sectionDynamic = 0.92 - 0.28 * Math.pow(progress, 1.1);
      return {
        sectionDynamic,
        timbralBloom: 0.94 - 0.22 * progress,
        vibratoScale: 0.95 - 0.15 * progress,
      };
    }
    case 'verse':
    default: {
      // Narrative breathing with gentle rise across the verse
      const sectionDynamic = 0.88 + 0.1 * Math.sin(progress * Math.PI * 0.85);
      return {
        sectionDynamic,
        timbralBloom: 0.96 + 0.08 * progress,
        vibratoScale: 1.04,
      };
    }
  }
}

/**
 * Compute the 4-bar phrase arch and intra-bar metrical beat weight.
 */
export function computePhraseAndBeatWeight(
  sectionBar: number,
  beat: number,
  durationSeconds: number,
  secPerBeat: number,
): number {
  // 4-bar phrase breathing: Bar 1 (statement), Bar 2 (rising), Bar 3 (climax), Bar 4 (cadence resolution)
  const phrasePos = ((Math.max(1, sectionBar) - 1) % 4) + 1;
  const phraseMultiplier =
    phrasePos === 1 ? 0.97 : phrasePos === 2 ? 1.03 : phrasePos === 3 ? 1.08 : 0.95;

  // Intra-bar beat hierarchy: Beat 1 (downbeat gravitas), Beat 3 (secondary pulse), off-beats
  const beatInt = Math.floor(beat);
  const beatFrac = beat - beatInt;
  let beatMultiplier = 1.0;
  if (beatFrac < 0.08) {
    if (beatInt === 1) beatMultiplier = 1.06;
    else if (beatInt === 3) beatMultiplier = 1.03;
    else beatMultiplier = 0.97;
  } else {
    // Syncopated off-beat notes that hold across a beat get a jazz/funk syncopation accent
    const durationBeats = secPerBeat > 0 ? durationSeconds / secPerBeat : 0.5;
    beatMultiplier = durationBeats >= 0.95 ? 1.04 : 0.94;
  }

  return phraseMultiplier * beatMultiplier;
}

/**
 * Compute the complete `ExpressionContour` for a single note event.
 */
export function computeNoteExpression(input: NoteExpressionInput): ExpressionContour {
  const kind = inferSectionEmotionKind(input.sectionKind);
  const { sectionDynamic, timbralBloom, vibratoScale } = computeSectionEmotionEnvelope(
    kind,
    input.sectionBar,
    input.sectionTotalBars,
  );
  const phraseBeatWeight = computePhraseAndBeatWeight(
    input.sectionBar,
    input.beat,
    input.durationSeconds,
    input.secPerBeat,
  );

  // Lead melody & solo roles step forward slightly during climactic sections
  const roleBoost =
    input.role === 'lead_melody' || input.role === 'satb_soprano'
      ? kind === 'chorus' || kind === 'solo'
        ? 1.08
        : 1.03
      : 1.0;

  const dynamicScale = Math.max(
    0.55,
    Math.min(1.35, sectionDynamic * phraseBeatWeight * roleBoost),
  );

  // Sustained Note Emotional Swell (CC11 Bow/Breath Swell for Strings, Winds, Brass, Choir)
  const isSustainingFamily =
    input.category === 'strings' ||
    input.category === 'woodwinds' ||
    input.category === 'brass' ||
    input.category === 'vocal';
  const art = (input.articulation || 'normal').toLowerCase();
  const isShortArticulation = art === 'staccato' || art === 'pizzicato';

  let swellPeakRatio = 1.0;
  let swellPeakPosition = 0.45;
  if (isSustainingFamily && !isShortArticulation && input.durationSeconds >= 0.36) {
    // Longer notes swell more deeply, especially in emotional sections or legato/tenuto/vibrato
    const lengthFactor = Math.min(1.0, (input.durationSeconds - 0.3) / 1.5);
    const artBoost = art === 'legato' || art === 'vibrato' || art === 'tenuto' ? 0.08 : 0.0;
    const climaxBoost = kind === 'chorus' || kind === 'solo' || kind === 'movement' ? 0.06 : 0.0;
    swellPeakRatio = 1.12 + 0.16 * lengthFactor + artBoost + climaxBoost;
    swellPeakPosition = art === 'accent' ? 0.32 : 0.46;
  }

  // Groove Swing + Rubato + Chord Arpeggiation Roll
  let timingOffsetSec = 0.0;
  const effectiveSwing =
    input.grooveSwing !== undefined && input.grooveSwing > 0.505
      ? input.grooveSwing
      : input.genre === 'jazz_swing'
        ? 0.62
        : input.genre === 'hiphop_groove'
          ? 0.55
          : 0.5;

  const beatFrac = input.beat - Math.floor(input.beat);
  // Apply swing delay to 8th-note offbeats (around X.5)
  if (effectiveSwing > 0.505 && Math.abs(beatFrac - 0.5) < 0.08) {
    timingOffsetSec += (effectiveSwing - 0.5) * input.secPerBeat * 0.85;
  }

  // Gentle chord roll (arpeggiation) on Piano, Guitar, and Strings when multiple notes strike together
  const chordIdx = input.chordVoiceIndex ?? 0;
  if (
    chordIdx > 0 &&
    (input.category === 'piano' || input.category === 'guitar' || input.category === 'strings')
  ) {
    const rollStepSec =
      kind === 'intro' || kind === 'outro' || input.genre === 'classical_strict' ? 0.0065 : 0.004;
    timingOffsetSec += Math.min(0.028, chordIdx * rollStepSec);
  }

  // Deterministic micro-rubato so ensemble attacks feel organic rather than mechanical
  if (input.category !== 'drums') {
    const hash =
      ((input.sectionBar * 31 + Math.round(input.beat * 12) * 17 + input.midi * 13) & 0xff) / 255.0;
    const rubatoSpan =
      input.genre === 'classical_strict' || input.genre === 'jazz_swing' ? 0.005 : 0.0025;
    timingOffsetSec += (hash - 0.5) * rubatoSpan;
  }

  return {
    dynamicScale,
    swellPeakRatio,
    swellPeakPosition,
    timingOffsetSec,
    timbralBloom,
    vibratoIntensityScale: vibratoScale,
  };
}
