/**
 * General MIDI (GM) Instrument Catalog, Genre Timbre Resolver, and Articulation Profiles.
 *
 * Maps `MusicInstrumentView` + `GenreStyleProfile` into concrete `ResolvedInstrumentPatch`
 * descriptors with General MIDI instrument names, GM program numbers, stereo stage
 * placement, and reverb send levels. The browser voice modules synthesize every patch.
 */

import type { InstrumentCategory, MusicInstrumentView } from '../MusicAudioEngine';
import type {
  AcousticSpaceProfile,
  ArticulationEnvelope,
  GenreStyleProfile,
  ResolvedInstrumentPatch,
  StagePosition3D,
  TimbreFamily,
} from './types';

/**
 * Standard Symphony Orchestra & Studio 3D Stage Seating Map:
 * - Row 1 (Front Stage, z = -1.8m .. -2.4m, y = 0.0m .. 0.08m):
 *   Grand Piano (-0.12), Solo Violin (-0.42), Solo Viola (+0.18), Solo Cello (+0.42), Guitars (-0.28).
 * - Row 2 (Mid-Front Strings & Rhythm, z = -2.8m .. -3.5m, y = 0.15m .. 0.25m):
 *   String Ensemble (-0.16, 3-desk divisi), Upright/Electric Bass (+0.24), Rhodes EP (+0.20).
 * - Row 3 (Center Woodwind Choir, z = -3.8m .. -4.4m, y = 0.35m):
 *   Flute (-0.24), Oboe (+0.14), Clarinet (-0.32), Saxophone (+0.22).
 * - Row 4 (Rear Elevated Riser — Brass, Contrabass, Percussion, Choir, z = -4.8m .. -6.2m, y = 0.55m .. 0.85m):
 *   French Horns (-0.48, 2-desk), Trumpets (+0.36, 2-desk), Trombones (+0.52, 2-desk),
 *   Contrabass (+0.62, 2-desk), Timpani/Drums (+0.06), Choir/Warm Pad (0.0, 3-desk).
 */
export function resolveStagePosition3D(
  timbreFamily: TimbreFamily,
  userPan: number = 0,
): StagePosition3D {
  const STAGE_MAP: Record<TimbreFamily, StagePosition3D> = {
    grand_piano: {
      panX: -0.1,
      elevationY: 0.0,
      depthZ: -1.85,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    rhodes_ep: {
      panX: -0.18,
      elevationY: 0.05,
      depthZ: -2.2,
      ensembleVoices: 2,
      ensembleDetuneCents: 3.5,
    },
    harpsichord: {
      panX: -0.22,
      elevationY: 0.0,
      depthZ: -2.1,
      ensembleVoices: 2,
      ensembleDetuneCents: 3.0,
    },
    solo_violin: {
      panX: -0.44,
      elevationY: 0.06,
      depthZ: -2.1,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    solo_viola: {
      panX: 0.18,
      elevationY: 0.1,
      depthZ: -2.5,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    solo_cello: {
      panX: 0.42,
      elevationY: 0.05,
      depthZ: -2.25,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    contrabass: {
      panX: 0.62,
      elevationY: 0.35,
      depthZ: -4.2,
      ensembleVoices: 2,
      ensembleDetuneCents: 4.5,
    },
    string_ensemble: {
      panX: -0.14,
      elevationY: 0.2,
      depthZ: -3.3,
      ensembleVoices: 3,
      ensembleDetuneCents: 5.5,
    },
    upright_bass: {
      panX: 0.22,
      elevationY: 0.1,
      depthZ: -2.9,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    electric_bass: {
      panX: 0.08,
      elevationY: 0.05,
      depthZ: -2.5,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    sub_808_bass: {
      panX: 0.0,
      elevationY: 0.0,
      depthZ: -1.6,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    flute: {
      panX: -0.24,
      elevationY: 0.35,
      depthZ: -3.8,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    oboe: {
      panX: 0.14,
      elevationY: 0.35,
      depthZ: -3.9,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    clarinet: {
      panX: -0.32,
      elevationY: 0.38,
      depthZ: -4.1,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    saxophone: {
      panX: 0.24,
      elevationY: 0.22,
      depthZ: -2.8,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    french_horn: {
      panX: -0.48,
      elevationY: 0.62,
      depthZ: -5.2,
      ensembleVoices: 2,
      ensembleDetuneCents: 4.2,
    },
    trumpet: {
      panX: 0.36,
      elevationY: 0.68,
      depthZ: -5.4,
      ensembleVoices: 2,
      ensembleDetuneCents: 3.8,
    },
    trombone: {
      panX: 0.52,
      elevationY: 0.7,
      depthZ: -5.6,
      ensembleVoices: 2,
      ensembleDetuneCents: 4.5,
    },
    nylon_guitar: {
      panX: -0.26,
      elevationY: 0.05,
      depthZ: -2.0,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    jazz_guitar: {
      panX: -0.3,
      elevationY: 0.08,
      depthZ: -2.3,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    steel_guitar: {
      panX: -0.28,
      elevationY: 0.06,
      depthZ: -2.2,
      ensembleVoices: 2,
      ensembleDetuneCents: 3.2,
    },
    warm_pad: {
      panX: 0.0,
      elevationY: 0.75,
      depthZ: -5.5,
      ensembleVoices: 3,
      ensembleDetuneCents: 6.5,
    },
    lead_synth: {
      panX: 0.05,
      elevationY: 0.2,
      depthZ: -2.4,
      ensembleVoices: 2,
      ensembleDetuneCents: 4.8,
    },
    choir_vocal: {
      panX: 0.0,
      elevationY: 0.85,
      depthZ: -6.0,
      ensembleVoices: 3,
      ensembleDetuneCents: 6.0,
    },
    drum_kit_acoustic: {
      panX: 0.0,
      elevationY: 0.35,
      depthZ: -3.6,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    drum_kit_jazz: {
      panX: -0.08,
      elevationY: 0.25,
      depthZ: -3.2,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    drum_kit_808: {
      panX: 0.0,
      elevationY: 0.0,
      depthZ: -1.8,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
    drum_kit_orchestral: {
      panX: 0.08,
      elevationY: 0.78,
      depthZ: -5.9,
      ensembleVoices: 1,
      ensembleDetuneCents: 0,
    },
  };

  const base = STAGE_MAP[timbreFamily] ?? STAGE_MAP.grand_piano;
  const effectivePan = Math.abs(userPan) > 0.01 ? userPan : base.panX;
  return {
    ...base,
    panX: Math.max(-1.0, Math.min(1.0, effectivePan)),
  };
}

const GM_PROGRAM_TO_PATCH: Record<number, { gm: string; family: TimbreFamily }> = {
  0: { gm: 'acoustic_grand_piano', family: 'grand_piano' },
  1: { gm: 'bright_acoustic_piano', family: 'grand_piano' },
  4: { gm: 'electric_piano_1', family: 'rhodes_ep' },
  5: { gm: 'electric_piano_2', family: 'rhodes_ep' },
  6: { gm: 'harpsichord', family: 'harpsichord' },
  24: { gm: 'acoustic_guitar_nylon', family: 'nylon_guitar' },
  25: { gm: 'acoustic_guitar_steel', family: 'steel_guitar' },
  26: { gm: 'electric_guitar_jazz', family: 'jazz_guitar' },
  27: { gm: 'electric_guitar_clean', family: 'jazz_guitar' },
  32: { gm: 'acoustic_bass', family: 'upright_bass' },
  33: { gm: 'electric_bass_finger', family: 'electric_bass' },
  34: { gm: 'electric_bass_pick', family: 'electric_bass' },
  38: { gm: 'synth_bass_1', family: 'sub_808_bass' },
  39: { gm: 'synth_bass_2', family: 'sub_808_bass' },
  40: { gm: 'violin', family: 'solo_violin' },
  41: { gm: 'viola', family: 'solo_viola' },
  42: { gm: 'cello', family: 'solo_cello' },
  43: { gm: 'contrabass', family: 'contrabass' },
  48: { gm: 'string_ensemble_1', family: 'string_ensemble' },
  49: { gm: 'string_ensemble_2', family: 'string_ensemble' },
  52: { gm: 'choir_aahs', family: 'choir_vocal' },
  56: { gm: 'trumpet', family: 'trumpet' },
  57: { gm: 'trombone', family: 'trombone' },
  60: { gm: 'french_horn', family: 'french_horn' },
  64: { gm: 'soprano_sax', family: 'saxophone' },
  65: { gm: 'alto_sax', family: 'saxophone' },
  66: { gm: 'tenor_sax', family: 'saxophone' },
  68: { gm: 'oboe', family: 'oboe' },
  71: { gm: 'clarinet', family: 'clarinet' },
  73: { gm: 'flute', family: 'flute' },
  80: { gm: 'lead_1_square', family: 'lead_synth' },
  81: { gm: 'lead_2_sawtooth', family: 'lead_synth' },
  88: { gm: 'pad_1_new_age', family: 'warm_pad' },
  89: { gm: 'pad_2_warm', family: 'warm_pad' },
};

export function normalizeGenreProfile(genre?: string | null): GenreStyleProfile {
  const g = (genre ?? '').toLowerCase();
  if (g.includes('classical')) return 'classical_strict';
  if (g.includes('jazz') || g.includes('swing') || g.includes('bossa')) return 'jazz_swing';
  if (g.includes('hiphop') || g.includes('hip_hop') || g.includes('trap') || g.includes('boom')) {
    return 'hiphop_groove';
  }
  return 'pop_ballad';
}

export function defaultSpaceForGenre(genre: GenreStyleProfile): AcousticSpaceProfile {
  switch (genre) {
    case 'classical_strict':
      return 'concert_hall';
    case 'jazz_swing':
      return 'jazz_club';
    case 'hiphop_groove':
      return 'tight_booth';
    case 'pop_ballad':
    default:
      return 'studio_room';
  }
}

/**
 * Resolve an instrument's concrete timbre family, General MIDI instrument name, and mix parameters.
 */
export function resolveInstrumentPatch(
  inst: MusicInstrumentView,
  genre: GenreStyleProfile = 'pop_ballad',
): ResolvedInstrumentPatch {
  const hint = `${inst.id} ${inst.name} ${inst.soundfont_patch} ${inst.role}`.toLowerCase();

  // 1. Drums & Percussion
  if (inst.category === 'drums' || inst.midi_channel === 10 || hint.includes('drum')) {
    let family: TimbreFamily = 'drum_kit_acoustic';
    let reverbSend = 0.16;
    if (genre === 'hiphop_groove' || hint.includes('808') || hint.includes('trap') || hint.includes('boom')) {
      family = 'drum_kit_808';
      reverbSend = 0.08;
    } else if (genre === 'jazz_swing' || hint.includes('jazz') || hint.includes('brush')) {
      family = 'drum_kit_jazz';
      reverbSend = 0.22;
    } else if (genre === 'classical_strict' || hint.includes('timpani') || hint.includes('orchestra')) {
      family = 'drum_kit_orchestral';
      reverbSend = 0.35;
    }
    const stagePos = resolveStagePosition3D(family, inst.pan);
    return {
      id: inst.id,
      name: inst.name,
      category: 'drums',
      timbreFamily: family,
      gmInstrumentName: 'synth_drum',
      midiProgram: 0,
      reverbSend,
      defaultPan: stagePos.panX,
      stagePosition: stagePos,
    };
  }

  // 2. Explicit non-zero GM program mapping if provided and not generic 0 on non-piano
  if (
    inst.midi_program > 0 &&
    Object.prototype.hasOwnProperty.call(GM_PROGRAM_TO_PATCH, inst.midi_program)
  ) {
    const mapped = GM_PROGRAM_TO_PATCH[inst.midi_program];
    const stagePos = resolveStagePosition3D(mapped.family, inst.pan);
    return {
      id: inst.id,
      name: inst.name,
      category: inst.category,
      timbreFamily: mapped.family,
      gmInstrumentName: mapped.gm,
      midiProgram: inst.midi_program,
      reverbSend: reverbSendForCategory(inst.category, genre),
      defaultPan: stagePos.panX,
      stagePosition: stagePos,
    };
  }

  // 3. Name / Patch / Role / Category + Genre heuristics
  switch (inst.category) {
    case 'piano': {
      if (hint.includes('rhodes') || hint.includes('electric') || hint.includes('ep')) {
        return makePatch(inst, 'rhodes_ep', 'electric_piano_1', 4, genre, 0.28);
      }
      if (hint.includes('harpsichord') || hint.includes('clav')) {
        return makePatch(inst, 'harpsichord', 'harpsichord', 6, genre, 0.32);
      }
      return makePatch(inst, 'grand_piano', 'acoustic_grand_piano', 0, genre, 0.32);
    }

    case 'strings': {
      if (hint.includes('contrabass') || hint.includes('double_bass')) {
        return makePatch(inst, 'contrabass', 'contrabass', 43, genre, 0.26);
      }
      if (hint.includes('cello') || hint.includes('violoncello') || inst.role === 'satb_bass') {
        return makePatch(inst, 'solo_cello', 'cello', 42, genre, 0.38);
      }
      if (hint.includes('viola') || inst.role === 'satb_tenor') {
        return makePatch(inst, 'solo_viola', 'viola', 41, genre, 0.38);
      }
      if (hint.includes('ensemble') || hint.includes('section') || hint.includes('pad')) {
        return makePatch(inst, 'string_ensemble', 'string_ensemble_1', 48, genre, 0.44);
      }
      return makePatch(inst, 'solo_violin', 'violin', 40, genre, 0.38);
    }

    case 'bass': {
      if (
        hint.includes('808') ||
        hint.includes('sub') ||
        hint.includes('synth') ||
        genre === 'hiphop_groove'
      ) {
        return makePatch(inst, 'sub_808_bass', 'synth_bass_1', 38, genre, 0.06);
      }
      if (
        hint.includes('upright') ||
        hint.includes('acoustic') ||
        hint.includes('contrabass') ||
        genre === 'jazz_swing' ||
        genre === 'classical_strict'
      ) {
        return makePatch(inst, 'upright_bass', 'acoustic_bass', 32, genre, 0.14);
      }
      return makePatch(inst, 'electric_bass', 'electric_bass_finger', 33, genre, 0.10);
    }

    case 'brass': {
      if (hint.includes('trombone')) {
        return makePatch(inst, 'trombone', 'trombone', 57, genre, 0.34);
      }
      if (hint.includes('horn')) {
        return makePatch(inst, 'french_horn', 'french_horn', 60, genre, 0.38);
      }
      if (hint.includes('sax')) {
        return makePatch(inst, 'saxophone', 'alto_sax', 65, genre, 0.32);
      }
      return makePatch(inst, 'trumpet', 'trumpet', 56, genre, 0.32);
    }

    case 'woodwinds': {
      if (hint.includes('sax')) {
        return makePatch(inst, 'saxophone', 'tenor_sax', 66, genre, 0.30);
      }
      if (hint.includes('flute')) {
        return makePatch(inst, 'flute', 'flute', 73, genre, 0.36);
      }
      if (hint.includes('oboe') || hint.includes('english_horn')) {
        return makePatch(inst, 'oboe', 'oboe', 68, genre, 0.34);
      }
      return makePatch(inst, 'clarinet', 'clarinet', 71, genre, 0.34);
    }

    case 'guitar': {
      if (hint.includes('jazz') || hint.includes('hollow') || genre === 'jazz_swing') {
        return makePatch(inst, 'jazz_guitar', 'electric_guitar_jazz', 26, genre, 0.28);
      }
      if (hint.includes('nylon') || hint.includes('classical') || genre === 'classical_strict') {
        return makePatch(inst, 'nylon_guitar', 'acoustic_guitar_nylon', 24, genre, 0.32);
      }
      return makePatch(inst, 'steel_guitar', 'acoustic_guitar_steel', 25, genre, 0.26);
    }

    case 'vocal':
      return makePatch(inst, 'choir_vocal', 'choir_aahs', 52, genre, 0.42);

    case 'synth':
    default: {
      if (hint.includes('lead') || inst.role === 'lead_melody') {
        return makePatch(inst, 'lead_synth', 'lead_2_sawtooth', 81, genre, 0.28);
      }
      return makePatch(inst, 'warm_pad', 'pad_2_warm', 89, genre, 0.40);
    }
  }
}

function reverbSendForCategory(category: InstrumentCategory, genre: GenreStyleProfile): number {
  const base =
    category === 'bass'
      ? 0.08
      : category === 'drums'
        ? 0.15
        : category === 'strings' || category === 'vocal'
          ? 0.38
          : 0.28;
  if (genre === 'classical_strict') return Math.min(0.55, base * 1.25);
  if (genre === 'hiphop_groove') return base * 0.65;
  return base;
}

function makePatch(
  inst: MusicInstrumentView,
  timbreFamily: TimbreFamily,
  gmInstrumentName: string,
  midiProgram: number,
  genre: GenreStyleProfile,
  baseReverb: number,
): ResolvedInstrumentPatch {
  const genreScale =
    genre === 'classical_strict' ? 1.2 : genre === 'hiphop_groove' ? 0.65 : 1.0;
  const stagePos = resolveStagePosition3D(timbreFamily, inst.pan);
  return {
    id: inst.id,
    name: inst.name,
    category: inst.category,
    timbreFamily,
    gmInstrumentName,
    midiProgram,
    reverbSend: Math.min(0.6, Math.max(0.04, baseReverb * genreScale)),
    defaultPan: stagePos.panX,
    stagePosition: stagePos,
  };
}

/**
 * Translate a note's `articulation` string and instrument category into DSP envelope modifiers.
 */
export function resolveArticulationEnvelope(
  articulation: string,
  category: InstrumentCategory,
): ArticulationEnvelope {
  const art = (articulation || 'normal').toLowerCase();
  const baseVibrato =
    category === 'strings' || category === 'woodwinds' || category === 'brass' || category === 'vocal'
      ? 7.5
      : 0;

  switch (art) {
    case 'staccato':
      return {
        gateRatio: 0.42,
        attackScale: 0.65,
        releaseScale: 0.45,
        velocityScale: 1.06,
        brightnessBoost: 1.18,
        vibratoDepthCents: 0,
        vibratoRateHz: 5.5,
        vibratoDelaySec: 0.2,
        isPlucked: false,
      };
    case 'legato':
      return {
        gateRatio: 1.06,
        attackScale: 1.35,
        releaseScale: 1.25,
        velocityScale: 0.96,
        brightnessBoost: 0.92,
        vibratoDepthCents: baseVibrato * 1.1,
        vibratoRateHz: 5.8,
        vibratoDelaySec: 0.14,
        isPlucked: false,
      };
    case 'tenuto':
      return {
        gateRatio: 1.0,
        attackScale: 0.95,
        releaseScale: 1.1,
        velocityScale: 1.04,
        brightnessBoost: 1.05,
        vibratoDepthCents: baseVibrato,
        vibratoRateHz: 5.6,
        vibratoDelaySec: 0.16,
        isPlucked: false,
      };
    case 'accent':
      return {
        gateRatio: 0.88,
        attackScale: 0.6,
        releaseScale: 0.9,
        velocityScale: 1.2,
        brightnessBoost: 1.38,
        vibratoDepthCents: baseVibrato * 0.9,
        vibratoRateHz: 5.8,
        vibratoDelaySec: 0.15,
        isPlucked: false,
      };
    case 'vibrato':
      return {
        gateRatio: 0.96,
        attackScale: 1.0,
        releaseScale: 1.15,
        velocityScale: 1.02,
        brightnessBoost: 1.06,
        vibratoDepthCents: Math.max(14, baseVibrato * 1.9),
        vibratoRateHz: 6.1,
        vibratoDelaySec: 0.08,
        isPlucked: false,
      };
    case 'pizzicato':
      return {
        gateRatio: 0.35,
        attackScale: 0.15,
        releaseScale: 0.4,
        velocityScale: 1.12,
        brightnessBoost: 1.25,
        vibratoDepthCents: 0,
        vibratoRateHz: 5.0,
        vibratoDelaySec: 0.5,
        isPlucked: true,
      };
    case 'normal':
    default:
      return {
        gateRatio: 0.9,
        attackScale: 1.0,
        releaseScale: 1.0,
        velocityScale: 1.0,
        brightnessBoost: 1.0,
        vibratoDepthCents: baseVibrato,
        vibratoRateHz: 5.6,
        vibratoDelaySec: 0.16,
        isPlucked: false,
      };
  }
}
