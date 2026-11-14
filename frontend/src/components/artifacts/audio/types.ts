/**
 * Core interfaces and types for the modular multi-genre Music Audio Engine.
 *
 * Designed for plug-and-play extensibility:
 * - New instrument synthesizers implement `VoiceModule` and register with `VoiceRegistry`.
 * - Genre profiles configure acoustic space (Reverb IR), humanization, and default patches.
 */

import type { InstrumentCategory, MusicInstrumentView } from '../MusicAudioEngine';

export type AcousticSpaceProfile =
  | 'concert_hall'
  | 'symphony_surround'
  | 'jazz_club'
  | 'studio_room'
  | 'tight_booth';

export interface StagePosition3D {
  /** Left (-1.0) to Right (+1.0) normalized stage azimuth. */
  panX: number;
  /** Riser elevation in meters (0.0m front floor .. +0.85m rear riser). */
  elevationY: number;
  /** Front-to-back stage depth in meters (-1.6m front soloist .. -6.5m rear riser). */
  depthZ: number;
  /** Number of unison players/desks to layer (1 = solo, 2..3 = orchestral section divisi). */
  ensembleVoices: number;
  /** Ensemble unison detune spread in cents for multi-desk richness. */
  ensembleDetuneCents: number;
}

export interface ExpressionContour {
  /** Dynamic gain multiplier from section emotion, phrase arch, and beat weight (0.55 .. 1.35). */
  dynamicScale: number;
  /** CC11-style mid-note expression swell peak ratio (1.0 = flat, 1.28 = expressive bowed/wind swell). */
  swellPeakRatio: number;
  /** Normalized position in note duration where the swell peaks (0.35 .. 0.65). */
  swellPeakPosition: number;
  /** Timing humanization, chord roll, and groove swing offset in seconds. */
  timingOffsetSec: number;
  /** Harmonic filter brightness multiplier at peak emotion. */
  timbralBloom: number;
  /** Vibrato depth multiplier during emotional climax/swell. */
  vibratoIntensityScale: number;
}

export type GenreStyleProfile =
  | 'pop_ballad'
  | 'hiphop_groove'
  | 'jazz_swing'
  | 'classical_strict';

export type TimbreFamily =
  | 'grand_piano'
  | 'rhodes_ep'
  | 'harpsichord'
  | 'solo_violin'
  | 'solo_viola'
  | 'solo_cello'
  | 'contrabass'
  | 'string_ensemble'
  | 'upright_bass'
  | 'electric_bass'
  | 'sub_808_bass'
  | 'trumpet'
  | 'trombone'
  | 'french_horn'
  | 'saxophone'
  | 'flute'
  | 'clarinet'
  | 'oboe'
  | 'nylon_guitar'
  | 'jazz_guitar'
  | 'steel_guitar'
  | 'warm_pad'
  | 'lead_synth'
  | 'choir_vocal'
  | 'drum_kit_acoustic'
  | 'drum_kit_jazz'
  | 'drum_kit_808'
  | 'drum_kit_orchestral';

export interface ResolvedInstrumentPatch {
  id: string;
  name: string;
  category: InstrumentCategory;
  timbreFamily: TimbreFamily;
  /** General MIDI instrument name for the resolved program (e.g. `acoustic_bass`). */
  gmInstrumentName: string;
  midiProgram: number;
  reverbSend: number;
  defaultPan: number;
  stagePosition?: StagePosition3D;
}

export interface ArticulationEnvelope {
  gateRatio: number;
  attackScale: number;
  releaseScale: number;
  velocityScale: number;
  brightnessBoost: number;
  vibratoDepthCents: number;
  vibratoRateHz: number;
  vibratoDelaySec: number;
  isPlucked: boolean;
}

export interface VoiceRenderContext {
  ctx: AudioContext;
  destination: AudioNode;
  noiseBuffer: AudioBuffer | null;
  midi: number;
  frequency: number;
  startAt: number;
  durationSeconds: number;
  velocity: number;
  volumeGain: number;
  articulation: string;
  articulationProfile: ArticulationEnvelope;
  writtenPitch?: string;
  instrument: MusicInstrumentView;
  patch: ResolvedInstrumentPatch;
  genre: GenreStyleProfile;
  expression?: ExpressionContour;
}

/**
 * Pluggable instrument voice synthesizer module.
 */
export interface VoiceModule {
  readonly id: string;
  supports(patch: ResolvedInstrumentPatch): boolean;
  trigger(rc: VoiceRenderContext): void;
}
