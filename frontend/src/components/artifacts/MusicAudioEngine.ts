/**
 * Modular in-browser WebAudio synthesizer and bar scheduler for `MusicView`.
 *
 * Every instrument is synthesized in the browser by the voice modules below. The one
 * exception is the grand piano, which plays recorded Salamander Grand Piano samples once
 * they have loaded (`audio/salamanderPiano.ts`) and the synthesized piano until then.
 *
 * Architecture:
 * - `audio/gmCatalog.ts`: General MIDI patch & genre timbre resolution + articulation envelopes.
 * - `audio/dspGraph.ts`: Stereo channel strips, algorithmic convolution reverb (Concert Hall,
 *   Jazz Club, Studio Room, Tight Booth), and master bus compressor/limiter.
 * - `audio/salamanderPiano.ts`: recorded grand piano (CC BY 3.0, credited in `MusicView`).
 * - `audio/voiceRegistry.ts` & `audio/voices/*`: Pluggable multi-partial physical-modeling
 *   synthesizers for Piano/Rhodes, Bowed/Pizzicato Strings, Upright/Electric/808 Bass,
 *   Brass/Woodwinds, Plucked Guitars/Pads, and Multi-Genre Drum Kits.
 *
 * Gracefully handles headless/mocked `window.AudioContext` environments (e.g., Vitest / jsdom).
 */

import {
  ACOUSTIC_SPACES,
  createPlaybackSessionBus,
  createSharedNoiseBuffer,
  type PlaybackSessionBus,
} from './audio/dspGraph';
import {
  computeNoteExpression,
  inferSectionEmotionKind,
  type NoteExpressionInput,
  type SectionEmotionKind,
} from './audio/expressionEngine';
import {
  defaultSpaceForGenre,
  normalizeGenreProfile,
  resolveArticulationEnvelope,
  resolveInstrumentPatch,
  resolveStagePosition3D,
} from './audio/gmCatalog';
import { isSampledPianoPatch, SalamanderPiano } from './audio/salamanderPiano';
import type {
  AcousticSpaceProfile,
  ExpressionContour,
  GenreStyleProfile,
  ResolvedInstrumentPatch,
  StagePosition3D,
  VoiceModule,
  VoiceRenderContext,
} from './audio/types';
import { defaultVoiceRegistry, VoiceRegistry } from './audio/voiceRegistry';

export {
  ACOUSTIC_SPACES,
  computeNoteExpression,
  defaultSpaceForGenre,
  inferSectionEmotionKind,
  normalizeGenreProfile,
  resolveArticulationEnvelope,
  resolveInstrumentPatch,
  resolveStagePosition3D,
};
export type {
  AcousticSpaceProfile,
  ExpressionContour,
  GenreStyleProfile,
  NoteExpressionInput,
  ResolvedInstrumentPatch,
  SectionEmotionKind,
  StagePosition3D,
  VoiceModule,
  VoiceRenderContext,
};

export type InstrumentCategory =
  | 'bass'
  | 'strings'
  | 'piano'
  | 'drums'
  | 'woodwinds'
  | 'brass'
  | 'synth'
  | 'guitar'
  | 'vocal';

export interface MusicInstrumentView {
  id: string;
  name: string;
  category: InstrumentCategory;
  role: string;
  midi_program: number;
  midi_channel: number;
  soundfont_patch: string;
  min_pitch: string;
  max_pitch: string;
  transposition_semitones: number;
  volume: number;
  pan: number;
  muted: boolean;
  solo: boolean;
}

export interface MusicTrackBarView {
  bar: number;
  chord?: string | null;
  notes: string[];
  repeat_bar?: number | null;
  transpose_semitones?: number;
  muted?: boolean;
}

export interface MusicTrackView {
  track_id: string;
  section_id: string;
  instrument_id: string;
  digest: string;
  groove_swing: number;
  bars: MusicTrackBarView[];
  can_revert: boolean;
}

export interface MusicBarView {
  global_bar: number;
  section_id: string;
  section_title: string;
  section_bar: number;
  chord: string;
  bpm: number;
  start_seconds: number;
  end_seconds: number;
}

export interface MusicNoteView {
  track_id: string;
  section_id: string;
  instrument_id: string;
  bar: number;
  global_bar: number;
  beat: number;
  duration: number;
  concert_midi: number;
  written_pitch: string;
  concert_pitch: string;
  velocity: number;
  articulation: string;
  start_seconds: number;
  duration_seconds: number;
}

export interface MusicProposalView {
  id: string;
  kind: 'track_patch' | 'track_replace' | 'instrument_change';
  summary: string;
  section_id?: string | null;
  instrument_id?: string | null;
  start_bar?: number | null;
  end_bar?: number | null;
  seen_digest: string;
  original_bars: MusicTrackBarView[];
  proposed_bars: MusicTrackBarView[];
}

export interface MusicRegionSelection {
  section_id: string;
  start_bar: number;
  end_bar: number;
  instrument_ids: string[];
}

export interface ShownSongView {
  song_id: string;
  title: string;
  genre: string;
  style_axioms: {
    profile: string;
    forbid_parallel_fifths_octaves: boolean;
    forbid_voice_crossing: boolean;
    require_kick_bass_lock: boolean;
    default_swing: number;
  };
  writable: boolean;
  holder: string | null;
  decide_code: string | null;
  /** Why changes cannot be saved now, in plain words, when `writable` is false. */
  decide_note?: string | null;
  tempo: number;
  time_signature: string;
  key_signature: string;
  total_bars: number;
  duration_seconds: number;
  sections: {
    id: string;
    title: string;
    kind: string;
    bars: number;
    chords: string[];
  }[];
  instruments: MusicInstrumentView[];
  tracks: MusicTrackView[];
  bars: MusicBarView[];
  notes: MusicNoteView[];
  proposals: MusicProposalView[];
}

export interface PlaybackOptions {
  notes: MusicNoteView[];
  bars: MusicBarView[];
  instruments: MusicInstrumentView[];
  sections?: ShownSongView['sections'];
  tracks?: MusicTrackView[];
  genre?: string;
  acousticSpace?: AcousticSpaceProfile;
  loopEnabled?: boolean;
  loopStartBar?: number;
  loopEndBar?: number;
  onBarChange?: (globalBar: number) => void;
  onStop?: () => void;
}

export interface AuditionNoteOptions {
  instrument?: MusicInstrumentView;
  writtenPitch?: string;
  velocity?: number;
  articulation?: string;
  genre?: string;
}

const NOTE_OFFSETS: Record<string, number> = {
  C: 0,
  D: 2,
  E: 4,
  F: 5,
  G: 7,
  A: 9,
  B: 11,
};

const DRUM_PITCHES: Record<string, number> = {
  KICK: 36,
  BD: 36,
  RIM: 37,
  SNARE: 38,
  SD: 38,
  CLAP: 39,
  CLOSED_HAT: 42,
  HIHAT: 42,
  HH: 42,
  LOW_TOM: 45,
  OPEN_HAT: 46,
  OH: 46,
  MID_TOM: 47,
  CRASH: 49,
  HIGH_TOM: 50,
  RIDE: 51,
};

export function midiToFrequency(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function parsePitchTokenToMidi(pitch: string): number {
  const cleaned = pitch.trim();
  const upper = cleaned.toUpperCase();
  if (upper === 'R' || upper === 'REST') return -1;
  if (Object.prototype.hasOwnProperty.call(DRUM_PITCHES, upper)) {
    return DRUM_PITCHES[upper];
  }
  const match = /^([A-Ga-g])([#b]?)(-?\d)$/.exec(cleaned);
  if (!match) return 60;
  const [, letter, acc, octStr] = match;
  let semi = NOTE_OFFSETS[letter.toUpperCase()] ?? 0;
  if (acc === '#') semi += 1;
  if (acc === 'b') semi -= 1;
  const octave = parseInt(octStr, 10);
  return Math.max(0, Math.min(127, (octave + 1) * 12 + semi));
}

function getAudioContextConstructor(): (new () => AudioContext) | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    AudioContext?: new () => AudioContext;
    webkitAudioContext?: new () => AudioContext;
  };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

/**
 * Filter playable notes according to instrument Solo/Mute state and loop bar bounds.
 */
export function filterPlayableNotes(
  notes: MusicNoteView[],
  instruments: MusicInstrumentView[],
  loopStartBar?: number,
  loopEndBar?: number,
): MusicNoteView[] {
  const anySolo = instruments.some((inst) => inst.solo);
  const activeInstIds = new Set(
    instruments
      .filter((inst) => (anySolo ? inst.solo && !inst.muted : !inst.muted))
      .map((inst) => inst.id),
  );
  return notes.filter((n) => {
    if (n.concert_midi < 0 || n.velocity <= 0) return false;
    if (!activeInstIds.has(n.instrument_id)) return false;
    if (loopStartBar !== undefined && n.global_bar < loopStartBar) return false;
    if (loopEndBar !== undefined && n.global_bar > loopEndBar) return false;
    return true;
  });
}

/**
 * Substitute proposed bars into `notes` when previewing a proposal in `'proposed'` mode.
 */
export function buildNotesWithProposalPreview(
  song: ShownSongView,
  proposal: MusicProposalView | null,
  mode: 'original' | 'proposed',
): MusicNoteView[] {
  if (!proposal || mode === 'original' || !proposal.section_id || !proposal.instrument_id) {
    return song.notes;
  }
  const startBar = proposal.start_bar ?? 1;
  const endBar = proposal.end_bar ?? startBar;
  const kept = song.notes.filter(
    (n) =>
      !(
        n.section_id === proposal.section_id &&
        n.instrument_id === proposal.instrument_id &&
        n.bar >= startBar &&
        n.bar <= endBar
      ),
  );

  const barLookup = new Map<number, MusicBarView>();
  for (const b of song.bars) {
    if (b.section_id === proposal.section_id) {
      barLookup.set(b.section_bar, b);
    }
  }

  const replaced: MusicNoteView[] = [];
  for (const pBar of proposal.proposed_bars) {
    const barMeta = barLookup.get(pBar.bar);
    if (!barMeta || pBar.muted) continue;
    const secPerBeat = 60.0 / barMeta.bpm;
    for (const token of pBar.notes) {
      const parts = token.split(':').map((s) => s.trim());
      if (parts.length < 3) continue;
      const beat = parseFloat(parts[0]) || 1.0;
      const pitchStr = parts[1];
      const duration = parseFloat(parts[2]) || 1.0;
      const midi = parsePitchTokenToMidi(pitchStr);
      let velocity = midi === -1 ? 0 : 85;
      let articulation = 'normal';
      for (const extra of parts.slice(3)) {
        if (extra.toLowerCase().startsWith('v')) {
          velocity = parseInt(extra.slice(1), 10) || 85;
        } else {
          articulation = extra.toLowerCase();
        }
      }
      replaced.push({
        track_id: `${proposal.section_id}_${proposal.instrument_id}`,
        section_id: proposal.section_id,
        instrument_id: proposal.instrument_id,
        bar: pBar.bar,
        global_bar: barMeta.global_bar,
        beat,
        duration,
        concert_midi: midi,
        written_pitch: midi === -1 ? 'REST' : pitchStr,
        concert_pitch: midi === -1 ? 'REST' : pitchStr,
        velocity,
        articulation,
        start_seconds: barMeta.start_seconds + (beat - 1.0) * secPerBeat,
        duration_seconds: duration * secPerBeat,
      });
    }
  }
  return [...kept, ...replaced].sort((a, b) => a.start_seconds - b.start_seconds);
}

function makeFallbackInstrument(category: InstrumentCategory): MusicInstrumentView {
  return {
    id: `audition_${category}`,
    name: category.toUpperCase(),
    category,
    role: 'lead_melody',
    midi_program: 0,
    midi_channel: category === 'drums' ? 10 : 1,
    soundfont_patch: 'gm_default',
    min_pitch: 'A0',
    max_pitch: 'C8',
    transposition_semitones: 0,
    volume: 85,
    pan: 0.0,
    muted: false,
    solo: false,
  };
}

export class MusicAudioEngine {
  private ctx: AudioContext | null = null;
  private noiseBuffer: AudioBuffer | null = null;
  private activeSessionBus: PlaybackSessionBus | null = null;
  private playing = false;
  private playbackGeneration = 0;
  private timerId: ReturnType<typeof setInterval> | null = null;
  private registry: VoiceRegistry;
  private piano = new SalamanderPiano();
  private acousticSpaceOverride: AcousticSpaceProfile | null = null;

  constructor(registry: VoiceRegistry = defaultVoiceRegistry) {
    this.registry = registry;
  }

  private ensureContext(): AudioContext | null {
    if (this.ctx) return this.ctx;
    const Ctor = getAudioContextConstructor();
    if (!Ctor) return null;
    try {
      this.ctx = new Ctor();
      this.noiseBuffer = createSharedNoiseBuffer(this.ctx);
      return this.ctx;
    } catch {
      return null;
    }
  }

  public isPlaying(): boolean {
    return this.playing;
  }

  public getAcousticSpaceOverride(): AcousticSpaceProfile | null {
    return this.acousticSpaceOverride;
  }

  public setAcousticSpaceOverride(space: AcousticSpaceProfile | null): void {
    this.acousticSpaceOverride = space;
  }

  /**
   * Register a custom `VoiceModule` plugin into this engine's synthesizer registry.
   */
  public registerVoiceModule(module: VoiceModule): void {
    this.registry.register(module);
  }

  /**
   * Start loading the recorded piano notes a song's grand piano parts will play. Other
   * instruments need nothing loaded. Safe to call repeatedly; already loaded notes are kept.
   */
  public preloadPiano(
    instruments: MusicInstrumentView[],
    notes: MusicNoteView[],
    genre?: string,
  ): Promise<void> {
    const ctx = this.ensureContext();
    if (!ctx) return Promise.resolve();
    const genreProfile = normalizeGenreProfile(genre);
    const pianoIds = new Set(
      instruments
        .filter((inst) => isSampledPianoPatch(resolveInstrumentPatch(inst, genreProfile)))
        .map((inst) => inst.id),
    );
    if (pianoIds.size === 0) return Promise.resolve();
    const midis = notes.filter((n) => pianoIds.has(n.instrument_id)).map((n) => n.concert_midi);
    return this.piano.preload(ctx, midis.length > 0 ? midis : [48, 60, 72]);
  }

  /** Play a note: the recorded piano when it applies and has loaded, otherwise synthesis. */
  private playNote(rc: VoiceRenderContext): void {
    if (this.piano.trigger(rc)) return;
    this.registry.triggerVoice(rc);
  }

  /**
   * Play a rich single-note audition when the user clicks a note or cell on the score.
   */
  public auditionNote(
    midi: number,
    category: InstrumentCategory = 'piano',
    options?: AuditionNoteOptions,
  ): void {
    if (midi < 0) return;
    const ctx = this.ensureContext();
    if (!ctx) return;
    try {
      if (ctx.state === 'suspended' && typeof ctx.resume === 'function') {
        void ctx.resume();
      }
      const genreProfile = normalizeGenreProfile(options?.genre);
      const inst = options?.instrument ?? makeFallbackInstrument(category);
      const patch = resolveInstrumentPatch(inst, genreProfile);
      const articulation = options?.articulation ?? 'normal';
      const articulationProfile = resolveArticulationEnvelope(articulation, inst.category);
      const velocity = options?.velocity ?? 88;
      const velNorm = Math.max(0.15, Math.min(1.0, velocity / 127));
      const volumeGain = ((inst.volume ?? 85) / 100) * Math.pow(velNorm, 0.85) * 0.25;

      const now = ctx.currentTime ?? 0;
      const space = this.acousticSpaceOverride ?? defaultSpaceForGenre(genreProfile);
      const auditionBus = createPlaybackSessionBus(ctx, space);
      const strip = auditionBus.createInstrumentStrip(
        inst.pan || patch.defaultPan,
        patch.reverbSend * 0.8,
        patch.stagePosition,
      );

      this.playNote({
        ctx,
        destination: strip,
        noiseBuffer: this.noiseBuffer,
        midi,
        frequency: midiToFrequency(midi),
        startAt: now,
        durationSeconds: 0.42,
        velocity,
        volumeGain,
        articulation,
        articulationProfile,
        writtenPitch: options?.writtenPitch,
        instrument: inst,
        patch,
        genre: genreProfile,
      });
      if (isSampledPianoPatch(patch)) void this.piano.preload(ctx, [midi]);
    } catch {
      // Safe no-op in headless/jsdom test environments.
    }
  }

  /**
   * Schedule and play the active notes across the selected or full bar range.
   */
  public start(options: PlaybackOptions): void {
    this.stop();
    this.playing = true;
    this.playbackGeneration += 1;
    const gen = this.playbackGeneration;

    const {
      notes,
      bars,
      instruments,
      sections = [],
      tracks = [],
      genre,
      acousticSpace,
      loopEnabled = false,
      loopStartBar,
      loopEndBar,
      onBarChange,
      onStop,
    } = options;

    const genreProfile = normalizeGenreProfile(genre);
    const space =
      acousticSpace ?? this.acousticSpaceOverride ?? defaultSpaceForGenre(genreProfile);

    const playable = filterPlayableNotes(notes, instruments, loopStartBar, loopEndBar);
    // Piano notes not loaded yet are synthesized; later passes pick up the recordings.
    void this.preloadPiano(instruments, playable, genre);
    const instMap = new Map(instruments.map((i) => [i.id, i]));
    const sectionMetaMap = new Map(sections.map((s) => [s.id, s]));
    const trackSwingMap = new Map(
      tracks.map((t) => [`${t.section_id}:${t.instrument_id}`, t.groove_swing]),
    );
    const barByGlobal = new Map(bars.map((b) => [b.global_bar, b]));
    const sectionBarCounts = new Map<string, number>();
    for (const b of bars) {
      const prev = sectionBarCounts.get(b.section_id) ?? 0;
      if (b.section_bar > prev) sectionBarCounts.set(b.section_id, b.section_bar);
    }

    const activeBars = bars.filter((b) => {
      if (loopStartBar !== undefined && b.global_bar < loopStartBar) return false;
      if (loopEndBar !== undefined && b.global_bar > loopEndBar) return false;
      return true;
    });

    const offsetSeconds = activeBars.length > 0 ? activeBars[0].start_seconds : 0;
    const totalWindowSeconds =
      activeBars.length > 0
        ? Math.max(0.25, activeBars[activeBars.length - 1].end_seconds - offsetSeconds)
        : 2.0;

    const barOrder = activeBars.length > 0 ? activeBars.map((b) => b.global_bar) : [1];
    onBarChange?.(barOrder[0]);

    const schedulePass = () => {
      const ctx = this.ensureContext();
      if (!ctx) return;
      try {
        if (ctx.state === 'suspended' && typeof ctx.resume === 'function') {
          void ctx.resume();
        }
        const baseNow = ctx.currentTime ?? 0;
        if (this.activeSessionBus) {
          this.activeSessionBus.stopAndDisconnect(baseNow);
        }
        const sessionBus = createPlaybackSessionBus(ctx, space);
        this.activeSessionBus = sessionBus;

        const patchCache = new Map<string, { patch: ResolvedInstrumentPatch; strip: AudioNode }>();
        for (const inst of instruments) {
          const patch = resolveInstrumentPatch(inst, genreProfile);
          const strip = sessionBus.createInstrumentStrip(
            inst.pan || patch.defaultPan,
            patch.reverbSend,
            patch.stagePosition,
          );
          patchCache.set(inst.id, { patch, strip });
        }

        // Track simultaneous notes per instrument+bar+beat for natural chord arpeggiation roll
        const chordRollCounter = new Map<string, number>();

        for (const n of playable) {
          const inst = instMap.get(n.instrument_id) ?? makeFallbackInstrument('piano');
          const resolved = patchCache.get(inst.id) ?? {
            patch: resolveInstrumentPatch(inst, genreProfile),
            strip: sessionBus.dryInput,
          };

          const chordKey = `${n.instrument_id}:${n.global_bar}:${n.beat.toFixed(2)}`;
          const chordVoiceIndex = chordRollCounter.get(chordKey) ?? 0;
          chordRollCounter.set(chordKey, chordVoiceIndex + 1);

          const barMeta = barByGlobal.get(n.global_bar);
          const secMeta = sectionMetaMap.get(n.section_id);
          const secTotalBars =
            secMeta?.bars ?? sectionBarCounts.get(n.section_id) ?? Math.max(1, n.bar);
          const secPerBeat = barMeta && barMeta.bpm > 0 ? 60.0 / barMeta.bpm : 0.6;
          const grooveSwing = trackSwingMap.get(`${n.section_id}:${n.instrument_id}`);

          const dur = Math.max(0.05, n.duration_seconds);
          const expression = computeNoteExpression({
            sectionId: n.section_id,
            sectionKind: secMeta?.kind,
            sectionBar: n.bar,
            sectionTotalBars: secTotalBars,
            beat: n.beat,
            durationSeconds: dur,
            secPerBeat,
            midi: n.concert_midi,
            velocity: n.velocity,
            articulation: n.articulation,
            category: inst.category,
            role: inst.role,
            genre: genreProfile,
            grooveSwing,
            chordVoiceIndex,
          });

          const startAt = Math.max(baseNow, baseNow + (n.start_seconds - offsetSeconds));
          const velNorm = Math.max(0.1, Math.min(1.0, n.velocity / 127));
          const volumeGain = ((inst.volume ?? 85) / 100) * Math.pow(velNorm, 0.85) * 0.26;
          const articulationProfile = resolveArticulationEnvelope(n.articulation, inst.category);

          this.playNote({
            ctx,
            destination: resolved.strip,
            noiseBuffer: this.noiseBuffer,
            midi: n.concert_midi,
            frequency: midiToFrequency(n.concert_midi),
            startAt,
            durationSeconds: dur,
            velocity: n.velocity,
            volumeGain,
            articulation: n.articulation,
            articulationProfile,
            writtenPitch: n.written_pitch,
            instrument: inst,
            patch: resolved.patch,
            genre: genreProfile,
            expression,
          });
        }
      } catch {
        // Ignore synthesis errors in non-browser test runners.
      }
    };

    schedulePass();

    let stepIndex = 0;
    const stepMs = Math.max(
      120,
      Math.round((totalWindowSeconds * 1000) / Math.max(1, barOrder.length)),
    );
    this.timerId = setInterval(() => {
      if (!this.playing || this.playbackGeneration !== gen) return;
      stepIndex += 1;
      if (stepIndex >= barOrder.length) {
        if (loopEnabled) {
          stepIndex = 0;
          onBarChange?.(barOrder[0]);
          schedulePass();
        } else {
          this.stop();
          onStop?.();
        }
      } else {
        onBarChange?.(barOrder[stepIndex]);
      }
    }, stepMs);
  }

  public stop(): void {
    this.playing = false;
    this.playbackGeneration += 1;
    if (this.timerId !== null) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
    if (this.activeSessionBus) {
      const now = this.ctx?.currentTime ?? 0;
      this.activeSessionBus.stopAndDisconnect(now);
      this.activeSessionBus = null;
    }
  }
}
