/**
 * Multi-Genre Bass Voice Module (`upright_bass`, `electric_bass`, `sub_808_bass`).
 *
 * - `upright_bass` (Jazz / Classical): Woody acoustic double-bass finger pluck with resonant
 *   body filter decay.
 * - `electric_bass` (Pop / Rock): Punchy fingered electric bass with tight low-mid definition.
 * - `sub_808_bass` (Hip-Hop / Boom-Bap): Deep 808 sub-bass with attack pitch punch and
 *   warm upper-harmonic saturation so sub notes translate on small speakers.
 */

import type { ResolvedInstrumentPatch, VoiceModule, VoiceRenderContext } from '../types';

const SUPPORTED_FAMILIES = new Set(['upright_bass', 'electric_bass', 'sub_808_bass']);

export const bassVoiceModule: VoiceModule = {
  id: 'voice.bass',

  supports(patch: ResolvedInstrumentPatch): boolean {
    return SUPPORTED_FAMILIES.has(patch.timbreFamily) || patch.category === 'bass';
  },

  trigger(rc: VoiceRenderContext): void {
    const {
      ctx,
      destination,
      frequency,
      startAt,
      durationSeconds,
      velocity,
      volumeGain,
      articulationProfile,
      patch,
    } = rc;

    const velNorm = Math.max(0.2, Math.min(1.0, velocity / 127));
    const gateDur = Math.max(0.06, durationSeconds * articulationProfile.gateRatio);
    const releaseSec = Math.max(0.06, 0.14 * articulationProfile.releaseScale);
    const totalEnd = startAt + gateDur + releaseSec;
    const peakGain = Math.max(0.01, volumeGain * articulationProfile.velocityScale * 1.15);

    const noteGain = ctx.createGain?.();
    if (!noteGain) return;

    const attackSec = Math.max(0.004, 0.008 * articulationProfile.attackScale);
    const sustainGain =
      patch.timbreFamily === 'upright_bass'
        ? Math.max(0.002, peakGain * 0.38)
        : Math.max(0.003, peakGain * 0.68);

    noteGain.gain?.setValueAtTime?.(0.0001, startAt);
    noteGain.gain?.linearRampToValueAtTime?.(peakGain, startAt + attackSec);
    noteGain.gain?.exponentialRampToValueAtTime?.(sustainGain, startAt + Math.min(gateDur, 0.35));
    noteGain.gain?.exponentialRampToValueAtTime?.(0.0001, totalEnd);

    let filterInput: AudioNode = noteGain;
    if (typeof ctx.createBiquadFilter === 'function') {
      try {
        const lp = ctx.createBiquadFilter();
        lp.type = 'lowpass';
        const startCutoff =
          patch.timbreFamily === 'sub_808_bass'
            ? frequency * (3.2 + velNorm * 1.8)
            : patch.timbreFamily === 'upright_bass'
              ? frequency * (4.8 + velNorm * 3.0)
              : frequency * (5.5 + velNorm * 4.0);
        const endCutoff = Math.max(110, frequency * 1.8);
        lp.frequency?.setValueAtTime?.(
          Math.min(3200, startCutoff * articulationProfile.brightnessBoost),
          startAt,
        );
        lp.frequency?.exponentialRampToValueAtTime?.(endCutoff, startAt + Math.min(gateDur, 0.28));
        lp.Q?.setValueAtTime?.(patch.timbreFamily === 'upright_bass' ? 1.8 : 1.1, startAt);
        lp.connect?.(noteGain);
        filterInput = lp;
      } catch {
        filterInput = noteGain;
      }
    }

    noteGain.connect?.(destination);

    // 1. Sub/Fundamental oscillator
    const subOsc = ctx.createOscillator?.();
    if (subOsc) {
      subOsc.type = patch.timbreFamily === 'sub_808_bass' ? 'sine' : 'triangle';
      if (patch.timbreFamily === 'sub_808_bass') {
        // 808 initial pitch punch (+4 semitones -> fundamental in 28ms)
        subOsc.frequency?.setValueAtTime?.(frequency * 1.26, startAt);
        subOsc.frequency?.exponentialRampToValueAtTime?.(frequency, startAt + 0.028);
      } else {
        subOsc.frequency?.setValueAtTime?.(frequency, startAt);
      }
      subOsc.connect?.(filterInput);
      subOsc.start?.(startAt);
      subOsc.stop?.(totalEnd + 0.01);
    }

    // 2. Upper harmonic / string definition oscillator (makes bass audible and warm on all speakers)
    const harmOsc = ctx.createOscillator?.();
    const harmGain = ctx.createGain?.();
    if (harmOsc && harmGain) {
      harmOsc.type = patch.timbreFamily === 'electric_bass' ? 'sawtooth' : 'triangle';
      harmOsc.frequency?.setValueAtTime?.(
        patch.timbreFamily === 'sub_808_bass' ? frequency * 2.0 : frequency,
        startAt,
      );
      const harmLevel =
        patch.timbreFamily === 'sub_808_bass'
          ? 0.24 * velNorm
          : patch.timbreFamily === 'upright_bass'
            ? 0.38 * velNorm
            : 0.45 * velNorm;
      harmGain.gain?.setValueAtTime?.(harmLevel, startAt);
      harmGain.gain?.exponentialRampToValueAtTime?.(0.01, totalEnd);
      harmOsc.connect?.(harmGain);
      harmGain.connect?.(filterInput);
      harmOsc.start?.(startAt);
      harmOsc.stop?.(totalEnd + 0.01);
    }
  },
};
