/**
 * Bowed & Plucked Strings Voice Module
 * (`solo_violin`, `solo_viola`, `solo_cello`, `contrabass`, `string_ensemble`).
 *
 * Implements wooden body formant filtering, multi-voice detuned bow chorus,
 * delayed expressive LFO vibrato, bow-hair rosin attack whisper, and pizzicato pluck mode.
 */

import type { ResolvedInstrumentPatch, VoiceModule, VoiceRenderContext } from '../types';

const SUPPORTED_FAMILIES = new Set([
  'solo_violin',
  'solo_viola',
  'solo_cello',
  'contrabass',
  'string_ensemble',
]);

interface StringBodySpec {
  formantHz: number;
  lowpassHz: number;
  detuneSpreadCents: number;
  baseAttackSec: number;
  baseReleaseSec: number;
}

function bodySpecForFamily(family: string): StringBodySpec {
  switch (family) {
    case 'contrabass':
      return {
        formantHz: 145,
        lowpassHz: 1500,
        detuneSpreadCents: 4.0,
        baseAttackSec: 0.055,
        baseReleaseSec: 0.22,
      };
    case 'solo_cello':
      return {
        formantHz: 245,
        lowpassHz: 2400,
        detuneSpreadCents: 4.5,
        baseAttackSec: 0.05,
        baseReleaseSec: 0.2,
      };
    case 'solo_viola':
      return {
        formantHz: 365,
        lowpassHz: 3000,
        detuneSpreadCents: 4.5,
        baseAttackSec: 0.045,
        baseReleaseSec: 0.18,
      };
    case 'string_ensemble':
      return {
        formantHz: 420,
        lowpassHz: 3400,
        detuneSpreadCents: 9.5,
        baseAttackSec: 0.085,
        baseReleaseSec: 0.28,
      };
    case 'solo_violin':
    default:
      return {
        formantHz: 490,
        lowpassHz: 3600,
        detuneSpreadCents: 4.2,
        baseAttackSec: 0.042,
        baseReleaseSec: 0.17,
      };
  }
}

export const stringsVoiceModule: VoiceModule = {
  id: 'voice.strings',

  supports(patch: ResolvedInstrumentPatch): boolean {
    return SUPPORTED_FAMILIES.has(patch.timbreFamily) || patch.category === 'strings';
  },

  trigger(rc: VoiceRenderContext): void {
    const {
      ctx,
      destination,
      noiseBuffer,
      frequency,
      startAt,
      durationSeconds,
      velocity,
      volumeGain,
      articulationProfile,
      patch,
    } = rc;

    const spec = bodySpecForFamily(patch.timbreFamily);
    const velNorm = Math.max(0.15, Math.min(1.0, velocity / 127));
    const peakGain = Math.max(0.008, volumeGain * articulationProfile.velocityScale * 0.92);

    const noteGain = ctx.createGain?.();
    if (!noteGain) return;

    // Pizzicato plucked mode vs Bowed mode
    if (articulationProfile.isPlucked) {
      const pluckEnd = startAt + Math.max(0.18, Math.min(0.45, durationSeconds * 0.65));
      noteGain.gain?.setValueAtTime?.(0.0001, startAt);
      noteGain.gain?.linearRampToValueAtTime?.(peakGain * 1.15, startAt + 0.005);
      noteGain.gain?.exponentialRampToValueAtTime?.(0.0001, pluckEnd);

      let target: AudioNode = noteGain;
      if (typeof ctx.createBiquadFilter === 'function') {
        try {
          const lp = ctx.createBiquadFilter();
          lp.type = 'lowpass';
          lp.frequency?.setValueAtTime?.(frequency * 3.8, startAt);
          lp.frequency?.exponentialRampToValueAtTime?.(frequency * 1.2, pluckEnd);
          lp.connect?.(noteGain);
          target = lp;
        } catch {
          target = noteGain;
        }
      }
      noteGain.connect?.(destination);

      const osc1 = ctx.createOscillator?.();
      const osc2 = ctx.createOscillator?.();
      if (osc1) {
        osc1.type = 'triangle';
        osc1.frequency?.setValueAtTime?.(frequency, startAt);
        osc1.connect?.(target);
        osc1.start?.(startAt);
        osc1.stop?.(pluckEnd + 0.01);
      }
      if (osc2) {
        osc2.type = 'sawtooth';
        osc2.frequency?.setValueAtTime?.(frequency, startAt);
        const g2 = ctx.createGain?.();
        if (g2) {
          g2.gain?.setValueAtTime?.(0.28, startAt);
          osc2.connect?.(g2);
          g2.connect?.(target);
        } else {
          osc2.connect?.(target);
        }
        osc2.start?.(startAt);
        osc2.stop?.(pluckEnd + 0.01);
      }
      return;
    }

    // Bowed String ADSR Envelope
    const gateDur = Math.max(0.07, durationSeconds * articulationProfile.gateRatio);
    const attackSec = Math.min(
      gateDur * 0.45,
      Math.max(0.016, spec.baseAttackSec * articulationProfile.attackScale),
    );
    const releaseSec = Math.max(0.07, spec.baseReleaseSec * articulationProfile.releaseScale);
    const totalEnd = startAt + gateDur + releaseSec;
    const sustainGain = Math.max(0.002, peakGain * 0.84);

    noteGain.gain?.setValueAtTime?.(0.0001, startAt);
    noteGain.gain?.linearRampToValueAtTime?.(peakGain, startAt + attackSec);
    noteGain.gain?.linearRampToValueAtTime?.(sustainGain, startAt + gateDur);
    noteGain.gain?.exponentialRampToValueAtTime?.(0.0001, totalEnd);

    // Spruce body formant + warm low-pass filter chain
    let voiceInput: AudioNode = noteGain;
    if (typeof ctx.createBiquadFilter === 'function') {
      try {
        const lowpass = ctx.createBiquadFilter();
        lowpass.type = 'lowpass';
        const cutoff = Math.min(
          6500,
          spec.lowpassHz * (0.75 + 0.45 * velNorm) * articulationProfile.brightnessBoost,
        );
        lowpass.frequency?.setValueAtTime?.(cutoff, startAt);
        lowpass.Q?.setValueAtTime?.(0.85, startAt);

        const bodyFormant = ctx.createBiquadFilter();
        bodyFormant.type = 'peaking';
        bodyFormant.frequency?.setValueAtTime?.(spec.formantHz, startAt);
        bodyFormant.Q?.setValueAtTime?.(1.6, startAt);
        bodyFormant.gain?.setValueAtTime?.(4.5, startAt);

        bodyFormant.connect?.(lowpass);
        lowpass.connect?.(noteGain);
        voiceInput = bodyFormant;
      } catch {
        voiceInput = noteGain;
      }
    }

    noteGain.connect?.(destination);

    // Delayed Expressive Vibrato LFO
    let lfoGain: GainNode | null = null;
    if (articulationProfile.vibratoDepthCents > 0 && gateDur > 0.18) {
      try {
        const lfo = ctx.createOscillator?.();
        lfoGain = ctx.createGain?.() ?? null;
        if (lfo && lfoGain) {
          lfo.type = 'sine';
          lfo.frequency?.setValueAtTime?.(articulationProfile.vibratoRateHz, startAt);
          // Convert cents to Hz deviation: freq * (2^(cents/1200) - 1)
          const devHz = frequency * (Math.pow(2, articulationProfile.vibratoDepthCents / 1200) - 1);
          const vibStart = Math.min(startAt + gateDur * 0.5, startAt + articulationProfile.vibratoDelaySec);
          lfoGain.gain?.setValueAtTime?.(0.0, startAt);
          lfoGain.gain?.setValueAtTime?.(0.0, vibStart);
          lfoGain.gain?.linearRampToValueAtTime?.(devHz, Math.min(totalEnd, vibStart + 0.14));
          lfo.connect?.(lfoGain);
          lfo.start?.(startAt);
          lfo.stop?.(totalEnd + 0.01);
        }
      } catch {
        lfoGain = null;
      }
    }

    // 3-layer bowed string oscillators: fundamental warm triangle + 2 detuned sawtooth bow layers
    const layers: { type: OscillatorType; detune: number; gain: number }[] = [
      { type: 'triangle', detune: 0, gain: 0.46 },
      { type: 'sawtooth', detune: -spec.detuneSpreadCents, gain: 0.32 },
      { type: 'sawtooth', detune: spec.detuneSpreadCents, gain: 0.32 },
    ];

    for (const layer of layers) {
      const osc = ctx.createOscillator?.();
      const lg = ctx.createGain?.();
      if (!osc || !lg) continue;
      osc.type = layer.type;
      osc.frequency?.setValueAtTime?.(frequency, startAt);
      if (osc.detune && typeof osc.detune.setValueAtTime === 'function') {
        osc.detune.setValueAtTime(layer.detune, startAt);
      }
      if (lfoGain && osc.frequency) {
        try {
          lfoGain.connect?.(osc.frequency);
        } catch {
          // Ignore AudioParam connection if unsupported in mock
        }
      }
      lg.gain?.setValueAtTime?.(layer.gain, startAt);
      osc.connect?.(lg);
      lg.connect?.(voiceInput);
      osc.start?.(startAt);
      osc.stop?.(totalEnd + 0.01);
    }

    // Subtle bow-hair rosin friction on initial bow stroke
    if (noiseBuffer && typeof ctx.createBufferSource === 'function') {
      try {
        const bowNoise = ctx.createBufferSource();
        bowNoise.buffer = noiseBuffer;
        const bowGain = ctx.createGain?.();
        if (bowGain) {
          bowGain.gain?.setValueAtTime?.(0.0001, startAt);
          bowGain.gain?.linearRampToValueAtTime?.(peakGain * 0.055 * velNorm, startAt + attackSec * 0.6);
          bowGain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + attackSec + 0.04);
          bowNoise.connect?.(bowGain);
          bowGain.connect?.(voiceInput);
          bowNoise.start?.(startAt, 0.1, attackSec + 0.05);
          bowNoise.stop?.(startAt + attackSec + 0.05);
        }
      } catch {
        // Ignore in minimal mock contexts
      }
    }
  },
};
