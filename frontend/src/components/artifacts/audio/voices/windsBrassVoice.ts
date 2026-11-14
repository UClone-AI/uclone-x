/**
 * Brass & Woodwinds Voice Module
 * (`trumpet`, `trombone`, `french_horn`, `saxophone`, `flute`, `clarinet`, `oboe`).
 *
 * - Brass: Dynamic lip-buzz harmonic filter bloom ("brass swell") + bell formant resonance.
 * - Saxophone / Oboe: Conical reed harmonic blend + vocal formant peak + subtle pitch scoop.
 * - Clarinet: Cylindrical odd-harmonic woody reed tone.
 * - Flute: Singing fundamental + overblown 2nd harmonic + airy embouchure breath turbulence.
 */

import type { ResolvedInstrumentPatch, VoiceModule, VoiceRenderContext } from '../types';

const SUPPORTED_FAMILIES = new Set([
  'trumpet',
  'trombone',
  'french_horn',
  'saxophone',
  'flute',
  'clarinet',
  'oboe',
]);

export const windsBrassVoiceModule: VoiceModule = {
  id: 'voice.winds_brass',

  supports(patch: ResolvedInstrumentPatch): boolean {
    return (
      SUPPORTED_FAMILIES.has(patch.timbreFamily) ||
      patch.category === 'brass' ||
      patch.category === 'woodwinds'
    );
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

    const velNorm = Math.max(0.15, Math.min(1.0, velocity / 127));
    const gateDur = Math.max(0.06, durationSeconds * articulationProfile.gateRatio);
    const isBrass =
      patch.category === 'brass' ||
      patch.timbreFamily === 'trumpet' ||
      patch.timbreFamily === 'trombone' ||
      patch.timbreFamily === 'french_horn';

    const attackSec = Math.min(
      gateDur * 0.4,
      Math.max(0.012, (isBrass ? 0.028 : 0.035) * articulationProfile.attackScale),
    );
    const releaseSec = Math.max(0.06, 0.14 * articulationProfile.releaseScale);
    const totalEnd = startAt + gateDur + releaseSec;
    const peakGain = Math.max(0.008, volumeGain * articulationProfile.velocityScale * 0.9);

    const noteGain = ctx.createGain?.();
    if (!noteGain) return;

    noteGain.gain?.setValueAtTime?.(0.0001, startAt);
    noteGain.gain?.linearRampToValueAtTime?.(peakGain, startAt + attackSec);
    noteGain.gain?.linearRampToValueAtTime?.(peakGain * 0.86, startAt + gateDur);
    noteGain.gain?.exponentialRampToValueAtTime?.(0.0001, totalEnd);

    let voiceInput: AudioNode = noteGain;
    if (typeof ctx.createBiquadFilter === 'function') {
      try {
        const lp = ctx.createBiquadFilter();
        lp.type = 'lowpass';
        const maxMult =
          patch.timbreFamily === 'french_horn'
            ? 3.6
            : patch.timbreFamily === 'flute'
              ? 3.2
              : patch.timbreFamily === 'clarinet'
                ? 4.2
                : 6.5;
        const peakCutoff = Math.min(
          8500,
          frequency * (2.0 + velNorm * maxMult) * articulationProfile.brightnessBoost,
        );
        // Brass filter swell on attack
        lp.frequency?.setValueAtTime?.(Math.max(250, frequency * 1.4), startAt);
        lp.frequency?.exponentialRampToValueAtTime?.(peakCutoff, startAt + attackSec + 0.015);
        lp.frequency?.exponentialRampToValueAtTime?.(
          Math.max(300, peakCutoff * 0.72),
          startAt + gateDur,
        );
        lp.Q?.setValueAtTime?.(isBrass ? 1.6 : 1.0, startAt);

        const formant = ctx.createBiquadFilter();
        formant.type = 'peaking';
        const formantHz =
          patch.timbreFamily === 'french_horn'
            ? 620
            : patch.timbreFamily === 'trombone'
              ? 780
              : patch.timbreFamily === 'saxophone'
                ? 1080
                : 1250;
        formant.frequency?.setValueAtTime?.(formantHz, startAt);
        formant.Q?.setValueAtTime?.(1.8, startAt);
        formant.gain?.setValueAtTime?.(3.8, startAt);

        formant.connect?.(lp);
        lp.connect?.(noteGain);
        voiceInput = formant;
      } catch {
        voiceInput = noteGain;
      }
    }

    noteGain.connect?.(destination);

    // Delayed Vibrato LFO
    let lfoGain: GainNode | null = null;
    if (articulationProfile.vibratoDepthCents > 0 && gateDur > 0.16) {
      try {
        const lfo = ctx.createOscillator?.();
        lfoGain = ctx.createGain?.() ?? null;
        if (lfo && lfoGain) {
          lfo.type = 'sine';
          lfo.frequency?.setValueAtTime?.(articulationProfile.vibratoRateHz, startAt);
          const devHz = frequency * (Math.pow(2, articulationProfile.vibratoDepthCents / 1200) - 1);
          const vibStart = startAt + articulationProfile.vibratoDelaySec;
          lfoGain.gain?.setValueAtTime?.(0, startAt);
          lfoGain.gain?.setValueAtTime?.(0, vibStart);
          lfoGain.gain?.linearRampToValueAtTime?.(devHz, Math.min(totalEnd, vibStart + 0.12));
          lfo.connect?.(lfoGain);
          lfo.start?.(startAt);
          lfo.stop?.(totalEnd + 0.01);
        }
      } catch {
        lfoGain = null;
      }
    }

    const primaryType: OscillatorType =
      patch.timbreFamily === 'flute'
        ? 'sine'
        : patch.timbreFamily === 'clarinet'
          ? 'square'
          : 'sawtooth';
    const secondaryType: OscillatorType =
      patch.timbreFamily === 'saxophone' ? 'square' : 'triangle';

    const osc1 = ctx.createOscillator?.();
    const osc2 = ctx.createOscillator?.();
    const g2 = ctx.createGain?.();

    if (osc1) {
      osc1.type = primaryType;
      osc1.frequency?.setValueAtTime?.(frequency, startAt);
      if (lfoGain && osc1.frequency) {
        try {
          lfoGain.connect?.(osc1.frequency);
        } catch {
          // Ignore
        }
      }
      osc1.connect?.(voiceInput);
      osc1.start?.(startAt);
      osc1.stop?.(totalEnd + 0.01);
    }

    if (osc2 && g2) {
      osc2.type = secondaryType;
      const freqRatio = patch.timbreFamily === 'flute' ? 2.0 : 1.0;
      osc2.frequency?.setValueAtTime?.(frequency * freqRatio, startAt);
      if (osc2.detune && typeof osc2.detune.setValueAtTime === 'function') {
        osc2.detune.setValueAtTime(3.2, startAt);
      }
      if (lfoGain && osc2.frequency) {
        try {
          lfoGain.connect?.(osc2.frequency);
        } catch {
          // Ignore
        }
      }
      g2.gain?.setValueAtTime?.(patch.timbreFamily === 'flute' ? 0.22 : 0.42, startAt);
      osc2.connect?.(g2);
      g2.connect?.(voiceInput);
      osc2.start?.(startAt);
      osc2.stop?.(totalEnd + 0.01);
    }

    // Airy embouchure / breath transient for woodwinds & brass
    if (noiseBuffer && typeof ctx.createBufferSource === 'function') {
      try {
        const breath = ctx.createBufferSource();
        breath.buffer = noiseBuffer;
        const bGain = ctx.createGain?.();
        if (bGain) {
          const breathAmt = patch.timbreFamily === 'flute' ? 0.065 : 0.035;
          bGain.gain?.setValueAtTime?.(peakGain * breathAmt * velNorm, startAt);
          bGain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + Math.min(gateDur, 0.12));
          breath.connect?.(bGain);
          bGain.connect?.(voiceInput);
          breath.start?.(startAt, 0.25, 0.14);
          breath.stop?.(startAt + 0.14);
        }
      } catch {
        // Ignore in minimal mock contexts
      }
    }
  },
};
