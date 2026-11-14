/**
 * Plucked Guitar, Analog Synth Pad/Lead, and Choir Vocal Voice Module
 * (`nylon_guitar`, `jazz_guitar`, `steel_guitar`, `warm_pad`, `lead_synth`, `choir_vocal`).
 */

import type { ResolvedInstrumentPatch, VoiceModule, VoiceRenderContext } from '../types';

const SUPPORTED_FAMILIES = new Set([
  'nylon_guitar',
  'jazz_guitar',
  'steel_guitar',
  'warm_pad',
  'lead_synth',
  'choir_vocal',
]);

export const guitarSynthVoiceModule: VoiceModule = {
  id: 'voice.guitar_synth',

  supports(patch: ResolvedInstrumentPatch): boolean {
    return (
      SUPPORTED_FAMILIES.has(patch.timbreFamily) ||
      patch.category === 'guitar' ||
      patch.category === 'synth' ||
      patch.category === 'vocal'
    );
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

    const velNorm = Math.max(0.15, Math.min(1.0, velocity / 127));
    const isGuitar =
      patch.category === 'guitar' ||
      patch.timbreFamily === 'nylon_guitar' ||
      patch.timbreFamily === 'jazz_guitar' ||
      patch.timbreFamily === 'steel_guitar';

    const gateDur = Math.max(0.06, durationSeconds * articulationProfile.gateRatio);
    const attackSec = isGuitar
      ? Math.max(0.003, 0.005 * articulationProfile.attackScale)
      : Math.max(0.025, 0.065 * articulationProfile.attackScale);
    const releaseSec = Math.max(0.08, (isGuitar ? 0.24 : 0.3) * articulationProfile.releaseScale);
    const totalEnd = startAt + gateDur + releaseSec;
    const peakGain = Math.max(0.008, volumeGain * articulationProfile.velocityScale);

    const noteGain = ctx.createGain?.();
    if (!noteGain) return;

    noteGain.gain?.setValueAtTime?.(0.0001, startAt);
    noteGain.gain?.linearRampToValueAtTime?.(peakGain, startAt + attackSec);
    if (isGuitar) {
      noteGain.gain?.exponentialRampToValueAtTime?.(
        Math.max(0.002, peakGain * 0.28),
        startAt + Math.min(gateDur, 0.45),
      );
    } else {
      noteGain.gain?.linearRampToValueAtTime?.(peakGain * 0.82, startAt + gateDur);
    }
    noteGain.gain?.exponentialRampToValueAtTime?.(0.0001, totalEnd);

    let voiceInput: AudioNode = noteGain;
    if (typeof ctx.createBiquadFilter === 'function') {
      try {
        const lp = ctx.createBiquadFilter();
        lp.type = 'lowpass';
        if (isGuitar) {
          const pluckBright =
            patch.timbreFamily === 'jazz_guitar'
              ? frequency * 3.6
              : patch.timbreFamily === 'nylon_guitar'
                ? frequency * 4.5
                : frequency * 6.2;
          lp.frequency?.setValueAtTime?.(
            Math.min(7500, pluckBright * (0.7 + 0.5 * velNorm)),
            startAt,
          );
          lp.frequency?.exponentialRampToValueAtTime?.(
            Math.max(240, frequency * 1.6),
            startAt + Math.min(gateDur, 0.25),
          );
          lp.Q?.setValueAtTime?.(1.2, startAt);
        } else {
          const padCutoff = Math.min(4800, frequency * (2.6 + velNorm * 2.5));
          lp.frequency?.setValueAtTime?.(padCutoff, startAt);
          lp.Q?.setValueAtTime?.(patch.timbreFamily === 'choir_vocal' ? 2.2 : 0.9, startAt);
        }
        lp.connect?.(noteGain);
        voiceInput = lp;
      } catch {
        voiceInput = noteGain;
      }
    }

    noteGain.connect?.(destination);

    // Dual chorused / plucked oscillators
    const osc1 = ctx.createOscillator?.();
    const osc2 = ctx.createOscillator?.();
    const g2 = ctx.createGain?.();

    if (osc1) {
      osc1.type = isGuitar ? 'triangle' : 'sawtooth';
      osc1.frequency?.setValueAtTime?.(frequency, startAt);
      if (!isGuitar && osc1.detune && typeof osc1.detune.setValueAtTime === 'function') {
        osc1.detune.setValueAtTime(-6.0, startAt);
      }
      osc1.connect?.(voiceInput);
      osc1.start?.(startAt);
      osc1.stop?.(totalEnd + 0.01);
    }

    if (osc2 && g2) {
      osc2.type = patch.timbreFamily === 'jazz_guitar' ? 'sine' : 'sawtooth';
      osc2.frequency?.setValueAtTime?.(
        patch.timbreFamily === 'jazz_guitar' ? frequency * 2.0 : frequency,
        startAt,
      );
      if (osc2.detune && typeof osc2.detune.setValueAtTime === 'function') {
        osc2.detune.setValueAtTime(isGuitar ? 1.8 : 6.0, startAt);
      }
      g2.gain?.setValueAtTime?.(isGuitar ? 0.35 * velNorm : 0.48, startAt);
      if (isGuitar) {
        g2.gain?.exponentialRampToValueAtTime?.(0.01, startAt + Math.min(gateDur, 0.22));
      }
      osc2.connect?.(g2);
      g2.connect?.(voiceInput);
      osc2.start?.(startAt);
      osc2.stop?.(totalEnd + 0.01);
    }
  },
};
