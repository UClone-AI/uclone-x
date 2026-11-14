/**
 * Piano & Keyboard Voice Module (`grand_piano`, `rhodes_ep`, `harpsichord`).
 *
 * Implements multi-partial harmonic synthesis with inharmonicity stretch,
 * velocity-sensitive dynamic low-pass filtering, FM tine bell overtones (for Rhodes EP),
 * and a physical hammer-felt strike transient.
 */

import type { ResolvedInstrumentPatch, VoiceModule, VoiceRenderContext } from '../types';

const SUPPORTED_FAMILIES = new Set(['grand_piano', 'rhodes_ep', 'harpsichord']);

export const pianoVoiceModule: VoiceModule = {
  id: 'voice.piano',

  supports(patch: ResolvedInstrumentPatch): boolean {
    return SUPPORTED_FAMILIES.has(patch.timbreFamily) || patch.category === 'piano';
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

    const velNorm = Math.max(0.1, Math.min(1.0, velocity / 127));
    const gateDur = Math.max(0.06, durationSeconds * articulationProfile.gateRatio);
    const releaseSec = Math.max(
      0.08,
      (patch.timbreFamily === 'harpsichord' ? 0.14 : 0.32) * articulationProfile.releaseScale,
    );
    const totalEnd = startAt + gateDur + releaseSec;
    const peakGain = Math.max(0.008, volumeGain * articulationProfile.velocityScale);

    // Master note envelope + dynamic velocity low-pass filter
    const noteGain = ctx.createGain?.();
    if (!noteGain) return;

    const attackSec = Math.max(0.003, 0.006 * articulationProfile.attackScale);
    const sustainLevel = Math.max(0.002, peakGain * (patch.timbreFamily === 'rhodes_ep' ? 0.52 : 0.34));

    noteGain.gain?.setValueAtTime?.(0.0001, startAt);
    noteGain.gain?.linearRampToValueAtTime?.(peakGain, startAt + attackSec);
    noteGain.gain?.exponentialRampToValueAtTime?.(sustainLevel, startAt + attackSec + gateDur * 0.65);
    noteGain.gain?.exponentialRampToValueAtTime?.(0.0001, totalEnd);

    let voiceTarget: AudioNode = noteGain;
    if (typeof ctx.createBiquadFilter === 'function') {
      try {
        const filter = ctx.createBiquadFilter();
        filter.type = 'lowpass';
        const baseCutoff = Math.min(
          12000,
          frequency * (2.8 + velNorm * 5.5) * articulationProfile.brightnessBoost,
        );
        const tailCutoff = Math.max(380, frequency * 2.1);
        filter.frequency?.setValueAtTime?.(baseCutoff, startAt);
        filter.frequency?.exponentialRampToValueAtTime?.(tailCutoff, startAt + gateDur + 0.05);
        filter.Q?.setValueAtTime?.(0.7, startAt);
        filter.connect?.(noteGain);
        voiceTarget = filter;
      } catch {
        voiceTarget = noteGain;
      }
    }

    noteGain.connect?.(destination);

    if (patch.timbreFamily === 'rhodes_ep') {
      triggerRhodesPartials(ctx, voiceTarget, frequency, startAt, totalEnd, velNorm);
      return;
    }

    if (patch.timbreFamily === 'harpsichord') {
      triggerHarpsichordPartials(ctx, voiceTarget, frequency, startAt, totalEnd);
      return;
    }

    // Acoustic Grand Piano: Unison detuned strings + fast-decaying upper harmonics + hammer felt thump
    const partials: {
      ratio: number;
      detuneCents: number;
      type: OscillatorType;
      relAmp: number;
      decayMult: number;
    }[] = [
      { ratio: 1.0, detuneCents: -1.6, type: 'triangle', relAmp: 0.62, decayMult: 1.0 },
      { ratio: 1.0, detuneCents: 1.6, type: 'sine', relAmp: 0.48, decayMult: 1.0 },
      { ratio: 2.0008, detuneCents: 0.5, type: 'sine', relAmp: 0.28 * velNorm, decayMult: 0.55 },
      { ratio: 3.0025, detuneCents: -0.8, type: 'sine', relAmp: 0.14 * velNorm * velNorm, decayMult: 0.35 },
    ];

    for (const p of partials) {
      const partialFreq = frequency * p.ratio;
      if (partialFreq > 15500) continue;
      const osc = ctx.createOscillator?.();
      const pGain = ctx.createGain?.();
      if (!osc || !pGain) continue;

      osc.type = p.type;
      osc.frequency?.setValueAtTime?.(partialFreq, startAt);
      if (osc.detune && typeof osc.detune.setValueAtTime === 'function') {
        osc.detune.setValueAtTime(p.detuneCents, startAt);
      }

      const pEnd = startAt + (gateDur + releaseSec) * p.decayMult;
      pGain.gain?.setValueAtTime?.(Math.max(0.001, p.relAmp), startAt);
      pGain.gain?.exponentialRampToValueAtTime?.(0.0001, Math.max(startAt + 0.04, pEnd));

      osc.connect?.(pGain);
      pGain.connect?.(voiceTarget);
      osc.start?.(startAt);
      osc.stop?.(Math.min(totalEnd + 0.02, Math.max(startAt + 0.05, pEnd + 0.01)));
    }

    // Hammer-felt physical strike transient
    if (noiseBuffer && typeof ctx.createBufferSource === 'function') {
      try {
        const noiseSrc = ctx.createBufferSource();
        noiseSrc.buffer = noiseBuffer;
        const hammerGain = ctx.createGain?.();
        if (hammerGain) {
          const strikeAmp = peakGain * 0.22 * velNorm;
          hammerGain.gain?.setValueAtTime?.(strikeAmp, startAt);
          hammerGain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + 0.022);
          if (typeof ctx.createBiquadFilter === 'function') {
            const bp = ctx.createBiquadFilter();
            bp.type = 'bandpass';
            bp.frequency?.setValueAtTime?.(Math.min(4200, frequency * 3.5 + 900), startAt);
            bp.Q?.setValueAtTime?.(1.4, startAt);
            noiseSrc.connect?.(bp);
            bp.connect?.(hammerGain);
          } else {
            noiseSrc.connect?.(hammerGain);
          }
          hammerGain.connect?.(destination);
          noiseSrc.start?.(startAt, 0, 0.03);
          noiseSrc.stop?.(startAt + 0.03);
        }
      } catch {
        // Ignore in minimal mock contexts
      }
    }
  },
};

function triggerRhodesPartials(
  ctx: AudioContext,
  target: AudioNode,
  frequency: number,
  startAt: number,
  totalEnd: number,
  velNorm: number,
): void {
  const carrier = ctx.createOscillator?.();
  const body = ctx.createOscillator?.();
  const tine = ctx.createOscillator?.();
  const tineGain = ctx.createGain?.();
  const bodyGain = ctx.createGain?.();

  if (!carrier) return;
  carrier.type = 'sine';
  carrier.frequency?.setValueAtTime?.(frequency, startAt);
  carrier.connect?.(target);
  carrier.start?.(startAt);
  carrier.stop?.(totalEnd + 0.01);

  if (body && bodyGain) {
    body.type = 'triangle';
    body.frequency?.setValueAtTime?.(frequency * 2.0, startAt);
    bodyGain.gain?.setValueAtTime?.(0.24 * velNorm, startAt);
    bodyGain.gain?.exponentialRampToValueAtTime?.(0.01, totalEnd);
    body.connect?.(bodyGain);
    bodyGain.connect?.(target);
    body.start?.(startAt);
    body.stop?.(totalEnd + 0.01);
  }

  // Classic Rhodes 7th-harmonic tine chime on attack
  if (tine && tineGain && frequency * 7 < 14000) {
    tine.type = 'sine';
    tine.frequency?.setValueAtTime?.(frequency * 7.0, startAt);
    tineGain.gain?.setValueAtTime?.(0.11 * velNorm * velNorm, startAt);
    tineGain.gain?.exponentialRampToValueAtTime?.(0.0001, Math.min(totalEnd, startAt + 0.16));
    tine.connect?.(tineGain);
    tineGain.connect?.(target);
    tine.start?.(startAt);
    tine.stop?.(Math.min(totalEnd, startAt + 0.17));
  }
}

function triggerHarpsichordPartials(
  ctx: AudioContext,
  target: AudioNode,
  frequency: number,
  startAt: number,
  totalEnd: number,
): void {
  const main = ctx.createOscillator?.();
  const oct = ctx.createOscillator?.();
  const octGain = ctx.createGain?.();
  if (!main) return;
  main.type = 'sawtooth';
  main.frequency?.setValueAtTime?.(frequency, startAt);
  main.connect?.(target);
  main.start?.(startAt);
  main.stop?.(totalEnd + 0.01);

  if (oct && octGain && frequency * 2 < 14000) {
    oct.type = 'triangle';
    oct.frequency?.setValueAtTime?.(frequency * 2.0, startAt);
    octGain.gain?.setValueAtTime?.(0.35, startAt);
    octGain.gain?.exponentialRampToValueAtTime?.(0.001, totalEnd);
    oct.connect?.(octGain);
    octGain.connect?.(target);
    oct.start?.(startAt);
    oct.stop?.(totalEnd + 0.01);
  }
}
