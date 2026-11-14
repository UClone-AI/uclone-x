/**
 * Studio Multi-Genre Drum & Percussion Synthesizer Module
 * (`drum_kit_acoustic`, `drum_kit_jazz`, `drum_kit_808`, `drum_kit_orchestral`).
 *
 * Synthesizes realistic percussion using:
 * - Kick (36): Pitch-swept membrane oscillator (155Hz -> 46Hz) + beater click transient.
 * - Snare (38): Dual drumhead shell modes (185Hz + 330Hz) + high-pass snare wire noise burst.
 * - Rimshot (37): High-Q woody resonant impulse.
 * - Hand Clap (39): Micro-staggered bandpass noise bursts (0ms, 11ms, 22ms, 34ms).
 * - Hi-Hats (42 Closed, 46 Open) & Cymbals (49 Crash, 51 Ride): 6-ratio metallic square
 *   matrix + high-passed sizzle noise + Ride bell ping (2750Hz).
 * - Toms (45 Low, 47 Mid, 50 High) / Timpani: Pitch-drop resonant drum shells.
 */

import type { ResolvedInstrumentPatch, VoiceModule, VoiceRenderContext } from '../types';

const SUPPORTED_FAMILIES = new Set([
  'drum_kit_acoustic',
  'drum_kit_jazz',
  'drum_kit_808',
  'drum_kit_orchestral',
]);

// Classic TR-808 / Cymbal metallic inharmonic frequency ratios
const METALLIC_FREQS = [205.3, 304.4, 369.6, 522.7, 680.1, 812.8];

export const drumKitVoiceModule: VoiceModule = {
  id: 'voice.drums',

  supports(patch: ResolvedInstrumentPatch): boolean {
    return SUPPORTED_FAMILIES.has(patch.timbreFamily) || patch.category === 'drums';
  },

  trigger(rc: VoiceRenderContext): void {
    const {
      ctx,
      destination,
      noiseBuffer,
      midi,
      startAt,
      velocity,
      volumeGain,
      writtenPitch,
      patch,
    } = rc;

    const velNorm = Math.max(0.15, Math.min(1.0, velocity / 127));
    const amp = Math.max(0.015, volumeGain * 1.35);
    const upper = (writtenPitch ?? '').toUpperCase();
    const is808 = patch.timbreFamily === 'drum_kit_808';
    const isJazz = patch.timbreFamily === 'drum_kit_jazz';
    const isOrch = patch.timbreFamily === 'drum_kit_orchestral';

    // Determine drum piece from MIDI note or pitch alias
    if (midi === 36 || upper === 'KICK' || upper === 'BD') {
      triggerKick(ctx, destination, noiseBuffer, startAt, amp, velNorm, is808, isJazz, isOrch);
      return;
    }
    if (midi === 37 || upper === 'RIM') {
      triggerRimshot(ctx, destination, noiseBuffer, startAt, amp, velNorm);
      return;
    }
    if (midi === 38 || upper === 'SNARE' || upper === 'SD') {
      triggerSnare(ctx, destination, noiseBuffer, startAt, amp, velNorm, is808, isJazz);
      return;
    }
    if (midi === 39 || upper === 'CLAP') {
      triggerClap(ctx, destination, noiseBuffer, startAt, amp, velNorm);
      return;
    }
    if (midi === 42 || upper === 'CLOSED_HAT' || upper === 'HIHAT' || upper === 'HH') {
      triggerHiHat(ctx, destination, noiseBuffer, startAt, amp, velNorm, false, isJazz);
      return;
    }
    if (midi === 46 || upper === 'OPEN_HAT' || upper === 'OH') {
      triggerHiHat(ctx, destination, noiseBuffer, startAt, amp, velNorm, true, isJazz);
      return;
    }
    if (midi === 49 || upper === 'CRASH') {
      triggerCymbal(ctx, destination, noiseBuffer, startAt, amp, velNorm, 'crash');
      return;
    }
    if (midi === 51 || upper === 'RIDE') {
      triggerCymbal(ctx, destination, noiseBuffer, startAt, amp, velNorm, 'ride');
      return;
    }

    // Toms (45 Low, 47 Mid, 50 High) or fallback pitched percussion
    const tomBaseHz = midi <= 45 ? 82 : midi <= 48 ? 115 : 155;
    triggerTom(ctx, destination, noiseBuffer, startAt, amp, velNorm, tomBaseHz, isOrch);
  },
};

function triggerKick(
  ctx: AudioContext,
  destination: AudioNode,
  noiseBuffer: AudioBuffer | null,
  startAt: number,
  amp: number,
  velNorm: number,
  is808: boolean,
  isJazz: boolean,
  isOrch: boolean,
): void {
  const osc = ctx.createOscillator?.();
  const gain = ctx.createGain?.();
  if (!osc || !gain) return;

  const decaySec = is808 ? 0.52 : isOrch ? 0.65 : isJazz ? 0.24 : 0.32;
  const startFreq = isJazz ? 125 : is808 ? 155 : 145;
  const endFreq = is808 ? 44 : isJazz ? 58 : 49;

  osc.type = 'sine';
  osc.frequency?.setValueAtTime?.(startFreq, startAt);
  osc.frequency?.exponentialRampToValueAtTime?.(endFreq, startAt + 0.042);
  osc.frequency?.exponentialRampToValueAtTime?.(Math.max(36, endFreq * 0.88), startAt + decaySec);

  gain.gain?.setValueAtTime?.(amp * 1.25, startAt);
  gain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + decaySec);

  osc.connect?.(gain);
  gain.connect?.(destination);
  osc.start?.(startAt);
  osc.stop?.(startAt + decaySec + 0.01);

  // Punchy 2nd harmonic knock for speaker translation
  const punchOsc = ctx.createOscillator?.();
  const punchGain = ctx.createGain?.();
  if (punchOsc && punchGain) {
    punchOsc.type = 'triangle';
    punchOsc.frequency?.setValueAtTime?.(startFreq * 1.6, startAt);
    punchOsc.frequency?.exponentialRampToValueAtTime?.(endFreq * 1.5, startAt + 0.035);
    punchGain.gain?.setValueAtTime?.(amp * 0.45 * velNorm, startAt);
    punchGain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + 0.065);
    punchOsc.connect?.(punchGain);
    punchGain.connect?.(destination);
    punchOsc.start?.(startAt);
    punchOsc.stop?.(startAt + 0.07);
  }

  // Beater click transient
  if (noiseBuffer && typeof ctx.createBufferSource === 'function') {
    try {
      const click = ctx.createBufferSource();
      click.buffer = noiseBuffer;
      const clickGain = ctx.createGain?.();
      if (clickGain) {
        clickGain.gain?.setValueAtTime?.(amp * 0.35 * velNorm, startAt);
        clickGain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + 0.016);
        if (typeof ctx.createBiquadFilter === 'function') {
          const bp = ctx.createBiquadFilter();
          bp.type = 'bandpass';
          bp.frequency?.setValueAtTime?.(isJazz ? 1600 : 2600, startAt);
          bp.Q?.setValueAtTime?.(1.4, startAt);
          click.connect?.(bp);
          bp.connect?.(clickGain);
        } else {
          click.connect?.(clickGain);
        }
        clickGain.connect?.(destination);
        click.start?.(startAt, 0, 0.02);
        click.stop?.(startAt + 0.02);
      }
    } catch {
      // Ignore in test mocks
    }
  }
}

function triggerSnare(
  ctx: AudioContext,
  destination: AudioNode,
  noiseBuffer: AudioBuffer | null,
  startAt: number,
  amp: number,
  velNorm: number,
  is808: boolean,
  isJazz: boolean,
): void {
  // 1. Dual drumhead resonant shell modes (185Hz & 330Hz)
  const shellFreqs = is808 ? [195, 345] : [180, 320];
  for (let i = 0; i < shellFreqs.length; i++) {
    const osc = ctx.createOscillator?.();
    const g = ctx.createGain?.();
    if (!osc || !g) continue;
    osc.type = 'triangle';
    osc.frequency?.setValueAtTime?.(shellFreqs[i] * 1.25, startAt);
    osc.frequency?.exponentialRampToValueAtTime?.(shellFreqs[i], startAt + 0.025);
    const modeAmp = i === 0 ? amp * 0.55 : amp * 0.32;
    g.gain?.setValueAtTime?.(modeAmp, startAt);
    g.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + 0.095);
    osc.connect?.(g);
    g.connect?.(destination);
    osc.start?.(startAt);
    osc.stop?.(startAt + 0.1);
  }

  // 2. Snare wire rattle (high-passed white noise burst)
  if (noiseBuffer && typeof ctx.createBufferSource === 'function') {
    try {
      const src = ctx.createBufferSource();
      src.buffer = noiseBuffer;
      const nGain = ctx.createGain?.();
      if (nGain) {
        const wireDecay = isJazz ? 0.24 : is808 ? 0.17 : 0.2;
        nGain.gain?.setValueAtTime?.(amp * 0.85 * velNorm, startAt);
        nGain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + wireDecay);
        if (typeof ctx.createBiquadFilter === 'function') {
          const hp = ctx.createBiquadFilter();
          hp.type = 'highpass';
          hp.frequency?.setValueAtTime?.(isJazz ? 950 : 1350, startAt);
          src.connect?.(hp);
          hp.connect?.(nGain);
        } else {
          src.connect?.(nGain);
        }
        nGain.connect?.(destination);
        src.start?.(startAt, 0.05, wireDecay + 0.02);
        src.stop?.(startAt + wireDecay + 0.02);
      }
    } catch {
      // Ignore in test mocks
    }
  }
}

function triggerRimshot(
  ctx: AudioContext,
  destination: AudioNode,
  noiseBuffer: AudioBuffer | null,
  startAt: number,
  amp: number,
  velNorm: number,
): void {
  const osc = ctx.createOscillator?.();
  const g = ctx.createGain?.();
  if (osc && g) {
    osc.type = 'triangle';
    osc.frequency?.setValueAtTime?.(460, startAt);
    osc.frequency?.exponentialRampToValueAtTime?.(380, startAt + 0.03);
    g.gain?.setValueAtTime?.(amp * 0.75 * velNorm, startAt);
    g.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + 0.045);
    osc.connect?.(g);
    g.connect?.(destination);
    osc.start?.(startAt);
    osc.stop?.(startAt + 0.05);
  }
  if (noiseBuffer && typeof ctx.createBufferSource === 'function') {
    try {
      const src = ctx.createBufferSource();
      src.buffer = noiseBuffer;
      const ng = ctx.createGain?.();
      if (ng) {
        ng.gain?.setValueAtTime?.(amp * 0.5, startAt);
        ng.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + 0.03);
        src.connect?.(ng);
        ng.connect?.(destination);
        src.start?.(startAt, 0.2, 0.035);
        src.stop?.(startAt + 0.035);
      }
    } catch {
      // Ignore
    }
  }
}

function triggerClap(
  ctx: AudioContext,
  destination: AudioNode,
  noiseBuffer: AudioBuffer | null,
  startAt: number,
  amp: number,
  velNorm: number,
): void {
  if (!noiseBuffer || typeof ctx.createBufferSource !== 'function') {
    triggerRimshot(ctx, destination, noiseBuffer, startAt, amp, velNorm);
    return;
  }
  // 4 micro-staggered handclap bursts (0ms, 10ms, 21ms, 32ms reverb tail)
  const offsets = [0, 0.01, 0.021, 0.032];
  for (let i = 0; i < offsets.length; i++) {
    try {
      const t = startAt + offsets[i];
      const dur = i === offsets.length - 1 ? 0.16 : 0.018;
      const src = ctx.createBufferSource();
      src.buffer = noiseBuffer;
      const g = ctx.createGain?.();
      if (!g) continue;
      g.gain?.setValueAtTime?.(amp * 0.7 * velNorm, t);
      g.gain?.exponentialRampToValueAtTime?.(0.0001, t + dur);
      if (typeof ctx.createBiquadFilter === 'function') {
        const bp = ctx.createBiquadFilter();
        bp.type = 'bandpass';
        bp.frequency?.setValueAtTime?.(1200, t);
        bp.Q?.setValueAtTime?.(1.6, t);
        src.connect?.(bp);
        bp.connect?.(g);
      } else {
        src.connect?.(g);
      }
      g.connect?.(destination);
      src.start?.(t, 0.3 + i * 0.04, dur + 0.01);
      src.stop?.(t + dur + 0.01);
    } catch {
      // Ignore
    }
  }
}

function triggerHiHat(
  ctx: AudioContext,
  destination: AudioNode,
  noiseBuffer: AudioBuffer | null,
  startAt: number,
  amp: number,
  velNorm: number,
  isOpen: boolean,
  isJazz: boolean,
): void {
  const decaySec = isOpen ? (isJazz ? 0.38 : 0.3) : isJazz ? 0.065 : 0.05;
  const hatGain = ctx.createGain?.();
  if (!hatGain) return;

  hatGain.gain?.setValueAtTime?.(amp * (isOpen ? 0.48 : 0.42) * velNorm, startAt);
  hatGain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + decaySec);

  let filterTarget: AudioNode = hatGain;
  if (typeof ctx.createBiquadFilter === 'function') {
    try {
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency?.setValueAtTime?.(isJazz ? 6200 : 7400, startAt);
      hp.connect?.(hatGain);
      filterTarget = hp;
    } catch {
      filterTarget = hatGain;
    }
  }
  hatGain.connect?.(destination);

  // Sizzle noise layer
  if (noiseBuffer && typeof ctx.createBufferSource === 'function') {
    try {
      const src = ctx.createBufferSource();
      src.buffer = noiseBuffer;
      src.connect?.(filterTarget);
      src.start?.(startAt, 0.15, decaySec + 0.01);
      src.stop?.(startAt + decaySec + 0.01);
    } catch {
      // Ignore
    }
  }

  // Metallic inharmonic square partials
  for (let i = 0; i < 3; i++) {
    const osc = ctx.createOscillator?.();
    const og = ctx.createGain?.();
    if (!osc || !og) continue;
    osc.type = 'square';
    osc.frequency?.setValueAtTime?.(METALLIC_FREQS[i * 2] * 8, startAt);
    og.gain?.setValueAtTime?.(0.14, startAt);
    osc.connect?.(og);
    og.connect?.(filterTarget);
    osc.start?.(startAt);
    osc.stop?.(startAt + decaySec + 0.01);
  }
}

function triggerCymbal(
  ctx: AudioContext,
  destination: AudioNode,
  noiseBuffer: AudioBuffer | null,
  startAt: number,
  amp: number,
  velNorm: number,
  kind: 'crash' | 'ride',
): void {
  const decaySec = kind === 'crash' ? 1.15 : 0.65;
  const cymGain = ctx.createGain?.();
  if (!cymGain) return;

  cymGain.gain?.setValueAtTime?.(amp * (kind === 'crash' ? 0.55 : 0.45) * velNorm, startAt);
  cymGain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + decaySec);

  let target: AudioNode = cymGain;
  if (typeof ctx.createBiquadFilter === 'function') {
    try {
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency?.setValueAtTime?.(kind === 'ride' ? 4800 : 5600, startAt);
      hp.connect?.(cymGain);
      target = hp;
    } catch {
      target = cymGain;
    }
  }
  cymGain.connect?.(destination);

  if (noiseBuffer && typeof ctx.createBufferSource === 'function') {
    try {
      const src = ctx.createBufferSource();
      src.buffer = noiseBuffer;
      src.connect?.(target);
      src.start?.(startAt, 0.4, decaySec + 0.02);
      src.stop?.(startAt + decaySec + 0.02);
    } catch {
      // Ignore
    }
  }

  // Ride Cymbal Bell Ping (essential for Jazz Swing ride patterns)
  const bellOsc = ctx.createOscillator?.();
  const bellGain = ctx.createGain?.();
  if (bellOsc && bellGain) {
    bellOsc.type = 'triangle';
    bellOsc.frequency?.setValueAtTime?.(kind === 'ride' ? 2780 : 1940, startAt);
    bellGain.gain?.setValueAtTime?.(amp * (kind === 'ride' ? 0.28 : 0.16) * velNorm, startAt);
    bellGain.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + (kind === 'ride' ? 0.26 : 0.35));
    bellOsc.connect?.(bellGain);
    bellGain.connect?.(destination);
    bellOsc.start?.(startAt);
    bellOsc.stop?.(startAt + 0.36);
  }
}

function triggerTom(
  ctx: AudioContext,
  destination: AudioNode,
  noiseBuffer: AudioBuffer | null,
  startAt: number,
  amp: number,
  velNorm: number,
  baseHz: number,
  isOrch: boolean,
): void {
  const osc = ctx.createOscillator?.();
  const g = ctx.createGain?.();
  if (!osc || !g) return;
  const decaySec = isOrch ? 0.55 : 0.28;
  osc.type = 'sine';
  osc.frequency?.setValueAtTime?.(baseHz * 1.45, startAt);
  osc.frequency?.exponentialRampToValueAtTime?.(baseHz, startAt + 0.055);
  g.gain?.setValueAtTime?.(amp * 0.9 * velNorm, startAt);
  g.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + decaySec);
  osc.connect?.(g);
  g.connect?.(destination);
  osc.start?.(startAt);
  osc.stop?.(startAt + decaySec + 0.01);

  if (noiseBuffer && typeof ctx.createBufferSource === 'function') {
    try {
      const stick = ctx.createBufferSource();
      stick.buffer = noiseBuffer;
      const sg = ctx.createGain?.();
      if (sg) {
        sg.gain?.setValueAtTime?.(amp * 0.25 * velNorm, startAt);
        sg.gain?.exponentialRampToValueAtTime?.(0.0001, startAt + 0.02);
        stick.connect?.(sg);
        sg.connect?.(destination);
        stick.start?.(startAt, 0, 0.025);
        stick.stop?.(startAt + 0.025);
      }
    } catch {
      // Ignore
    }
  }
}
