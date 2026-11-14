/**
 * Studio Mixing Console, Algorithmic Convolution Reverb, Stereo Channel Strips, and Master Bus.
 *
 * Provides:
 * - Procedural Stereo Impulse Response (IR) generation for 4 acoustic spaces:
 *   `concert_hall` (Classical), `jazz_club` (Jazz), `studio_room` (Pop), `tight_booth` (Hip-Hop).
 * - Master Bus `DynamicsCompressorNode` + Console Warmth EQ to glue multi-track ensembles
 *   without digital clipping.
 * - Per-instrument Channel Strip with `StereoPannerNode` and configurable Dry/Reverb send routing.
 * - Session-scoped master bus so `stop()` fades out in 15ms and silences all scheduled notes.
 * - Full fallback safety when running under lightweight mocked `AudioContext` in unit tests.
 */

import type { AcousticSpaceProfile, StagePosition3D } from './types';

export interface SpaceDescriptor {
  id: AcousticSpaceProfile;
  label: string;
  durationSec: number;
  decayRate: number;
  dampingCutoffHz: number;
  wetReturnGain: number;
  earlyReflections: { delayMs: number; gainL: number; gainR: number }[];
}

export const ACOUSTIC_SPACES: Record<AcousticSpaceProfile, SpaceDescriptor> = {
  concert_hall: {
    id: 'concert_hall',
    label: 'Concert Hall',
    durationSec: 2.35,
    decayRate: 2.65,
    dampingCutoffHz: 3900,
    wetReturnGain: 0.45,
    earlyReflections: [
      { delayMs: 14, gainL: 0.38, gainR: 0.16 },
      { delayMs: 23, gainL: 0.17, gainR: 0.37 },
      { delayMs: 34, gainL: 0.31, gainR: 0.24 },
      { delayMs: 49, gainL: 0.22, gainR: 0.32 },
      { delayMs: 68, gainL: 0.25, gainR: 0.21 },
      { delayMs: 89, gainL: 0.18, gainR: 0.24 },
    ],
  },
  symphony_surround: {
    id: 'symphony_surround',
    label: 'Symphony 3D Surround',
    durationSec: 2.75,
    decayRate: 2.35,
    dampingCutoffHz: 4200,
    wetReturnGain: 0.52,
    earlyReflections: [
      { delayMs: 12, gainL: 0.42, gainR: 0.14 },
      { delayMs: 19, gainL: 0.15, gainR: 0.41 },
      { delayMs: 29, gainL: 0.34, gainR: 0.22 },
      { delayMs: 41, gainL: 0.21, gainR: 0.35 },
      { delayMs: 56, gainL: 0.29, gainR: 0.27 },
      { delayMs: 74, gainL: 0.24, gainR: 0.30 },
      { delayMs: 96, gainL: 0.20, gainR: 0.22 },
    ],
  },
  jazz_club: {
    id: 'jazz_club',
    label: 'Jazz Club',
    durationSec: 1.25,
    decayRate: 3.9,
    dampingCutoffHz: 3200,
    wetReturnGain: 0.34,
    earlyReflections: [
      { delayMs: 11, gainL: 0.4, gainR: 0.22 },
      { delayMs: 23, gainL: 0.22, gainR: 0.38 },
      { delayMs: 37, gainL: 0.24, gainR: 0.24 },
    ],
  },
  studio_room: {
    id: 'studio_room',
    label: 'Studio Room',
    durationSec: 0.9,
    decayRate: 4.8,
    dampingCutoffHz: 4600,
    wetReturnGain: 0.28,
    earlyReflections: [
      { delayMs: 9, gainL: 0.35, gainR: 0.2 },
      { delayMs: 18, gainL: 0.2, gainR: 0.35 },
    ],
  },
  tight_booth: {
    id: 'tight_booth',
    label: 'Tight Booth',
    durationSec: 0.45,
    decayRate: 7.2,
    dampingCutoffHz: 2900,
    wetReturnGain: 0.18,
    earlyReflections: [{ delayMs: 7, gainL: 0.3, gainR: 0.28 }],
  },
};

/**
 * Create a reusable 2-second white/pink-blended noise buffer for percussion, breath, and hammer transients.
 */
export function createSharedNoiseBuffer(ctx: AudioContext): AudioBuffer | null {
  if (typeof ctx.createBuffer !== 'function') return null;
  try {
    const sampleRate = ctx.sampleRate || 44100;
    const length = Math.floor(sampleRate * 2.0);
    const buffer = ctx.createBuffer(1, length, sampleRate);
    const data = buffer.getChannelData(0);
    // Deterministic PRNG so audio tests & renders are reproducible
    let seed = 0x5f3759df;
    let b0 = 0;
    let b1 = 0;
    let b2 = 0;
    for (let i = 0; i < length; i++) {
      seed = (seed * 1664525 + 1013904223) | 0;
      const white = (seed / 0x7fffffff);
      // Light pinking filter + white blend for natural acoustic transients
      b0 = 0.99765 * b0 + white * 0.099046;
      b1 = 0.963 * b1 + white * 0.2965164;
      b2 = 0.57 * b2 + white * 1.0526913;
      const pink = (b0 + b1 + b2 + white * 0.1848) * 0.14;
      data[i] = white * 0.65 + pink * 0.35;
    }
    return buffer;
  } catch {
    return null;
  }
}

/**
 * Synthesize a stereo impulse response buffer for a given `AcousticSpaceProfile`.
 */
export function buildImpulseResponseBuffer(
  ctx: AudioContext,
  space: AcousticSpaceProfile,
): AudioBuffer | null {
  if (typeof ctx.createBuffer !== 'function') return null;
  try {
    const desc = ACOUSTIC_SPACES[space] ?? ACOUSTIC_SPACES.studio_room;
    const sampleRate = ctx.sampleRate || 44100;
    const length = Math.max(1, Math.floor(sampleRate * desc.durationSec));
    const buffer = ctx.createBuffer(2, length, sampleRate);
    const left = buffer.getChannelData(0);
    const right = buffer.getChannelData(1);

    // 1. Early reflections
    for (const er of desc.earlyReflections) {
      const idx = Math.floor((er.delayMs / 1000) * sampleRate);
      if (idx >= 0 && idx < length) {
        left[idx] += er.gainL;
        right[idx] += er.gainR;
      }
    }

    // 2. Diffuse stereo reverb tail with progressive high-frequency air absorption
    let seedL = 13374201;
    let seedR = 98765431;
    let lpL = 0;
    let lpR = 0;
    for (let i = 0; i < length; i++) {
      const t = i / sampleRate;
      const preDelayEnv = Math.min(1.0, i / (sampleRate * 0.012));
      const decayEnv = Math.exp(-desc.decayRate * (t / desc.durationSec));
      const env = preDelayEnv * decayEnv * 0.28;

      seedL = (seedL * 1664525 + 1013904223) | 0;
      seedR = (seedR * 22695477 + 1) | 0;
      const nL = seedL / 0x7fffffff;
      const nR = seedR / 0x7fffffff;

      // Progressive air damping (tail gets warmer over time)
      const dampAlpha = Math.max(0.12, 0.55 - 0.38 * (i / length));
      lpL = lpL + dampAlpha * (nL - lpL);
      lpR = lpR + dampAlpha * (nR - lpR);

      left[i] += lpL * env;
      right[i] += lpR * env;
    }

    return buffer;
  } catch {
    return null;
  }
}

export interface PlaybackSessionBus {
  readonly dryInput: AudioNode;
  readonly reverbInput: AudioNode | null;
  createInstrumentStrip(
    pan: number,
    reverbSend: number,
    stagePos?: StagePosition3D,
  ): AudioNode;
  stopAndDisconnect(now: number): void;
}

/**
 * Create a session-scoped mixing console bus connected to `ctx.destination`.
 * When `stopAndDisconnect` is called, all scheduled voices routed through this session
 * are smoothly ramped down in 15ms and disconnected.
 */
export function createPlaybackSessionBus(
  ctx: AudioContext,
  space: AcousticSpaceProfile,
): PlaybackSessionBus {
  const sessionMaster = ctx.createGain?.();
  if (!sessionMaster) {
    // Fallback for bare test mocks without GainNode
    return {
      dryInput: ctx.destination,
      reverbInput: null,
      createInstrumentStrip: () => ctx.destination,
      stopAndDisconnect: () => {},
    };
  }

  const now = ctx.currentTime ?? 0;
  sessionMaster.gain?.setValueAtTime?.(0.92, now);

  // Master bus dynamics compressor + warmth filter (if supported by AudioContext)
  let finalNode: AudioNode = sessionMaster;
  try {
    if (typeof ctx.createBiquadFilter === 'function') {
      const warmthFilter = ctx.createBiquadFilter();
      warmthFilter.type = 'lowpass';
      warmthFilter.frequency?.setValueAtTime?.(14500, now);
      warmthFilter.Q?.setValueAtTime?.(0.5, now);
      finalNode.connect?.(warmthFilter);
      finalNode = warmthFilter;
    }
    if (typeof ctx.createDynamicsCompressor === 'function') {
      const comp = ctx.createDynamicsCompressor();
      comp.threshold?.setValueAtTime?.(-14, now);
      comp.knee?.setValueAtTime?.(12, now);
      comp.ratio?.setValueAtTime?.(3.2, now);
      comp.attack?.setValueAtTime?.(0.004, now);
      comp.release?.setValueAtTime?.(0.18, now);
      finalNode.connect?.(comp);
      finalNode = comp;
    }
  } catch {
    // Safe fallback if optional nodes throw in test mocks
  }

  finalNode.connect?.(ctx.destination);

  // Convolution reverb bus
  let reverbBus: GainNode | null = null;
  try {
    if (typeof ctx.createConvolver === 'function' && typeof ctx.createGain === 'function') {
      const ir = buildImpulseResponseBuffer(ctx, space);
      if (ir) {
        const convolver = ctx.createConvolver();
        convolver.buffer = ir;
        reverbBus = ctx.createGain();
        const wetGain = ctx.createGain();
        const spaceDesc = ACOUSTIC_SPACES[space] ?? ACOUSTIC_SPACES.studio_room;
        wetGain.gain?.setValueAtTime?.(spaceDesc.wetReturnGain, now);

        reverbBus.connect?.(convolver);
        convolver.connect?.(wetGain);
        wetGain.connect?.(sessionMaster);
      }
    }
  } catch {
    reverbBus = null;
  }

  const stripCache = new Map<string, AudioNode>();

  return {
    dryInput: sessionMaster,
    reverbInput: reverbBus,
    createInstrumentStrip(
      pan: number,
      reverbSend: number,
      stagePos?: StagePosition3D,
    ): AudioNode {
      const effectivePan = Math.max(-1, Math.min(1, pan));
      const depthZ = stagePos?.depthZ ?? -2.4;
      const elevationY = stagePos?.elevationY ?? 0.1;
      const key = `${effectivePan.toFixed(2)}:${reverbSend.toFixed(2)}:${depthZ.toFixed(1)}:${elevationY.toFixed(1)}`;
      const cached = stripCache.get(key);
      if (cached) return cached;

      const stripInput = ctx.createGain?.();
      if (!stripInput) return sessionMaster;

      let stageChain: AudioNode = stripInput;

      // 1. Acoustic Stage Depth Propagation Delay & Distance Air Absorption
      // Sound travels at ~343 m/s: rear orchestral risers (-5.8m) arrive ~11ms after front soloists (-1.8m).
      const distanceMeters = Math.max(1.5, Math.abs(depthZ));
      try {
        if (typeof ctx.createDelay === 'function' && distanceMeters > 2.1) {
          const depthDelaySec = Math.min(0.022, (distanceMeters - 1.8) / 343.0);
          if (depthDelaySec > 0.001) {
            const delayNode = ctx.createDelay(0.05);
            delayNode.delayTime?.setValueAtTime?.(depthDelaySec, now);
            stageChain.connect?.(delayNode);
            stageChain = delayNode;
          }
        }
        if (typeof ctx.createBiquadFilter === 'function' && distanceMeters > 3.2) {
          const airFilter = ctx.createBiquadFilter();
          airFilter.type = 'lowpass';
          const airCutoff = Math.max(11500, 18500 - (distanceMeters - 2.0) * 1100);
          airFilter.frequency?.setValueAtTime?.(airCutoff, now);
          stageChain.connect?.(airFilter);
          stageChain = airFilter;
        }
      } catch {
        // Fallback if DelayNode or BiquadFilter unsupported in test mock
      }

      // 2. Stereo Console Panning + 3D Binaural HRTF Surround Stage Positioning
      let pannedOut: AudioNode = stageChain;
      try {
        const use3DHrtf =
          typeof ctx.createPanner === 'function' &&
          (space === 'symphony_surround' || space === 'concert_hall' || stagePos !== undefined);

        if (use3DHrtf) {
          const panner3d = ctx.createPanner();
          panner3d.panningModel = 'HRTF';
          panner3d.distanceModel = 'inverse';
          panner3d.refDistance = 2.2;
          panner3d.maxDistance = 25.0;
          panner3d.rolloffFactor = space === 'symphony_surround' ? 0.42 : 0.28;

          const xMeters = effectivePan * (space === 'symphony_surround' ? 4.4 : 3.5);
          if (panner3d.positionX && panner3d.positionY && panner3d.positionZ) {
            panner3d.positionX.setValueAtTime?.(xMeters, now);
            panner3d.positionY.setValueAtTime?.(elevationY, now);
            panner3d.positionZ.setValueAtTime?.(depthZ, now);
          } else if (typeof panner3d.setPosition === 'function') {
            panner3d.setPosition(xMeters, elevationY, depthZ);
          }

          // Blend 3D HRTF image with direct stereo foundation so bass & center punch stay solid
          const hrtfGain = ctx.createGain?.();
          const directStereoGain = ctx.createGain?.();
          const mergeBus = ctx.createGain?.();
          if (hrtfGain && directStereoGain && mergeBus) {
            const hrtfMix = space === 'symphony_surround' ? 0.72 : 0.48;
            hrtfGain.gain?.setValueAtTime?.(hrtfMix, now);
            directStereoGain.gain?.setValueAtTime?.(1.0 - hrtfMix * 0.45, now);

            stageChain.connect?.(panner3d);
            panner3d.connect?.(hrtfGain);
            hrtfGain.connect?.(mergeBus);

            if (typeof ctx.createStereoPanner === 'function' && Math.abs(effectivePan) > 0.01) {
              const stereoPanner = ctx.createStereoPanner();
              stereoPanner.pan?.setValueAtTime?.(effectivePan, now);
              stageChain.connect?.(stereoPanner);
              stereoPanner.connect?.(directStereoGain);
            } else {
              stageChain.connect?.(directStereoGain);
            }
            directStereoGain.connect?.(mergeBus);
            pannedOut = mergeBus;
          } else {
            stageChain.connect?.(panner3d);
            pannedOut = panner3d;
          }
        } else if (typeof ctx.createStereoPanner === 'function' && Math.abs(effectivePan) > 0.01) {
          const panner = ctx.createStereoPanner();
          panner.pan?.setValueAtTime?.(effectivePan, now);
          stageChain.connect?.(panner);
          pannedOut = panner;
        }
      } catch {
        pannedOut = stageChain;
      }

      pannedOut.connect?.(sessionMaster);

      // 3. Depth-Aware Hall Reverb Send (rear orchestral sections excite more hall reflections)
      const depthReverbScale = 1.0 + Math.max(0, (distanceMeters - 2.0) * 0.11);
      const effectiveReverbSend = Math.min(0.82, reverbSend * depthReverbScale);

      if (reverbBus && effectiveReverbSend > 0.01) {
        try {
          const sendGain = ctx.createGain?.();
          if (sendGain) {
            sendGain.gain?.setValueAtTime?.(effectiveReverbSend, now);
            pannedOut.connect?.(sendGain);
            sendGain.connect?.(reverbBus);
          }
        } catch {
          // Ignore send routing error in mock environments
        }
      }

      stripCache.set(key, stripInput);
      return stripInput;
    },
    stopAndDisconnect(stopTime: number): void {
      try {
        const t = Math.max(ctx.currentTime ?? 0, stopTime);
        sessionMaster.gain?.cancelScheduledValues?.(t);
        sessionMaster.gain?.setValueAtTime?.(sessionMaster.gain?.value ?? 0.9, t);
        sessionMaster.gain?.exponentialRampToValueAtTime?.(0.0001, t + 0.018);
        setTimeout(() => {
          try {
            finalNode.disconnect?.();
            sessionMaster.disconnect?.();
          } catch {
            // Ignore disconnect errors
          }
        }, 35);
      } catch {
        try {
          sessionMaster.disconnect?.();
        } catch {
          // Ignore
        }
      }
    },
  };
}
