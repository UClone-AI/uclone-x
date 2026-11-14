/**
 * Recorded grand piano for `grand_piano` patches.
 *
 * Samples: Salamander Grand Piano by Alexander Holm, licensed CC BY 3.0
 * (https://creativecommons.org/licenses/by/3.0/). `MusicView` shows the credit wherever a
 * song uses the grand piano. One recording every minor third is fetched on demand and
 * pitch-shifted to the nearest note.
 *
 * Until a recording has loaded — or when the host cannot be reached — `trigger` returns
 * `false` and the caller plays the synthesized piano voice instead. That is silent by design:
 * the user hears a piano either way, and no error is shown.
 */

import type { ResolvedInstrumentPatch, VoiceRenderContext } from './types';

export const SALAMANDER_PIANO_CREDIT =
  'Piano samples: Salamander Grand Piano by Alexander Holm (CC BY 3.0)';
export const SALAMANDER_PIANO_LICENSE_URL = 'https://creativecommons.org/licenses/by/3.0/';

const SALAMANDER_PIANO_BASE = 'https://tonejs.github.io/audio/salamander';

/** Recorded anchor pitches, every minor third from A0 to C8. */
const SALAMANDER_ANCHORS: { midi: number; file: string }[] = [
  { midi: 21, file: 'A0.mp3' },
  { midi: 24, file: 'C1.mp3' },
  { midi: 27, file: 'Ds1.mp3' },
  { midi: 30, file: 'Fs1.mp3' },
  { midi: 33, file: 'A1.mp3' },
  { midi: 36, file: 'C2.mp3' },
  { midi: 39, file: 'Ds2.mp3' },
  { midi: 42, file: 'Fs2.mp3' },
  { midi: 45, file: 'A2.mp3' },
  { midi: 48, file: 'C3.mp3' },
  { midi: 51, file: 'Ds3.mp3' },
  { midi: 54, file: 'Fs3.mp3' },
  { midi: 57, file: 'A3.mp3' },
  { midi: 60, file: 'C4.mp3' },
  { midi: 63, file: 'Ds4.mp3' },
  { midi: 66, file: 'Fs4.mp3' },
  { midi: 69, file: 'A4.mp3' },
  { midi: 72, file: 'C5.mp3' },
  { midi: 75, file: 'Ds5.mp3' },
  { midi: 78, file: 'Fs5.mp3' },
  { midi: 81, file: 'A5.mp3' },
  { midi: 84, file: 'C6.mp3' },
  { midi: 87, file: 'Ds6.mp3' },
  { midi: 90, file: 'Fs6.mp3' },
  { midi: 93, file: 'A6.mp3' },
  { midi: 96, file: 'C7.mp3' },
  { midi: 99, file: 'Ds7.mp3' },
  { midi: 102, file: 'Fs7.mp3' },
  { midi: 105, file: 'A7.mp3' },
  { midi: 108, file: 'C8.mp3' },
];

export function isSampledPianoPatch(patch: ResolvedInstrumentPatch): boolean {
  return patch.timbreFamily === 'grand_piano';
}

export function nearestSalamanderAnchor(targetMidi: number): { midi: number; file: string } {
  let best = SALAMANDER_ANCHORS[0];
  let bestDist = Math.abs(targetMidi - best.midi);
  for (const cand of SALAMANDER_ANCHORS) {
    const dist = Math.abs(targetMidi - cand.midi);
    if (dist < bestDist) {
      best = cand;
      bestDist = dist;
    }
  }
  return best;
}

export class SalamanderPiano {
  private decodedByAnchor = new Map<number, AudioBuffer>();
  private inflight = new Map<number, Promise<void>>();

  /** Fetch and decode the recordings nearest to `midis`. Never throws. */
  public async preload(ctx: AudioContext, midis: number[]): Promise<void> {
    if (typeof fetch !== 'function' || typeof ctx.decodeAudioData !== 'function') return;
    const anchors = new Map<number, string>();
    for (const m of midis) {
      if (m < 0) continue;
      const a = nearestSalamanderAnchor(m);
      anchors.set(a.midi, a.file);
    }
    await Promise.all(
      Array.from(anchors.entries()).map(([anchorMidi, file]) =>
        this.loadAnchor(ctx, anchorMidi, file),
      ),
    );
  }

  private loadAnchor(ctx: AudioContext, anchorMidi: number, file: string): Promise<void> {
    if (this.decodedByAnchor.has(anchorMidi)) return Promise.resolve();
    const pending = this.inflight.get(anchorMidi);
    if (pending) return pending;
    const promise = (async () => {
      try {
        const res = await fetch(`${SALAMANDER_PIANO_BASE}/${file}`);
        if (!res.ok) return;
        const decoded = await ctx.decodeAudioData(await res.arrayBuffer());
        if (decoded) this.decodedByAnchor.set(anchorMidi, decoded);
      } catch {
        // Unreachable host or undecodable file: the synthesized piano plays instead.
      } finally {
        this.inflight.delete(anchorMidi);
      }
    })();
    this.inflight.set(anchorMidi, promise);
    return promise;
  }

  /**
   * Play one piano note from the nearest loaded recording. Returns `false` when no recording
   * is loaded yet, so the caller can synthesize the note.
   */
  public trigger(rc: VoiceRenderContext): boolean {
    const { ctx, destination, midi, startAt, durationSeconds, velocity, volumeGain } = rc;
    const { articulationProfile, expression } = rc;
    if (!isSampledPianoPatch(rc.patch) || typeof ctx.createBufferSource !== 'function') {
      return false;
    }

    let chosenMidi: number | null = null;
    for (let offset = 0; offset <= 12 && chosenMidi === null; offset++) {
      if (this.decodedByAnchor.has(midi - offset)) chosenMidi = midi - offset;
      else if (this.decodedByAnchor.has(midi + offset)) chosenMidi = midi + offset;
    }
    const buffer = chosenMidi === null ? undefined : this.decodedByAnchor.get(chosenMidi);
    if (chosenMidi === null || !buffer) return false;

    try {
      const dynamicScale = expression?.dynamicScale ?? 1.0;
      const timbralBloom = expression?.timbralBloom ?? 1.0;
      const start = Math.max(ctx.currentTime ?? 0, startAt + (expression?.timingOffsetSec ?? 0));
      const velNorm = Math.max(0.15, Math.min(1.0, (velocity * Math.min(1.15, dynamicScale)) / 127));
      const gateDur = Math.max(0.06, durationSeconds * articulationProfile.gateRatio);
      const releaseSec = Math.max(0.08, 0.44 * articulationProfile.releaseScale);
      const end = start + gateDur + releaseSec;
      const peakGain = Math.max(
        0.012,
        volumeGain * dynamicScale * articulationProfile.velocityScale * 1.6,
      );
      const attackSec = Math.max(0.002, 0.004 * articulationProfile.attackScale);

      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.playbackRate?.setValueAtTime?.(Math.pow(2, (midi - chosenMidi) / 12), start);

      const env = ctx.createGain?.();
      if (!env) return false;
      env.gain?.setValueAtTime?.(0.0001, start);
      env.gain?.linearRampToValueAtTime?.(peakGain, start + attackSec);
      env.gain?.setValueAtTime?.(peakGain * 0.94, start + gateDur);
      env.gain?.exponentialRampToValueAtTime?.(0.0001, end);

      if (typeof ctx.createBiquadFilter === 'function') {
        const lp = ctx.createBiquadFilter();
        lp.type = 'lowpass';
        lp.frequency?.setValueAtTime?.(
          Math.min(18000, (1800 + velNorm * 13500) * articulationProfile.brightnessBoost * timbralBloom),
          start,
        );
        src.connect?.(lp);
        lp.connect?.(env);
      } else {
        src.connect?.(env);
      }
      env.connect?.(destination);
      src.start?.(start);
      src.stop?.(end + 0.03);
      return true;
    } catch {
      return false;
    }
  }
}
