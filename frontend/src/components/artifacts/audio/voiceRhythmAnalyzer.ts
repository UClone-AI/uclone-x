/**
 * Lightweight browser-side Voice & Tap Rhythm Analyzer for `MusicView`.
 *
 * Extracts onset beats, note sustain durations, velocity accents, frequency bands
 * (`low` / `mid` / `high`), and estimated BPM from a mono PCM `Float32Array` buffer
 * (or interactive tap intervals) and quantizes them into Bar-Grid Step DSL skeletons
 * that the Composer LLM can directly map onto target score bars.
 */

import type { MusicRegionSelection } from '../MusicAudioEngine';

export type VoiceBand = 'low' | 'mid' | 'high';

export interface ExtractedVoiceHit {
  onsetSec: number;
  durationSec: number;
  velocity: number;
  band: VoiceBand;
}

export interface VoiceRhythmSketch {
  detectedBpm: number;
  barsCount: number;
  hitCount: number;
  /** One quantized rhythm skeleton string per bar (`B1`, `B2`, ...). */
  barPatterns: string[];
}

export interface AnalyzeVoiceOptions {
  bpm?: number;
  maxBars?: number;
  beatsPerBar?: number;
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function quantizeStep(value: number, step = 0.5): number {
  return Math.round(value / step) * step;
}

function formatBeatNum(n: number): string {
  const rounded = Math.round(n * 100) / 100;
  return Number.isInteger(rounded) ? `${rounded}.0` : `${rounded}`;
}

/**
 * Extract vocal/percussive hits (`onsetSec`, `durationSec`, `velocity`, `band`) from mono PCM samples.
 */
export function extractHitsFromPcm(
  samples: Float32Array,
  sampleRate: number,
): ExtractedVoiceHit[] {
  if (samples.length === 0 || sampleRate <= 0) return [];

  const frameSize = Math.max(64, Math.floor(sampleRate * 0.02)); // 20ms window
  const numFrames = Math.floor(samples.length / frameSize);
  if (numFrames === 0) return [];

  const rmsList = new Float32Array(numFrames);
  const zcrList = new Float32Array(numFrames);
  let maxRms = 0;

  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSize;
    const end = start + frameSize;
    let sumSq = 0;
    let zeroCrossings = 0;
    for (let i = start; i < end; i++) {
      const s = samples[i];
      sumSq += s * s;
      if (i > start && (samples[i - 1] >= 0) !== (s >= 0)) {
        zeroCrossings++;
      }
    }
    const rms = Math.sqrt(sumSq / frameSize);
    rmsList[f] = rms;
    zcrList[f] = zeroCrossings / frameSize;
    if (rms > maxRms) maxRms = rms;
  }

  if (maxRms < 0.008) return [];

  const onsetThreshold = Math.max(0.015, maxRms * 0.25);
  const gateThreshold = onsetThreshold * 0.55;
  const minInterOnsetFrames = Math.max(2, Math.round(0.08 / 0.02)); // >= 80ms between hits

  const hits: ExtractedVoiceHit[] = [];
  let f = 0;
  let lastOnsetFrame = -minInterOnsetFrames;

  while (f < numFrames) {
    const prevRms = f > 0 ? rmsList[f - 1] : 0;
    const currRms = rmsList[f];
    if (
      currRms >= onsetThreshold &&
      currRms > prevRms * 1.25 &&
      f - lastOnsetFrame >= minInterOnsetFrames
    ) {
      const onsetFrame = f;
      lastOnsetFrame = onsetFrame;
      let peakRms = currRms;
      let zcrSum = zcrList[f];
      let zcrCount = 1;
      let endFrame = f + 1;

      while (endFrame < numFrames && rmsList[endFrame] >= gateThreshold) {
        if (rmsList[endFrame] > peakRms) peakRms = rmsList[endFrame];
        if (endFrame - onsetFrame < 4) {
          zcrSum += zcrList[endFrame];
          zcrCount++;
        }
        endFrame++;
      }

      const avgZcr = zcrSum / zcrCount;
      const band: VoiceBand =
        avgZcr < 0.055 ? 'low' : avgZcr >= 0.16 ? 'high' : 'mid';
      const normAmp = clamp(peakRms / maxRms, 0.2, 1.0);
      const velocity = Math.round(64 + normAmp * 46);
      const onsetSec = (onsetFrame * frameSize) / sampleRate;
      const durationSec = Math.max(0.08, ((endFrame - onsetFrame) * frameSize) / sampleRate);

      hits.push({ onsetSec, durationSec, velocity, band });
      f = endFrame;
    } else {
      f++;
    }
  }

  return hits;
}

/**
 * Estimate BPM from inter-onset intervals when at least 3 hits exist; otherwise return `fallbackBpm`.
 */
export function estimateBpmFromHits(hits: readonly ExtractedVoiceHit[], fallbackBpm = 96): number {
  if (hits.length < 3) return clamp(Math.round(fallbackBpm), 60, 200);
  const iois: number[] = [];
  for (let i = 1; i < hits.length; i++) {
    const dt = hits[i].onsetSec - hits[i - 1].onsetSec;
    if (dt >= 0.18 && dt <= 1.5) {
      iois.push(dt);
    }
  }
  if (iois.length === 0) return clamp(Math.round(fallbackBpm), 60, 200);
  iois.sort((a, b) => a - b);
  const medianIoi = iois[Math.floor(iois.length / 2)];
  // Normalize sub-beat IOIs (e.g. eighth notes) into quarter-note BPM range 68..165
  let candidateBpm = 60 / medianIoi;
  while (candidateBpm > 165) candidateBpm /= 2;
  while (candidateBpm < 68) candidateBpm *= 2;
  return clamp(Math.round(candidateBpm), 60, 200);
}

/**
 * Quantize extracted voice/tap hits onto a 4/4 bar grid and produce Bar-Grid Step DSL skeletons.
 */
export function quantizeHitsToSketch(
  hits: readonly ExtractedVoiceHit[],
  options: AnalyzeVoiceOptions = {},
): VoiceRhythmSketch {
  const beatsPerBar = options.beatsPerBar ?? 4;
  const maxBars = clamp(options.maxBars ?? 2, 1, 8);
  const detectedBpm = options.bpm
    ? clamp(Math.round(options.bpm), 60, 200)
    : estimateBpmFromHits(hits, 96);

  if (hits.length === 0) {
    return {
      detectedBpm,
      barsCount: 1,
      hitCount: 0,
      barPatterns: [`1.0:REST:${formatBeatNum(beatsPerBar)}`],
    };
  }

  const secPerBeat = 60 / detectedBpm;
  const firstOnset = hits[0].onsetSec;

  interface QuantizedEvent {
    barIdx: number;
    beatOffset: number; // 0 .. beatsPerBar - 0.5
    durationBeats: number;
    velocity: number;
    band: VoiceBand;
  }

  const quantized: QuantizedEvent[] = [];
  for (const hit of hits) {
    const relBeats = Math.max(0, (hit.onsetSec - firstOnset) / secPerBeat);
    const snappedRel = quantizeStep(relBeats, 0.5);
    const barIdx = Math.floor(snappedRel / beatsPerBar);
    if (barIdx >= maxBars) continue;
    const beatOffset = Math.round((snappedRel - barIdx * beatsPerBar) * 100) / 100;
    const rawDurBeats = Math.max(0.5, quantizeStep(hit.durationSec / secPerBeat, 0.5));
    const maxDurInBar = beatsPerBar - beatOffset;
    const durationBeats = clamp(rawDurBeats, 0.5, maxDurInBar);

    const existingIdx = quantized.findIndex(
      (e) => e.barIdx === barIdx && Math.abs(e.beatOffset - beatOffset) < 0.05,
    );
    if (existingIdx >= 0) {
      if (hit.velocity > quantized[existingIdx].velocity) {
        quantized[existingIdx] = {
          barIdx,
          beatOffset,
          durationBeats,
          velocity: hit.velocity,
          band: hit.band,
        };
      }
    } else {
      quantized.push({
        barIdx,
        beatOffset,
        durationBeats,
        velocity: hit.velocity,
        band: hit.band,
      });
    }
  }

  const highestBarIdx = quantized.reduce((acc, e) => Math.max(acc, e.barIdx), 0);
  const barsCount = clamp(highestBarIdx + 1, 1, maxBars);
  const barPatterns: string[] = [];

  for (let b = 0; b < barsCount; b++) {
    const barEvents = quantized
      .filter((e) => e.barIdx === b)
      .sort((x, y) => x.beatOffset - y.beatOffset);

    if (barEvents.length === 0) {
      barPatterns.push(`1.0:REST:${formatBeatNum(beatsPerBar)}`);
      continue;
    }

    const tokens: string[] = [];
    let cursor = 0.0;

    for (let i = 0; i < barEvents.length; i++) {
      const ev = barEvents[i];
      const nextOffset = i + 1 < barEvents.length ? barEvents[i + 1].beatOffset : beatsPerBar;
      if (ev.beatOffset > cursor + 0.01) {
        const restDur = ev.beatOffset - cursor;
        tokens.push(`${formatBeatNum(1.0 + cursor)}:REST:${formatBeatNum(restDur)}`);
        cursor = ev.beatOffset;
      }
      const availDur = Math.max(0.5, nextOffset - cursor);
      const actualDur = Math.min(ev.durationBeats, availDur);
      const articulation =
        ev.velocity >= 98
          ? ':accent'
          : actualDur <= 0.5 && availDur >= 1.0
            ? ':staccato'
            : actualDur >= 1.5
              ? ':tenuto'
              : '';
      tokens.push(
        `${formatBeatNum(1.0 + cursor)}:${ev.band}:${formatBeatNum(actualDur)}:v${ev.velocity}${articulation}`,
      );
      cursor += actualDur;
    }

    if (cursor < beatsPerBar - 0.01) {
      tokens.push(`${formatBeatNum(1.0 + cursor)}:REST:${formatBeatNum(beatsPerBar - cursor)}`);
    }

    barPatterns.push(tokens.join(', '));
  }

  return {
    detectedBpm,
    barsCount,
    hitCount: hits.length,
    barPatterns,
  };
}

/**
 * Analyze a mono PCM `Float32Array` buffer directly into a `VoiceRhythmSketch`.
 */
export function analyzeVoiceRhythmBuffer(
  samples: Float32Array,
  sampleRate: number,
  options: AnalyzeVoiceOptions = {},
): VoiceRhythmSketch {
  const hits = extractHitsFromPcm(samples, sampleRate);
  return quantizeHitsToSketch(hits, options);
}

/**
 * Format a `VoiceRhythmSketch` and target `MusicRegionSelection` into the structured prompt tag
 * sent to the Composer LLM.
 */
export function formatVoiceRhythmPromptTag(
  sketch: VoiceRhythmSketch,
  selection: MusicRegionSelection,
): string {
  const instLabel = selection.instrument_ids.join(',') || 'violin_solo';
  const barsSpec = sketch.barPatterns
    .map((pat, idx) => `B${selection.start_bar + idx}="${pat}"`)
    .join(' | ');
  return `[${selection.section_id} | Bar ${selection.start_bar}-${selection.end_bar} | ${instLabel}] [VoiceRhythm | detected_bpm=${sketch.detectedBpm} | bars=${sketch.barsCount} | ${barsSpec}]`;
}
