/**
 * Tier-aware NLE Timeline & 3-Way Color Grading Compositor Engine (`VideoCompositorEngine`).
 *
 * Evaluates a `PlaybackBundle` (at `preview` 540p proxy, `source` raw/derived vault, or
 * `master` final tier) for any frame on the timeline, computing:
 * - Active video clips and transition crossfades
 * - Combined 3-Way Color Grading (Lift / Gamma / Gain, Contrast, Saturation, Temperature,
 *   Tint, Vignette, and LUT preset)
 * - Active animated title graphic overlays (`fade_up`, `typewriter`, `pop`, `slide_left`, `glitch_in`)
 * - Hybrid HTML5 `<video>` + Canvas2D Program Monitor rendering
 */

export type QualityTier = 'preview' | 'source' | 'master';

export type LutPreset =
  | 'clean_neutral'
  | 'rec709_standard'
  | 'cinematic_teal_orange'
  | 'teal_orange_cinema'
  | 'kodak_2383_warm'
  | 'warm_sunset'
  | 'vintage_kodak'
  | 'vintage_sepia'
  | 'bleach_bypass_gritty'
  | 'noir_bw'
  | 'noir_monochrome'
  | 'cool_cyberpunk'
  | 'cyberpunk_neon';

export type BakeOpKind =
  | 'stabilize'
  | 'optical_flow_slowmo'
  | 'reverse'
  | 'speed_ramp'
  | 'auto_reframe_9_16'
  | 'denoise_audio'
  | 'audio_denoise'
  | 'compound_bake';

export type TransitionKind =
  | 'none'
  | 'cross_dissolve'
  | 'dip_to_black'
  | 'dip_to_white'
  | 'wipe_left'
  | 'whip_pan';

export type TitleStylePreset =
  | 'minimal_cinema'
  | 'minimal_lower_third'
  | 'lower_third_news'
  | 'cinema_serif_center'
  | 'bold_pop'
  | 'bold_impact_headline'
  | 'neon_glitch'
  | 'neon_cyber_subtitle'
  | 'documentary_caption';

export type TitlePosition =
  | 'center'
  | 'lower_third_left'
  | 'lower_third_right'
  | 'top_banner'
  | 'lower_left'
  | 'lower_center'
  | 'top_left'
  | 'top_center';

export type TitleAnimation =
  | 'none'
  | 'fade_in'
  | 'fade_up'
  | 'typewriter'
  | 'glitch_in'
  | 'pop'
  | 'slide_left';

export interface ColorGrade {
  lift: [number, number, number];
  gamma: [number, number, number];
  gain: [number, number, number];
  contrast: number;
  saturation: number;
  temperature: number;
  tint: number;
  lut: LutPreset;
  vignette: number;
}

export interface ClipTransform {
  scale: number;
  position_x: number;
  position_y: number;
  rotation_deg: number;
  opacity?: number;
}

export interface ClipTransition {
  kind: TransitionKind;
  duration_frames: number;
}

export interface RawFallbackRef {
  asset_id: string;
  source_in_frame: number;
  source_out_frame: number;
}

export interface VideoClipView {
  clip_id: string;
  asset_id: string;
  timeline_in_frame: number;
  source_in_frame: number;
  source_out_frame: number;
  speed: number;
  volume: number;
  muted: boolean;
  transform: ClipTransform;
  color_grade: ColorGrade;
  transition_out?: ClipTransition | null;
  raw_fallback_ref?: RawFallbackRef | null;
  label?: string;
  note?: string;
}

export interface TitleClipView {
  title_id: string;
  text: string;
  subtitle?: string;
  style_preset: TitleStylePreset;
  timeline_in_frame: number;
  timeline_out_frame: number;
  position: TitlePosition;
  animation?: TitleAnimation;
  animation_in?: TitleAnimation;
  animation_out?: TitleAnimation;
  font_family?: string;
  font_size_px: number;
  color_hex: string;
  backdrop_opacity?: number;
  backdrop_blur?: number;
}

export interface TierStreamFile {
  path: string;
  sha256: string;
  width: number;
  height: number;
  size_bytes: number;
}

export interface BakeOperation {
  op: BakeOpKind;
  rate: number;
  strength: number;
}

export interface AssetLineage {
  parent_asset_ids: string[];
  source_in_frame: number;
  source_out_frame: number;
  recipe_kind: 'heavy_transform' | 'compound_bake' | 'audio_stem_clean';
  operations: BakeOperation[];
}

export interface MediaAssetView {
  asset_id: string;
  kind: 'raw' | 'derived';
  filename: string;
  immutable: boolean;
  fps: number;
  total_frames: number;
  duration_ms: number;
  tiers: {
    source: TierStreamFile;
    preview: TierStreamFile;
  };
  lineage?: AssetLineage | null;
  has_audio: boolean;
  index_status: 'pending' | 'indexed' | 'failed';
  indexed_frame_count: number;
}

export interface ShownAssetView {
  asset: MediaAssetView;
  digest: string;
  start_timecode: string;
  end_timecode: string;
}

export interface ShownTrackView {
  track_id: string;
  kind: string;
  name: string;
  index: number;
  digest: string;
  muted: boolean;
  solo: boolean;
  locked: boolean;
  clips: VideoClipView[];
  titles: TitleClipView[];
  can_revert: boolean;
}

export interface CompiledClip {
  clip_id: string;
  track_id: string;
  asset_id: string;
  asset_kind: string;
  resolved_tier: QualityTier;
  stream_path: string;
  stream_sha256: string;
  stream_width: number;
  stream_height: number;
  source_in_frame: number;
  source_out_frame: number;
  timeline_in_frame: number;
  timeline_out_frame: number;
  duration_frames: number;
  start_sec: number;
  end_sec: number;
  in_timecode: string;
  out_timecode: string;
  speed: number;
  volume: number;
  muted: boolean;
  transform: ClipTransform;
  color_grade: ColorGrade;
  transition_out?: ClipTransition | null;
  can_unbake: boolean;
  raw_parent_asset_id?: string | null;
}

export interface CompiledTitle {
  title_id: string;
  track_id: string;
  timeline_in_frame: number;
  timeline_out_frame: number;
  start_sec: number;
  end_sec: number;
  in_timecode: string;
  out_timecode: string;
  clip: TitleClipView;
}

export interface CompiledTrack {
  track_id: string;
  kind: 'video' | 'title' | 'fx' | 'audio';
  name: string;
  index: number;
  muted: boolean;
  solo: boolean;
  locked: boolean;
  clips: CompiledClip[];
  titles: CompiledTitle[];
}

export interface PlaybackBundle {
  project_id: string;
  title: string;
  fps: number;
  width: number;
  height: number;
  active_tier: QualityTier;
  total_frames: number;
  duration_sec: number;
  total_timecode: string;
  master_grade: ColorGrade;
  tracks: CompiledTrack[];
  ffmpeg_filtergraph: string;
}

export interface ShownProposalView {
  id: string;
  kind: 'track_replace' | 'manifest_replace';
  summary: string;
  track_id?: string | null;
  seen_digest: string;
  original_clips: VideoClipView[];
  proposed_clips: VideoClipView[];
  original_titles: TitleClipView[];
  proposed_titles: TitleClipView[];
  preview_bundle: PlaybackBundle;
}

export interface VideoAuditFinding {
  code: string;
  severity: 'error' | 'warning';
  message: string;
  frame?: number;
  track_id?: string | null;
  clip_id?: string | null;
  asset_id?: string | null;
}

export interface MasterRenderResult {
  project_id: string;
  tier: QualityTier;
  output_path: string;
  sha256: string;
  width: number;
  height: number;
  fps: number;
  total_frames: number;
  duration_sec: number;
  relinked_source_assets: string[];
  ffmpeg_filtergraph: string;
}

export type StyleProfileInput =
  | string
  | {
      genre: string;
      aspect_ratio: string;
      pacing: string;
      color_mood: LutPreset;
      audio_lufs_target: number;
    };

export interface ShownVideoProject {
  project_id: string;
  title: string;
  style_profile: StyleProfileInput;
  quality_tiers: {
    preview: {
      tier: string;
      width?: number;
      height?: number;
      codec?: string;
      crf?: number;
      max_keyframe_interval?: number;
      bitrate_kbps?: number;
      preserve_original_bitstream?: boolean;
      audio_lufs_target?: number;
      description?: string;
    };
    source: {
      tier: string;
      width?: number;
      height?: number;
      codec?: string;
      crf?: number;
      preserve_bitstream?: boolean;
      preserve_original_bitstream?: boolean;
      read_only_mode?: string;
      description?: string;
    };
    master: {
      tier: string;
      width?: number;
      height?: number;
      codec?: string;
      crf?: number;
      pixel_format?: string;
      audio_lufs_target?: number;
      description?: string;
    };
  };
  active_tier: QualityTier;
  writable: boolean;
  holder: string | null;
  decide_code: string | null;
  decide_note?: string | null;
  fps: number;
  width: number;
  height: number;
  total_frames: number;
  duration_seconds: number;
  total_timecode: string;
  sections: Array<{
    id: string;
    title: string;
    kind: string;
    start_frame: number;
    end_frame: number;
  }>;
  markers: Array<{
    marker_id: string;
    frame: number;
    label: string;
    color: string;
  }>;
  assets: ShownAssetView[];
  tracks: ShownTrackView[];
  playback: PlaybackBundle;
  proposals: ShownProposalView[];
  audit_findings: VideoAuditFinding[];
  last_master_receipt?: MasterRenderResult | null;
}

export interface VideoProjectSummary {
  id: string;
  title: string;
  style_profile: string;
  fps: number;
  active_playback_tier: QualityTier;
}

export interface FrameRangeHit {
  frame_id: string;
  asset_id: string;
  shot_id: string;
  peak_frame: number;
  in_frame: number;
  out_frame: number;
  in_timecode: string;
  out_timecode: string;
  duration_frames: number;
  shot_type: string;
  camera_motion: string;
  score: number;
  caption: string;
  visual_tags: string[];
  transcript_slice: string;
}

export interface FrameSearchRanking {
  query: string;
  method: string;
  total_indexed_frames: number;
  hits: FrameRangeHit[];
}

export interface VideoRegionSelection {
  track_id: string;
  start_frame: number;
  end_frame: number;
  clip_id?: string | null;
}

export interface ActiveTitleRenderState {
  title: CompiledTitle;
  progress: number;
  opacity: number;
  translateY: number;
  visibleText: string;
}

export interface FrameCompositionState {
  frame: number;
  timecode: string;
  activeTier: QualityTier;
  resolutionLabel: string;
  primaryClip: CompiledClip | null;
  secondaryClip: CompiledClip | null;
  sourceFrame: number;
  transitionKind: TransitionKind;
  transitionProgress: number;
  effectiveGrade: ColorGrade;
  cssFilter: string;
  lutOverlay: { color: string; opacity: number; label: string };
  activeTitles: ActiveTitleRenderState[];
}

export const LUT_PRESETS: ReadonlyArray<{ id: LutPreset; label: string; badgeColor: string }> = [
  { id: 'clean_neutral', label: 'Clean Neutral', badgeColor: '#94a3b8' },
  { id: 'cinematic_teal_orange', label: 'Cinematic Teal & Orange', badgeColor: '#06b6d4' },
  { id: 'teal_orange_cinema', label: 'Teal & Orange Cinema', badgeColor: '#0891b2' },
  { id: 'rec709_standard', label: 'Rec.709 Standard', badgeColor: '#38bdf8' },
  { id: 'kodak_2383_warm', label: 'Kodak 2383 Warm', badgeColor: '#f59e0b' },
  { id: 'warm_sunset', label: 'Warm Golden Sunset', badgeColor: '#fb923c' },
  { id: 'vintage_kodak', label: 'Vintage Kodak Film', badgeColor: '#d97706' },
  { id: 'vintage_sepia', label: 'Vintage Sepia', badgeColor: '#b45309' },
  { id: 'bleach_bypass_gritty', label: 'Bleach Bypass Gritty', badgeColor: '#a1a1aa' },
  { id: 'cool_cyberpunk', label: 'Cool Cyberpunk', badgeColor: '#c084fc' },
  { id: 'cyberpunk_neon', label: 'Cyberpunk Neon', badgeColor: '#d946ef' },
  { id: 'noir_bw', label: 'Noir B&W', badgeColor: '#cbd5e1' },
  { id: 'noir_monochrome', label: 'Noir Monochrome', badgeColor: '#e2e8f0' },
];

export function describeStyleProfile(
  styleProfile: StyleProfileInput,
  width = 1920,
  height = 1080
): { genre: string; aspect_ratio: string } {
  if (typeof styleProfile === 'object' && styleProfile !== null) {
    return {
      genre: styleProfile.genre || 'cinematic_film',
      aspect_ratio: styleProfile.aspect_ratio || (width >= height ? '16:9' : '9:16'),
    };
  }
  const aspect = width >= height ? '16:9' : '9:16';
  switch (styleProfile) {
    case 'documentary_clean':
      return { genre: 'Documentary Clean', aspect_ratio: aspect };
    case 'fast_social_cut':
      return { genre: 'Fast Social Cut', aspect_ratio: aspect };
    case 'vintage_analog':
      return { genre: 'Vintage Analog', aspect_ratio: aspect };
    case 'cinematic_film':
    default:
      return { genre: String(styleProfile || 'Cinematic Film'), aspect_ratio: aspect };
  }
}

export function framesToTimecode(frame: number, fps: number): string {
  const safeFps = Math.max(1, Math.round(fps || 30));
  const clamped = Math.max(0, Math.floor(frame));
  const ff = clamped % safeFps;
  const totalSec = Math.floor(clamped / safeFps);
  const ss = totalSec % 60;
  const totalMin = Math.floor(totalSec / 60);
  const mm = totalMin % 60;
  const hh = Math.floor(totalMin / 60);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(hh)}:${pad(mm)}:${pad(ss)}:${pad(ff)}`;
}

export function evaluateLutTintOverlay(lut: LutPreset): {
  color: string;
  opacity: number;
  label: string;
} {
  switch (lut) {
    case 'kodak_2383_warm':
    case 'warm_sunset':
      return { color: '#f59e0b', opacity: 0.14, label: 'Kodak 2383 Warm' };
    case 'cinematic_teal_orange':
    case 'teal_orange_cinema':
      return { color: '#0891b2', opacity: 0.16, label: 'Teal & Orange Cinema' };
    case 'bleach_bypass_gritty':
      return { color: '#71717a', opacity: 0.2, label: 'Bleach Bypass Gritty' };
    case 'vintage_kodak':
    case 'vintage_sepia':
      return { color: '#b45309', opacity: 0.22, label: 'Vintage Film' };
    case 'cool_cyberpunk':
    case 'cyberpunk_neon':
      return { color: '#c026d3', opacity: 0.18, label: 'Cyberpunk Neon' };
    case 'noir_bw':
    case 'noir_monochrome':
      return { color: '#0f172a', opacity: 0.25, label: 'Noir Monochrome' };
    case 'rec709_standard':
      return { color: '#38bdf8', opacity: 0.05, label: 'Rec.709 Standard' };
    default:
      return { color: '#000000', opacity: 0.0, label: 'Clean Neutral' };
  }
}

function normalizeGammaComponent(val: number): number {
  return Math.abs(val - 1.0) < 1e-6 ? 0.0 : val;
}

export function combineColorGrades(clipGrade: ColorGrade, masterGrade: ColorGrade): ColorGrade {
  return {
    lift: [
      clipGrade.lift[0] + masterGrade.lift[0],
      clipGrade.lift[1] + masterGrade.lift[1],
      clipGrade.lift[2] + masterGrade.lift[2],
    ],
    gamma: [
      normalizeGammaComponent(clipGrade.gamma[0]) + normalizeGammaComponent(masterGrade.gamma[0]),
      normalizeGammaComponent(clipGrade.gamma[1]) + normalizeGammaComponent(masterGrade.gamma[1]),
      normalizeGammaComponent(clipGrade.gamma[2]) + normalizeGammaComponent(masterGrade.gamma[2]),
    ],
    gain: [
      clipGrade.gain[0] * masterGrade.gain[0],
      clipGrade.gain[1] * masterGrade.gain[1],
      clipGrade.gain[2] * masterGrade.gain[2],
    ],
    contrast: Number((clipGrade.contrast * masterGrade.contrast).toFixed(3)),
    saturation: Number((clipGrade.saturation * masterGrade.saturation).toFixed(3)),
    temperature: clipGrade.temperature + masterGrade.temperature,
    tint: clipGrade.tint + masterGrade.tint,
    lut: clipGrade.lut !== 'clean_neutral' ? clipGrade.lut : masterGrade.lut,
    vignette: Math.min(1, clipGrade.vignette + masterGrade.vignette),
  };
}

export function evaluateColorGradeCssFilter(grade: ColorGrade): string {
  const avgGain = (grade.gain[0] + grade.gain[1] + grade.gain[2]) / 3;
  const avgLift = (grade.lift[0] + grade.lift[1] + grade.lift[2]) / 3;
  const avgGamma =
    (normalizeGammaComponent(grade.gamma[0]) +
      normalizeGammaComponent(grade.gamma[1]) +
      normalizeGammaComponent(grade.gamma[2])) /
    3;
  const brightness = Math.max(0.2, Math.min(2.2, avgGain + avgLift * 0.5 + avgGamma * 0.35));
  const contrast = Math.max(0.2, Math.min(2.5, grade.contrast));
  const saturate =
    grade.lut === 'noir_monochrome' || grade.lut === 'noir_bw'
      ? 0
      : grade.lut === 'bleach_bypass_gritty'
        ? grade.saturation * 0.55
        : Math.max(0, Math.min(3.0, grade.saturation));
  const sepia =
    grade.lut === 'vintage_sepia' || grade.lut === 'vintage_kodak'
      ? 0.45
      : grade.temperature > 15
        ? 0.15
        : 0;
  const hueRotate = Math.round(grade.tint * 0.4 - grade.temperature * 0.2);
  return `brightness(${brightness.toFixed(2)}) contrast(${contrast.toFixed(2)}) saturate(${saturate.toFixed(2)}) sepia(${sepia.toFixed(2)}) hue-rotate(${hueRotate}deg)`;
}

export function evaluateFrameComposition(
  bundle: PlaybackBundle,
  frame: number
): FrameCompositionState {
  const clampedFrame = Math.max(0, Math.min(Math.max(0, bundle.total_frames - 1), Math.floor(frame)));
  const timecode = framesToTimecode(clampedFrame, bundle.fps);

  const soloTracks = bundle.tracks.filter((t) => t.solo && !t.muted);
  const candidateTracks = soloTracks.length > 0 ? soloTracks : bundle.tracks;

  let primaryClip: CompiledClip | null = null;
  let secondaryClip: CompiledClip | null = null;
  for (const track of candidateTracks) {
    if (track.muted || track.kind !== 'video') continue;
    for (const clip of track.clips) {
      if (clampedFrame >= clip.timeline_in_frame && clampedFrame < clip.timeline_out_frame) {
        if (!primaryClip) {
          primaryClip = clip;
        } else if (!secondaryClip) {
          secondaryClip = clip;
        }
      }
    }
  }

  const effectiveGrade = primaryClip
    ? combineColorGrades(primaryClip.color_grade, bundle.master_grade)
    : bundle.master_grade;

  let sourceFrame = 0;
  let transitionKind: TransitionKind = 'none';
  let transitionProgress = 0;

  if (primaryClip) {
    const timelineOffset = clampedFrame - primaryClip.timeline_in_frame;
    sourceFrame =
      primaryClip.source_in_frame + Math.round(timelineOffset * (primaryClip.speed || 1));
    if (
      primaryClip.transition_out &&
      primaryClip.transition_out.kind !== 'none' &&
      primaryClip.transition_out.duration_frames > 0
    ) {
      const transStart =
        primaryClip.timeline_out_frame - primaryClip.transition_out.duration_frames;
      if (clampedFrame >= transStart) {
        transitionKind = primaryClip.transition_out.kind;
        transitionProgress = Math.min(
          1,
          Math.max(
            0,
            (clampedFrame - transStart) / primaryClip.transition_out.duration_frames
          )
        );
      }
    }
  }

  const activeTitles: ActiveTitleRenderState[] = [];
  for (const track of candidateTracks) {
    if (track.muted) continue;
    for (const t of track.titles) {
      if (clampedFrame >= t.timeline_in_frame && clampedFrame < t.timeline_out_frame) {
        const span = Math.max(1, t.timeline_out_frame - t.timeline_in_frame);
        const rel = clampedFrame - t.timeline_in_frame;
        const progress = rel / span;
        const animIn = Math.min(1, rel / Math.min(15, span * 0.25));
        let opacity = 1;
        let translateY = 0;
        let visibleText = t.clip.text;
        const animKind: TitleAnimation = t.clip.animation_in ?? t.clip.animation ?? 'fade_up';

        if (animKind === 'fade_up' || animKind === 'fade_in') {
          opacity = animIn;
          translateY = animKind === 'fade_up' ? (1 - animIn) * 14 : 0;
        } else if (animKind === 'pop' || animKind === 'glitch_in') {
          opacity = animIn;
          translateY = (1 - animIn) * -6;
        } else if (animKind === 'typewriter') {
          const chars = Math.max(
            1,
            Math.ceil(t.clip.text.length * Math.min(1, progress * 1.6))
          );
          visibleText = t.clip.text.slice(0, chars);
        }

        activeTitles.push({
          title: t,
          progress,
          opacity,
          translateY,
          visibleText,
        });
      }
    }
  }

  const resolutionLabel = primaryClip
    ? `${primaryClip.stream_width}x${primaryClip.stream_height} (${bundle.active_tier.toUpperCase()})`
    : `${bundle.width}x${bundle.height} (${bundle.active_tier.toUpperCase()})`;

  return {
    frame: clampedFrame,
    timecode,
    activeTier: bundle.active_tier,
    resolutionLabel,
    primaryClip,
    secondaryClip,
    sourceFrame,
    transitionKind,
    transitionProgress,
    effectiveGrade,
    cssFilter: evaluateColorGradeCssFilter(effectiveGrade),
    lutOverlay: evaluateLutTintOverlay(effectiveGrade.lut),
    activeTitles,
  };
}

export function renderCompositionToCanvas(
  ctx: CanvasRenderingContext2D,
  comp: FrameCompositionState,
  width: number,
  height: number,
  videoElement?: HTMLVideoElement | null
): void {
  ctx.save();
  ctx.clearRect(0, 0, width, height);

  const grade = comp.effectiveGrade;
  const tempShiftR = Math.round(grade.temperature * 0.9 + grade.lift[0] * 110);
  const tempShiftG = Math.round(grade.tint * 0.6 + grade.lift[1] * 110);
  const tempShiftB = Math.round(-grade.temperature * 0.9 + grade.lift[2] * 110);
  const clampByte = (v: number) => Math.max(0, Math.min(255, Math.round(v)));

  if (comp.primaryClip) {
    const isDerived = comp.primaryClip.asset_kind === 'derived';
    const hasRealVideoFrame =
      Boolean(videoElement) &&
      (videoElement?.readyState ?? 0) >= 2 &&
      (videoElement?.videoWidth ?? 0) > 0;

    if (hasRealVideoFrame && videoElement) {
      ctx.save();
      const tx = (comp.primaryClip.transform?.position_x ?? 0) * width * 0.25;
      const ty = (comp.primaryClip.transform?.position_y ?? 0) * height * 0.25;
      const scale = comp.primaryClip.transform?.scale ?? 1.0;
      ctx.translate(width / 2 + tx, height / 2 + ty);
      ctx.scale(scale, scale);
      ctx.drawImage(videoElement, -width / 2, -height / 2, width, height);
      ctx.restore();
    } else {
      // Rich procedural cinema scene with animated skyline, bokeh, and subject motion
      const isInterview =
        comp.primaryClip.asset_id.includes('interview') ||
        comp.primaryClip.asset_id.includes('cam');
      const baseR = clampByte((isDerived ? 26 : isInterview ? 42 : 18) * grade.gain[0] + tempShiftR);
      const baseG = clampByte((isDerived ? 48 : isInterview ? 28 : 34) * grade.gain[1] + tempShiftG);
      const baseB = clampByte((isDerived ? 86 : isInterview ? 22 : 68) * grade.gain[2] + tempShiftB);

      const grad = ctx.createLinearGradient(0, 0, width, height);
      grad.addColorStop(0, `rgb(${baseR}, ${baseG}, ${baseB})`);
      grad.addColorStop(
        1,
        `rgb(${clampByte(baseR + 54)}, ${clampByte(baseG + 32)}, ${clampByte(baseB + 28)})`
      );
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

      // Animated horizon & bokeh lights
      const phase = comp.sourceFrame * 0.04;
      const sunX = width * (0.28 + 0.44 * ((Math.sin(phase * 0.5) + 1) / 2));
      const sunY = height * (0.36 + 0.08 * Math.cos(phase));
      ctx.fillStyle = isDerived ? 'rgba(56, 189, 248, 0.28)' : 'rgba(251, 191, 36, 0.32)';
      if (typeof ctx.arc === 'function' && typeof ctx.fill === 'function') {
        ctx.beginPath();
        ctx.arc(sunX, sunY, 34, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillRect(sunX - 17, sunY - 17, 34, 34);
      }

      // Animated skyline / studio silhouettes
      const gridOffset = (comp.sourceFrame * 3) % 48;
      ctx.fillStyle = 'rgba(9, 13, 22, 0.55)';
      for (let x = -gridOffset; x < width; x += 48) {
        const barH = 26 + ((Math.abs(Math.floor(x + comp.sourceFrame) * 17) % 5) * 11);
        ctx.fillRect(x, height - barH - 22, 34, barH);
      }

      // Motion grid lines
      ctx.strokeStyle = isDerived ? 'rgba(56, 189, 248, 0.22)' : 'rgba(148, 163, 184, 0.16)';
      ctx.lineWidth = 1;
      for (let x = -gridOffset; x < width; x += 48) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
    }

    // B-roll picture-in-picture indicator when a secondary video track clip is active
    if (comp.secondaryClip) {
      ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
      ctx.fillRect(width - 172, 14, 158, 42);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
      if (typeof ctx.strokeRect === 'function') {
        ctx.strokeRect(width - 172, 14, 158, 42);
      }
      ctx.fillStyle = '#38bdf8';
      ctx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(`B-ROLL OVERLAY (${comp.secondaryClip.track_id.toUpperCase()})`, width - 164, 29);
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(`${comp.secondaryClip.clip_id}`, width - 164, 44);
    }

    // Clip metadata HUD inside monitor canvas
    ctx.fillStyle = 'rgba(9, 13, 22, 0.65)';
    ctx.fillRect(8, 8, Math.min(width - 16, 460), 40);
    ctx.fillStyle = 'rgba(226, 232, 240, 0.92)';
    ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const tierBadge =
      comp.activeTier === 'preview'
        ? 'TIER 1 PROXY (540p)'
        : comp.activeTier === 'source'
          ? 'TIER 2 SOURCE VAULT'
          : 'TIER 3 MASTER RENDER';
    ctx.fillText(
      `${comp.primaryClip.clip_id} • ${comp.primaryClip.asset_id} [${comp.primaryClip.asset_kind.toUpperCase()}] • ${tierBadge}`,
      14,
      24
    );
    ctx.fillStyle = 'rgba(148, 163, 184, 0.85)';
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(
      `Stream: ${comp.primaryClip.stream_path} (${comp.primaryClip.stream_width}x${comp.primaryClip.stream_height}) • Src Frame ${comp.sourceFrame}`,
      14,
      40
    );
  } else {
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('NO CLIP AT PLAYHEAD', Math.max(16, width / 2 - 68), height / 2);
  }

  // 2. LUT Tint Overlay
  if (comp.lutOverlay.opacity > 0) {
    ctx.fillStyle = comp.lutOverlay.color;
    ctx.globalAlpha = comp.lutOverlay.opacity;
    ctx.fillRect(0, 0, width, height);
    ctx.globalAlpha = 1;
  }

  // 3. Vignette
  if (grade.vignette > 0.01) {
    const rad = ctx.createRadialGradient(
      width / 2,
      height / 2,
      width * 0.25,
      width / 2,
      height / 2,
      width * 0.65
    );
    rad.addColorStop(0, 'rgba(0,0,0,0)');
    rad.addColorStop(1, `rgba(0,0,0,${Math.min(0.85, grade.vignette).toFixed(2)})`);
    ctx.fillStyle = rad;
    ctx.fillRect(0, 0, width, height);
  }

  // 4. Transition Overlay
  if (comp.transitionKind !== 'none' && comp.transitionProgress > 0) {
    if (comp.transitionKind === 'wipe_left') {
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.fillRect(width * (1 - comp.transitionProgress), 0, width * comp.transitionProgress, height);
    } else {
      ctx.globalAlpha = comp.transitionProgress * 0.75;
      ctx.fillStyle = comp.transitionKind === 'dip_to_white' ? '#ffffff' : '#000000';
      ctx.fillRect(0, 0, width, height);
      ctx.globalAlpha = 1;
    }
  }

  // 5. Title Overlays
  for (const activeTitle of comp.activeTitles) {
    const { clip } = activeTitle.title;
    ctx.save();
    ctx.globalAlpha = activeTitle.opacity;
    const pos = clip.position || 'center';
    const boxY =
      pos.startsWith('top')
        ? 54 + activeTitle.translateY
        : pos === 'center'
          ? height / 2 - 18 + activeTitle.translateY
          : height - 56 + activeTitle.translateY;
    const backdropAlpha = clip.backdrop_opacity ?? clip.backdrop_blur ?? 0.25;
    if (backdropAlpha > 0) {
      ctx.fillStyle = `rgba(15, 23, 42, ${Math.min(0.9, backdropAlpha).toFixed(2)})`;
      ctx.fillRect(16, boxY - 20, width - 32, clip.subtitle ? 44 : 30);
    }
    ctx.fillStyle = clip.color_hex || '#ffffff';
    ctx.font = `700 ${Math.min(22, Math.max(12, Math.round((clip.font_size_px || 42) * 0.35)))}px Inter, system-ui, sans-serif`;
    ctx.fillText(activeTitle.visibleText, 24, boxY);
    if (clip.subtitle) {
      ctx.fillStyle = 'rgba(203, 213, 225, 0.9)';
      ctx.font = '11px Inter, system-ui, sans-serif';
      ctx.fillText(clip.subtitle, 24, boxY + 16);
    }
    ctx.restore();
  }

  ctx.restore();
}
