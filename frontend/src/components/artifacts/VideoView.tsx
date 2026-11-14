import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, Pause, Play, RefreshCw, Scissors, Sparkles } from 'lucide-react';
import { Button } from '../ui/Button';
import { CoreFailure, failureOf, plainFailure } from '../../lib/coreFailure';
import { openConfirmedWindow } from '../../lib/person';
import {
  describeStyleProfile,
  evaluateFrameComposition,
  framesToTimecode,
  LUT_PRESETS,
  renderCompositionToCanvas,
  type BakeOpKind,
  type CompiledClip,
  type FrameRangeHit,
  type FrameSearchRanking,
  type LutPreset,
  type QualityTier,
  type ShownAssetView,
  type ShownProposalView,
  type ShownTrackView,
  type ShownVideoProject,
  type TitleAnimation,
  type TitleStylePreset,
  type VideoRegionSelection,
} from './VideoCompositorEngine';

export interface VideoViewProps {
  projectId: string;
  /**
   * The conversation this view is docked beside. A decision made here is sent with it, so a
   * video project no live conversation holds is opened in this one instead of refusing the click.
   */
  conversationId?: string | null;
  onBack?: () => void;
  refreshKey?: unknown;
  onSelectRegion?: (selection: VideoRegionSelection | null) => void;
}

const BASE = '/api/artifacts/library/videos';

async function readVideoJson<T>(res: Response): Promise<T> {
  if (!res.ok) throw await failureOf(res);
  return (await res.json()) as T;
}

const postVideoJson = (url: string, body: unknown): Promise<Response> =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

export const VideoView: React.FC<VideoViewProps> = ({
  projectId,
  conversationId,
  onBack,
  refreshKey,
  onSelectRegion,
}) => {
  const [project, setProject] = useState<ShownVideoProject | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<CoreFailure | null>(null);
  const [actionError, setActionError] = useState<CoreFailure | null>(null);
  const [busyAction, setBusyAction] = useState<boolean>(false);

  // Transport & Playhead
  const [playheadFrame, setPlayheadFrame] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [loopPlayback, setLoopPlayback] = useState<boolean>(true);
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);
  const [abMode, setAbMode] = useState<'original' | 'proposed'>('original');
  const [activeProposalId, setActiveProposalId] = useState<string | null>(null);
  const isScrubbingRef = useRef<boolean>(false);

  // Media Pool & Semantic Search
  const [mediaFilter, setMediaFilter] = useState<'all' | 'raw' | 'derived'>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchRanking, setSearchRanking] = useState<FrameSearchRanking | null>(null);
  const [searching, setSearching] = useState<boolean>(false);

  // Selection & Inspector
  const [selectedTrackId, setSelectedTrackId] = useState<string>('v1_main');
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [inspectorTab, setInspectorTab] = useState<'color' | 'titles' | 'bake'>('color');
  const [timelineTool, setTimelineTool] = useState<'select' | 'trim' | 'blade'>('select');
  const [snapEnabled, setSnapEnabled] = useState<boolean>(true);
  const [linkedSelection, setLinkedSelection] = useState<boolean>(true);
  const [timelineZoom, setTimelineZoom] = useState<number>(1.0);

  // Color Grade Draft
  const [draftLut, setDraftLut] = useState<LutPreset>('clean_neutral');
  const [draftContrast, setDraftContrast] = useState<number>(1.0);
  const [draftSaturation, setDraftSaturation] = useState<number>(1.0);
  const [draftTemperature, setDraftTemperature] = useState<number>(0.0);
  const [draftVignette, setDraftVignette] = useState<number>(0.0);
  const [draftLiftB, setDraftLiftB] = useState<number>(0.0);
  const [draftGainR, setDraftGainR] = useState<number>(1.0);

  // Title Draft
  const [titleText, setTitleText] = useState<string>('DIRECTOR CUT');
  const [titleSubtitle, setTitleSubtitle] = useState<string>('');
  const [titlePreset, setTitlePreset] = useState<TitleStylePreset>('minimal_lower_third');
  const [titleAnimation, setTitleAnimation] = useState<TitleAnimation>('fade_up');
  const [titleDurationFrames] = useState<number>(90);

  // Bake Derived Draft
  const [bakeOp, setBakeOp] = useState<BakeOpKind>('optical_flow_slowmo');
  const [bakeRate, setBakeRate] = useState<number>(0.5);
  const [bakeStrength] = useState<number>(0.85);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [videoReady, setVideoReady] = useState<boolean>(false);

  const loadProject = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BASE}/${encodeURIComponent(projectId)}`);
      const data = await readVideoJson<ShownVideoProject>(res);
      setProject(data);
      if (data.tracks.length > 0) {
        const firstVideo = data.tracks.find((t) => t.kind === 'video') ?? data.tracks[0];
        setSelectedTrackId((prev) =>
          data.tracks.some((t) => t.track_id === prev) ? prev : firstVideo.track_id
        );
        if (firstVideo.clips.length > 0) {
          setSelectedClipId((prev) =>
            prev && firstVideo.clips.some((c) => c.clip_id === prev)
              ? prev
              : firstVideo.clips[0].clip_id
          );
        }
      }
      if (data.proposals.length > 0) {
        setActiveProposalId((prev) =>
          prev && data.proposals.some((p) => p.id === prev) ? prev : data.proposals[0].id
        );
      } else {
        setActiveProposalId(null);
        setAbMode('original');
      }
    } catch (err) {
      setError(
        err instanceof CoreFailure
          ? err
          : new CoreFailure(500, plainFailure(err, 'Could not load the video project.'))
      );
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void loadProject();
  }, [loadProject, refreshKey]);

  // Active playback bundle (Original A vs Proposed B)
  const activeBundle = useMemo(() => {
    if (!project) return null;
    if (abMode === 'proposed' && project.proposals.length > 0) {
      const prop =
        project.proposals.find((p) => p.id === activeProposalId) ?? project.proposals[0];
      return prop.preview_bundle;
    }
    return project.playback;
  }, [abMode, activeProposalId, project]);

  // Selected track & clip
  const selectedTrack: ShownTrackView | null = useMemo(() => {
    if (!project) return null;
    return (
      project.tracks.find((t) => t.track_id === selectedTrackId) ?? project.tracks[0] ?? null
    );
  }, [project, selectedTrackId]);

  const selectedCompiledClip: CompiledClip | null = useMemo(() => {
    if (!activeBundle) return null;
    for (const track of activeBundle.tracks) {
      for (const clip of track.clips) {
        if (clip.clip_id === selectedClipId) return clip;
      }
    }
    return null;
  }, [activeBundle, selectedClipId]);

  // Sync draft color controls when selected clip changes
  useEffect(() => {
    if (!selectedCompiledClip) return;
    const g = selectedCompiledClip.color_grade;
    setDraftLut(g.lut);
    setDraftContrast(g.contrast);
    setDraftSaturation(g.saturation);
    setDraftTemperature(g.temperature);
    setDraftVignette(g.vignette);
    setDraftLiftB(g.lift[2]);
    setDraftGainR(g.gain[0]);
  }, [selectedCompiledClip]);

  // Playback RAF / interval loop
  useEffect(() => {
    if (!isPlaying || !activeBundle) return;
    const fps = Math.max(1, activeBundle.fps || 30);
    const total = Math.max(1, activeBundle.total_frames || 300);
    const effectiveFps = Math.max(1, fps * playbackRate);
    const intervalMs = Math.max(16, Math.round(1000 / effectiveFps));
    const timer = window.setInterval(() => {
      setPlayheadFrame((prev) => {
        if (prev + 1 >= total) {
          if (loopPlayback) return 0;
          setIsPlaying(false);
          return total - 1;
        }
        return prev + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [isPlaying, activeBundle, loopPlayback, playbackRate]);

  // Frame composition & Canvas render
  const frameComposition = useMemo(() => {
    if (!activeBundle) return null;
    return evaluateFrameComposition(activeBundle, playheadFrame);
  }, [activeBundle, playheadFrame]);

  const activeStreamUrl = useMemo(() => {
    if (!project || !frameComposition?.primaryClip) return null;
    const clip = frameComposition.primaryClip;
    return `${BASE}/${encodeURIComponent(project.project_id)}/stream/${encodeURIComponent(
      clip.resolved_tier
    )}/${encodeURIComponent(clip.asset_id)}?sha=${encodeURIComponent(
      clip.stream_sha256.slice(0, 12)
    )}`;
  }, [frameComposition?.primaryClip, project]);

  const drawMonitorCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frameComposition) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    renderCompositionToCanvas(
      ctx,
      frameComposition,
      canvas.width,
      canvas.height,
      videoReady ? videoRef.current : null
    );
  }, [frameComposition, videoReady]);

  useEffect(() => {
    const videoEl = videoRef.current;
    const clip = frameComposition?.primaryClip;
    if (videoEl && videoReady && clip && activeBundle) {
      const fps = Math.max(1, activeBundle.fps || 30);
      const localOffset = Math.max(0, playheadFrame - clip.timeline_in_frame);
      const sourceFrame = clip.source_in_frame + localOffset * (clip.speed || 1.0);
      const targetSec = Math.max(0, sourceFrame / fps);
      const targetRate = Math.max(0.25, Math.min(4.0, (clip.speed || 1.0) * playbackRate));
      try {
        if (Math.abs((videoEl.playbackRate || 1.0) - targetRate) > 0.05) {
          videoEl.playbackRate = targetRate;
        }
        if (isPlaying) {
          if (videoEl.paused) {
            void videoEl.play?.()?.catch?.(() => {});
          }
          if (Math.abs((videoEl.currentTime || 0) - targetSec) > 0.25) {
            videoEl.currentTime = targetSec;
          }
        } else {
          if (!videoEl.paused) {
            videoEl.pause?.();
          }
          if (Math.abs((videoEl.currentTime || 0) - targetSec) > 0.02) {
            videoEl.currentTime = targetSec;
          }
        }
      } catch {
        /* jsdom or unseekable stream */
      }
    }
    drawMonitorCanvas();
  }, [activeBundle, drawMonitorCanvas, frameComposition, isPlaying, playbackRate, playheadFrame, videoReady]);

  const handleSelectClip = useCallback(
    (trackId: string, clip: CompiledClip) => {
      setSelectedTrackId(trackId);
      setSelectedClipId(clip.clip_id);
      setPlayheadFrame(clip.timeline_in_frame);
      onSelectRegion?.({
        track_id: trackId,
        start_frame: clip.timeline_in_frame,
        end_frame: clip.timeline_out_frame,
        clip_id: clip.clip_id,
      });
    },
    [onSelectRegion]
  );

  // Mutation helper
  const runMutation = useCallback(
    async (endpoint: string, payload: Record<string, unknown>) => {
      setBusyAction(true);
      setActionError(null);
      try {
        const res = await postVideoJson(
          `${BASE}/${encodeURIComponent(projectId)}${endpoint}`,
          {
            ...payload,
            room_id: conversationId ?? undefined,
          }
        );
        const updated = await readVideoJson<ShownVideoProject>(res);
        setProject(updated);
      } catch (err) {
        setActionError(
          err instanceof CoreFailure
            ? err
            : new CoreFailure(500, plainFailure(err, 'Could not apply the video studio action.'))
        );
      } finally {
        setBusyAction(false);
      }
    },
    [conversationId, projectId]
  );

  const handleSwitchTier = useCallback(
    (tier: QualityTier) => {
      if (tier === 'master') {
        void runMutation('/render-master', {});
      } else {
        void runMutation('/tier', { tier });
      }
    },
    [runMutation]
  );

  const handleSearchFrames = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!searchQuery.trim()) return;
      setSearching(true);
      setActionError(null);
      try {
        const res = await postVideoJson(
          `${BASE}/${encodeURIComponent(projectId)}/search`,
          {
            query: searchQuery.trim(),
            top_k: 6,
          }
        );
        const ranking = await readVideoJson<FrameSearchRanking>(res);
        setSearchRanking(ranking);
      } catch (err) {
        setActionError(
          err instanceof CoreFailure
            ? err
            : new CoreFailure(500, plainFailure(err, 'Could not apply the video studio action.'))
        );
      } finally {
        setSearching(false);
      }
    },
    [projectId, searchQuery]
  );

  const handleIngestSampleMedia = useCallback(() => {
    const idx = (project?.assets.length ?? 0) + 1;
    const assetId = `raw_cam_${String(idx).padStart(2, '0')}`;
    void runMutation('/media', {
      asset_id: assetId,
      filename: `${assetId}_take.mp4`,
      fps: project?.fps ?? 30,
      total_frames: 300,
      width: 1920,
      height: 1080,
    });
  }, [project, runMutation]);

  const handleUploadVideoFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      e.target.value = '';
      const rawStem = file.name
        .replace(/\.[^.]+$/, '')
        .toLowerCase()
        .replace(/[^a-z0-9_]+/g, '_')
        .replace(/^_+|_+$/g, '');
      const idx = (project?.assets.length ?? 0) + 1;
      const assetId = rawStem || `upload_cam_${String(idx).padStart(2, '0')}`;
      try {
        const buf = await file.arrayBuffer();
        const bytes = new Uint8Array(buf);
        let binary = '';
        const chunkSize = 0x8000;
        for (let i = 0; i < bytes.length; i += chunkSize) {
          binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
        }
        const payloadB64 = window.btoa(binary);
        await runMutation('/media', {
          asset_id: assetId,
          filename: file.name,
          fps: project?.fps ?? 30,
          total_frames: 300,
          width: project?.width ?? 1920,
          height: project?.height ?? 1080,
          payload_b64: payloadB64,
        });
      } catch (err) {
        setActionError(
          err instanceof CoreFailure
            ? err
            : new CoreFailure(500, plainFailure(err, 'Could not upload the video file.'))
        );
      }
    },
    [project, runMutation]
  );

  const handleAppendToTimeline = useCallback(
    (assetId: string, sourceIn: number, sourceOut: number, label?: string) => {
      if (!project) return;
      const targetTrack =
        (selectedTrack?.kind === 'video' ? selectedTrack : null) ??
        project.tracks.find((t) => t.kind === 'video') ??
        project.tracks[0];
      if (!targetTrack) return;
      void runMutation(`/tracks/${encodeURIComponent(targetTrack.track_id)}/edit`, {
        action: 'append_clip',
        seen_digest: targetTrack.digest,
        asset_id: assetId,
        source_in_frame: sourceIn,
        source_out_frame: Math.max(sourceIn + 15, sourceOut),
        label: label ?? '',
      });
    },
    [project, runMutation, selectedTrack]
  );

  const handleToggleTrackState = useCallback(
    (track: ShownTrackView, field: 'muted' | 'solo' | 'locked') => {
      void runMutation(`/tracks/${encodeURIComponent(track.track_id)}/edit`, {
        action: 'set_track_state',
        seen_digest: track.digest,
        [field]: !track[field],
      });
    },
    [runMutation]
  );

  const handleSplitAtPlayhead = useCallback(() => {
    if (!selectedTrack || !selectedCompiledClip) return;
    const midFrame =
      playheadFrame > selectedCompiledClip.timeline_in_frame &&
      playheadFrame < selectedCompiledClip.timeline_out_frame
        ? playheadFrame
        : Math.floor(
            (selectedCompiledClip.timeline_in_frame +
              selectedCompiledClip.timeline_out_frame) /
              2
          );
    void runMutation(`/tracks/${encodeURIComponent(selectedTrack.track_id)}/edit`, {
      action: 'split_clip',
      seen_digest: selectedTrack.digest,
      clip_id: selectedCompiledClip.clip_id,
      split_at_frame: midFrame,
    });
  }, [playheadFrame, runMutation, selectedCompiledClip, selectedTrack]);

  const handleTrimInAtPlayhead = useCallback(() => {
    if (!selectedTrack || !selectedCompiledClip) return;
    const clip = selectedCompiledClip;
    const speed = clip.speed || 1.0;
    const isInside =
      playheadFrame > clip.timeline_in_frame && playheadFrame < clip.timeline_out_frame;
    const frameDelta = isInside
      ? playheadFrame - clip.timeline_in_frame
      : Math.min(15, Math.max(1, Math.floor(clip.duration_frames * 0.25)));
    const newSourceIn = Math.min(
      clip.source_out_frame - 1,
      clip.source_in_frame + Math.max(1, Math.round(frameDelta * speed))
    );
    void runMutation(`/tracks/${encodeURIComponent(selectedTrack.track_id)}/edit`, {
      action: 'trim_clip',
      seen_digest: selectedTrack.digest,
      clip_id: clip.clip_id,
      source_in_frame: newSourceIn,
      ripple: true,
    });
  }, [playheadFrame, runMutation, selectedCompiledClip, selectedTrack]);

  const handleTrimOutAtPlayhead = useCallback(() => {
    if (!selectedTrack || !selectedCompiledClip) return;
    const clip = selectedCompiledClip;
    const speed = clip.speed || 1.0;
    const isInside =
      playheadFrame > clip.timeline_in_frame && playheadFrame < clip.timeline_out_frame;
    const newSourceOut = isInside
      ? Math.max(
          clip.source_in_frame + 1,
          clip.source_in_frame + Math.round((playheadFrame - clip.timeline_in_frame) * speed)
        )
      : Math.max(
          clip.source_in_frame + 1,
          clip.source_out_frame -
            Math.max(1, Math.round(Math.min(15, Math.floor(clip.duration_frames * 0.25)) * speed))
        );
    void runMutation(`/tracks/${encodeURIComponent(selectedTrack.track_id)}/edit`, {
      action: 'trim_clip',
      seen_digest: selectedTrack.digest,
      clip_id: clip.clip_id,
      source_out_frame: newSourceOut,
      ripple: true,
    });
  }, [playheadFrame, runMutation, selectedCompiledClip, selectedTrack]);

  const handleDeleteClip = useCallback(() => {
    if (!selectedTrack || !selectedCompiledClip) return;
    void runMutation(`/tracks/${encodeURIComponent(selectedTrack.track_id)}/edit`, {
      action: 'delete_clip',
      seen_digest: selectedTrack.digest,
      clip_id: selectedCompiledClip.clip_id,
      ripple: true,
    });
  }, [runMutation, selectedCompiledClip, selectedTrack]);

  const handleApplyColorGrade = useCallback(() => {
    if (!selectedTrack || !selectedCompiledClip) return;
    void runMutation(`/tracks/${encodeURIComponent(selectedTrack.track_id)}/edit`, {
      action: 'grade_clip',
      seen_digest: selectedTrack.digest,
      clip_id: selectedCompiledClip.clip_id,
      color_grade: {
        lut: draftLut,
        contrast: draftContrast,
        saturation: draftSaturation,
        temperature: draftTemperature,
        vignette: draftVignette,
        lift: [0.0, 0.0, draftLiftB],
        gamma: [1.0, 1.0, 1.0],
        gain: [draftGainR, 1.0, 1.0],
        tint: 0.0,
      },
    });
  }, [
    draftContrast,
    draftGainR,
    draftLiftB,
    draftLut,
    draftSaturation,
    draftTemperature,
    draftVignette,
    runMutation,
    selectedCompiledClip,
    selectedTrack,
  ]);

  const handleAddTitleOverlay = useCallback(() => {
    if (!project || !titleText.trim()) return;
    const titleTrack =
      project.tracks.find((t) => t.kind === 'title') ?? selectedTrack ?? project.tracks[0];
    if (!titleTrack) return;
    const existingIds = new Set(titleTrack.titles.map((t) => t.title_id));
    let idx = titleTrack.titles.length + 1;
    let titleId = `title_${String(idx).padStart(2, '0')}`;
    while (existingIds.has(titleId)) {
      idx += 1;
      titleId = `title_${String(idx).padStart(2, '0')}`;
    }
    void runMutation(`/tracks/${encodeURIComponent(titleTrack.track_id)}/edit`, {
      action: 'set_title',
      seen_digest: titleTrack.digest,
      title_clip: {
        title_id: titleId,
        text: titleText.trim(),
        subtitle: titleSubtitle.trim(),
        style_preset: titlePreset,
        timeline_in_frame: playheadFrame,
        timeline_out_frame: playheadFrame + Math.max(15, titleDurationFrames),
        position: 'lower_left',
        animation: titleAnimation,
        animation_in: titleAnimation,
        font_size_px: 42,
        color_hex: '#FFFFFF',
        backdrop_opacity: 0.45,
        backdrop_blur: 0.45,
      },
    });
  }, [
    playheadFrame,
    project,
    runMutation,
    selectedTrack,
    titleAnimation,
    titleDurationFrames,
    titlePreset,
    titleSubtitle,
    titleText,
  ]);

  const handleDeleteTitleOverlay = useCallback(
    (trackId: string, titleId: string) => {
      if (!project) return;
      const targetTrack = project.tracks.find((t) => t.track_id === trackId);
      if (!targetTrack) return;
      void runMutation(`/tracks/${encodeURIComponent(targetTrack.track_id)}/edit`, {
        action: 'delete_title',
        seen_digest: targetTrack.digest,
        title_id: titleId,
      });
    },
    [project, runMutation]
  );

  const handleBakeDerived = useCallback(() => {
    if (!project) return;
    const parentId =
      selectedCompiledClip?.asset_id ?? project.assets[0]?.asset.asset_id ?? null;
    if (!parentId) return;
    const inFrame = selectedCompiledClip?.source_in_frame ?? 0;
    const outFrame = selectedCompiledClip?.source_out_frame ?? 150;
    void runMutation('/bake', {
      parent_asset_id: parentId,
      source_in_frame: inFrame,
      source_out_frame: Math.max(inFrame + 15, outFrame),
      operation_kind: bakeOp,
      rate: bakeRate,
      strength: bakeStrength,
      track_id: selectedTrack?.track_id,
      clip_id: selectedCompiledClip?.clip_id,
    });
  }, [
    bakeOp,
    bakeRate,
    bakeStrength,
    project,
    runMutation,
    selectedCompiledClip,
    selectedTrack,
  ]);

  const handleUnbakeToRaw = useCallback(() => {
    if (!selectedTrack || !selectedCompiledClip) return;
    void runMutation(`/tracks/${encodeURIComponent(selectedTrack.track_id)}/edit`, {
      action: 'unbake_to_raw',
      seen_digest: selectedTrack.digest,
      clip_id: selectedCompiledClip.clip_id,
    });
  }, [runMutation, selectedCompiledClip, selectedTrack]);

  const handleRevertTrack = useCallback(
    (track: ShownTrackView) => {
      void runMutation(`/tracks/${encodeURIComponent(track.track_id)}/edit`, {
        action: 'revert',
        seen_digest: track.digest,
      });
    },
    [runMutation]
  );

  const handleApproveProposal = useCallback(
    (prop: ShownProposalView) => {
      void runMutation(`/proposals/${encodeURIComponent(prop.id)}/approve`, {
        seen_digest: prop.seen_digest,
      });
    },
    [runMutation]
  );

  const handleRejectProposal = useCallback(
    (prop: ShownProposalView) => {
      void runMutation(`/proposals/${encodeURIComponent(prop.id)}/reject`, {
        seen_digest: prop.seen_digest,
      });
    },
    [runMutation]
  );

  const handleApproveAllProposals = useCallback(() => {
    void runMutation('/proposals/approve-all', {});
  }, [runMutation]);

  const scrubToClientX = useCallback(
    (clientX: number, rect: DOMRect) => {
      if (!activeBundle) return;
      const total = Math.max(1, activeBundle.total_frames);
      const width = rect.width > 0 ? rect.width : 400;
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / width));
      let targetFrame = Math.min(total - 1, Math.max(0, Math.round(ratio * total)));
      if (snapEnabled && project) {
        const snapPoints = new Set<number>([0, total - 1]);
        for (const sec of project.sections) {
          snapPoints.add(sec.start_frame);
          snapPoints.add(sec.end_frame);
        }
        for (const track of activeBundle.tracks) {
          for (const clip of track.clips) {
            snapPoints.add(clip.timeline_in_frame);
            snapPoints.add(clip.timeline_out_frame);
          }
        }
        for (const pt of snapPoints) {
          if (Math.abs(targetFrame - pt) <= 3) {
            targetFrame = Math.min(total - 1, Math.max(0, pt));
            break;
          }
        }
      }
      setPlayheadFrame(targetFrame);
    },
    [activeBundle, project, snapEnabled]
  );

  // Global NLE Keyboard Shortcuts (Space, J/K/L, ArrowLeft/Right, Home/End, A/T/B, S, [/], Delete, N, Cmd/Ctrl+Z)
  useEffect(() => {
    if (!project || !activeBundle) return;
    const total = Math.max(1, activeBundle.total_frames);
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName;
        if (
          tag === 'INPUT' ||
          tag === 'TEXTAREA' ||
          tag === 'SELECT' ||
          target.isContentEditable
        ) {
          return;
        }
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (selectedTrack?.can_revert && !busyAction) {
          handleRevertTrack(selectedTrack);
        }
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      switch (e.key) {
        case ' ':
        case 'k':
        case 'K':
          e.preventDefault();
          setIsPlaying((p) => !p);
          break;
        case 'ArrowLeft':
        case 'j':
        case 'J': {
          e.preventDefault();
          setIsPlaying(false);
          const step = e.shiftKey ? 10 : 1;
          setPlayheadFrame((f) => Math.max(0, f - step));
          break;
        }
        case 'ArrowRight':
        case 'l':
        case 'L': {
          e.preventDefault();
          setIsPlaying(false);
          const step = e.shiftKey ? 10 : 1;
          setPlayheadFrame((f) => Math.min(total - 1, f + step));
          break;
        }
        case 'Home':
          e.preventDefault();
          setIsPlaying(false);
          setPlayheadFrame(0);
          break;
        case 'End':
          e.preventDefault();
          setIsPlaying(false);
          setPlayheadFrame(Math.max(0, total - 1));
          break;
        case 'a':
        case 'A':
          e.preventDefault();
          setTimelineTool('select');
          break;
        case 't':
        case 'T':
          e.preventDefault();
          setTimelineTool('trim');
          break;
        case 'b':
        case 'B':
          e.preventDefault();
          setTimelineTool('blade');
          break;
        case 'n':
        case 'N':
          e.preventDefault();
          setSnapEnabled((s) => !s);
          break;
        case 's':
        case 'S':
          e.preventDefault();
          if (!busyAction) handleSplitAtPlayhead();
          break;
        case '[':
        case 'i':
        case 'I':
          e.preventDefault();
          if (!busyAction) handleTrimInAtPlayhead();
          break;
        case ']':
        case 'o':
        case 'O':
          e.preventDefault();
          if (!busyAction) handleTrimOutAtPlayhead();
          break;
        case 'Delete':
        case 'Backspace':
          e.preventDefault();
          if (!busyAction) handleDeleteClip();
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    activeBundle,
    busyAction,
    handleDeleteClip,
    handleRevertTrack,
    handleSplitAtPlayhead,
    handleTrimInAtPlayhead,
    handleTrimOutAtPlayhead,
    project,
    selectedTrack,
  ]);

  const filteredAssets: ShownAssetView[] = useMemo(() => {
    if (!project) return [];
    return project.assets.filter((item) => {
      if (mediaFilter === 'raw') return item.asset.kind === 'raw';
      if (mediaFilter === 'derived') return item.asset.kind === 'derived';
      return true;
    });
  }, [mediaFilter, project]);

  if (loading && !project) {
    return (
      <div
        data-testid="video-view-loading"
        className="p-6 text-xs text-slate-400 flex items-center gap-2"
      >
        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
        <span>Loading video studio...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div data-testid="video-view-error" className="p-4 space-y-3 text-xs">
        <p className="text-rose-300">{error.message}</p>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => void loadProject()}>
            Retry
          </Button>
          {onBack && (
            <Button size="sm" variant="ghost" onClick={onBack}>
              Back
            </Button>
          )}
        </div>
      </div>
    );
  }

  if (!project || !activeBundle || !frameComposition) return null;

  const totalFrames = Math.max(1, activeBundle.total_frames);
  const styleInfo = describeStyleProfile(project.style_profile, project.width, project.height);

  return (
    <div
      data-testid="video-view"
      className="flex flex-col h-full min-h-0 bg-slate-950 text-slate-100 select-none overflow-y-auto"
    >
      {/* Top Studio Header Bar: Title + 3-Tier Quality Switcher */}
      <div className="px-3 py-2 border-b border-slate-800 bg-slate-900/80 flex flex-wrap items-center justify-between gap-2 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          {onBack && (
            <button
              type="button"
              data-testid="video-back-button"
              onClick={onBack}
              className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white"
              title="Back to files"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
            </button>
          )}
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span
                data-testid="video-project-title"
                className="text-xs font-semibold text-white truncate"
              >
                {project.title}
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 font-mono">
                {project.fps}fps • {styleInfo.aspect_ratio}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 font-mono">
              {styleInfo.genre} • {project.total_timecode} ({project.total_frames}f)
            </div>
          </div>
        </div>

        {/* 3-Tier Resolution & Quality Switcher */}
        <div
          data-testid="video-tier-switcher"
          className="flex items-center gap-1 bg-slate-950/90 p-0.5 rounded-lg border border-slate-800"
        >
          <button
            type="button"
            data-testid="tier-btn-preview"
            disabled={busyAction}
            onClick={() => handleSwitchTier('preview')}
            className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${
              project.active_tier === 'preview'
                ? 'bg-emerald-600/30 text-emerald-300 border border-emerald-500/50'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            title="Tier 1: Fast 540p Short-GOP Proxy for real-time scrubbing"
          >
            ⚡ Proxy 540p
          </button>
          <button
            type="button"
            data-testid="tier-btn-source"
            disabled={busyAction}
            onClick={() => handleSwitchTier('source')}
            className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${
              project.active_tier === 'source'
                ? 'bg-cyan-600/30 text-cyan-300 border border-cyan-500/50'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            title="Tier 2: Immutable Original/Derived Vault Stream"
          >
            🔍 Source Raw
          </button>
          <button
            type="button"
            data-testid="tier-btn-master"
            disabled={busyAction}
            onClick={() => handleSwitchTier('master')}
            className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${
              project.active_tier === 'master'
                ? 'bg-amber-600/30 text-amber-300 border border-amber-500/50'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            title="Tier 3: Relink to Vault Source & Render Final Master"
          >
            🏆 Render Master
          </button>
        </div>
      </div>

      {/* Writable / Lease Banner & Action Refusal Banner */}
      {!project.writable && project.decide_note && (
        <div
          data-testid="video-readonly-banner"
          className="px-3 py-1.5 bg-amber-950/50 border-b border-amber-800/60 text-[11px] text-amber-200"
        >
          {project.decide_note}
        </div>
      )}

      {actionError && (
        <div
          data-testid="video-action-error"
          className="px-3 py-2 bg-rose-950/70 border-b border-rose-800/70 text-xs text-rose-200 flex items-center justify-between gap-2"
        >
          <span>{actionError.message}</span>
          {actionError.status === 403 && (
            <Button
              size="sm"
              variant="outline"
              data-testid="video-open-confirmed-window"
              onClick={() => {
                void openConfirmedWindow().catch((err) =>
                  setActionError(
                    err instanceof CoreFailure
                      ? err
                      : new CoreFailure(
                          500,
                          plainFailure(err, 'Could not open a confirmed window.')
                        )
                  )
                );
              }}
            >
              Open a confirmed window
            </Button>
          )}
        </div>
      )}

      {project.last_master_receipt && (
        <div
          data-testid="video-master-receipt"
          className="px-3 py-1.5 bg-emerald-950/50 border-b border-emerald-700/50 text-[11px] text-emerald-200 font-mono flex items-center justify-between gap-2"
        >
          <span>
            Master Export Ready: {project.last_master_receipt.output_path} (
            {project.last_master_receipt.width}x{project.last_master_receipt.height}, SHA-256{' '}
            {project.last_master_receipt.sha256.slice(0, 10)})
          </span>
          <span className="text-emerald-300">
            Relinked {project.last_master_receipt.relinked_source_assets.length} source vault
            asset(s)
          </span>
        </div>
      )}

      {/* Pending Proposals Bar with A/B Audition Toggle */}
      {project.proposals.length > 0 && (
        <div
          data-testid="video-proposals-bar"
          className="px-3 py-2 bg-indigo-950/60 border-b border-indigo-700/60 space-y-2 shrink-0"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs font-semibold text-indigo-200 flex items-center gap-2">
              <span>Pending Timeline Proposals ({project.proposals.length})</span>
              <div className="flex items-center gap-1 bg-slate-900/90 p-0.5 rounded border border-indigo-500/40">
                <button
                  type="button"
                  data-testid="video-ab-original"
                  onClick={() => setAbMode('original')}
                  className={`px-2 py-0.5 rounded text-[10px] ${
                    abMode === 'original'
                      ? 'bg-slate-700 text-white'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Original (A)
                </button>
                <button
                  type="button"
                  data-testid="video-ab-proposed"
                  onClick={() => setAbMode('proposed')}
                  className={`px-2 py-0.5 rounded text-[10px] ${
                    abMode === 'proposed'
                      ? 'bg-indigo-600 text-white'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Proposed (B)
                </button>
              </div>
            </div>
            {project.proposals.length > 1 && (
              <button
                type="button"
                data-testid="video-approve-all-btn"
                disabled={busyAction}
                onClick={handleApproveAllProposals}
                className="px-2.5 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-[11px] font-medium"
              >
                Approve All ({project.proposals.length})
              </button>
            )}
          </div>
          <div className="space-y-1.5">
            {project.proposals.map((prop) => (
              <div
                key={prop.id}
                data-testid={`video-proposal-${prop.id}`}
                className="p-2 rounded bg-slate-900/90 border border-indigo-500/40 flex flex-wrap items-center justify-between gap-2 text-xs"
              >
                <div className="min-w-0">
                  <span className="font-mono text-indigo-300 mr-1.5">{prop.id}</span>
                  <span className="text-slate-200">{prop.summary}</span>
                  {prop.track_id && (
                    <span className="ml-2 text-[10px] font-mono text-slate-400">
                      [{prop.track_id}]
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    data-testid={`video-approve-${prop.id}`}
                    disabled={busyAction}
                    onClick={() => handleApproveProposal(prop)}
                    className="px-2 py-1 rounded bg-emerald-600/80 hover:bg-emerald-500 text-white text-[11px] font-medium"
                  >
                    Approve
                  </button>
                  <button
                    type="button"
                    data-testid={`video-reject-${prop.id}`}
                    disabled={busyAction}
                    onClick={() => handleRejectProposal(prop)}
                    className="px-2 py-1 rounded bg-rose-600/40 hover:bg-rose-600/70 text-rose-200 text-[11px]"
                  >
                    Reject
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upper Studio Workspace: Program Monitor + 3-Way Color & FX Inspector */}
      <div className="p-3 grid grid-cols-1 gap-3 border-b border-slate-800 shrink-0">
        {/* DaVinci Program Monitor */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-2.5 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono">
            <span className="text-slate-400">
              PROGRAM MONITOR •{' '}
              <span data-testid="video-monitor-tier" className="text-cyan-300">
                {frameComposition.resolutionLabel}
              </span>
            </span>
            <span data-testid="video-timecode-hud" className="text-amber-300 font-semibold">
              {frameComposition.timecode} / {activeBundle.total_timecode} (F{playheadFrame})
            </span>
          </div>

          <div className="relative rounded overflow-hidden border border-slate-800 bg-black aspect-video flex items-center justify-center">
            {activeStreamUrl && (
              <video
                ref={videoRef}
                data-testid="video-stream-element"
                src={activeStreamUrl}
                muted
                playsInline
                preload="auto"
                onLoadedData={() => {
                  setVideoReady(true);
                  drawMonitorCanvas();
                }}
                onSeeked={() => drawMonitorCanvas()}
                onError={() => setVideoReady(false)}
                className="hidden"
              />
            )}
            <canvas
              ref={canvasRef}
              data-testid="video-program-canvas"
              width={480}
              height={270}
              onClick={() => setIsPlaying((p) => !p)}
              style={{ filter: frameComposition.cssFilter }}
              className="w-full h-full object-cover cursor-pointer"
              title="Click monitor or press Space to Play / Pause"
            />
            {frameComposition.activeTitles.length > 0 && (
              <div
                data-testid="video-monitor-title-overlay"
                className="sr-only"
              >
                {frameComposition.activeTitles.map((t) => t.visibleText).join(' | ')}
              </div>
            )}
          </div>

          {/* Transport & Direct Timeline Action Bar */}
          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
            <div className="flex flex-wrap items-center gap-1">
              <button
                type="button"
                data-testid="video-jump-start"
                onClick={() => {
                  setIsPlaying(false);
                  setPlayheadFrame(0);
                }}
                className="px-1.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-mono"
                title="Jump to Start (Home)"
              >
                |◀◀
              </button>
              <button
                type="button"
                data-testid="video-step-back"
                onClick={() => {
                  setIsPlaying(false);
                  setPlayheadFrame((f) => Math.max(0, f - 1));
                }}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-mono"
                title="Step Back 1 Frame (← or J)"
              >
                |◀
              </button>
              <button
                type="button"
                data-testid="video-play-toggle"
                onClick={() => setIsPlaying((p) => !p)}
                className="px-2.5 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-[11px] font-medium flex items-center gap-1"
                title="Play / Pause (Space or K)"
              >
                {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                <span>{isPlaying ? 'Pause' : 'Play'}</span>
                <span className="text-[9px] font-mono opacity-80 ml-0.5">[Space]</span>
              </button>
              <button
                type="button"
                data-testid="video-step-forward"
                onClick={() => {
                  setIsPlaying(false);
                  setPlayheadFrame((f) => Math.min(totalFrames - 1, f + 1));
                }}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-mono"
                title="Step Forward 1 Frame (→ or L)"
              >
                ▶|
              </button>
              <button
                type="button"
                data-testid="video-jump-end"
                onClick={() => {
                  setIsPlaying(false);
                  setPlayheadFrame(Math.max(0, totalFrames - 1));
                }}
                className="px-1.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-mono"
                title="Jump to End (End)"
              >
                ▶▶|
              </button>
              <button
                type="button"
                data-testid="video-loop-toggle"
                onClick={() => setLoopPlayback((l) => !l)}
                className={`px-1.5 py-1 rounded text-[10px] font-mono border ${
                  loopPlayback
                    ? 'bg-cyan-600/20 text-cyan-300 border-cyan-500/40'
                    : 'bg-slate-900 text-slate-500 border-slate-800'
                }`}
                title="Toggle Loop Playback"
              >
                🔁 Loop
              </button>
              <select
                data-testid="video-playback-rate"
                value={playbackRate}
                onChange={(e) => setPlaybackRate(Number(e.target.value))}
                className="bg-slate-900 border border-slate-800 rounded px-1.5 py-1 text-[10px] font-mono text-slate-200"
                title="Playback Speed"
              >
                <option value={0.5}>0.5x</option>
                <option value={1}>1x</option>
                <option value={2}>2x</option>
              </select>
            </div>

            <div className="flex flex-wrap items-center gap-1">
              <button
                type="button"
                data-testid="video-trim-in-btn"
                disabled={busyAction || !selectedCompiledClip}
                onClick={handleTrimInAtPlayhead}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-amber-200 text-[11px] font-mono disabled:opacity-40"
                title="Trim Clip In-Point ([ or I)"
              >
                [ Trim In
              </button>
              <button
                type="button"
                data-testid="video-trim-out-btn"
                disabled={busyAction || !selectedCompiledClip}
                onClick={handleTrimOutAtPlayhead}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-amber-200 text-[11px] font-mono disabled:opacity-40"
                title="Trim Clip Out-Point (] or O)"
              >
                Trim Out ]
              </button>
              <button
                type="button"
                data-testid="video-split-clip-btn"
                disabled={busyAction || !selectedCompiledClip}
                onClick={handleSplitAtPlayhead}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[11px] flex items-center gap-1 disabled:opacity-40"
                title="Split selected clip (S)"
              >
                <Scissors className="w-3 h-3" />
                <span>Split Clip</span>
              </button>
              <button
                type="button"
                data-testid="video-quick-bake-btn"
                disabled={busyAction || project.assets.length === 0}
                onClick={handleBakeDerived}
                className="px-2 py-1 rounded bg-violet-600/30 hover:bg-violet-600/50 border border-violet-500/40 text-violet-200 text-[11px] flex items-center gap-1 disabled:opacity-40"
                title="Bake heavy transform into a new immutable derived asset while keeping raw untouched"
              >
                <Sparkles className="w-3 h-3" />
                <span>Bake Derived</span>
              </button>
              {selectedCompiledClip?.can_unbake && (
                <button
                  type="button"
                  data-testid="video-unbake-btn"
                  disabled={busyAction}
                  onClick={handleUnbakeToRaw}
                  className="px-2 py-1 rounded bg-amber-600/30 hover:bg-amber-600/50 border border-amber-500/40 text-amber-200 text-[11px]"
                  title="Restore clip to original raw vault asset"
                >
                  ↩ Unbake to Raw
                </button>
              )}
              <button
                type="button"
                data-testid="video-delete-clip-btn"
                disabled={busyAction || !selectedCompiledClip}
                onClick={handleDeleteClip}
                className="px-2 py-1 rounded bg-rose-900/40 hover:bg-rose-800/60 text-rose-200 text-[11px] disabled:opacity-40"
              >
                Ripple Delete
              </button>
            </div>
          </div>
        </div>

        {/* DaVinci 3-Way Color, Titles & Bake FX Inspector */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-2.5 space-y-2.5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
            <div className="flex items-center gap-1">
              <button
                type="button"
                data-testid="inspector-tab-color"
                onClick={() => setInspectorTab('color')}
                className={`px-2 py-1 rounded text-[11px] font-medium ${
                  inspectorTab === 'color'
                    ? 'bg-cyan-600/30 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                🎨 3-Way Color & LUT
              </button>
              <button
                type="button"
                data-testid="inspector-tab-titles"
                onClick={() => setInspectorTab('titles')}
                className={`px-2 py-1 rounded text-[11px] font-medium ${
                  inspectorTab === 'titles'
                    ? 'bg-cyan-600/30 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                🔤 Titles & Lower-Thirds
              </button>
              <button
                type="button"
                data-testid="inspector-tab-bake"
                onClick={() => setInspectorTab('bake')}
                className={`px-2 py-1 rounded text-[11px] font-medium ${
                  inspectorTab === 'bake'
                    ? 'bg-violet-600/30 text-violet-300 border border-violet-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                🔥 Heavy Bake FX
              </button>
            </div>
            {selectedCompiledClip && (
              <span
                data-testid="video-selection-badge"
                className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300"
              >
                🎯 [{selectedTrackId} | F{selectedCompiledClip.timeline_in_frame}–
                {selectedCompiledClip.timeline_out_frame} | {selectedCompiledClip.clip_id}]
              </span>
            )}
          </div>

          {inspectorTab === 'color' && (
            <div data-testid="video-color-panel" className="space-y-2 text-[11px]">
              <div className="grid grid-cols-2 gap-2">
                <label className="space-y-1">
                  <span className="text-slate-400">3D LUT Preset</span>
                  <select
                    data-testid="video-lut-select"
                    value={draftLut}
                    onChange={(e) => setDraftLut(e.target.value as LutPreset)}
                    className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200"
                  >
                    {LUT_PRESETS.map((lut) => (
                      <option key={lut.id} value={lut.id}>
                        {lut.label}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="flex items-end">
                  <button
                    type="button"
                    data-testid="video-apply-grade-btn"
                    disabled={busyAction || !selectedCompiledClip}
                    onClick={handleApplyColorGrade}
                    className="w-full py-1.5 px-3 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium disabled:opacity-40"
                  >
                    Apply Color Grade
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 pt-1">
                <label className="space-y-0.5">
                  <span className="text-slate-400 flex justify-between">
                    <span>Contrast</span>
                    <span className="font-mono">{draftContrast.toFixed(2)}</span>
                  </span>
                  <input
                    type="range"
                    data-testid="video-contrast-slider"
                    min="0.5"
                    max="1.8"
                    step="0.05"
                    value={draftContrast}
                    onChange={(e) => setDraftContrast(Number(e.target.value))}
                    className="w-full accent-cyan-400"
                  />
                </label>
                <label className="space-y-0.5">
                  <span className="text-slate-400 flex justify-between">
                    <span>Saturation</span>
                    <span className="font-mono">{draftSaturation.toFixed(2)}</span>
                  </span>
                  <input
                    type="range"
                    data-testid="video-saturation-slider"
                    min="0"
                    max="2.0"
                    step="0.05"
                    value={draftSaturation}
                    onChange={(e) => setDraftSaturation(Number(e.target.value))}
                    className="w-full accent-cyan-400"
                  />
                </label>
                <label className="space-y-0.5">
                  <span className="text-slate-400 flex justify-between">
                    <span>Temp</span>
                    <span className="font-mono">{draftTemperature}</span>
                  </span>
                  <input
                    type="range"
                    data-testid="video-temp-slider"
                    min="-50"
                    max="50"
                    step="5"
                    value={draftTemperature}
                    onChange={(e) => setDraftTemperature(Number(e.target.value))}
                    className="w-full accent-amber-400"
                  />
                </label>
              </div>

              {/* 3-Way Color Wheels Summary (Lift / Gamma / Gain + Vignette) */}
              <div className="grid grid-cols-3 gap-2 pt-1">
                <label className="space-y-0.5">
                  <span className="text-slate-400 flex justify-between">
                    <span>Lift (Shadows B)</span>
                    <span className="font-mono">{draftLiftB.toFixed(2)}</span>
                  </span>
                  <input
                    type="range"
                    min="-0.3"
                    max="0.3"
                    step="0.02"
                    value={draftLiftB}
                    onChange={(e) => setDraftLiftB(Number(e.target.value))}
                    className="w-full accent-sky-400"
                  />
                </label>
                <label className="space-y-0.5">
                  <span className="text-slate-400 flex justify-between">
                    <span>Gain (Highlights R)</span>
                    <span className="font-mono">{draftGainR.toFixed(2)}</span>
                  </span>
                  <input
                    type="range"
                    min="0.6"
                    max="1.5"
                    step="0.05"
                    value={draftGainR}
                    onChange={(e) => setDraftGainR(Number(e.target.value))}
                    className="w-full accent-amber-400"
                  />
                </label>
                <label className="space-y-0.5">
                  <span className="text-slate-400 flex justify-between">
                    <span>Vignette</span>
                    <span className="font-mono">{draftVignette.toFixed(2)}</span>
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="0.8"
                    step="0.05"
                    value={draftVignette}
                    onChange={(e) => setDraftVignette(Number(e.target.value))}
                    className="w-full accent-purple-400"
                  />
                </label>
              </div>
            </div>
          )}

          {inspectorTab === 'titles' && (
            <div data-testid="video-titles-panel" className="space-y-2 text-[11px]">
              <div className="grid grid-cols-2 gap-2">
                <label className="space-y-0.5">
                  <span className="text-slate-400">Title Text</span>
                  <input
                    type="text"
                    data-testid="video-title-input"
                    value={titleText}
                    onChange={(e) => setTitleText(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200"
                  />
                </label>
                <label className="space-y-0.5">
                  <span className="text-slate-400">Subtitle</span>
                  <input
                    type="text"
                    value={titleSubtitle}
                    onChange={(e) => setTitleSubtitle(e.target.value)}
                    placeholder="Optional subtitle"
                    className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200"
                  />
                </label>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <label className="space-y-0.5">
                  <span className="text-slate-400">Style Preset</span>
                  <select
                    value={titlePreset}
                    onChange={(e) => setTitlePreset(e.target.value as TitleStylePreset)}
                    className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200"
                  >
                    <option value="minimal_lower_third">Minimal Lower Third</option>
                    <option value="cinema_serif_center">Cinema Serif Center</option>
                    <option value="bold_impact_headline">Bold Impact Headline</option>
                    <option value="neon_cyber_subtitle">Neon Cyber Subtitle</option>
                    <option value="documentary_caption">Documentary Caption</option>
                  </select>
                </label>
                <label className="space-y-0.5">
                  <span className="text-slate-400">Animation</span>
                  <select
                    value={titleAnimation}
                    onChange={(e) => setTitleAnimation(e.target.value as TitleAnimation)}
                    className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200"
                  >
                    <option value="fade_up">Fade Up</option>
                    <option value="typewriter">Typewriter</option>
                    <option value="pop">Pop</option>
                    <option value="slide_left">Slide Left</option>
                    <option value="none">None</option>
                  </select>
                </label>
                <div className="flex items-end">
                  <button
                    type="button"
                    data-testid="video-add-title-btn"
                    disabled={busyAction}
                    onClick={handleAddTitleOverlay}
                    className="w-full py-1 px-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium"
                  >
                    + Add Title at F{playheadFrame}
                  </button>
                </div>
              </div>
            </div>
          )}

          {inspectorTab === 'bake' && (
            <div data-testid="video-bake-panel" className="space-y-2 text-[11px]">
              <p className="text-slate-400">
                Heavy transformations render a new immutable derived asset in{' '}
                <code className="text-violet-300">vault/</code> while keeping the original raw
                upload untouched.
              </p>
              <div className="grid grid-cols-3 gap-2">
                <label className="space-y-0.5">
                  <span className="text-slate-400">Heavy Operation</span>
                  <select
                    data-testid="video-bake-op-select"
                    value={bakeOp}
                    onChange={(e) => setBakeOp(e.target.value as BakeOpKind)}
                    className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200"
                  >
                    <option value="optical_flow_slowmo">Optical-Flow Slow Motion</option>
                    <option value="stabilize">Camera Stabilization</option>
                    <option value="auto_reframe_9_16">9:16 Vertical Auto-Reframe</option>
                    <option value="reverse">Reverse Playback</option>
                    <option value="speed_ramp">Speed Ramp</option>
                    <option value="audio_denoise">Spectral Audio Denoise</option>
                  </select>
                </label>
                <label className="space-y-0.5">
                  <span className="text-slate-400">Speed Rate ({bakeRate}x)</span>
                  <input
                    type="range"
                    min="0.25"
                    max="2.0"
                    step="0.25"
                    value={bakeRate}
                    onChange={(e) => setBakeRate(Number(e.target.value))}
                    className="w-full accent-violet-400"
                  />
                </label>
                <div className="flex items-end">
                  <button
                    type="button"
                    data-testid="video-bake-submit-btn"
                    disabled={busyAction || project.assets.length === 0}
                    onClick={handleBakeDerived}
                    className="w-full py-1 px-2 rounded bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium"
                  >
                    🔥 Bake New Asset
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Media Pool (Immutable Raw + Derived Lineage) & Semantic Frame Search Bin */}
      <div className="p-3 border-b border-slate-800 bg-slate-900/40 space-y-2.5 shrink-0">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-semibold text-slate-200">Media Vault Pool</span>
            <div className="flex items-center gap-1 bg-slate-950 p-0.5 rounded border border-slate-800 text-[10px]">
              <button
                type="button"
                data-testid="media-filter-all"
                onClick={() => setMediaFilter('all')}
                className={`px-1.5 py-0.5 rounded ${
                  mediaFilter === 'all' ? 'bg-slate-800 text-white' : 'text-slate-400'
                }`}
              >
                All ({project.assets.length})
              </button>
              <button
                type="button"
                data-testid="media-filter-raw"
                onClick={() => setMediaFilter('raw')}
                className={`px-1.5 py-0.5 rounded ${
                  mediaFilter === 'raw' ? 'bg-slate-800 text-amber-300' : 'text-slate-400'
                }`}
              >
                🔒 Raw ({project.assets.filter((a) => a.asset.kind === 'raw').length})
              </button>
              <button
                type="button"
                data-testid="media-filter-derived"
                onClick={() => setMediaFilter('derived')}
                className={`px-1.5 py-0.5 rounded ${
                  mediaFilter === 'derived' ? 'bg-slate-800 text-violet-300' : 'text-slate-400'
                }`}
              >
                ✨ Derived ({project.assets.filter((a) => a.asset.kind === 'derived').length})
              </button>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/webm"
              data-testid="video-file-upload-input"
              onChange={(e) => void handleUploadVideoFile(e)}
              className="hidden"
            />
            <button
              type="button"
              data-testid="video-upload-mp4-btn"
              disabled={busyAction}
              onClick={() => fileInputRef.current?.click()}
              className="px-2 py-1 rounded bg-emerald-600/25 hover:bg-emerald-600/40 text-emerald-200 border border-emerald-500/40 text-[11px]"
              title="Upload a real .mp4 video into the immutable vault"
            >
              📤 Upload .mp4
            </button>
            <button
              type="button"
              data-testid="video-ingest-btn"
              disabled={busyAction}
              onClick={handleIngestSampleMedia}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-cyan-500/30 text-[11px]"
            >
              + Ingest Immutable Raw Video
            </button>
          </div>
        </div>

        {/* Asset Cards with Lineage */}
        <div
          data-testid="video-media-pool-list"
          className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 max-h-36 overflow-y-auto"
        >
          {filteredAssets.map(({ asset, end_timecode }) => (
            <div
              key={asset.asset_id}
              data-testid={`video-asset-card-${asset.asset_id}`}
              className="p-2 rounded bg-slate-950/90 border border-slate-800 flex flex-col justify-between text-[11px]"
            >
              <div className="flex items-center justify-between gap-1">
                <span className="font-mono font-medium text-slate-200 truncate">
                  {asset.kind === 'raw' ? '🔒' : '✨'} {asset.asset_id}
                </span>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    data-testid={`video-append-asset-${asset.asset_id}`}
                    disabled={busyAction}
                    onClick={() =>
                      handleAppendToTimeline(
                        asset.asset_id,
                        0,
                        Math.min(150, asset.total_frames),
                        asset.filename
                      )
                    }
                    className="px-1.5 py-0.5 rounded bg-cyan-600/25 hover:bg-cyan-600/45 border border-cyan-500/40 text-cyan-200 text-[9px] font-mono"
                    title="Append this asset onto the active video track"
                  >
                    + Append
                  </button>
                  <span
                    className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${
                      asset.kind === 'raw'
                        ? 'bg-amber-500/15 text-amber-300'
                        : 'bg-violet-500/20 text-violet-300'
                    }`}
                  >
                    {asset.kind.toUpperCase()} • {asset.tiers.source.width}x
                    {asset.tiers.source.height}
                  </span>
                </div>
              </div>
              <div className="text-[10px] text-slate-400 font-mono mt-0.5 flex items-center justify-between">
                <span>
                  {end_timecode} ({asset.total_frames}f)
                </span>
                <span>SHA {asset.tiers.source.sha256.slice(0, 8)}</span>
              </div>
              {asset.lineage && (
                <div
                  data-testid={`video-asset-lineage-${asset.asset_id}`}
                  className="mt-1 pt-1 border-t border-slate-800/80 text-[10px] text-violet-300 font-mono truncate"
                >
                  ↳ Derived from {asset.lineage.parent_asset_ids.join(', ')} [F
                  {asset.lineage.source_in_frame}..{asset.lineage.source_out_frame}] via{' '}
                  {asset.lineage.operations.map((o) => o.op).join('+')}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Frame-Level Semantic Search Form */}
        <form onSubmit={(e) => void handleSearchFrames(e)} className="flex items-center gap-1.5">
          <input
            type="text"
            data-testid="video-frame-search-input"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search frames semantically (e.g. golden hour aerial, close-up smile, neon street)..."
            className="flex-1 bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200"
          />
          <button
            type="submit"
            data-testid="video-frame-search-btn"
            disabled={searching}
            className="px-2.5 py-1 rounded bg-cyan-600/80 hover:bg-cyan-500 text-white text-xs font-medium"
          >
            {searching ? '...' : 'Search Frames'}
          </button>
        </form>

        {searchRanking && (
          <div
            data-testid="video-frame-search-results"
            className="p-2 rounded bg-slate-950 border border-slate-800 space-y-1.5 text-[11px]"
          >
            <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
              <span>
                Query: &ldquo;{searchRanking.query}&rdquo; ({searchRanking.hits.length} shot hits)
              </span>
              <span data-testid="video-search-method-badge" className="text-cyan-300">
                Method: {searchRanking.method}
              </span>
            </div>
            {searchRanking.hits.length === 0 ? (
              <div className="text-slate-500 text-[11px]">No matching frame ranges found.</div>
            ) : (
              <div className="space-y-1">
                {searchRanking.hits.map((hit: FrameRangeHit) => (
                  <div
                    key={hit.frame_id}
                    data-testid={`video-search-hit-${hit.frame_id}`}
                    className="p-1.5 rounded bg-slate-900 border border-slate-800 flex items-center justify-between gap-2"
                  >
                    <div className="min-w-0">
                      <div className="font-mono text-[10px] text-cyan-300">
                        {hit.asset_id} • {hit.in_timecode}–{hit.out_timecode} (F{hit.in_frame}..
                        {hit.out_frame}) • {hit.shot_type} • score {hit.score.toFixed(2)}
                      </div>
                      <div className="text-slate-300 truncate">{hit.caption}</div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <button
                        type="button"
                        onClick={() => setPlayheadFrame(hit.in_frame)}
                        className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-[10px] font-mono text-amber-300"
                      >
                        Cue F{hit.in_frame}
                      </button>
                      <button
                        type="button"
                        data-testid={`video-insert-hit-${hit.frame_id}`}
                        disabled={busyAction}
                        onClick={() =>
                          handleAppendToTimeline(
                            hit.asset_id,
                            hit.in_frame,
                            hit.out_frame,
                            hit.caption
                          )
                        }
                        className="px-2 py-0.5 rounded bg-cyan-600/30 hover:bg-cyan-600/50 border border-cyan-500/40 text-[10px] font-mono text-cyan-200"
                        title="Insert this semantic shot range onto the active video track"
                      >
                        + Insert Shot
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Multi-Track NLE Timeline (DaVinci Resolve Edit Page Timeline Bar) */}
      <div
        data-testid="video-timeline-bar"
        className="p-2.5 space-y-1.5 flex-1 min-h-0 bg-[#14171d] border-t border-slate-800"
      >
        {/* DaVinci Resolve Top NLE Edit Toolbar */}
        <div
          data-testid="video-nle-toolbar"
          className="px-2.5 py-1.5 rounded-t bg-[#1c2029] border border-slate-800 flex flex-wrap items-center justify-between gap-2 text-[11px]"
        >
          {/* Left: Pointer / Trim / Blade Mode + Insert/Append + Snap/Link/Marker */}
          <div className="flex flex-wrap items-center gap-1">
            <div className="flex items-center bg-[#11141a] p-0.5 rounded border border-slate-800">
              <button
                type="button"
                data-testid="nle-tool-select"
                onClick={() => setTimelineTool('select')}
                className={`px-2 py-0.5 rounded text-[10px] font-medium flex items-center gap-1 ${
                  timelineTool === 'select'
                    ? 'bg-rose-600/30 text-rose-300 border border-rose-500/50'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title="Selection Mode (A) — Select and cue clips"
              >
                <span>↖</span>
                <span>Select</span>
              </button>
              <button
                type="button"
                data-testid="nle-tool-trim"
                onClick={() => setTimelineTool('trim')}
                className={`px-2 py-0.5 rounded text-[10px] font-medium flex items-center gap-1 ${
                  timelineTool === 'trim'
                    ? 'bg-amber-600/30 text-amber-300 border border-amber-500/50'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title="Trim Edit Mode (T) — Ripple trim clip boundaries"
              >
                <span>[|]</span>
                <span>Trim</span>
              </button>
              <button
                type="button"
                data-testid="nle-tool-blade"
                onClick={() => setTimelineTool('blade')}
                className={`px-2 py-0.5 rounded text-[10px] font-medium flex items-center gap-1 ${
                  timelineTool === 'blade'
                    ? 'bg-rose-600/40 text-rose-200 border border-rose-500/60'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title="Blade / Razor Edit Mode (B) — Click any clip on the timeline bar to split at that frame"
              >
                <Scissors className="w-3 h-3" />
                <span>Blade</span>
              </button>
            </div>

            <div className="h-4 w-px bg-slate-800 mx-0.5" />

            <button
              type="button"
              data-testid="nle-toggle-snap"
              onClick={() => setSnapEnabled((s) => !s)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${
                snapEnabled
                  ? 'bg-cyan-600/20 text-cyan-300 border-cyan-500/40'
                  : 'bg-slate-900 text-slate-500 border-slate-800'
              }`}
              title="Toggle Timeline Snapping (N)"
            >
              🧲 Snap
            </button>
            <button
              type="button"
              data-testid="nle-toggle-link"
              onClick={() => setLinkedSelection((l) => !l)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${
                linkedSelection
                  ? 'bg-cyan-600/20 text-cyan-300 border-cyan-500/40'
                  : 'bg-slate-900 text-slate-500 border-slate-800'
              }`}
              title="Toggle Linked Selection (Ctrl+Shift+L)"
            >
              🔗 Link
            </button>

            {project.sections.length > 0 && (
              <div className="flex items-center gap-1 ml-1">
                {project.sections.map((sec) => (
                  <button
                    key={sec.id}
                    type="button"
                    onClick={() => setPlayheadFrame(sec.start_frame)}
                    className="px-1.5 py-0.5 rounded bg-emerald-950/60 hover:bg-emerald-900/70 border border-emerald-600/40 text-[9px] font-mono text-emerald-300"
                    title={`Marker: ${sec.title} (F${sec.start_frame}–${sec.end_frame})`}
                  >
                    🟢 {sec.title}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Right: Scrubber + DaVinci Zoom Slider (− [━━●━━] +) */}
          <div className="flex items-center gap-2.5">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-mono text-slate-400">Scrub</span>
              <input
                type="range"
                data-testid="video-playhead-scrubber"
                min={0}
                max={Math.max(1, totalFrames - 1)}
                value={playheadFrame}
                onChange={(e) => setPlayheadFrame(Number(e.target.value))}
                className="w-28 accent-rose-500"
              />
              <span className="text-[10px] font-mono text-amber-300">
                F{playheadFrame}/{totalFrames}
              </span>
            </div>

            <div className="h-4 w-px bg-slate-800" />

            <div className="flex items-center gap-1 text-[10px] font-mono text-slate-400">
              <span>🔍−</span>
              <input
                type="range"
                data-testid="video-timeline-zoom"
                min={1}
                max={2.5}
                step={0.25}
                value={timelineZoom}
                onChange={(e) => setTimelineZoom(Number(e.target.value))}
                className="w-16 accent-slate-300"
                title={`Timeline Zoom: ${Math.round(timelineZoom * 100)}%`}
              />
              <span>+</span>
            </div>
          </div>
        </div>

        {/* DaVinci Horizontal Timeline Container: Timecode Ruler + Playhead + Filmstrip Track Bars */}
        <div
          data-testid="video-timeline-tracks"
          className="rounded-b border border-slate-800 bg-[#181b22] overflow-x-auto"
        >
          <div style={{ minWidth: `${Math.round(timelineZoom * 100)}%` }}>
            {/* Top SMPTE Timecode Ruler Bar with Fine Graduation Ticks & Marker Flags */}
            <div className="flex border-b border-slate-800 bg-[#12151b] text-[10px] font-mono text-slate-400">
              <div className="w-36 shrink-0 px-2.5 py-1.5 border-r border-slate-800 flex items-center justify-between bg-[#161920]">
                <span className="text-[11px] font-bold text-slate-200">{frameComposition.timecode}</span>
                <span className="text-[9px] text-rose-400 uppercase">{timelineTool}</span>
              </div>
              <div
                data-testid="video-timeline-ruler"
                onPointerDown={(e) => {
                  isScrubbingRef.current = true;
                  e.currentTarget.setPointerCapture?.(e.pointerId);
                  scrubToClientX(e.clientX, e.currentTarget.getBoundingClientRect());
                }}
                onPointerMove={(e) => {
                  if (!isScrubbingRef.current) return;
                  scrubToClientX(e.clientX, e.currentTarget.getBoundingClientRect());
                }}
                onPointerUp={(e) => {
                  isScrubbingRef.current = false;
                  e.currentTarget.releasePointerCapture?.(e.pointerId);
                }}
                onClick={(e) => {
                  scrubToClientX(e.clientX, e.currentTarget.getBoundingClientRect());
                }}
                className="flex-1 relative h-8 cursor-pointer select-none overflow-hidden bg-[#14171e]"
                title="Click or drag anywhere on the SMPTE timecode ruler to scrub the playhead"
              >
                {/* Fine Top Tick Marks (DaVinci Ruler Graduation) */}
                {Array.from({ length: 41 }).map((_, idx) => {
                  const pct = (idx / 40) * 100;
                  const isMajor = idx % 10 === 0;
                  const isMedium = idx % 5 === 0;
                  return (
                    <div
                      key={idx}
                      style={{ left: `${pct}%` }}
                      className={`absolute top-0 w-px pointer-events-none ${
                        isMajor
                          ? 'h-3 bg-slate-500'
                          : isMedium
                            ? 'h-2 bg-slate-600/80'
                            : 'h-1.5 bg-slate-700/60'
                      }`}
                    />
                  );
                })}

                {/* Major SMPTE Timecode Stamps (01:00:00:00 style) */}
                {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
                  const tickFrame = Math.min(totalFrames, Math.round(pct * totalFrames));
                  return (
                    <div
                      key={pct}
                      style={{
                        left: `${pct * 100}%`,
                        transform: pct === 1 ? 'translateX(-100%)' : undefined,
                      }}
                      className="absolute top-2.5 bottom-0 px-1 pointer-events-none flex items-center"
                    >
                      <span className="text-[9px] text-slate-400 tracking-tight">
                        {framesToTimecode(tickFrame, project.fps)}
                      </span>
                    </div>
                  );
                })}

                {/* Green Section Marker Shields on Ruler */}
                {project.sections.map((sec) => {
                  const markerPct = Math.min(98, Math.max(1, (sec.start_frame / totalFrames) * 100));
                  return (
                    <div
                      key={sec.id}
                      style={{ left: `${markerPct}%` }}
                      className="absolute bottom-0.5 -ml-1.5 w-2.5 h-2.5 rounded-t-sm bg-emerald-500 border border-emerald-300 pointer-events-none"
                      title={sec.title}
                    />
                  );
                })}

                {/* DaVinci Red Playhead Cap on Ruler */}
                <div
                  data-testid="video-playhead-line"
                  style={{
                    left: `${Math.min(100, Math.max(0, (playheadFrame / totalFrames) * 100))}%`,
                  }}
                  className="absolute top-0 bottom-0 w-0.5 bg-rose-500 pointer-events-none z-30"
                >
                  <div className="w-3 h-2.5 -ml-[5px] bg-rose-500 rounded-t-sm shadow" />
                </div>
              </div>
            </div>

            {/* DaVinci Horizontal Filmstrip Track Lanes */}
            <div className="divide-y divide-slate-800/90">
              {activeBundle.tracks.map((track) => {
                const rawTrack = project.tracks.find((t) => t.track_id === track.track_id);
                const playheadPct = Math.min(
                  100,
                  Math.max(0, (playheadFrame / totalFrames) * 100)
                );
                return (
                  <div
                    key={track.track_id}
                    data-testid={`video-track-${track.track_id}`}
                    className="flex items-stretch min-h-[68px] bg-[#181b22]"
                  >
                    {/* Left DaVinci Track Patch Header Column */}
                    <div className="w-36 shrink-0 px-2 py-1.5 bg-[#1d212a] border-r border-slate-800 flex flex-col justify-between text-[11px]">
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-mono font-semibold text-cyan-300 truncate">
                          {track.track_id}
                        </span>
                        <span className="text-slate-400 text-[9px] uppercase font-mono">
                          {track.kind}
                        </span>
                      </div>
                      <div className="flex items-center justify-between gap-1 mt-1">
                        {rawTrack && (
                          <div className="flex items-center gap-0.5">
                            <button
                              type="button"
                              data-testid={`video-track-mute-${track.track_id}`}
                              disabled={busyAction}
                              onClick={() => handleToggleTrackState(rawTrack, 'muted')}
                              className={`px-1.5 py-0.5 rounded text-[9px] font-mono ${
                                rawTrack.muted
                                  ? 'bg-rose-600/50 text-rose-100 border border-rose-400/60'
                                  : 'bg-[#12151b] text-slate-400 hover:text-slate-200 border border-slate-800'
                              }`}
                              title="Mute track"
                            >
                              M
                            </button>
                            <button
                              type="button"
                              data-testid={`video-track-solo-${track.track_id}`}
                              disabled={busyAction}
                              onClick={() => handleToggleTrackState(rawTrack, 'solo')}
                              className={`px-1.5 py-0.5 rounded text-[9px] font-mono ${
                                rawTrack.solo
                                  ? 'bg-amber-600/50 text-amber-100 border border-amber-400/60'
                                  : 'bg-[#12151b] text-slate-400 hover:text-slate-200 border border-slate-800'
                              }`}
                              title="Solo track"
                            >
                              S
                            </button>
                            <button
                              type="button"
                              data-testid={`video-track-lock-${track.track_id}`}
                              disabled={busyAction}
                              onClick={() => handleToggleTrackState(rawTrack, 'locked')}
                              className={`px-1.5 py-0.5 rounded text-[9px] font-mono ${
                                rawTrack.locked
                                  ? 'bg-violet-600/50 text-violet-100 border border-violet-400/60'
                                  : 'bg-[#12151b] text-slate-400 hover:text-slate-200 border border-slate-800'
                              }`}
                              title="Lock track"
                            >
                              🔒
                            </button>
                          </div>
                        )}
                        {rawTrack?.can_revert && (
                          <button
                            type="button"
                            data-testid={`video-revert-${track.track_id}`}
                            disabled={busyAction}
                            onClick={() => handleRevertTrack(rawTrack)}
                            className="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-[9px] text-amber-300"
                          >
                            Undo
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Right Proportional DaVinci Filmstrip Timeline Bar */}
                    <div
                      data-testid={`video-track-bar-${track.track_id}`}
                      onClick={(e) => {
                        scrubToClientX(e.clientX, e.currentTarget.getBoundingClientRect());
                      }}
                      className={`flex-1 relative min-h-[68px] bg-[#14171e] overflow-hidden ${
                        timelineTool === 'blade'
                          ? 'cursor-crosshair'
                          : timelineTool === 'trim'
                            ? 'cursor-ew-resize'
                            : 'cursor-pointer'
                      }`}
                    >
                      {/* Vertical Quarter Grid Lines */}
                      {[0.25, 0.5, 0.75].map((pct) => (
                        <div
                          key={pct}
                          style={{ left: `${pct * 100}%` }}
                          className="absolute top-0 bottom-0 w-px bg-slate-800/50 pointer-events-none"
                        />
                      ))}

                      {/* DaVinci Filmstrip Video & Waveform Audio Clips */}
                      {track.clips.map((clip) => {
                        const isSelected = selectedClipId === clip.clip_id;
                        const leftPct = Math.max(
                          0,
                          Math.min(96, (clip.timeline_in_frame / totalFrames) * 100)
                        );
                        const widthPct = Math.max(
                          7,
                          Math.min(
                            100 - leftPct,
                            ((clip.timeline_out_frame - clip.timeline_in_frame) / totalFrames) *
                              100
                          )
                        );
                        const isDerived = clip.asset_kind === 'derived';
                        const isAudio = track.kind === 'audio';
                        const isInterviewThumb =
                          clip.asset_id.includes('interview') || clip.asset_id.includes('cam');

                        return (
                          <button
                            key={clip.clip_id}
                            type="button"
                            data-testid={`video-clip-${clip.clip_id}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              if (timelineTool === 'blade' && rawTrack) {
                                const rect = e.currentTarget.getBoundingClientRect();
                                const clickRatio =
                                  rect.width > 0
                                    ? Math.max(
                                        0.15,
                                        Math.min(0.85, (e.clientX - rect.left) / rect.width)
                                      )
                                    : 0.5;
                                const splitFrame = Math.round(
                                  clip.timeline_in_frame +
                                    (clip.timeline_out_frame - clip.timeline_in_frame) *
                                      clickRatio
                                );
                                if (
                                  splitFrame > clip.timeline_in_frame &&
                                  splitFrame < clip.timeline_out_frame
                                ) {
                                  void runMutation(
                                    `/tracks/${encodeURIComponent(rawTrack.track_id)}/edit`,
                                    {
                                      action: 'split_clip',
                                      seen_digest: rawTrack.digest,
                                      clip_id: clip.clip_id,
                                      split_at_frame: splitFrame,
                                    }
                                  );
                                  return;
                                }
                              }
                              if (timelineTool === 'trim' && rawTrack) {
                                const rect = e.currentTarget.getBoundingClientRect();
                                const clickRatio =
                                  rect.width > 0
                                    ? Math.max(
                                        0.1,
                                        Math.min(0.9, (e.clientX - rect.left) / rect.width)
                                      )
                                    : 0.75;
                                const speed = clip.speed || 1.0;
                                if (clickRatio < 0.5) {
                                  const deltaFrames = Math.max(
                                    1,
                                    Math.round(clip.duration_frames * clickRatio * 0.5 * speed)
                                  );
                                  void runMutation(
                                    `/tracks/${encodeURIComponent(rawTrack.track_id)}/edit`,
                                    {
                                      action: 'trim_clip',
                                      seen_digest: rawTrack.digest,
                                      clip_id: clip.clip_id,
                                      source_in_frame: Math.min(
                                        clip.source_out_frame - 1,
                                        clip.source_in_frame + deltaFrames
                                      ),
                                      ripple: true,
                                    }
                                  );
                                } else {
                                  const keepFrames = Math.max(
                                    1,
                                    Math.round(clip.duration_frames * clickRatio * speed)
                                  );
                                  void runMutation(
                                    `/tracks/${encodeURIComponent(rawTrack.track_id)}/edit`,
                                    {
                                      action: 'trim_clip',
                                      seen_digest: rawTrack.digest,
                                      clip_id: clip.clip_id,
                                      source_out_frame: Math.max(
                                        clip.source_in_frame + 1,
                                        clip.source_in_frame + keepFrames
                                      ),
                                      ripple: true,
                                    }
                                  );
                                }
                                return;
                              }
                              handleSelectClip(track.track_id, clip);
                            }}
                            style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                            className={`absolute top-1 bottom-1 rounded-sm text-left transition-all overflow-hidden flex flex-col justify-between select-none ${
                              isSelected
                                ? 'ring-2 ring-rose-500 border border-rose-400 z-10 shadow-lg'
                                : 'border border-slate-950/90 hover:brightness-110'
                            }`}
                            title={`${clip.clip_id} (${clip.asset_id}.mp4) • Timeline F${clip.timeline_in_frame}–${clip.timeline_out_frame} • Source F${clip.source_in_frame}–${clip.source_out_frame}`}
                          >
                            {isAudio ? (
                              /* DaVinci Forest-Green Audio Clip Bar with Waveform */
                              <>
                                <div className="flex-1 w-full bg-[#387556] relative flex items-center px-1.5 overflow-hidden">
                                  {/* 0.0 dB Unity Gain Line */}
                                  <div className="absolute inset-x-0 top-1/2 h-px bg-white/30 pointer-events-none" />
                                  {/* Simulated Stereo Waveform Bars */}
                                  <div className="w-full flex items-center justify-between gap-0.5 h-7 opacity-80">
                                    {Array.from({ length: 28 }).map((_, wIdx) => {
                                      const barPct =
                                        25 + ((Math.abs((wIdx * 19 + clip.source_in_frame) % 7)) * 10);
                                      return (
                                        <div
                                          key={wIdx}
                                          style={{ height: `${barPct}%` }}
                                          className="flex-1 bg-[#74c69d] rounded-full"
                                        />
                                      );
                                    })}
                                  </div>
                                </div>
                                <div className="h-4 px-1.5 bg-[#295940] border-t border-black/30 flex items-center justify-between text-[9px] font-mono text-emerald-50 truncate w-full">
                                  <span className="truncate">
                                    🔗 {clip.asset_id}.mp4 ({clip.clip_id})
                                  </span>
                                  <span className="shrink-0 text-[8px] text-emerald-200">
                                    {clip.volume}%
                                  </span>
                                </div>
                              </>
                            ) : (
                              /* DaVinci Steel-Blue Video Filmstrip Bar with Head & Tail Thumbnails */
                              <>
                                <div
                                  className={`flex-1 w-full flex items-stretch overflow-hidden relative ${
                                    isDerived ? 'bg-[#52438f]' : 'bg-[#3b6998]'
                                  }`}
                                >
                                  {/* Left (Head) Frame Thumbnail Preview */}
                                  <div
                                    className="w-14 shrink-0 h-full border-r border-black/40 relative overflow-hidden flex flex-col justify-between p-1"
                                    style={{
                                      background: isInterviewThumb
                                        ? 'linear-gradient(135deg, #4a2c1d 0%, #8c583a 55%, #1e293b 100%)'
                                        : 'linear-gradient(135deg, #0f172a 0%, #c2410c 55%, #38bdf8 100%)',
                                    }}
                                  >
                                    <div className="flex items-center justify-between">
                                      <span className="w-2 h-2 rounded-full bg-amber-300/90 shadow" />
                                      <span className="text-[7px] font-mono text-white/90 bg-black/50 px-0.5 rounded">
                                        IN
                                      </span>
                                    </div>
                                    <span className="text-[7px] font-mono text-slate-100 bg-black/60 px-0.5 rounded truncate">
                                      F{clip.source_in_frame}
                                    </span>
                                  </div>

                                  {/* Center Steel-Blue Body (pointed to by arrows in DaVinci UI) */}
                                  <div className="flex-1 px-2 py-1 flex items-center justify-between min-w-0 overflow-hidden">
                                    <div className="min-w-0 truncate">
                                      <span className="text-[9px] font-mono font-semibold text-white/95 truncate block">
                                        {isDerived ? '✨ DERIVED BAKE' : '🎬 VIDEO CLIP'}
                                      </span>
                                      <span className="text-[8px] font-mono text-sky-100/80 truncate block">
                                        F{clip.timeline_in_frame}–{clip.timeline_out_frame} (
                                        {framesToTimecode(clip.timeline_in_frame, project.fps)})
                                      </span>
                                    </div>
                                    <div className="flex items-center gap-1 shrink-0">
                                      <span className="text-[8px] font-mono px-1 py-0.5 rounded bg-black/35 text-cyan-200">
                                        {clip.color_grade.lut}
                                      </span>
                                      {clip.transition_out && (
                                        <span className="text-[8px] font-mono px-1 py-0.5 rounded bg-amber-500/30 text-amber-200">
                                          ⇥ {clip.transition_out.kind}
                                        </span>
                                      )}
                                    </div>
                                  </div>

                                  {/* Right (Tail) Frame Thumbnail Preview */}
                                  <div
                                    className="w-14 shrink-0 h-full border-l border-black/40 relative overflow-hidden hidden sm:flex flex-col justify-between p-1"
                                    style={{
                                      background: isInterviewThumb
                                        ? 'linear-gradient(225deg, #3b2316 0%, #a16207 50%, #0f172a 100%)'
                                        : 'linear-gradient(225deg, #1e1b4b 0%, #0284c7 55%, #f59e0b 100%)',
                                    }}
                                  >
                                    <div className="flex items-center justify-between">
                                      <span className="text-[7px] font-mono text-white/90 bg-black/50 px-0.5 rounded">
                                        OUT
                                      </span>
                                      <span className="w-2 h-2 rounded-full bg-sky-300/90 shadow" />
                                    </div>
                                    <span className="text-[7px] font-mono text-slate-100 bg-black/60 px-0.5 rounded truncate text-right">
                                      F{clip.source_out_frame}
                                    </span>
                                  </div>
                                </div>

                                {/* Bottom DaVinci Clip Label Ribbon (`🔗 AA0113_01.mov`) */}
                                <div
                                  className={`h-4 px-1.5 border-t border-black/35 flex items-center justify-between text-[9px] font-mono text-white truncate w-full ${
                                    isDerived ? 'bg-[#3d316e]' : 'bg-[#2b527a]'
                                  }`}
                                >
                                  <span className="truncate">
                                    🔗 {clip.asset_id}.mp4 • {clip.clip_id}
                                  </span>
                                  <span className="shrink-0 text-[8px] text-sky-200/90">
                                    {clip.duration_frames}f
                                  </span>
                                </div>
                              </>
                            )}
                          </button>
                        );
                      })}

                      {/* Proportional Title Clip Blocks */}
                      {track.titles.map((t) => {
                        const leftPct = Math.max(
                          0,
                          Math.min(96, (t.timeline_in_frame / totalFrames) * 100)
                        );
                        const widthPct = Math.max(
                          8,
                          Math.min(
                            100 - leftPct,
                            ((t.timeline_out_frame - t.timeline_in_frame) / totalFrames) * 100
                          )
                        );
                        return (
                          <div
                            key={t.title_id}
                            data-testid={`video-title-chip-${t.title_id}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              setPlayheadFrame(t.timeline_in_frame);
                            }}
                            style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                            className="absolute top-2 bottom-2 px-2 py-1 rounded-sm text-left border border-amber-400/70 bg-[#7c4a1a] hover:bg-[#8f5620] text-amber-50 overflow-hidden flex flex-col justify-between shadow cursor-pointer"
                            title={`${t.clip.text} (F${t.timeline_in_frame}–${t.timeline_out_frame})`}
                          >
                            <div className="flex items-center justify-between gap-1">
                              <div className="font-mono text-[10px] font-semibold truncate">
                                🔤 {t.clip.text}
                              </div>
                              <button
                                type="button"
                                data-testid={`video-delete-title-${t.title_id}`}
                                disabled={busyAction}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteTitleOverlay(track.track_id, t.title_id);
                                }}
                                className="px-1 rounded bg-black/40 hover:bg-rose-600/80 text-[8px] font-mono text-amber-100 shrink-0"
                                title="Delete title overlay"
                              >
                                ×
                              </button>
                            </div>
                            <div className="text-[8px] font-mono text-amber-200/90 truncate">
                              F{t.timeline_in_frame}–{t.timeline_out_frame} ({t.clip.style_preset})
                            </div>
                          </div>
                        );
                      })}

                      {track.clips.length === 0 && track.titles.length === 0 && (
                        <span className="px-3 py-5 text-[10px] text-slate-600 italic block">
                          Empty track bar
                        </span>
                      )}

                      {/* Continuous Vertical Red Playhead Line Across Track Bar */}
                      <div
                        style={{ left: `${playheadPct}%` }}
                        className="absolute top-0 bottom-0 w-0.5 bg-rose-500 pointer-events-none z-30"
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
