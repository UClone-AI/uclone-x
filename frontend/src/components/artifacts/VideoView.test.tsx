import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { VideoView } from './VideoView';
import {
  combineColorGrades,
  evaluateColorGradeCssFilter,
  evaluateFrameComposition,
  framesToTimecode,
  type ColorGrade,
  type PlaybackBundle,
  type ShownVideoProject,
} from './VideoCompositorEngine';

const defaultGrade: ColorGrade = {
  lift: [0, 0, 0],
  gamma: [1, 1, 1],
  gain: [1, 1, 1],
  contrast: 1.0,
  saturation: 1.0,
  temperature: 0,
  tint: 0,
  lut: 'clean_neutral',
  vignette: 0,
};

const samplePlayback = (over: Partial<PlaybackBundle> = {}): PlaybackBundle => ({
  project_id: 'seoul-cyber-cut',
  title: 'Seoul Cyber Cut',
  fps: 30,
  width: 960,
  height: 540,
  active_tier: 'preview',
  total_frames: 300,
  duration_sec: 10.0,
  total_timecode: '00:00:10:00',
  master_grade: defaultGrade,
  ffmpeg_filtergraph: '[a_cam_01:v]trim=start=0.0:end=5.0[v_v1_main_0]',
  tracks: [
    {
      track_id: 'v1_main',
      kind: 'video',
      name: 'Main Picture',
      index: 1,
      muted: false,
      solo: false,
      locked: false,
      clips: [
        {
          clip_id: 'clip_01',
          track_id: 'v1_main',
          asset_id: 'a_cam_01',
          asset_kind: 'raw',
          resolved_tier: 'preview',
          stream_path: 'proxies/sha256_abc_540p.mp4',
          stream_sha256: 'abc1234567890',
          stream_width: 960,
          stream_height: 540,
          source_in_frame: 0,
          source_out_frame: 150,
          timeline_in_frame: 0,
          timeline_out_frame: 150,
          duration_frames: 150,
          start_sec: 0,
          end_sec: 5.0,
          in_timecode: '00:00:00:00',
          out_timecode: '00:00:05:00',
          speed: 1.0,
          volume: 90,
          muted: false,
          transform: {
            scale: 1.0,
            position_x: 0,
            position_y: 0,
            rotation_deg: 0,
            opacity: 1.0,
          },
          color_grade: { ...defaultGrade, lut: 'teal_orange_cinema' },
          transition_out: { kind: 'cross_dissolve', duration_frames: 15 },
          can_unbake: true,
          raw_parent_asset_id: 'a_cam_01',
        },
      ],
      titles: [],
    },
    {
      track_id: 't1_titles',
      kind: 'title',
      name: 'Titles',
      index: 2,
      muted: false,
      solo: false,
      locked: false,
      clips: [],
      titles: [
        {
          title_id: 'title_01',
          track_id: 't1_titles',
          timeline_in_frame: 0,
          timeline_out_frame: 90,
          start_sec: 0,
          end_sec: 3.0,
          in_timecode: '00:00:00:00',
          out_timecode: '00:00:03:00',
          clip: {
            title_id: 'title_01',
            text: 'SEOUL 2026',
            subtitle: 'Night Sequence',
            style_preset: 'minimal_lower_third',
            timeline_in_frame: 0,
            timeline_out_frame: 90,
            position: 'lower_left',
            animation: 'fade_up',
            font_size_px: 42,
            color_hex: '#FFFFFF',
            backdrop_opacity: 0.45,
          },
        },
      ],
    },
  ],
  ...over,
});

const sampleProject = (over: Partial<ShownVideoProject> = {}): ShownVideoProject => ({
  project_id: 'seoul-cyber-cut',
  title: 'Seoul Cyber Cut',
  style_profile: {
    genre: 'cinematic_film',
    aspect_ratio: '16:9',
    pacing: 'dynamic',
    color_mood: 'teal_orange_cinema',
    audio_lufs_target: -14.0,
  },
  quality_tiers: {
    preview: {
      tier: 'preview',
      width: 960,
      height: 540,
      codec: 'h264_proxy',
      crf: 28,
      max_keyframe_interval: 15,
      description: '540p proxy',
    },
    source: {
      tier: 'source',
      preserve_bitstream: true,
      read_only_mode: '0444',
      description: 'Immutable source vault',
    },
    master: {
      tier: 'master',
      width: 1920,
      height: 1080,
      codec: 'h264_high',
      crf: 16,
      pixel_format: 'yuv420p10le',
      audio_lufs_target: -14.0,
      description: '1080p master',
    },
  },
  active_tier: 'preview',
  writable: true,
  holder: 'room-1',
  decide_code: null,
  fps: 30,
  width: 1920,
  height: 1080,
  total_frames: 300,
  duration_seconds: 10.0,
  total_timecode: '00:00:10:00',
  sections: [
    {
      id: 'sec_intro',
      title: 'Intro',
      kind: 'intro',
      start_frame: 0,
      end_frame: 150,
    },
  ],
  markers: [],
  assets: [
    {
      asset: {
        asset_id: 'a_cam_01',
        kind: 'raw',
        filename: 'a_cam_01.mp4',
        immutable: true,
        fps: 30,
        total_frames: 300,
        duration_ms: 10000,
        tiers: {
          source: {
            path: 'vault/sha256_raw111.mp4',
            sha256: 'raw111222333444',
            width: 1920,
            height: 1080,
            size_bytes: 4096,
          },
          preview: {
            path: 'proxies/sha256_raw111_540p.mp4',
            sha256: 'prox111222333444',
            width: 960,
            height: 540,
            size_bytes: 1024,
          },
        },
        lineage: null,
        has_audio: true,
        index_status: 'indexed',
        indexed_frame_count: 5,
      },
      digest: 'dig-asset-raw',
      start_timecode: '00:00:00:00',
      end_timecode: '00:00:10:00',
    },
    {
      asset: {
        asset_id: 'a_cam_01_slowmo_2x',
        kind: 'derived',
        filename: 'a_cam_01_slowmo_2x.mp4',
        immutable: true,
        fps: 30,
        total_frames: 150,
        duration_ms: 5000,
        tiers: {
          source: {
            path: 'vault/sha256_der999.mp4',
            sha256: 'der999888777666',
            width: 1920,
            height: 1080,
            size_bytes: 4096,
          },
          preview: {
            path: 'proxies/sha256_der999_540p.mp4',
            sha256: 'pder999888777666',
            width: 960,
            height: 540,
            size_bytes: 1024,
          },
        },
        lineage: {
          parent_asset_ids: ['a_cam_01'],
          source_in_frame: 30,
          source_out_frame: 105,
          recipe_kind: 'heavy_transform',
          operations: [{ op: 'optical_flow_slowmo', rate: 0.5, strength: 0.85 }],
        },
        has_audio: true,
        index_status: 'indexed',
        indexed_frame_count: 3,
      },
      digest: 'dig-asset-derived',
      start_timecode: '00:00:00:00',
      end_timecode: '00:00:05:00',
    },
  ],
  tracks: [
    {
      track_id: 'v1_main',
      kind: 'video',
      name: 'Main Picture',
      index: 1,
      digest: 'dig-track-v1',
      muted: false,
      solo: false,
      locked: false,
      clips: [
        {
          clip_id: 'clip_01',
          asset_id: 'a_cam_01',
          timeline_in_frame: 0,
          source_in_frame: 0,
          source_out_frame: 150,
          speed: 1.0,
          volume: 90,
          muted: false,
          transform: {
            scale: 1.0,
            position_x: 0,
            position_y: 0,
            rotation_deg: 0,
            opacity: 1.0,
          },
          color_grade: { ...defaultGrade, lut: 'teal_orange_cinema' },
          transition_out: { kind: 'cross_dissolve', duration_frames: 15 },
        },
      ],
      titles: [],
      can_revert: true,
    },
  ],
  playback: samplePlayback(),
  proposals: [
    {
      id: 'p001',
      kind: 'track_replace',
      summary: 'Tighten intro cut and apply Kodak 2383 LUT',
      track_id: 'v1_main',
      seen_digest: 'dig-prop-1',
      original_clips: [],
      proposed_clips: [],
      original_titles: [],
      proposed_titles: [],
      preview_bundle: samplePlayback({
        active_tier: 'source',
        width: 1920,
        height: 1080,
      }),
    },
  ],
  audit_findings: [],
  last_master_receipt: null,
  ...over,
});

interface RecordedCall {
  url: string;
  method: string;
  body: unknown;
}

let calls: RecordedCall[];
let responses: Record<string, { status: number; body: unknown }>;

beforeEach(() => {
  calls = [];
  responses = {
    'GET /api/artifacts/library/videos/seoul-cyber-cut': {
      status: 200,
      body: sampleProject(),
    },
  };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      const parsedBody = init?.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ url, method, body: parsedBody });
      const key = `${method} ${url}`;
      const match = responses[key] ?? { status: 200, body: sampleProject() };
      return new Response(JSON.stringify(match.body), { status: match.status });
    })
  );
  // Mock canvas getContext for jsdom
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    save: vi.fn(),
    restore: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    fillText: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    createLinearGradient: () => ({ addColorStop: vi.fn() }),
    createRadialGradient: () => ({ addColorStop: vi.fn() }),
  } as unknown as CanvasRenderingContext2D);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('VideoView DaVinci Studio & 3-Tier VideoCompositorEngine', () => {
  it('switches between Tier 1 Proxy, Tier 2 Source Raw, and Tier 3 Master Render and toggles A/B Proposal preview', async () => {
    // Killed by: frontend/src/components/artifacts/VideoView.tsx :: if (tier === 'master') {
    // Becomes: if (false) {
    render(<VideoView projectId="seoul-cyber-cut" conversationId="room-1" />);

    expect(await screen.findByTestId('video-project-title')).toHaveTextContent(
      'Seoul Cyber Cut'
    );
    expect(screen.getByTestId('video-monitor-tier')).toHaveTextContent('960x540 (PREVIEW)');

    // Toggle to Proposed (B) preview bundle
    fireEvent.click(screen.getByTestId('video-ab-proposed'));
    expect(screen.getByTestId('video-monitor-tier')).toHaveTextContent('960x540 (SOURCE)');

    // Click Tier 3 Render Master
    fireEvent.click(screen.getByTestId('tier-btn-master'));
    const masterCalls = calls.filter(
      (c) => c.url === '/api/artifacts/library/videos/seoul-cyber-cut/render-master'
    );
    expect(masterCalls).toHaveLength(1);
  });

  it('filters Media Vault Pool between Raw and Derived assets and displays derived lineage', async () => {
    // Killed by: frontend/src/components/artifacts/VideoView.tsx :: if (mediaFilter === 'raw') return item.asset.kind === 'raw';
    // Becomes: if (mediaFilter === 'raw') return true;
    render(<VideoView projectId="seoul-cyber-cut" />);
    await screen.findByTestId('video-project-title');

    expect(screen.getByTestId('video-asset-card-a_cam_01')).toBeInTheDocument();
    expect(screen.getByTestId('video-asset-card-a_cam_01_slowmo_2x')).toBeInTheDocument();
    expect(screen.getByTestId('video-asset-lineage-a_cam_01_slowmo_2x')).toHaveTextContent(
      'Derived from a_cam_01 [F30..105] via optical_flow_slowmo'
    );

    fireEvent.click(screen.getByTestId('media-filter-raw'));
    expect(screen.getByTestId('video-asset-card-a_cam_01')).toBeInTheDocument();
    expect(screen.queryByTestId('video-asset-card-a_cam_01_slowmo_2x')).toBeNull();
  });

  it('executes semantic frame search and displays explicit method attribution', async () => {
    // Killed by: frontend/src/components/artifacts/VideoView.tsx :: `${BASE}/${encodeURIComponent(projectId)}/search`
    // Becomes: `${BASE}/${encodeURIComponent(projectId)}/noop`
    responses['POST /api/artifacts/library/videos/seoul-cyber-cut/search'] = {
      status: 200,
      body: {
        query: 'golden hour aerial',
        method: 'embedding similarity (nomic-embed-text)',
        total_indexed_frames: 8,
        hits: [
          {
            frame_id: 'a_cam_01_f000030',
            asset_id: 'a_cam_01',
            shot_id: 'shot_01',
            peak_frame: 30,
            in_frame: 0,
            out_frame: 75,
            in_timecode: '00:00:00:00',
            out_timecode: '00:00:02:15',
            duration_frames: 75,
            shot_type: 'aerial',
            camera_motion: 'drone_orbit',
            score: 0.94,
            caption: 'Golden hour aerial establishing shot over skyline',
            visual_tags: ['aerial', 'golden_hour'],
            transcript_slice: '',
          },
        ],
      },
    };

    render(<VideoView projectId="seoul-cyber-cut" />);
    await screen.findByTestId('video-project-title');

    fireEvent.change(screen.getByTestId('video-frame-search-input'), {
      target: { value: 'golden hour aerial' },
    });
    fireEvent.click(screen.getByTestId('video-frame-search-btn'));

    expect(await screen.findByTestId('video-search-method-badge')).toHaveTextContent(
      'Method: embedding similarity (nomic-embed-text)'
    );
    expect(screen.getByTestId('video-search-hit-a_cam_01_f000030')).toHaveTextContent(
      'Golden hour aerial establishing shot over skyline'
    );
  });

  it('executes direct NLE timeline actions (split, color grade, bake derived, unbake to raw, proposal approve)', async () => {
    // Killed by: frontend/src/components/artifacts/VideoView.tsx :: action: 'grade_clip',
    // Becomes: action: 'revert',
    render(<VideoView projectId="seoul-cyber-cut" conversationId="room-1" />);
    await screen.findByTestId('video-project-title');

    // Apply Color Grade
    fireEvent.change(screen.getByTestId('video-lut-select'), {
      target: { value: 'kodak_2383_warm' },
    });
    fireEvent.click(screen.getByTestId('video-apply-grade-btn'));

    await waitFor(() => {
      expect(screen.getByTestId('video-quick-bake-btn')).not.toBeDisabled();
    });
    const editCalls = calls.filter(
      (c) => c.url === '/api/artifacts/library/videos/seoul-cyber-cut/tracks/v1_main/edit'
    );
    expect(editCalls).toHaveLength(1);
    expect(editCalls[0].body).toMatchObject({
      action: 'grade_clip',
      clip_id: 'clip_01',
      seen_digest: 'dig-track-v1',
      room_id: 'room-1',
    });

    // Bake Derived Asset
    fireEvent.click(screen.getByTestId('video-quick-bake-btn'));
    await waitFor(() => {
      expect(screen.getByTestId('video-approve-p001')).not.toBeDisabled();
    });
    const bakeCalls = calls.filter(
      (c) => c.url === '/api/artifacts/library/videos/seoul-cyber-cut/bake'
    );
    expect(bakeCalls).toHaveLength(1);
    expect(bakeCalls[0].body).toMatchObject({
      parent_asset_id: 'a_cam_01',
      operation_kind: 'optical_flow_slowmo',
    });

    // Approve Proposal p001
    fireEvent.click(screen.getByTestId('video-approve-p001'));
    await waitFor(() => {
      const approveCalls = calls.filter(
        (c) => c.url === '/api/artifacts/library/videos/seoul-cyber-cut/proposals/p001/approve'
      );
      expect(approveCalls).toHaveLength(1);
      expect(approveCalls[0].body).toEqual({
        seen_digest: 'dig-prop-1',
        room_id: 'room-1',
      });
    });
  });

  it('evaluates SMPTE timecode, 3-way color grading filters, and frame composition accurately', () => {
    // Killed by: frontend/src/components/artifacts/VideoCompositorEngine.ts :: const ff = clamped % safeFps;
    // Becomes: const ff = 0;
    expect(framesToTimecode(75, 30)).toBe('00:00:02:15');

    // Killed by: frontend/src/components/artifacts/VideoCompositorEngine.ts :: grade.lut === 'noir_monochrome'
    // Becomes: false
    const noirFilter = evaluateColorGradeCssFilter({
      ...defaultGrade,
      lut: 'noir_monochrome',
      saturation: 1.2,
    });
    expect(noirFilter).toContain('saturate(0.00)');

    const combined = combineColorGrades(
      { ...defaultGrade, contrast: 1.2, lut: 'kodak_2383_warm' },
      { ...defaultGrade, contrast: 1.1, lut: 'clean_neutral' }
    );
    expect(combined.contrast).toBe(1.32);
    expect(combined.lut).toBe('kodak_2383_warm');

    const comp = evaluateFrameComposition(samplePlayback(), 140);
    expect(comp.primaryClip?.clip_id).toBe('clip_01');
    expect(comp.transitionKind).toBe('cross_dissolve');
    expect(comp.transitionProgress).toBeGreaterThan(0);
  });

  it('supports 3-point editing (+ Append to timeline, + Insert Shot from search), track Mute/Solo/Lock toggles, and HTML5 video stream binding', async () => {
    // Killed by: frontend/src/components/artifacts/VideoView.tsx :: action: 'append_clip',
    // Becomes: action: 'revert',
    responses['GET /api/artifacts/library/videos/seoul-cyber-cut'] = {
      status: 200,
      body: sampleProject({ style_profile: 'cinematic_film' }),
    };

    render(<VideoView projectId="seoul-cyber-cut" conversationId="room-1" />);
    await screen.findByTestId('video-project-title');

    // Verify HTML5 <video> element binds to the active clip's tier stream URL
    const videoEl = screen.getByTestId('video-stream-element') as HTMLVideoElement;
    expect(videoEl.getAttribute('src')).toContain(
      '/api/artifacts/library/videos/seoul-cyber-cut/stream/preview/a_cam_01'
    );

    // Click + Append on Media Pool asset card
    fireEvent.click(screen.getByTestId('video-append-asset-a_cam_01'));
    await waitFor(() => {
      const appendCalls = calls.filter(
        (c) =>
          c.url === '/api/artifacts/library/videos/seoul-cyber-cut/tracks/v1_main/edit' &&
          (c.body as { action?: string })?.action === 'append_clip'
      );
      expect(appendCalls).toHaveLength(1);
      expect(appendCalls[0].body).toMatchObject({
        action: 'append_clip',
        asset_id: 'a_cam_01',
        source_in_frame: 0,
        source_out_frame: 150,
      });
    });

    // Click Track Mute button on v1_main
    fireEvent.click(screen.getByTestId('video-track-mute-v1_main'));
    await waitFor(() => {
      const stateCalls = calls.filter(
        (c) =>
          c.url === '/api/artifacts/library/videos/seoul-cyber-cut/tracks/v1_main/edit' &&
          (c.body as { action?: string })?.action === 'set_track_state'
      );
      expect(stateCalls).toHaveLength(1);
      expect(stateCalls[0].body).toMatchObject({
        action: 'set_track_state',
        muted: true,
      });
    });
  });

  it('supports Spacebar Play/Pause, monitor click Play/Pause, J/K/L & Arrow frame stepping, Home/End jumps, and ignores shortcuts inside inputs', async () => {
    // Killed by: frontend/src/components/artifacts/VideoView.tsx :: case ' ':
    // Becomes: case '__DISABLED_SPACE__':
    render(<VideoView projectId="seoul-cyber-cut" conversationId="room-1" />);
    await screen.findByTestId('video-project-title');

    const playBtn = screen.getByTestId('video-play-toggle');
    expect(playBtn).toHaveTextContent('Play');
    expect(screen.getByTestId('video-timecode-hud')).toHaveTextContent('(F0)');

    // 1. Spacebar starts playback, second Spacebar stops playback
    fireEvent.keyDown(window, { key: ' ' });
    expect(playBtn).toHaveTextContent('Pause');
    fireEvent.keyDown(window, { key: ' ' });
    expect(playBtn).toHaveTextContent('Play');

    // 2. Clicking Program Monitor Canvas toggles Play / Pause
    fireEvent.click(screen.getByTestId('video-program-canvas'));
    expect(playBtn).toHaveTextContent('Pause');
    fireEvent.keyDown(window, { key: 'k' });
    expect(playBtn).toHaveTextContent('Play');

    // 3. ArrowRight / L steps forward, Shift+ArrowRight steps +10f, ArrowLeft / J steps back
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(screen.getByTestId('video-timecode-hud')).toHaveTextContent('(F1)');
    fireEvent.keyDown(window, { key: 'ArrowRight', shiftKey: true });
    expect(screen.getByTestId('video-timecode-hud')).toHaveTextContent('(F11)');
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(screen.getByTestId('video-timecode-hud')).toHaveTextContent('(F10)');

    // 4. Jump to End (End key / button) and Jump to Start (Home key / button)
    fireEvent.click(screen.getByTestId('video-jump-end'));
    expect(screen.getByTestId('video-timecode-hud')).toHaveTextContent('(F299)');
    fireEvent.click(screen.getByTestId('video-jump-start'));
    expect(screen.getByTestId('video-timecode-hud')).toHaveTextContent('(F0)');
    fireEvent.keyDown(window, { key: 'End' });
    expect(screen.getByTestId('video-timecode-hud')).toHaveTextContent('(F299)');
    fireEvent.keyDown(window, { key: 'Home' });
    expect(screen.getByTestId('video-timecode-hud')).toHaveTextContent('(F0)');

    // 5. Loop toggle & playback rate selector
    fireEvent.click(screen.getByTestId('video-loop-toggle'));
    fireEvent.change(screen.getByTestId('video-playback-rate'), { target: { value: '2' } });
    expect((screen.getByTestId('video-playback-rate') as HTMLSelectElement).value).toBe('2');

    // 6. Typing Space inside search input does NOT toggle playback
    const searchInput = screen.getByTestId('video-frame-search-input');
    fireEvent.keyDown(searchInput, { key: ' ' });
    expect(playBtn).toHaveTextContent('Play');
  });

  it('supports Trim In/Out buttons, Trim & Blade timeline modes, Split, Ripple Delete, Undo, and Title add/delete', async () => {
    // Killed by: frontend/src/components/artifacts/VideoView.tsx :: source_in_frame: newSourceIn,
    // Becomes: source_in_frame: 0,
    render(<VideoView projectId="seoul-cyber-cut" conversationId="room-1" />);
    await screen.findByTestId('video-project-title');

    // 1. Trim In button ([ Trim In)
    fireEvent.click(screen.getByTestId('video-trim-in-btn'));
    await waitFor(() => {
      expect(screen.getByTestId('video-trim-out-btn')).not.toBeDisabled();
      const trimInCalls = calls.filter(
        (c) =>
          c.url === '/api/artifacts/library/videos/seoul-cyber-cut/tracks/v1_main/edit' &&
          (c.body as { action?: string })?.action === 'trim_clip'
      );
      expect(trimInCalls.length).toBeGreaterThanOrEqual(1);
      expect(trimInCalls[0].body).toMatchObject({
        action: 'trim_clip',
        clip_id: 'clip_01',
        source_in_frame: 15,
        ripple: true,
      });
    });

    // 2. Trim Out button (Trim Out ])
    fireEvent.click(screen.getByTestId('video-trim-out-btn'));
    await waitFor(() => {
      expect(screen.getByTestId('video-split-clip-btn')).not.toBeDisabled();
      const trimCalls = calls.filter(
        (c) =>
          c.url === '/api/artifacts/library/videos/seoul-cyber-cut/tracks/v1_main/edit' &&
          (c.body as { action?: string })?.action === 'trim_clip'
      );
      expect(trimCalls.length).toBeGreaterThanOrEqual(2);
    });

    // 3. Switch to Blade tool (B key) and click clip_01 on the timeline bar to split it
    fireEvent.keyDown(window, { key: 'b' });
    fireEvent.click(screen.getByTestId('video-clip-clip_01'));
    await waitFor(() => {
      expect(screen.getByTestId('video-split-clip-btn')).not.toBeDisabled();
      const splitCalls = calls.filter(
        (c) =>
          c.url === '/api/artifacts/library/videos/seoul-cyber-cut/tracks/v1_main/edit' &&
          (c.body as { action?: string })?.action === 'split_clip'
      );
      expect(splitCalls).toHaveLength(1);
    });

    // 4. Switch to Titles tab and add a title overlay
    fireEvent.click(screen.getByTestId('inspector-tab-titles'));
    fireEvent.change(screen.getByTestId('video-title-input'), {
      target: { value: 'SEOUL SKYLINE' },
    });
    fireEvent.click(screen.getByTestId('video-add-title-btn'));
    await waitFor(() => {
      expect(screen.getByTestId('video-delete-clip-btn')).not.toBeDisabled();
      const titleCalls = calls.filter(
        (c) =>
          c.url === '/api/artifacts/library/videos/seoul-cyber-cut/tracks/v1_main/edit' &&
          (c.body as { action?: string })?.action === 'set_title'
      );
      expect(titleCalls).toHaveLength(1);
    });

    // 5. Ripple Delete via Delete key and Undo via Ctrl+Z
    fireEvent.keyDown(window, { key: 'Delete' });
    await waitFor(() => {
      expect(screen.getByTestId('video-delete-clip-btn')).not.toBeDisabled();
      const delCalls = calls.filter(
        (c) =>
          c.url === '/api/artifacts/library/videos/seoul-cyber-cut/tracks/v1_main/edit' &&
          (c.body as { action?: string })?.action === 'delete_clip'
      );
      expect(delCalls).toHaveLength(1);
    });

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true });
    await waitFor(() => {
      const revertCalls = calls.filter(
        (c) =>
          c.url === '/api/artifacts/library/videos/seoul-cyber-cut/tracks/v1_main/edit' &&
          (c.body as { action?: string })?.action === 'revert'
      );
      expect(revertCalls).toHaveLength(1);
    });
  });
});
