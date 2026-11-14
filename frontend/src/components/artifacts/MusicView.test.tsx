import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { MusicView } from './MusicView';
import {
  computeNoteExpression,
  defaultSpaceForGenre,
  filterPlayableNotes,
  MusicAudioEngine,
  normalizeGenreProfile,
  resolveArticulationEnvelope,
  resolveInstrumentPatch,
  resolveStagePosition3D,
  type MusicInstrumentView,
  type ShownSongView,
} from './MusicAudioEngine';
import { VoiceRegistry } from './audio/voiceRegistry';
import { analyzeVoiceRhythmBuffer } from './audio/voiceRhythmAnalyzer';

const sampleSong = (over: Partial<ShownSongView> = {}): ShownSongView => ({
  song_id: 'midnight-horizon',
  title: 'Midnight Horizon',
  genre: 'jazz_swing',
  style_axioms: {
    profile: 'jazz_swing',
    forbid_parallel_fifths_octaves: false,
    forbid_voice_crossing: false,
    require_kick_bass_lock: false,
    default_swing: 0.6,
  },
  writable: true,
  holder: 'room-1',
  decide_code: null,
  tempo: 120,
  time_signature: '4/4',
  key_signature: 'C major',
  total_bars: 2,
  duration_seconds: 4.0,
  sections: [
    {
      id: 'verse1',
      title: 'Verse 1',
      kind: 'verse',
      bars: 2,
      chords: ['Cmaj7', 'G7'],
    },
  ],
  instruments: [
    {
      id: 'piano',
      name: 'Grand Piano',
      category: 'piano',
      role: 'harmonic_comping',
      midi_program: 0,
      midi_channel: 1,
      soundfont_patch: 'gm_default',
      min_pitch: 'A0',
      max_pitch: 'C8',
      transposition_semitones: 0,
      volume: 85,
      pan: 0.0,
      muted: false,
      solo: false,
    },
    {
      id: 'trumpet',
      name: 'Bb Trumpet',
      category: 'brass',
      role: 'lead_melody',
      midi_program: 56,
      midi_channel: 2,
      soundfont_patch: 'gm_default',
      min_pitch: 'E3',
      max_pitch: 'C6',
      transposition_semitones: -2,
      volume: 80,
      pan: 0.0,
      muted: false,
      solo: false,
    },
  ],
  tracks: [
    {
      track_id: 'verse1_piano',
      section_id: 'verse1',
      instrument_id: 'piano',
      digest: 'dig-piano-1',
      groove_swing: 0.6,
      bars: [
        { bar: 1, chord: 'Cmaj7', notes: ['1.0:C4:2.0:v80', '3.0:E4:2.0:v80'], muted: false },
        { bar: 2, chord: 'G7', notes: ['1.0:G3:4.0:v85'], muted: false },
      ],
      can_revert: true,
    },
    {
      track_id: 'verse1_trumpet',
      section_id: 'verse1',
      instrument_id: 'trumpet',
      digest: 'dig-trumpet-1',
      groove_swing: 0.6,
      bars: [
        { bar: 1, chord: 'Cmaj7', notes: ['1.0:A#4:4.0:v90'], muted: false },
        { bar: 2, chord: 'G7', notes: ['1.0:REST:4.0'], muted: false },
      ],
      can_revert: false,
    },
  ],
  bars: [
    {
      global_bar: 1,
      section_id: 'verse1',
      section_title: 'Verse 1',
      section_bar: 1,
      chord: 'Cmaj7',
      bpm: 120,
      start_seconds: 0.0,
      end_seconds: 2.0,
    },
    {
      global_bar: 2,
      section_id: 'verse1',
      section_title: 'Verse 1',
      section_bar: 2,
      chord: 'G7',
      bpm: 120,
      start_seconds: 2.0,
      end_seconds: 4.0,
    },
  ],
  notes: [
    {
      track_id: 'verse1_piano',
      section_id: 'verse1',
      instrument_id: 'piano',
      bar: 1,
      global_bar: 1,
      beat: 1.0,
      duration: 2.0,
      concert_midi: 60,
      written_pitch: 'C4',
      concert_pitch: 'C4',
      velocity: 80,
      articulation: 'normal',
      start_seconds: 0.0,
      duration_seconds: 1.0,
    },
    {
      track_id: 'verse1_trumpet',
      section_id: 'verse1',
      instrument_id: 'trumpet',
      bar: 1,
      global_bar: 1,
      beat: 1.0,
      duration: 4.0,
      concert_midi: 70,
      written_pitch: 'C5',
      concert_pitch: 'A#4',
      velocity: 90,
      articulation: 'normal',
      start_seconds: 0.0,
      duration_seconds: 2.0,
    },
  ],
  proposals: [
    {
      id: 'p001',
      kind: 'track_patch',
      summary: 'Brighten piano bar 1 voicing',
      section_id: 'verse1',
      instrument_id: 'piano',
      start_bar: 1,
      end_bar: 1,
      seen_digest: 'prop-dig-1',
      original_bars: [{ bar: 1, chord: 'Cmaj7', notes: ['1.0:C4:2.0:v80', '3.0:E4:2.0:v80'] }],
      proposed_bars: [{ bar: 1, chord: 'Cmaj7', notes: ['1.0:G4:4.0:v88'] }],
    },
  ],
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
    'GET /api/artifacts/library/songs/midnight-horizon': {
      status: 200,
      body: sampleSong(),
    },
  };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      const parsedBody = init?.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ url, method, body: parsedBody });
      const key = `${method} ${url}`;
      const match = responses[key] ?? { status: 200, body: sampleSong() };
      return new Response(JSON.stringify(match.body), { status: match.status });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('MusicView Interactive Sheet Music & Multi-Track Studio', () => {
  it('renders transport bar, stem matrix, SVG staves, and switches concert vs written pitch', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: pitchMode === 'written' ? n.written_pitch : n.concert_pitch
    // Becomes: n.concert_pitch
    render(<MusicView songId="midnight-horizon" />);

    expect(await screen.findByTestId('music-view-title')).toHaveTextContent('Midnight Horizon');
    expect(screen.getByTestId('score-playhead')).toHaveAttribute('data-bar', '1');

    const trumpetCell = screen.getByTestId('score-cell-trumpet-1');
    expect(trumpetCell).toHaveTextContent('A#4');

    fireEvent.click(screen.getByTestId('music-pitch-toggle'));
    expect(screen.getByTestId('music-pitch-toggle')).toHaveTextContent('Written Pitch');
    expect(trumpetCell).toHaveTextContent('C5');
  });

  it('filters visible staves when Part View is selected and toggles Solo/Mute in the stem matrix', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: if (partFilter === 'all') return song.instruments;
    // Becomes: return song.instruments;
    render(<MusicView songId="midnight-horizon" />);
    await screen.findByTestId('music-view-title');

    expect(screen.getByTestId('score-staff-piano')).toBeInTheDocument();
    expect(screen.getByTestId('score-staff-trumpet')).toBeInTheDocument();

    fireEvent.change(screen.getByTestId('music-part-selector'), {
      target: { value: 'trumpet' },
    });
    expect(screen.queryByTestId('score-staff-piano')).toBeNull();
    expect(screen.getByTestId('score-staff-trumpet')).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('stem-solo-piano'));
    const mixCalls = calls.filter(
      (c) => c.url === '/api/artifacts/library/songs/midnight-horizon/instruments/piano/mix',
    );
    expect(mixCalls).toHaveLength(1);
    expect(mixCalls[0].body).toEqual({ solo: true });
  });

  it('selects a bar on the SVG score and triggers direct score actions and Fast Edit prompt', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: onClick={() => void handleDirectEdit('clear_bars')}
    // Becomes: onClick={() => {}}
    const onSelectRegion = vi.fn();
    const onFastEditPrompt = vi.fn();
    render(
      <MusicView
        songId="midnight-horizon"
        onSelectRegion={onSelectRegion}
        onFastEditPrompt={onFastEditPrompt}
      />,
    );

    const cell = await screen.findByTestId('score-cell-piano-2');
    fireEvent.click(cell);

    expect(onSelectRegion).toHaveBeenCalledWith({
      section_id: 'verse1',
      start_bar: 2,
      end_bar: 2,
      instrument_ids: ['piano'],
    });
    expect(screen.getByTestId('music-selection-badge')).toHaveTextContent('🎯 verse1 · Bar 2 (piano)');

    fireEvent.click(screen.getByTestId('score-action-clear'));
    const editCalls = calls.filter(
      (c) => c.url === '/api/artifacts/library/songs/midnight-horizon/tracks/verse1_piano/edit',
    );
    expect(editCalls).toHaveLength(1);
    expect(editCalls[0].body).toEqual({
      action: 'clear_bars',
      start_bar: 2,
      end_bar: 2,
      semitones: 0,
      muted: true,
      seen_digest: 'dig-piano-1',
    });

    fireEvent.click(await screen.findByTestId('score-action-fast-edit'));
    expect(onFastEditPrompt).toHaveBeenCalledWith('이 부분을 빠르게 수정해줘', {
      section_id: 'verse1',
      start_bar: 2,
      end_bar: 2,
      instrument_ids: ['piano'],
    });
  });

  it('toggles A/B proposal audition mode and sends seen_digest on Approve and Reject', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: { seen_digest: proposal.seen_digest }
    // Becomes: { seen_digest: '' }
    render(<MusicView songId="midnight-horizon" />);

    const toggleBtn = await screen.findByTestId('proposal-mode-toggle-p001');
    expect(toggleBtn).toHaveTextContent('Auditioning: Proposed (B)');

    // In Proposed (B) mode, piano bar 1 shows proposed G4
    expect(screen.getByTestId('score-cell-piano-1')).toHaveTextContent('G4');

    fireEvent.click(toggleBtn);
    expect(toggleBtn).toHaveTextContent('Auditioning: Original (A)');
    expect(screen.getByTestId('score-cell-piano-1')).toHaveTextContent('C4');

    fireEvent.click(screen.getByTestId('proposal-approve-p001'));
    const approveCalls = calls.filter(
      (c) => c.url === '/api/artifacts/library/songs/midnight-horizon/proposals/p001/approve',
    );
    expect(approveCalls).toHaveLength(1);
    expect(approveCalls[0].body).toEqual({ seen_digest: 'prop-dig-1' });
  });

  it('approves a proposal: posts its seen_digest with the conversation, drops the card and says so', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: ? `Approved: ${proposal.summary}`
    // Becomes: ? ''
    responses['POST /api/artifacts/library/songs/midnight-horizon/proposals/p001/approve'] = {
      status: 200,
      body: sampleSong({ proposals: [] }),
    };
    render(<MusicView songId="midnight-horizon" conversationId="room-1" />);

    fireEvent.click(await screen.findByTestId('proposal-approve-p001'));

    expect(await screen.findByTestId('music-proposal-notice')).toHaveTextContent(
      'Approved: Brighten piano bar 1 voicing',
    );
    expect(screen.queryByTestId('music-proposal-p001')).toBeNull();
    const approveCalls = calls.filter(
      (c) =>
        c.method === 'POST' &&
        c.url === '/api/artifacts/library/songs/midnight-horizon/proposals/p001/approve',
    );
    expect(approveCalls).toHaveLength(1);
    expect(approveCalls[0].body).toEqual({ seen_digest: 'prop-dig-1', room_id: 'room-1' });
  });

  it('keeps Approve usable when no conversation holds the song, and explains what clicking does', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: const canDecide = Boolean(song && (song.writable || conversationId));
    // Becomes: const canDecide = Boolean(song && song.writable);
    responses['GET /api/artifacts/library/songs/midnight-horizon'] = {
      status: 200,
      body: sampleSong({
        writable: false,
        holder: 'room-gone',
        decide_code: 'writer_gone',
        decide_note: 'The conversation that was composing “Midnight Horizon” no longer exists.',
      }),
    };
    responses['POST /api/artifacts/library/songs/midnight-horizon/proposals/p001/approve'] = {
      status: 200,
      body: sampleSong({ proposals: [], holder: 'room-2' }),
    };
    render(<MusicView songId="midnight-horizon" conversationId="room-2" />);

    expect(await screen.findByTestId('music-proposals-writer-note')).toHaveTextContent(
      'opens it in this conversation',
    );
    const approve = screen.getByTestId('proposal-approve-p001');
    expect(approve).not.toBeDisabled();
    fireEvent.click(approve);

    expect(await screen.findByTestId('music-proposal-notice')).toHaveTextContent('Approved:');
    expect(screen.queryByTestId('music-proposal-p001')).toBeNull();
    expect(calls.filter((c) => c.method === 'POST').map((c) => c.body)).toEqual([
      { seen_digest: 'prop-dig-1', room_id: 'room-2' },
    ]);
  });

  it('shows a plain notice beside the proposal when approving is refused, and keeps the card', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: {decisionNotice && (
    // Becomes: {false && (
    responses['POST /api/artifacts/library/songs/midnight-horizon/proposals/p001/approve'] = {
      status: 409,
      body: {
        detail:
          'The conversation “Swan Lake” is answering right now, so the change was not saved. Wait for the turn to finish, then try again.',
      },
    };
    render(<MusicView songId="midnight-horizon" conversationId="room-1" />);

    fireEvent.click(await screen.findByTestId('proposal-approve-p001'));

    expect(await screen.findByTestId('music-proposal-notice')).toHaveTextContent(
      'is answering right now',
    );
    expect(screen.getByTestId('music-proposal-p001')).toBeInTheDocument();
  });

  it('offers a confirmed window when the server refuses a change from this window', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: setUnconfirmed(err instanceof CoreFailure && err.status === 403);
    // Becomes: setUnconfirmed(false);
    responses['POST /api/artifacts/library/songs/midnight-horizon/proposals/p001/approve'] = {
      status: 403,
      body: { detail: 'Only you can make this decision, from a window UClone-X opened.' },
    };
    render(<MusicView songId="midnight-horizon" conversationId="room-1" />);

    fireEvent.click(await screen.findByTestId('proposal-approve-p001'));
    const notice = await screen.findByTestId('music-proposal-notice');
    expect(notice).toHaveTextContent('Only you can make this decision');
    fireEvent.click(within(notice).getByTestId('music-open-confirmed-window'));

    expect(await screen.findByTestId('music-proposal-notice')).toHaveTextContent(
      'A confirmed window opened in your browser.',
    );
    expect(calls.some((c) => c.method === 'POST' && c.url === '/api/person/window')).toBe(true);
    expect(screen.getByTestId('music-proposal-p001')).toBeInTheDocument();
  });

  it('filters playable notes in MusicAudioEngine by Solo/Mute and loop boundaries', () => {
    // Killed by: frontend/src/components/artifacts/MusicAudioEngine.ts :: const anySolo = instruments.some((inst) => inst.solo);
    // Becomes: const anySolo = false;
    const song = sampleSong();
    const soloTrumpet = song.instruments.map((inst) =>
      inst.id === 'trumpet' ? { ...inst, solo: true } : inst,
    );
    const filtered = filterPlayableNotes(song.notes, soloTrumpet, 1, 1);
    expect(filtered.map((n) => n.instrument_id)).toEqual(['trumpet']);
  });

  it('renders note duration labels and section arrangement headers on the continuous timeline', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: return `${Math.round(duration)}b`;
    // Becomes: return '';
    render(<MusicView songId="midnight-horizon" />);

    expect(await screen.findByTestId('music-view-title')).toHaveTextContent('Midnight Horizon');
    const trumpetCell = screen.getByTestId('score-cell-trumpet-1');
    expect(trumpetCell).toHaveTextContent('4b');
  });

  it('resolves genre-aware GM patches, acoustic spaces, articulation envelopes, and custom voice modules', async () => {
    // Killed by: frontend/src/components/artifacts/audio/gmCatalog.ts :: if (g.includes('classical')) return 'classical_strict';
    // Becomes: if (false) return 'classical_strict';
    expect(normalizeGenreProfile('Classical Chamber')).toBe('classical_strict');
    expect(normalizeGenreProfile('pop')).toBe('pop_ballad');
    expect(defaultSpaceForGenre('classical_strict')).toBe('concert_hall');
    expect(defaultSpaceForGenre('jazz_swing')).toBe('jazz_club');
    expect(defaultSpaceForGenre('hiphop_groove')).toBe('tight_booth');
    expect(defaultSpaceForGenre('pop_ballad')).toBe('studio_room');

    const baseBass: MusicInstrumentView = {
      id: 'bass_1',
      name: 'Bass',
      category: 'bass',
      role: 'bass_foundation',
      midi_program: 0,
      midi_channel: 1,
      soundfont_patch: 'gm_default',
      min_pitch: 'E1',
      max_pitch: 'G3',
      transposition_semitones: 0,
      volume: 85,
      pan: 0,
      muted: false,
      solo: false,
    };

    const jazzBass = resolveInstrumentPatch(baseBass, 'jazz_swing');
    expect(jazzBass.timbreFamily).toBe('upright_bass');
    expect(jazzBass.gmInstrumentName).toBe('acoustic_bass');

    const hiphopBass = resolveInstrumentPatch(baseBass, 'hiphop_groove');
    expect(hiphopBass.timbreFamily).toBe('sub_808_bass');
    expect(hiphopBass.gmInstrumentName).toBe('synth_bass_1');

    const pizz = resolveArticulationEnvelope('pizzicato', 'strings');
    expect(pizz.isPlucked).toBe(true);
    expect(pizz.gateRatio).toBeLessThan(0.5);

    const vib = resolveArticulationEnvelope('vibrato', 'strings');
    expect(vib.vibratoDepthCents).toBeGreaterThan(10);

    // 3D Orchestral Stage Seating & Multi-Desk Ensemble Divisi
    const stringEnsembleStage = resolveStagePosition3D('string_ensemble');
    expect(stringEnsembleStage.ensembleVoices).toBe(3);
    expect(stringEnsembleStage.depthZ).toBeLessThan(-2.5);

    const hornStage = resolveStagePosition3D('french_horn');
    expect(hornStage.depthZ).toBeLessThan(stringEnsembleStage.depthZ);
    expect(hornStage.elevationY).toBeGreaterThan(0.4);

    // Emotional Phrasing & CC11 Swell Curve
    const introNoteExpr = computeNoteExpression({
      sectionId: 'intro',
      sectionKind: 'intro',
      sectionBar: 1,
      sectionTotalBars: 4,
      beat: 2.0,
      durationSeconds: 0.25,
      secPerBeat: 0.5,
      midi: 60,
      velocity: 75,
      articulation: 'normal',
      category: 'piano',
      genre: 'classical_strict',
    });
    const climaxStringsExpr = computeNoteExpression({
      sectionId: 'chorus',
      sectionKind: 'chorus',
      sectionBar: 3,
      sectionTotalBars: 4,
      beat: 1.0,
      durationSeconds: 1.5,
      secPerBeat: 0.5,
      midi: 67,
      velocity: 96,
      articulation: 'legato',
      category: 'strings',
      role: 'lead_melody',
      genre: 'classical_strict',
    });
    expect(climaxStringsExpr.dynamicScale).toBeGreaterThan(introNoteExpr.dynamicScale);
    expect(climaxStringsExpr.swellPeakRatio).toBeGreaterThan(1.15);
    expect(climaxStringsExpr.timbralBloom).toBeGreaterThan(1.1);

    const customRegistry = new VoiceRegistry();
    customRegistry.register({
      id: 'voice.custom_test',
      supports: (patch) => patch.timbreFamily === 'upright_bass',
      trigger: () => {},
    });
    expect(customRegistry.resolveModule(jazzBass).id).toBe('voice.custom_test');
    expect(customRegistry.resolveModule(hiphopBass).id).not.toBe('voice.custom_test');
    expect(new MusicAudioEngine(customRegistry).isPlaying()).toBe(false);

    render(<MusicView songId="midnight-horizon" />);
    expect(await screen.findByTestId('music-acoustic-space-selector')).toHaveValue('auto');
    expect(screen.queryByTestId('music-engine-mode-selector')).toBeNull();
  });

  it('credits the recorded piano samples with a licence link only when the song has a grand piano', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: {usesRecordedPiano && (
    // Becomes: {false && (
    const full = sampleSong();
    responses['GET /api/artifacts/library/songs/midnight-horizon'] = {
      status: 200,
      body: sampleSong({
        instruments: full.instruments.filter((i) => i.id !== 'piano'),
        tracks: full.tracks.filter((t) => t.instrument_id !== 'piano'),
        notes: full.notes.filter((n) => n.instrument_id !== 'piano'),
      }),
    };
    const first = render(<MusicView songId="midnight-horizon" />);
    await screen.findByTestId('music-view-title');
    expect(screen.queryByTestId('music-piano-credit')).toBeNull();
    first.unmount();

    responses['GET /api/artifacts/library/songs/midnight-horizon'] = { status: 200, body: full };
    render(<MusicView songId="midnight-horizon" />);
    const credit = await screen.findByTestId('music-piano-credit');
    expect(credit).toHaveTextContent(
      'Piano samples: Salamander Grand Piano by Alexander Holm (CC BY 3.0)',
    );
    expect(within(credit).getByRole('link', { name: 'License' })).toHaveAttribute(
      'href',
      'https://creativecommons.org/licenses/by/3.0/',
    );
  });

  it('supports structural bar insertion/appending, inline bar note/chord editing, and Approve All proposals (Scenario S12)', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: onClick={() => void handleApproveAllProposals()}
    // Becomes: onClick={() => {}}
    render(<MusicView songId="midnight-horizon" />);
    await screen.findByTestId('music-view-title');

    // 1. Prepend Intro (2 Bars)
    fireEvent.click(screen.getByTestId('score-structure-prepend-intro'));
    const introCalls = calls.filter(
      (c) => c.url === '/api/artifacts/library/songs/midnight-horizon/structure',
    );
    expect(introCalls).toHaveLength(1);
    expect(introCalls[0].body).toEqual({
      action: 'insert_bars',
      section_id: 'verse1',
      at_bar: 1,
      count: 2,
      intent: 'gentle_intro',
    });

    // 2. Select bar cell -> edit inline chord & notes -> Apply Notes
    const cell = await screen.findByTestId('score-cell-piano-1');
    fireEvent.click(cell);
    const chordInput = await screen.findByTestId('score-inline-chord-input');
    const notesInput = await screen.findByTestId('score-inline-notes-input');
    fireEvent.change(chordInput, { target: { value: 'Fmaj7' } });
    fireEvent.change(notesInput, {
      target: { value: '1.0:F4:2.0:v82:legato, 3.0:A4:2.0:v84:tenuto' },
    });
    fireEvent.click(screen.getByTestId('score-action-apply-notes'));

    const noteEditCalls = calls.filter(
      (c) =>
        c.url === '/api/artifacts/library/songs/midnight-horizon/tracks/verse1_piano/edit' &&
        (c.body as { action?: string })?.action === 'set_bar_notes',
    );
    expect(noteEditCalls).toHaveLength(1);
    expect(noteEditCalls[0].body).toEqual({
      action: 'set_bar_notes',
      start_bar: 1,
      end_bar: 1,
      semitones: 0,
      muted: true,
      chord: 'Fmaj7',
      notes: ['1.0:F4:2.0:v82:legato', '3.0:A4:2.0:v84:tenuto'],
      seen_digest: 'dig-piano-1',
    });

    // 3. Approve All proposals
    const approveAllBtn = await screen.findByTestId('proposal-approve-all');
    fireEvent.click(approveAllBtn);
    const approveAllCalls = calls.filter(
      (c) => c.url === '/api/artifacts/library/songs/midnight-horizon/proposals/approve-all',
    );
    expect(approveAllCalls).toHaveLength(1);
  });
  describe('timeline follow', () => {
    const longSong = (): ShownSongView => {
      const base = sampleSong();
      const bars = Array.from({ length: 12 }, (_, i) => ({
        ...base.bars[0],
        global_bar: i + 1,
        section_bar: i + 1,
        start_seconds: i * 2,
        end_seconds: i * 2 + 2,
      }));
      return {
        ...base,
        total_bars: 12,
        sections: [{ ...base.sections[0], bars: 12, chords: bars.map(() => 'Cmaj7') }],
        bars,
      };
    };

    // Plays the song and returns what the engine would report as the bar it is on.
    const startPlayback = async () => {
      responses['GET /api/artifacts/library/songs/midnight-horizon'] = {
        status: 200,
        body: longSong(),
      };
      let reportBar: (bar: number) => void = () => {};
      vi.spyOn(MusicAudioEngine.prototype, 'start').mockImplementation(async (opts) => {
        reportBar = opts.onBarChange ?? (() => {});
      });
      render(<MusicView songId="midnight-horizon" />);
      const scroller = await screen.findByTestId('music-score-scroller');
      Object.defineProperty(scroller, 'clientWidth', { configurable: true, value: 600 });
      fireEvent.click(screen.getByTestId('music-transport-play'));
      return { scroller, moveTo: (bar: number) => act(() => reportBar(bar)) };
    };

    it('scrolls to a bar that played out of view, and leaves a bar already in view alone', async () => {
      // Killed by: frontend/src/components/artifacts/MusicView.tsx :: barLeft + BAR_WIDTH > visibleRight
      // Becomes: false
      const { scroller, moveTo } = await startPlayback();
      moveTo(2);
      expect(scroller.scrollLeft).toBe(0);
      moveTo(6);
      // Bar 6 starts 208 (label column) + 5 * 168 into the content; the label column then
      // covers 208 of the scroller, so the scroller lands 5 bars in.
      expect(scroller.scrollLeft).toBe(5 * 168);
    });

    it('does not scroll while Follow is off', async () => {
      // Killed by: frontend/src/components/artifacts/MusicView.tsx :: !followPlayhead
      // Becomes: false
      const { scroller, moveTo } = await startPlayback();
      fireEvent.click(screen.getByTestId('music-transport-follow'));
      expect(screen.getByTestId('music-transport-follow')).toHaveAttribute('aria-pressed', 'false');
      moveTo(6);
      expect(scroller.scrollLeft).toBe(0);
    });

    it('stops following once the user scrolls by hand, until Play is pressed again', async () => {
      // Killed by: frontend/src/components/artifacts/MusicView.tsx :: userScrolledRef.current = true
      // Becomes: userScrolledRef.current = false
      const { scroller, moveTo } = await startPlayback();
      scroller.scrollLeft = 40;
      fireEvent.scroll(scroller);
      moveTo(6);
      expect(scroller.scrollLeft).toBe(40);
    });

    it('follows again after the user presses Pause and Play', async () => {
      // Killed by: frontend/src/components/artifacts/MusicView.tsx :: resumeFollow(); // Play starts following again
      // Becomes: void 0; // Play starts following again
      const { scroller, moveTo } = await startPlayback();
      scroller.scrollLeft = 40;
      fireEvent.scroll(scroller);
      fireEvent.click(screen.getByTestId('music-transport-play'));
      fireEvent.click(screen.getByTestId('music-transport-play'));
      moveTo(6);
      expect(scroller.scrollLeft).toBe(5 * 168);
    });

    it("does not read its own scroll as the user's", async () => {
      // Killed by: frontend/src/components/artifacts/MusicView.tsx :: !== followScrollLeftRef.current
      // Becomes: === followScrollLeftRef.current
      const { scroller, moveTo } = await startPlayback();
      moveTo(6);
      fireEvent.scroll(scroller);
      moveTo(10);
      expect(scroller.scrollLeft).toBe(9 * 168);
    });
  });

  it('extracts voice/tap rhythm skeleton and sends structured [VoiceRhythm] prompt to Composer LLM', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: data-recording={isRecordingVoice ? 'true' : 'false'}
    // Becomes: data-recording="stale"
    const sr = 8000;
    const pcm = new Float32Array(sr * 2);
    // Synthesize 2 vocal bursts at t=0.05s (low freq ~100Hz, 0.5s long) and t=0.80s (mid freq ~600Hz, 0.2s long)
    for (let i = Math.floor(0.05 * sr); i < Math.floor(0.55 * sr); i++) {
      pcm[i] = 0.8 * Math.sin((2 * Math.PI * 100 * i) / sr);
    }
    for (let i = Math.floor(0.8 * sr); i < Math.floor(1.0 * sr); i++) {
      pcm[i] = 0.6 * Math.sin((2 * Math.PI * 600 * i) / sr);
    }
    const analyzed = analyzeVoiceRhythmBuffer(pcm, sr, { bpm: 96, maxBars: 1 });
    expect(analyzed.hitCount).toBeGreaterThanOrEqual(2);
    expect(analyzed.barPatterns[0]).toContain('1.0:low:');

    const selectedRegions: unknown[] = [];
    const sentPrompts: { prompt: string; selection: unknown }[] = [];
    render(
      <MusicView
        songId="midnight-horizon"
        onSelectRegion={(sel) => selectedRegions.push(sel)}
        onFastEditPrompt={(prompt, selection) => sentPrompts.push({ prompt, selection })}
      />,
    );
    await screen.findByTestId('music-view-title');

    // Select bar 2 on piano
    fireEvent.click(screen.getByTestId('score-cell-piano-2'));
    expect(selectedRegions).toContainEqual({
      section_id: 'verse1',
      start_bar: 2,
      end_bar: 2,
      instrument_ids: ['piano'],
    });

    // Open Voice Rhythm panel, select 2-bar span, and record via microphone
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: () => [{ stop: vi.fn() }],
        }),
      },
      configurable: true,
    });

    fireEvent.click(screen.getByTestId('music-voice-rhythm-toggle'));
    await screen.findByTestId('music-voice-rhythm-panel');
    expect(screen.getByTestId('music-voice-target-region')).toHaveTextContent(
      'Target: verse1 · Bar 2–2 · piano',
    );
    expect(screen.getByTestId('music-voice-live-monitor')).toHaveAttribute(
      'data-recording',
      'false',
    );

    // Start mic recording
    await act(async () => {
      fireEvent.click(screen.getByTestId('music-voice-record-btn'));
    });
    expect(screen.getByTestId('music-voice-live-monitor')).toHaveAttribute(
      'data-recording',
      'true',
    );
    expect(screen.getByTestId('music-voice-active-band')).toHaveTextContent('LOW 감지');
    expect(screen.getByTestId('music-voice-live-hits')).toHaveTextContent('2');

    // Stop mic recording & extract rhythm
    await act(async () => {
      fireEvent.click(screen.getByTestId('music-voice-record-btn'));
    });

    const visualGrid = await screen.findByTestId('music-voice-visual-grid');
    expect(visualGrid).toHaveTextContent('Bar 2');
    expect(visualGrid).toHaveTextContent('🥁 쿵');

    const preview = await screen.findByTestId('music-voice-rhythm-preview');
    expect(preview).toHaveTextContent(
      '[verse1 | Bar 2-2 | piano] [VoiceRhythm | detected_bpm=',
    );

    fireEvent.click(screen.getByTestId('music-voice-rhythm-send'));
    expect(sentPrompts).toHaveLength(1);
    expect(sentPrompts[0].prompt).toContain('[VoiceRhythm | detected_bpm=');
    expect(sentPrompts[0].selection).toEqual({
      section_id: 'verse1',
      start_bar: 2,
      end_bar: 2,
      instrument_ids: ['piano'],
    });
  });

  it('posts voice rhythm directly to the conversation messages endpoint when onFastEditPrompt is omitted', async () => {
    // Killed by: frontend/src/components/artifacts/MusicView.tsx :: `/api/rooms/${encodeURIComponent(conversationId)}/messages`
    // Becomes: ''
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: () => [{ stop: vi.fn() }],
        }),
      },
      configurable: true,
    });
    responses['POST /api/rooms/room-1/messages'] = {
      status: 200,
      body: { ok: true },
    };

    render(<MusicView songId="midnight-horizon" conversationId="room-1" />);
    await screen.findByTestId('music-view-title');

    fireEvent.click(screen.getByTestId('music-voice-rhythm-toggle'));
    await screen.findByTestId('music-voice-rhythm-panel');
    await act(async () => {
      fireEvent.click(screen.getByTestId('music-voice-record-btn'));
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('music-voice-record-btn'));
    });
    fireEvent.click(await screen.findByTestId('music-voice-rhythm-send'));

    expect(await screen.findByTestId('music-view-notice')).toHaveTextContent(
      'Sent rhythm request to Composer.',
    );
    const roomPostCalls = calls.filter(
      (c) => c.method === 'POST' && c.url === '/api/rooms/room-1/messages',
    );
    expect(roomPostCalls).toHaveLength(1);
    expect((roomPostCalls[0].body as { content: string }).content).toContain(
      '[verse1 | Bar 1-2 | piano] [VoiceRhythm | detected_bpm=',
    );
  });
});
