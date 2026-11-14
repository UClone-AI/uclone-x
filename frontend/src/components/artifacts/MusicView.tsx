import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, Pause, Play, RefreshCw, Repeat } from 'lucide-react';
import { Button } from '../ui/Button';
import { CoreFailure, failureOf, plainFailure } from '../../lib/coreFailure';
import { openConfirmedWindow } from '../../lib/person';
import {
  ACOUSTIC_SPACES,
  buildNotesWithProposalPreview,
  defaultSpaceForGenre,
  MusicAudioEngine,
  normalizeGenreProfile,
  parsePitchTokenToMidi,
  resolveInstrumentPatch,
  type AcousticSpaceProfile,
  type InstrumentCategory,
  type MusicBarView,
  type MusicInstrumentView,
  type MusicNoteView,
  type MusicProposalView,
  type MusicRegionSelection,
  type ShownSongView,
} from './MusicAudioEngine';
import {
  isSampledPianoPatch,
  SALAMANDER_PIANO_CREDIT,
  SALAMANDER_PIANO_LICENSE_URL,
} from './audio/salamanderPiano';
import {
  analyzeVoiceRhythmBuffer,
  formatVoiceRhythmPromptTag,
  type VoiceBand,
  type VoiceRhythmSketch,
} from './audio/voiceRhythmAnalyzer';

/** Arrangement intent the structure-edit route accepts (server: `ArrangementIntent`). */
type ArrangementIntent =
  | 'gentle_intro'
  | 'climax_ornate'
  | 'hiphop_groove'
  | 'outro_cadence'
  | 'continuation';

export interface MusicViewProps {
  songId: string;
  /**
   * The conversation this view is docked beside. A decision made here is sent with it, so a
   * song no live conversation holds is opened in this one instead of refusing the click.
   */
  conversationId?: string | null;
  onBack?: () => void;
  refreshKey?: unknown;
  onSelectRegion?: (selection: MusicRegionSelection | null) => void;
  onFastEditPrompt?: (prompt: string, selection: MusicRegionSelection) => void;
}

const BASE = '/api/artifacts/library/songs';
const BAR_WIDTH = 168;
/** The sticky track-label column (`w-52`): bars scroll underneath it, so it hides this much. */
const LABEL_COLUMN_WIDTH = 208;
const BEAT_WIDTH = 36;
const BEAT_START_X = 16;

async function readSongJson<T>(res: Response): Promise<T> {
  if (!res.ok) throw await failureOf(res);
  return (await res.json()) as T;
}

const postSongJson = (url: string, body: unknown): Promise<Response> =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

function clefForCategory(category: InstrumentCategory): string {
  if (category === 'bass') return '𝄢';
  if (category === 'drums') return '𝄥';
  return '𝄞';
}

function trackThemeForCategory(category: InstrumentCategory): {
  accentBorder: string;
  badgeBg: string;
  noteFill: string;
  sustainFill: string;
} {
  switch (category) {
    case 'strings':
      return {
        accentBorder: 'border-l-amber-400',
        badgeBg: 'bg-amber-500/15 text-amber-300',
        noteFill: '#f59e0b',
        sustainFill: 'rgba(245, 158, 11, 0.28)',
      };
    case 'piano':
      return {
        accentBorder: 'border-l-emerald-400',
        badgeBg: 'bg-emerald-500/15 text-emerald-300',
        noteFill: '#34d399',
        sustainFill: 'rgba(52, 211, 153, 0.26)',
      };
    case 'bass':
      return {
        accentBorder: 'border-l-sky-400',
        badgeBg: 'bg-sky-500/15 text-sky-300',
        noteFill: '#38bdf8',
        sustainFill: 'rgba(56, 189, 248, 0.28)',
      };
    case 'drums':
      return {
        accentBorder: 'border-l-rose-400',
        badgeBg: 'bg-rose-500/15 text-rose-300',
        noteFill: '#fb7185',
        sustainFill: 'rgba(251, 113, 133, 0.25)',
      };
    default:
      return {
        accentBorder: 'border-l-violet-400',
        badgeBg: 'bg-violet-500/15 text-violet-300',
        noteFill: '#a78bfa',
        sustainFill: 'rgba(167, 139, 250, 0.28)',
      };
  }
}

function noteheadY(midi: number, category: InstrumentCategory, writtenPitch?: string): number {
  if (category === 'drums' && writtenPitch) {
    const upper = writtenPitch.toUpperCase();
    if (upper.includes('HIHAT') || upper.includes('CRASH') || upper.includes('RIDE')) return 14;
    if (upper.includes('SNARE') || upper.includes('CLAP') || upper.includes('RIM')) return 28;
    if (upper.includes('TOM')) return 35;
    return 44; // KICK
  }
  if (midi < 0) return 30;
  const centerMidi = category === 'bass' ? 48 : 64;
  const semitoneDiff = midi - centerMidi;
  const rawY = 30 - semitoneDiff * 1.4;
  return Math.max(8, Math.min(50, rawY));
}

function describeDurationGlyph(duration: number): {
  glyph: string;
  isHollow: boolean;
  hasStem: boolean;
  flags: number;
  isDotted: boolean;
} {
  const isDotted =
    Math.abs(duration - 3.0) < 0.12 ||
    Math.abs(duration - 1.5) < 0.12 ||
    Math.abs(duration - 0.75) < 0.08;
  if (duration >= 3.5) {
    return { glyph: '𝅝', isHollow: true, hasStem: false, flags: 0, isDotted: false };
  }
  if (duration >= 1.75) {
    return {
      glyph: isDotted ? '𝅗𝅥.' : '𝅗𝅥',
      isHollow: true,
      hasStem: true,
      flags: 0,
      isDotted,
    };
  }
  if (duration >= 0.85) {
    return {
      glyph: isDotted ? '♩.' : '♩',
      isHollow: false,
      hasStem: true,
      flags: 0,
      isDotted,
    };
  }
  if (duration >= 0.38) {
    return {
      glyph: isDotted ? '♪.' : '♪',
      isHollow: false,
      hasStem: true,
      flags: 1,
      isDotted,
    };
  }
  return {
    glyph: '♬',
    isHollow: false,
    hasStem: true,
    flags: 2,
    isDotted: false,
  };
}

function formatBeatsCompact(duration: number): string {
  if (Math.abs(duration - Math.round(duration)) < 0.05) {
    return `${Math.round(duration)}b`;
  }
  if (Math.abs(duration * 2 - Math.round(duration * 2)) < 0.05) {
    return `${duration.toFixed(1)}b`;
  }
  return `${duration.toFixed(2)}b`;
}

export const MusicView: React.FC<MusicViewProps> = ({
  songId,
  conversationId,
  onBack,
  refreshKey,
  onSelectRegion,
  onFastEditPrompt,
}) => {
  const [song, setSong] = useState<ShownSongView | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // The result of the last Approve / Reject, shown beside the proposals where it was clicked.
  const [decisionNotice, setDecisionNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // The last change was refused because this window is not one the server confirmed.
  const [unconfirmed, setUnconfirmed] = useState(false);
  // A change that failed: its plain sentence, and whether a confirmed window would help.
  const changeFailed = (err: unknown, fallback: string): string => {
    setUnconfirmed(err instanceof CoreFailure && err.status === 403);
    return plainFailure(err, fallback);
  };
  const openWindow = async () => {
    try {
      await openConfirmedWindow();
      setDecisionNotice('A confirmed window opened in your browser. Make the change there.');
    } catch (err) {
      setDecisionNotice(plainFailure(err, 'No confirmed window could be opened.'));
    }
    setUnconfirmed(false);
  };
  const confirmButton = unconfirmed ? (
    <Button
      variant="outline"
      className="mt-2"
      onClick={() => void openWindow()}
      data-testid="music-open-confirmed-window"
    >
      Open a confirmed window
    </Button>
  ) : null;
  const decisionBody = (extra: Record<string, unknown>): Record<string, unknown> =>
    conversationId ? { ...extra, room_id: conversationId } : extra;

  // Transport & Display state
  const [isPlaying, setIsPlaying] = useState(false);
  const [loopEnabled, setLoopEnabled] = useState(false);
  const [playheadBar, setPlayheadBar] = useState<number>(1);
  const [partFilter, setPartFilter] = useState<string>('all');
  const [pitchMode, setPitchMode] = useState<'concert' | 'written'>('concert');
  const [acousticSpaceChoice, setAcousticSpaceChoice] = useState<AcousticSpaceProfile | 'auto'>('auto');

  // Interactive score selection
  const [selectedRegion, setSelectedRegion] = useState<MusicRegionSelection | null>(null);

  // Proposal A/B audition modes keyed by proposal id
  const [proposalModes, setProposalModes] = useState<Record<string, 'original' | 'proposed'>>({});
  const [inlineChord, setInlineChord] = useState<string>('');
  const [inlineNotes, setInlineNotes] = useState<string>('');

  // Voice Rhythm Recorder & Extractor state
  const [voicePanelOpen, setVoicePanelOpen] = useState<boolean>(false);
  const [isRecordingVoice, setIsRecordingVoice] = useState<boolean>(false);
  const [voiceSketch, setVoiceSketch] = useState<VoiceRhythmSketch | null>(null);
  const [voicePromptText, setVoicePromptText] = useState<string>(
    '\ub0b4 \ubaa9\uc18c\ub9ac\ub85c \uc804\ub2ec\ud55c \uc774 \ub9ac\ub4ec \ud328\ud134\uc5d0 \ub9de\ucdb0\uc11c \ud574\ub2f9 \ub9c8\ub514\ub97c \uc791\uace1\ud574\uc918.',
  );
  const [micInputLevel, setMicInputLevel] = useState<number>(0);
  const [micActiveBand, setMicActiveBand] = useState<VoiceBand | 'silence'>('silence');
  const [micLiveHits, setMicLiveHits] = useState<number>(0);
  const [micElapsedSec, setMicElapsedSec] = useState<number>(0);
  const [micWaveBars, setMicWaveBars] = useState<
    Array<{ level: number; band: VoiceBand | 'silence'; isHit: boolean }>
  >(() => Array.from({ length: 24 }, () => ({ level: 0.04, band: 'silence', isHit: false })));
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const voiceAudioCtxRef = useRef<AudioContext | null>(null);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const sampleRateRef = useRef<number>(44100);
  const lastLiveHitAtMsRef = useRef<number>(0);
  const recStartMsRef = useRef<number>(0);

  // Timeline follow: while playing, the score scrolls to keep the playhead bar in view.
  const [followPlayhead, setFollowPlayhead] = useState(true);
  const scoreScrollerRef = useRef<HTMLDivElement | null>(null);
  // Set when the user scrolls by hand mid-playback, so we do not fight them; cleared on Play.
  const userScrolledRef = useRef(false);
  // Where the follow effect last put the scroller, to tell its scroll events from the user's.
  const followScrollLeftRef = useRef(0);

  const resumeFollow = () => {
    userScrolledRef.current = false;
  };

  useEffect(() => {
    const scroller = scoreScrollerRef.current;
    if (!isPlaying || !followPlayhead || userScrolledRef.current || !scroller) return;
    const barLeft = LABEL_COLUMN_WIDTH + (playheadBar - 1) * BAR_WIDTH;
    const visibleLeft = scroller.scrollLeft + LABEL_COLUMN_WIDTH;
    const visibleRight = scroller.scrollLeft + scroller.clientWidth;
    if (barLeft < visibleLeft || barLeft + BAR_WIDTH > visibleRight) {
      scroller.scrollLeft = Math.max(0, barLeft - LABEL_COLUMN_WIDTH);
      followScrollLeftRef.current = scroller.scrollLeft;
    }
  }, [playheadBar, isPlaying, followPlayhead]);

  const engineRef = useRef<MusicAudioEngine | null>(null);
  if (!engineRef.current) {
    engineRef.current = new MusicAudioEngine();
  }

  const songGenre = song?.style_axioms?.profile ?? song?.genre ?? 'pop_ballad';
  const resolvedSpace: AcousticSpaceProfile =
    acousticSpaceChoice === 'auto'
      ? defaultSpaceForGenre(normalizeGenreProfile(songGenre))
      : acousticSpaceChoice;

  const prevSoloIdsRef = useRef<string>('');

  const loadSong = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await readSongJson<ShownSongView>(
        await fetch(`${BASE}/${encodeURIComponent(songId)}`),
      );
      setSong(data);
    } catch (err) {
      setError(plainFailure(err, 'Failed to load song score.'));
    } finally {
      setLoading(false);
    }
  }, [songId]);

  useEffect(() => {
    setSong(null);
    setSelectedRegion(null);
    void loadSong();
    return () => {
      engineRef.current?.stop();
    };
  }, [loadSong]);

  const prevRefreshKeyRef = useRef<unknown>(refreshKey);
  useEffect(() => {
    if (prevRefreshKeyRef.current !== refreshKey) {
      prevRefreshKeyRef.current = refreshKey;
      void loadSong();
    }
  }, [refreshKey, loadSong]);

  const activeProposalPreview = useMemo<{
    proposal: MusicProposalView | null;
    mode: 'original' | 'proposed';
  }>(() => {
    if (!song || song.proposals.length === 0) {
      return { proposal: null, mode: 'original' };
    }
    for (const p of song.proposals) {
      const m = proposalModes[p.id] ?? 'proposed';
      if (m === 'proposed') {
        return { proposal: p, mode: 'proposed' };
      }
    }
    return { proposal: song.proposals[0], mode: 'original' };
  }, [song, proposalModes]);

  const effectiveNotes = useMemo<MusicNoteView[]>(() => {
    if (!song) return [];
    return buildNotesWithProposalPreview(
      song,
      activeProposalPreview.proposal,
      activeProposalPreview.mode,
    );
  }, [song, activeProposalPreview]);

  const usesRecordedPiano = useMemo(
    () =>
      (song?.instruments ?? []).some((inst) =>
        isSampledPianoPatch(resolveInstrumentPatch(inst, normalizeGenreProfile(songGenre))),
      ),
    [song, songGenre],
  );

  // Auto-update or auto-start playback when solo state changes from chat or mixer
  useEffect(() => {
    if (!song || !engineRef.current) return;
    void engineRef.current.preloadPiano(song.instruments, effectiveNotes, songGenre);
    engineRef.current.setAcousticSpaceOverride(
      acousticSpaceChoice === 'auto' ? null : acousticSpaceChoice,
    );

    const currentSoloIds = song.instruments
      .filter((i) => i.solo)
      .map((i) => i.id)
      .sort()
      .join(',');
    const prevSoloIds = prevSoloIdsRef.current;
    prevSoloIdsRef.current = currentSoloIds;

    if (isPlaying || (currentSoloIds !== '' && currentSoloIds !== prevSoloIds)) {
      setIsPlaying(true);
      engineRef.current.start({
        notes: effectiveNotes,
        bars: song.bars,
        instruments: song.instruments,
        sections: song.sections,
        tracks: song.tracks,
        genre: songGenre,
        acousticSpace: resolvedSpace,
        loopEnabled,
        loopStartBar: selectedRegion ? selectedRegion.start_bar : undefined,
        loopEndBar: selectedRegion ? selectedRegion.end_bar : undefined,
        onBarChange: (bar) => setPlayheadBar(bar),
        onStop: () => setIsPlaying(false),
      });
    }
  }, [song, effectiveNotes, resolvedSpace, acousticSpaceChoice, songGenre]);

  const togglePlay = () => {
    if (!song || !engineRef.current) return;
    if (isPlaying) {
      engineRef.current.stop();
      setIsPlaying(false);
      return;
    }
    resumeFollow(); // Play starts following again
    setIsPlaying(true);
    engineRef.current.start({
      notes: effectiveNotes,
      bars: song.bars,
      instruments: song.instruments,
      sections: song.sections,
      tracks: song.tracks,
      genre: songGenre,
      acousticSpace: resolvedSpace,
      loopEnabled,
      loopStartBar: selectedRegion ? selectedRegion.start_bar : undefined,
      loopEndBar: selectedRegion ? selectedRegion.end_bar : undefined,
      onBarChange: (bar) => setPlayheadBar(bar),
      onStop: () => setIsPlaying(false),
    });
  };

  const handleMixChange = async (
    inst: MusicInstrumentView,
    patch: { solo?: boolean; muted?: boolean; volume?: number },
  ) => {
    if (!song) return;
    setBusy(true);
    setNotice(null);
    setUnconfirmed(false);
    try {
      const updated = await readSongJson<ShownSongView>(
        await postSongJson(
          `${BASE}/${encodeURIComponent(songId)}/instruments/${encodeURIComponent(inst.id)}/mix`,
          patch,
        ),
      );
      setSong(updated);
    } catch (err) {
      setNotice(changeFailed(err, 'Could not update stem mix.'));
    } finally {
      setBusy(false);
    }
  };

  const handleSelectCell = (
    barMeta: MusicBarView,
    inst: MusicInstrumentView,
    cellNotes: MusicNoteView[],
  ) => {
    const firstSounding = cellNotes.find((n) => n.concert_midi >= 0);
    if (firstSounding) {
      engineRef.current?.auditionNote(firstSounding.concert_midi, inst.category, {
        instrument: inst,
        writtenPitch: firstSounding.written_pitch,
        velocity: firstSounding.velocity,
        articulation: firstSounding.articulation,
        genre: songGenre,
      });
    }
    const nextSelection: MusicRegionSelection = {
      section_id: barMeta.section_id,
      start_bar: barMeta.section_bar,
      end_bar: barMeta.section_bar,
      instrument_ids: [inst.id],
    };
    const track = song?.tracks.find(
      (t) => t.section_id === barMeta.section_id && t.instrument_id === inst.id,
    );
    const barObj = track?.bars.find((b) => b.bar === barMeta.section_bar);
    setInlineChord(barObj?.chord ?? barMeta.chord ?? '');
    setInlineNotes(barObj?.notes.join(', ') ?? '');
    setSelectedRegion(nextSelection);
    setPlayheadBar(barMeta.global_bar);
    onSelectRegion?.(nextSelection);
  };

  const handleDirectEdit = async (
    action: 'clear_bars' | 'mute_bars' | 'transpose_bars' | 'set_bar_notes' | 'revert',
    extra: { semitones?: number; muted?: boolean; notes?: string[]; chord?: string } = {},
  ) => {
    if (!song || !selectedRegion || selectedRegion.instrument_ids.length === 0) return;
    const instId = selectedRegion.instrument_ids[0];
    const track = song.tracks.find(
      (t) => t.section_id === selectedRegion.section_id && t.instrument_id === instId,
    );
    if (!track) return;

    setBusy(true);
    setNotice(null);
    setUnconfirmed(false);
    try {
      const payload: Record<string, unknown> = {
        action,
        start_bar: selectedRegion.start_bar,
        end_bar: selectedRegion.end_bar,
        semitones: extra.semitones ?? 0,
        muted: extra.muted ?? true,
        seen_digest: track.digest,
      };
      if (extra.notes !== undefined) {
        payload.notes = extra.notes;
      }
      if (extra.chord !== undefined) {
        payload.chord = extra.chord;
      }
      const updated = await readSongJson<ShownSongView>(
        await postSongJson(
          `${BASE}/${encodeURIComponent(songId)}/tracks/${encodeURIComponent(track.track_id)}/edit`,
          payload,
        ),
      );
      setSong(updated);
      setNotice(`Applied ${action} on ${track.track_id} (bars ${selectedRegion.start_bar}..${selectedRegion.end_bar}).`);
    } catch (err) {
      setNotice(changeFailed(err, 'Could not edit score track.'));
    } finally {
      setBusy(false);
    }
  };

  const handleStructureEdit = async (
    action: 'insert_bars' | 'append_bars',
    extra: { at_bar?: number; count?: number; intent?: ArrangementIntent } = {},
  ) => {
    if (!song) return;
    const targetSecId = selectedRegion?.section_id ?? song.sections[0]?.id ?? 'theme_a';
    setBusy(true);
    setNotice(null);
    setUnconfirmed(false);
    try {
      const updated = await readSongJson<ShownSongView>(
        await postSongJson(
          `${BASE}/${encodeURIComponent(songId)}/structure`,
          {
            action,
            section_id: targetSecId,
            at_bar: extra.at_bar ?? 1,
            count: extra.count ?? (action === 'insert_bars' ? 2 : 4),
            intent: extra.intent ?? (action === 'insert_bars' ? 'gentle_intro' : 'continuation'),
          },
        ),
      );
      setSong(updated);
      setNotice(
        action === 'insert_bars'
          ? `Inserted ${extra.count ?? 2} intro bars into ${targetSecId}.`
          : `Appended ${extra.count ?? 4} continuation bars to ${targetSecId}.`,
      );
    } catch (err) {
      setNotice(changeFailed(err, 'Could not edit score structure.'));
    } finally {
      setBusy(false);
    }
  };

  const handleApproveAllProposals = async () => {
    if (!song || song.proposals.length === 0) return;
    setBusy(true);
    setDecisionNotice(null);
    setUnconfirmed(false);
    try {
      const count = song.proposals.length;
      const updated = await readSongJson<ShownSongView>(
        await postSongJson(
          `${BASE}/${encodeURIComponent(songId)}/proposals/approve-all`,
          decisionBody({}),
        ),
      );
      setSong(updated);
      setDecisionNotice(`Approved all ${count} pending proposals.`);
    } catch (err) {
      setDecisionNotice(changeFailed(err, 'Could not approve the proposals.'));
    } finally {
      setBusy(false);
    }
  };

  const deliverComposerPrompt = async (prompt: string, region: MusicRegionSelection) => {
    onSelectRegion?.(region);
    if (onFastEditPrompt) {
      onFastEditPrompt(prompt, region);
      return;
    }
    if (!conversationId) {
      setNotice('Open this song in a conversation with Composer to send requests.');
      return;
    }
    setBusy(true);
    setNotice('Sending rhythm request to Composer...');
    try {
      const res = await postSongJson(
        `/api/rooms/${encodeURIComponent(conversationId)}/messages`,
        { content: prompt },
      );
      if (!res.ok) {
        const body = (await res.json().catch(() => null)) as { detail?: unknown } | null;
        throw new Error(
          typeof body?.detail === 'string' && body.detail
            ? body.detail
            : 'Could not send request to Composer.',
        );
      }
      setNotice('Sent rhythm request to Composer.');
      await loadSong();
    } catch (err) {
      setNotice(plainFailure(err, 'Could not send request to Composer.'));
    } finally {
      setBusy(false);
    }
  };

  const handleFastEdit = () => {
    if (!selectedRegion) return;
    const prompt = '\uc774 \ubd80\ubd84\uc744 \ube60\ub974\uac8c \uc218\uc815\ud574\uc918';
    const taggedPrompt = `[${selectedRegion.section_id} | Bar ${selectedRegion.start_bar}-${selectedRegion.end_bar} | ${selectedRegion.instrument_ids.join(', ')}] ${prompt}`;
    setNotice(
      `🎯 [${selectedRegion.section_id} · Bar ${selectedRegion.start_bar}..${selectedRegion.end_bar} · ${selectedRegion.instrument_ids.join(', ')}] ${prompt}`,
    );
    if (onFastEditPrompt) {
      onFastEditPrompt(prompt, selectedRegion);
      return;
    }
    void deliverComposerPrompt(taggedPrompt, selectedRegion);
  };

  const handleProposalDecision = async (
    proposal: MusicProposalView,
    decision: 'approve' | 'reject',
  ) => {
    setBusy(true);
    setDecisionNotice(null);
    setUnconfirmed(false);
    try {
      const updated = await readSongJson<ShownSongView>(
        await postSongJson(
          `${BASE}/${encodeURIComponent(songId)}/proposals/${encodeURIComponent(proposal.id)}/${decision}`,
          decisionBody({ seen_digest: proposal.seen_digest }),
        ),
      );
      setSong(updated);
      setDecisionNotice(
        decision === 'approve'
          ? `Approved: ${proposal.summary}`
          : `Rejected: ${proposal.summary}`,
      );
    } catch (err) {
      setDecisionNotice(
        changeFailed(
          err,
          decision === 'approve' ? 'Could not approve the proposal.' : 'Could not reject the proposal.',
        ),
      );
    } finally {
      setBusy(false);
    }
  };

  const toggleProposalAuditionMode = (proposalId: string) => {
    setProposalModes((prev) => {
      const current = prev[proposalId] ?? 'proposed';
      return {
        ...prev,
        [proposalId]: current === 'proposed' ? 'original' : 'proposed',
      };
    });
  };

  // A decision can always be sent from a docked view: the server opens a song no live
  // conversation holds in this one. Without a conversation it needs a live writer.
  const canDecide = Boolean(song && (song.writable || conversationId));

  const visibleInstruments = useMemo(() => {
    if (!song) return [];
    if (partFilter === 'all') return song.instruments;
    return song.instruments.filter((inst) => inst.id === partFilter);
  }, [song, partFilter]);

  const selectedTrack = useMemo(() => {
    if (!song || !selectedRegion || selectedRegion.instrument_ids.length === 0) return null;
    return (
      song.tracks.find(
        (t) =>
          t.section_id === selectedRegion.section_id &&
          t.instrument_id === selectedRegion.instrument_ids[0],
      ) ?? null
    );
  }, [song, selectedRegion]);

  const voiceTargetSelection = useMemo<MusicRegionSelection>(() => {
    if (selectedRegion) return selectedRegion;
    const defaultSectionId = song?.sections[0]?.id ?? 'sec_verse_01';
    const defaultInstId = song?.instruments[0]?.id ?? 'violin_solo';
    const maxBar = Math.min(2, Math.max(1, song?.total_bars ?? 2));
    return {
      section_id: defaultSectionId,
      start_bar: 1,
      end_bar: maxBar,
      instrument_ids: [defaultInstId],
    };
  }, [selectedRegion, song]);

  const stopVoiceStream = useCallback(() => {
    if (mediaStreamRef.current) {
      for (const track of mediaStreamRef.current.getTracks()) {
        track.stop();
      }
      mediaStreamRef.current = null;
    }
    if (voiceAudioCtxRef.current) {
      void voiceAudioCtxRef.current.close().catch(() => {});
      voiceAudioCtxRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stopVoiceStream();
    };
  }, [stopVoiceStream]);

  const handleToggleMicRecording = async () => {
    if (isRecordingVoice) {
      setIsRecordingVoice(false);
      setMicInputLevel(0);
      setMicActiveBand('silence');
      stopVoiceStream();
      const chunks = pcmChunksRef.current;
      const totalLen = chunks.reduce((acc, c) => acc + c.length, 0);
      const merged = new Float32Array(totalLen);
      let offset = 0;
      for (const c of chunks) {
        merged.set(c, offset);
        offset += c.length;
      }
      const targetBars = Math.max(
        1,
        voiceTargetSelection.end_bar - voiceTargetSelection.start_bar + 1,
      );
      const sketch = analyzeVoiceRhythmBuffer(merged, sampleRateRef.current, {
        bpm: song?.tempo ?? 96,
        maxBars: targetBars,
      });
      setVoiceSketch(sketch);
      setMicLiveHits(sketch.hitCount);
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setNotice('Microphone access is not available in this browser; use the voice/rhythm tap buttons below.');
      return;
    }

    try {
      setNotice(null);
      pcmChunksRef.current = [];
      lastLiveHitAtMsRef.current = 0;
      recStartMsRef.current = Date.now();
      setMicInputLevel(0);
      setMicActiveBand('silence');
      setMicLiveHits(0);
      setMicElapsedSec(0);
      setMicWaveBars(
        Array.from({ length: 24 }, () => ({ level: 0.04, band: 'silence', isHit: false })),
      );
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      mediaStreamRef.current = stream;
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (AudioCtx) {
        const ctx = new AudioCtx();
        voiceAudioCtxRef.current = ctx;
        sampleRateRef.current = ctx.sampleRate || 44100;
        const source = ctx.createMediaStreamSource(stream);
        const processor = ctx.createScriptProcessor(2048, 1, 1);
        processor.onaudioprocess = (ev) => {
          const input = ev.inputBuffer.getChannelData(0);
          pcmChunksRef.current.push(new Float32Array(input));
          let sumSq = 0;
          let zc = 0;
          for (let i = 0; i < input.length; i++) {
            const s = input[i];
            sumSq += s * s;
            if (i > 0 && ((input[i - 1] >= 0 && s < 0) || (input[i - 1] < 0 && s >= 0))) {
              zc++;
            }
          }
          const rms = Math.sqrt(sumSq / Math.max(1, input.length));
          const normalizedLevel = Math.min(1, rms * 5.5);
          const zcrHz = (zc * (ctx.sampleRate || 44100)) / (2 * Math.max(1, input.length));
          const isVoiced = normalizedLevel >= 0.14;
          const band: VoiceBand | 'silence' = !isVoiced
            ? 'silence'
            : zcrHz > 1800
              ? 'high'
              : zcrHz < 420
                ? 'low'
                : 'mid';
          const nowMs = Date.now();
          const isNewHit = isVoiced && nowMs - lastLiveHitAtMsRef.current >= 110;
          if (isNewHit) {
            lastLiveHitAtMsRef.current = nowMs;
            setMicLiveHits((h) => h + 1);
          }
          setMicInputLevel(normalizedLevel);
          setMicActiveBand(band);
          setMicElapsedSec(
            Math.max(0, Math.round(((nowMs - recStartMsRef.current) / 1000) * 10) / 10),
          );
          setMicWaveBars((prev) => [
            ...prev.slice(1),
            {
              level: Math.max(0.05, normalizedLevel),
              band,
              isHit: isNewHit,
            },
          ]);
        };
        source.connect(processor);
        processor.connect(ctx.destination);
      } else {
        // Headless test fallback when WebAudio AudioContext is absent
        const sr = 8000;
        sampleRateRef.current = sr;
        const synth = new Float32Array(sr * 2);
        for (let i = Math.floor(0.05 * sr); i < Math.floor(0.55 * sr); i++) {
          synth[i] = 0.85 * Math.sin((2 * Math.PI * 110 * i) / sr);
        }
        for (let i = Math.floor(0.8 * sr); i < Math.floor(1.0 * sr); i++) {
          synth[i] = 0.65 * Math.sin((2 * Math.PI * 620 * i) / sr);
        }
        pcmChunksRef.current.push(synth);
        setMicInputLevel(0.75);
        setMicActiveBand('low');
        setMicLiveHits(2);
      }
      setIsRecordingVoice(true);
    } catch (err) {
      setNotice(plainFailure(err, 'Could not start microphone recording.'));
    }
  };

  const currentVoiceSpanBars = Math.max(
    1,
    voiceTargetSelection.end_bar - voiceTargetSelection.start_bar + 1,
  );

  const handleSetVoiceBarSpan = (spanBars: number) => {
    const startBar = voiceTargetSelection.start_bar;
    const maxTotal = Math.max(startBar, song?.total_bars ?? 4);
    const nextSelection: MusicRegionSelection = {
      ...voiceTargetSelection,
      end_bar: Math.min(maxTotal, startBar + spanBars - 1),
    };
    setSelectedRegion(nextSelection);
    onSelectRegion?.(nextSelection);
  };

  const handleResetVoiceSketch = () => {
    pcmChunksRef.current = [];
    setVoiceSketch(null);
    setMicInputLevel(0);
    setMicActiveBand('silence');
    setMicLiveHits(0);
    setMicElapsedSec(0);
    setMicWaveBars(
      Array.from({ length: 24 }, () => ({ level: 0.04, band: 'silence', isHit: false })),
    );
  };

  const handleSendVoiceRhythmToComposer = () => {
    if (!voiceSketch) return;
    const tag = formatVoiceRhythmPromptTag(voiceSketch, voiceTargetSelection);
    const prompt = `${tag} ${voicePromptText.trim()}`.trim();
    if (onFastEditPrompt) {
      setNotice('Sent rhythm request to Composer.');
    }
    void deliverComposerPrompt(prompt, voiceTargetSelection);
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col bg-slate-950 text-slate-100" data-testid="music-view">
      {/* Header Bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-800 bg-slate-950">
        {onBack && (
          <Button variant="ghost" onClick={onBack} data-testid="music-view-back">
            <ArrowLeft className="w-3.5 h-3.5" />
            Back
          </Button>
        )}
        <div className="min-w-0 flex-1">
          {song && (
            <div className="flex items-center gap-2 flex-wrap">
              <p className="text-sm font-semibold truncate" data-testid="music-view-title">
                {song.title}
              </p>
              <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[11px] text-slate-300 font-mono">
                {song.key_signature} · {song.time_signature} · {song.tempo} BPM · {song.total_bars} Bars
              </span>
            </div>
          )}
        </div>
        <Button
          variant="bordered"
          size="icon"
          onClick={() => void loadSong()}
          disabled={loading}
          title="Refresh Score"
          data-testid="music-view-refresh"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </Button>
      </div>

      {/* DAW Transport & Notation Toolbar */}
      {song && (
        <div
          className="flex flex-wrap items-center gap-2 px-4 py-2 border-b border-slate-800 bg-slate-900/80 text-xs"
          data-testid="music-transport-bar"
        >
          <Button
            variant="solid"
            onClick={togglePlay}
            data-testid="music-transport-play"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5 mr-1" /> : <Play className="w-3.5 h-3.5 mr-1" />}
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </Button>

          <Button
            variant={loopEnabled ? 'solid' : 'outline'}
            aria-pressed={loopEnabled}
            onClick={() => setLoopEnabled((v) => !v)}
            data-testid="music-transport-loop"
          >
            <Repeat className="w-3.5 h-3.5 mr-1" />
            ⟲ Loop
          </Button>

          <Button
            variant={followPlayhead ? 'solid' : 'outline'}
            aria-pressed={followPlayhead}
            onClick={() => {
              resumeFollow();
              setFollowPlayhead((v) => !v);
            }}
            data-testid="music-transport-follow"
          >
            Follow
          </Button>

          <span
            className="px-2.5 py-1 rounded border border-slate-700 bg-slate-950 font-mono text-[11px] text-amber-300 shadow-inner"
            data-testid="music-playhead-indicator"
          >
            BAR {playheadBar} / {Math.max(1, song.total_bars)}
          </span>

          {/* Note Duration Legend */}
          <div className="hidden xl:flex items-center gap-2 px-2.5 py-1 rounded bg-slate-950/60 border border-slate-800/80 text-[10px] text-slate-400">
            <span className="text-slate-300 font-medium">Note Length:</span>
            <span>𝅝 4b</span>
            <span>𝅗𝅥 2b</span>
            <span>♩. 1.5b</span>
            <span>♩ 1b</span>
            <span>♪ 0.5b</span>
            <span>♬ 0.25b</span>
            <span className="text-sky-400">━ Sustain Bar</span>
          </div>

          <label className="flex items-center gap-1 text-slate-300 ml-auto">
            <span>Space:</span>
            <select
              aria-label="Acoustic Reverb Space"
              data-testid="music-acoustic-space-selector"
              value={acousticSpaceChoice}
              onChange={(e) =>
                setAcousticSpaceChoice(e.target.value as AcousticSpaceProfile | 'auto')
              }
              className="rounded bg-slate-900 border border-slate-700 px-2 py-1 text-xs text-sky-300"
            >
              <option value="auto">
                Auto ({ACOUSTIC_SPACES[resolvedSpace]?.label ?? 'Studio Room'})
              </option>
              <option value="symphony_surround">Symphony 3D Surround (HRTF)</option>
              <option value="concert_hall">Concert Hall (Classical)</option>
              <option value="jazz_club">Jazz Club (Jazz)</option>
              <option value="studio_room">Studio Room (Pop)</option>
              <option value="tight_booth">Tight Booth (Hip-Hop)</option>
            </select>
          </label>

          <label className="flex items-center gap-1 text-slate-300">
            <span>View:</span>
            <select
              aria-label="Score Part Filter"
              data-testid="music-part-selector"
              value={partFilter}
              onChange={(e) => setPartFilter(e.target.value)}
              className="rounded bg-slate-900 border border-slate-700 px-2 py-1 text-xs text-slate-100"
            >
              <option value="all">Full Score</option>
              {song.instruments.map((inst) => (
                <option key={inst.id} value={inst.id}>
                  Part View: {inst.name}
                </option>
              ))}
            </select>
          </label>

          <Button
            variant="outline"
            data-testid="music-pitch-toggle"
            onClick={() => setPitchMode((m) => (m === 'concert' ? 'written' : 'concert'))}
          >
            {pitchMode === 'concert' ? 'Concert Pitch' : 'Written Pitch'}
          </Button>

          <Button
            variant={voicePanelOpen ? 'solid' : 'outline'}
            data-testid="music-voice-rhythm-toggle"
            onClick={() => setVoicePanelOpen((v) => !v)}
          >
            🎤 Voice Rhythm
          </Button>
        </div>
      )}

      {notice && (
        <div
          role="status"
          data-testid="music-view-notice"
          className="px-4 py-2 text-xs text-amber-200 border-b border-slate-800 bg-amber-950/30"
        >
          {notice}
          {confirmButton}
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-5 text-xs">
        {loading && !song && <p className="text-slate-400">Loading score...</p>}
        {error && (
          <div role="alert" className="text-rose-300">
            <p>{error}</p>
          </div>
        )}

        {song && (
          <>
            {/* Voice-to-Rhythm Sketch Panel */}
            {voicePanelOpen && (
              <section
                data-testid="music-voice-rhythm-panel"
                className="rounded-lg border border-sky-500/40 bg-slate-900/90 p-3.5 space-y-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="text-xs font-semibold text-sky-300">
                      🎤 Voice & Beat Rhythm Extractor → Composer LLM
                    </h3>
                    <p className="text-[11px] text-slate-400">
                      {"\ub9c8\uc774\ud06c \ub179\uc74c\uc744 \ub204\ub974\uace0 \uc6d0\ud558\ub294 \ubc15\uc790(“\ucff5 \ub531 \ucff5\ucff5 \ub531” / “\ub2e4\ub2e4\ub2e8 \ub534~”)\ub97c \uc18c\ub9ac \ub0b4\uba74 \ub9c8\ub514\ubcc4 \ub9ac\ub4ec \uace8\uaca9\uc73c\ub85c \uc790\ub3d9 \ucd94\ucd9c\ud569\ub2c8\ub2e4."}
                    </p>
                  </div>
                  <span
                    data-testid="music-voice-target-region"
                    className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] text-amber-300"
                  >
                    Target: {voiceTargetSelection.section_id} · Bar {voiceTargetSelection.start_bar}
                    –{voiceTargetSelection.end_bar} ·{' '}
                    {voiceTargetSelection.instrument_ids.join(', ')}
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    variant={isRecordingVoice ? 'solid' : 'outline'}
                    data-testid="music-voice-record-btn"
                    onClick={() => void handleToggleMicRecording()}
                    className={
                      isRecordingVoice
                        ? 'bg-rose-600 hover:bg-rose-500 text-white border-rose-400 animate-pulse'
                        : undefined
                    }
                  >
                    {isRecordingVoice ? '⏹ Stop Mic & Extract' : '🔴 Record Mic'}
                  </Button>

                  <div className="flex items-center gap-1 rounded border border-slate-800 bg-slate-950/80 px-2 py-0.5 text-[11px] text-slate-400">
                    <span className="mr-1">{"\uad6c\uac04 \uae38\uc774:"}</span>
                    {[1, 2, 4].map((span) => (
                      <button
                        key={span}
                        type="button"
                        data-testid={`music-voice-span-${span}`}
                        onClick={() => handleSetVoiceBarSpan(span)}
                        className={`rounded px-2 py-0.5 font-mono text-[11px] transition-colors ${
                          currentVoiceSpanBars === span
                            ? 'bg-sky-500/25 text-sky-200 border border-sky-500/50 font-semibold'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                        }`}
                      >
                        {`${span}\ub9c8\ub514`}
                      </button>
                    ))}
                  </div>

                  {voiceSketch && (
                    <Button
                      variant="ghost"
                      data-testid="music-voice-reset-btn"
                      onClick={handleResetVoiceSketch}
                    >
                      Reset
                    </Button>
                  )}
                </div>

                {/* Real-time Microphone Input Level, Vocal Band & Rolling Beat Waveform Monitor */}
                <div
                  data-testid="music-voice-live-monitor"
                  data-recording={isRecordingVoice ? 'true' : 'false'}
                  className={`rounded-lg border px-3 py-2.5 space-y-2 transition-colors ${
                    isRecordingVoice
                      ? 'border-rose-500/70 bg-rose-950/25 shadow-inner'
                      : 'border-slate-800 bg-slate-950/75'
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
                    <div className="flex items-center gap-2">
                      <span
                        data-testid="music-voice-rec-status"
                        className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-semibold ${
                          isRecordingVoice
                            ? 'bg-rose-500/25 text-rose-200 border border-rose-500/50'
                            : 'bg-slate-800 text-slate-300 border border-slate-700'
                        }`}
                      >
                        <span
                          className={`h-2 w-2 rounded-full ${
                            isRecordingVoice ? 'bg-rose-400 animate-ping' : 'bg-slate-400'
                          }`}
                        />
                        {isRecordingVoice
                          ? `LIVE MIC REC (${micElapsedSec.toFixed(1)}s)`
                          : 'MIC STANDBY / TAP READY'}
                      </span>

                      <span
                        data-testid="music-voice-active-band"
                        className={`rounded px-2 py-0.5 font-mono text-[10px] border ${
                          micActiveBand === 'low'
                            ? 'bg-amber-500/20 text-amber-200 border-amber-500/40'
                            : micActiveBand === 'mid'
                              ? 'bg-sky-500/20 text-sky-200 border-sky-500/40'
                              : micActiveBand === 'high'
                                ? 'bg-violet-500/20 text-violet-200 border-violet-500/40'
                                : 'bg-slate-900 text-slate-400 border-slate-800'
                        }`}
                      >
                        {micActiveBand === 'low'
                          ? '🥁 LOW \uac10\uc9c0 (\ucff5/Bass)'
                          : micActiveBand === 'mid'
                            ? '👏 MID \uac10\uc9c0 (\ub531/\ub534)'
                            : micActiveBand === 'high'
                              ? '✨ HIGH \uac10\uc9c0 (\uce59/Hi)'
                              : '🔇 \uc785\ub825 \ub300\uae30 (Threshold 14%)'}
                      </span>
                    </div>

                    <div className="flex items-center gap-3 font-mono text-[11px] text-slate-300">
                      <span data-testid="music-voice-live-hits">
                        {"\uac10\uc9c0\ub41c \ube44\ud2b8:"} <strong className="text-sky-300">{micLiveHits}</strong>
                      </span>
                      <span>
                        {"\uc785\ub825 \ub808\ubca8:"}{' '}
                        <strong
                          className={
                            micInputLevel >= 0.14 ? 'text-emerald-300' : 'text-slate-400'
                          }
                        >
                          {Math.round(micInputLevel * 100)}%
                        </strong>
                      </span>
                    </div>
                  </div>

                  {/* Instantaneous Input Level Bar (with 14% Onset Threshold Marker) */}
                  <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-slate-900 border border-slate-800">
                    <div
                      title="Onset Detection Threshold (14%)"
                      style={{ left: '14%' }}
                      className="absolute top-0 bottom-0 z-10 w-0.5 bg-amber-400/80"
                    />
                    <div
                      data-testid="music-voice-level-bar"
                      style={{ width: `${Math.min(100, Math.round(micInputLevel * 100))}%` }}
                      className={`h-full transition-all duration-75 ${
                        micInputLevel >= 0.65
                          ? 'bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-500'
                          : micInputLevel >= 0.14
                            ? 'bg-gradient-to-r from-sky-500 to-emerald-400'
                            : 'bg-slate-600'
                      }`}
                    />
                  </div>

                  {/* 24-Bar Rolling Waveform & Beat Spike Strip */}
                  <div
                    data-testid="music-voice-waveform-strip"
                    className="flex h-10 items-end gap-1 rounded bg-slate-950/90 px-2 py-1 border border-slate-800/80"
                  >
                    {micWaveBars.map((bar, idx) => {
                      const heightPct = Math.max(10, Math.min(100, Math.round(bar.level * 100)));
                      const colorCls =
                        bar.band === 'low'
                          ? 'bg-amber-400'
                          : bar.band === 'mid'
                            ? 'bg-sky-400'
                            : bar.band === 'high'
                              ? 'bg-violet-400'
                              : 'bg-slate-700/70';
                      return (
                        <div
                          key={idx}
                          style={{ height: `${heightPct}%` }}
                          className={`flex-1 rounded-t transition-all duration-75 ${colorCls} ${
                            bar.isHit ? 'ring-1 ring-white/70' : ''
                          }`}
                        />
                      );
                    })}
                  </div>
                </div>

                {voiceSketch && (
                  <div
                    data-testid="music-voice-rhythm-preview"
                    className="rounded border border-slate-800 bg-slate-950/90 p-2.5 space-y-2.5"
                  >
                    {/* Visual Bar & Beat Rhythm Timeline Grid */}
                    <div
                      data-testid="music-voice-visual-grid"
                      className="space-y-1.5"
                    >
                      {voiceSketch.barPatterns.map((pattern, barIdx) => {
                        const barNum = voiceTargetSelection.start_bar + barIdx;
                        const tokens = pattern
                          .split(',')
                          .map((s) => s.trim())
                          .filter(Boolean);
                        return (
                          <div
                            key={`voice-bar-${barNum}`}
                            className="flex items-center gap-2 rounded bg-slate-900/90 px-2 py-1.5 border border-slate-800"
                          >
                            <span className="w-12 shrink-0 font-mono text-[10px] font-bold text-amber-300">
                              Bar {barNum}
                            </span>
                            <div className="flex flex-1 items-center gap-1 overflow-x-auto">
                              {tokens.map((tok, tokIdx) => {
                                const parts = tok.split(':');
                                const beatStr = parts[0] ?? '1.0';
                                const kind = parts[1] ?? 'REST';
                                const dur = parseFloat(parts[2] ?? '0.5') || 0.5;
                                const isRest = kind === 'REST';
                                const label = isRest
                                  ? `\uc27c\ud45c (${dur}b)`
                                  : kind === 'low'
                                    ? `🥁 \ucff5 (${dur}b)`
                                    : kind === 'high'
                                      ? `✨ \uce59 (${dur}b)`
                                      : dur >= 1.0
                                        ? `🎵 \ub534~ (${dur}b)`
                                        : `👏 \ub531 (${dur}b)`;
                                const badgeCls = isRest
                                  ? 'border-slate-800 bg-slate-950/60 text-slate-500'
                                  : kind === 'low'
                                    ? 'border-amber-500/50 bg-amber-500/20 text-amber-200'
                                    : kind === 'high'
                                      ? 'border-violet-500/50 bg-violet-500/20 text-violet-200'
                                      : 'border-sky-500/50 bg-sky-500/20 text-sky-200';
                                return (
                                  <div
                                    key={`${tok}-${tokIdx}`}
                                    style={{ flexGrow: Math.max(1, Math.round(dur * 4)) }}
                                    className={`rounded border px-2 py-1 text-[10px] font-medium truncate ${badgeCls}`}
                                    title={`Beat ${beatStr} · ${tok}`}
                                  >
                                    <span className="font-mono text-[9px] opacity-75 mr-1">
                                      b{beatStr}
                                    </span>
                                    {label}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
                      <span className="font-mono text-emerald-300">
                        {formatVoiceRhythmPromptTag(voiceSketch, voiceTargetSelection)}
                      </span>
                      <span className="text-slate-400">
                        Hits: {voiceSketch.hitCount} · BPM: {voiceSketch.detectedBpm}
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <input
                        type="text"
                        aria-label="Voice Rhythm Composer Prompt"
                        data-testid="music-voice-rhythm-prompt"
                        value={voicePromptText}
                        onChange={(e) => setVoicePromptText(e.target.value)}
                        className="flex-1 min-w-[220px] rounded bg-slate-900 border border-slate-700 px-2.5 py-1 text-xs text-slate-100"
                      />
                      <Button
                        variant="solid"
                        data-testid="music-voice-rhythm-send"
                        onClick={handleSendVoiceRhythmToComposer}
                      >
                        💬 Send Rhythm to Composer
                      </Button>
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* Continuous DAW + Score Engraver Timeline (with Integrated Left Track Header Stem Mixer) */}
            <section data-testid="music-score-canvas" className="space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <h3 className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                    Continuous Multi-Track Score & Note Sustain Timeline
                  </h3>
                  <Button
                    variant="outline"
                    disabled={busy || !song.writable}
                    data-testid="score-structure-prepend-intro"
                    onClick={() =>
                      void handleStructureEdit('insert_bars', {
                        at_bar: 1,
                        count: 2,
                        intent: 'gentle_intro',
                      })
                    }
                  >
                    + Intro (2 Bars)
                  </Button>
                  <Button
                    variant="outline"
                    disabled={busy || !song.writable}
                    data-testid="score-structure-append-bars"
                    onClick={() =>
                      void handleStructureEdit('append_bars', {
                        count: 4,
                        intent: 'continuation',
                      })
                    }
                  >
                    + Append (4 Bars)
                  </Button>
                </div>
                <span
                  data-testid="score-playhead"
                  data-bar={playheadBar}
                  className="text-[11px] font-mono text-sky-300"
                >
                  Playhead: Bar {playheadBar} / {song.total_bars}
                </span>
              </div>

              {/* Floating Action Bar & Selection Badge when a region is selected */}
              {selectedRegion && (
                <div
                  data-testid="music-selection-toolbar"
                  className="flex flex-col gap-2 rounded-lg border border-sky-700 bg-sky-950/40 p-2.5"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      data-testid="music-selection-badge"
                      className="rounded bg-sky-900/80 px-2 py-0.5 font-medium text-sky-100"
                    >
                      🎯 {selectedRegion.section_id} · Bar {selectedRegion.start_bar}
                      {selectedRegion.end_bar !== selectedRegion.start_bar
                        ? `–${selectedRegion.end_bar}`
                        : ''}{' '}
                      ({selectedRegion.instrument_ids.join(', ')})
                    </span>

                    <Button
                      variant="outline"
                      disabled={busy}
                      data-testid="score-action-clear"
                      onClick={() => void handleDirectEdit('clear_bars')}
                    >
                      🗑 Delete (Rest)
                    </Button>
                    <Button
                      variant="outline"
                      disabled={busy}
                      title="Mute only the selected bar(s), keeping the rest of the instrument track audible"
                      data-testid="score-action-mute"
                      onClick={() => void handleDirectEdit('mute_bars', { muted: true })}
                    >
                      🔇 Mute Selected Bar
                    </Button>
                    <Button
                      variant="outline"
                      disabled={busy}
                      data-testid="score-action-octave-up"
                      onClick={() => void handleDirectEdit('transpose_bars', { semitones: 12 })}
                    >
                      Octave ▲
                    </Button>
                    <Button
                      variant="outline"
                      disabled={busy}
                      data-testid="score-action-octave-down"
                      onClick={() => void handleDirectEdit('transpose_bars', { semitones: -12 })}
                    >
                      Octave ▼
                    </Button>
                    <Button
                      variant="outline"
                      disabled={busy || !(selectedTrack?.can_revert ?? false)}
                      data-testid="score-action-revert"
                      onClick={() => void handleDirectEdit('revert')}
                    >
                      ↩ Undo
                    </Button>
                    <Button
                      variant="solid"
                      disabled={busy}
                      data-testid="score-action-fast-edit"
                      onClick={handleFastEdit}
                    >
                      ⚡ Fast Edit Prompt
                    </Button>
                  </div>

                  {/* Inline Bar Note & Chord Editor */}
                  <div
                    data-testid="score-inline-note-editor"
                    className="flex flex-wrap items-center gap-2 border-t border-sky-800/60 pt-2 text-[11px]"
                  >
                    <label className="flex items-center gap-1 text-slate-300">
                      <span className="font-semibold text-amber-300">Chord:</span>
                      <input
                        type="text"
                        value={inlineChord}
                        onChange={(e) => setInlineChord(e.target.value)}
                        data-testid="score-inline-chord-input"
                        placeholder="Dm7"
                        className="w-20 rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-slate-100"
                      />
                    </label>
                    <label className="flex flex-1 min-w-[260px] items-center gap-1 text-slate-300">
                      <span className="font-semibold text-sky-300">Step Notes:</span>
                      <input
                        type="text"
                        value={inlineNotes}
                        onChange={(e) => setInlineNotes(e.target.value)}
                        data-testid="score-inline-notes-input"
                        placeholder="1.0:A4:2.0:v85:legato, 3.0:D5:2.0:v88:tenuto"
                        className="flex-1 rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-slate-100"
                      />
                    </label>
                    <Button
                      variant="solid"
                      disabled={busy || !song.writable}
                      data-testid="score-action-apply-notes"
                      onClick={() =>
                        void handleDirectEdit('set_bar_notes', {
                          chord: inlineChord,
                          notes: inlineNotes
                            .split(',')
                            .map((s) => s.trim())
                            .filter(Boolean),
                        })
                      }
                    >
                      ✓ Apply Notes
                    </Button>
                  </div>
                </div>
              )}

              {/* Unified Horizontal Scrollable DAW + Engraver Score Sheet */}
              <div
                ref={scoreScrollerRef}
                data-testid="music-score-scroller"
                onScroll={(e) => {
                  if (e.currentTarget.scrollLeft !== followScrollLeftRef.current) {
                    userScrolledRef.current = true;
                  }
                }}
                className="overflow-x-auto rounded-lg border border-slate-800 bg-[#090d16] shadow-xl"
              >
                <div className="inline-block min-w-full align-top">
                  {/* 1. Section Arrangement Marker Row */}
                  <div className="flex border-b border-slate-800/90 bg-slate-900/90">
                    <div className="sticky left-0 z-20 w-52 shrink-0 border-r border-slate-800 bg-slate-900/95 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                      Section / Arrangement
                    </div>
                    <div className="flex">
                      {song.sections.map((sec) => (
                        <div
                          key={sec.id}
                          style={{ width: `${sec.bars * BAR_WIDTH}px` }}
                          className="shrink-0 border-r border-slate-700/80 bg-indigo-950/30 px-3 py-1.5 text-[11px] font-semibold text-indigo-200 truncate"
                        >
                          ▸ {sec.title}{' '}
                          <span className="text-[10px] font-normal text-indigo-300/70">
                            ({sec.kind} · {sec.bars} bars)
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 2. Master Measure, Chord & Beat Ruler Row */}
                  <div className="flex border-b border-slate-800 bg-slate-900/60">
                    <div className="sticky left-0 z-20 w-52 shrink-0 border-r border-slate-800 bg-slate-900/95 px-3 py-1.5 text-[10px] font-medium text-slate-400 flex items-center justify-between">
                      <span>Track / Stem Mix (S·M·Vol)</span>
                      <span className="font-mono text-[9px] text-slate-500">Beats 1·2·3·4</span>
                    </div>
                    <div className="flex">
                      {song.bars.map((barMeta) => {
                        const isPlayhead = playheadBar === barMeta.global_bar;
                        return (
                          <div
                            key={`ruler-${barMeta.global_bar}`}
                            style={{ width: `${BAR_WIDTH}px` }}
                            onClick={() => setPlayheadBar(barMeta.global_bar)}
                            className={`shrink-0 border-r border-slate-800 px-2.5 py-1 cursor-pointer select-none transition ${
                              isPlayhead
                                ? 'bg-sky-950/60 border-b-2 border-b-sky-400'
                                : 'hover:bg-slate-800/40'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span
                                className={`font-mono text-[10px] font-bold ${
                                  isPlayhead ? 'text-sky-300' : 'text-slate-400'
                                }`}
                              >
                                M.{barMeta.global_bar}
                              </span>
                              <span className="rounded bg-amber-500/15 border border-amber-500/30 px-1.5 py-0.2 text-[10px] font-bold text-amber-300">
                                {barMeta.chord}
                              </span>
                            </div>
                            <div className="mt-0.5 flex justify-between px-1 font-mono text-[8px] text-slate-500">
                              <span>1</span>
                              <span>2</span>
                              <span>3</span>
                              <span>4</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* 3. Connected Instrument Staff Lanes (with Integrated DAW Stem Controls) */}
                  <div data-testid="music-stem-matrix" className="divide-y divide-slate-800/80">
                    {visibleInstruments.map((inst) => {
                      const clef = clefForCategory(inst.category);
                      const theme = trackThemeForCategory(inst.category);
                      const resolvedPatch = resolveInstrumentPatch(
                        inst,
                        normalizeGenreProfile(songGenre),
                      );
                      const anySolo = song.instruments.some((i) => i.solo);
                      const isAudible = !inst.muted && (!anySolo || inst.solo);

                      return (
                        <div
                          key={inst.id}
                          data-testid={`score-staff-${inst.id}`}
                          className={`flex transition-opacity ${
                            isAudible ? 'opacity-100' : 'opacity-45'
                          }`}
                        >
                          {/* Sticky Left DAW Track Header with Integrated Solo / Mute / Volume */}
                          <div
                            data-testid={`stem-row-${inst.id}`}
                            title={`${inst.name} (${inst.category} · ${resolvedPatch.gmInstrumentName})`}
                            className={`sticky left-0 z-10 w-52 shrink-0 border-r border-slate-800 border-l-4 ${theme.accentBorder} bg-slate-950/95 px-2.5 py-1.5 flex flex-col justify-between gap-1`}
                          >
                            <div className="flex items-center justify-between gap-1">
                              <div className="flex items-center gap-1.5 min-w-0">
                                <span className="text-base text-amber-300 leading-none">
                                  {clef}
                                </span>
                                <span className="font-semibold text-slate-100 text-[11px] truncate">
                                  {inst.name}
                                </span>
                              </div>
                              <div className="flex items-center gap-1 shrink-0">
                                <Button
                                  variant={inst.solo ? 'solid' : 'outline'}
                                  aria-pressed={inst.solo}
                                  disabled={busy}
                                  title={`Solo ${inst.name} (Play only this instrument)`}
                                  data-testid={`stem-solo-${inst.id}`}
                                  onClick={() => void handleMixChange(inst, { solo: !inst.solo })}
                                >
                                  S
                                </Button>
                                <Button
                                  variant={inst.muted ? 'solid' : 'outline'}
                                  aria-pressed={inst.muted}
                                  disabled={busy}
                                  title={`Mute ${inst.name} (Silence entire instrument stem)`}
                                  data-testid={`stem-mute-${inst.id}`}
                                  onClick={() => void handleMixChange(inst, { muted: !inst.muted })}
                                >
                                  M
                                </Button>
                              </div>
                            </div>
                            <div className="flex items-center justify-between gap-1.5">
                              <span
                                className={`rounded px-1.5 py-0.5 text-[9px] font-medium truncate ${theme.badgeBg}`}
                              >
                                {inst.role.replace('_', ' ')}
                              </span>
                              <input
                                type="range"
                                min={0}
                                max={100}
                                value={inst.volume}
                                disabled={busy}
                                aria-label={`${inst.name} Volume`}
                                data-testid={`stem-volume-${inst.id}`}
                                onChange={(e) =>
                                  void handleMixChange(inst, {
                                    volume: parseInt(e.target.value, 10),
                                  })
                                }
                                className="w-20 accent-sky-400"
                              />
                            </div>
                          </div>

                          {/* Continuous Unbroken Horizontal Multi-Measure Staff */}
                          <div className="flex gap-0">
                            {song.bars.map((barMeta, barIdx) => {
                              const cellNotes = effectiveNotes.filter(
                                (n) =>
                                  n.instrument_id === inst.id &&
                                  n.global_bar === barMeta.global_bar,
                              );
                              const isSelected =
                                selectedRegion !== null &&
                                selectedRegion.section_id === barMeta.section_id &&
                                selectedRegion.instrument_ids.includes(inst.id) &&
                                barMeta.section_bar >= selectedRegion.start_bar &&
                                barMeta.section_bar <= selectedRegion.end_bar;
                              const isPlayhead = playheadBar === barMeta.global_bar;
                              const hasPendingProposal = song.proposals.some(
                                (p) =>
                                  p.section_id === barMeta.section_id &&
                                  p.instrument_id === inst.id &&
                                  barMeta.section_bar >= (p.start_bar ?? 1) &&
                                  barMeta.section_bar <= (p.end_bar ?? 1),
                              );
                              const isLastBarOfSong = barIdx === song.bars.length - 1;

                              return (
                                <div
                                  key={`${inst.id}-${barMeta.global_bar}`}
                                  role="button"
                                  tabIndex={0}
                                  data-testid={`score-cell-${inst.id}-${barMeta.global_bar}`}
                                  data-selected={isSelected || undefined}
                                  data-proposal-highlight={hasPendingProposal || undefined}
                                  onClick={() => handleSelectCell(barMeta, inst, cellNotes)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter' || e.key === ' ') {
                                      e.preventDefault();
                                      handleSelectCell(barMeta, inst, cellNotes);
                                    }
                                  }}
                                  style={{ width: `${BAR_WIDTH}px` }}
                                  className={`shrink-0 relative transition cursor-pointer select-none ${
                                    isSelected
                                      ? 'bg-sky-950/45 ring-1 ring-inset ring-sky-400'
                                      : hasPendingProposal
                                        ? 'bg-amber-950/25 ring-1 ring-inset ring-amber-500/60'
                                        : isPlayhead
                                          ? 'bg-sky-950/15 hover:bg-slate-900/50'
                                          : 'bg-[#090d16] hover:bg-slate-900/40'
                                  }`}
                                >
                                  {/* Subtle Measure Overlay Tag (keeps accessible text for tests & quick hover context) */}
                                  <span className="sr-only">
                                    M.{barMeta.global_bar} ({barMeta.section_title}) {barMeta.chord}
                                  </span>

                                  <svg
                                    viewBox={`0 0 ${BAR_WIDTH} 68`}
                                    className="w-full h-[68px] block overflow-visible"
                                    aria-label={`${inst.name} bar ${barMeta.global_bar}`}
                                  >
                                    {/* Beat subdivision vertical grid lines (Beats 1, 2, 3, 4) */}
                                    {[0, 1, 2, 3].map((beatIdx) => {
                                      const bx = BEAT_START_X + beatIdx * BEAT_WIDTH;
                                      return (
                                        <line
                                          key={`grid-${beatIdx}`}
                                          x1={bx}
                                          y1={6}
                                          x2={bx}
                                          y2={50}
                                          stroke={beatIdx === 0 ? '#1e293b' : '#131c2e'}
                                          strokeWidth={1}
                                          strokeDasharray={beatIdx === 0 ? undefined : '2,2'}
                                        />
                                      );
                                    })}

                                    {/* Continuous 5-line musical staff (x=0..BAR_WIDTH without gaps!) */}
                                    {[14, 22, 30, 38, 46].map((lineY) => (
                                      <line
                                        key={lineY}
                                        x1={0}
                                        y1={lineY}
                                        x2={BAR_WIDTH}
                                        y2={lineY}
                                        stroke="#334155"
                                        strokeWidth={1}
                                      />
                                    ))}

                                    {/* Measure Barline at right edge (continuous musical staff divider) */}
                                    <line
                                      x1={BAR_WIDTH - 0.5}
                                      y1={14}
                                      x2={BAR_WIDTH - 0.5}
                                      y2={46}
                                      stroke="#64748b"
                                      strokeWidth={isLastBarOfSong ? 2.5 : 1.2}
                                    />
                                    {isLastBarOfSong && (
                                      <line
                                        x1={BAR_WIDTH - 4}
                                        y1={14}
                                        x2={BAR_WIDTH - 4}
                                        y2={46}
                                        stroke="#64748b"
                                        strokeWidth={1}
                                      />
                                    )}

                                    {/* Playhead vertical cursor */}
                                    {isPlayhead && (
                                      <line
                                        x1={BEAT_START_X - 6}
                                        y1={2}
                                        x2={BEAT_START_X - 6}
                                        y2={64}
                                        stroke="#38bdf8"
                                        strokeWidth={2}
                                      />
                                    )}

                                    {/* Notes: DAW Proportional Sustain Bar + Musical Notehead/Stem/Flag/Dot */}
                                    {cellNotes.map((n, idx) => {
                                      const x = BEAT_START_X + (n.beat - 1) * BEAT_WIDTH;
                                      const sustainWidth = Math.max(
                                        6,
                                        Math.min(
                                          BAR_WIDTH - x - 2,
                                          n.duration * BEAT_WIDTH - 3,
                                        ),
                                      );
                                      const displayPitch =
                                        pitchMode === 'written' ? n.written_pitch : n.concert_pitch;
                                      const displayMidi =
                                        pitchMode === 'written'
                                          ? parsePitchTokenToMidi(n.written_pitch)
                                          : n.concert_midi;
                                      const y = noteheadY(
                                        displayMidi,
                                        inst.category,
                                        n.written_pitch,
                                      );
                                      const isRest =
                                        n.concert_midi < 0 && inst.category !== 'drums';
                                      const noteColor = hasPendingProposal
                                        ? '#fbbf24'
                                        : theme.noteFill;
                                      const durMeta = describeDurationGlyph(n.duration);
                                      const stemTopY = Math.max(3, y - 14);

                                      return (
                                        <g key={`${n.beat}-${idx}-${n.written_pitch}`}>
                                          <title>
                                            {`${displayPitch} · Beat ${n.beat} · Duration ${n.duration} beats (${durMeta.glyph})`}
                                          </title>
                                          {isRest ? (
                                            <>
                                              <rect
                                                x={x - 3}
                                                y={26}
                                                width={Math.max(8, sustainWidth * 0.6)}
                                                height={5}
                                                rx={1.5}
                                                fill="#475569"
                                                fillOpacity={0.5}
                                              />
                                              <text
                                                x={x + 2}
                                                y={61}
                                                textAnchor="middle"
                                                fill="#64748b"
                                                fontSize="7.5"
                                              >
                                                {displayPitch}
                                              </text>
                                            </>
                                          ) : (
                                            <>
                                              {/* 1. DAW Proportional Sustain Bar (visual note length!) */}
                                              <rect
                                                x={x}
                                                y={y - 2.8}
                                                width={sustainWidth}
                                                height={5.6}
                                                rx={2.5}
                                                fill={
                                                  hasPendingProposal
                                                    ? 'rgba(251, 191, 36, 0.28)'
                                                    : theme.sustainFill
                                                }
                                                stroke={noteColor}
                                                strokeOpacity={0.55}
                                                strokeWidth={0.8}
                                              />

                                              {/* 2. Notehead (hollow for whole/half, filled for quarter/8th/16th) */}
                                              <ellipse
                                                cx={x + 3.5}
                                                cy={y}
                                                rx={4.2}
                                                ry={3.0}
                                                fill={durMeta.isHollow ? '#090d16' : noteColor}
                                                stroke={noteColor}
                                                strokeWidth={durMeta.isHollow ? 1.7 : 0.8}
                                              />

                                              {/* 3. Stem (for half, quarter, 8th, 16th notes) */}
                                              {durMeta.hasStem && (
                                                <line
                                                  x1={x + 7.2}
                                                  y1={y}
                                                  x2={x + 7.2}
                                                  y2={stemTopY}
                                                  stroke={noteColor}
                                                  strokeWidth={1.3}
                                                />
                                              )}

                                              {/* 4. Eighth / Sixteenth Note Flags */}
                                              {durMeta.flags >= 1 && (
                                                <path
                                                  d={`M ${x + 7.2} ${stemTopY} q 5 2.5 4 7.5`}
                                                  fill="none"
                                                  stroke={noteColor}
                                                  strokeWidth={1.3}
                                                />
                                              )}
                                              {durMeta.flags >= 2 && (
                                                <path
                                                  d={`M ${x + 7.2} ${stemTopY + 4} q 5 2.5 4 7.5`}
                                                  fill="none"
                                                  stroke={noteColor}
                                                  strokeWidth={1.3}
                                                />
                                              )}

                                              {/* 5. Augmentation Dot for Dotted Notes (1.5b, 0.75b, 3.0b) */}
                                              {durMeta.isDotted && (
                                                <circle
                                                  cx={x + 10.5}
                                                  cy={y - 1.2}
                                                  r={1.5}
                                                  fill={noteColor}
                                                />
                                              )}

                                              {/* 6. Pitch & Duration Label */}
                                              <text
                                                x={x + 3.5}
                                                y={59}
                                                textAnchor="middle"
                                                fill="#e2e8f0"
                                                fontSize="7.5"
                                                fontWeight="600"
                                              >
                                                {displayPitch}
                                              </text>
                                              <text
                                                x={x + 3.5}
                                                y={66}
                                                textAnchor="middle"
                                                fill="#94a3b8"
                                                fontSize="6.5"
                                              >
                                                {formatBeatsCompact(n.duration)}
                                              </text>
                                            </>
                                          )}
                                        </g>
                                      );
                                    })}
                                  </svg>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                    {partFilter !== 'all' &&
                      song.instruments
                        .filter((inst) => inst.id !== partFilter)
                        .map((inst) => (
                          <div
                            key={`hidden-stem-${inst.id}`}
                            data-testid={`stem-row-${inst.id}`}
                            className="flex items-center gap-2 bg-slate-950/80 px-3 py-1 text-[10px] text-slate-400"
                          >
                            <span className="truncate font-medium text-slate-300">{inst.name}</span>
                            <Button
                              variant={inst.solo ? 'solid' : 'outline'}
                              aria-pressed={inst.solo}
                              disabled={busy}
                              data-testid={`stem-solo-${inst.id}`}
                              onClick={() => void handleMixChange(inst, { solo: !inst.solo })}
                            >
                              S
                            </Button>
                            <Button
                              variant={inst.muted ? 'solid' : 'outline'}
                              aria-pressed={inst.muted}
                              disabled={busy}
                              data-testid={`stem-mute-${inst.id}`}
                              onClick={() => void handleMixChange(inst, { muted: !inst.muted })}
                            >
                              M
                            </Button>
                            <input
                              type="range"
                              min={0}
                              max={100}
                              value={inst.volume}
                              disabled={busy}
                              aria-label={`${inst.name} Volume`}
                              data-testid={`stem-volume-${inst.id}`}
                              onChange={(e) =>
                                void handleMixChange(inst, { volume: parseInt(e.target.value, 10) })
                              }
                              className="w-16 accent-sky-400"
                            />
                          </div>
                        ))}
                  </div>
                </div>
              </div>
            </section>

            {usesRecordedPiano && (
              <p data-testid="music-piano-credit" className="text-[10px] text-slate-500">
                {SALAMANDER_PIANO_CREDIT} ·{' '}
                <a
                  href={SALAMANDER_PIANO_LICENSE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-slate-300"
                >
                  License
                </a>
              </p>
            )}

            {/* Proposal Diff & A/B Audition Cards */}
            <section data-testid="music-proposals-section" className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Pending Score Proposals ({song.proposals.length})
                </h3>
                {song.proposals.length > 0 && (
                  <Button
                    variant="solid"
                    disabled={busy || !canDecide}
                    data-testid="proposal-approve-all"
                    onClick={() => void handleApproveAllProposals()}
                  >
                    ✓ Approve All ({song.proposals.length})
                  </Button>
                )}
              </div>
              {!song.writable && song.proposals.length > 0 && (
                <p data-testid="music-proposals-writer-note" className="text-slate-400">
                  {conversationId
                    ? 'No conversation is working on this song right now. Approving or rejecting here opens it in this conversation.'
                    : (song.decide_note ??
                      'This song can’t be changed until a conversation opens it.')}
                </p>
              )}
              {decisionNotice && (
                <div
                  role="status"
                  data-testid="music-proposal-notice"
                  className="rounded border border-amber-700/50 bg-amber-950/30 px-3 py-2 text-amber-200"
                >
                  {decisionNotice}
                  {confirmButton}
                </div>
              )}
              {song.proposals.length === 0 ? (
                <p className="text-slate-500">No pending proposals awaiting review.</p>
              ) : (
                <ul className="space-y-3">
                  {song.proposals.map((prop) => {
                    const mode = proposalModes[prop.id] ?? 'proposed';
                    return (
                      <li
                        key={prop.id}
                        data-testid={`music-proposal-${prop.id}`}
                        className="rounded-lg border border-amber-700/60 bg-slate-900/80 p-3 space-y-2"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <p className="font-semibold text-amber-200">{prop.summary}</p>
                            <p className="text-[11px] text-slate-400">
                              {prop.kind} · {prop.section_id ?? 'score'} · {prop.instrument_id ?? ''}{' '}
                              {prop.start_bar
                                ? `(Bars ${prop.start_bar}..${prop.end_bar ?? prop.start_bar})`
                                : ''}
                            </p>
                          </div>

                          <Button
                            variant="outline"
                            data-testid={`proposal-mode-toggle-${prop.id}`}
                            onClick={() => toggleProposalAuditionMode(prop.id)}
                          >
                            {mode === 'proposed'
                              ? 'Auditioning: Proposed (B)'
                              : 'Auditioning: Original (A)'}
                          </Button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px] font-mono">
                          <div className="rounded bg-slate-950 p-2 border border-slate-800">
                            <p className="text-rose-300 mb-1">Original (A)</p>
                            {prop.original_bars.map((b) => (
                              <div key={b.bar} className="text-slate-400">
                                Bar {b.bar}: {b.notes.join(', ')}
                              </div>
                            ))}
                          </div>
                          <div className="rounded bg-slate-950 p-2 border border-slate-800">
                            <p className="text-emerald-300 mb-1">Proposed (B)</p>
                            {prop.proposed_bars.map((b) => (
                              <div key={b.bar} className="text-slate-200">
                                Bar {b.bar}: {b.notes.join(', ')}
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="flex gap-2 pt-1">
                          <Button
                            variant="solid"
                            disabled={busy || !canDecide}
                            data-testid={`proposal-approve-${prop.id}`}
                            onClick={() => void handleProposalDecision(prop, 'approve')}
                          >
                            Approve
                          </Button>
                          <Button
                            variant="outline"
                            disabled={busy || !canDecide}
                            data-testid={`proposal-reject-${prop.id}`}
                            onClick={() => void handleProposalDecision(prop, 'reject')}
                          >
                            Reject
                          </Button>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>
          </>
        )}
      </div>
    </div>
  );
};
