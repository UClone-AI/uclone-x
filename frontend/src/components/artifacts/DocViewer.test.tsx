import { afterEach, beforeEach, describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { DocViewer, FILES_READ_FAILED, FILE_READ_FAILED } from './DocViewer';
import { expectPlain } from '../../test/plainCopy';
import type { RoomArtifact, RoomArtifacts } from '../../lib/roomDock';

const art = (over: Partial<RoomArtifact>): RoomArtifact => ({
  id: 'art_1',
  path: 'docs/report.md',
  name: 'report.md',
  title: 'report.md',
  type: 'document',
  participant_id: 'scout',
  tool_name: 'write_file',
  turn_id: 't1',
  tool_call_id: 'c1',
  written_at: '2026-09-22T10:00:00Z',
  write_count: 1,
  writers: ['scout'],
  exists: true,
  exists_reason: null,
  size_bytes: 512,
  ...over,
});

const SCOPE_NOTE =
  'This list shows files the clones saved by name; a file written another way, such as by a shell command or a helper, may not appear here.';

const listing = (over: Partial<RoomArtifacts>): RoomArtifacts => ({
  room_id: 'room-a',
  artifacts: [],
  total: 0,
  unattributed_writes: 0,
  unattributed_note: null,
  unrecorded_turns: 0,
  unrecorded_note: null,
  unsaved_turns: 0,
  record_gaps: [],
  scope_note: SCOPE_NOTE,
  turn_running: false,
  reason: null,
  ...over,
});

let answers: Record<string, unknown>;
let requested: string[];

beforeEach(() => {
  answers = {};
  requested = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      requested.push(url);
      const body = answers[url];
      if (body === undefined) {
        return new Response(JSON.stringify({ detail: `no stub for ${url}` }), { status: 404 });
      }
      return typeof body === 'string'
        ? new Response(body, { status: 200 })
        : new Response(JSON.stringify(body), { status: 200 });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const LIST_A = '/api/rooms/room-a/artifacts';

describe('DocViewer, scoped to the conversation (#1354, #1356)', () => {
  it('lists only the files this conversation wrote, read by room id', async () => {
    answers[LIST_A] = listing({ artifacts: [art({})], total: 1 });
    answers['/api/artifacts/content?path=docs%2Freport.md'] = '# Audio Interface Report';
    render(<DocViewer roomId="room-a" />);

    expect(await screen.findByText('Audio Interface Report')).toBeInTheDocument();
    expect(requested[0]).toBe(LIST_A);
    expect(requested.some((u) => u.includes('session_id'))).toBe(false);
    expect(screen.getByTestId('artifact-selector')).toHaveValue('docs/report.md');
  });

  it('re-reads when the conversation changes', async () => {
    answers[LIST_A] = listing({ artifacts: [art({})] });
    answers['/api/rooms/room-b/artifacts'] = listing({
      room_id: 'room-b',
      artifacts: [art({ path: 'b/plan.md', name: 'plan.md' })],
    });
    answers['/api/artifacts/content?path=docs%2Freport.md'] = 'A text';
    answers['/api/artifacts/content?path=b%2Fplan.md'] = 'B text';
    const { rerender } = render(<DocViewer roomId="room-a" />);
    await screen.findByText('A text');
    rerender(<DocViewer roomId="room-b" />);
    await screen.findByText('B text');
    expect(screen.getByTestId('artifact-selector')).toHaveValue('b/plan.md');
    expect(screen.queryByRole('option', { name: /report\.md/ })).toBeNull();
  });

  it('says why the list is empty, in the Core’s words (P6)', async () => {
    const reason = `No files are listed yet. ${SCOPE_NOTE} It may also be missing files because a turn is still running and its files are listed when it finishes.`;
    answers[LIST_A] = listing({ reason, turn_running: true });
    render(<DocViewer roomId="room-a" />);
    expect(await screen.findByTestId('doc-reason')).toHaveTextContent(reason);
    // The empty list's reason already carries the scope and the gaps; they are not repeated.
    expect(screen.queryByTestId('doc-list-scope')).toBeNull();
  });

  // Killed by: frontend/src/components/artifacts/DocViewer.tsx :: ? listing.reason ?? `No files are listed yet. ${listing.scope_note ?? ''}`.trim()
  // Becomes: ? listing.reason ?? 'No file has been written in this conversation yet.'
  it('says an empty list is only empty, with what it covers, when the Core gives no reason', async () => {
    answers[LIST_A] = listing({});
    render(<DocViewer roomId="room-a" />);
    expect(await screen.findByTestId('doc-reason')).toHaveTextContent(
      `No files are listed yet. ${SCOPE_NOTE}`,
    );
    expect(screen.getByRole('option', { name: '(No files listed)' })).toBeInTheDocument();
  });

  // Killed by: frontend/src/components/artifacts/DocViewer.tsx :: {gaps.length > 0 && (
  // Becomes: {false && gaps.length > 0 && (
  it('beside a list, says what it covers and the gaps the Core knows of, in its words', async () => {
    answers[LIST_A] = listing({
      artifacts: [art({})],
      unattributed_writes: 2,
      unrecorded_turns: 1,
      record_gaps: [
        '2 tool calls in this conversation may have written files without naming them',
        '1 turn ended before its tool calls were saved',
      ],
    });
    answers['/api/artifacts/content?path=docs%2Freport.md'] = 'text';
    render(<DocViewer roomId="room-a" />);
    expect(await screen.findByTestId('doc-scope-note')).toHaveTextContent(SCOPE_NOTE);
    const gaps = screen.getByTestId('doc-record-gaps');
    expect(gaps).toHaveTextContent('It may also be missing files because:');
    expect(gaps).toHaveTextContent('may have written files without naming them');
    expect(gaps).toHaveTextContent('ended before its tool calls were saved');
    expect(screen.queryByTestId('doc-turn-running')).toBeNull();
  });

  it('names no gap it was not given, and says so while a turn is still running', async () => {
    answers[LIST_A] = listing({ artifacts: [art({})], turn_running: true });
    answers['/api/artifacts/content?path=docs%2Freport.md'] = 'text';
    render(<DocViewer roomId="room-a" />);
    expect(await screen.findByTestId('doc-turn-running')).toHaveTextContent(
      'A turn is still running; the files it saves are listed when it finishes.',
    );
    expect(screen.getByTestId('doc-scope-note')).toHaveTextContent(SCOPE_NOTE);
    expect(screen.queryByTestId('doc-record-gaps')).toBeNull();
  });

  it('says it could not check a file, in the Core’s words, and does not read it', async () => {
    answers[LIST_A] = listing({
      artifacts: [art({ exists: null, exists_reason: 'This file is outside the project, so it was not checked.' })],
    });
    render(<DocViewer roomId="room-a" />);
    expect(await screen.findByTestId('doc-unchecked-file')).toHaveTextContent(
      'outside the project, so it was not checked',
    );
    expect(requested.some((u) => u.startsWith('/api/artifacts/content'))).toBe(false);
  });

  it('fronts the file it was asked to open, and follows a later request', async () => {
    answers[LIST_A] = listing({
      artifacts: [art({}), art({ id: 'art_2', path: 'out/plan.md', name: 'plan.md' })],
    });
    answers['/api/artifacts/content?path=docs%2Freport.md'] = 'report body';
    answers['/api/artifacts/content?path=out%2Fplan.md'] = 'plan body';
    const { rerender } = render(<DocViewer roomId="room-a" selectedArtifactPath="out/plan.md" />);
    expect(await screen.findByText('plan body')).toBeInTheDocument();
    expect(screen.getByTestId('artifact-selector')).toHaveValue('out/plan.md');

    rerender(<DocViewer roomId="room-a" selectedArtifactPath="docs/report.md" />);
    expect(await screen.findByText('report body')).toBeInTheDocument();
  });

  it('reports a choice made in its own selector upward', async () => {
    answers[LIST_A] = listing({
      artifacts: [art({}), art({ id: 'art_2', path: 'out/plan.md', name: 'plan.md' })],
    });
    answers['/api/artifacts/content?path=docs%2Freport.md'] = 'report body';
    answers['/api/artifacts/content?path=out%2Fplan.md'] = 'plan body';
    const onSelectArtifact = vi.fn();
    render(<DocViewer roomId="room-a" onSelectArtifact={onSelectArtifact} />);
    await screen.findByText('report body');
    fireEvent.change(screen.getByTestId('artifact-selector'), { target: { value: 'out/plan.md' } });
    expect(onSelectArtifact).toHaveBeenCalledWith('out/plan.md');
    expect(await screen.findByText('plan body')).toBeInTheDocument();
  });

  it('says so when a listed file is no longer on disk', async () => {
    answers[LIST_A] = listing({ artifacts: [art({ exists: false, size_bytes: null })] });
    render(<DocViewer roomId="room-a" />);
    expect(await screen.findByTestId('doc-missing-file')).toHaveTextContent(
      'no longer on disk',
    );
  });

  it('renders an image file as a preview', async () => {
    answers[LIST_A] = listing({
      artifacts: [art({ path: 'artifacts/images/scenic.png', name: 'scenic.png', type: 'image' })],
    });
    render(<DocViewer roomId="room-a" />);
    await screen.findByTestId('image-artifact-preview');
    expect(screen.getByRole('img')).toHaveAttribute(
      'src',
      '/api/artifacts/content?path=artifacts%2Fimages%2Fscenic.png',
    );
    expect(screen.getByText('Open original in new tab')).toBeInTheDocument();
  });

  it('makes no request and says why when no conversation is open', async () => {
    render(<DocViewer roomId={null} />);
    expect(screen.getByTestId('doc-reason')).toHaveTextContent('No conversation is open');
    await waitFor(() => expect(requested).toEqual([]));
  });
});

/**
 * A failed read, as the browser meets it (#1435). Nothing is cleaned before Docs sees it: an
 * unreachable runtime is a real `fetch` to a closed port, and a crashed one is a real
 * `Response` carrying the plain-text 500 the server sends.
 */
describe('DocViewer: a failed read shows no transport text (#1435)', () => {
  const CONTENT_A = '/api/artifacts/content?path=docs%2Freport.md';
  const plain500 = () =>
    new Response('Internal Server Error', {
      status: 500,
      statusText: 'Internal Server Error',
      headers: { 'content-type': 'text/plain; charset=utf-8' },
    });

  // Killed by: frontend/src/components/artifacts/DocViewer.tsx :: (listRead.fault.detail ?? FILES_READ_FAILED)
  // Becomes: (listRead.error ?? FILES_READ_FAILED)
  it('an unreachable runtime reads as a plain sentence for the file list', async () => {
    vi.unstubAllGlobals();
    const realFetch = globalThis.fetch;
    // Port 1 on loopback: nothing listens there, so the connection is refused.
    vi.stubGlobal('fetch', (url: string) => realFetch(`http://127.0.0.1:1${url}`));

    render(<DocViewer roomId="room-a" />);
    await waitFor(
      () => expect(screen.getByTestId('doc-reason')).toHaveTextContent(FILES_READ_FAILED),
      { timeout: 3000 },
    );
    expectPlain(screen.getByTestId('doc-reason').textContent);
  });

  // Killed by: frontend/src/lib/roomDock.ts :: let detail: string | null = null;
  // Becomes: let detail: string | null = message;
  it('a plain-text 500 reads as a plain sentence for the file list, not its status line', async () => {
    vi.stubGlobal('fetch', async () => plain500());

    render(<DocViewer roomId="room-a" />);
    const reason = await screen.findByTestId('doc-reason');
    await waitFor(() => expect(reason).toHaveTextContent(FILES_READ_FAILED));
    expectPlain(reason.textContent);
  });

  // Killed by: frontend/src/components/artifacts/DocViewer.tsx :: {FILE_READ_FAILED}
  // Becomes: {error}
  it('an unreachable runtime reads as a plain sentence for a file', async () => {
    vi.unstubAllGlobals();
    const realFetch = globalThis.fetch;
    vi.stubGlobal('fetch', (url: string) =>
      url === LIST_A
        ? Promise.resolve(
            new Response(JSON.stringify(listing({ artifacts: [art({})], total: 1 })), { status: 200 }),
          )
        : // Port 1 on loopback: nothing listens there, so the connection is refused.
          realFetch(`http://127.0.0.1:1${url}`),
    );

    render(<DocViewer roomId="room-a" />);
    const failed = await screen.findByTestId('doc-read-failed', undefined, { timeout: 3000 });
    expect(failed).toHaveTextContent(FILE_READ_FAILED);
    expectPlain(failed.textContent);
  });

  // Killed by: frontend/src/components/artifacts/DocViewer.tsx :: {FILE_READ_FAILED}
  // Becomes: {error}
  it('a plain-text 500 reads as a plain sentence for a file, not its status line', async () => {
    vi.stubGlobal('fetch', async (url: string) =>
      url === CONTENT_A
        ? plain500()
        : new Response(JSON.stringify(listing({ artifacts: [art({})], total: 1 })), { status: 200 }),
    );

    render(<DocViewer roomId="room-a" />);
    const failed = await screen.findByTestId('doc-read-failed');
    expect(failed).toHaveTextContent(FILE_READ_FAILED);
    expectPlain(failed.textContent);
  });

  // Killed by: frontend/src/components/artifacts/DocViewer.tsx :: (listRead.fault.detail ?? FILES_READ_FAILED)
  // Becomes: (FILES_READ_FAILED)
  it('a refusal in the Core’s own plain words is shown as it gave them', async () => {
    const detail = 'No conversation with that name is open.';
    vi.stubGlobal(
      'fetch',
      async () =>
        new Response(JSON.stringify({ detail }), {
          status: 404,
          headers: { 'content-type': 'application/json' },
        }),
    );

    render(<DocViewer roomId="room-a" />);
    const reason = await screen.findByTestId('doc-reason');
    await waitFor(() => expect(reason).toHaveTextContent(detail));
    expectPlain(reason.textContent);
  });

  it('renders image metadata card when companion json exists', async () => {
    answers[LIST_A] = listing({
      artifacts: [
        art({ path: 'artifacts/images/img_a1b2c3.png', name: 'img_a1b2c3.png' }),
        art({ path: 'artifacts/images/img_a1b2c3.json', name: 'img_a1b2c3.json' }),
      ],
      total: 2,
    });
    answers['/api/artifacts/content?path=artifacts%2Fimages%2Fimg_a1b2c3.json'] = {
      id: 'a1b2c3',
      prompt: 'a scenic mountain at sunset',
      seed: 42,
      engine: 'diffusers-sdxl',
      style: 'artistic',
      width: 1024,
      height: 768,
      duration_seconds: 1.5,
    };
    render(<DocViewer roomId="room-a" selectedArtifactPath="artifacts/images/img_a1b2c3.png" />);
    expect(screen.getByTestId('image-artifact-preview')).toBeInTheDocument();
    expect(await screen.findByTestId('image-metadata-card')).toBeInTheDocument();
    expect(screen.getByText('a scenic mountain at sunset')).toBeInTheDocument();
    expect(screen.getByText('Seed: 42')).toBeInTheDocument();
    expect(screen.getByText('diffusers-sdxl')).toBeInTheDocument();
  });
});

