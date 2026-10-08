import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RoomConversation } from './RoomConversation';
import type { RoomState, RoomTranscriptMessage } from '../../types';
import { EMPTY_LIVE } from '../../lib/rooms';
import { LocaleProvider } from '../../i18n';
import { leftoverEnglish } from '../../test/leftoverEnglish';

/**
 * The Core's own notes, and the missing-image directive in a reply, in the reader's language
 * (step 5 of `multilingual-ui.md`, §3.3).
 *
 * A note is stored with a `code` and `params`, and its `content` is the English fallback. So a
 * Korean screen that shows `content` is the defect these cases exist for, and a note with no
 * code, or one this head does not know, must still show `content` rather than nothing.
 */

const ENGLISH_REGISTERED =
  '🔄 **Repeating task started** (every 5 minutes, ID: `job-7`):\n"status check"\n\n' +
  '*To stop it, type `/loop stop` or press the stop button at the top of the conversation.*';

const note = (over: Partial<RoomTranscriptMessage>): RoomTranscriptMessage => ({
  seq: 2,
  sender_id: 'system',
  content: 'English fallback',
  kind: 'note',
  created_at: '2026-09-25T00:00:00Z',
  completed: true,
  ...over,
});

const NOTES: RoomTranscriptMessage[] = [
  note({ seq: 2, code: 'loop.help', content: 'help (fallback)' }),
  note({
    seq: 3,
    code: 'loop.registered',
    params: { job_id: 'job-7', interval_seconds: 300, prompt: 'status check' },
    content: ENGLISH_REGISTERED,
  }),
  note({
    seq: 4,
    code: 'loop.active',
    params: { job_id: 'job-7', interval_seconds: 45, prompt: 'status check' },
  }),
  note({ seq: 5, code: 'loop.none_active' }),
  note({ seq: 6, code: 'loop.stopped' }),
  note({ seq: 7, code: 'loop.nothing_to_stop' }),
  note({ seq: 8, code: 'loop.missing_prompt' }),
  note({ seq: 9, code: 'loop.interval_too_short', params: { interval_seconds: 1 } }),
  note({ seq: 10, code: 'loop.no_interval' }),
  note({
    seq: 13,
    code: 'loop.resumed',
    params: { job_id: 'job-7', interval_seconds: 600, prompt: 'status check' },
  }),
];

// Its `reason` is the Core's own sentence, so it is kept out of NOTES' no-English check.
const RUN_FAILED = note({
  seq: 14,
  code: 'loop.run_failed',
  params: { job_id: 'job-7', run: 3, reason: 'Reason.', interval_seconds: 600 },
});

const REPLY: RoomTranscriptMessage = {
  seq: 11,
  sender_id: 'scout',
  content:
    'Here it is: :missing-image{file="img_%EA%B7%B8%EB%A6%BC.png"}\n\nAlso :missing-image-link{file="img_2.png"}.\n\n' +
    'Written literally: `:missing-image{file="kept.png"}`',
  kind: 'utterance',
  created_at: '2026-09-25T00:00:00Z',
  completed: true,
};

const room = (transcript: RoomTranscriptMessage[]): RoomState => ({
  room_id: 'r1',
  title: 'Loop test',
  participants: [
    { id: 'user', kind: 'human', display_name: 'Kenny' },
    { id: 'scout', kind: 'agent', display_name: 'Scout' },
  ],
  transcript: [
    {
      seq: 1,
      sender_id: 'user',
      content: '/loop 5m status check',
      kind: 'utterance',
      created_at: '2026-09-25T00:00:00Z',
      completed: true,
    },
    ...transcript,
  ],
  turn_state: { agent_turns_since_human: 0 },
  policy: {
    max_agent_turns_per_human_message: 3,
    max_span_tokens: 8000,
    transcript_window: 15,
    hesitation_seconds: 0,
    default_responder_id: '',
  },
});

const renderIn = (hints: readonly string[], transcript: RoomTranscriptMessage[]) =>
  render(
    <LocaleProvider hints={hints}>
      <RoomConversation
        room={room(transcript)}
        availableAgents={[]}
        live={EMPTY_LIVE}
        onSend={() => {}}
        onStop={() => {}}
        onRetry={() => {}}
        onAddAgent={() => {}}
        onTyping={() => {}}
        onOpenTurn={() => {}}
        draft=""
        onDraftChange={() => {}}
      />
    </LocaleProvider>,
  );

const notesText = () =>
  NOTES.map((n) => screen.getByTestId(`note-${n.seq}`).textContent ?? '').join('\n');

const notesRoot = (): HTMLElement => {
  const root = document.createElement('div');
  for (const n of NOTES) root.appendChild(screen.getByTestId(`note-${n.seq}`).cloneNode(true));
  return root;
};

describe('the Core’s notes in the chosen language', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 503, json: async () => ({}) })));
  });
  afterEach(() => vi.unstubAllGlobals());

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: <MessageBody text={noticeText(message, t)} />
  // Becomes: <MessageBody text={message.content} />
  it('words every /loop note in Korean, from its code, with the values in place', () => {
    renderIn(['ko-KR'], [...NOTES, RUN_FAILED]);

    expect(screen.getByTestId('note-3')).toHaveTextContent(
      '🔄 반복 작업을 등록했습니다 (5분마다 실행, ID: job-7): "status check"',
    );
    expect(screen.getByTestId('note-13')).toHaveTextContent(
      '앱이 다시 시작되어 반복 작업을 이어서 실행합니다 (10분마다, ID: job-7)',
    );
    expect(screen.getByTestId('note-14')).toHaveTextContent(
      '반복 작업 job-7의 3번째 실행이 실패했습니다: Reason. 10분 뒤에 다시 시도합니다.',
    );
    expect(screen.getByTestId('note-4')).toHaveTextContent('job-7 (45초마다)');
    expect(screen.getByTestId('note-9')).toHaveTextContent('1초에 한 번까지만');
    // The command syntax stays literal.
    expect(screen.getByTestId('note-2')).toHaveTextContent('/loop list');
    expect(notesText()).not.toContain('fallback');
    expect(notesText()).not.toMatch(/\{\w+\}/);
    expect(leftoverEnglish(notesRoot(), ['notices'])).toEqual([]);
  });

  // Killed by: frontend/src/lib/notices.ts :: seconds < 60 || seconds % 60 !== 0
  // Becomes: seconds < 60
  it('words them in English, with the interval in seconds or minutes as it divides', () => {
    renderIn(['en-US'], [
      ...NOTES,
      note({
        seq: 12,
        code: 'loop.active',
        params: { job_id: 'job-8', interval_seconds: 90, prompt: 'p' },
      }),
    ]);

    expect(screen.getByTestId('note-3')).toHaveTextContent(
      'Repeating task started (every 5 minutes, ID: job-7): "status check"',
    );
    expect(screen.getByTestId('note-4')).toHaveTextContent('job-7 (every 45 seconds)');
    expect(screen.getByTestId('note-12')).toHaveTextContent('job-8 (every 90 seconds)');
    expect(screen.getByTestId('note-9')).toHaveTextContent('at most once every 1 second.');
    expect(screen.getByTestId('note-6')).toHaveTextContent('The repeating task was stopped.');
    expect(leftoverEnglish(notesRoot(), ['notices']).length).toBeGreaterThan(4);
  });

  // Killed by: frontend/src/lib/notices.ts :: if (!code || !isKnownCode(code, t)) return content;
  // Becomes: if (!code) return content;
  it('shows the stored sentence for a note with no code, an unknown code, or missing values', () => {
    renderIn(['ko-KR'], [
      note({ seq: 2, content: '🔄 **반복 작업 등록됨** (old note)' }),
      note({ seq: 3, code: 'loop.from_a_newer_core', content: 'A newer notice.' }),
      note({ seq: 4, code: 'loop.registered', params: { job_id: 'job-7' }, content: 'Incomplete params.' }),
    ]);

    expect(screen.getByTestId('note-2')).toHaveTextContent('반복 작업 등록됨 (old note)');
    expect(screen.getByTestId('note-3')).toHaveTextContent('A newer notice.');
    expect(screen.getByTestId('note-4')).toHaveTextContent('Incomplete params.');
  });
});

describe('a missing image in a reply, in the chosen language', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 503, json: async () => ({}) })));
  });
  afterEach(() => vi.unstubAllGlobals());

  // Killed by: frontend/src/components/RichText.tsx :: const preprocessed = preprocessLaTeX(preprocessImageTags(preprocessMissingImages(rawText)));
  // Becomes: const preprocessed = preprocessLaTeX(preprocessImageTags(rawText));
  it('draws the directive as a notice in Korean, with the file name decoded', () => {
    const { baseElement } = renderIn(['ko-KR'], [REPLY]);

    const notices = screen.getAllByTestId('missing-image-notice');
    expect(notices).toHaveLength(2);
    expect(notices[0]).toHaveTextContent('img_그림.png이(가) 만들어지지 않았으므로');
    expect(notices[1]).toHaveTextContent('만들어지지 않은 이미지로 가는 링크입니다: img_2.png');
    expect(baseElement.textContent).not.toContain(':missing-image{file="img_');
    // Inside code it is text someone wrote, not a directive.
    expect(baseElement.textContent).toContain(':missing-image{file="kept.png"}');
    expect(leftoverEnglish(baseElement, ['notices'])).toEqual([]);
  });

  // Killed by: frontend/src/components/RichText.tsx :: <span>{fmt(link ? t.link : t.image, { file })}</span>
  // Becomes: <span>{fmt(t.image, { file })}</span>
  it('draws it in English on an English screen, the image and the link each as themselves', () => {
    renderIn(['en-US'], [REPLY]);

    const notices = screen.getAllByTestId('missing-image-notice');
    expect(notices[0]).toHaveTextContent(
      'This image is not shown because the image tool did not run, so img_그림.png was never created.',
    );
    expect(notices[1]).toHaveTextContent('Link to an image that was never created: img_2.png');
  });

  it('leaves a directive whose file is not encoded as the text it is', () => {
    renderIn(['en-US'], [{ ...REPLY, content: 'x :missing-image{file="a b"} y' }]);
    expect(screen.queryByTestId('missing-image-notice')).toBeNull();
    expect(screen.getByText(/:missing-image\{file="a b"\}/)).toBeInTheDocument();
  });
});
