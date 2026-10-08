import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { RoomConversation } from './RoomConversation';
import type { RoomState, RoomTranscriptMessage } from '../../types';
import { EMPTY_LIVE } from '../../lib/rooms';
import { LocaleProvider, fmt } from '../../i18n';
import { en } from '../../i18n/en';
import { ko } from '../../i18n/ko';

/**
 * The conversation and its composer follow the language control (step 4 of
 * `multilingual-ui.md`).
 *
 * Every case in `RoomConversation.test.tsx` renders without a provider and so reads English;
 * these render inside one, the way `main.tsx` does, and read Korean.
 */

const message = (over: Partial<RoomTranscriptMessage>): RoomTranscriptMessage => ({
  seq: 1,
  sender_id: 'user',
  content: 'hello',
  kind: 'utterance',
  created_at: '2026-09-14T00:00:00Z',
  completed: true,
  ...over,
});

const room = (over: Partial<RoomState> = {}): RoomState => ({
  room_id: 'r1',
  title: 'Architecture triage',
  participants: [
    { id: 'user', kind: 'human', display_name: 'Kenny' },
    { id: 'scout', kind: 'agent', display_name: 'Scout' },
  ],
  transcript: [message({ seq: 1 })],
  turn_state: { agent_turns_since_human: 0 },
  policy: {
    max_agent_turns_per_human_message: 3,
    max_span_tokens: 8000,
    transcript_window: 15,
    hesitation_seconds: 0,
    default_responder_id: '',
  },
  ...over,
});

type Props = React.ComponentProps<typeof RoomConversation>;

const Host: React.FC<{ hints: readonly string[]; over?: Partial<Props> }> = ({ hints, over = {} }) => {
  const [draft, setDraft] = React.useState(over.draft ?? '');
  return (
    <LocaleProvider hints={hints}>
      <RoomConversation
        room={room()}
        availableAgents={[]}
        live={EMPTY_LIVE}
        onSend={() => {}}
        onStop={() => {}}
        onRetry={() => {}}
        onAddAgent={() => {}}
        onTyping={() => {}}
        onOpenTurn={() => {}}
        {...over}
        draft={draft}
        onDraftChange={setDraft}
      />
    </LocaleProvider>
  );
};

class FakeRecognition {
  static last: FakeRecognition | null = null;
  lang = '';
  continuous = false;
  interimResults = true;
  onresult: unknown = null;
  onerror: ((event: { error: string }) => void) | null = null;
  onend: (() => void) | null = null;
  constructor() {
    FakeRecognition.last = this;
  }
  start() {}
  stop() {
    this.onend?.();
  }
  abort() {}
}

describe('the conversation in the chosen language', () => {
  beforeEach(() => {
    window.localStorage.clear();
    // The provider reads the saved choice; no answer keeps the hint it was given.
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 503, json: async () => ({}) })));
  });
  afterEach(() => vi.unstubAllGlobals());

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: placeholder={c.placeholder}
  // Becomes: placeholder="Send a message"
  it('writes the composer and the header in Korean, and in English again when switched', () => {
    const { rerender } = render(<Host hints={['ko-KR']} />);

    expect(screen.getByTestId('room-composer')).toHaveAttribute('placeholder', ko.composer.placeholder);
    expect(screen.getByLabelText(ko.composer.sendLabel)).toBeInTheDocument();
    expect(screen.getByLabelText(ko.conversation.header.addSomeone)).toBeInTheDocument();
    expect(screen.queryByLabelText(en.composer.sendLabel)).toBeNull();

    rerender(<Host hints={['en-US']} />);
    expect(screen.getByTestId('room-composer')).toHaveAttribute('placeholder', 'Send a message');
    expect(screen.getByLabelText('Send (Enter)')).toBeInTheDocument();
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {fmt(t.row.imageNegativeAdded, { label, tags: message.image_negative_added.join(', ') })}
  // Becomes: {fmt(en.conversation.row.imageNegativeAdded, { label, tags: message.image_negative_added.join(', ') })}
  it('says in Korean what was added to a picture request (#1865)', () => {
    render(
      <Host
        hints={['ko-KR']}
        over={{
          room: room({
            transcript: [
              message({
                seq: 1,
                sender_id: 'scout',
                content: '그렸습니다.',
                image_prompt_added: ['masterpiece'],
                image_negative_added: ['blurry', 'text'],
              }),
            ],
          }),
        }}
      />,
    );

    expect(screen.getByTestId('row-image-prompt-added-1')).toHaveTextContent(
      fmt(ko.conversation.row.imagePromptAdded, { label: 'Scout', tags: 'masterpiece' }),
    );
    const left = screen.getByTestId('row-image-negative-added-1');
    expect(left).toHaveTextContent('Scout이(가) 그림에서 빼도록 요청했습니다: blurry, text.');
    expect(left).not.toHaveTextContent(/asked|leave out/);
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {refusalRemedy(message.refusal, t.outcome)}
  // Becomes: {refusalRemedy(message.refusal)}
  it('says a refused turn and its remedy in Korean, with the clone named', () => {
    render(
      <Host
        hints={['ko-KR']}
        over={{
          room: room({
            transcript: [
              message({
                seq: 1,
                sender_id: 'scout',
                content: '',
                error: 'BudgetExceeded: Session token limit of 5 reached',
                refusal: 'budget_exceeded',
              }),
            ],
          }),
        }}
      />,
    );

    expect(screen.getByTestId('row-error-1')).toHaveTextContent(
      fmt(ko.conversation.outcome.refused, {
        label: 'Scout',
        reason: ko.conversation.outcome.refusalReason.budget_exceeded,
      }),
    );
    expect(screen.getByTestId('row-remedy-1')).toHaveTextContent(
      ko.conversation.outcome.refusalRemedy.budget_exceeded,
    );
    expect(screen.getByTestId('row-error-1')).not.toHaveTextContent('Session token limit');
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: {sendFailureNotice(sendError.cause, c.sendFailure)}
  // Becomes: {sendFailureNotice(sendError.cause)}
  it('says a failed send in the language on screen, and again after a switch', async () => {
    const onSend = vi.fn(async () => {
      throw new Error('fetch failed');
    });
    const { rerender } = render(<Host hints={['ko-KR']} over={{ onSend, draft: 'hello' }} />);

    await act(async () => {
      fireEvent.click(screen.getByLabelText(ko.composer.sendLabel));
    });
    const notice = screen.getByTestId('send-error');
    expect(notice).toHaveTextContent(
      fmt(ko.composer.sendFailure.unconfirmed, { reason: ko.composer.sendFailure.appFault }),
    );
    expect(notice).not.toHaveTextContent('fetch failed');

    // The failure is kept, not its sentence, so the line follows the control.
    rerender(<Host hints={['en-US']} over={{ onSend, draft: 'hello' }} />);
    expect(screen.getByTestId('send-error')).toHaveTextContent(
      'Could not confirm whether this conversation has your message: Something went wrong in the app.',
    );
  });

  // Killed by: frontend/src/components/rooms/RoomConversation.tsx :: dictationLang(language, typeof navigator === 'undefined' ? undefined : navigator.language),
  // Becomes: navigator.language,
  it('listens for Korean when the screen is in Korean, whatever the browser says', () => {
    vi.stubGlobal('SpeechRecognition', FakeRecognition);
    render(<Host hints={['ko-KR']} />);

    fireEvent.click(screen.getByTestId('dictate'));
    // jsdom's `navigator.language` is `en-US`; the reader chose Korean.
    expect(FakeRecognition.last?.lang).toBe('ko-KR');

    act(() => FakeRecognition.last?.onerror?.({ error: 'not-allowed' }));
    act(() => FakeRecognition.last?.onend?.());
    expect(screen.getByTestId('dictation-message')).toHaveTextContent(ko.composer.dictation.notAllowed);
  });
});
