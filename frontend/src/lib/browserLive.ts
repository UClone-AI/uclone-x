import { useCallback, useEffect, useRef, useState } from 'react';
import type { Messages } from '../i18n';
import { fmt } from '../i18n';
import type { RoomToolUse } from '../types';

/**
 * The dock's Browser tab, live (`browser-agent.md` §3.4; §6 step 3).
 *
 * One WebSocket per open conversation, `/api/browser/{conversation}` (#2122's route), served by
 * the Core's `browser/live.py`. It carries codes, never copy: which tabs are open, which clone is acting,
 * a problem code, steps as they end, and -- only while the Browser tab is on screen -- the
 * shown tab's frames as binary JPEG and an overlay box before each action. Every sentence the
 * reader sees is worded here, from the catalog.
 *
 * The socket, not the room's event stream, because the stream publishes a turn's tool events
 * only once the turn has landed: a banner or a step line read from it would lag the browser by
 * a whole turn.
 */

export interface BrowserLiveTab {
  clone: string;
  /** From 1, in the clone's own order. */
  index: number;
  url: string;
  title: string;
  /** The clone's current tab. */
  current: boolean;
  /** Who drives the tab: U0 after Take over, else the clone. */
  controller: 'user' | 'clone';
}

/** A problem the head words itself; any other code is not shown. */
export type BrowserProblem = 'chrome_missing' | 'browser_closed';

export interface BrowserLiveState {
  tabs: BrowserLiveTab[];
  acting: { clone: string; action: string } | null;
  problem: BrowserProblem | null;
  shown: { clone: string; index: number } | null;
  /** Whether U0's Chrome is paired and connected (R1). */
  extension: boolean;
  asked_user?: {
    clone: string;
    kind: string;
    site: string;
    message: string;
  } | null;
}

export interface BrowserStep {
  clone: string;
  action: string;
  /** The accessible name of the element the step touched, or `''`. Never typed text. */
  element: string;
  ok: boolean;
}

export interface BrowserOverlay {
  clone: string;
  /** In the page's CSS pixels. */
  box: { x: number; y: number; width: number; height: number };
  label: string;
  action: string;
}

export interface BrowserFrameSize {
  width: number;
  height: number;
}

/** What the Browser tab drives without re-rendering the application on every frame. */
export interface BrowserChannel {
  /** Ask for frames (while the tab is on screen) at about `width` CSS pixels. */
  setView: (on: boolean, width: number) => void;
  /** Watch one tab. */
  pickTab: (clone: string, index: number) => void;
  /** #2122's controls on the shown tab. */
  control: (kind: 'take_over' | 'give_back' | 'stop') => void;
  takeOver: () => void;
  giveBack: () => void;
  stop: () => void;
  newTab: (url?: string) => void;
  navigate: (url: string) => void;
  handTo: (target: string) => void;
  /** U0's mouse, key or text, for the shown tab while it is theirs. */
  input: (message: Record<string, unknown>) => void;
  onFrame: (listener: (jpeg: Blob, size: BrowserFrameSize | null) => void) => () => void;
  onOverlay: (listener: (overlay: BrowserOverlay) => void) => () => void;
}

export type LiveConnection = 'connecting' | 'open' | 'lost';

export interface BrowserLive {
  connection: LiveConnection;
  state: BrowserLiveState | null;
  /** This turn's steps, oldest first, until `clearSteps`. */
  steps: BrowserStep[];
  /** The last step in this conversation, kept across turns for the Browser tab. */
  lastStep: BrowserStep | null;
  clearSteps: () => void;
  channel: BrowserChannel;
}

/** The minimum of `WebSocket` the hook uses, so tests can hand it a fake. */
export interface LiveSocketLike {
  binaryType: string;
  onopen: (() => void) | null;
  onclose: (() => void) | null;
  onerror: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  send: (data: string) => void;
  close: () => void;
  readonly readyState: number;
}

export type LiveSocketFactory = (url: string) => LiveSocketLike;

const OPEN = 1;
export const RECONNECT_MS = 2000;

export const liveUrl = (conversation: string, location: Location = window.location): string =>
  `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/browser/${encodeURIComponent(conversation)}`;

const defaultFactory: LiveSocketFactory = (url) => new WebSocket(url) as unknown as LiveSocketLike;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const str = (value: unknown): string => (typeof value === 'string' ? value : '');

/** A `state` message as data, or `null` when it is not one. Unknown problem codes read as none. */
export const parseState = (message: Record<string, unknown>): BrowserLiveState => {
  const tabs = Array.isArray(message.tabs)
    ? message.tabs.filter(isRecord).map((t) => ({
        clone: str(t.clone),
        index: typeof t.index === 'number' ? t.index : 0,
        url: str(t.url),
        title: str(t.title),
        current: t.current === true,
        controller: t.controller === 'user' ? ('user' as const) : ('clone' as const),
      }))
    : [];
  const acting = isRecord(message.acting)
    ? { clone: str(message.acting.clone), action: str(message.acting.action) }
    : null;
  const problem =
    message.problem === 'chrome_missing' || message.problem === 'browser_closed'
      ? message.problem
      : null;
  const shown =
    isRecord(message.shown) && typeof message.shown.index === 'number'
      ? { clone: str(message.shown.clone), index: message.shown.index }
      : null;
  const asked_user = isRecord(message.asked_user)
    ? {
        clone: str(message.asked_user.clone),
        kind: str(message.asked_user.kind),
        site: str(message.asked_user.site),
        message: str(message.asked_user.message),
      }
    : null;
  return { tabs, acting, problem, shown, extension: message.extension === true, asked_user };
};

/**
 * The live view of one conversation's browser. `conversation` null closes the socket.
 *
 * The socket stays open while the conversation is on screen, frames or not: the banner's
 * acting state, the step lines and the dock's first-use opening all come from it.
 */
export const useBrowserLive = (
  conversation: string | null,
  factory: LiveSocketFactory = defaultFactory,
): BrowserLive => {
  const [connection, setConnection] = useState<LiveConnection>('connecting');
  const [state, setState] = useState<BrowserLiveState | null>(null);
  const [steps, setSteps] = useState<BrowserStep[]>([]);
  const [lastStep, setLastStep] = useState<BrowserStep | null>(null);
  const socketRef = useRef<LiveSocketLike | null>(null);
  const viewRef = useRef<{ on: boolean; width: number }>({ on: false, width: 0 });
  const frameListeners = useRef(new Set<(jpeg: Blob, size: BrowserFrameSize | null) => void>());
  const overlayListeners = useRef(new Set<(overlay: BrowserOverlay) => void>());
  const sizeRef = useRef<BrowserFrameSize | null>(null);
  const factoryRef = useRef(factory);
  factoryRef.current = factory;

  const sendNow = useCallback((message: Record<string, unknown>) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === OPEN) socket.send(JSON.stringify(message));
  }, []);

  useEffect(() => {
    setState(null);
    setSteps([]);
    setLastStep(null);
    sizeRef.current = null;
    if (conversation === null) {
      setConnection('connecting');
      return undefined;
    }
    let disposed = false;
    let retry: ReturnType<typeof setTimeout> | null = null;

    const connect = (): void => {
      setConnection((now) => (now === 'lost' ? 'lost' : 'connecting'));
      let socket: LiveSocketLike;
      try {
        socket = factoryRef.current(liveUrl(conversation));
      } catch {
        setConnection('lost');
        retry = setTimeout(connect, RECONNECT_MS);
        return;
      }
      socket.binaryType = 'blob';
      socketRef.current = socket;
      socket.onopen = () => {
        if (disposed) return;
        setConnection('open');
        const view = viewRef.current;
        if (view.on) socket.send(JSON.stringify({ type: 'view', on: true, width: view.width }));
      };
      socket.onmessage = (event) => {
        if (disposed) return;
        const { data } = event;
        if (typeof data !== 'string') {
          if (data instanceof Blob) {
            frameListeners.current.forEach((listener) => listener(data, sizeRef.current));
          }
          return;
        }
        let message: unknown;
        try {
          message = JSON.parse(data);
        } catch {
          return;
        }
        if (!isRecord(message)) return;
        switch (message.type) {
          case 'state':
            setState(parseState(message));
            break;
          case 'step': {
            const step: BrowserStep = {
              clone: str(message.clone),
              action: str(message.action),
              element: str(message.element),
              ok: message.ok === true,
            };
            setSteps((now) => [...now, step]);
            setLastStep(step);
            break;
          }
          case 'overlay': {
            const box = message.box;
            const sides = isRecord(box) ? [box.x, box.y, box.width, box.height] : [];
            if (
              sides.length === 4 &&
              sides.every((n) => typeof n === 'number' && Number.isFinite(n))
            ) {
              const [x, y, width, height] = sides as number[];
              const overlay: BrowserOverlay = {
                clone: str(message.clone),
                box: { x, y, width, height },
                label: str(message.label),
                action: str(message.action),
              };
              overlayListeners.current.forEach((listener) => listener(overlay));
            }
            break;
          }
          case 'frame':
            if (typeof message.width === 'number' && typeof message.height === 'number') {
              sizeRef.current = { width: message.width, height: message.height };
            }
            break;
          default:
            break;
        }
      };
      socket.onerror = () => undefined;
      socket.onclose = () => {
        if (disposed) return;
        socketRef.current = null;
        sizeRef.current = null;
        setConnection('lost');
        retry = setTimeout(connect, RECONNECT_MS);
      };
    };

    connect();
    return () => {
      disposed = true;
      if (retry !== null) clearTimeout(retry);
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, [conversation]);

  const channelRef = useRef<BrowserChannel | null>(null);
  if (channelRef.current === null) {
    channelRef.current = {
      setView: (on, width) => {
        const rounded = Math.round(width);
        const before = viewRef.current;
        if (before.on === on && (!on || before.width === rounded)) return;
        viewRef.current = { on, width: rounded };
        sendNow({ type: 'view', on, width: rounded });
      },
      pickTab: (clone, index) => sendNow({ type: 'tab', clone, index }),
      control: (kind) => sendNow({ type: kind }),
      takeOver: () => sendNow({ type: 'take_over' }),
      giveBack: () => sendNow({ type: 'give_back' }),
      stop: () => sendNow({ type: 'stop' }),
      newTab: (url) => sendNow({ type: 'new_tab', url: url || 'about:blank' }),
      navigate: (url) => sendNow({ type: 'navigate', url }),
      handTo: (target) => sendNow({ type: 'hand_to', target }),
      input: (message) => sendNow({ ...message, type: 'input' }),
      onFrame: (listener) => {
        frameListeners.current.add(listener);
        return () => frameListeners.current.delete(listener);
      },
      onOverlay: (listener) => {
        overlayListeners.current.add(listener);
        return () => overlayListeners.current.delete(listener);
      },
    };
  }
  const clearSteps = useCallback(() => setSteps([]), []);
  return { connection, state, steps, lastStep, clearSteps, channel: channelRef.current };
};

// -- wording ------------------------------------------------------------------------------

type StepCopy = Messages['dock']['browser']['steps'];

const PLAIN: ReadonlySet<string> = new Set([
  'open', 'snapshot', 'read', 'find', 'click', 'type', 'select', 'check', 'press', 'scroll',
  'upload', 'wait', 'back', 'forward', 'reload', 'tab',
]);

const WITH_ELEMENT: Readonly<Record<string, keyof StepCopy>> = {
  click: 'clickOn',
  type: 'typeOn',
  select: 'selectOn',
  check: 'checkOn',
  upload: 'uploadOn',
};

/**
 * One step in words: the action and the element's name, never what was typed, never the
 * tool's own error. A step that failed says only that it did not work.
 */
export const stepSentence = (copy: StepCopy, step: Pick<BrowserStep, 'action' | 'element' | 'ok'>): string => {
  const named = WITH_ELEMENT[step.action];
  let sentence: string;
  if (named && step.element) {
    sentence = fmt(copy[named], { element: step.element });
  } else if (PLAIN.has(step.action)) {
    sentence = copy[step.action as keyof StepCopy];
  } else {
    sentence = copy.other;
  }
  return step.ok ? sentence : fmt(copy.failed, { step: sentence });
};

const ACTION_KEY = /"action"\s*:\s*"([a-z_]+)"/;
const ELEMENT_KEY = /^\{"element":\s*("(?:[^"\\]|\\.)*")/;

/**
 * A recorded `browser` call as a step: the action from its arguments, the element from the
 * result, which leads with it (`browser/tool.py`). Both previews are cut to a bound, so each is
 * read by pattern rather than parsed whole; a preview that names neither still gives a step.
 */
export const recordedStep = (use: RoomToolUse): BrowserStep => {
  const action = ACTION_KEY.exec(use.arguments_preview)?.[1] ?? '';
  let element = '';
  const quoted = ELEMENT_KEY.exec(use.output_preview)?.[1];
  if (quoted) {
    try {
      const parsed: unknown = JSON.parse(quoted);
      if (typeof parsed === 'string') element = parsed;
    } catch {
      element = '';
    }
  }
  return { clone: use.participant_id, action, element, ok: use.status === 'success' };
};

/** The tool's registered name. */
export const BROWSER_TOOL = 'browser';
