import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, Globe, StopCircle, UserCheck } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import type { Messages } from '../../i18n';
import {
  stepSentence,
  useBrowserLive,
  type BrowserFrameSize,
  type BrowserLive,
  type BrowserLiveState,
  type BrowserOverlay,
  type LiveConnection,
} from '../../lib/browserLive';
import type { RoomState } from '../../types';

type Copy = Messages['dock']['browser'];

/** How long an action's outline stays before it has faded (§3.4: "about 600 ms"). */
export const OVERLAY_MS = 600;

const GET_CHROME_URL = 'https://www.google.com/chrome/';

/** A clone's name as the conversation shows it; its id when it has none. */
export const cloneName = (room: RoomState | null | undefined, clone: string): string =>
  room?.participants.find((p) => p.id === clone)?.display_name || clone;

export type BannerKind =
  | 'noRoom'
  | 'connecting'
  | 'lost'
  | 'chromeMissing'
  | 'closed'
  | 'askedUser'
  | 'userControl'
  | 'acting'
  | 'idle'
  | 'none';

/** Which banner the tab shows: one fact, always in words, never an empty panel. */
export const bannerKind = (
  roomId: string | null,
  connection: LiveConnection,
  state: BrowserLiveState | null,
): BannerKind => {
  if (roomId === null) return 'noRoom';
  if (state === null) return connection === 'lost' ? 'lost' : 'connecting';
  if (state.asked_user) return 'askedUser';
  const shown = state.shown;
  const shownTab = shown
    ? state.tabs.find((tab) => tab.clone === shown.clone && tab.index === shown.index)
    : undefined;
  if (shownTab?.controller === 'user') return 'userControl';
  if (state.acting) return 'acting';
  if (state.problem === 'chrome_missing') return 'chromeMissing';
  if (state.problem === 'browser_closed') return 'closed';
  return state.tabs.length > 0 ? 'idle' : 'none';
};

/** Whether the page is on screen (not a background browser tab or a minimised window). */
const useDocumentVisible = (): boolean => {
  const read = (): boolean =>
    typeof document === 'undefined' || document.visibilityState !== 'hidden';
  const [visible, setVisible] = useState(read);
  useEffect(() => {
    const update = (): void => setVisible(read());
    document.addEventListener('visibilitychange', update);
    return () => document.removeEventListener('visibilitychange', update);
  }, []);
  return visible;
};

export interface BrowserPanelProps {
  roomId: string | null;
  /** The open conversation, for its clones' names. */
  room?: RoomState | null;
  /**
   * The conversation's browser socket (`lib/browserLive.ts`). `App` holds it, so the step lines
   * under a turn read the same socket; without it the panel opens its own.
   */
  live?: BrowserLive;
  /** Open Settings ▸ Browser, to connect U0's Chrome. */
  onConnect?: () => void;
}

/**
 * The dock's Browser tab (§3.4): the clone's tab, live, with an outline on the element it is
 * about to touch, a banner that names who is driving, and #2122's Take over / Give back /
 * Stop. It is mounted only while it is the dock's surface, so frames are asked for only while
 * it is on screen, at the width the dock gives it.
 */
export const BrowserPanel: React.FC<BrowserPanelProps> = ({ roomId, room, live, onConnect }) => {
  const t = useCopy().dock.browser;
  const own = useBrowserLive(live ? null : roomId);
  const { state, connection, lastStep, channel } = live ?? own;
  const kind = bannerKind(roomId, connection, state);
  const pageVisible = useDocumentVisible();

  const frameBoxRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [frameSize, setFrameSize] = useState<BrowserFrameSize | null>(null);
  const [overlay, setOverlay] = useState<(BrowserOverlay & { at: number }) | null>(null);
  const [fading, setFading] = useState(false);
  const [shownWidth, setShownWidth] = useState(0);
  const [addressInput, setAddressInput] = useState('');

  const viewing = pageVisible && roomId !== null;

  // Frames only while the tab is on screen, sized to the dock.
  useEffect(() => {
    const box = frameBoxRef.current;
    const measure = (): number => Math.max(0, box?.clientWidth ?? 0);
    channel.setView(viewing, measure() || 640);
    setShownWidth(measure());
    if (!box || typeof ResizeObserver === 'undefined') return () => channel.setView(false, 0);
    const observer = new ResizeObserver(() => {
      const width = measure();
      setShownWidth(width);
      if (viewing && width > 0) channel.setView(true, width);
    });
    observer.observe(box);
    return () => {
      observer.disconnect();
      channel.setView(false, 0);
    };
  }, [channel, viewing]);

  // Draw each frame without re-rendering for it.
  useEffect(() => {
    let drawing = false;
    return channel.onFrame((jpeg, size) => {
      if (size) {
        setFrameSize((now) =>
          now?.width === size.width && now?.height === size.height ? now : size,
        );
      }
      const canvas = canvasRef.current;
      const context = canvas?.getContext?.('2d');
      if (!canvas || !context || drawing || typeof createImageBitmap !== 'function') return;
      drawing = true;
      void createImageBitmap(jpeg)
        .then((bitmap) => {
          if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
          if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
          context.drawImage(bitmap, 0, 0);
          bitmap.close();
          setFrameSize((now) => now ?? { width: bitmap.width, height: bitmap.height });
        })
        .catch(() => undefined)
        .finally(() => {
          drawing = false;
        });
    });
  }, [channel]);

  useEffect(
    () =>
      channel.onOverlay((next) => {
        setFading(false);
        setOverlay({ ...next, at: Date.now() });
      }),
    [channel],
  );
  useEffect(() => {
    if (!overlay) return undefined;
    const fade = setTimeout(() => setFading(true), 50);
    const clear = setTimeout(() => setOverlay(null), OVERLAY_MS);
    return () => {
      clearTimeout(fade);
      clearTimeout(clear);
    };
  }, [overlay]);

  const shown = state?.shown ?? null;
  const shownTab = shown
    ? (state?.tabs.find((tab) => tab.clone === shown.clone && tab.index === shown.index) ?? null)
    : null;
  const shownUrl = shownTab?.url ?? '';
  useEffect(() => setAddressInput(shownUrl), [shownUrl]);

  if (roomId === null) {
    return (
      <div
        data-testid="browser-no-room"
        className="h-full flex flex-col items-center justify-center p-6 text-center text-slate-400 text-xs"
      >
        <Globe className="w-8 h-8 text-slate-600 mb-2" />
        <p>{t.noRoom}</p>
      </div>
    );
  }

  const isUserControl = kind === 'userControl';
  const driverName = cloneName(room, shownTab?.clone ?? state?.acting?.clone ?? '');
  const idleName = cloneName(
    room,
    lastStep?.clone ?? shown?.clone ?? state?.tabs[0]?.clone ?? '',
  );
  const scale =
    frameSize && frameSize.width > 0 && shownWidth > 0 ? shownWidth / frameSize.width : 0;

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>): void => {
    if (!isUserControl || !frameSize) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = Math.round(((e.clientX - rect.left) * frameSize.width) / rect.width);
    const y = Math.round(((e.clientY - rect.top) * frameSize.height) / rect.height);
    for (const mouseType of ['mousePressed', 'mouseReleased']) {
      channel.input({
        event: 'mouse',
        mouse_type: mouseType,
        x,
        y,
        button: 'left',
        click_count: 1,
      });
    }
  };

  return (
    <div
      data-testid="browser-panel"
      className="h-full flex flex-col bg-slate-950 text-slate-200 select-none min-h-0"
    >
      <Banner
        kind={kind}
        copy={t}
        actingName={state?.acting ? cloneName(room, state.acting.clone) : ''}
        idleName={idleName}
        driverName={driverName}
        askedName={state?.asked_user ? cloneName(room, state.asked_user.clone) : ''}
        askedSite={state?.asked_user?.site || ''}
        hasTab={shownTab !== null}
        live={connection === 'open'}
        showConnect={
          Boolean(state && !state.extension && onConnect) && (kind === 'none' || kind === 'idle')
        }
        clones={
          room
            ? room.participants.filter((p) => p.kind === 'agent')
            : state?.tabs
                .map((t) => ({ id: t.clone, display_name: cloneName(room, t.clone) }))
                .filter((c, i, arr) => arr.findIndex((x) => x.id === c.id) === i) ?? []
        }
        onTakeOver={() => channel.takeOver()}
        onGiveBack={() => channel.giveBack()}
        onHandTo={(target) => channel.handTo(target)}
        onStop={() => channel.stop()}
        onConnect={onConnect}
      />
      {state !== null && connection === 'lost' ? (
        <p data-testid="browser-lost" className="px-3 py-1 text-[11px] text-slate-500">
          {t.lost}
        </p>
      ) : null}

      {/* Tabs Strip */}
      <div
        role="group"
        aria-label={t.tabsLabel}
        data-testid="browser-tabs"
        className="px-2 pt-1.5 flex items-center gap-1 bg-slate-900/70 border-b border-slate-800 overflow-x-auto shrink-0 scrollbar-none"
      >
        {!state || state.tabs.length === 0 ? (
          <span className="text-[11px] text-slate-500 px-2 py-1 italic">{t.noTabs}</span>
        ) : (
          state.tabs.map((tab) => {
            const on = shown?.clone === tab.clone && shown.index === tab.index;
            const name = fmt(t.tabName, {
              name: cloneName(room, tab.clone),
              title: tab.title || t.untitled,
            });
            return (
              <button
                key={`${tab.clone}:${tab.index}`}
                type="button"
                data-testid={`browser-tab-${tab.clone}-${tab.index}`}
                aria-pressed={on}
                onClick={() => channel.pickTab(tab.clone, tab.index)}
                className={`max-w-[160px] px-2.5 py-1 text-[11px] rounded-t border-t border-l border-r truncate flex items-center gap-1.5 transition-colors ${
                  on
                    ? 'bg-slate-950 border-slate-700 text-slate-100 font-semibold'
                    : 'bg-slate-900/40 border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
                title={name}
              >
                <Globe className="w-3 h-3 shrink-0 text-slate-400" />
                <span className="truncate">{name}</span>
              </button>
            );
          })
        )}
        <button
          type="button"
          data-testid="browser-btn-new-tab"
          aria-label={t.newTab || 'New tab'}
          title={t.newTab || 'New tab'}
          onClick={() => channel.newTab()}
          className="px-2 py-1 text-xs font-bold rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors shrink-0"
        >
          +
        </button>
      </div>

      {/* Address Bar */}
      <div
        data-testid="browser-address-bar"
        className="px-3 py-1.5 bg-slate-950 border-b border-slate-800 flex items-center gap-2 shrink-0"
      >
        <Globe className="w-3.5 h-3.5 text-slate-500 shrink-0" />
        <input
          type="text"
          aria-label={t.address}
          readOnly={!isUserControl}
          value={addressInput}
          onChange={(e) => setAddressInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && isUserControl && addressInput.trim()) {
              channel.navigate(addressInput.trim());
            }
          }}
          placeholder={t.addressPlaceholder}
          className={`w-full bg-slate-900 border rounded px-2.5 py-0.5 text-xs font-mono transition-colors ${
            isUserControl
              ? 'border-blue-500/60 text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-400'
              : 'border-slate-800 text-slate-400 cursor-default select-all'
          }`}
        />
      </div>

      {/* Screencast Viewport: as wide as the dock, as tall as the page's aspect needs. */}
      <div data-testid="browser-viewport" className="flex-1 overflow-auto bg-slate-950 p-2 min-h-0">
        <div ref={frameBoxRef} className="relative w-full">
          {shownTab ? (
            <>
              <canvas
                ref={canvasRef}
                data-testid="browser-canvas"
                role="img"
                aria-label={fmt(t.view.label, { name: cloneName(room, shownTab.clone) })}
                onClick={handleCanvasClick}
                className={`block w-full h-auto rounded border border-slate-800/80 bg-white ${
                  isUserControl ? 'cursor-pointer' : 'cursor-default'
                }`}
              />
              {frameSize === null ? (
                <p
                  data-testid="browser-view-waiting"
                  className="absolute inset-0 flex items-center justify-center text-xs text-slate-500"
                >
                  {viewing ? t.view.waiting : t.view.paused}
                </p>
              ) : null}
              {overlay && scale > 0 ? (
                <div
                  data-testid="browser-overlay"
                  aria-hidden="true"
                  style={{
                    left: overlay.box.x * scale,
                    top: overlay.box.y * scale,
                    width: overlay.box.width * scale,
                    height: overlay.box.height * scale,
                    opacity: fading ? 0 : 1,
                    transition: `opacity ${OVERLAY_MS - 50}ms ease-out`,
                  }}
                  className="absolute pointer-events-none rounded border-2 border-cyan-400 bg-cyan-400/10"
                >
                  {overlay.label ? (
                    <span className="absolute -top-5 left-0 max-w-[200px] truncate rounded bg-cyan-500 px-1.5 py-0.5 text-[10px] font-semibold text-slate-950">
                      {overlay.label}
                    </span>
                  ) : null}
                </div>
              ) : null}
            </>
          ) : null}
        </div>
      </div>

      {/* Last Step Footer */}
      <div
        data-testid="browser-last-step"
        className="px-3 py-1.5 text-[11px] text-slate-400 border-t border-slate-800 bg-slate-900/60 flex items-center justify-between shrink-0"
      >
        <span className="truncate">
          {lastStep
            ? fmt(t.lastStep, { step: stepSentence(t.steps, lastStep) })
            : t.lastStepNone}
        </span>
      </div>
    </div>
  );
};

const Banner: React.FC<{
  kind: BannerKind;
  copy: Copy;
  actingName: string;
  idleName: string;
  driverName: string;
  askedName?: string;
  askedSite?: string;
  hasTab: boolean;
  live: boolean;
  showConnect: boolean;
  clones: Array<{ id: string; display_name?: string }>;
  onTakeOver: () => void;
  onGiveBack: () => void;
  onHandTo: (target: string) => void;
  onStop: () => void;
  onConnect?: () => void;
}> = ({
  kind,
  copy,
  actingName,
  idleName,
  driverName,
  askedName,
  askedSite,
  hasTab,
  live,
  showConnect,
  clones,
  onTakeOver,
  onGiveBack,
  onHandTo,
  onStop,
  onConnect,
}) => {
  const [handToOpen, setHandToOpen] = useState(false);
  const text: Record<BannerKind, string> = {
    noRoom: copy.noRoom,
    connecting: copy.connecting,
    lost: copy.lost,
    chromeMissing: copy.banner.chromeMissing,
    closed: copy.banner.closed,
    askedUser:
      kind === 'askedUser'
        ? fmt(copy.banner.askedUser, { name: askedName || '', site: askedSite || '' })
        : '',
    userControl: copy.banner.userControl,
    acting: kind === 'acting' ? fmt(copy.banner.acting, { name: actingName }) : '',
    idle: kind === 'idle' ? fmt(copy.banner.idle, { name: idleName }) : '',
    none: copy.banner.none,
  };
  const isUserControl = kind === 'userControl';
  return (
    <div
      data-testid="browser-banner"
      data-state={kind}
      role="status"
      className={`px-3 py-2 flex flex-wrap items-center justify-between gap-2 border-b text-xs transition-colors shrink-0 ${
        isUserControl
          ? 'bg-blue-950/60 border-blue-800/80 text-blue-200'
          : 'bg-slate-900/90 border-slate-800 text-slate-200'
      }`}
    >
      <div className="flex items-center gap-2 min-w-0">
        <span
          aria-hidden="true"
          className={`w-2 h-2 rounded-full shrink-0 ${
            live
              ? isUserControl
                ? 'bg-blue-400'
                : kind === 'acting'
                  ? 'bg-emerald-400 animate-pulse'
                  : 'bg-emerald-400'
              : 'bg-slate-600'
          }`}
        />
        <span className="min-w-0 font-medium">{text[kind]}</span>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        {kind === 'chromeMissing' ? (
          <a
            data-testid="browser-get-chrome"
            href={GET_CHROME_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="underline text-cyan-300"
          >
            {copy.getChrome}
          </a>
        ) : null}
        {kind === 'askedUser' ? (
          <button
            type="button"
            data-testid="browser-btn-take-over"
            onClick={onTakeOver}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-amber-600 hover:bg-amber-500 text-white text-[11px] font-medium transition-colors"
          >
            <span>{copy.takeOver}</span>
          </button>
        ) : null}
        {hasTab && isUserControl ? (
          <div className="relative inline-flex items-center gap-1.5">
            <button
              type="button"
              data-testid="browser-btn-give-back"
              onClick={onGiveBack}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white text-[11px] font-medium transition-colors"
            >
              <UserCheck className="w-3.5 h-3.5" />
              <span>{fmt(copy.giveBack, { name: driverName })}</span>
            </button>
            {clones.length > 0 ? (
              <div className="relative">
                <button
                  type="button"
                  data-testid="browser-btn-hand-to"
                  onClick={() => setHandToOpen((prev) => !prev)}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-[11px] font-medium transition-colors"
                >
                  <span>{copy.handTo || 'Hand to…'}</span>
                  <ChevronDown className="w-3 h-3 text-slate-400" />
                </button>
                {handToOpen ? (
                  <div
                    data-testid="browser-hand-to-menu"
                    className="absolute right-0 top-full mt-1 z-20 min-w-[140px] rounded-md bg-slate-900 border border-slate-700 shadow-lg py-1 text-xs"
                  >
                    {clones.map((c) => (
                      <button
                        key={c.id}
                        type="button"
                        data-testid={`browser-hand-to-${c.id}`}
                        onClick={() => {
                          setHandToOpen(false);
                          onHandTo(c.id);
                        }}
                        className="w-full text-left px-3 py-1.5 hover:bg-slate-800 text-slate-200 hover:text-white transition-colors"
                      >
                        {c.display_name || c.id}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}
        {hasTab && !isUserControl && kind !== 'askedUser' ? (
          <>
            <button
              type="button"
              data-testid="browser-btn-take-over"
              onClick={onTakeOver}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-[11px] font-medium transition-colors"
            >
              <span>{copy.takeOver}</span>
            </button>
            {kind === 'acting' ? (
              <button
                type="button"
                data-testid="browser-btn-stop"
                onClick={onStop}
                className="inline-flex items-center gap-1 px-2 py-1 rounded bg-rose-950/60 hover:bg-rose-900/80 text-rose-300 border border-rose-800/60 text-[11px] font-medium transition-colors"
              >
                <StopCircle className="w-3.5 h-3.5" />
                <span>{copy.stop}</span>
              </button>
            ) : null}
          </>
        ) : null}
      </div>
      {showConnect ? (
        <div
          data-testid="browser-connect-hint"
          className="basis-full flex flex-wrap items-center gap-2 text-[11px] text-slate-400"
        >
          <span>{copy.connect.hint}</span>
          <button
            type="button"
            data-testid="browser-connect"
            onClick={onConnect}
            className="underline text-cyan-300"
          >
            {copy.connect.action}
          </button>
        </div>
      ) : null}
    </div>
  );
};
