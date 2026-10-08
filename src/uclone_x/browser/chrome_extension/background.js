// UClone-X in U0's own Chrome: the R1 route of the browser agent.
//
// The service worker dials the Core over a loopback WebSocket, pairs with the code Settings
// shows, and then relays the Chrome DevTools Protocol between the Core and the tabs it opens
// for clones. The wire is CDP's own framing, with the tab id as the session id:
//
//   Core -> here   {id, method, params, sessionId?}    a CDP command, or a UClone.* one
//   here -> Core   {id, result} | {id, error}          its answer
//   here -> Core   {method, params, sessionId}         a CDP event from a tab
//
// Nothing here hides that the tab is controlled: Chrome shows its debugger bar on every
// attached tab, and closing it detaches, which is reported as UClone.detached with Chrome's
// reason ("canceled_by_user" is the person taking the tab back).

const PROTOCOL_VERSION = "1.3";
const GROUP_TITLE = "UClone-X";
const KEEPALIVE_MS = 20000; // a WebSocket message every <30 s keeps the worker alive (Chrome 116+)
const REDIAL_ALARM = "uclone-redial";

let socket = null;
let keepalive = null;
let groupId = null;
const attached = new Set(); // tab ids with the debugger attached
const enabled = new Map(); // tab id -> CDP "*.enable" calls to replay after a reattach

/** "8765-<token>" -> {port, token}, or null when it is not a pairing code. */
function parseCode(code) {
  const match = /^\s*(\d{2,5})-([0-9a-f]{16,128})\s*$/i.exec(code || "");
  if (!match) return null;
  const port = Number(match[1]);
  return port > 0 && port < 65536 ? { port, token: match[2].toLowerCase() } : null;
}

async function setStatus(status) {
  await chrome.storage.local.set({ status });
}

async function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const { pairingCode, refusedCode } = await chrome.storage.local.get(["pairingCode", "refusedCode"]);
  const pairing = parseCode(pairingCode);
  if (!pairing) {
    await setStatus("unpaired");
    return;
  }
  if (refusedCode === pairingCode) return; // refused already: wait for a new code
  await setStatus("connecting");
  const ws = new WebSocket(`ws://127.0.0.1:${pairing.port}/api/browser/extension`);
  socket = ws;
  let paired = false;
  ws.onopen = () => {
    ws.send(
      JSON.stringify({ type: "hello", token: pairing.token, version: chrome.runtime.getManifest().version })
    );
  };
  ws.onmessage = async (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (!paired) {
      if (message.type === "paired") {
        paired = true;
        await setStatus("connected");
        keepalive = setInterval(() => send({ keepalive: true }), KEEPALIVE_MS);
      } else if (message.type === "refused") {
        await chrome.storage.local.set({ status: "refused", refusedCode: pairingCode });
      }
      return;
    }
    if (typeof message.id === "number" && typeof message.method === "string") {
      await answer(message);
    }
  };
  ws.onclose = async () => {
    if (socket !== ws) return; // replaced by a newer dial
    socket = null;
    clearInterval(keepalive);
    keepalive = null;
    const { status } = await chrome.storage.local.get("status");
    if (status !== "refused") await setStatus(paired ? "connecting" : "unreachable");
  };
}

function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

async function answer(message) {
  try {
    const result = await handle(message.method, message.params || {}, message.sessionId);
    send({ id: message.id, result: result || {} });
  } catch (error) {
    send({ id: message.id, error: { code: -32000, message: String((error && error.message) || error) } });
  }
}

async function handle(method, params, sessionId) {
  switch (method) {
    case "UClone.openTab":
      return openTab(params.url || "about:blank");
    case "UClone.closeTab":
      await release(Number(params.tabId));
      await chrome.tabs.remove(Number(params.tabId));
      return {};
    case "UClone.releaseTab":
      await release(Number(params.tabId));
      return {};
    default: {
      if (sessionId === undefined || sessionId === null) {
        throw new Error(`${method} needs a tab`);
      }
      const tabId = Number(sessionId);
      await ensureAttached(tabId);
      const result = await chrome.debugger.sendCommand({ tabId }, method, params);
      if (method.endsWith(".enable")) {
        const calls = enabled.get(tabId) || new Map();
        calls.set(method, params);
        enabled.set(tabId, calls);
      }
      return result;
    }
  }
}

async function openTab(url) {
  const tab = await chrome.tabs.create({ url, active: false });
  try {
    if (groupId !== null) {
      await chrome.tabs.group({ groupId, tabIds: [tab.id] });
    } else {
      throw new Error("no group yet");
    }
  } catch {
    groupId = await chrome.tabs.group({ tabIds: [tab.id] });
    await chrome.tabGroups.update(groupId, { title: GROUP_TITLE, color: "blue" });
  }
  await ensureAttached(tab.id);
  return { tabId: String(tab.id) };
}

// A tab the person took back (closed the debugger bar) is attached again on the clone's next
// call, and the bar comes back with it. The domains the Core enabled are enabled again, so its
// events keep arriving as before.
async function ensureAttached(tabId) {
  if (attached.has(tabId)) return;
  await chrome.debugger.attach({ tabId }, PROTOCOL_VERSION);
  attached.add(tabId);
  const calls = enabled.get(tabId);
  if (calls) {
    for (const [method, params] of calls) {
      await chrome.debugger.sendCommand({ tabId }, method, params);
    }
  }
}

async function release(tabId) {
  enabled.delete(tabId);
  if (!attached.has(tabId)) return;
  attached.delete(tabId);
  try {
    await chrome.debugger.detach({ tabId });
  } catch {
    // Already detached: the person closed the bar, or the tab is gone.
  }
}

chrome.debugger.onEvent.addListener((source, method, params) => {
  if (source.tabId === undefined || !attached.has(source.tabId)) return;
  send({ method, params: params || {}, sessionId: String(source.tabId) });
});

chrome.debugger.onDetach.addListener((source, reason) => {
  if (source.tabId === undefined || !attached.delete(source.tabId)) return;
  send({ method: "UClone.detached", params: { tabId: String(source.tabId), reason }, sessionId: String(source.tabId) });
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (!attached.delete(tabId) && !enabled.has(tabId)) return;
  enabled.delete(tabId);
  send({ method: "UClone.tabClosed", params: { tabId: String(tabId) }, sessionId: String(tabId) });
});

// An idle worker is suspended and its timers die with it, so it is woken to redial.
chrome.alarms.create(REDIAL_ALARM, { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === REDIAL_ALARM) connect();
});
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener(connect);
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local" || !changes.pairingCode) return;
  const old = socket;
  socket = null;
  if (old) old.close();
  chrome.storage.local.remove("refusedCode").then(connect);
});

connect();

// The options page asks for a fresh dial after saving, even when the code did not change.
chrome.runtime.onMessage.addListener((message) => {
  if (message && message.type === "redial") {
    chrome.storage.local.remove("refusedCode").then(connect);
  }
});
