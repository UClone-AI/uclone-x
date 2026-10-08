"""Rules behind the page actions that need no browser (design `browser-agent.md` §3.3).

Key names and chords, the password-field rule of §3.5, what an open dialog says, how far
a scroll goes, and the page-side functions the service runs on an element. The service
in `service.py` sends them over CDP; everything here is testable without Chrome.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from uclone_x.browser.snapshot import Entry
from uclone_x.errors import PlainRefusalError

SECRET_FIELD_REFUSAL = (
    "That field takes a password or a one-time code, which a clone never types. "
    "Use sign_in, or ask the person to take over."
)
STALE_REF = "That element is no longer on the page — take a new snapshot."
POPUP_NOT_FOLLOWED = (
    "The page opened a new tab, but in your own Chrome the clone can't follow it yet, so it "
    "stays on this tab. If the new page's address is shown here, open it with action=open."
)
"""Said when the link cannot attach to a pop-up: the extension attaches only to tabs it opened."""

_SECRET_AUTOCOMPLETE = frozenset({"one-time-code", "current-password", "new-password"})


def is_secret_field(input_type: str, autocomplete: str) -> bool:
    """Whether §3.5 reserves the field for `sign_in` or the person."""
    if input_type.strip().lower() == "password":
        return True
    return bool(_SECRET_AUTOCOMPLETE & set(autocomplete.lower().split()))


# -- keys --------------------------------------------------------------------------------

# name -> (key, code, windowsVirtualKeyCode, modifier bit)
_MODIFIERS = {
    "alt": ("Alt", "AltLeft", 18, 1),
    "control": ("Control", "ControlLeft", 17, 2),
    "meta": ("Meta", "MetaLeft", 91, 4),
    "shift": ("Shift", "ShiftLeft", 16, 8),
}
_MODIFIER_ALIASES = {
    "ctrl": "control",
    "cmd": "meta",
    "command": "meta",
    "⌘": "meta",
    "option": "alt",
    "opt": "alt",
}

# name -> (key, code, windowsVirtualKeyCode, text)
_KEYS: dict[str, tuple[str, str, int, str]] = {
    "enter": ("Enter", "Enter", 13, "\r"),
    "tab": ("Tab", "Tab", 9, ""),
    "escape": ("Escape", "Escape", 27, ""),
    "backspace": ("Backspace", "Backspace", 8, ""),
    "delete": ("Delete", "Delete", 46, ""),
    "space": (" ", "Space", 32, " "),
    "arrowup": ("ArrowUp", "ArrowUp", 38, ""),
    "arrowdown": ("ArrowDown", "ArrowDown", 40, ""),
    "arrowleft": ("ArrowLeft", "ArrowLeft", 37, ""),
    "arrowright": ("ArrowRight", "ArrowRight", 39, ""),
    "home": ("Home", "Home", 36, ""),
    "end": ("End", "End", 35, ""),
    "pageup": ("PageUp", "PageUp", 33, ""),
    "pagedown": ("PageDown", "PageDown", 34, ""),
}
_KEY_ALIASES = {
    "return": "enter",
    "esc": "escape",
    "up": "arrowup",
    "down": "arrowdown",
    "left": "arrowleft",
    "right": "arrowright",
    "del": "delete",
}
# With ⌘ or Ctrl alone, these letters are editing commands Chrome only runs when told to.
_COMMANDS = {"a": "selectAll", "c": "copy", "x": "cut", "v": "paste", "z": "undo"}

_UNKNOWN_KEY = (
    'I don\'t know the key "{chord}". Use a key name such as Enter, Escape, Tab, '
    "ArrowDown or Backspace, a single character, or a chord such as Control+A."
)


@dataclass(frozen=True)
class KeyPress:
    """One key, with the modifiers held while it is pressed."""

    key: str
    code: str
    key_code: int
    text: str = ""
    modifiers: int = 0
    held: tuple[tuple[str, str, int], ...] = ()
    commands: tuple[str, ...] = ()


def parse_chord(chord: str) -> KeyPress:
    """`Enter`, `a`, `Shift+Tab`, `Meta+A` … as the key events will need them."""
    refused = PlainRefusalError(_UNKNOWN_KEY.format(chord=chord), reason_code="unknown_key")
    parts = chord.split("+") if chord not in {"+", ""} else [chord]
    if chord.endswith("++"):
        parts = [*chord[:-2].split("+"), "+"]
    *mods, last = parts
    held: list[tuple[str, str, int]] = []
    bits = 0
    for raw in mods:
        name = _MODIFIER_ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if name not in _MODIFIERS:
            raise refused
        key, code, key_code, bit = _MODIFIERS[name]
        if not bits & bit:
            held.append((key, code, key_code))
            bits |= bit
    name = last.strip().lower() if len(last.strip()) > 1 else last
    name = _KEY_ALIASES.get(name, name)
    if name in _KEYS:
        key, code, key_code, text = _KEYS[name]
    elif len(last) == 1 and last.isprintable():
        key, code, key_code, text = _character(last, shift=bool(bits & 8))
    else:
        raise refused
    editing = bits in {2, 4} or bits in {2 | 8, 4 | 8}
    commands: tuple[str, ...] = ()
    if editing and len(key) == 1 and key.lower() in _COMMANDS:
        command = _COMMANDS[key.lower()]
        commands = ("redo",) if command == "undo" and bits & 8 else (command,)
    if bits & (1 | 2 | 4):
        text = ""  # a chord with Alt, Ctrl or ⌘ types nothing
    return KeyPress(key, code, key_code, text, bits, tuple(held), commands)


def _character(char: str, *, shift: bool) -> tuple[str, str, int, str]:
    if char.isalpha() and char.isascii():
        shown = char.upper() if shift else char.lower()
        return shown, f"Key{char.upper()}", ord(char.upper()), shown
    if char.isdigit() and char.isascii():
        return char, f"Digit{char}", ord(char), char
    return char, "", 0, char


def key_events(press: KeyPress) -> list[dict[str, Any]]:
    """The `Input.dispatchKeyEvent` calls for one press, in order."""
    events: list[dict[str, Any]] = []
    bits = 0
    for key, code, key_code in press.held:
        bits |= next(m[3] for m in _MODIFIERS.values() if m[0] == key)
        events.append(_key_event("rawKeyDown", key, code, key_code, bits))
    down = _key_event(
        "keyDown" if press.text else "rawKeyDown",
        press.key,
        press.code,
        press.key_code,
        press.modifiers,
    )
    if press.text:
        down["text"] = press.text
        down["unmodifiedText"] = press.text
    if press.commands:
        down["commands"] = list(press.commands)
    events.append(down)
    events.append(_key_event("keyUp", press.key, press.code, press.key_code, press.modifiers))
    for key, code, key_code in reversed(press.held):
        bits &= ~next(m[3] for m in _MODIFIERS.values() if m[0] == key)
        events.append(_key_event("keyUp", key, code, key_code, bits))
    return events


def _key_event(kind: str, key: str, code: str, key_code: int, modifiers: int) -> dict[str, Any]:
    return {
        "type": kind,
        "key": key,
        "code": code,
        "windowsVirtualKeyCode": key_code,
        "modifiers": modifiers,
    }


# -- observations ------------------------------------------------------------------------


def dialog_note(dialog: Mapping[str, Any]) -> str:
    """What an open JavaScript dialog says and how the clone answers it."""
    kind = str(dialog.get("type", "alert"))
    message = str(dialog.get("message", ""))
    if kind == "confirm":
        how = "Answer it with press Enter (OK) or press Escape (Cancel)."
    elif kind == "prompt":
        how = "Answer it with type and your text, or press Escape to cancel."
    elif kind == "beforeunload":
        how = "Press Enter to leave the page or press Escape to stay."
    else:
        how = "Close it with press Enter."
    return f'The page shows a {kind} dialog: "{message}". {how}'


def scroll_delta(direction: str, amount: float, width: float, height: float) -> tuple[int, int]:
    """Pixels to scroll: `amount` screens, each 80% of the visible size so lines overlap."""
    step = 0.8 * amount
    if direction in {"left", "right"}:
        dx = round(width * step)
        return (dx if direction == "right" else -dx, 0)
    dy = round(height * step)
    return (0, dy if direction == "down" else -dy)


_OPTION_ROLES = frozenset({"option", "menuitem", "menuitemradio", "menuitemcheckbox", "treeitem"})


def pick_option(entries: Sequence[Entry], wanted: str) -> Entry | None:
    """The option of an opened custom dropdown whose name matches, exact names first."""
    needle = wanted.strip().casefold()
    options = [e for e in entries if e.role in _OPTION_ROLES and e.ref]
    for entry in options:
        if entry.name.strip().casefold() == needle:
            return entry
    for entry in options:
        if needle and needle in entry.name.casefold():
            return entry
    return None


# -- page-side functions (Runtime.callFunctionOn with the element as `this`) ---------------

FIELD_INFO_JS = """function () {
  const el = this;
  const tag = el.tagName ? el.tagName.toLowerCase() : '';
  const type = (el.getAttribute && (el.getAttribute('type') || '')) || '';
  const textTypes = ['', 'text', 'search', 'email', 'url', 'tel', 'number', 'password',
                     'date', 'datetime-local', 'month', 'time', 'week'];
  const editable = (tag === 'input' && textTypes.includes(type.toLowerCase())) ||
                   tag === 'textarea' || !!el.isContentEditable;
  const role = (el.getAttribute && el.getAttribute('role')) || '';
  let checked = null;
  if (tag === 'input' && (el.type === 'checkbox' || el.type === 'radio')) checked = el.checked;
  else if (el.getAttribute && el.hasAttribute('aria-checked'))
    checked = el.getAttribute('aria-checked') === 'true';
  else if (el.getAttribute && el.hasAttribute('aria-pressed'))
    checked = el.getAttribute('aria-pressed') === 'true';
  return {
    tag, type: tag === 'input' ? el.type : type, role,
    autocomplete: (el.getAttribute && el.getAttribute('autocomplete')) || '',
    editable, checked,
    disabled: !!el.disabled || (el.getAttribute && el.getAttribute('aria-disabled') === 'true'),
    readonly: !!el.readOnly,
    radio: (tag === 'input' && el.type === 'radio') || role === 'radio',
    select: tag === 'select', file: tag === 'input' && el.type === 'file',
  };
}"""

# Resolves once the element is visible, enabled, holding still and not covered, or with the
# last reason it was not after `timeout` ms. Timers, not animation frames: a background tab
# pauses animation frames but still runs timers.
ACTIONABLE_JS = """function (timeout, hitTest) {
  // A checkbox styled away (zero-size, transparent) is used through its label.
  let el = this;
  if (this.labels && this.labels.length) {
    const r = this.getBoundingClientRect(), s = getComputedStyle(this);
    if (r.width < 2 || r.height < 2 || s.display === 'none' || s.visibility === 'hidden' ||
        s.opacity === '0') el = this.labels[0];
  }
  const self = this;
  const until = Date.now() + timeout;
  const rectOf = () => { const r = el.getBoundingClientRect();
                         return [r.left, r.top, r.width, r.height]; };
  const check = (before) => {
    if (!self.isConnected) return {reason: 'detached'};
    const style = getComputedStyle(el);
    const [x, y, w, h] = rectOf();
    if (w === 0 || h === 0 || style.visibility === 'hidden' || style.display === 'none')
      return {reason: 'hidden'};
    if (self.disabled || self.getAttribute('aria-disabled') === 'true')
      return {reason: 'disabled'};
    if (before && (before[0] !== x || before[1] !== y || before[2] !== w || before[3] !== h))
      return {reason: 'moving'};
    const cx = x + w / 2, cy = y + h / 2;
    if (hitTest) {
      let hit = document.elementFromPoint(cx, cy);
      while (hit && hit.shadowRoot && hit.shadowRoot.elementFromPoint) {
        const inner = hit.shadowRoot.elementFromPoint(cx, cy);
        if (!inner || inner === hit) break;
        hit = inner;
      }
      const label = self.labels && self.labels.length ? self.labels[0] : null;
      if (hit && hit !== el && !el.contains(hit) && hit !== self &&
          !(label && label.contains(hit)))
        return {reason: 'covered'};
    }
    return {reason: '', x: cx, y: cy, box: rectOf()};
  };
  return new Promise((resolve) => {
    const step = () => {
      const before = el.isConnected ? rectOf() : null;
      setTimeout(() => {
        const result = check(before);
        if (!result.reason || result.reason === 'detached' || Date.now() >= until) resolve(result);
        else setTimeout(step, 50);
      }, 30);
    };
    step();
  });
}"""

IS_FOCUSED_JS = """function () {
  let active = document.activeElement;
  while (active && active.shadowRoot && active.shadowRoot.activeElement)
    active = active.shadowRoot.activeElement;
  return active === this || this.contains(active);
}"""

SELECT_ALL_JS = """function () {
  if (typeof this.select === 'function') { this.select(); return; }
  const range = document.createRange(); range.selectNodeContents(this);
  const sel = getSelection(); sel.removeAllRanges(); sel.addRange(range);
}"""

CARET_TO_END_JS = """function () {
  if (typeof this.setSelectionRange === 'function') {
    try { const n = this.value.length; this.setSelectionRange(n, n); return true; }
    catch (e) { return false; }
  }
  const range = document.createRange(); range.selectNodeContents(this); range.collapse(false);
  const sel = getSelection(); sel.removeAllRanges(); sel.addRange(range);
  return true;
}"""

# Chooses the options of a native <select> by label or value, as a person picking them would.
SELECT_OPTIONS_JS = """function (wanted) {
  const norm = (s) => s.trim().toLowerCase();
  const options = Array.from(this.options);
  const missing = [];
  const chosen = new Set();
  for (const w of wanted) {
    const hit = options.find((o) => norm(o.label) === norm(w) || o.value === w) ||
                options.find((o) => norm(o.label).includes(norm(w)));
    if (hit) chosen.add(hit); else missing.push(w);
  }
  if (missing.length) return {missing, options: options.map((o) => o.label)};
  if (!this.multiple && chosen.size > 1) return {multiple: false};
  for (const o of options) o.selected = chosen.has(o);
  this.dispatchEvent(new Event('input', {bubbles: true}));
  this.dispatchEvent(new Event('change', {bubbles: true}));
  return {missing: []};
}"""

ELEMENT_SCROLL_JS = """function () {
  const r = this.getBoundingClientRect();
  const x = Math.min(Math.max(r.left + r.width / 2, 1), innerWidth - 1);
  const y = Math.min(Math.max(r.top + r.height / 2, 1), innerHeight - 1);
  return {x, y, width: this.clientWidth || r.width, height: this.clientHeight || r.height};
}"""

ELEMENT_SCROLL_POSITION_JS = """function () {
  const room = this.scrollHeight - this.clientHeight;
  return room > 0 ? Math.round(this.scrollTop / room * 100) : -1;
}"""

PAGE_SCROLL_POSITION = (
    "(() => { const s = document.scrollingElement || document.documentElement;"
    " const room = s.scrollHeight - innerHeight;"
    " return room > 0 ? Math.round(scrollY / room * 100) : -1; })()"
)

ACTIVE_FIELD = """(() => {
  let a = document.activeElement;
  while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
  if (!a || !a.getAttribute) return {type: '', autocomplete: ''};
  return {type: a.tagName === 'INPUT' ? a.type : '',
          autocomplete: a.getAttribute('autocomplete') || ''};
})()"""


def wait_for_text_expression(text: str, timeout_ms: int) -> str:
    """A promise that resolves `true` once `text` shows on the page, or `false` at the timeout."""
    return f"""new Promise((resolve) => {{
  const wanted = {json.dumps(text)}.toLowerCase();
  const seen = () => (document.body ? document.body.innerText : '').toLowerCase().includes(wanted);
  if (seen()) return resolve(true);
  let queued = false;
  const observer = new MutationObserver(() => {{
    if (queued) return; queued = true;
    setTimeout(() => {{ queued = false;
      if (seen()) {{ observer.disconnect(); clearTimeout(timer); resolve(true); }} }}, 50);
  }});
  observer.observe(document, {{subtree: true, childList: true, characterData: true,
                                attributes: true}});
  const timer = setTimeout(() => {{ observer.disconnect(); resolve(seen()); }}, {timeout_ms});
}})"""
