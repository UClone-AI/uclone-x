"""Pure rules behind the browser's page actions (design `browser-agent.md` §3.3, §3.5)."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from uclone_x.browser.actions import (
    POPUP_NOT_FOLLOWED,
    SECRET_FIELD_REFUSAL,
    dialog_note,
    is_secret_field,
    key_events,
    parse_chord,
    pick_option,
    scroll_delta,
)
from uclone_x.browser.cdp import CdpError
from uclone_x.browser.link import BrowserLink
from uclone_x.browser.service import (
    BrowserService,
    _Tab,  # pyright: ignore[reportPrivateUsage]
    _Window,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.browser.snapshot import Entry
from uclone_x.errors import PlainRefusalError


@pytest.mark.parametrize(
    ("input_type", "autocomplete"),
    [
        ("password", ""),
        ("PASSWORD", "username"),
        ("text", "one-time-code"),
        ("text", "section-login current-password"),
        ("", "new-password"),
        ("tel", "One-Time-Code"),
    ],
)
def test_password_and_one_time_code_fields_are_secret(input_type: str, autocomplete: str) -> None:
    assert is_secret_field(input_type, autocomplete)


@pytest.mark.parametrize(
    ("input_type", "autocomplete"),
    [("text", ""), ("email", "username"), ("search", "off"), ("text", "password-hint")],
)
def test_ordinary_fields_are_not_secret(input_type: str, autocomplete: str) -> None:
    assert not is_secret_field(input_type, autocomplete)


def test_the_secret_field_refusal_is_plain_copy() -> None:
    assert "Use sign_in, or ask the person to take over" in SECRET_FIELD_REFUSAL
    for internal in ("autocomplete", "type=", "CDP", "backend", "Error"):
        assert internal not in SECRET_FIELD_REFUSAL


def test_enter_inserts_a_carriage_return() -> None:
    press = parse_chord("Enter")
    assert (press.key, press.code, press.key_code, press.text) == ("Enter", "Enter", 13, "\r")
    assert press.modifiers == 0


def test_chords_set_modifier_bits_and_editing_commands() -> None:
    select_all = parse_chord("Meta+A")
    assert select_all.modifiers == 4
    assert select_all.text == ""
    assert select_all.commands == ("selectAll",)
    assert parse_chord("ctrl+a").commands == ("selectAll",)
    assert parse_chord("Cmd+Shift+Z").commands == ("redo",)
    assert parse_chord("Shift+Tab").modifiers == 8
    assert parse_chord("Alt+ArrowLeft").modifiers == 1


def test_aliases_and_case_are_forgiven() -> None:
    assert parse_chord("esc").key == "Escape"
    assert parse_chord("return").key == "Enter"
    assert parse_chord("down").key == "ArrowDown"
    assert parse_chord("space").text == " "
    assert parse_chord("Shift+a").text == "A"
    assert parse_chord("7").code == "Digit7"


@pytest.mark.parametrize("chord", ["Hyper", "Control+", "Meta+Shift", "Control+Foo+A", ""])
def test_an_unknown_key_is_refused_in_plain_words(chord: str) -> None:
    with pytest.raises(PlainRefusalError) as refused:
        parse_chord(chord)
    message = str(refused.value)
    assert "Enter" in message
    assert "Error" not in message


def test_key_events_press_modifiers_first_and_release_them_last() -> None:
    events = key_events(parse_chord("Control+A"))
    assert [(e["type"], e["key"]) for e in events] == [
        ("rawKeyDown", "Control"),
        ("rawKeyDown", "a"),
        ("keyUp", "a"),
        ("keyUp", "Control"),
    ]
    assert events[1]["modifiers"] == 2
    assert events[1]["commands"] == ["selectAll"]


def test_a_key_that_types_text_is_sent_as_key_down_with_its_text() -> None:
    events = key_events(parse_chord("Enter"))
    assert events[0] == {
        "type": "keyDown",
        "key": "Enter",
        "code": "Enter",
        "windowsVirtualKeyCode": 13,
        "modifiers": 0,
        "text": "\r",
        "unmodifiedText": "\r",
    }
    assert events[1]["type"] == "keyUp"


def test_scroll_moves_most_of_a_screen_per_unit() -> None:
    assert scroll_delta("down", 1, 1000, 800) == (0, 640)
    assert scroll_delta("up", 2, 1000, 800) == (0, -1280)
    assert scroll_delta("right", 1, 1000, 800) == (800, 0)
    assert scroll_delta("left", 0.5, 1000, 800) == (-400, 0)


def test_dialog_notes_say_how_to_answer() -> None:
    alert = dialog_note({"type": "alert", "message": "저장했습니다"})
    assert '"저장했습니다"' in alert
    assert "press Enter" in alert
    confirm = dialog_note({"type": "confirm", "message": "Delete?"})
    assert "press Enter" in confirm and "press Escape" in confirm
    prompt = dialog_note({"type": "prompt", "message": "Name?", "defaultPrompt": ""})
    assert "type" in prompt


def _option(name: str, role: str = "option", ref: str = "e1") -> Entry:
    return Entry(role=role, name=name, depth=0, ref=ref)


def test_pick_option_prefers_an_exact_name_over_a_partial_one() -> None:
    entries = [
        _option("Seoul Station", ref="e1"),
        _option("Seoul", ref="e2"),
        Entry(role="button", name="Seoul", depth=0, ref="e3"),
    ]
    assert pick_option(entries, "seoul") == entries[1]
    assert pick_option(entries, "Station") == entries[0]
    assert pick_option(entries, "Busan") is None
    assert pick_option([_option("부산", role="menuitem")], "부산") is not None


class _PopupLink:
    """Just enough of a link for pop-up adoption: one pop-up listed, adoption refused."""

    def __init__(self, listing: Exception | None = None) -> None:
        self.listing = listing

    async def send(self, tab: str, method: str, params: object = None) -> dict[str, object]:
        assert method == "Target.getTargets"
        if self.listing is not None:
            raise self.listing
        return {"targetInfos": [{"type": "page", "targetId": "T-POP", "openerId": tab}]}

    async def adopt_tab(self, opener: str, tab: str) -> str:
        raise PlainRefusalError(POPUP_NOT_FOLLOWED, reason_code="popup_not_followed")


@pytest.mark.parametrize("listing", [None, CdpError("Target.getTargets", -32000, "Not allowed")])
async def test_a_pop_up_the_link_cannot_follow_is_a_note_not_a_failure(
    listing: Exception | None,
) -> None:
    # Over R1 the click still happened; the clone is told why it stays on its tab.
    service = BrowserService(_no_link)
    opener = _Tab(target="T-1", events=asyncio.Queue())
    window = _Window([opener])

    adopted, note = await service._adopt_popups(  # pyright: ignore[reportPrivateUsage]
        cast(BrowserLink, _PopupLink(listing)), window, opener
    )

    assert not adopted
    assert note == POPUP_NOT_FOLLOWED
    assert window.tabs == [opener]
    assert "Not allowed" not in POPUP_NOT_FOLLOWED


async def _no_link() -> BrowserLink:
    raise AssertionError("the test hands the link in directly")
