"""The accessibility snapshot a clone reads (design `browser-agent.md` §3.2).

Trees here are hand-built in the shape `Accessibility.getFullAXTree` returns; the e2e suite
checks the same functions against a real Chromium.
"""

from __future__ import annotations

from typing import Any

from uclone_x.browser.snapshot import RefTable, build_entries, diff, find, render


def _node(
    node_id: int,
    role: str,
    name: str = "",
    *,
    children: tuple[int, ...] = (),
    parent: int | None = None,
    backend: int | None = None,
    value: str | None = None,
    ignored: bool = False,
    props: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "nodeId": str(node_id),
        "ignored": ignored,
        "role": {"type": "role", "value": role},
        "name": {"type": "computedString", "value": name},
        "childIds": [str(c) for c in children],
        "backendDOMNodeId": backend if backend is not None else node_id + 1000,
    }
    if parent is not None:
        node["parentId"] = str(parent)
    if value is not None:
        node["value"] = {"type": "string", "value": value}
    if props:
        node["properties"] = [{"name": k, "value": {"value": v}} for k, v in props.items()]
    return node


def _form_tree() -> list[dict[str, Any]]:
    return [
        _node(1, "RootWebArea", "양식", children=(2,)),
        _node(2, "generic", ignored=True, parent=1, children=(3, 5, 6, 8)),
        _node(3, "heading", "여행 찾기", parent=2, children=(4,), props={"level": 1}),
        _node(4, "StaticText", "여행 찾기", parent=3),
        _node(5, "textbox", "검색어", parent=2, value="부산", props={"required": True}),
        _node(6, "button", "보내기", parent=2, children=(7,)),
        _node(7, "StaticText", "보내기", parent=6),
        _node(8, "checkbox", "동의", parent=2, props={"checked": "true", "disabled": True}),
    ]


def test_controls_get_refs_and_text_is_not_repeated() -> None:
    text = render(build_entries(_form_tree(), RefTable()))

    assert text.splitlines() == [
        '- heading "여행 찾기" [level=1]',
        '- textbox "검색어" [ref=e1, required] value="부산"',
        '- button "보내기" [ref=e2]',
        '- checkbox "동의" [ref=e3, checked, disabled]',
    ]


def test_a_ref_stays_with_its_element_across_snapshots() -> None:
    refs = RefTable()
    build_entries(_form_tree(), refs)
    reordered = _form_tree()
    reordered[1]["childIds"] = ["8", "6", "5", "3"]

    text = render(build_entries(reordered, refs))

    assert '- checkbox "동의" [ref=e3, checked, disabled]' in text
    assert refs.node_for("e3") == 1008
    assert refs.node_for("e9") is None


def test_an_open_dialog_is_listed_first() -> None:
    nodes = [
        _node(1, "RootWebArea", children=(2, 3)),
        _node(2, "link", "홈", parent=1),
        _node(3, "dialog", "쿠키 안내", parent=1, children=(4,)),
        _node(4, "button", "동의", parent=3),
    ]

    lines = render(build_entries(nodes, RefTable())).splitlines()

    assert lines == [
        '- dialog "쿠키 안내"',
        '  - button "동의" [ref=e1]',
        '- link "홈" [ref=e2]',
    ]


def test_a_long_list_is_collapsed_after_ten_items() -> None:
    items = tuple(range(10, 40))
    nodes = [
        _node(1, "RootWebArea", children=(2,)),
        _node(2, "list", parent=1, children=items),
        *[_node(i, "listitem", f"항목 {i}", parent=2, children=(i + 100,)) for i in items],
        *[_node(i + 100, "link", f"항목 {i}", parent=i) for i in items],
    ]

    entries = build_entries(nodes, RefTable())
    text = render(entries)

    assert 'link "항목 19"' in text
    assert 'link "항목 20"' not in text
    assert text.splitlines()[-1] == "  - …and 20 more"
    # find looks past the collapse.
    hits = find(build_entries(nodes, RefTable(), collapse=False), "항목 35")
    assert [h.line() for h in hits] == ['- link "항목 35" [ref=e26]']


def test_layout_containers_with_nothing_readable_are_dropped() -> None:
    nodes = [
        _node(1, "RootWebArea", children=(2, 4)),
        _node(2, "table", parent=1, children=(3,)),
        _node(3, "row", parent=2),
        _node(4, "link", "다음", parent=1),
    ]

    assert render(build_entries(nodes, RefTable())) == '- link "다음" [ref=e1]'


def test_a_long_snapshot_is_cut_with_a_note_saying_how_to_go_on() -> None:
    ids = tuple(range(2, 400))
    nodes = [
        _node(1, "RootWebArea", children=ids),
        *[_node(i, "link", f"링크 번호 {i}", parent=1) for i in ids],
    ]

    text = render(build_entries(nodes, RefTable()), max_chars=500)

    assert len(text) < 700
    assert text.splitlines()[-1].endswith("use find to reach elements further down the page")


def test_an_empty_page_says_so() -> None:
    assert render([]) == "(The page shows nothing a clone can read.)"


def test_diff_names_added_removed_and_changed_elements() -> None:
    refs = RefTable()
    before = build_entries(_form_tree(), refs)
    changed = _form_tree()
    changed[4]["value"] = {"value": "제주"}
    changed[1]["childIds"] = ["3", "5", "9"]
    changed.append(_node(9, "StaticText", "보냈습니다", parent=2))
    after = build_entries(changed, refs)

    assert diff(before, after).splitlines() == [
        '~ textbox "검색어" [ref=e1, required] value="제주"',
        '- button "보내기" [ref=e2]',
        '- checkbox "동의" [ref=e3, checked, disabled]',
        '+ text "보냈습니다"',
    ]
    assert diff(after, after) == "Nothing on the page changed."


def test_find_lists_controls_before_plain_text() -> None:
    nodes = [
        _node(1, "RootWebArea", children=(2, 3)),
        _node(2, "StaticText", "다음 페이지로 가려면", parent=1),
        _node(3, "link", "다음 페이지", parent=1),
    ]

    hits = find(build_entries(nodes, RefTable()), "다음 페이지")

    assert [h.line() for h in hits] == [
        '- link "다음 페이지" [ref=e1]',
        '- text "다음 페이지로 가려면"',
    ]
    assert find(build_entries(nodes, RefTable()), "  ") == []
