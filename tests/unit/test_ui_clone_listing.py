"""The head's list of installed clones -- `GET /api/clones`.

Written before `uclone_x.ui.clones` existed. What they pin, in order of what it would
cost to get wrong:

* **A fresh install is not an empty screen.** `GET /api/agents` returns live instances,
  so on an install where nothing has been spawned it is empty -- which is exactly the
  first screen a new user sees. The clones themselves are directories under the agents
  root, and this route is the only thing that reads them.
* **Running is reported, never inferred.** A clone that has a home and has never run and
  a clone that is running are drawn differently. If the wire carried only the home, the
  surface would have to cross-reference a second endpoint and guess.
* **Three absences do not render as one empty list.** Nothing installed, a root that
  cannot be read, and one damaged home are three facts; a bare `[]` is the same value for
  all three, and only two of them are something a reader can act on.
* **One damaged home does not empty the list.** The damage belongs to that row.
* **Running means running, on the path a user actually takes.** D1 makes every new
  conversation a room, so a clone is normally seated by a `RoomAgentResolver` and never
  appears in the session manager's chat map. A join that reads only that map calls the
  ordinary running clone "Installed and not running".
* **One predicate, not two.** The sentence above the list counts the rows the list holds.
  Counting the *disk* listing instead prints "No clone is installed yet." above a clone.
* **The wire speaks the product's vocabulary.** A reason is rendered verbatim, so a
  sentence half in the Core's words (`agent`, `agent home`, `id file`) is the confusion
  design §3.1.1 draws the boundary to end.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentState
from uclone_x.core.agent_home import AGENT_ID_PREFIX, AGENTS_DIR_ENV_VAR
from uclone_x.llm import MockLLMConnector
from uclone_x.ui.app import AgentSessionManager, create_ui_app


def _client(tmp_path: Path, session_mgr: AgentSessionManager | None = None) -> TestClient:
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(
        static_dir=tmp_path / "static",
        storage_dir=storage_dir,
        llm=MockLLMConnector(),
        session_manager=session_mgr,
    )
    return TestClient(app)


def _seat_a_live_clone(session_mgr: AgentSessionManager, username: str) -> None:
    """Register a running instance under `username`, as a chat turn would."""
    agent = BaseAgent(
        config=AgentConfig(agent_id=username, name=username),
        bus=session_mgr.bus,
        llm=MockLLMConnector(),
        tools=session_mgr.tools,
        context=AgentContext(
            session_id=f"sess_{username}",
            agent_id=username,
            current_state=AgentState.IDLE,
        ),
        store=session_mgr.core_store,
    )
    session_mgr._agents[f"{username}:sess_{username}"] = agent  # pyright: ignore[reportPrivateUsage]


def _body(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/clones")
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], response.json())


def _wait_for_transcript(client: TestClient, room_id: str, rows: int) -> None:
    """Poll a conversation until the cascade has written `rows`, or give up loudly.

    Sending answers 202 and runs the turns behind it, so there is no response to await.
    Polling the record is what a head does too.
    """
    deadline = time.monotonic() + 10.0
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = client.get(f"/api/rooms/{room_id}").json()
        if len(latest.get("transcript", [])) >= rows:
            return
        time.sleep(0.05)
    raise AssertionError(f"conversation never reached {rows} rows; last was {latest}")


def _reason_of(body: dict[str, Any], name: str) -> str:
    return cast(str, next(c for c in body["clones"] if c["name"] == name)["reason"])


#: Stand-ins for the two things a reason renders verbatim off the reader's own machine:
#: the folder the clones are in, and the names they were given. Neither stand-in carries
#: `agent`, `username` or `clone`. A stand-in holding a word a vocabulary assertion looks
#: for answers that assertion with the fixture rather than the product -- which is how the
#: first two versions of the vocabulary check below came to check nothing.
_A_FOLDER_ON_THIS_MACHINE = "<a folder on this machine>"
_A_NAME_CHOSEN_HERE = "<a name chosen here>"

#: The words this file judges a sentence by. The guards below read them from here; the
#: assertions spell them out one at a time, because each says something different about
#: the word it names. Adding a word here does not add an assertion for it.
_WORDS_UNDER_TEST = ("agent", "username", "clone")

#: The subset the wire rule forbids. A fixture name carrying one of these is taken out of
#: the sentence along with the product's own use of it, so the absence assertion would then
#: hold over text the leak had already been removed from. A name carrying `clone` is fine:
#: removing it can only make the presence assertion stricter, never vacuous.
_WORDS_THIS_FILE_FORBIDS = ("agent", "username")


def _without_what_the_fixture_supplied(sentence: str, root: Path, names: Iterable[str]) -> str:
    """`sentence` with the fixture's own root and clone names replaced by stand-ins.

    A path and a clone's name are the reader's own text, not copy this layer composed --
    rendering the wrong folder would be a different defect, and neither is a vocabulary
    choice -- so both are taken out before the words are judged. What is left is exactly
    the half `uclone_x.ui.clones` writes, which is the half the vocabulary rule binds.

    Longest name first, so a name that contains another is not half-replaced.
    """
    for stand_in in (_A_FOLDER_ON_THIS_MACHINE, _A_NAME_CHOSEN_HERE):
        assert not any(word in stand_in.lower() for word in _WORDS_UNDER_TEST), (
            f"a stand-in carrying a word under test answers the assertion itself: {stand_in}"
        )
    composed = sentence.replace(str(root), _A_FOLDER_ON_THIS_MACHINE)
    for name in sorted(names, key=len, reverse=True):
        assert not any(word in name.lower() for word in _WORDS_THIS_FILE_FORBIDS), (
            "a fixture name carrying a forbidden word takes the product's own out with it,"
            f" and the absence assertion then passes on nothing: {name}"
        )
        composed = composed.replace(name, _A_NAME_CHOSEN_HERE)
    return composed


def test_a_clone_that_has_never_run_is_listed_as_dormant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route this repository already had would answer nothing here.

    `GET /api/agents` lists live instances, and on a fresh install nothing has been
    spawned -- so a surface built on it renders an empty list over an install that has
    clones in it. The decision recorded in design §6.9 is `dormant`, not `present`:
    everything in this list is present by virtue of being listed, so `present` would
    encode nothing and the surface would be back to inferring.

    Killed by: src/uclone_x/ui/clones.py :: status=CloneStatus.DORMANT,
    Becomes: status=CloneStatus.LIVE,
    """
    agents_root = tmp_path / "agents"
    (agents_root / "scout").mkdir(parents=True)
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))

    body = _body(_client(tmp_path))

    assert [clone["name"] for clone in body["clones"]] == ["scout"]
    only = body["clones"][0]
    assert only["status"] == "dormant"
    assert only["id"] is None
    assert only["reason"], "a row that says nothing about its state is the defect"


def test_a_running_clone_is_reported_live_rather_than_left_to_be_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The surface draws a running clone differently, so the fact has to arrive typed.

    Killed by: src/uclone_x/ui/clones.py :: if entry.username in running:
    Becomes: if entry.username in frozenset():
    """
    agents_root = tmp_path / "agents"
    (agents_root / "scout").mkdir(parents=True)
    (agents_root / "scout" / "id").write_text(f"{AGENT_ID_PREFIX}abc\n", encoding="utf-8")
    (agents_root / "archivist").mkdir(parents=True)
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))

    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions")
    _seat_a_live_clone(session_mgr, "scout")

    body = _body(_client(tmp_path, session_mgr))

    by_name = {clone["name"]: clone for clone in body["clones"]}
    assert by_name["scout"]["status"] == "live"
    assert by_name["scout"]["id"] == f"{AGENT_ID_PREFIX}abc"
    assert by_name["archivist"]["status"] == "dormant"


def test_nothing_installed_and_an_unreadable_root_are_not_the_same_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both carry `clones: []`, and only one of them means "install a clone".

    The precedent is `AcpTab.tsx` and its `does not present absence as an empty session
    list` test: the presence block is rendered before the list and states its own reason.

    Killed by: src/uclone_x/ui/clones.py :: root_state=root_state,
    Becomes: root_state=CloneRootState.READABLE,
    """
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(empty_root))
    nothing_installed = _body(_client(tmp_path))

    locked_root = tmp_path / "locked"
    locked_root.mkdir()
    (locked_root / "scout").mkdir()
    os.chmod(locked_root, 0o000)
    try:
        if os.access(locked_root, os.R_OK):  # pragma: no cover - only for a privileged user
            pytest.skip("this user can read a mode-000 directory, so there is no fault")
        monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(locked_root))
        unreadable = _body(_client(tmp_path))
    finally:
        os.chmod(locked_root, 0o700)

    assert nothing_installed["clones"] == []
    assert unreadable["clones"] == []
    assert nothing_installed["root_state"] == "readable"
    assert unreadable["root_state"] == "unreadable"
    assert nothing_installed["reason"] != unreadable["reason"]
    assert str(locked_root) in unreadable["reason"]


def test_a_missing_root_is_its_own_answer_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped override and a root nothing has created yet look the same from here.

    So the reason names the directory it looked in and the variable that redirects it,
    which is the remedy for one of them and harmless for the other.

    Killed by: src/uclone_x/ui/clones.py :: f"there: {root}. {_CHECK_THE_OVERRIDE}"
    Becomes: f"there."
    """
    absent = tmp_path / "nowhere"
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(absent))

    body = _body(_client(tmp_path))

    assert body["clones"] == []
    assert body["root_state"] == "missing"
    assert str(absent) in body["reason"]
    assert AGENTS_DIR_ENV_VAR in body["reason"]


def test_one_unreadable_home_keeps_its_row_instead_of_emptying_the_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping the damaged row would render it identically to "not installed".

    It is the damaged home, not the absent one, that a reader can do something about --
    and the name is taken either way, so a row that vanishes is the one misleading answer.

    Killed by: src/uclone_x/ui/clones.py :: status=CloneStatus.UNREADABLE,
    Becomes: status=CloneStatus.DORMANT,
    """
    agents_root = tmp_path / "agents"
    (agents_root / "scout").mkdir(parents=True)
    (agents_root / "wrecked").mkdir(parents=True)
    (agents_root / "wrecked" / "id").write_text("  \n", encoding="utf-8")
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))

    body = _body(_client(tmp_path))

    by_name = {clone["name"]: clone for clone in body["clones"]}
    assert set(by_name) == {"scout", "wrecked"}
    assert body["root_state"] == "readable"
    assert by_name["scout"]["status"] == "dormant"
    assert by_name["wrecked"]["status"] == "unreadable"
    assert by_name["wrecked"]["id"] is None
    assert str(agents_root / "wrecked" / "id") in by_name["wrecked"]["reason"]


def test_a_running_clone_whose_home_is_gone_is_still_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone the user is talking to must not be missing from the list of clones.

    The home is minted on bring-up, so this is the case where it was removed underneath a
    running instance. Reporting only what is on disk would drop the row while the surface
    is rendering that clone's replies.

    Killed by: src/uclone_x/ui/clones.py :: for username in sorted(running - on_disk):
    Becomes: for username in sorted(frozenset()):
    """
    agents_root = tmp_path / "agents"
    agents_root.mkdir(parents=True)
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))

    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions")
    _seat_a_live_clone(session_mgr, "scout")

    body = _body(_client(tmp_path, session_mgr))

    assert [clone["name"] for clone in body["clones"]] == ["scout"]
    assert body["clones"][0]["status"] == "live"
    reason = body["clones"][0]["reason"]
    # Truthiness would pass on a row that says the opposite of every other field on it.
    assert "Running now" in reason, "the row that is running must not say it is not"
    assert "not running" not in reason
    assert str(agents_root) in reason, "the row has to say its files are not on disk"


def test_a_clone_answering_in_a_conversation_is_reported_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default way to be running is to be seated in a conversation, not in the chat map.

    `AgentSessionManager.list_agents()` returns only `_agents`, the legacy chat map. A
    clone seated in a conversation is built and cached by that room's `RoomAgentResolver`
    (`uclone_x.room.resolver`), and nothing writes it back into the session manager. Under
    D1 (design §1.3) *every* new conversation is a room, single-clone ones included, so a
    join that reads only the chat map calls the ordinary running clone "Installed and not
    running" while the user is reading the reply it just wrote.

    Killed by: src/uclone_x/ui/clones.py :: return chatting | room_stack.seated_agent_ids()
    Becomes: return chatting
    """
    agents_root = tmp_path / "agents"
    (agents_root / "scout").mkdir(parents=True)
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))

    app = create_ui_app(
        static_dir=tmp_path / "static",
        storage_dir=tmp_path / "sessions",
        llm=MockLLMConnector(),
    )
    # Context-managed: sending answers 202 and leaves the cascade running as a task, which
    # needs the client's shared event loop to still be open.
    with TestClient(app) as client:
        created = client.post("/api/rooms", json={"title": "Index tuning", "agent_ids": ["scout"]})
        assert created.status_code == 201, created.text
        room_id = created.json()["room_id"]

        sent = client.post(f"/api/rooms/{room_id}/messages", json={"content": "are you there?"})
        assert sent.status_code == 202, sent.text
        # Two joins, the message, then the reply: the turn has run.
        _wait_for_transcript(client, room_id, rows=4)

        body = _body(client)

    assert [clone["name"] for clone in body["clones"]] == ["scout"]
    only = body["clones"][0]
    assert only["status"] == "live", "the clone that just answered is the one called dormant"
    assert "Running now" in only["reason"]


def test_a_deleted_conversation_stops_reporting_its_clone_as_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversation that no longer exists must not leave a clone reported as running.

    `_RoomRuntime` was introduced so that `RoomStack.forget` cannot drop a deleted room's
    orchestrator and leave its resolver behind: `seated_agent_ids` unions over the live
    rooms, so a runtime outliving its room keeps answering "this clone is seated" with
    nothing left that could disagree. That was the whole stated justification for the
    refactor and no test held it -- measured with the drop removed, `GET /api/rooms`
    returns no conversations and `GET /api/clones` still says `live`, "Running now."

    Killed by: src/uclone_x/ui/rooms.py :: self._rooms.pop(room_id, None)
    Becomes: self._rooms.get(room_id)
    """
    agents_root = tmp_path / "agents"
    (agents_root / "scout").mkdir(parents=True)
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))

    app = create_ui_app(
        static_dir=tmp_path / "static",
        storage_dir=tmp_path / "sessions",
        llm=MockLLMConnector(),
    )
    # Context-managed for the same reason as the conversation test above: sending answers
    # 202 and leaves the cascade running on the client's shared event loop.
    with TestClient(app) as client:
        created = client.post("/api/rooms", json={"title": "Index tuning", "agent_ids": ["scout"]})
        assert created.status_code == 201, created.text
        room_id = created.json()["room_id"]

        sent = client.post(f"/api/rooms/{room_id}/messages", json={"content": "are you there?"})
        assert sent.status_code == 202, sent.text
        _wait_for_transcript(client, room_id, rows=4)
        # The seat has to exist before its removal can mean anything.
        assert _reason_of(_body(client), "scout").startswith("Running now")

        deleted = client.delete(f"/api/rooms/{room_id}")
        assert deleted.status_code == 204, deleted.text
        assert client.get("/api/rooms").json()["rooms"] == []

        body = _body(client)

    assert [clone["name"] for clone in body["clones"]] == ["scout"]
    only = body["clones"][0]
    assert only["status"] == "dormant", (
        "the deleted conversation's resolver is still reporting its clone as running"
    )
    assert "Running now" not in only["reason"]


def test_two_unusable_names_say_which_rule_each_one_broke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Scout/` and `my clone/` are refused for two different reasons, and say so.

    The Core computed the answer -- `refuse_an_unusable_username` names the offending
    characters and the allowed set, and `_describe_agent_home` records it on the entry --
    and the row is where a reader meets it. Rendering both as "that is not a name a clone
    can have" tells someone their folder is wrong and leaves them no way to derive the
    fix, which is an absence that states no cause (P6).

    The root mirrors the product default, `~/.uclone/agents`. An earlier version put the
    fixture somewhere that happened not to be called `agents`, so that the vocabulary
    assertions at the end would pass -- which is the dodge this file removes 200 lines
    below, and it cannot be both filed as a defect and recommended here. The path is
    instead taken out of the sentence before the words are judged, by the one helper both
    tests share.

    Killed by: src/uclone_x/ui/clones.py :: {entry.cause} Its folder is
    Becomes: Its folder is
    """
    clones_root = tmp_path / ".uclone" / "agents"
    (clones_root / "Scout").mkdir(parents=True)
    (clones_root / "my clone").mkdir(parents=True)
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(clones_root))

    body = _body(_client(tmp_path))

    by_name = {clone["name"]: clone for clone in body["clones"]}
    assert set(by_name) == {"Scout", "my clone"}
    assert by_name["Scout"]["status"] == "unreadable"
    assert by_name["my clone"]["status"] == "unreadable"

    uppercase = _reason_of(body, "Scout")
    spaced = _reason_of(body, "my clone")
    assert uppercase != spaced, "two different rules rendered as one sentence"
    assert "'S'" in uppercase, "the row never says which character is refused"
    assert "' '" in spaced, "the row never says which character is refused"
    assert "lowercase letters" in uppercase, "the row states no rule the reader could meet"
    # What crosses is the rule, not the Core's sentence about it (design §3.1.1).
    for sentence in (uppercase, spaced):
        composed = _without_what_the_fixture_supplied(sentence, clones_root, by_name)
        assert "agent" not in composed.lower(), f"the Core's word reached the wire: {composed}"
        assert "username" not in composed.lower(), f"the Core's word reached the wire: {composed}"


def test_a_clone_that_has_never_run_says_so_rather_than_only_that_it_is_not_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`id: null` is the ordinary state of something nobody has started, not damage.

    A row saying only "installed and not running" leaves the empty id looking like the
    damaged row below it, which is the one thing it is not. The two dormant sentences
    therefore differ, and this asserts the difference rather than the truthiness
    `test_a_clone_that_has_never_run_is_listed_as_dormant` settled for.

    Killed by: src/uclone_x/ui/clones.py :: if entry.agent_id is None:
    Becomes: if False:
    """
    agents_root = tmp_path / "agents"
    (agents_root / "archivist").mkdir(parents=True)
    (agents_root / "scout").mkdir(parents=True)
    (agents_root / "scout" / "id").write_text(f"{AGENT_ID_PREFIX}abc\n", encoding="utf-8")
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))

    body = _body(_client(tmp_path))

    never_run = _reason_of(body, "archivist")
    has_run = _reason_of(body, "scout")

    assert never_run != has_run, "a clone that has never run says what one that has says"
    assert "has never been started" in never_run
    assert "has never been started" not in has_run
    assert "Installed and not running" in has_run


def test_the_empty_root_sentence_names_the_folder_it_looked_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This is the first screen a new user sees, so it says where to put a clone.

    "No clone is installed yet." on its own is a dead end: the folder is configurable,
    and a reader with no path in front of them cannot tell an empty installation from one
    that is looking in the wrong place. `test_nothing_installed_and_an_unreadable_root_are_
    not_the_same_answer` asserts only that this sentence differs from the unreadable-root
    one, which stays true however much of it is deleted.

    Killed by: src/uclone_x/ui/clones.py :: f"No clone is installed yet. Clones live in {root}, and it is empty."
    Becomes: f"No clone is installed yet."
    """
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(empty_root))

    body = _body(_client(tmp_path))

    assert body["clones"] == []
    assert body["root_state"] == "readable"
    assert str(empty_root) in body["reason"], "the first screen does not say where it looked"
    assert "is empty" in body["reason"], (
        "the reason does not say the folder was read and held nothing"
    )


def test_the_sentence_above_the_list_counts_the_rows_the_list_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row and the sentence rendered above it must not contradict each other.

    `CloneListing.reason` is rendered before the list and never collapsed into it, so a
    count taken from the *disk* listing while the list also carries running clones with no
    home puts the two predicates in contradiction -- and this is the exact state the
    running-clone-without-a-home loop was added for. #1088 reintroduced; §3.2.6 states the
    requirement: one predicate, not two. `len(clones)` is the count.

    Killed by: src/uclone_x/ui/clones.py :: _root_reason(root_state, listing.root, listing.cause, len(clones))
    Becomes: _root_reason(root_state, listing.root, listing.cause, len(listing.homes))
    """
    agents_root = tmp_path / "agents"
    agents_root.mkdir(parents=True)
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))

    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions")
    _seat_a_live_clone(session_mgr, "scout")

    body = _body(_client(tmp_path, session_mgr))

    assert [clone["name"] for clone in body["clones"]] == ["scout"]
    assert body["root_state"] == "readable"
    assert "No clone is installed" not in body["reason"], (
        "the sentence rendered above the list contradicted the row directly below it"
    )
    assert "One clone" in body["reason"]


def test_a_missing_or_unreadable_root_counts_the_rows_it_holds_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same contradiction, reached through the other two root states.

    A root that is not there, or cannot be read, still carries the running clones that
    have no home -- so the sentence above the list has to reconcile with them rather than
    announce an emptiness the list disproves.

    Killed by: src/uclone_x/ui/clones.py :: counted = "One clone" if count == 1 else f"{count} clones"
    Becomes: counted = "No clone"
    """
    absent = tmp_path / "nowhere"
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(absent))
    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions")
    _seat_a_live_clone(session_mgr, "scout")

    missing = _body(_client(tmp_path, session_mgr))

    assert [clone["name"] for clone in missing["clones"]] == ["scout"]
    assert missing["root_state"] == "missing"
    assert "One clone" in missing["reason"]
    assert str(absent) in missing["reason"]

    locked_root = tmp_path / "locked"
    locked_root.mkdir()
    os.chmod(locked_root, 0o000)
    try:
        if os.access(locked_root, os.R_OK):  # pragma: no cover - only for a privileged user
            pytest.skip("this user can read a mode-000 directory, so there is no fault")
        monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(locked_root))
        unreadable = _body(_client(tmp_path, session_mgr))
    finally:
        os.chmod(locked_root, 0o700)

    assert [clone["name"] for clone in unreadable["clones"]] == ["scout"]
    assert unreadable["root_state"] == "unreadable"
    assert "One clone" in unreadable["reason"]
    # The list may be short, which is the one thing an unreadable root adds here.
    assert "more may be installed" in unreadable["reason"]


def test_each_row_says_its_own_state_rather_than_repeating_one_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The change's stated purpose is that three absences do not render as one.

    That lands in the per-row `reason`, and only the *root* reasons were ever checked for
    differing. A dormant row that claims to be running, or a running row told it is not,
    is exactly what this endpoint exists to stop -- so the rows are asserted on their own
    rendered content rather than on truthiness.

    Killed by: src/uclone_x/ui/clones.py :: f"Installed and not running. Its files are in {entry.path}."
    Becomes: f"Running now. Its files are in {entry.path}."
    """
    agents_root = tmp_path / "agents"
    (agents_root / "scout").mkdir(parents=True)
    (agents_root / "scout" / "id").write_text(f"{AGENT_ID_PREFIX}abc\n", encoding="utf-8")
    (agents_root / "archivist").mkdir(parents=True)
    (agents_root / "archivist" / "id").write_text(f"{AGENT_ID_PREFIX}def\n", encoding="utf-8")
    (agents_root / "wrecked").mkdir(parents=True)
    (agents_root / "wrecked" / "id").write_text("  \n", encoding="utf-8")
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))

    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions")
    _seat_a_live_clone(session_mgr, "scout")

    body = _body(_client(tmp_path, session_mgr))

    live = _reason_of(body, "scout")
    dormant = _reason_of(body, "archivist")
    damaged = _reason_of(body, "wrecked")

    assert len({live, dormant, damaged}) == 3, "three states rendered as fewer than three"
    assert "Running now" in live
    assert "Installed and not running" in dormant
    assert "Running now" not in dormant, "a clone that is not running must not say it is"
    assert "cannot be started" in damaged


def test_no_reason_carries_the_cores_vocabulary_onto_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One object, one word for it. Design §3.1.1 puts that boundary at the wire.

    `CloneSummary.reason` and `CloneListing.reason` are rendered verbatim, so concatenating
    the Core's own sentence makes a U0 read one sentence that says `clone` in the head's
    half and `agent`, `agent home` and `id file` in the Core's -- the confusion the
    boundary exists to end, in the module whose docstring claims to have ended it.

    **The root mirrors the product default, and everything the fixture supplied is taken
    out before the words are judged.** `DEFAULT_AGENTS_ROOT` is `~/.uclone/agents` and
    paths are rendered verbatim, so every sentence a real installation produces carries
    `agent` in its path; a test rooted anywhere else passes on its fixture rather than on
    the product. The path and the clone names are therefore replaced by stand-ins that
    carry none of the words under test -- the substitution twice removed the word this
    test then looked for, so `_without_what_the_fixture_supplied` refuses a stand-in that
    could answer an assertion on the fixture's behalf, and refuses a fixture *name* that
    would take a forbidden word out of the sentence along with the product's own use of
    it. A name carrying `clone` is allowed: removing it can only make the presence
    assertion stricter.

    **Where the product's own noun is required, and where it is not.** A live or dormant
    row says what is true of one clone without naming its kind (`Running now. Its files
    are in ...`); the sentence above the list names it for them. Requiring `clone` in all
    five sentences asserts something the product does not do, and the only reason the
    earlier version appeared to pass was that its stand-in supplied the word. What is
    required instead is that the sentence above the list names what it lists, and that a
    row a reader has to act on *opens* by saying what cannot be started -- the opening,
    because `clone` appearing anywhere in a four-sentence damage row is satisfied by a
    clause the rewording never touched.

    Killed by: src/uclone_x/ui/clones.py :: Installed, and has never been started -- it is given an identity the first
    Becomes: This agent home holds no id file yet.
    Killed by: src/uclone_x/ui/clones.py :: f"This clone cannot be started: the identity recorded at {entry.id_path} is "
    Becomes: f"This thing cannot be started: the identity recorded at {entry.id_path} is "
    Killed by: src/uclone_x/ui/clones.py :: f"This clone cannot be started: {entry.username!r} is not a name a clone can "
    Becomes: f"This thing cannot be started. {entry.username!r} is not a name a clone can "
    """
    clones_root = tmp_path / ".uclone" / "agents"
    (clones_root / "scout").mkdir(parents=True)
    (clones_root / "scout" / "id").write_text(f"{AGENT_ID_PREFIX}abc\n", encoding="utf-8")
    (clones_root / "archivist").mkdir(parents=True)
    (clones_root / "wrecked").mkdir(parents=True)
    (clones_root / "wrecked" / "id").write_text("  \n", encoding="utf-8")
    # An unusable name, so that the explanation the Core computes for one is judged here
    # too. `my clone` rather than `Scout`: this machine's filesystem is case-insensitive,
    # so `Scout/` and the `scout/` above would be one directory -- which is the rule the
    # name check exists for, arriving as a fixture collision.
    (clones_root / "my clone").mkdir(parents=True)
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(clones_root))
    assert "agents" in str(clones_root), (
        "the fixture no longer mirrors the product default, which is the point of it"
    )

    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions")
    _seat_a_live_clone(session_mgr, "scout")

    body = _body(_client(tmp_path, session_mgr))

    installed = {clone["name"] for clone in body["clones"]}
    assert installed == {"scout", "archivist", "wrecked", "my clone"}

    rendered = [body["reason"], *(clone["reason"] for clone in body["clones"])]
    assert len(rendered) == 5
    for sentence in rendered:
        assert str(clones_root) in sentence, (
            f"a sentence with no path in it cannot have one excluded: {sentence}"
        )
        composed = _without_what_the_fixture_supplied(sentence, clones_root, installed)
        assert "agent" not in composed.lower(), f"the Core's word reached the wire: {composed}"
        assert "username" not in composed.lower(), f"the Core's word reached the wire: {composed}"

    above_the_list = _without_what_the_fixture_supplied(body["reason"], clones_root, installed)
    assert "clone" in above_the_list.lower(), (
        f"the sentence above the list never names what it is a list of: {above_the_list}"
    )
    damaged = [clone for clone in body["clones"] if clone["status"] == "unreadable"]
    assert len(damaged) == 2, "the fixture no longer produces the damaged rows this judges"
    for clone in damaged:
        composed = _without_what_the_fixture_supplied(clone["reason"], clones_root, installed)
        # Without a colon `split` returns the whole sentence, and this would widen back to
        # `"clone" in composed` -- which the row satisfies from a later clause either way.
        assert ":" in composed, f"a damaged row no longer opens with a clause: {composed}"
        opening = composed.split(":", 1)[0]
        # A colon anywhere else in the row can stand in for the one that ended the opening
        # clause: the name fault carries the Core's `Allowed: ...` explanation, so rewriting
        # the real colon leaves `split` to find *that* one and hand back three sentences,
        # which say `clone` in a clause the rewording never touched. The opening is one
        # clause, so a sentence boundary inside it means this is not the opening.
        assert "." not in opening, (
            f"the opening clause runs past a sentence boundary, so a later clause is "
            f"answering for it: {composed}"
        )
        assert "clone" in opening.lower(), (
            f"a damaged row never says what it is that cannot be started: {composed}"
        )
    # `1 clone(s) installed` is not copy anyone would write.
    assert "(s)" not in body["reason"]
