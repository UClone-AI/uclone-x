"""One directory per agent: the name is refused, not repaired, and the id is written once."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from unittest import mock

import pytest

from uclone_x.core.agent_home import (
    AGENT_ID_PREFIX,
    AGENTS_DIR_ENV_VAR,
    DEFAULT_AGENTS_ROOT,
    AgentHome,
    AgentHomeError,
    AgentHomeFault,
    AgentHomeRootState,
    AgentHomeState,
    default_agents_root,
    list_agent_homes,
    refuse_an_unusable_username,
)


def test_a_home_is_located_without_creating_anything(tmp_path: Path) -> None:
    """Asking where an agent's state would live must not bring that agent into being.

    Callers that only want a path -- a status report, a diagnostic, a dry run -- would
    otherwise leave a trail of empty homes for agents that were never started.

    Killed by: src/uclone_x/core/agent_home.py :: return self.path / "memory.json"
    Becomes: return self.path / "cross_session.json"
    """
    home = AgentHome.for_username("scout", root=tmp_path)

    assert home.path == tmp_path / "scout"
    assert home.id_path == tmp_path / "scout" / "id"
    assert home.memory_path == tmp_path / "scout" / "memory.json"
    assert not home.path.exists()


def test_names_that_used_to_collide_are_each_refused(tmp_path: Path) -> None:
    """`a.b`, `a/b` and `a b` no longer resolve onto `a_b`.

    The old derivation folded every character it disliked into `_`, so those three names
    and `a_b` addressed one file. Silently merging four agents' memories is exactly the
    substitution P6 forbids, so each unusable name raises against itself.

    Killed by: src/uclone_x/core/agent_home.py :: if not _USERNAME_RE.match(username):
    Becomes: if False:
    """
    for rejected in ("a.b", "a/b", "a b", "../escape", "Scout", "-lead", "trail_", ""):
        with pytest.raises(AgentHomeError):
            AgentHome.for_username(rejected, root=tmp_path)

    assert AgentHome.for_username("a_b", root=tmp_path).path == tmp_path / "a_b"


def test_a_username_longer_than_a_directory_name_is_refused(tmp_path: Path) -> None:
    """The limit is checked before the pattern, so the message says which rule failed.

    Killed by: src/uclone_x/core/agent_home.py :: if len(username) > _MAX_USERNAME_LENGTH:
    Becomes: if False:
    """
    with pytest.raises(AgentHomeError) as excinfo:
        refuse_an_unusable_username("a" * 65)

    assert "65 characters" in str(excinfo.value)
    refuse_an_unusable_username("a" * 64)


def test_the_id_is_minted_once_and_read_thereafter(tmp_path: Path) -> None:
    """An agent whose identifier changes is a different agent to every record of it.

    The reported id is the one on disk, read back after publishing rather than assumed
    from the candidate: on the losing side of a mint race the candidate is not the agent's
    id, and a caller told otherwise would record a value nothing else will ever agree with.

    Killed by: src/uclone_x/core/agent_home.py :: settled = self._read_id_if_present()
    Becomes: settled = f"{AGENT_ID_PREFIX}{uuid.uuid4().hex}"
    """
    home = AgentHome.for_username("archivist", root=tmp_path)

    first = home.agent_id()
    assert first.startswith(AGENT_ID_PREFIX)
    assert home.id_path.read_text(encoding="utf-8").strip() == first

    assert home.agent_id() == first
    assert AgentHome.for_username("archivist", root=tmp_path).agent_id() == first


def test_two_agents_do_not_share_an_identifier(tmp_path: Path) -> None:
    """Each home mints its own value.

    Killed by: src/uclone_x/core/agent_home.py :: self._publish_id(f"{AGENT_ID_PREFIX}{uuid.uuid4().hex}")
    Becomes: self._publish_id(f"{AGENT_ID_PREFIX}fixed")
    """
    one = AgentHome.for_username("one", root=tmp_path).agent_id()
    two = AgentHome.for_username("two", root=tmp_path).agent_id()

    assert one != two


def test_an_empty_id_file_is_refused_rather_than_replaced(tmp_path: Path) -> None:
    """A home with a blank id is damage, and minting over it would hide the damage.

    Whatever emptied the file may be a crash mid-write or a bad restore; either way the
    agent's recorded identity is gone, and quietly issuing a new one would make every
    system still holding the old id wrong with no error anywhere (P6).

    Killed by: src/uclone_x/core/agent_home.py :: if not recorded:
    Becomes: if False:
    """
    home = AgentHome.for_username("truncated", root=tmp_path)
    home.path.mkdir(parents=True)
    home.id_path.write_text("  \n", encoding="utf-8")

    with pytest.raises(AgentHomeError) as excinfo:
        home.agent_id()

    assert "truncated" in str(excinfo.value)


def test_the_root_defaults_under_the_home_directory_without_an_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no `UCLONE_AGENTS_DIR`, agents live under the invoking user's home.

    Killed by: src/uclone_x/core/agent_home.py :: return DEFAULT_AGENTS_ROOT
    Becomes: return Path("agents")
    """
    monkeypatch.delenv(AGENTS_DIR_ENV_VAR, raising=False)

    assert default_agents_root() == DEFAULT_AGENTS_ROOT
    assert DEFAULT_AGENTS_ROOT.is_absolute()


def test_an_override_is_expanded_before_it_becomes_a_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`~/elsewhere` has to become a real path, not a directory literally named `~`.

    Killed by: src/uclone_x/core/agent_home.py :: return Path(override).expanduser()
    Becomes: return Path(override)
    """
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, "~/uclone-agents-test")

    assert default_agents_root() == Path.home() / "uclone-agents-test"


def test_a_trailing_newline_does_not_smuggle_a_second_home_past_the_rule(tmp_path: Path) -> None:
    r"""`scout\n` is not a second spelling of `scout`; it is not a username at all.

    Python's `$` matches immediately before a trailing newline, so the first version of
    this rule accepted `"scout\n"` and minted it a directory of its own. Two homes, two
    ids, and one rendering in every listing and log -- the ambiguity the whole rule exists
    to prevent, reachable from any caller that does not strip its input.

    Killed by: src/uclone_x/core/agent_home.py :: _USERNAME_RE = re.compile(r"\A[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?\Z")
    Becomes: _USERNAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
    """
    for smuggled in ("scout\n", "scout\r\n", "\nscout"):
        with pytest.raises(AgentHomeError):
            AgentHome.for_username(smuggled, root=tmp_path)


def test_a_windows_device_name_is_refused_on_every_platform(tmp_path: Path) -> None:
    """`con` names a device on Windows, so it names no directory there.

    The rule's justification is that one username means one directory on every machine.
    A name that resolves to a device on one of them fails that on its own terms, and
    refusing it only on Windows would make the roster platform-dependent.

    Killed by: src/uclone_x/core/agent_home.py :: if username in _RESERVED_DEVICE_NAMES:
    Becomes: if False:
    """
    for reserved in ("con", "nul", "aux", "prn", "com1", "lpt9"):
        with pytest.raises(AgentHomeError) as excinfo:
            AgentHome.for_username(reserved, root=tmp_path)
        assert "reserved device name" in str(excinfo.value)

    AgentHome.for_username("console", root=tmp_path)
    AgentHome.for_username("com10", root=tmp_path)


def test_the_id_becomes_visible_complete_or_not_at_all(tmp_path: Path) -> None:
    """No reader can observe an `id` file that exists and is empty.

    `O_CREAT | O_EXCL` published the name before the bytes, so a second process starting
    at the same moment read an empty file -- and an empty `id` is refused and never
    repaired, so that race turned a benign concurrent startup into a permanent "this home
    is damaged". Staging then linking makes the name appear only once the bytes are there.

    Killed by: src/uclone_x/core/agent_home.py :: os.link(staged, self.id_path)
    Becomes: self.id_path.touch()
    """
    home = AgentHome.for_username("stager", root=tmp_path)
    observed: list[str] = []

    real_link = os.link

    def watching_link(source: str, target: str) -> None:
        # Stand where a concurrent reader stands: after the id file has a name.
        real_link(source, target)
        observed.append(home.id_path.read_text(encoding="utf-8"))

    with mock.patch.object(os, "link", watching_link):
        minted = home.agent_id()

    assert observed == [minted + "\n"]
    assert sorted(p.name for p in home.path.iterdir()) == ["id"]


def test_the_loser_of_a_mint_race_reports_the_winners_id(tmp_path: Path) -> None:
    """Whoever links first owns the id; the other discards its candidate and reads.

    Killed by: src/uclone_x/core/agent_home.py :: except FileExistsError:
    Becomes: except ValueError:
    """
    home = AgentHome.for_username("contended", root=tmp_path)
    winner = f"{AGENT_ID_PREFIX}winner"

    def losing_link(source: str, target: str) -> None:
        Path(target).write_text(winner + "\n", encoding="utf-8")
        raise FileExistsError(target)

    with mock.patch.object(os, "link", losing_link):
        assert home.agent_id() == winner

    assert sorted(p.name for p in home.path.iterdir()) == ["id"]


def test_a_regular_file_where_a_home_belongs_is_an_agent_home_fault(tmp_path: Path) -> None:
    """A bare `OSError` from here is reported by the UI as a session fault.

    `_translate_session_error` names `AgentHomeError`; an `OSError` escaping `mkdir` falls
    past it to a 500 labelled "Session operation failed" with
    `component=uclone_x.agent.session` -- the mis-attribution this module exists to stop,
    arriving through a different door (P6).

    Killed by: src/uclone_x/core/agent_home.py :: f"agent {self.username!r} has no usable home at {self.path}: {exc}"
    Becomes: f"agent has no usable home"
    """
    (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")

    with pytest.raises(AgentHomeError) as excinfo:
        AgentHome.for_username("blocked", root=tmp_path).agent_id()

    message = str(excinfo.value)
    assert "blocked" in message and "no usable home" in message
    assert str(tmp_path / "blocked") in message, "the reader has to be told which path"


def test_a_root_that_cannot_hold_a_link_is_an_agent_home_fault(tmp_path: Path) -> None:
    """`UCLONE_AGENTS_DIR` on exFAT, FAT32 or a network mount has no hard links.

    `os.link` fails there with `EPERM` or `ENOTSUP`, which is neither the race
    `FileExistsError` nor a session fault. Left bare it reaches the UI as a 500 naming the
    session store, which holds nothing to fix.

    Killed by: src/uclone_x/core/agent_home.py :: except OSError as link_failure:
    Becomes: except InterruptedError as link_failure:
    """
    home = AgentHome.for_username("nolinks", root=tmp_path)

    def unsupported_link(source: str, target: str) -> None:
        raise OSError(errno.EPERM, "Operation not permitted", target)

    with mock.patch.object(os, "link", unsupported_link):
        with pytest.raises(AgentHomeError) as excinfo:
            home.agent_id()

    assert "no identity" in str(excinfo.value)
    assert not list(home.path.iterdir()), "a failed publish leaves no stage behind"


def test_an_empty_agents_root_is_reported_as_such_and_not_as_a_bare_empty_list(
    tmp_path: Path,
) -> None:
    """ "Nothing is installed" is a fact, and it has to be stated rather than implied.

    An empty tuple is the same value three different situations produce -- nothing
    installed, a root that cannot be read, and a root that is not there -- so the tuple
    alone decides nothing for the caller that has to draw it (P6). `root_state` is what
    decides it, and it is a value rather than a sentence so that the head states the fact
    in the product's own vocabulary (design §3.1.1).

    Killed by: src/uclone_x/core/agent_home.py :: root_state=AgentHomeRootState.READABLE
    Becomes: root_state=AgentHomeRootState.MISSING
    """
    listing = list_agent_homes(root=tmp_path)

    assert listing.root == tmp_path
    assert listing.root_state is AgentHomeRootState.READABLE
    assert listing.homes == ()
    assert listing.cause == "", "nothing failed, so there is no error to quote"


def test_a_missing_agents_root_is_not_the_same_answer_as_an_empty_one(tmp_path: Path) -> None:
    """A root that is not there is a different fact from a root holding nothing.

    It is also the shape a mistyped `UCLONE_AGENTS_DIR` takes, which is why the listing
    carries the directory it looked in -- the head names it, and names the variable, in
    the sentence it renders.

    Killed by: src/uclone_x/core/agent_home.py :: root_state=AgentHomeRootState.MISSING
    Becomes: root_state=AgentHomeRootState.READABLE
    """
    absent = tmp_path / "nowhere"

    listing = list_agent_homes(root=absent)

    assert listing.root_state is AgentHomeRootState.MISSING
    assert listing.homes == ()
    assert listing.root == absent, "the directory it looked in, for the reader to check"
    assert listing.cause == "", "ENOENT's text adds nothing the state and the path do not"


def test_an_unreadable_agents_root_is_not_reported_as_nothing_installed(tmp_path: Path) -> None:
    """A permission fault must not read as "you have no agents".

    The two render identically as an empty list, and only one of them is something the
    reader can act on.

    Killed by: src/uclone_x/core/agent_home.py :: root_state=AgentHomeRootState.UNREADABLE
    Becomes: root_state=AgentHomeRootState.MISSING
    """
    root = tmp_path / "locked"
    root.mkdir()
    (root / "scout").mkdir()
    os.chmod(root, 0o000)
    try:
        if os.access(root, os.R_OK):  # pragma: no cover - only true for a privileged user
            pytest.skip("this user can read a mode-000 directory, so there is no fault to report")
        listing = list_agent_homes(root=root)
    finally:
        os.chmod(root, 0o700)

    assert listing.root_state is AgentHomeRootState.UNREADABLE
    assert listing.homes == ()
    assert listing.root == root
    assert listing.cause, "the operating system's own words, which the head may show as-is"


def test_a_home_that_has_never_run_is_listed_with_no_identity(tmp_path: Path) -> None:
    """An installed agent that has never been brought up is still installed.

    Its `id` is minted on first use, so "no id file" is the ordinary state of a home
    nothing has run yet -- not damage. That is the distinction `fault` carries: a home
    reported with a fault is one a head tells the reader to repair, and telling someone to
    repair a clone they have simply never started is the mis-attribution P6 forbids.

    Killed by: src/uclone_x/core/agent_home.py :: fault=None,
    Becomes: fault=AgentHomeFault.EMPTY_ID,
    """
    (tmp_path / "scout").mkdir()

    listing = list_agent_homes(root=tmp_path)

    assert [home.username for home in listing.homes] == ["scout"]
    only = listing.homes[0]
    assert only.state is AgentHomeState.INSTALLED
    assert only.agent_id is None
    assert only.path == tmp_path / "scout"
    assert only.fault is None, "never started is not damage"
    assert only.cause == ""


def test_one_unreadable_home_does_not_empty_the_listing(tmp_path: Path) -> None:
    """A single damaged home is a property of that row, not of the list.

    An `id` file that exists and holds nothing is refused rather than repaired
    (`_read_id_if_present`), and a listing that let that refusal escape would report one
    damaged directory as though no agent were installed at all.

    Killed by: src/uclone_x/core/agent_home.py :: except AgentHomeError as damage:
    Becomes: except FileNotFoundError as damage:
    """
    (tmp_path / "scout").mkdir()
    (tmp_path / "scout" / "id").write_text(f"{AGENT_ID_PREFIX}abc\n", encoding="utf-8")
    (tmp_path / "wrecked").mkdir()
    (tmp_path / "wrecked" / "id").write_text("   \n", encoding="utf-8")

    listing = list_agent_homes(root=tmp_path)

    by_name = {home.username: home for home in listing.homes}
    assert set(by_name) == {"scout", "wrecked"}
    assert by_name["scout"].state is AgentHomeState.INSTALLED
    assert by_name["scout"].agent_id == f"{AGENT_ID_PREFIX}abc"
    assert by_name["wrecked"].state is AgentHomeState.UNREADABLE
    assert by_name["wrecked"].agent_id is None
    assert by_name["wrecked"].fault is AgentHomeFault.EMPTY_ID
    assert by_name["wrecked"].id_path == tmp_path / "wrecked" / "id"


def test_an_id_that_cannot_be_read_at_all_is_a_different_fault_from_an_empty_one(
    tmp_path: Path,
) -> None:
    """An `id` that cannot be opened is not an `id` that is blank.

    `recorded_agent_id` raises `AgentHomeError` only for the file it read and found
    empty. Everything else the filesystem refuses -- a permission, a directory standing
    where the file should be, a device error -- arrives as a bare `OSError`, and an
    enumeration that caught only the first would let it escape and empty the whole
    listing for one damaged home. The two also have different remedies, so they must not
    arrive as one fault: a blank id is restored from a backup, an unopenable one is very
    often just a mode.

    A directory at `id` rather than a `chmod` because a privileged user can read a
    mode-000 file, and a test that skips for root proves nothing on the machine it
    skipped on.

    Killed by: src/uclone_x/core/agent_home.py :: fault=AgentHomeFault.UNREADABLE_ID,
    Becomes: fault=AgentHomeFault.EMPTY_ID,
    """
    (tmp_path / "scout").mkdir()
    (tmp_path / "scout" / "id").mkdir()

    listing = list_agent_homes(root=tmp_path)

    assert [home.username for home in listing.homes] == ["scout"]
    only = listing.homes[0]
    assert only.state is AgentHomeState.UNREADABLE
    assert only.agent_id is None
    assert only.fault is AgentHomeFault.UNREADABLE_ID
    assert only.cause, "the operating system said why, and the reader needs it"


@pytest.mark.parametrize(
    ("refused", "rule_stated"),
    [
        pytest.param("my clone", "' '", id="character-class"),
        pytest.param("con", "reserved device name", id="reserved-device-name"),
        pytest.param("a" * 100, "100 characters long", id="over-length"),
        pytest.param("", "has to be called something", id="empty-name"),
    ],
)
def test_every_refused_name_hands_a_head_the_rule_it_broke(refused: str, rule_stated: str) -> None:
    """Each of the four refusals states its own rule in words a head may render as they are.

    `AgentHomeEntry.cause` is `AgentHomeError.explanation` verbatim, and a head drops it
    into a sentence unchanged: `... so nothing can be installed under it. {cause} Its
    folder is ...`. An explanation left empty therefore renders as a double space where
    the rule should be -- the absence stating no cause that P6 forbids, and the reason the
    field exists. Only one of the four was pinned; `con` and a 100-character name are both
    directory names a filesystem accepts, so both reach a reader.

    The empty name cannot be a directory name, so `list_agent_homes` never reaches it; it
    is reached here through `refuse_an_unusable_username` directly, which is also how
    `AgentHome.for_username("")` reaches it, so the branch is asserted rather than excused.

    `agent` and `username` are this module's nouns and may not cross the wire beside the
    head's word for the same object (design §3.1.1). That boundary was being enforced for
    one branch in four.

    Killed by: src/uclone_x/core/agent_home.py :: explanation=f"{detail} {_ALLOWED_IN_A_NAME}",
    Becomes: explanation="",
    Killed by: src/uclone_x/core/agent_home.py :: f"{username!r} is a reserved device name on Windows, where it names a "
    Becomes: f""
    Killed by: src/uclone_x/core/agent_home.py :: f"It is {len(username)} characters long, and the limit is {_MAX_USERNAME_LENGTH}."
    Becomes: f""
    Killed by: src/uclone_x/core/agent_home.py :: explanation="The name is empty, and a folder has to be called something.",
    Becomes: explanation="",
    """
    with pytest.raises(AgentHomeError) as excinfo:
        refuse_an_unusable_username(refused)

    explanation = excinfo.value.explanation
    assert explanation, "a head renders this verbatim, so empty is a blank where a rule goes"
    assert rule_stated in explanation, (
        f"the refusal never says which rule was broken: {explanation!r}"
    )
    assert "agent" not in explanation.lower(), (
        f"the Core's word would reach the wire: {explanation}"
    )
    assert "username" not in explanation.lower(), (
        f"the Core's word would reach the wire: {explanation}"
    )


def test_a_directory_no_username_rule_allows_is_reported_rather_than_skipped(
    tmp_path: Path,
) -> None:
    """`Scout/` occupies the name `scout` on a case-insensitive filesystem.

    Dropping it silently would show the reader a root with nothing in it while every
    attempt to use that name failed, which is the substitution `refuse_an_unusable_username`
    exists to stop, moved one layer out.

    Killed by: src/uclone_x/core/agent_home.py :: except AgentHomeError as unusable:
    Becomes: except ValueError as unusable:
    """
    (tmp_path / "Scout").mkdir()

    listing = list_agent_homes(root=tmp_path)

    assert [home.username for home in listing.homes] == ["Scout"]
    assert listing.homes[0].state is AgentHomeState.UNREADABLE
    assert listing.homes[0].fault is AgentHomeFault.UNUSABLE_NAME


def test_a_file_where_a_home_should_be_is_reported_as_damage(tmp_path: Path) -> None:
    """`<root>/scout` as a regular file is the `FileExistsError` `agent_id()` documents.

    It is not an absent agent: the name is taken, and every bring-up under it fails.

    Killed by: src/uclone_x/core/agent_home.py :: if not path.is_dir():
    Becomes: if False:  # the id read then reports a bare OSError instead
    """
    (tmp_path / "scout").write_text("not a directory", encoding="utf-8")

    listing = list_agent_homes(root=tmp_path)

    assert [home.username for home in listing.homes] == ["scout"]
    assert listing.homes[0].state is AgentHomeState.UNREADABLE
    assert listing.homes[0].fault is AgentHomeFault.NOT_A_DIRECTORY


def test_filesystem_bookkeeping_entries_are_not_reported_as_broken_agents(
    tmp_path: Path,
) -> None:
    """`.DS_Store` is not a damaged agent, and reporting it as one would be noise on every Mac.

    A leading dot cannot be a username under any circumstance, so such an entry never
    competes for one.

    Killed by: src/uclone_x/core/agent_home.py :: if not entry.name.startswith(".")
    Becomes: if not entry.name.startswith("x")
    """
    (tmp_path / ".DS_Store").write_text("", encoding="utf-8")
    (tmp_path / "scout").mkdir()

    listing = list_agent_homes(root=tmp_path)

    assert [home.username for home in listing.homes] == ["scout"]


def test_the_listing_reads_the_root_the_override_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that passes no root gets the same root every other caller resolves.

    `UCLONE_AGENTS_DIR` redirects where homes live; a listing that read
    `DEFAULT_AGENTS_ROOT` regardless would report the developer's own agents to a test,
    an isolated run, or a second install.

    Killed by: src/uclone_x/core/agent_home.py ::
    agents_root = root if root is not None else default_agents_root()
    Becomes: agents_root = root if root is not None else DEFAULT_AGENTS_ROOT
    """
    (tmp_path / "scout").mkdir()
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path))

    listing = list_agent_homes()

    assert listing.root == tmp_path
    assert [home.username for home in listing.homes] == ["scout"]
