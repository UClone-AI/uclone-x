"""R2's launch line and the helpers around it (design §3.7), without starting Chrome."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.browser.chrome import (
    CHROME_PATH_ENV_VAR,
    PORT_FILE,
    R2Link,
    browser_endpoint,
    find_chrome,
    free_port,
    launch_args,
    recorded_port,
)
from uclone_x.errors import PlainRefusalError


def test_the_launch_line_carries_no_automation_switch(tmp_path: Path) -> None:
    args = launch_args(Path("/chrome"), tmp_path, 9222)

    assert args[0] == "/chrome"
    assert f"--user-data-dir={tmp_path}" in args
    assert "--remote-debugging-port=9222" in args
    assert "--remote-debugging-address=127.0.0.1" in args
    assert args[-1] == "about:blank"
    # These show the automation bar or set navigator.webdriver (step 0 spike).
    assert not [a for a in args if "automation" in a or "headless" in a]


def test_port_zero_is_refused_because_it_sets_webdriver(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="webdriver"):
        launch_args(Path("/chrome"), tmp_path, 0)


def test_the_recorded_port_is_read_back_and_junk_is_ignored(tmp_path: Path) -> None:
    assert recorded_port(tmp_path) is None
    (tmp_path / PORT_FILE).write_text("52011\n", encoding="utf-8")
    assert recorded_port(tmp_path) == 52011
    (tmp_path / PORT_FILE).write_text("0", encoding="utf-8")
    assert recorded_port(tmp_path) is None
    (tmp_path / PORT_FILE).write_text("port", encoding="utf-8")
    assert recorded_port(tmp_path) is None


def test_the_chrome_override_names_the_binary(tmp_path: Path) -> None:
    binary = tmp_path / "chrome"
    binary.write_bytes(b"")

    assert find_chrome({CHROME_PATH_ENV_VAR: str(binary)}) == binary
    assert find_chrome({CHROME_PATH_ENV_VAR: str(tmp_path / "missing")}) is None


async def test_nothing_on_the_port_means_no_endpoint() -> None:
    assert await browser_endpoint(free_port()) is None


async def test_no_chrome_is_refused_in_plain_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CHROME_PATH_ENV_VAR, str(tmp_path / "missing"))

    with pytest.raises(PlainRefusalError) as refused:
        await R2Link.start(tmp_path / "profile")

    assert refused.value.reason_code == "no_chrome"
    assert "Install Chrome" in str(refused.value)


async def test_a_chrome_that_exits_at_once_is_refused(tmp_path: Path) -> None:
    exits = tmp_path / "chrome"
    exits.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    exits.chmod(0o755)

    with pytest.raises(PlainRefusalError) as refused:
        await R2Link.start(tmp_path / "profile", chrome=exits, timeout=5)

    assert refused.value.reason_code == "chrome_did_not_start"
