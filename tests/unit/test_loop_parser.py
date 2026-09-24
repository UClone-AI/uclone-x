"""Unit tests for interval string and natural language loop parser (Issue FR-Loop, P4)."""

import pytest

from uclone_x.agent.loop.parser import (
    MIN_INTERVAL_SECONDS,
    parse_interval_string,
    parse_loop_command_input,
)


def test_parse_interval_string_valid() -> None:
    assert parse_interval_string("10s") == 10.0
    assert parse_interval_string("30sec") == 30.0
    assert parse_interval_string("5m") == 300.0
    assert parse_interval_string("15min") == 900.0
    assert parse_interval_string("1h") == 3600.0
    assert parse_interval_string("2d") == 172800.0
    assert parse_interval_string("1.5h") == 5400.0
    # Korean units
    assert parse_interval_string("10초") == 10.0
    assert parse_interval_string("5분") == 300.0
    assert parse_interval_string("1시간") == 3600.0
    assert parse_interval_string("2일") == 172800.0


def test_parse_interval_string_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid interval format"):
        parse_interval_string("invalid")

    with pytest.raises(ValueError, match="Unknown time unit"):
        parse_interval_string("10xyz")

    with pytest.raises(ValueError, match=f"at least {MIN_INTERVAL_SECONDS:.1f}s"):
        parse_interval_string("0.1s")


def test_parse_loop_command_input_structured() -> None:
    seconds, prompt = parse_loop_command_input("5m ./ucx test check")
    assert seconds == 300.0
    assert prompt == "./ucx test check"

    seconds, prompt = parse_loop_command_input("30s check server status")
    assert seconds == 30.0
    assert prompt == "check server status"

    seconds, prompt = parse_loop_command_input("1h run daily report")
    assert seconds == 3600.0
    assert prompt == "run daily report"


def test_parse_loop_command_input_natural_language_korean() -> None:
    seconds, prompt = parse_loop_command_input("5분마다 ./ucx test check 돌리고 보고해줘")
    assert seconds == 300.0
    assert prompt == "./ucx test check 돌리고 보고해줘"

    seconds, prompt = parse_loop_command_input("30초 간격으로 서버 헬스체크 확인해줘")
    assert seconds == 30.0
    assert prompt == "서버 헬스체크 확인해줘"

    seconds, prompt = parse_loop_command_input("1시간 주기로 리포트 생성")
    assert seconds == 3600.0
    assert prompt == "리포트 생성"


def test_parse_loop_command_input_natural_language_english() -> None:
    seconds, prompt = parse_loop_command_input("every 5 minutes check git log and report")
    assert seconds == 300.0
    assert prompt == "check git log and report"

    seconds, prompt = parse_loop_command_input("every 30 seconds ping health endpoint")
    assert seconds == 30.0
    assert prompt == "ping health endpoint"

    seconds, prompt = parse_loop_command_input("with interval of 10m summarize changes")
    assert seconds == 600.0
    assert prompt == "summarize changes"


def test_parse_loop_command_input_missing_prompt_or_interval() -> None:
    with pytest.raises(ValueError, match="Usage: /loop"):
        parse_loop_command_input("")

    with pytest.raises(ValueError, match="Missing prompt after interval"):
        parse_loop_command_input("5m")

    with pytest.raises(ValueError, match="Could not recognize interval"):
        parse_loop_command_input("just run this without any interval")
