# pyright: reportPrivateUsage=false
"""A clone's display name: free text per locale, beside the ASCII `name` that stays its id.

A person asked whether a clone can be called "잠 꾸러기". The `name` cannot -- it is the file
name, the agent id and the `@mention` token, so `_USERNAME_RE` keeps it lowercase ASCII --
but the clone can carry `display_name: {ko: 잠 꾸러기}`, which every label reads instead.
The shape is the one the clone-store design gives: a saved clone keeps it in its
`clone.yaml` (clone-data-scopes §3.3), and a package or workspace file carries the same key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import pytest
import yaml
from fastapi.testclient import TestClient

from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.persona_store import (
    BUILTIN_PERSONAS_DIR,
    DISPLAY_NAME_MAX_LENGTH,
    PersonaLoadError,
    YamlFilePersonaStore,
)
from uclone_x.core.agent_home import CLONE_FILE_NAME, clone_handles, default_agents_root
from uclone_x.i18n.language import Language
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.ui.app import create_ui_app

SLEEPY = "잠 꾸러기"


def _client(workspace: Path) -> TestClient:
    app = create_ui_app(
        static_dir=workspace / "static",
        storage_dir=workspace / "sessions",
        llm=MockLLMConnector(default_response="ok"),
        workspace_dir=workspace,
    )
    return TestClient(app)


def _draft(name: str = "sleepy", **overrides: Any) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "name": name,
        "role": "Night Owl",
        "system_prompt": "You nap.",
    }
    draft.update(overrides)
    return draft


def _clone_text(handle: str) -> str:
    """The `clone.yaml` a save wrote for `handle`, in the per-test agents root."""
    root = default_agents_root()
    (agent_id,) = clone_handles(root)[handle]
    return (root / agent_id / CLONE_FILE_NAME).read_text(encoding="utf-8")


def _listed(client: TestClient) -> dict[str, dict[str, Any]]:
    res = client.get("/api/clones")
    assert res.status_code == 200
    return {p["name"]: p for p in res.json()["clones"]}


@pytest.fixture
def workspace(tmp_path: Path, builtin_personas_absent: None) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _write_persona_file(directory: Path, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sleepy.yaml").write_text(
        f"name: sleepy\nrole: Night Owl\nsystem_prompt: You nap.\n{body}", encoding="utf-8"
    )


def test_a_korean_display_name_round_trips_through_the_file_unchanged(workspace: Path) -> None:
    """Saved, it is readable Korean in the file, and a fresh loader reads it back as sent.

    Killed by: src/uclone_x/agent/persona_store.py :: data["display_name"] = display_name_from_mapping(data, file_path)
    Becomes: data.pop("display_name", None)
    """
    client = _client(workspace)

    res = client.post("/api/clones", json=_draft(display_name={"ko": SLEEPY}))

    assert res.status_code == 201, res.text
    text = _clone_text("sleepy")
    assert SLEEPY in text  # written as itself, not as 잠 escapes
    assert yaml.safe_load(text)["display_name"] == {"ko": SLEEPY}
    fresh = PersonaRegistry(workspace_root=workspace, include_defaults=False)
    loaded = fresh.get_persona("sleepy")
    assert loaded is not None
    assert loaded.name == "sleepy"
    assert loaded.display_name == {"ko": SLEEPY}


def test_the_persona_listing_carries_the_display_name(workspace: Path) -> None:
    """The head labels a clone from the listing, so the listing must carry the name.

    Killed by: src/uclone_x/ui/app.py :: "display_name": dict(persona.display_name),
    Becomes: "display_name": {},
    """
    client = _client(workspace)
    assert (
        client.post(
            "/api/clones", json=_draft(display_name={"ko": SLEEPY, "en": "Sleepyhead"})
        ).status_code
        == 201
    )

    assert _listed(client)["sleepy"]["display_name"] == {"ko": SLEEPY, "en": "Sleepyhead"}


def test_a_clone_with_no_display_name_keeps_the_key_out_of_its_file(workspace: Path) -> None:
    client = _client(workspace)

    assert client.post("/api/clones", json=_draft()).status_code == 201

    text = _clone_text("sleepy")
    assert "display_name" not in yaml.safe_load(text)
    assert _listed(client)["sleepy"]["display_name"] == {}


def test_the_name_itself_still_follows_the_handle_rule(workspace: Path) -> None:
    """ "잠 꾸러기" is a display name, not a name: the name is the id, and stays ASCII."""
    client = _client(workspace)

    res = client.post("/api/clones", json=_draft(name=SLEEPY))

    assert res.status_code == 422
    assert "cannot be a persona name" in str(res.json()["detail"])
    assert SLEEPY not in clone_handles(default_agents_root())


def test_a_display_name_with_a_control_character_is_refused_in_plain_words(
    workspace: Path,
) -> None:
    """A line break or tab in a label breaks every row that shows it; it is refused, not kept.

    Killed by: src/uclone_x/agent/persona_store.py :: if any(unicodedata.category(ch) == "Cc" for ch in name):
    Becomes: if False:

    Killed by: src/uclone_x/agent/persona_store.py :: raise PersonaWriteRefused(sentence if sentence.endswith(".") else f"{sentence}.")
    Becomes: raise PersonaWriteRefused(f"{sentence}.")
    """
    client = _client(workspace)

    res = client.post("/api/clones", json=_draft(display_name={"ko": "잠\n꾸러기"}))

    assert res.status_code == 422
    detail = str(res.json()["detail"])
    assert "control character" in detail
    assert "one line" in detail
    assert detail.endswith("visible characters only."), detail
    assert not detail.endswith(".."), detail
    assert "sleepy" not in clone_handles(default_agents_root())


def test_an_over_long_display_name_is_refused_with_the_limit(workspace: Path) -> None:
    """Killed by: src/uclone_x/agent/persona_store.py :: if len(name) > DISPLAY_NAME_MAX_LENGTH:
    Becomes: if False:
    """
    client = _client(workspace)
    at_limit = "가" * DISPLAY_NAME_MAX_LENGTH

    over = client.post("/api/clones", json=_draft(display_name={"ko": at_limit + "가"}))
    fits = client.post("/api/clones", json=_draft(display_name={"ko": at_limit}))

    assert over.status_code == 422
    assert f"{DISPLAY_NAME_MAX_LENGTH} characters or fewer" in str(over.json()["detail"])
    assert fits.status_code == 201, fits.text


@pytest.mark.parametrize(
    ("body", "words"),
    [
        (f"display_name: {SLEEPY}\n", "must map a locale to a name"),
        ("display_name: [a, b]\n", "must map a locale to a name"),
        ("display_name: {ko: '   '}\n", "must map a locale to a non-empty name"),
        ("display_name: {ko: 3}\n", "must map a locale to a non-empty name"),
        ('display_name: {ko: "a\\tb"}\n', "control character"),
        (f"display_name: {{ko: {'x' * (DISPLAY_NAME_MAX_LENGTH + 1)}}}\n", "characters or fewer"),
    ],
)
def test_a_file_with_an_unusable_display_name_does_not_load(
    tmp_path: Path, body: str, words: str
) -> None:
    """The loader refuses the file and says which key is wrong, rather than dropping it."""
    _write_persona_file(tmp_path, body)

    with pytest.raises(PersonaLoadError) as caught:
        YamlFilePersonaStore(tmp_path)

    assert "sleepy.yaml" in str(caught.value)
    assert words in str(caught.value)


def test_a_display_name_is_trimmed_when_loaded(tmp_path: Path) -> None:
    _write_persona_file(tmp_path, f"display_name:\n  ko: '  {SLEEPY}  '\n  en: Sleepyhead\n")

    persona = YamlFilePersonaStore(tmp_path).get_persona("sleepy")

    assert persona is not None
    assert persona.display_name == {"ko": SLEEPY, "en": "Sleepyhead"}


def test_every_builtin_clone_is_named_in_every_language_the_ui_writes() -> None:
    """A shipped clone is labelled in the reader's language from install on, never by its id.

    The languages are read from `Language`, so adding one to the UI makes this fail until
    every builtin is named in it; a new builtin cannot ship without its names either.
    """
    builtins = YamlFilePersonaStore(BUILTIN_PERSONAS_DIR).list_personas()
    languages = get_args(Language)

    assert builtins, "no builtin persona was loaded, so nothing was checked"
    assert languages, "no UI language was found, so nothing was checked"
    missing = {
        persona.name: [lang for lang in languages if not persona.display_name.get(lang, "").strip()]
        for persona in builtins
    }
    assert {name: langs for name, langs in missing.items() if langs} == {}
