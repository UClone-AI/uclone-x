"""Which tests the suite's persona-registry reset reaches, and which it must leave alone.

`tests/conftest.py::_isolate_default_persona_registry` drops the process-wide default persona
registry before a test (#967). Two failures bound it, one in each direction:

- **Without the reset**, six unit tests passed only when an earlier test in the same process had
  already built the registry, and failed on a parallel worker where none had.
- **With the reset on browser tests**, a module-scoped UI server running in this process lost the
  registry it was serving from, and the persona-badge E2E case failed every time (PR #977).

Both tests below prime the registry at module scope, before any function-scoped fixture runs,
which is exactly where a module-scoped UI server builds it. Each then reads what reached the test
body. A pytest-level check rather than a unit test of a helper: the property is the fixture's
wiring and ordering, which a helper called by hand would not exercise.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from uclone_x.agent import persona_registry

#: A registry built before the test, standing in for the one a module-scoped server holds.
_PRIMED = object()


def _cached_registry() -> object:
    return persona_registry._default_persona_registry  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope="module")
def registry_primed_for_the_module() -> Iterator[None]:
    """Leave `_PRIMED` in the process-wide cache for the whole module, as a UI server would."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(persona_registry, "_default_persona_registry", _PRIMED)
        yield


@pytest.mark.usefixtures("registry_primed_for_the_module")
def test_a_unit_test_starts_without_a_registry_an_earlier_one_built() -> None:
    """A non-browser test must not inherit a registry built before it.

    Killed by: tests/conftest.py ::
        monkeypatch.setattr(persona_registry, "_default_persona_registry", None)
    Becomes: pass
    """
    assert _cached_registry() is None


@pytest.mark.e2e
@pytest.mark.usefixtures("registry_primed_for_the_module")
def test_a_browser_test_keeps_the_registry_its_module_scoped_server_built() -> None:
    """An `e2e` test must see the registry that was built before it, untouched.

    Marked `e2e` because the marker is what the reset reads, so it is the gate's browser step,
    not its worker step, that runs this test.

    Killed by: tests/conftest.py :: if node.get_closest_marker("e2e") is not None:
    Becomes: if False:
    """
    assert _cached_registry() is _PRIMED
