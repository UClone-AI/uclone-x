"""The tool that repairs this installation, rather than telling the user how to (#1107).

Every other tool here does something the agent could not otherwise do. This one exists
because of something the agent *did*: asked three times to install a missing image
engine, it wrote the user a `pip install` how-to and called nothing
(`~/.uclone/sessions/core/sess_default.json`, 2026-09-18). `bash_run` was registered,
advertised, and unused. The diagnosis in #1107 is that a capability with no stated role
is not reached for -- the one routing rule the system prompt states, `IMAGE_GENERATION`,
names the one tool the agent reliably called.

So the point of a named `install_package` over "the agent could shell out" is not
mechanism, it is address. A tool with this name in the inventory, a failure message that
names it, and `ENVIRONMENT_REPAIR` in the prompt are three statements of the same fact
in the three places a model actually reads.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from uclone_x.core.environment_install import (
    INSTALLABLE_EXTRAS,
    INSTALLABLE_PACKAGES,
    install_into_running_environment,
    resolve_installable,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext


class InstallPackageParams(BaseModel):
    """Parameters for installing one of this project's optional components."""

    package: str = Field(
        description=(
            "The optional component to install. Either an extra of this project "
            f"({', '.join(sorted(INSTALLABLE_EXTRAS))}) -- written bare as `media` or as "
            "`uclone-x[media]` -- or one of the engine distributions "
            f"({', '.join(sorted(INSTALLABLE_PACKAGES))})."
        )
    )


class InstallPackageTool(BaseTool[InstallPackageParams]):
    """Install a missing optional component into the running environment."""

    name = "install_package"
    writes_files: ClassVar[bool] = True  # can create, modify or delete a file on the host (#1167)
    description = (
        "Install one of this project's optional components into the environment this "
        "agent is running in, so a capability that failed for a missing dependency "
        "works on the next attempt. Use this when a tool reports a missing dependency, "
        "or when the user asks you to install something -- do not answer with shell "
        "commands for the user to run. Only this project's own extras and engines can "
        "be installed; anything else is refused. The install can take several minutes "
        "and the result says whether it actually succeeded."
    )
    params_type = InstallPackageParams

    def run(self, params: InstallPackageParams, context: ToolContext) -> dict[str, Any]:
        """Resolve the request against the allowlist, then install it."""
        requirements, reason = resolve_installable(params.package)
        if requirements is None:
            # A refusal is a result, not an exception: the model needs to read why and
            # pick a name from the list, which an error string thrown past it does not
            # allow. `installed: false` keeps it from reading this as a success.
            return {
                "installed": False,
                "requested": params.package,
                "detail": reason,
            }

        installed, detail = install_into_running_environment(*requirements)
        named = ", ".join(requirements)
        return {
            "installed": installed,
            "requested": params.package,
            "resolved": list(requirements),
            "detail": (
                f"`{named}` is installed. Retry what failed; the new package is "
                "importable only in a fresh import, so report honestly if the retry "
                "still fails."
                if installed
                else detail
            ),
        }
