"""Where a person acts on a failed turn, on the command line.

The Core's failure sentence says what stopped and whose side it is on; the sentences here
say where to change it -- a flag, a variable, a command (#1630). Both `ucx run` and
`ucx room` print them.

Kept apart from `cli/commands/run.py` so that `ucx room show`, which only renders a stored
failure, does not import the turn runner and everything it pulls in (the tool registry,
telemetry, the sandbox) to print one line (#1648). Nothing here reads a file or touches the
network; `provider_key_remedy` reads the environment through the provider table.
"""

from __future__ import annotations

from uclone_x.llm.providers import PROVIDERS, env_key

__all__ = [
    "MODEL_WITHOUT_TOOLS_REMEDY",
    "PROVIDER_FAILURE_REMEDIES",
    "provider_key_remedy",
]


#: Where to act on a model without tool support, on the command line. The Core's sentence
#: says which model to pick and leaves where to the head (P8); here it is the flag.
MODEL_WITHOUT_TOOLS_REMEDY = "Choose it with --model."

#: Where to act on a hosted provider's failure, on the command line (#1630). The Core's
#: sentence says what stopped and whose side it is on; this says where to change it.
PROVIDER_FAILURE_REMEDIES: dict[str, str] = {
    "model_without_tools": MODEL_WITHOUT_TOOLS_REMEDY,
    # Worded for both a model the provider no longer serves and one never chosen.
    "model_unavailable": "Choose a model with --model.",
    "provider_auth": "Save a new key with `ucx key set <provider>`.",
    "provider_unreachable": "If you set a custom endpoint, check that address too.",
}


def provider_key_remedy(provider: str | None) -> str:
    """Where to put a new key for `provider` (a display name, as failures carry it).

    A rejected key is the case the connector's own "no key" error never reaches -- the key
    is set, just wrong -- so the remedy names where *this* key came from: the variable,
    when one is set (it outranks the saved key, so saving another would change nothing),
    else the command that saves one for this provider. The provider is looked up in the
    provider table by its display name.
    """
    spec = next((s for s in PROVIDERS.values() if s.display_name == provider), None)
    if spec is None or not spec.key_env_vars:
        return PROVIDER_FAILURE_REMEDIES["provider_auth"]
    overriding = env_key(spec.id)
    if overriding is not None:
        return f"Set a new key in {overriding[1]}."
    return f"Save a new key with `ucx key set {spec.id}`."
