"""Extensions: what a domain adds to the core without the core naming it (#2205).

The core asks `registry` for what extensions add -- tools, turn lifecycle hooks, protected
workspace folders, a leased folder kind for the Files screen, an `a2a_call` character
lookup, and HTTP routes -- and never imports an extension's package. The contract is
`contract.Extension`.
"""

from __future__ import annotations

from uclone_x.extensions.contract import Extension, ExtensionError, ProtectedRoot
from uclone_x.extensions.registry import (
    ENTRY_POINT_GROUP,
    IN_TREE_MODULE,
    a2a_character_lookup,
    discover,
    extension_lifecycle_hooks,
    leased_folder_kinds,
    loaded_extensions,
    mount_extension_routes,
    protected_roots,
    use_extensions,
    with_extension_tools,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "IN_TREE_MODULE",
    "Extension",
    "ExtensionError",
    "ProtectedRoot",
    "a2a_character_lookup",
    "discover",
    "extension_lifecycle_hooks",
    "leased_folder_kinds",
    "loaded_extensions",
    "mount_extension_routes",
    "protected_roots",
    "use_extensions",
    "with_extension_tools",
]
