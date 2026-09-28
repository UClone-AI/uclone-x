"""Cross-subsystem contracts that belong to no single subsystem."""

# `host`, `workspace`, `session_store` and `capability` are not re-exported here. The
# first reason given was that a re-export would make `import uclone_x.core` expensive.
# That is true of `core.host` alone, which reaches into other subsystems, while the other
# three and this package itself stay within `core/` and `errors`. Compare with `python -c
# "import sys, uclone_x.core.host; print(sum(m.startswith('uclone_x') for m in
# sys.modules))"` against the same line for `uclone_x.core`. (An earlier version of this
# comment called that reason wrong, because this package then pulled in every subsystem
# through `core/session_diagnostics.py`; it no longer imports that module.)
# The reason that covers all four is narrower: these are contracts nothing consumes yet, so a
# re-export would fix `from uclone_x.core import HostProtocol` as the obvious spelling
# before the shape has met a consumer.
#
# `secrets` is not in that list and is not omitted for that reason. It has a consumer —
# `sandbox/models.py` imports it — which reaches it through `sandbox.models` rather than
# from here, because C7 relocated the predicate without moving its importers. Re-exporting
# it would add a third spelling for one function while that is still true.

from uclone_x.core.immutable import (
    ImmutableIntMapping,
    ImmutableJsonMapping,
    ImmutableMapping,
    ImmutableStrMapping,
    freeze_mapping,
    unwrap_immutable,
)
from uclone_x.core.log_inspector import (
    LogEntry,
    get_default_log_dir,
    get_log_file,
    parse_log_line,
    read_logs,
)
from uclone_x.core.logging_setup import (
    JsonLogFormatter,
    setup_application_logging,
)
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
    require_provenance,
)

__all__ = [
    "AttemptRecord",
    "ExecutionPath",
    "ImmutableIntMapping",
    "ImmutableJsonMapping",
    "ImmutableMapping",
    "ImmutableStrMapping",
    "JsonLogFormatter",
    "LogEntry",
    "Provenance",
    "ServiceRef",
    "freeze_mapping",
    "get_default_log_dir",
    "get_log_file",
    "parse_log_line",
    "read_logs",
    "require_provenance",
    "setup_application_logging",
    "unwrap_immutable",
]
