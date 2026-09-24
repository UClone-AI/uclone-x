"""Cross-subsystem contracts that belong to no single subsystem."""

# `host`, `workspace`, `session_store` and `capability` are not re-exported here. The
# reason first given — that re-export would make `import uclone_x.core` expensive — was
# wrong: importing this package already pulls 79 `uclone_x` modules across every
# subsystem through `core/session_diagnostics.py`, and `core.host` costs about four more.
# The actual reason is narrower: these are contracts nothing consumes yet, so a
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
