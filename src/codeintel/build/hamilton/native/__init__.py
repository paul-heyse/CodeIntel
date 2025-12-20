"""Native Hamilton target support.

This package contains pure Hamilton implementations of build targets,
replacing the plugin-based wrapper approach with explicit compute + materialize nodes.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.build.hamilton.native.executor import NativeTargetExecutor
    from codeintel.build.hamilton.native.ibis_helpers import (
        filter_for_snapshot,
        filter_tables_for_snapshot,
        select_snapshot_columns,
    )
    from codeintel.build.hamilton.run_records import (
        NativeRunInfo,
        create_run_record,
        save_manifest,
        should_skip_native_target,
    )

    def _typecheck_exports() -> tuple[object, ...]:
        return (
            NativeRunInfo,
            NativeTargetExecutor,
            create_run_record,
            filter_for_snapshot,
            filter_tables_for_snapshot,
            save_manifest,
            select_snapshot_columns,
            should_skip_native_target,
        )

_EXPORTS: dict[str, tuple[str, str]] = {
    "NativeRunInfo": ("codeintel.build.hamilton.run_records", "NativeRunInfo"),
    "create_run_record": ("codeintel.build.hamilton.run_records", "create_run_record"),
    "save_manifest": ("codeintel.build.hamilton.run_records", "save_manifest"),
    "should_skip_native_target": (
        "codeintel.build.hamilton.run_records",
        "should_skip_native_target",
    ),
    "NativeTargetExecutor": (
        "codeintel.build.hamilton.native.executor",
        "NativeTargetExecutor",
    ),
    "filter_for_snapshot": (
        "codeintel.build.hamilton.native.ibis_helpers",
        "filter_for_snapshot",
    ),
    "filter_tables_for_snapshot": (
        "codeintel.build.hamilton.native.ibis_helpers",
        "filter_tables_for_snapshot",
    ),
    "select_snapshot_columns": (
        "codeintel.build.hamilton.native.ibis_helpers",
        "select_snapshot_columns",
    ),
}

__all__ = [
    "NativeRunInfo",
    "NativeTargetExecutor",
    "create_run_record",
    "filter_for_snapshot",
    "filter_tables_for_snapshot",
    "save_manifest",
    "select_snapshot_columns",
    "should_skip_native_target",
]


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
