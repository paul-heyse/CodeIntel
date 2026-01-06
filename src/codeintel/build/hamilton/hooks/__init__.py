"""Hamilton lifecycle hooks for build execution.

This module consolidates all Hamilton adapter hooks into a single location:
- ManifestHook: Skip logic and manifest persistence
- TelemetryHook: Node-level execution telemetry
- LifecycleHooks: Progress bars, timing, and conditional execution

Hooks are composable via Hamilton's Builder.with_adapters() pattern.

Example
-------
>>> from codeintel.build.hamilton.hooks import build_hooks
>>> hooks = build_hooks(run_id, writer)
>>> driver = Builder().with_modules(modules).with_adapters(*hooks).build()
Using progress bars:

>>> from codeintel.build.hamilton.hooks import create_progress_hook
>>> progress = create_progress_hook("Building targets")
>>> driver = Builder().with_adapters(progress).build()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.hamilton.hooks.event_stream import LifecycleEventStreamHook

# Re-export from lifecycle hooks (progress, timing, conditional)
from codeintel.build.hamilton.hooks.lifecycle import (
    BuildTimingHook,
    ConditionalHook,
    NodeTimingRecord,
    ProgressBarHook,
    create_progress_hook,
)

# Re-export from telemetry hook
from codeintel.build.hamilton.hooks.telemetry_hook import (
    NodeExecutionRecord,
    NodeIOTelemetryHook,
    NodeTelemetryHook,
)

# Re-export from manifest hook
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    compute_target_options_hash,
    save_manifest,
)

if TYPE_CHECKING:
    from codeintel.build.hamilton.run_writer import BuildRunWriter


@dataclass(frozen=True, slots=True)
class HookOptions:
    """Configuration options for `build_hooks`."""

    enable_telemetry: bool = True
    enable_io_telemetry: bool = True
    enable_progress: bool = False
    enable_timing: bool = False
    progress_desc: str = "Building targets"
    telemetry_output_path: Path | None = None
    io_telemetry_output_path: Path | None = None


def build_hooks(
    run_id: str,
    writer: BuildRunWriter,
    *,
    options: HookOptions | None = None,
) -> list[object]:
    """Build the standard hook set for build execution.

    Creates a list of Hamilton lifecycle hooks configured for the build system.
    Hooks are composable and can be passed to Builder.with_adapters().

    Parameters
    ----------
    run_id
        Build run identifier for telemetry grouping.
    writer
        Build run writer used for persistence operations.
    options
        Hook options. When omitted, uses the defaults from `HookOptions`.

    Returns
    -------
    list[object]
        List of configured hook instances.

    Examples
    --------
    >>> hooks = build_hooks("run-123", writer)
    >>> len(hooks)
    1  # telemetry

    >>> hooks = build_hooks("run-123", writer, options=HookOptions(enable_progress=True))
    >>> len(hooks)
    2  # telemetry + progress
    """
    if options is None:
        options = HookOptions()
    hooks: list[object] = []

    if options.enable_telemetry:
        hooks.append(
            NodeTelemetryHook(
                run_id,
                writer,
                output_path=options.telemetry_output_path,
            )
        )
    if options.enable_io_telemetry and options.io_telemetry_output_path is not None:
        hooks.append(
            NodeIOTelemetryHook(
                run_id,
                output_path=options.io_telemetry_output_path,
            )
        )

    if options.enable_progress:
        hooks.append(create_progress_hook(options.progress_desc))

    if options.enable_timing:
        hooks.append(BuildTimingHook())

    return hooks


__all__ = [
    "BuildTimingHook",
    "ConditionalHook",
    "HookOptions",
    "LifecycleEventStreamHook",
    "NodeExecutionRecord",
    "NodeIOTelemetryHook",
    "NodeTelemetryHook",
    "NodeTimingRecord",
    "ProgressBarHook",
    "TargetRunRecord",
    "build_hooks",
    "compute_target_options_hash",
    "create_progress_hook",
    "save_manifest",
]
