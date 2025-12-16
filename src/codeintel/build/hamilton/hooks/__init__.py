"""Hamilton lifecycle hooks for build execution.

This module consolidates all Hamilton adapter hooks into a single location:
- ManifestHook: Skip logic and manifest persistence
- TelemetryHook: Node-level execution telemetry
- ContractHook: Schema validation enforcement and result capture
- LifecycleHooks: Progress bars, timing, and conditional execution

Hooks are composable via Hamilton's Builder.with_adapters() pattern.

Example
-------
>>> from codeintel.build.hamilton.hooks import build_hooks
>>> hooks = build_hooks(run_id, gateway, graph, enable_validation=True)
>>> driver = (
...     Builder()
...     .with_modules(modules)
...     .with_adapters(*hooks)
...     .build()
... )
>>> # After execution, get validation summary
>>> contract_hook = next(h for h in hooks if isinstance(h, ContractEnforcementHook))
>>> summary = contract_hook.get_validation_summary()

Using progress bars:

>>> from codeintel.build.hamilton.hooks import create_progress_hook
>>> progress = create_progress_hook("Building targets")
>>> driver = Builder().with_adapters(progress).build()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export from contract hook
from codeintel.build.hamilton.hooks.contract_hook import (
    ContractEnforcementHook,
    ValidationResult,
    ValidationSummary,
)

# Re-export from lifecycle hooks (progress, timing, conditional)
from codeintel.build.hamilton.hooks.lifecycle import (
    BuildTimingHook,
    ConditionalHook,
    NodeTimingRecord,
    ProgressBarHook,
    create_progress_hook,
)

# Re-export from manifest hook
from codeintel.build.hamilton.hooks.manifest_hook import (
    ManifestSaveRequest,
    SkipCheckRequest,
    TargetRunRecord,
    compute_target_input_hash,
    compute_target_input_hash_with_deps,
    compute_target_options_hash,
    save_manifest,
    should_skip,
)

# Re-export from telemetry hook
from codeintel.build.hamilton.hooks.telemetry_hook import (
    NodeExecutionRecord,
    NodeTelemetryHook,
)

if TYPE_CHECKING:
    from codeintel.build.targets import TargetGraph
    from codeintel.storage.gateway import StorageGateway


def build_hooks(
    run_id: str,
    gateway: StorageGateway,
    graph: TargetGraph,
    *,
    strict_contracts: bool = False,
    enable_validation: bool = True,
    enable_telemetry: bool = True,
    enable_progress: bool = False,
    enable_timing: bool = False,
    progress_desc: str = "Building targets",
) -> list[object]:
    """Build the standard hook set for build execution.

    Creates a list of Hamilton lifecycle hooks configured for the build system.
    Hooks are composable and can be passed to Builder.with_adapters().

    Parameters
    ----------
    run_id
        Build run identifier for telemetry grouping.
    gateway
        Storage gateway for persistence operations.
    graph
        Target graph for contract enforcement lookups.
    strict_contracts
        When True, enable strict schema validation on writes (Pandera-style).
    enable_validation
        When True, enable Hamilton-native validation via @check_output_custom.
        Validation results are captured for reporting. Default True.
    enable_telemetry
        When True, enable node-level telemetry recording.
    enable_progress
        When True, enable tqdm progress bar display.
    enable_timing
        When True, enable detailed node timing collection.
    progress_desc
        Description for the progress bar.

    Returns
    -------
    list[object]
        List of configured hook instances.

    Notes
    -----
    When enable_validation is True, a ContractEnforcementHook is added to
    capture validation results from Hamilton's @check_output_custom decorator.
    Access results via hook.get_validation_summary() after execution.

    Examples
    --------
    >>> hooks = build_hooks("run-123", gateway, graph)
    >>> len(hooks)
    2  # telemetry + validation

    >>> hooks = build_hooks("run-123", gateway, graph, strict_contracts=True)
    >>> len(hooks)
    2  # telemetry + strict validation

    >>> hooks = build_hooks("run-123", gateway, graph, enable_progress=True)
    >>> len(hooks)
    3  # telemetry + validation + progress
    """
    hooks: list[object] = []

    if enable_telemetry:
        hooks.append(NodeTelemetryHook(run_id, gateway))

    # Enable validation by default (captures @check_output_custom results)
    # Use strict mode if strict_contracts is True
    if enable_validation or strict_contracts:
        hooks.append(ContractEnforcementHook(graph, strict=strict_contracts))

    if enable_progress:
        hooks.append(create_progress_hook(progress_desc))

    if enable_timing:
        hooks.append(BuildTimingHook())

    return hooks


__all__ = [
    "BuildTimingHook",
    "ConditionalHook",
    "ContractEnforcementHook",
    "ManifestSaveRequest",
    "NodeExecutionRecord",
    "NodeTelemetryHook",
    "NodeTimingRecord",
    "ProgressBarHook",
    "SkipCheckRequest",
    "TargetRunRecord",
    "ValidationResult",
    "ValidationSummary",
    "build_hooks",
    "compute_target_input_hash",
    "compute_target_input_hash_with_deps",
    "compute_target_options_hash",
    "create_progress_hook",
    "save_manifest",
    "should_skip",
]
