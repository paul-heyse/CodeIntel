"""Plugin execution infrastructure.

This package contains execution context and executor implementations:

Context Types
-------------
- PluginExecutionContext: Runtime context for plugin execution
- PluginScratch: Inter-plugin data sharing store
- ConfigProvider: Typed configuration access
- PluginExecutionContextBuilder: Builder pattern for contexts
- BaseExecutorContext: Base class for domain-specific executor contexts

Executor Types
--------------
- BasePluginExecutor: Abstract base class for plugin executors
- BaseExecutionPolicy: Common execution policy settings

Settings
--------
- PluginExecutionSettings: Per-plugin execution settings
- PluginSeverity: Severity level type
- build_plugin_settings_from_policy: Build settings from policy

Manifest
--------
- PluginExecutionManifest: Execution history tracking
- ManifestState: State for skip detection
- InputHashPayload: Inputs for content hashing
- compute_input_hash: Compute content hash
- compute_options_hash: Compute options hash
- is_unchanged: Check for unchanged inputs
- create_skip_record: Create skip execution record
- build_manifest_entry: Build manifest entry from record

Tracking
--------
- record_plugin_steps: Record plugin results to pipeline tracking
- complete_run_from_records: Complete a pipeline run from results
"""

from __future__ import annotations

from codeintel.core.plugins.execution.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.core.plugins.execution.executor import BasePluginExecutor
from codeintel.core.plugins.execution.executor_context import BaseExecutorContext
from codeintel.core.plugins.execution.manifest import (
    InputHashPayload,
    ManifestState,
    PluginExecutionManifest,
    build_manifest_entry,
    compute_input_hash,
    compute_options_hash,
    create_skip_record,
    is_unchanged,
)
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.core.plugins.execution.settings import (
    PluginExecutionSettings,
    PluginSeverity,
    build_plugin_settings_from_policy,
)
from codeintel.core.plugins.execution.tracking import (
    complete_run_from_records,
    record_plugin_steps,
)

__all__ = [
    "BaseExecutionPolicy",
    "BaseExecutorContext",
    "BasePluginExecutor",
    "ConfigProvider",
    "InputHashPayload",
    "ManifestState",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginExecutionManifest",
    "PluginExecutionSettings",
    "PluginScratch",
    "PluginSeverity",
    "build_manifest_entry",
    "build_plugin_settings_from_policy",
    "complete_run_from_records",
    "compute_input_hash",
    "compute_options_hash",
    "create_skip_record",
    "is_unchanged",
    "record_plugin_steps",
]
