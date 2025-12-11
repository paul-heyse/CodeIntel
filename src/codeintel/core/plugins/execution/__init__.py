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
from codeintel.core.plugins.execution.executor import (
    BasePluginExecutor,
    DefaultPluginExecutionStrategy,
    ExecutionOptions,
    ExecutionReportContext,
    ExecutionStrategy,
    ExecutionStrategyContext,
    PluginExecutionStrategy,
)
from codeintel.core.plugins.execution.executor_context import BaseExecutorContext
from codeintel.core.plugins.execution.manifest import (
    InputHashPayload,
    ManifestQuery,
    ManifestState,
    ManifestStore,
    PluginExecutionManifest,
    build_manifest_entry,
    build_upstream_state_from_records,
    compute_input_hash,
    compute_options_hash,
    compute_scope_id,
    create_skip_record,
    is_unchanged,
)
from codeintel.core.plugins.execution.options import (
    ConfigSource,
    EmptyConfigSource,
    PluginConfigBundle,
    PluginOptionsResolver,
    ProfiledConfigSource,
)
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.core.plugins.execution.profile_loader import (
    load_profile_from_file,
    load_profiles_from_directory,
)
from codeintel.core.plugins.execution.profiles import (
    CI_PROFILE,
    FAST_PROFILE,
    FULL_PROFILE,
    ExecutionProfile,
    get_profile,
    list_profiles,
    register_profile,
)
from codeintel.core.plugins.execution.run_context import (
    PluginRunContext,
    RunContextInputs,
    prepare_plugin_run,
)
from codeintel.core.plugins.execution.settings import (
    PluginExecutionSettings,
    PluginSeverity,
    build_plugin_settings_from_policy,
)
from codeintel.core.plugins.execution.tracking import (
    FatalHandling,
    TrackingOptions,
    complete_run_from_records,
    record_plugin_steps,
)

__all__ = [
    "CI_PROFILE",
    "FAST_PROFILE",
    "FULL_PROFILE",
    "BaseExecutionPolicy",
    "BaseExecutorContext",
    "BasePluginExecutor",
    "ConfigProvider",
    "ConfigSource",
    "DefaultPluginExecutionStrategy",
    "EmptyConfigSource",
    "ExecutionOptions",
    "ExecutionProfile",
    "ExecutionReportContext",
    "ExecutionStrategy",
    "ExecutionStrategyContext",
    "FatalHandling",
    "InputHashPayload",
    "ManifestQuery",
    "ManifestState",
    "ManifestStore",
    "PluginConfigBundle",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginExecutionManifest",
    "PluginExecutionSettings",
    "PluginExecutionStrategy",
    "PluginOptionsResolver",
    "PluginRunContext",
    "PluginScratch",
    "PluginSeverity",
    "ProfiledConfigSource",
    "RunContextInputs",
    "TrackingOptions",
    "build_manifest_entry",
    "build_plugin_settings_from_policy",
    "build_upstream_state_from_records",
    "complete_run_from_records",
    "compute_input_hash",
    "compute_options_hash",
    "compute_scope_id",
    "create_skip_record",
    "get_profile",
    "is_unchanged",
    "list_profiles",
    "load_profile_from_file",
    "load_profiles_from_directory",
    "prepare_plugin_run",
    "record_plugin_steps",
    "register_profile",
]
