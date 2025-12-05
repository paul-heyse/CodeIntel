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
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.core.plugins.execution.tracking import (
    complete_run_from_records,
    record_plugin_steps,
)

__all__ = [
    "BaseExecutionPolicy",
    "BaseExecutorContext",
    "BasePluginExecutor",
    "ConfigProvider",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginScratch",
    "complete_run_from_records",
    "record_plugin_steps",
]
