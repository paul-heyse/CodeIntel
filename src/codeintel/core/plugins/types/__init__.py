"""Plugin type definitions.

This package contains core type definitions for the plugin system:

Protocol Types
--------------
- PluginProtocol: Core synchronous plugin interface
- AsyncPluginProtocol: Async plugin interface for tool plugins
- PluginMetadata: Static metadata about a plugin
- PluginDependency: Dependency specification
- ValidationOutcome: Contract validation outcome

Result Types
------------
- PluginResult: Execution result with success/failure
- PluginStatus: Status literals (succeeded, failed, skipped)
- PluginExecutionRecord: Record of a single plugin execution
- BasePluginExecutionRecord: Base class for domain-specific records

Report Types
------------
- ExecutionStatus: Aggregate run status
- BaseExecutionReport: Base class for execution reports

Type Literals
-------------
- PluginKind, PluginStage, PluginSeverity, etc.
"""

from __future__ import annotations

from codeintel.core.plugins.types.async_protocol import (
    AsyncPluginProtocol,
    AsyncPluginWithCleanup,
)
from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
)
from codeintel.core.plugins.types.protocol import (
    CapabilityKind,
    InputSource,
    PluginCapability,
    PluginInputSpec,
    PluginIsolation,
    PluginKind,
    PluginMetadata,
    PluginOutputSpec,
    PluginProtocol,
    PluginResourceHints,
    PluginSeverity,
    PluginStage,
    ValidationOutcome,
)
from codeintel.core.plugins.types.report import (
    BaseExecutionReport,
    ExecutionStatus,
)
from codeintel.core.plugins.types.result import (
    BasePluginExecutionRecord,
    PluginExecutionRecord,
    PluginResult,
    PluginStatus,
)

__all__ = [
    "AsyncPluginProtocol",
    "AsyncPluginWithCleanup",
    "BaseExecutionReport",
    "BasePluginExecutionRecord",
    "CapabilityKind",
    "CorePluginMetadata",
    "ExecutionStatus",
    "InputSource",
    "PluginCapability",
    "PluginDomain",
    "PluginExecutionRecord",
    "PluginInputSpec",
    "PluginIsolation",
    "PluginKind",
    "PluginMetadata",
    "PluginOutputSpec",
    "PluginProtocol",
    "PluginResourceHints",
    "PluginResult",
    "PluginSeverity",
    "PluginStage",
    "PluginStatus",
    "ValidationOutcome",
]
