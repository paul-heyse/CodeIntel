"""Domain-agnostic plugin traits for capability-based composition.

This package defines protocol classes (traits) that plugins can implement
to declare specific capabilities. These traits are domain-agnostic and
can be used by both graph and analytics plugins.

Trait Categories
----------------
**Isolation**
    IsolatedPlugin - For plugins requiring process or thread isolation

**Caching**
    CacheAwarePlugin / CacheAwareMixin - For plugins that participate in caching

**Retry**
    RetryablePlugin / RetryableMixin - For plugins with custom retry behavior
    get_retry_policy - Utility to get a RetryPolicy from any plugin

**Progress**
    ProgressReportingPlugin / ProgressReportingMixin - For progress reporting

**Incremental**
    IncrementalPlugin - For plugins that support incremental execution

**Data Sharing**
    ScratchContext / WithDependencyData - For plugins that share data via scratch

Domain-specific traits (like GraphAwarePlugin for analytics) should remain
in their respective domain modules.
"""

from __future__ import annotations

from codeintel.core.plugins.traits.caching import (
    CacheAwareMixin,
    CacheAwarePlugin,
    is_cache_aware,
)
from codeintel.core.plugins.traits.incremental import (
    IncrementalPlugin,
    is_incremental,
)
from codeintel.core.plugins.traits.isolated import (
    IsolatedPlugin,
    is_isolated,
)
from codeintel.core.plugins.traits.progress import (
    ProgressReportingMixin,
    ProgressReportingPlugin,
    is_progress_reporting,
)
from codeintel.core.plugins.traits.retryable import (
    RetryableMixin,
    RetryablePlugin,
    get_retry_policy,
    is_retryable,
)
from codeintel.core.plugins.traits.scratch import (
    ScratchContext,
    WithDependencyData,
)

__all__ = [
    "CacheAwareMixin",
    "CacheAwarePlugin",
    "IncrementalPlugin",
    "IsolatedPlugin",
    "ProgressReportingMixin",
    "ProgressReportingPlugin",
    "RetryableMixin",
    "RetryablePlugin",
    "ScratchContext",
    "WithDependencyData",
    "get_retry_policy",
    "is_cache_aware",
    "is_incremental",
    "is_isolated",
    "is_progress_reporting",
    "is_retryable",
]
