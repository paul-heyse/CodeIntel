"""Factories for building shared query services.

Note
----
This module is a backward-compatibility shim. The canonical implementation
now lives in ``codeintel.serving.bootstrap``.

Import Pattern
--------------
New code should import from ``codeintel.serving.bootstrap``:

    from codeintel.serving.bootstrap import (
        build_service_from_config,
        DatasetRegistryOptions,
        ServiceBuildOptions,
    )

Legacy imports from this module will continue to work:

    from codeintel.serving.services.factory import build_service_from_config
"""

from __future__ import annotations

# Re-export from bootstrap for backward compatibility
from codeintel.serving.bootstrap import (
    DatasetRegistryOptions,
    ServiceBuildOptions,
    build_http_query_service,
    build_local_query_service,
    build_service_from_config,
    get_observability_from_config,
)

# Re-export ServiceObservability for backward compatibility
from codeintel.serving.services.observability import ServiceObservability

# Re-export from wiring for backward compatibility
from codeintel.serving.services.wiring import BackendResource, build_backend_resource

__all__ = [
    "BackendResource",
    "DatasetRegistryOptions",
    "ServiceBuildOptions",
    "ServiceObservability",
    "build_backend_resource",
    "build_http_query_service",
    "build_local_query_service",
    "build_service_from_config",
    "get_observability_from_config",
]
