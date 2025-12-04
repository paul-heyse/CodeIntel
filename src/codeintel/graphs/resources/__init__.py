"""Resource provider pattern for dependency injection.

This package implements a typed resource provider pattern that enables
dependency injection for graph plugins, allowing clean separation between
computation and infrastructure concerns.

Key Components
--------------
- ResourceProvider: Base protocol for all resource providers (from core)
- ResourceRegistry: DI container for registering and resolving resources (from core)
- CatalogResource: Function catalog access
- GraphResource: Graph engine access
- StorageResource: Storage operations

Example
-------
```python
from codeintel.core.resources import ResourceRegistry
from codeintel.graphs.resources import CatalogResource

# Build registry
resources = ResourceRegistry()
resources.register_provider(CatalogResource(catalog))

# Resolve in plugin
catalog = resources.require_by_name(CatalogResource.RESOURCE_NAME)
spans = catalog.function_spans
```
"""

from __future__ import annotations

from codeintel.core.resources import ResourceProvider, ResourceProviderBase, ResourceRegistry
from codeintel.graphs.resources.catalog import CatalogResource
from codeintel.graphs.resources.graphs import GraphResource
from codeintel.graphs.resources.storage import StorageResource

__all__ = [
    "CatalogResource",
    "GraphResource",
    "ResourceProvider",
    "ResourceProviderBase",
    "ResourceRegistry",
    "StorageResource",
]
