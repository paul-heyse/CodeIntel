"""Resource provider pattern for dependency injection.

This package implements a typed resource provider pattern that enables
dependency injection for graph plugins, allowing clean separation between
computation and infrastructure concerns.

Key Components
--------------
- ResourceProvider: Base protocol for all resource providers
- ResourceContainer: DI container for registering and resolving resources
- CatalogResource: Function catalog access
- GraphResource: Graph engine access
- StorageResource: Storage operations

Example
-------
```python
from codeintel.graphs.resources import ResourceContainer, CatalogResource

# Build container
container = ResourceContainer()
container.register(CatalogResource(catalog))

# Resolve in plugin
catalog = container.require(CatalogResource)
spans = catalog.function_spans
```
"""

from __future__ import annotations

from codeintel.graphs.resources.catalog import CatalogResource
from codeintel.graphs.resources.container import ResourceContainer
from codeintel.graphs.resources.graphs import GraphResource
from codeintel.graphs.resources.protocol import ResourceProvider
from codeintel.graphs.resources.storage import StorageResource

__all__ = [
    "CatalogResource",
    "GraphResource",
    "ResourceContainer",
    "ResourceProvider",
    "StorageResource",
]
