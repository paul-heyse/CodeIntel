"""Resource provider pattern for dependency injection.

This package implements a typed resource provider pattern that enables
dependency injection for graph plugins, allowing clean separation between
computation and infrastructure concerns.

Key Components
--------------
- ResourceProvider: Base protocol for all resource providers (from core)
- ResourceRegistry: DI container for registering and resolving resources (from core)
- CatalogService: Unified function catalog access (from core.catalog)
- GraphProvider: Lazy GraphBundle provider (graphs.runtime-backed)
- GraphResource: Graph engine access
- StorageResource: Storage operations

Example
-------
```python
from codeintel.core.resources import ResourceRegistry
from codeintel.core.catalog import CatalogService

resources = ResourceRegistry()
service = CatalogService.from_db(gateway, repo=repo, commit=commit)
resources.register_provider(service)

catalog = resources.require_by_name(CatalogService.RESOURCE_NAME)
spans = catalog.function_spans
```
"""

from __future__ import annotations

from codeintel.build.graphs.resources.graph_provider import GraphProvider
from codeintel.build.graphs.resources.graphs import GraphResource
from codeintel.build.graphs.resources.storage import StorageResource
from codeintel.core.catalog import CatalogService
from codeintel.core.resources import ResourceProvider, ResourceProviderBase, ResourceRegistry

__all__ = [
    "CatalogService",
    "GraphProvider",
    "GraphResource",
    "ResourceProvider",
    "ResourceProviderBase",
    "ResourceRegistry",
    "StorageResource",
]
