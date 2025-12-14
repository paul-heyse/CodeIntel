"""Unified service infrastructure.

This module provides the core service patterns for the codebase,
including protocols, base classes, lifecycle management, and
dependency injection.

Examples
--------
Create a simple service:

>>> from codeintel.core.services import BaseService
>>>
>>> class MyService(BaseService):
...     SERVICE_NAME = "my_service"
...
...     def _do_initialize(self) -> None:
...         self._data = load_data()
...
...     def _do_shutdown(self) -> None:
...         self._data = None

Manage multiple services:

>>> from codeintel.core.services import ServiceLifecycle
>>>
>>> lifecycle = ServiceLifecycle()
>>> lifecycle.register(db_service, priority=1)
>>> lifecycle.register(cache_service, priority=2)
>>>
>>> with lifecycle:
...     # Services are started
...     run_application()
... # Services are stopped

Use the service registry:

>>> from codeintel.core.services import ServiceRegistry
>>>
>>> registry = ServiceRegistry()
>>> registry.register_singleton(DatabaseService, db)
>>> registry.register_factory(CacheService, CacheService)
>>>
>>> db = registry.get(DatabaseService)
>>> cache = registry.get(CacheService)
"""

from codeintel.core.services.base import (
    BaseService,
    CachedService,
    LazyService,
    ServiceError,
    ServiceInitializationError,
    ServiceNotReadyError,
)
from codeintel.core.services.lifecycle import (
    ServiceLifecycle,
    ServiceLifecycleError,
)
from codeintel.core.services.protocol import (
    HealthCheckProtocol,
    HealthStatus,
    ServiceProtocol,
    ServiceState,
)
from codeintel.core.services.registry import (
    ServiceAlreadyRegisteredError,
    ServiceEntry,
    ServiceNotFoundError,
    ServiceRegistry,
)

__all__ = [
    "BaseService",
    "CachedService",
    "HealthCheckProtocol",
    "HealthStatus",
    "LazyService",
    "ServiceAlreadyRegisteredError",
    "ServiceEntry",
    "ServiceError",
    "ServiceInitializationError",
    "ServiceLifecycle",
    "ServiceLifecycleError",
    "ServiceNotFoundError",
    "ServiceNotReadyError",
    "ServiceProtocol",
    "ServiceRegistry",
    "ServiceState",
]
