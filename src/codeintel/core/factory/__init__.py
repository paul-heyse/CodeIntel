"""Unified factory infrastructure.

This module provides core factory patterns for the codebase,
including protocols, base classes, and registry.

Examples
--------
Creating a factory:

>>> from codeintel.core.factory import BaseFactory
>>>
>>> class ConfigFactory(BaseFactory[dict[str, str]]):
...     FACTORY_NAME = "config"
...
...     def _do_create(self, **kwargs: object) -> dict[str, str]:
...         return {"env": str(kwargs.get("env", "dev"))}

Using the factory registry:

>>> from codeintel.core.factory import FactoryRegistry
>>>
>>> registry = FactoryRegistry()
>>> registry.register(config_factory)
>>> config = registry.get("config").create(env="prod")
"""

from codeintel.core.factory.base import (
    BaseFactory,
    CachingFactory,
)
from codeintel.core.factory.protocol import (
    CachingFactoryProtocol,
    FactoryCreationError,
    FactoryError,
    FactoryProtocol,
)
from codeintel.core.factory.registry import (
    FactoryNotFoundError,
    FactoryRegistry,
)

__all__ = [
    "BaseFactory",
    "CachingFactory",
    "CachingFactoryProtocol",
    "FactoryCreationError",
    "FactoryError",
    "FactoryNotFoundError",
    "FactoryProtocol",
    "FactoryRegistry",
]
