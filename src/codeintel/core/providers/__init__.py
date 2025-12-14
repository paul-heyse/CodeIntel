"""Unified provider infrastructure.

This module provides core provider patterns for the codebase,
including protocols, base classes, and lazy loading utilities.

Examples
--------
Using a base provider:

>>> from codeintel.core.providers import BaseProvider
>>>
>>> class ConfigProvider(BaseProvider[dict[str, str]]):
...     PROVIDER_NAME = "config"
...
...     def _load(self) -> dict[str, str]:
...         return load_config_file()

Using the lazy provider decorator:

>>> from codeintel.core.providers import lazy_provider
>>>
>>> @lazy_provider("settings")
... def get_settings() -> dict[str, str]:
...     return load_settings()
>>> get_settings.get()
{'key': 'value'}

Using make_provider:

>>> from codeintel.core.providers import make_provider
>>>
>>> provider = make_provider(load_data, name="data")
>>> data = provider.get()
"""

from codeintel.core.providers.base import (
    BaseProvider,
    CachedProvider,
    LazyProvider,
)
from codeintel.core.providers.lazy import (
    lazy_provider,
    make_provider,
)
from codeintel.core.providers.protocol import (
    OptionalProviderProtocol,
    ProviderError,
    ProviderNotAvailableError,
    ProviderNotLoadedError,
    ProviderProtocol,
)

__all__ = [
    "BaseProvider",
    "CachedProvider",
    "LazyProvider",
    "OptionalProviderProtocol",
    "ProviderError",
    "ProviderNotAvailableError",
    "ProviderNotLoadedError",
    "ProviderProtocol",
    "lazy_provider",
    "make_provider",
]
