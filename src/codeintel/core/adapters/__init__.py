"""Unified adapter infrastructure.

This module provides core adapter patterns for hexagonal architecture,
including protocols and base classes.

Examples
--------
Creating an adapter:

>>> from codeintel.core.adapters import BaseAdapter
>>>
>>> class FileAdapter(BaseAdapter):
...     ADAPTER_NAME = "file_storage"
...
...     def _do_initialize(self) -> None:
...         self._path.mkdir(exist_ok=True)
...
...     def _do_close(self) -> None:
...         pass
...
...     @property
...     def is_available(self) -> bool:
...         return self._path.exists()
"""

from codeintel.core.adapters.base import BaseAdapter
from codeintel.core.adapters.protocol import (
    AdapterError,
    AdapterInitializationError,
    AdapterNotAvailableError,
    AdapterProtocol,
    PortProtocol,
)

__all__ = [
    "AdapterError",
    "AdapterInitializationError",
    "AdapterNotAvailableError",
    "AdapterProtocol",
    "BaseAdapter",
    "PortProtocol",
]
