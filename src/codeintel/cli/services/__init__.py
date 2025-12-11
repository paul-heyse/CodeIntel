"""Unified service layer for CLI commands.

This package provides service abstractions that consolidate duplicated logic:

- ``ParamService``: Unified parameter coercion
- ``RuntimeService``: Project/runtime resolution with caching
- ``StorageService``: Gateway lifecycle management
- ``ServingService``: Serving operation invocation
- ``JobService``: Background job management

All services are designed for lazy initialization and proper lifecycle management
through context managers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from codeintel.cli.services.jobs import JobService
    from codeintel.cli.services.params import ParamService
    from codeintel.cli.services.runtime import RuntimeService
    from codeintel.cli.services.serving import ServingService
    from codeintel.cli.services.storage import StorageService
else:
    JobService = cast("Any", None)
    ParamService = cast("Any", None)
    RuntimeService = cast("Any", None)
    ServingService = cast("Any", None)
    StorageService = cast("Any", None)

__all__ = [
    "JobService",
    "ParamService",
    "RuntimeService",
    "ServingService",
    "StorageService",
]

_SERVICE_MODULES = {
    "JobService": "codeintel.cli.services.jobs",
    "ParamService": "codeintel.cli.services.params",
    "RuntimeService": "codeintel.cli.services.runtime",
    "ServingService": "codeintel.cli.services.serving",
    "StorageService": "codeintel.cli.services.storage",
}


def __getattr__(name: str) -> object:
    module_path = _SERVICE_MODULES.get(name)
    if module_path is None:
        message = f"module {__name__} has no attribute {name}"
        raise AttributeError(message)
    module = importlib.import_module(module_path)
    return getattr(module, name)
