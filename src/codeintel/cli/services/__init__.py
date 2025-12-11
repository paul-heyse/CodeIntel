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

from codeintel.cli.services.jobs import JobService
from codeintel.cli.services.params import ParamService
from codeintel.cli.services.runtime import RuntimeService
from codeintel.cli.services.serving import ServingService
from codeintel.cli.services.storage import StorageService

__all__ = [
    "JobService",
    "ParamService",
    "RuntimeService",
    "ServingService",
    "StorageService",
]
