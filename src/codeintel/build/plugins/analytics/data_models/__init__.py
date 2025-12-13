"""Data models analytics plugins using the new protocol."""

from __future__ import annotations

from codeintel.build.plugins.analytics.data_models.build import DataModelsPlugin
from codeintel.build.plugins.analytics.data_models.usage import DataModelUsagePlugin

__all__ = [
    "DataModelUsagePlugin",
    "DataModelsPlugin",
]
