"""Data models analytics plugins using the new protocol."""

from __future__ import annotations

from codeintel.analytics.plugins.data_models.build import DataModelsPlugin
from codeintel.analytics.plugins.data_models.usage import DataModelUsagePlugin

__all__ = [
    "DataModelUsagePlugin",
    "DataModelsPlugin",
]
