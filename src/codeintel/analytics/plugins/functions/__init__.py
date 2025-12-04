"""Function analytics plugins using the new protocol.

This module provides function-level analytics plugins migrated
to the new unified plugin protocol.
"""

from __future__ import annotations

from codeintel.analytics.plugins.functions.ast_features import (
    FunctionAstFeaturesPlugin,
)
from codeintel.analytics.plugins.functions.contracts import (
    FunctionContractsPlugin,
)
from codeintel.analytics.plugins.functions.effects import (
    FunctionEffectsPlugin,
)
from codeintel.analytics.plugins.functions.history import (
    FunctionHistoryPlugin,
)
from codeintel.analytics.plugins.functions.metrics import (
    FunctionMetricsPlugin,
)

__all__ = [
    "FunctionAstFeaturesPlugin",
    "FunctionContractsPlugin",
    "FunctionEffectsPlugin",
    "FunctionHistoryPlugin",
    "FunctionMetricsPlugin",
]
