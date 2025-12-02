"""Function analytics plugins using the new protocol.

This module provides function-level analytics plugins migrated
to the new unified plugin protocol.
"""

from __future__ import annotations

from codeintel.analytics.core.plugins.functions.ast_features import (
    FunctionAstFeaturesPlugin,
)
from codeintel.analytics.core.plugins.functions.contracts import (
    FunctionContractsPlugin,
)
from codeintel.analytics.core.plugins.functions.effects import (
    FunctionEffectsPlugin,
)
from codeintel.analytics.core.plugins.functions.history import (
    FunctionHistoryPlugin,
)
from codeintel.analytics.core.plugins.functions.metrics import (
    FunctionMetricsPlugin,
)

__all__ = [
    "FunctionAstFeaturesPlugin",
    "FunctionContractsPlugin",
    "FunctionEffectsPlugin",
    "FunctionHistoryPlugin",
    "FunctionMetricsPlugin",
]
