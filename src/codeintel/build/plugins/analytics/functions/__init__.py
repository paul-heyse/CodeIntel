"""Function analytics plugins using the new protocol.

This module provides function-level analytics plugins migrated
to the new unified plugin protocol.
"""

from __future__ import annotations

from codeintel.build.plugins.analytics.functions.ast_features import (
    FunctionAstFeaturesPlugin,
)
from codeintel.build.plugins.analytics.functions.contracts import (
    FunctionContractsPlugin,
)
from codeintel.build.plugins.analytics.functions.effects import (
    FunctionEffectsPlugin,
)
from codeintel.build.plugins.analytics.functions.history import (
    FunctionHistoryPlugin,
)
from codeintel.build.plugins.analytics.functions.metrics import (
    FunctionMetricsPlugin,
)

__all__ = [
    "FunctionAstFeaturesPlugin",
    "FunctionContractsPlugin",
    "FunctionEffectsPlugin",
    "FunctionHistoryPlugin",
    "FunctionMetricsPlugin",
]
