"""Native analytics modules (inferable tabular outputs)."""

from __future__ import annotations

from codeintel.build.hamilton.native.analytics.cfg_dfg_metrics import (
    cfg_block_metrics__base,
    cfg_block_metrics__table,
    cfg_dfg_metrics__table_materializations,
    cfg_function_metrics__base,
    cfg_function_metrics__table,
    cfg_function_metrics_ext__base,
    cfg_function_metrics_ext__table,
    dfg_block_metrics__base,
    dfg_block_metrics__table,
    dfg_function_metrics__base,
    dfg_function_metrics__table,
    dfg_function_metrics_ext__base,
    dfg_function_metrics_ext__table,
    t__cfg_dfg_metrics,
)
from codeintel.build.hamilton.native.analytics.function_ast_features import (
    function_ast_features__base,
    function_ast_features__table,
    t__function_ast_features,
)
from codeintel.build.hamilton.native.analytics.function_types import (
    function_types__base,
    function_types__table,
    t__function_types,
)
from codeintel.build.hamilton.native.analytics.tables_coverage import (
    coverage_functions__table,
    t__coverage_functions,
)
from codeintel.build.hamilton.native.analytics.tables_dependencies import (
    external_dependencies__table,
    external_dependency_calls__table,
    external_deps__table_materializations,
    t__external_deps,
)
from codeintel.build.hamilton.native.analytics.tables_functions import (
    function_metrics__base,
    function_metrics__table,
    t__function_metrics,
)
from codeintel.build.hamilton.native.analytics.tables_modules import (
    module_profile__base,
    module_profile__table,
    t__module_profile,
)
from codeintel.build.hamilton.native.analytics.tables_risk import (
    risk_factors__base,
    risk_factors__table,
    t__risk_factors,
)

__all__ = [
    "cfg_block_metrics__base",
    "cfg_block_metrics__table",
    "cfg_dfg_metrics__table_materializations",
    "cfg_function_metrics__base",
    "cfg_function_metrics__table",
    "cfg_function_metrics_ext__base",
    "cfg_function_metrics_ext__table",
    "coverage_functions__table",
    "dfg_block_metrics__base",
    "dfg_block_metrics__table",
    "dfg_function_metrics__base",
    "dfg_function_metrics__table",
    "dfg_function_metrics_ext__base",
    "dfg_function_metrics_ext__table",
    "external_dependencies__table",
    "external_dependency_calls__table",
    "external_deps__table_materializations",
    "function_ast_features__base",
    "function_ast_features__table",
    "function_metrics__base",
    "function_metrics__table",
    "function_types__base",
    "function_types__table",
    "module_profile__base",
    "module_profile__table",
    "risk_factors__base",
    "risk_factors__table",
    "t__cfg_dfg_metrics",
    "t__coverage_functions",
    "t__external_deps",
    "t__function_ast_features",
    "t__function_metrics",
    "t__function_types",
    "t__module_profile",
    "t__risk_factors",
]
