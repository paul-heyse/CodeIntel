"""Native analytics modules (relation-first)."""

from __future__ import annotations

from codeintel.build.hamilton.native.analytics.tables_coverage import (
    coverage_functions__table,
    t__coverage_functions,
)
from codeintel.build.hamilton.native.analytics.tables_dependencies import (
    external_deps__calls_table,
    external_deps__table,
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
)
from codeintel.build.hamilton.native.analytics.tables_profiles import (
    file_profile__table,
    function_profile__table,
    profiles__table_materializations,
    t__profiles,
    t__test_profile,
    test_profile__table,
)
from codeintel.build.hamilton.native.analytics.tables_risk import (
    risk_factors__base,
    risk_factors__table,
    t__risk_factors,
)

__all__ = [
    "coverage_functions__table",
    "external_deps__calls_table",
    "external_deps__table",
    "external_deps__table_materializations",
    "file_profile__table",
    "function_metrics__base",
    "function_metrics__table",
    "function_profile__table",
    "module_profile__base",
    "module_profile__table",
    "profiles__table_materializations",
    "risk_factors__base",
    "risk_factors__table",
    "t__coverage_functions",
    "t__external_deps",
    "t__function_metrics",
    "t__profiles",
    "t__risk_factors",
    "t__test_profile",
    "test_profile__table",
]
