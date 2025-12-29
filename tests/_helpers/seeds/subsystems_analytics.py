"""Extended subsystem seed pack with risk and clustering data.

This module provides SubsystemAnalyticsPack which seeds subsystem-related tables
with additional data required for analytics tests including risk factors,
import graph edges for clustering, symbol use edges, and config values.

The pack is designed for tests like test_subsystems.py that need comprehensive
subsystem analytics data beyond the basic SubsystemPack.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.fixtures.rows import (
    ConfigValueRow,
    FunctionMetricsRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RiskFactorRow,
    SubsystemModuleRow,
    SubsystemRow,
    SymbolUseEdgeRow,
    dataclass_row,
    insert_rows,
)
from tests._helpers.seeds.core import (
    CORE_PACK,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Subsystem Analytics Constants
# =============================================================================

# GOIDs for subsystem analytics functions (different from core GOIDs)
GOID_API_HANDLER = 10
GOID_CORE_SERVICE = 11
GOID_MISC_UTIL = 12

# Subsystem identifiers for analytics
SUBSYSTEM_API = "api"
SUBSYSTEM_CORE_ANALYTICS = "core_analytics"

# Module paths specific to analytics tests
MOD_API_PATH = "pkg/api.py"
MOD_CORE_PATH = "pkg/core.py"
MOD_MISC_PATH = "pkg/misc.py"
MOD_API_FQN = "pkg.api"
MOD_CORE_FQN = "pkg.core"
MOD_MISC_FQN = "pkg.misc"


# =============================================================================
# Subsystem Analytics Pack Implementation
# =============================================================================


@dataclass
class SubsystemAnalyticsPack:
    """Seed pack for comprehensive subsystem analytics data.

    Seeds modules, import graph edges, symbol use edges, config values,
    function metrics, risk factors, subsystem modules, and subsystems
    tables with data suitable for clustering and risk aggregation tests.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_import_edges : bool
        Whether to seed import graph edges for clustering.
    include_symbol_edges : bool
        Whether to seed symbol use edges.
    include_config_values : bool
        Whether to seed config values.
    include_function_metrics : bool
        Whether to seed function metrics.
    include_risk_factors : bool
        Whether to seed risk factors.
    """

    name: str = "subsystems_analytics"
    include_import_edges: bool = True
    include_symbol_edges: bool = True
    include_config_values: bool = True
    include_function_metrics: bool = True
    include_risk_factors: bool = True

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for base module data.
        """
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply subsystem analytics seeds to the test context.

        Seeds all subsystem analytics related tables.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)

        # Seed additional modules for subsystem tests
        self._seed_modules(ctx)

        if self.include_import_edges:
            self._seed_import_edges(ctx)

        if self.include_symbol_edges:
            self._seed_symbol_edges(ctx)

        if self.include_config_values:
            self._seed_config_values(ctx)

        if self.include_function_metrics:
            self._seed_function_metrics(ctx, now)

        if self.include_risk_factors:
            self._seed_risk_factors(ctx)

        # Seed subsystem structure
        self._seed_subsystem_modules(ctx)
        self._seed_subsystems(ctx, now)

    @staticmethod
    def _seed_modules(ctx: TestContext) -> None:
        """Seed additional modules for subsystem analytics.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            dataclass_row(
                ModuleRow,
                module=MOD_API_FQN,
                path=MOD_API_PATH,
                repo=ctx.repo,
                commit=ctx.commit,
                tags=["api"],
            ),
            dataclass_row(
                ModuleRow,
                module=MOD_CORE_FQN,
                path=MOD_CORE_PATH,
                repo=ctx.repo,
                commit=ctx.commit,
                tags=["api"],
            ),
            dataclass_row(
                ModuleRow,
                module=MOD_MISC_FQN,
                path=MOD_MISC_PATH,
                repo=ctx.repo,
                commit=ctx.commit,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_import_edges(ctx: TestContext) -> None:
        """Seed import graph edges for clustering.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            dataclass_row(
                ImportGraphEdgeRow,
                repo=ctx.repo,
                commit=ctx.commit,
                src_module=MOD_API_FQN,
                dst_module=MOD_CORE_FQN,
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            ),
            dataclass_row(
                ImportGraphEdgeRow,
                repo=ctx.repo,
                commit=ctx.commit,
                src_module=MOD_CORE_FQN,
                dst_module=MOD_API_FQN,
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_symbol_edges(ctx: TestContext) -> None:
        """Seed symbol use edges.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            dataclass_row(
                SymbolUseEdgeRow,
                symbol="sym_core",
                def_path=MOD_CORE_PATH,
                use_path=MOD_API_PATH,
                same_file=False,
                same_module=False,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_config_values(ctx: TestContext) -> None:
        """Seed config values.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            dataclass_row(
                ConfigValueRow,
                repo=ctx.repo,
                commit=ctx.commit,
                config_path="cfg/app.yaml",
                format="yaml",
                key="feature.flag",
                reference_paths=[],
                reference_modules=[MOD_API_FQN, MOD_CORE_FQN],
                reference_count=2,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_function_metrics(ctx: TestContext, now: datetime) -> None:
        """Seed function metrics for subsystem functions.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows = [
            dataclass_row(
                FunctionMetricsRow,
                function_goid_h128=GOID_API_HANDLER,
                urn=f"goid:{ctx.repo}#python:function:{MOD_API_FQN}.handler",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_API_PATH,
                language="python",
                kind="function",
                qualname=f"{MOD_API_FQN}.handler",
                start_line=1,
                end_line=2,
                loc=4,
                logical_loc=3,
                param_count=1,
                positional_params=1,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=1,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=1,
                max_nesting_depth=1,
                stmt_count=2,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=now,
            ),
            dataclass_row(
                FunctionMetricsRow,
                function_goid_h128=GOID_CORE_SERVICE,
                urn=f"goid:{ctx.repo}#python:function:{MOD_CORE_FQN}.service",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_CORE_PATH,
                language="python",
                kind="function",
                qualname=f"{MOD_CORE_FQN}.service",
                start_line=1,
                end_line=5,
                loc=8,
                logical_loc=6,
                param_count=2,
                positional_params=2,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=1,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=3,
                max_nesting_depth=2,
                stmt_count=5,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_risk_factors(ctx: TestContext) -> None:
        """Seed risk factors for subsystem functions.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            dataclass_row(
                RiskFactorRow,
                function_goid_h128=GOID_API_HANDLER,
                repo=ctx.repo,
                commit=ctx.commit,
                risk_score=1,
                risk_level="low",
                cyclomatic_complexity=1,
                fan_in_count=0,
                fan_out_count=1,
                has_tests=True,
            ),
            dataclass_row(
                RiskFactorRow,
                function_goid_h128=GOID_CORE_SERVICE,
                repo=ctx.repo,
                commit=ctx.commit,
                risk_score=7,
                risk_level="high",
                cyclomatic_complexity=3,
                fan_in_count=2,
                fan_out_count=2,
                has_tests=True,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_subsystem_modules(ctx: TestContext) -> None:
        """Seed subsystem module assignments.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            dataclass_row(
                SubsystemModuleRow,
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_API,
                module=MOD_API_FQN,
                role="entrypoint",
            ),
            dataclass_row(
                SubsystemModuleRow,
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_CORE_ANALYTICS,
                module=MOD_CORE_FQN,
                role="internal",
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_subsystems(ctx: TestContext, now: datetime) -> None:
        """Seed subsystems table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows = [
            dataclass_row(
                SubsystemRow,
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_API,
                name="API Layer",
                description="API entry points",
                module_count=1,
                modules_json=[MOD_API_FQN],
                entrypoints_json=[MOD_API_FQN],
                internal_edge_count=0,
                external_edge_count=1,
                fan_in=0,
                fan_out=1,
                function_count=1,
                avg_risk_score=0.1,
                max_risk_score=0.1,
                high_risk_function_count=0,
                risk_level="low",
                created_at=now,
            ),
            dataclass_row(
                SubsystemRow,
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_CORE_ANALYTICS,
                name="Core Analytics",
                description="Core business logic",
                module_count=1,
                modules_json=[MOD_CORE_FQN],
                entrypoints_json=[],
                internal_edge_count=0,
                external_edge_count=1,
                fan_in=1,
                fan_out=0,
                function_count=1,
                avg_risk_score=0.7,
                max_risk_score=0.7,
                high_risk_function_count=1,
                risk_level="high",
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)


# Default instance for common usage
SUBSYSTEM_ANALYTICS_PACK = SubsystemAnalyticsPack()


__all__ = [
    "GOID_API_HANDLER",
    "GOID_CORE_SERVICE",
    "GOID_MISC_UTIL",
    "MOD_API_FQN",
    "MOD_API_PATH",
    "MOD_CORE_FQN",
    "MOD_CORE_PATH",
    "MOD_MISC_FQN",
    "MOD_MISC_PATH",
    "SUBSYSTEM_ANALYTICS_PACK",
    "SUBSYSTEM_API",
    "SUBSYSTEM_CORE_ANALYTICS",
    "SubsystemAnalyticsPack",
]
