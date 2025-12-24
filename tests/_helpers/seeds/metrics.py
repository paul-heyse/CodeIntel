"""Metrics seed pack for function metrics and risk factors.

This module provides the MetricsPack which seeds analytics tables:
function_metrics, goid_risk_factors, typedness, static_diagnostics,
and graph_metrics_modules_ext.

The pack depends on CORE_PACK and uses its GOID definitions to create
realistic metrics data for functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.fixtures.rows import (
    FunctionMetricsRow,
    GraphMetricsModulesExtRow,
    RiskFactorRow,
    StaticDiagnosticsRow,
    TypednessRow,
    dataclass_row,
    insert_rows,
)
from tests._helpers.seeds.core import (
    CORE_PACK,
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_FQN,
    MOD_A_PATH,
    MOD_B_FQN,
    MOD_B_PATH,
    MOD_C_FQN,
    MOD_C_PATH,
    MOD_UTIL_FQN,
    MOD_UTIL_PATH,
)
from tests._helpers.seeds.function_types import FunctionTypesPack

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Metrics Pack Implementation
# =============================================================================


@dataclass
class MetricsPack:
    """Seed pack for function metrics and analytics data.

    Seeds function metrics, risk factors, typedness, static diagnostics,
    and graph metrics tables. Creates realistic metrics using GOIDs
    from CORE_PACK.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_function_metrics : bool
        Whether to seed function metrics.
    include_risk_factors : bool
        Whether to seed risk factors.
    include_typedness : bool
        Whether to seed typedness data.
    include_static_diagnostics : bool
        Whether to seed static diagnostics.
    include_graph_metrics : bool
        Whether to seed graph metrics.
    """

    name: str = "metrics"
    include_function_metrics: bool = True
    include_risk_factors: bool = True
    include_typedness: bool = True
    include_static_diagnostics: bool = True
    include_graph_metrics: bool = True

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for GOID data.
        """
        return (CORE_PACK, FunctionTypesPack())

    def apply(self, ctx: TestContext) -> None:
        """Apply metrics seeds to the test context.

        Seeds function metrics, risk factors, and related tables.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)

        if self.include_function_metrics:
            self._seed_function_metrics(ctx, now)

        if self.include_risk_factors:
            self._seed_risk_factors(ctx)

        if self.include_typedness:
            self._seed_typedness(ctx)

        if self.include_static_diagnostics:
            self._seed_static_diagnostics(ctx)

        if self.include_graph_metrics:
            self._seed_graph_metrics(ctx, now)

    @staticmethod
    def _seed_function_metrics(ctx: TestContext, now: datetime) -> None:
        """Seed function metrics table.

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
                function_goid_h128=GOID_FUNC_A,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_A_PATH}#func_a",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_A_PATH,
                language="python",
                kind="function",
                qualname="func_a",
                start_line=1,
                end_line=10,
                loc=10,
                logical_loc=8,
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
                stmt_count=8,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=now,
            ),
            dataclass_row(
                FunctionMetricsRow,
                function_goid_h128=GOID_FUNC_B,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_B_PATH}#func_b",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_B_PATH,
                language="python",
                kind="function",
                qualname="func_b",
                start_line=1,
                end_line=15,
                loc=15,
                logical_loc=12,
                param_count=1,
                positional_params=1,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=True,
                is_generator=False,
                return_count=2,
                yield_count=0,
                raise_count=1,
                cyclomatic_complexity=5,
                max_nesting_depth=3,
                stmt_count=12,
                decorator_count=1,
                has_docstring=True,
                complexity_bucket="medium",
                created_at=now,
            ),
            dataclass_row(
                FunctionMetricsRow,
                function_goid_h128=GOID_FUNC_C,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_C_PATH}#func_c",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_C_PATH,
                language="python",
                kind="function",
                qualname="func_c",
                start_line=1,
                end_line=8,
                loc=8,
                logical_loc=6,
                param_count=0,
                positional_params=0,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=True,
                return_count=0,
                yield_count=3,
                raise_count=0,
                cyclomatic_complexity=2,
                max_nesting_depth=1,
                stmt_count=6,
                decorator_count=0,
                has_docstring=False,
                complexity_bucket="low",
                created_at=now,
            ),
            dataclass_row(
                FunctionMetricsRow,
                function_goid_h128=GOID_HELPER,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_UTIL_PATH}#helper",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_UTIL_PATH,
                language="python",
                kind="function",
                qualname="helper",
                start_line=1,
                end_line=5,
                loc=5,
                logical_loc=4,
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
                max_nesting_depth=0,
                stmt_count=4,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_risk_factors(ctx: TestContext) -> None:
        """Seed risk factors table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            dataclass_row(
                RiskFactorRow,
                function_goid_h128=GOID_FUNC_A,
                repo=ctx.repo,
                commit=ctx.commit,
                risk_score=1,
                risk_level="low",
                cyclomatic_complexity=3,
                fan_in_count=1,
                fan_out_count=2,
                has_tests=True,
            ),
            dataclass_row(
                RiskFactorRow,
                function_goid_h128=GOID_FUNC_B,
                repo=ctx.repo,
                commit=ctx.commit,
                risk_score=4,
                risk_level="medium",
                cyclomatic_complexity=5,
                fan_in_count=3,
                fan_out_count=4,
                has_tests=True,
            ),
            dataclass_row(
                RiskFactorRow,
                function_goid_h128=GOID_FUNC_C,
                repo=ctx.repo,
                commit=ctx.commit,
                risk_score=2,
                risk_level="low",
                cyclomatic_complexity=2,
                fan_in_count=0,
                fan_out_count=1,
                has_tests=True,
            ),
            dataclass_row(
                RiskFactorRow,
                function_goid_h128=GOID_HELPER,
                repo=ctx.repo,
                commit=ctx.commit,
                risk_score=1,
                risk_level="low",
                cyclomatic_complexity=1,
                fan_in_count=0,
                fan_out_count=0,
                has_tests=True,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_typedness(ctx: TestContext) -> None:
        """Seed typedness table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            dataclass_row(
                TypednessRow,
                repo=ctx.repo,
                commit=ctx.commit,
                path=MOD_A_PATH,
                type_error_count=0,
                annotation_ratio="0.9",
                untyped_defs=1,
                overlay_needed=False,
            ),
            dataclass_row(
                TypednessRow,
                repo=ctx.repo,
                commit=ctx.commit,
                path=MOD_B_PATH,
                type_error_count=1,
                annotation_ratio="0.7",
                untyped_defs=2,
                overlay_needed=True,
            ),
            dataclass_row(
                TypednessRow,
                repo=ctx.repo,
                commit=ctx.commit,
                path=MOD_C_PATH,
                type_error_count=2,
                annotation_ratio="0.3",
                untyped_defs=3,
                overlay_needed=True,
            ),
            dataclass_row(
                TypednessRow,
                repo=ctx.repo,
                commit=ctx.commit,
                path=MOD_UTIL_PATH,
                type_error_count=0,
                annotation_ratio="0.95",
                untyped_defs=0,
                overlay_needed=False,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_static_diagnostics(ctx: TestContext) -> None:
        """Seed static diagnostics table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            dataclass_row(
                StaticDiagnosticsRow,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_A_PATH,
                pyrefly_errors=0,
                pyright_errors=0,
                ruff_errors=0,
                total_errors=0,
                has_errors=False,
            ),
            dataclass_row(
                StaticDiagnosticsRow,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_B_PATH,
                pyrefly_errors=1,
                pyright_errors=0,
                ruff_errors=0,
                total_errors=1,
                has_errors=True,
            ),
            dataclass_row(
                StaticDiagnosticsRow,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_C_PATH,
                pyrefly_errors=1,
                pyright_errors=1,
                ruff_errors=0,
                total_errors=2,
                has_errors=True,
            ),
            dataclass_row(
                StaticDiagnosticsRow,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_UTIL_PATH,
                pyrefly_errors=0,
                pyright_errors=0,
                ruff_errors=0,
                total_errors=0,
                has_errors=False,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_graph_metrics(ctx: TestContext, now: datetime) -> None:
        """Seed graph metrics modules ext table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows = [
            dataclass_row(
                GraphMetricsModulesExtRow,
                repo=ctx.repo,
                commit=ctx.commit,
                module=MOD_A_FQN,
                import_betweenness=0.3,
                import_closeness=0.5,
                import_eigenvector=0.4,
                import_harmonic=0.45,
                import_k_core=2,
                import_constraint=0.6,
                import_effective_size=1.5,
                import_community_id=0,
                import_component_id=0,
                import_component_size=4,
                import_scc_id=0,
                import_scc_size=1,
                created_at=now,
            ),
            dataclass_row(
                GraphMetricsModulesExtRow,
                repo=ctx.repo,
                commit=ctx.commit,
                module=MOD_B_FQN,
                import_betweenness=0.5,
                import_closeness=0.6,
                import_eigenvector=0.5,
                import_harmonic=0.55,
                import_k_core=2,
                import_constraint=0.5,
                import_effective_size=2.0,
                import_community_id=0,
                import_component_id=0,
                import_component_size=4,
                import_scc_id=0,
                import_scc_size=1,
                created_at=now,
            ),
            dataclass_row(
                GraphMetricsModulesExtRow,
                repo=ctx.repo,
                commit=ctx.commit,
                module=MOD_C_FQN,
                import_betweenness=0.1,
                import_closeness=0.4,
                import_eigenvector=0.3,
                import_harmonic=0.35,
                import_k_core=1,
                import_constraint=0.7,
                import_effective_size=1.0,
                import_community_id=1,
                import_component_id=0,
                import_component_size=4,
                import_scc_id=1,
                import_scc_size=1,
                created_at=now,
            ),
            dataclass_row(
                GraphMetricsModulesExtRow,
                repo=ctx.repo,
                commit=ctx.commit,
                module=MOD_UTIL_FQN,
                import_betweenness=0.05,
                import_closeness=0.3,
                import_eigenvector=0.2,
                import_harmonic=0.25,
                import_k_core=1,
                import_constraint=0.8,
                import_effective_size=0.5,
                import_community_id=1,
                import_component_id=0,
                import_component_size=4,
                import_scc_id=2,
                import_scc_size=1,
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)


# Default instance for common usage
METRICS_PACK = MetricsPack()


__all__ = [
    "METRICS_PACK",
    "MetricsPack",
]
