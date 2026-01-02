"""Metrics seed pack for static diagnostics and graph metrics.

This module provides the MetricsPack which seeds analytics tables:
static_diagnostics and graph_metrics_modules_ext.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.fixtures.rows import (
    GraphMetricsModulesExtRow,
    StaticDiagnosticsRow,
    dataclass_row,
    insert_rows,
)
from tests._helpers.seeds.core import (
    CORE_PACK,
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
    """Seed pack for static diagnostics and graph metrics data.

    Seeds static diagnostics and graph metrics tables. Creates realistic metrics
    using GOIDs from CORE_PACK.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_static_diagnostics : bool
        Whether to seed static diagnostics.
    include_graph_metrics : bool
        Whether to seed graph metrics.
    """

    name: str = "metrics"
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

        Seeds static diagnostics and graph metrics tables.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)

        if self.include_static_diagnostics:
            self._seed_static_diagnostics(ctx)

        if self.include_graph_metrics:
            self._seed_graph_metrics(ctx, now)

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
