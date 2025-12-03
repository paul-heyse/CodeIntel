"""Subsystem seed pack for architectural grouping data.

This module provides the SubsystemPack which seeds analytics.subsystems and
analytics.subsystem_modules tables with architectural grouping data.

The pack depends on CORE_PACK and uses its module definitions to create
realistic subsystem groupings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.builders import (
    SubsystemModuleRow,
    SubsystemRow,
    insert_subsystem_modules,
    insert_subsystems,
)
from tests._helpers.seeds.core import (
    CORE_PACK,
    MOD_A_FQN,
    MOD_B_FQN,
    MOD_C_FQN,
    MOD_UTIL_FQN,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Subsystem Data Constants
# =============================================================================

# Subsystem identifiers
SUBSYSTEM_CORE = "core"
SUBSYSTEM_UTIL = "util"


# =============================================================================
# Subsystem Pack Implementation
# =============================================================================


@dataclass
class SubsystemPack:
    """Seed pack for subsystem architecture data.

    Seeds analytics.subsystems and analytics.subsystem_modules tables with
    architectural grouping data based on modules from CORE_PACK.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_risk_scores : bool
        Whether to include risk score data.
    """

    name: str = "subsystems"
    include_risk_scores: bool = True

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for module data.
        """
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply subsystem seeds to the test context.

        Seeds analytics.subsystems and analytics.subsystem_modules.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)
        self._seed_subsystem_modules(ctx)
        self._seed_subsystems(ctx, now)

    @staticmethod
    def _seed_subsystem_modules(ctx: TestContext) -> None:
        """Seed the subsystem_modules table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            # Core subsystem contains mod_a, mod_b, mod_c
            SubsystemModuleRow(
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_CORE,
                module=MOD_A_FQN,
                role="entrypoint",
            ),
            SubsystemModuleRow(
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_CORE,
                module=MOD_B_FQN,
                role="internal",
            ),
            SubsystemModuleRow(
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_CORE,
                module=MOD_C_FQN,
                role="internal",
            ),
            # Util subsystem contains only util
            SubsystemModuleRow(
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_UTIL,
                module=MOD_UTIL_FQN,
                role="shared",
            ),
        ]
        insert_subsystem_modules(ctx.gateway, rows)

    def _seed_subsystems(self, ctx: TestContext, now: datetime) -> None:
        """Seed the subsystems table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        avg_risk = 0.35 if self.include_risk_scores else None
        max_risk = 0.55 if self.include_risk_scores else None

        rows = [
            SubsystemRow(
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_CORE,
                name="Core",
                description="Core business logic subsystem",
                module_count=3,
                modules_json=json.dumps([MOD_A_FQN, MOD_B_FQN, MOD_C_FQN]),
                entrypoints_json=json.dumps([MOD_A_FQN]),
                internal_edge_count=3,
                external_edge_count=1,
                fan_in=0,
                fan_out=1,
                function_count=3,
                avg_risk_score=avg_risk,
                max_risk_score=max_risk,
                high_risk_function_count=0,
                risk_level="low",
                created_at=now,
            ),
            SubsystemRow(
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=SUBSYSTEM_UTIL,
                name="Utilities",
                description="Shared utility functions",
                module_count=1,
                modules_json=json.dumps([MOD_UTIL_FQN]),
                entrypoints_json=json.dumps([]),
                internal_edge_count=0,
                external_edge_count=1,
                fan_in=1,
                fan_out=0,
                function_count=1,
                avg_risk_score=0.25 if self.include_risk_scores else None,
                max_risk_score=0.25 if self.include_risk_scores else None,
                high_risk_function_count=0,
                risk_level="low",
                created_at=now,
            ),
        ]
        insert_subsystems(ctx.gateway, rows)


# Default instance for common usage
SUBSYSTEM_PACK = SubsystemPack()


__all__ = [
    "SUBSYSTEM_CORE",
    "SUBSYSTEM_PACK",
    "SUBSYSTEM_UTIL",
    "SubsystemPack",
]
