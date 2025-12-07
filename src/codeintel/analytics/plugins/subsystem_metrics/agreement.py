"""Subsystem agreement plugin.

Compare subsystem assignments with import community labels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


class SubsystemAgreementPlugin(TargetPlugin):
    """Compare subsystem assignments with import community labels.

    Checks consistency between:
    - Inferred subsystem assignments
    - Import graph community detection
    - Identifies disagreement areas

    Outputs
    -------
    - analytics.subsystem_agreement: Per-module agreement status
    """

    plugin_name: ClassVar[str] = "subsystem_agreement"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Compare subsystem assignments with import community labels."
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result.
        """
        _ = self  # Protocol method requires instance

        repo = ctx.snapshot.repo
        commit = ctx.snapshot.commit

        try:
            log.info("Computing subsystem agreement for %s@%s", repo, commit)
            compute_subsystem_agreement(ctx.gateway, repo=repo, commit=commit)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Subsystem agreement computation failed: {e}")

        # Count rows written
        row = ctx.gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.subsystem_agreement
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        log.info("Subsystem agreement completed: %d rows", row_count)
        return TargetResult.succeeded(row_counts={"analytics.subsystem_agreement": row_count})


__all__ = ["SubsystemAgreementPlugin"]
