"""Subsystem agreement plugin.

Compare subsystem assignments with import community labels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata

log = logging.getLogger(__name__)


SUBSYSTEM_AGREEMENT_METADATA = CorePluginMetadata(
    name="analytics.subsystem_agreement",
    version="3.0.0",
    description="Compare subsystem assignments with import community labels.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="subsystem",
    provides=("analytics.subsystem_agreement",),
    requires=("analytics.subsystems", "analytics.subsystem_modules"),
    produces_tables=("analytics.subsystem_agreement",),
    consumes_tables=("analytics.subsystems", "analytics.subsystem_modules"),
)


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
    _core_metadata: ClassVar[CorePluginMetadata] = SUBSYSTEM_AGREEMENT_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

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
        _ = self

        repo = ctx.snapshot.repo
        commit = ctx.snapshot.commit

        try:
            log.info("Computing subsystem agreement for %s@%s", repo, commit)
            compute_subsystem_agreement(ctx.gateway, repo=repo, commit=commit)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Subsystem agreement computation failed: {e}")

        row = ctx.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.subsystem_agreement
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        log.info("Subsystem agreement completed: %d rows", row_count)
        return TargetResult.succeeded(row_counts={"analytics.subsystem_agreement": row_count})


__all__ = ["SUBSYSTEM_AGREEMENT_METADATA", "SubsystemAgreementPlugin"]
