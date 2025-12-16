"""Plugin stubs for backward compatibility during migration.

This module provides stub plugin classes that can be used during the
migration from plugins to native Hamilton modules. These stubs are
intended for test helpers only and will be removed after Phase 6.

Phase 2: Migration stubs for ingestion domain.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.plugin import MetadataPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


def _deprecation_warning(name: str) -> None:
    """Emit a deprecation warning for plugin stub usage.

    Parameters
    ----------
    name
        Plugin name.
    """
    warnings.warn(
        f"{name} is deprecated. Use native Hamilton modules instead. "
        "See: codeintel.build.hamilton.native.ingestion",
        DeprecationWarning,
        stacklevel=3,
    )


REPO_SCAN_METADATA = CorePluginMetadata(
    name="ingest.repo_scan.stub",
    version="3.0.0",
    description="[DEPRECATED] Use native modules.py instead.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="discovery",
    provides=("core.modules", "core.repo_map", "core.file_state"),
    requires=(),
    produces_tables=("core.modules", "core.repo_map", "core.file_state"),
    consumes_tables=(),
    supports_incremental=True,
    scope_aware=True,
)


class RepoScanPlugin(MetadataPlugin):
    """Stub for RepoScanPlugin.

    .. deprecated::
        Use native Hamilton module instead:
        ``codeintel.build.hamilton.native.ingestion.modules``
    """

    _core_metadata: ClassVar[CorePluginMetadata] = REPO_SCAN_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Return a skipped result indicating plugin migration.

        Parameters
        ----------
        ctx
            Execution context (unused).

        Returns
        -------
        TargetResult
            Skipped result indicating migration.
        """
        _deprecation_warning(self.__class__.__name__)
        return TargetResult.skipped()


TESTS_INGEST_METADATA = CorePluginMetadata(
    name="ingest.tests.stub",
    version="3.0.0",
    description="[DEPRECATED] Use native tests.py instead.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="analysis",
    provides=("analytics.test_results",),
    requires=("core.modules",),
    produces_tables=("analytics.test_results",),
    consumes_tables=("core.modules",),
    supports_incremental=True,
    scope_aware=True,
)


class TestsIngestPlugin(MetadataPlugin):
    """Stub for TestsIngestPlugin.

    .. deprecated::
        Use native Hamilton module instead:
        ``codeintel.build.hamilton.native.ingestion.tests``
    """

    _core_metadata: ClassVar[CorePluginMetadata] = TESTS_INGEST_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Return a skipped result indicating plugin migration.

        Parameters
        ----------
        ctx
            Execution context (unused).

        Returns
        -------
        TargetResult
            Skipped result indicating migration.
        """
        _deprecation_warning(self.__class__.__name__)
        return TargetResult.skipped()


__all__ = [
    "RepoScanPlugin",
    "TestsIngestPlugin",
]

