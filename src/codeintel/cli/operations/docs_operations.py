"""Documentation operation specifications.

Define operation specs for the docs command group including
status and generate commands.

Note: These register to the LEGACY registry for backward compatibility.
New handler registrations are in handlers/docs.py (NEW registry).
"""

from __future__ import annotations

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import DocsGenerateResult, DocsStatusResult
from codeintel.cli.execution import OperationCategory, OperationSpec
from codeintel.cli.introspection.registry import register_operation


def _docs_status_handler() -> CliResult[DocsStatusResult]:
    """Check documentation status handler.

    Returns
    -------
    CliResult[DocsStatusResult]
        Documentation status result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    return CliResult.ok(
        DocsStatusResult(
            generated_count=0,
            pending_count=0,
            stale_count=0,
            last_generated=None,
        )
    )


def _docs_generate_handler(
    *,
    targets: list[str] | None = None,
    force: bool = False,
) -> CliResult[DocsGenerateResult]:
    """Generate documentation handler.

    Parameters
    ----------
    targets
        Specific targets to generate docs for.
    force
        Force regeneration even if up-to-date.

    Returns
    -------
    CliResult[DocsGenerateResult]
        Documentation generation result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    _ = targets
    _ = force
    return CliResult.ok(
        DocsGenerateResult(
            generated=[],
            skipped=[],
            errors=[],
        )
    )


# Docs Status Operation (registers to LEGACY registry)
DOCS_STATUS_SPEC: OperationSpec[DocsStatusResult] = register_operation(
    OperationSpec(
        operation_id="docs.status",
        handler=_docs_status_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="Check documentation generation status",
    )
)

# Docs Generate Operation (registers to LEGACY registry)
DOCS_GENERATE_SPEC: OperationSpec[DocsGenerateResult] = register_operation(
    OperationSpec(
        operation_id="docs.generate",
        handler=_docs_generate_handler,
        category=OperationCategory.BUILD,
        param_schema=None,
        requires_progress=True,
        estimated_duration=60.0,
        description="Generate documentation",
    )
)

__all__ = [
    "DOCS_GENERATE_SPEC",
    "DOCS_STATUS_SPEC",
]
