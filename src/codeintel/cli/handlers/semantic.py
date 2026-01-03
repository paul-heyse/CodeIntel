"""Handlers for semantic registry CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import fail_execution_failed
from codeintel.cli.handlers.runtime_helpers import (
    CliRuntimeComposeOptions,
    compose_cli_runtime_bundle,
)
from codeintel.cli.resolution.errors import ResolutionError
from codeintel.serving.semantic.registry_compiler import (
    SemanticTagValidationError,
    compile_semantic_registry,
)

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext


def semantic_compile_handler(ctx: CommandContext) -> CliResult[str]:
    """Compile the semantic registry for the current runtime graph.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[str]
        Semantic registry JSON payload.
    """
    output_file = ctx.params.get_str("output_file")
    version = ctx.params.get_str("version") or "v1"

    try:
        runtime = ctx.runtime
    except ResolutionError as exc:
        return fail_execution_failed("semantic", str(exc))

    runtime_bundle = compose_cli_runtime_bundle(
        runtime=runtime,
        gateway=ctx.gateway,
        options=CliRuntimeComposeOptions(verbosity=ctx.verbosity),
    )
    schema_index = runtime_bundle.schema_index
    if schema_index is None:
        return fail_execution_failed("semantic", "Schema index unavailable", status=500)

    try:
        registry = compile_semantic_registry(
            tag_query=runtime_bundle.tag_query,
            schema_provider=schema_index.schema_provider(),
            version=version,
        )
    except SemanticTagValidationError as exc:
        message = "Semantic registry validation failed:\n"
        message += "\n".join(f"- {issue.node}: {issue.message}" for issue in exc.issues)
        return fail_execution_failed("semantic", message, status=409)

    payload = registry.to_json() + "\n"
    if output_file and output_file != "-":
        Path(output_file).write_text(payload, encoding="utf-8")

    return CliResult.ok(
        payload,
        metadata={"view_count": len(registry.views), "version": registry.version},
    )


__all__ = [
    "semantic_compile_handler",
]
