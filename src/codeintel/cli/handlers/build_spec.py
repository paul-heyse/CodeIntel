"""Handlers for BuildSpec CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.spec import BuildSpecCompileOptions, buildspec_to_json, compile_buildspec
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import fail_execution_failed

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext


def build_spec_compile_handler(ctx: CommandContext) -> CliResult[str]:
    """Compile a BuildSpec for the current repository configuration.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[str]
        JSON BuildSpec payload.
    """
    output_format = ctx.params.get_str("output_format") or "json"
    output_file = ctx.params.get_str("output_file")
    include_columns = ctx.params.get_bool("include_columns")

    if output_format != "json":
        return fail_execution_failed(
            "build",
            f"Unsupported BuildSpec format: {output_format}",
            status=400,
        )

    spec = compile_buildspec(options=BuildSpecCompileOptions(include_columns=include_columns))
    payload = buildspec_to_json(spec, indent=2)

    if output_file and output_file != "-":
        Path(output_file).write_text(payload, encoding="utf-8")

    return CliResult.ok(
        payload,
        metadata={
            "target_count": len(spec.targets),
            "dataset_count": len(spec.datasets),
            "buildspec_hash": spec.buildspec_hash,
        },
    )


__all__ = [
    "build_spec_compile_handler",
]
