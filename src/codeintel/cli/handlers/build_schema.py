"""Handlers for build schema commands."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest
from codeintel.build.schemas.provider_declared import declared_schema_provider
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import (
    fail_execution_failed,
    fail_file_not_found,
    fail_invalid_module,
    fail_invalid_targets,
    fail_missing_required,
)

if TYPE_CHECKING:
    from codeintel.build.targets import TargetModule
    from codeintel.cli.context import CommandContext


_VALID_MODULES: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics", "export")


@dataclass(frozen=True)
class _SchemaSelection:
    targets: tuple[str, ...] | None
    module: TargetModule | None
    all_targets: bool
    only_native: bool
    infer_native: bool
    stable: bool


class _InvalidModuleError(ValueError):
    def __init__(self, module: str) -> None:
        super().__init__(module)
        self.module = module


def _parse_module(module_raw: str | None) -> TargetModule | None:
    if module_raw is None:
        return None
    if module_raw not in _VALID_MODULES:
        raise _InvalidModuleError(module_raw)
    return cast("TargetModule", module_raw)


def _parse_selection(ctx: CommandContext) -> _SchemaSelection:
    targets_list = ctx.params.get_list("targets")
    targets: tuple[str, ...] | None = tuple(targets_list) if targets_list else None
    module = _parse_module(ctx.params.get_str("module"))
    return _SchemaSelection(
        targets=targets,
        module=module,
        all_targets=ctx.params.get_bool("all_targets"),
        only_native=ctx.params.get_bool("only_native"),
        infer_native=ctx.params.get_bool("infer_native"),
        stable=ctx.params.get_bool("stable"),
    )


@dataclass(frozen=True)
class _CompiledManifest:
    payload: str
    table_count: int


def _compile_manifest(ctx: CommandContext) -> _CompiledManifest:
    selection = _parse_selection(ctx)
    request = SchemaManifestRequest(
        targets=selection.targets,
        module=selection.module,
        all_targets=selection.all_targets,
        only_native=selection.only_native,
        infer_native=selection.infer_native,
        stable=selection.stable,
    )
    manifest = compile_schema_manifest(
        provider=declared_schema_provider(),
        request=request,
    )
    payload = json.dumps(manifest.to_json_obj(), indent=2, sort_keys=True) + "\n"
    return _CompiledManifest(payload=payload, table_count=len(manifest.tables))


def build_schema_compile_handler(ctx: CommandContext) -> CliResult[str]:
    """Compile a schema manifest for a target selection.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[str]
        JSON schema manifest payload.
    """
    output_format = ctx.params.get_str("output_format") or "json"
    output_file = ctx.params.get_str("output_file")

    if output_format != "json":
        return fail_execution_failed(
            "build",
            f"Unsupported schema manifest format: {output_format}",
            status=400,
        )

    try:
        compiled = _compile_manifest(ctx)
    except _InvalidModuleError as exc:
        return fail_invalid_module(exc.module, _VALID_MODULES)
    except KeyError as exc:
        return fail_invalid_targets(str(exc))
    except (RuntimeError, TypeError, ValueError) as exc:
        return fail_execution_failed("build", str(exc), status=500)

    payload = compiled.payload
    if output_file and output_file != "-":
        Path(output_file).write_text(payload, encoding="utf-8")

    return CliResult.ok(payload, metadata={"table_count": compiled.table_count})


def build_schema_diff_handler(ctx: CommandContext) -> CliResult[str]:
    """Compare a compiled schema manifest to an expected manifest file.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[str]
        Success message when manifests match; otherwise a drift diff failure.
    """
    expected_file = ctx.params.get_str("expected_file")
    if not expected_file:
        return fail_missing_required("expected")

    expected_path = Path(expected_file)
    if not expected_path.exists():
        return fail_file_not_found(str(expected_path), domain="build")

    try:
        expected_obj = json.loads(expected_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return fail_execution_failed(
            "build",
            f"Expected manifest is not valid JSON: {exc}",
            status=400,
        )
    if not isinstance(expected_obj, dict):
        return fail_execution_failed(
            "build",
            "Expected manifest must be a JSON object",
            status=400,
        )

    result: CliResult[str] = fail_execution_failed(
        "build",
        "Schema diff failed unexpectedly",
        status=500,
    )

    try:
        compiled = _compile_manifest(ctx)
    except _InvalidModuleError as exc:
        result = fail_invalid_module(exc.module, _VALID_MODULES)
    except KeyError as exc:
        result = fail_invalid_targets(str(exc))
    except (RuntimeError, TypeError, ValueError) as exc:
        result = fail_execution_failed("build", str(exc), status=500)
    else:
        expected_json = json.dumps(expected_obj, indent=2, sort_keys=True) + "\n"
        actual_json = compiled.payload

        if expected_json == actual_json:
            result = CliResult.ok("Schema manifest matches expected.\n")
        else:
            diff = "".join(
                difflib.unified_diff(
                    expected_json.splitlines(keepends=True),
                    actual_json.splitlines(keepends=True),
                    fromfile=f"expected:{expected_path}",
                    tofile="actual",
                )
            )
            message = (
                "Schema drift detected (expected vs actual).\n\n"
                f"{diff}\n"
                "Update the expected manifest by re-running:\n"
                f"  codeintel build schema compile --output {expected_path}\n"
            )
            result = fail_execution_failed("build", message, status=409)

    return result


__all__ = [
    "build_schema_compile_handler",
    "build_schema_diff_handler",
]
