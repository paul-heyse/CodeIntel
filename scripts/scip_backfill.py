"""Backfill SCIP tables and validate row counts for a repo snapshot."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.storage.queries.safe import safe_count_with_scope

from codeintel.build.config import load_build_config
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.executor import HamiltonBuildExecutor
from codeintel.build.providers import create_default_providers
from codeintel.build.settings import get_build_settings
from codeintel.cli.resolution.errors import ResolutionError
from codeintel.cli.resolution.runtime import resolve_from_params
from codeintel.core.errors.storage import StorageConnectionError
from codeintel.core.execution import new_run_context
from codeintel.core.runtime.loader import load_execution_context
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.factory import open_gateway

if TYPE_CHECKING:
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)

SCIP_TABLES = (
    "core.scip_symbols",
    "core.scip_occurrences",
    "core.scip_symbol_information",
    "core.scip_symbol_relationships",
    "core.scip_diagnostics",
    "core.scip_external_symbols",
    "core.scip_module_state",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SCIP datasets and validate row counts.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd).",
    )
    parser.add_argument("--repo", type=str, default=None, help="Repository slug.")
    parser.add_argument("--commit", type=str, default=None, help="Commit SHA.")
    parser.add_argument("--db-path", type=Path, default=None, help="DuckDB path override.")
    parser.add_argument("--build-dir", type=Path, default=None, help="Build directory.")
    parser.add_argument(
        "--min-symbols",
        type=int,
        default=1,
        help="Minimum core.scip_symbols rows.",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=1,
        help="Minimum core.scip_occurrences rows.",
    )
    parser.add_argument(
        "--min-symbol-info",
        type=int,
        default=1,
        help="Minimum core.scip_symbol_information rows.",
    )
    parser.add_argument(
        "--min-diagnostics",
        type=int,
        default=0,
        help="Minimum core.scip_diagnostics rows.",
    )
    parser.add_argument(
        "--min-diagnostic-ratio",
        type=float,
        default=0.0,
        help="Minimum diagnostics/occurrences ratio.",
    )
    parser.add_argument(
        "--require-module-state",
        action="store_true",
        help="Fail if core.scip_module_state is empty.",
    )
    parser.add_argument(
        "--validate-outputs",
        action="store_true",
        help="Enable Pandera validation for produced datasets.",
    )
    parser.add_argument(
        "--strict-contracts",
        action="store_true",
        help="Enable strict contract validation for the build.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _resolve_runtime(args: argparse.Namespace) -> ResolvedRuntime:
    params: dict[str, object] = {"project_root": args.repo_root}
    if args.repo is not None:
        params["repo"] = args.repo
    if args.commit is not None:
        params["commit"] = args.commit
    if args.db_path is not None:
        params["db_path"] = args.db_path
    if args.build_dir is not None:
        params["build_dir"] = args.build_dir
    return resolve_from_params(params)


def _build_env(
    runtime: ResolvedRuntime,
    args: argparse.Namespace,
    *,
    gateway: StorageGateway,
) -> BuildEnv:
    providers = create_default_providers(runtime.tools)
    config = load_build_config(runtime.snapshot.repo_root)
    settings = get_build_settings()
    run_context = new_run_context(
        snapshot=runtime.snapshot,
        kind="full",
        trigger="scip_backfill",
        requested_datasets=("scip",),
    )
    execution_context = load_execution_context(primitives=runtime.primitives, run=run_context)
    manifests = gateway.build.list_manifests(
        repo=runtime.snapshot.repo,
        commit=runtime.snapshot.commit,
    )
    manifest_index = {manifest.target: manifest for manifest in manifests}
    return BuildEnv(
        gateway=gateway,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        providers=providers,
        config=config,
        settings=settings,
        execution_context=execution_context,
        manifest_index=manifest_index,
        validate_outputs=args.validate_outputs,
        strict_contracts=args.strict_contracts,
    )


def _validate_row_counts(
    *,
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    args: argparse.Namespace,
) -> int:
    expectations = {
        "core.scip_symbols": args.min_symbols,
        "core.scip_occurrences": args.min_occurrences,
        "core.scip_symbol_information": args.min_symbol_info,
        "core.scip_diagnostics": args.min_diagnostics,
    }
    failures: list[str] = []
    counts: dict[str, int | None] = {}
    for table_key in SCIP_TABLES:
        count = safe_count_with_scope(gateway, table_key, snapshot)
        counts[table_key] = count
        LOG.info("scip.backfill.count %s=%s", table_key, count)
    for table_key, minimum in expectations.items():
        count = counts.get(table_key)
        if count is None:
            failures.append(f"{table_key} count unavailable")
            continue
        if count < minimum:
            failures.append(f"{table_key} has {count} rows (< {minimum})")
    module_state = counts.get("core.scip_module_state")
    if args.require_module_state and (module_state is None or module_state == 0):
        failures.append("core.scip_module_state is empty")
    diagnostic_ratio = _diagnostic_ratio(counts)
    if diagnostic_ratio is not None and diagnostic_ratio < args.min_diagnostic_ratio:
        failures.append(
            f"diagnostic ratio {diagnostic_ratio:.4f} < {args.min_diagnostic_ratio:.4f}"
        )
    if failures:
        for failure in failures:
            LOG.error("scip.backfill.validation_failed %s", failure)
        return 1
    return 0


def _diagnostic_ratio(counts: dict[str, int | None]) -> float | None:
    occurrences = counts.get("core.scip_occurrences")
    diagnostics = counts.get("core.scip_diagnostics")
    if occurrences is None or diagnostics is None:
        return None
    if occurrences == 0:
        return 0.0 if diagnostics == 0 else None
    return diagnostics / occurrences


def _validate_artifacts(scip_dir: Path) -> int:
    failures: list[str] = []
    index_path = scip_dir / "index.scip"
    proto_path = scip_dir / "proto" / "scip_pb2.py"
    if not index_path.is_file():
        failures.append(f"Missing index.scip at {index_path}")
    if not proto_path.is_file():
        failures.append(f"Missing scip_pb2.py at {proto_path}")
    if failures:
        for failure in failures:
            LOG.error("scip.backfill.artifact_missing %s", failure)
        return 1
    return 0


def main() -> int:
    """Run the SCIP backfill and validation flow.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        runtime = _resolve_runtime(args)
    except ResolutionError:
        LOG.exception("scip.backfill.runtime_resolution_failed")
        return 1

    storage_config = StorageConfig.for_ingest(runtime.paths.db_path)
    try:
        gateway = open_gateway(storage_config)
    except StorageConnectionError:
        LOG.exception("scip.backfill.gateway_open_failed")
        return 1
    try:
        env = _build_env(runtime, args, gateway=gateway)
        executor = HamiltonBuildExecutor(profile=runtime.project.default_profile)
        result = executor.run(env=env, targets=["scip"])
        if not result.success or result.failed_targets:
            LOG.error("scip.backfill.build_failed %s", result.failed_targets)
            return 1
        artifact_status = _validate_artifacts(env.paths.scip_dir)
        if artifact_status != 0:
            return artifact_status
        return _validate_row_counts(gateway=gateway, snapshot=runtime.snapshot, args=args)
    finally:
        gateway.close()


if __name__ == "__main__":
    sys.exit(main())
