"""CLI entrypoint for CodeIntel pipeline and document exports."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from codeintel.config.models import (
    CodeIntelConfig,
    FunctionAnalyticsOverrides,
    PathsConfig,
    RepoConfig,
)
from codeintel.config.parser_types import FunctionParserKind
from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.ingestion.source_scanner import ScanConfig
from codeintel.mcp.backend import DuckDBBackend
from codeintel.orchestration.prefect_flow import ExportArgs, export_docs_flow
from codeintel.services.errors import ExportError, log_problem, problem
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

LOG = logging.getLogger("codeintel.cli")
CommandHandler = Callable[[argparse.Namespace], int]


# ---------------------------------------------------------------------------
# Argument parsing / logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbosity: int) -> None:
    """
    Configure logging based on -v/--verbose count.

    0 -> WARNING, 1 -> INFO, 2+ -> DEBUG.
    """
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _add_common_repo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(),
        help="Path to the repository root (default: current directory)",
    )
    p.add_argument(
        "--repo",
        required=True,
        help="Repository slug (used in GOIDs and exports, e.g. 'my-org/my-repo')",
    )
    p.add_argument(
        "--commit",
        required=True,
        help="Commit SHA for this analysis run (embedded into GOIDs)",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=Path("build/db/codeintel_prefect.duckdb"),
        help="Path to the DuckDB database (default: build/db/codeintel_prefect.duckdb)",
    )
    p.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Build directory (default: build/)",
    )
    p.add_argument(
        "--document-output-dir",
        type=Path,
        default=None,
        help="Override Document Output/ directory (default: <repo-root>/Document Output)",
    )


def _add_subsystem_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register subsystem-related subcommands."""
    p_subsys = subparsers.add_parser("subsystem", help="Subsystem exploration helpers")
    subsys_sub = p_subsys.add_subparsers(dest="subcommand", required=True)

    p_subsys_list = subsys_sub.add_parser(
        "list",
        help="List inferred subsystems with role/risk metadata.",
    )
    _add_common_repo_args(p_subsys_list)
    p_subsys_list.add_argument("--role", help="Filter subsystems by role tag")
    p_subsys_list.add_argument("--q", help="Search substring on name/description")
    p_subsys_list.add_argument("--limit", type=int, default=None, help="Limit subsystem count")
    p_subsys_list.set_defaults(func=_cmd_subsystem_list)

    p_subsys_show = subsys_sub.add_parser(
        "show",
        help="Show subsystem detail and modules.",
    )
    _add_common_repo_args(p_subsys_show)
    p_subsys_show.add_argument("--subsystem-id", required=True, help="Subsystem identifier")
    p_subsys_show.set_defaults(func=_cmd_subsystem_show)

    p_subsys_modules = subsys_sub.add_parser(
        "module-memberships",
        help="List subsystem memberships for a module.",
    )
    _add_common_repo_args(p_subsys_modules)
    p_subsys_modules.add_argument("--module", required=True, help="Module name (e.g., pkg.mod)")
    p_subsys_modules.set_defaults(func=_cmd_subsystem_modules)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeintel",
        description="CodeIntel metadata pipeline and export CLI",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------------------------------------------------------------
    # pipeline run
    # -----------------------------------------------------------------------
    p_pipeline = subparsers.add_parser("pipeline", help="Run enrichment pipeline")
    pipeline_sub = p_pipeline.add_subparsers(dest="subcommand", required=True)

    p_run = pipeline_sub.add_parser(
        "run",
        help="Run the full pipeline via Prefect (targets are ignored; runs export_docs_flow).",
    )
    _add_common_repo_args(p_run)
    p_run.add_argument(
        "--target",
        action="append",
        default=None,
        help=(
            "Name of a pipeline step to run (can be specified multiple times). "
            "Defaults to 'export_docs' if omitted."
        ),
    )
    p_run.add_argument(
        "--skip-scip",
        action="store_true",
        help="Skip SCIP ingestion (sets CODEINTEL_SKIP_SCIP=true).",
    )
    p_run.add_argument(
        "--function-fail-on-missing-spans",
        action="store_true",
        help="Fail pipeline when function spans are missing or files cannot be parsed.",
    )
    p_run.add_argument(
        "--function-parser",
        choices=[kind.value for kind in FunctionParserKind],
        help="Optional parser selector for function analytics (e.g., 'python').",
    )
    p_run.set_defaults(func=_cmd_pipeline_run)

    # -----------------------------------------------------------------------
    # docs export
    # -----------------------------------------------------------------------
    p_docs = subparsers.add_parser("docs", help="Document Output helpers")
    docs_sub = p_docs.add_subparsers(dest="subcommand", required=True)

    p_export = docs_sub.add_parser(
        "export",
        help="Export Parquet + JSONL datasets from DuckDB into Document Output/",
    )
    _add_common_repo_args(p_export)
    p_export.add_argument(
        "--validate",
        dest="validate_exports",
        action="store_true",
        help="Validate exported datasets against JSON Schema definitions.",
    )
    p_export.add_argument(
        "--schema",
        dest="schemas",
        action="append",
        help="Schema name to validate (can be repeated). Defaults to the standard export set.",
    )
    p_export.set_defaults(func=_cmd_docs_export)

    # -----------------------------------------------------------------------
    # IDE helpers
    # -----------------------------------------------------------------------
    p_ide = subparsers.add_parser("ide", help="IDE helper commands")
    ide_sub = p_ide.add_subparsers(dest="subcommand", required=True)

    p_ide_hints = ide_sub.add_parser(
        "hints",
        help="Emit IDE hints (module + subsystem context) for a relative file path.",
    )
    _add_common_repo_args(p_ide_hints)
    p_ide_hints.add_argument(
        "--rel-path",
        required=True,
        help="File path relative to repo root (e.g., pkg/module.py)",
    )
    p_ide_hints.set_defaults(func=_cmd_ide_hints)

    _add_subsystem_subparser(subparsers)

    return parser


def make_parser() -> argparse.ArgumentParser:
    """
    Public helper to construct the CLI parser (for tests/tools).

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all subcommands registered.
    """
    return _make_parser()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _build_config_from_args(args: argparse.Namespace) -> CodeIntelConfig:
    paths_cfg = PathsConfig(
        repo_root=args.repo_root,
        build_dir=args.build_dir,
        db_path=args.db_path,
        document_output_dir=args.document_output_dir,
    )
    repo_cfg = RepoConfig(repo=args.repo, commit=args.commit)
    return CodeIntelConfig.from_cli_args(repo_cfg=repo_cfg, paths_cfg=paths_cfg)


def _open_gateway(cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
    """
    Open a StorageGateway and ensure schemas/views exist for write mode.

    Parameters
    ----------
    cfg:
        CodeIntel configuration holding paths.
    read_only:
        Whether to open the database read-only.

    Returns
    -------
    StorageGateway
        Gateway bound to the configured DuckDB database.
    """
    cfg.paths.db_dir.mkdir(parents=True, exist_ok=True)
    gateway_cfg = StorageConfig(
        db_path=cfg.paths.db_path,
        read_only=read_only,
        apply_schema=not read_only,
        ensure_views=not read_only,
        validate_schema=not read_only,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
    )
    return open_gateway(gateway_cfg)


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    repo_root = cfg.paths.repo_root
    db_path = cfg.paths.db_path
    build_dir = cfg.paths.build_dir
    output_dir = cfg.paths.document_output_dir
    if output_dir is not None:
        os.environ.setdefault("CODEINTEL_OUTPUT_DIR", str(output_dir))

    if args.skip_scip:
        os.environ["CODEINTEL_SKIP_SCIP"] = "true"

    parser_kind = FunctionParserKind(args.function_parser) if args.function_parser else None
    overrides = FunctionAnalyticsOverrides(
        fail_on_missing_spans=args.function_fail_on_missing_spans,
        parser=parser_kind,
    )

    targets = list(args.target) if args.target else None
    LOG.info(
        "Running Prefect export_docs_flow for repo=%s commit=%s targets=%s",
        cfg.repo.repo,
        cfg.repo.commit,
        targets,
    )

    export_docs_flow(
        args=ExportArgs(
            repo_root=repo_root,
            repo=cfg.repo.repo,
            commit=cfg.repo.commit,
            db_path=db_path,
            build_dir=build_dir,
            tools=cfg.tools,
            scan_config=ScanConfig(repo_root=repo_root),
            function_overrides=overrides,
        ),
        targets=targets,
    )
    return 0


def _cmd_ide_hints(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    backend = DuckDBBackend(gateway=gateway, repo=cfg.repo.repo, commit=cfg.repo.commit)
    response = backend.get_file_hints(rel_path=args.rel_path)
    if not response.found or not response.hints:
        LOG.error("No IDE hints found for %s", args.rel_path)
        return 1

    payload = {
        "rel_path": args.rel_path,
        "hints": [hint.model_dump() for hint in response.hints],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_list(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    backend = DuckDBBackend(gateway=gateway, repo=cfg.repo.repo, commit=cfg.repo.commit)
    response = backend.list_subsystems(
        limit=args.limit,
        role=args.role,
        q=args.q,
    )
    payload = {
        "subsystems": [row.model_dump() for row in response.subsystems],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_show(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    backend = DuckDBBackend(gateway=gateway, repo=cfg.repo.repo, commit=cfg.repo.commit)
    response = backend.get_subsystem_modules(subsystem_id=args.subsystem_id)
    if not response.found or response.subsystem is None:
        LOG.error("Subsystem not found: %s", args.subsystem_id)
        return 1
    payload = {
        "subsystem": response.subsystem.model_dump(),
        "modules": [row.model_dump() for row in response.modules],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_modules(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    backend = DuckDBBackend(gateway=gateway, repo=cfg.repo.repo, commit=cfg.repo.commit)
    response = backend.get_module_subsystems(module=args.module)
    payload = {
        "found": response.found,
        "memberships": [row.model_dump() for row in response.memberships],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_docs_export(args: argparse.Namespace) -> int:
    """
    Export only, assuming the pipeline has already populated the DuckDB db.

    This is what `scripts/generate_documents.sh` should call:

        codeintel docs export --repo <slug> --commit <sha>

    Returns
    -------
    int
        Exit code (0 on success).

    Raises
    ------
    RuntimeError
        If document_output_dir could not be resolved.
    """
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)

    out_dir = cfg.paths.document_output_dir
    if out_dir is None:
        message = "document_output_dir was not resolved"
        raise RuntimeError(message)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Exporting Parquet + JSONL datasets into %s", out_dir)
    schemas = list(args.schemas) if getattr(args, "schemas", None) else None
    try:
        export_all_parquet(
            gateway,
            out_dir,
            validate_exports=bool(getattr(args, "validate_exports", False)),
            schemas=schemas,
        )
        export_all_jsonl(
            gateway,
            out_dir,
            validate_exports=bool(getattr(args, "validate_exports", False)),
            schemas=schemas,
        )
    except ExportError as exc:
        log_problem(LOG, exc.problem_detail)
        return 1

    LOG.info("Export complete.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    """
    CLI entrypoint for the CodeIntel pipeline and export commands.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to sys.argv).

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = _make_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    _setup_logging(args.verbose)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        func: CommandHandler = args.func
        return int(func(args))
    except Exception as exc:  # noqa: BLE001 pragma: no cover - error path
        pd = problem(
            code="cli.failure",
            title="CLI command failed",
            detail=str(exc),
            extras={"command": args.command},
        )
        log_problem(LOG, pd)
        return 1


if __name__ == "__main__":
    sys.exit(main())
