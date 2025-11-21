"""CLI entrypoint for CodeIntel pipeline and document exports."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import duckdb

from codeintel.config.models import CodeIntelConfig
from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.orchestration.pipeline import run_pipeline
from codeintel.orchestration.steps import PipelineContext
from codeintel.storage.duckdb_client import DuckDBClient, DuckDBConfig
from codeintel.storage.views import create_all_views

LOG = logging.getLogger("codeintel.cli")


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
        default=Path("build/db/codeintel.duckdb"),
        help="Path to the DuckDB database (default: build/db/codeintel.duckdb)",
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
        help="Run pipeline steps up to one or more target steps (e.g. risk_factors, export_docs)",
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
    p_export.set_defaults(func=_cmd_docs_export)

    return parser


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _build_config_from_args(args: argparse.Namespace) -> CodeIntelConfig:
    return CodeIntelConfig.from_cli_args(
        repo_root=args.repo_root,
        db_path=args.db_path,
        build_dir=args.build_dir,
        document_output_dir=args.document_output_dir,
        repo=args.repo,
        commit=args.commit,
    )


def _open_connection(cfg: CodeIntelConfig, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection and ensure schemas/views exist for write mode.

    Parameters
    ----------
    cfg:
        CodeIntel configuration holding paths.
    read_only:
        Whether to open the database read-only.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Live DuckDB connection.
    """
    client = DuckDBClient(
        DuckDBConfig(
            db_path=cfg.paths.db_path,
            read_only=read_only,
        )
    )
    cfg.paths.db_dir.mkdir(parents=True, exist_ok=True)
    con = client.con
    if not read_only:
        create_all_views(con)
    return con


def _build_pipeline_context(cfg: CodeIntelConfig) -> PipelineContext:
    return PipelineContext(
        repo_root=cfg.paths.repo_root,
        db_path=cfg.paths.db_path,
        build_dir=cfg.paths.build_dir,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        extra={},
    )


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    con = _open_connection(cfg, read_only=False)

    ctx = _build_pipeline_context(cfg)

    targets: list[str]
    if args.target is None or len(args.target) == 0:
        targets = cfg.default_targets
    else:
        targets = list(args.target)

    LOG.info(
        "Running pipeline for repo=%s commit=%s targets=%s",
        cfg.repo.repo,
        cfg.repo.commit,
        targets,
    )

    run_pipeline(ctx, con, targets=targets)
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
    """
    cfg = _build_config_from_args(args)
    con = _open_connection(cfg, read_only=True)

    out_dir = cfg.document_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Exporting Parquet + JSONL datasets into %s", out_dir)
    export_all_parquet(con, out_dir)
    export_all_jsonl(con, out_dir)

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
        return int(args.func(args))  # type: ignore[misc]
    except Exception:
        LOG.exception("Command failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
