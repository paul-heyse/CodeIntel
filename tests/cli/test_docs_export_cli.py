"""Tests for docs export validator invocation."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.commands.docs import run_docs_export
from codeintel.config.models import (
    CliConfigOptions,
    CliPathsInput,
    CodeIntelConfig,
    RepoConfig,
)
from codeintel.export.runner import ExportOptions
from codeintel.storage.gateway import StorageGateway
from tests._helpers import provision_docs_export_ready


def test_docs_export_invokes_validator_before_exports(tmp_path: Path) -> None:
    """Docs export uses the provided validator before running exports."""
    events: list[str] = []

    # Provision the database with required seeds, then close the connection
    # so run_docs_export can open its own connection to the same file
    ctx = provision_docs_export_ready(tmp_path, db_path=tmp_path / "db.duckdb", file_backed=True)
    ctx.close()

    def validator(_gateway: StorageGateway) -> None:
        events.append("validator")

    def export_runner(
        *, gateway: StorageGateway, output_dir: Path, options: ExportOptions | None = None
    ) -> list[Path]:
        if options is None:
            pytest.fail("Expected options")
        options.validator(gateway)
        events.append(f"export:{output_dir}")
        return []

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths_cfg = CliPathsInput(
        repo_root=tmp_path,
        build_dir=tmp_path / "build",
        db_path=tmp_path / "db.duckdb",
        document_output_dir=out_dir,
    )
    repo_cfg = RepoConfig(repo="demo/repo", commit="deadbeef")
    cfg = CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=CliConfigOptions(),
    )

    run_docs_export(
        cfg=cfg,
        validate_exports=False,
        schemas=None,
        datasets=None,
        require_normalized_macros=False,
        validator=validator,
        export_runner=export_runner,
    )

    expected_events = ["validator", f"export:{out_dir}"]
    assert events == expected_events, f"Unexpected event order: {events}"
