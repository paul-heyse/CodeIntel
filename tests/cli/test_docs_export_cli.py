"""Tests for CLI docs export validator invocation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from codeintel.cli.main import cmd_docs_export
from codeintel.config.models import CodeIntelConfig
from codeintel.docs_export.runner import ExportOptions
from codeintel.storage.gateway import StorageGateway


class _StubGateway:
    """Minimal gateway stub."""

    con: object = object()


def _make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=tmp_path,
        repo="demo/repo",
        commit="deadbeef",
        db_path=tmp_path / "db.duckdb",
        build_dir=tmp_path / "build",
        document_output_dir=tmp_path / "out",
        schemas=None,
        datasets=None,
        validate_exports=False,
        nx_gpu=False,
        nx_backend="auto",
        verbose=0,
    )


def test_cmd_docs_export_invokes_validator_before_exports(tmp_path: Path) -> None:
    """CLI docs export uses the provided validator before running exports."""
    events: list[str] = []

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

    def gateway_factory(cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
        if not read_only:
            pytest.fail("Expected read-only gateway")
        _ = cfg  # unused in stub
        return _StubGateway()  # type: ignore[return-value]

    args = _make_args(tmp_path)
    # Ensure output dir exists for runner invocation.
    args.document_output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = cmd_docs_export(
        args,
        validator=validator,
        export_runner=export_runner,
        gateway_factory=gateway_factory,  # type: ignore[arg-type]
    )

    if exit_code != 0:
        pytest.fail(f"Unexpected exit: {exit_code}")
    expected_events = ["validator", f"export:{args.document_output_dir}"]
    if events != expected_events:
        pytest.fail(f"Unexpected event order: {events}")
