"""Tests for Iceberg scan helpers."""

from __future__ import annotations

from codeintel.core.config.settings import IcebergSettings
from codeintel.serving.semantic.iceberg_scans import resolve_iceberg_ref_for_identity


def test_resolve_iceberg_ref_for_identity_prefers_read_ref() -> None:
    """Prefer the read ref when read is disabled."""
    settings = IcebergSettings(read_enabled=False, read_ref="main")
    ref = resolve_iceberg_ref_for_identity(run_id="run-1", commit="abc", settings=settings)
    assert ref == "main"


def test_resolve_iceberg_ref_for_identity_prefers_run_id() -> None:
    """Prefer the run id ref when read is enabled."""
    settings = IcebergSettings(read_enabled=True)
    ref = resolve_iceberg_ref_for_identity(run_id="run-1", commit="abc", settings=settings)
    assert ref == "run/run-1"


def test_resolve_iceberg_ref_for_identity_falls_back_to_commit() -> None:
    """Fall back to commit ref when no run id is provided."""
    settings = IcebergSettings(read_enabled=True)
    ref = resolve_iceberg_ref_for_identity(run_id=None, commit="abc", settings=settings)
    assert ref == "commit/abc"
