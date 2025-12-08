"""Regression tests for ingestion context helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.ingestion_context import (
    LegacyTargetContextOptions,
    RecordingGateway,
    make_target_context,
    make_target_context_from_modules,
)


def test_make_target_context_accepts_overrides(tmp_path: Path) -> None:
    """Ensure overrides populate context fields correctly."""
    gateway = RecordingGateway()
    ctx = make_target_context(
        repo_root=tmp_path,
        modules=("a.py", "b.py"),
        gateway=gateway,
        type_checker="checker",
        snapshot=("repo-x", "sha-y"),
        use_real_gateway=False,
        tmp_path=tmp_path / "alt",
    )

    expect_equal(ctx.repo_root, tmp_path)
    expect_equal(ctx.repo, "repo-x")
    expect_equal(ctx.commit, "sha-y")
    expect_true(ctx.gateway is gateway)
    expect_true(ctx.resources.gateway is gateway)
    expect_equal(ctx.resources.modules, ("a.py", "b.py"))
    expect_equal(ctx.resources.type_checker, "checker")


def test_make_target_context_rejects_mixed_options_and_overrides(tmp_path: Path) -> None:
    """Passing both options and overrides should fail fast."""
    opts = LegacyTargetContextOptions()
    with pytest.raises(ValueError, match="either options"):
        make_target_context(repo_root=tmp_path, options=opts, modules=("x.py",))


def test_make_target_context_rejects_unknown_override(tmp_path: Path) -> None:
    """Unknown override keys must raise to keep API strict."""
    with pytest.raises(ValueError, match="Unexpected overrides"):
        make_target_context(repo_root=tmp_path, bogus="nope")  # type: ignore[arg-type]


def test_make_target_context_from_modules_shortcut(tmp_path: Path) -> None:
    """Shortcut helper should forward modules and snapshot."""
    ctx = make_target_context_from_modules(
        repo_root=tmp_path,
        modules=("pkg/mod.py",),
        snapshot=("repo1", "sha1"),
        use_real_gateway=False,
        gateway=RecordingGateway(),
    )

    expect_equal(ctx.repo, "repo1")
    expect_equal(ctx.commit, "sha1")
    expect_equal(ctx.resources.modules, ("pkg/mod.py",))
    expect_true(isinstance(ctx.gateway, RecordingGateway))
    expect_true(ctx.resources.gateway is ctx.gateway)
