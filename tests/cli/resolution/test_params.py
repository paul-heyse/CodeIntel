"""Tests for RuntimeParams."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.commands import RuntimeCLI
from codeintel.cli.resolution import BackendFlags, RuntimeParams
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_none,
    expect_true,
)


def _force_setattr(target: object, name: str, value: object) -> None:
    """Attempt to set an attribute on a frozen dataclass for testing."""
    setattr(target, name, value)


def test_backend_flags_defaults() -> None:
    """Verify BackendFlags has expected defaults."""
    flags = BackendFlags()

    expect_true(not flags.use_gpu)
    expect_equal(flags.backend, "auto")
    expect_true(not flags.strict)


def test_runtime_params_defaults() -> None:
    """Verify RuntimeParams has expected defaults."""
    params = RuntimeParams()

    expect_is_none(params.project_root)
    expect_is_none(params.repo)
    expect_is_none(params.commit)
    expect_is_none(params.db_path)
    expect_true(not params.backend.use_gpu)


def test_runtime_params_minimal_factory() -> None:
    """Verify minimal() creates minimal params."""
    params = RuntimeParams.minimal(Path("/project"))

    expect_equal(params.project_root, Path("/project"))
    expect_is_none(params.repo)


def test_runtime_params_minimal_factory_no_args() -> None:
    """Verify minimal() works with no arguments."""
    params = RuntimeParams.minimal()

    expect_is_none(params.project_root)


def test_runtime_params_from_dict() -> None:
    """Verify from_dict creates params from dictionary."""
    data = {
        "project_root": "/project",
        "repo": "org/repo",
        "commit": "abc123",
        "db_path": "/db/test.duckdb",
        "backend": {"use_gpu": True},
    }

    params = RuntimeParams.from_dict(data)

    expect_equal(params.project_root, Path("/project"))
    expect_equal(params.repo, "org/repo")
    expect_equal(params.commit, "abc123")
    expect_equal(params.db_path, Path("/db/test.duckdb"))
    expect_true(params.backend.use_gpu)


def test_runtime_params_from_dict_use_gpu() -> None:
    """Verify from_dict honors top-level GPU flags."""
    data = {
        "use_gpu": True,
    }

    params = RuntimeParams.from_dict(data)

    expect_true(params.backend.use_gpu)
    expect_equal(params.backend.backend, "auto")


def test_runtime_params_from_dict_nx_flags() -> None:
    """Verify from_dict maps NetworkX flags to backend flags."""
    data = {
        "nx_backend": "nx-cugraph",
        "nx_gpu_mode": "strict",
    }

    params = RuntimeParams.from_dict(data)

    expect_equal(params.backend.backend, "nx-cugraph")
    expect_true(params.backend.use_gpu)
    expect_true(params.backend.strict)


def test_runtime_params_to_dict() -> None:
    """Verify to_dict creates dictionary from params."""
    params = RuntimeParams(
        project_root=Path("/project"),
        repo="org/repo",
        commit="abc123",
    )

    data = params.to_dict()

    expect_equal(data["project_root"], "/project")
    expect_equal(data["repo"], "org/repo")
    expect_equal(data["commit"], "abc123")


def test_runtime_params_from_cyclopts() -> None:
    """Verify from_cyclopts converts RuntimeCLI."""
    cli = RuntimeCLI(
        project_root=Path("/project"),
        repo="org/repo",
        commit="abc123",
    )

    params = RuntimeParams.from_cyclopts(cli)

    expect_equal(params.project_root, Path("/project"))
    expect_equal(params.repo, "org/repo")
    expect_equal(params.commit, "abc123")


def test_runtime_params_immutable() -> None:
    """Verify RuntimeParams is immutable."""
    params = RuntimeParams(repo="org/repo")

    with pytest.raises(AttributeError):
        _force_setattr(params, "repo", "other/repo")
