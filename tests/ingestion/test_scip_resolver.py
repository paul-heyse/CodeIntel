"""Tests for SCIP resolver utilities.

This module tests the SCIP resolver helper functions that normalize
SCIP ingestion inputs into a resolved configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.ingestion.infrastructure import (
    ResolvedScipConfig,
    ScipPathConfig,
    ScipResolverInput,
    resolve_scip_inputs,
)
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_is_none,
    expect_true,
)
from tests._helpers.ingestion import (
    build_scip_repo_paths,
    module_records_for_paths,
)

if TYPE_CHECKING:
    from pathlib import Path


EXPECTED_START_LINE = 1
EXPECTED_END_LINE = 1


def _scip_paths(tmp_path: Path) -> tuple[Path, Path]:
    context = build_scip_repo_paths(tmp_path)
    return context.repo_root, context.build_dir


def test_resolved_scip_config_create_minimal(tmp_path: Path) -> None:
    """Test creating ResolvedScipConfig with minimal required fields."""
    repo_root, build_dir = _scip_paths(tmp_path)
    doc_dir = build_dir / "docs"

    config = ResolvedScipConfig(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=doc_dir,
        scip_python_bin=None,
        modules=[],
    )

    expect_equal(config.repo, "test-org/test-repo")
    expect_equal(config.commit, "abc123")
    expect_equal(config.repo_root, repo_root)
    expect_equal(config.build_dir, build_dir)
    expect_equal(config.document_output_dir, doc_dir)
    expect_is_none(config.scip_python_bin)
    expect_equal(config.modules, [])


def test_resolved_scip_config_create_with_modules(tmp_path: Path) -> None:
    """Test creating ResolvedScipConfig with module records."""
    repo_root, build_dir = _scip_paths(tmp_path)
    module = module_records_for_paths(["pkg/mod.py"], repo_root)[0]

    config = ResolvedScipConfig(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=build_dir / "docs",
        scip_python_bin="/usr/bin/scip-python",
        modules=[module],
    )

    expect_equal(len(config.modules), 1)
    expect_equal(config.modules[0].module_name, "pkg.mod")
    expect_equal(config.scip_python_bin, "/usr/bin/scip-python")


def test_resolved_scip_config_frozen_dataclass(tmp_path: Path) -> None:
    """Test that ResolvedScipConfig is immutable."""
    config = ResolvedScipConfig(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=tmp_path,
        build_dir=tmp_path / "build",
        document_output_dir=tmp_path / "docs",
        scip_python_bin=None,
        modules=[],
    )

    assert_cannot_setattr(config, "repo", "new-repo")


def test_scip_resolver_input_create_empty() -> None:
    """Test creating ScipResolverInput with all defaults."""
    inputs = ScipResolverInput()

    expect_is_none(inputs.repo)
    expect_is_none(inputs.commit)
    expect_is_none(inputs.repo_root)
    expect_is_none(inputs.build_dir)
    expect_is_none(inputs.document_output_dir)
    expect_is_none(inputs.scip_python_bin)
    expect_is_none(inputs.modules)


def test_scip_resolver_input_create_with_explicit_params(tmp_path: Path) -> None:
    """Test creating ScipResolverInput with explicit parameters."""
    repo_root, build_dir = _scip_paths(tmp_path)
    doc_dir = build_dir / "docs"

    inputs = ScipResolverInput(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=doc_dir,
        scip_python_bin="/usr/bin/scip-python",
    )

    expect_equal(inputs.repo, "test-org/test-repo")
    expect_equal(inputs.commit, "abc123")
    expect_equal(inputs.repo_root, repo_root)
    expect_equal(inputs.build_dir, build_dir)
    expect_equal(inputs.document_output_dir, doc_dir)
    expect_equal(inputs.scip_python_bin, "/usr/bin/scip-python")


def test_scip_resolver_input_create_with_modules(tmp_path: Path) -> None:
    """Test creating ScipResolverInput with pre-computed modules."""
    repo_root, build_dir = _scip_paths(tmp_path)
    modules = module_records_for_paths(["pkg/mod.py"], repo_root)

    inputs = ScipResolverInput(
        repo="test-repo",
        commit="abc",
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=build_dir / "docs",
        modules=modules,
    )

    resolved_modules = inputs.modules
    if resolved_modules is None:
        pytest.fail("Expected modules to be populated")

    expect_equal(len(resolved_modules), 1)
    expect_equal(resolved_modules[0].module_name, "pkg.mod")


def test_scip_resolver_input_frozen_dataclass() -> None:
    """Test that ScipResolverInput is immutable."""
    inputs = ScipResolverInput(repo="test-repo")

    assert_cannot_setattr(inputs, "repo", "new-repo")


def test_resolve_scip_inputs_with_explicit_params(tmp_path: Path) -> None:
    """Test resolving with explicit ScipResolverInput.build()."""
    repo_root, build_dir = _scip_paths(tmp_path)
    doc_dir = build_dir / "docs"

    result = resolve_scip_inputs(
        [],
        ScipResolverInput.build(
            repo="test-org/test-repo",
            commit="abc123",
            paths=ScipPathConfig.from_strings(
                repo_root=repo_root,
                build_dir=build_dir,
                document_output_dir=doc_dir,
            ),
        ),
    )

    expect_true(isinstance(result, ResolvedScipConfig))
    expect_equal(result.repo, "test-org/test-repo")
    expect_equal(result.commit, "abc123")
    expect_equal(result.repo_root, repo_root)
    expect_equal(result.build_dir, build_dir)
    expect_equal(result.document_output_dir, doc_dir)


def test_resolve_scip_inputs_with_scip_resolver_input(tmp_path: Path) -> None:
    """Test resolving with ScipResolverInput dataclass."""
    repo_root, build_dir = _scip_paths(tmp_path)
    doc_dir = build_dir / "docs"

    inputs = ScipResolverInput(
        repo="test-org/test-repo",
        commit="def456",
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=doc_dir,
        scip_python_bin="/usr/bin/scip-python",
    )

    result = resolve_scip_inputs([], inputs)

    expect_equal(result.repo, "test-org/test-repo")
    expect_equal(result.commit, "def456")
    expect_equal(result.scip_python_bin, "/usr/bin/scip-python")


def test_resolve_scip_inputs_with_modules_sequence(tmp_path: Path) -> None:
    """Test resolving with modules passed as first argument."""
    repo_root, build_dir = _scip_paths(tmp_path)
    modules = module_records_for_paths(["pkg/mod.py"], repo_root)

    result = resolve_scip_inputs(
        modules,
        ScipResolverInput.build(
            repo="demo/repo",
            commit="deadbeef",
            paths=ScipPathConfig.from_strings(
                repo_root=repo_root,
                build_dir=build_dir,
                document_output_dir=build_dir / "docs",
            ),
        ),
    )

    expect_equal(len(result.modules), len(modules))
    expect_equal(result.modules[0].module_name, modules[0].module_name)


def test_resolve_scip_inputs_with_modules_in_input(tmp_path: Path) -> None:
    """Test resolving with modules passed via ScipResolverInput."""
    repo_root, build_dir = _scip_paths(tmp_path)
    modules = module_records_for_paths(["pkg/mod.py"], repo_root)

    result = resolve_scip_inputs(
        [],
        ScipResolverInput.build(
            repo="demo/repo",
            commit="deadbeef",
            paths=ScipPathConfig.from_strings(
                repo_root=repo_root,
                build_dir=build_dir,
                document_output_dir=build_dir / "docs",
            ),
            modules=modules,
        ),
    )

    expect_equal(len(result.modules), len(modules))
    expect_equal(result.modules[0].module_name, modules[0].module_name)


def test_resolve_scip_inputs_missing_repo_raises_value_error(tmp_path: Path) -> None:
    """Test that missing repo parameter raises ValueError."""
    with pytest.raises(ValueError, match=r"repo.*required"):
        resolve_scip_inputs(
            [],
            ScipResolverInput.build(
                commit="abc",
                paths=ScipPathConfig.from_strings(
                    repo_root=tmp_path,
                    build_dir=tmp_path / "build",
                    document_output_dir=tmp_path / "docs",
                ),
            ),
        )


def test_resolve_scip_inputs_missing_commit_raises_value_error(tmp_path: Path) -> None:
    """Test that missing commit parameter raises ValueError."""
    with pytest.raises(ValueError, match=r"commit.*required"):
        resolve_scip_inputs(
            [],
            ScipResolverInput.build(
                repo="test-repo",
                paths=ScipPathConfig.from_strings(
                    repo_root=tmp_path,
                    build_dir=tmp_path / "build",
                    document_output_dir=tmp_path / "docs",
                ),
            ),
        )


def test_resolve_scip_inputs_missing_repo_root_raises_value_error(tmp_path: Path) -> None:
    """Test that missing repo_root parameter raises ValueError."""
    with pytest.raises(ValueError, match=r"repo_root.*required"):
        resolve_scip_inputs(
            [],
            ScipResolverInput.build(
                repo="test-repo",
                commit="abc",
                paths=ScipPathConfig.from_strings(
                    build_dir=tmp_path / "build",
                    document_output_dir=tmp_path / "docs",
                ),
            ),
        )


def test_resolve_scip_inputs_missing_build_dir_raises_value_error(tmp_path: Path) -> None:
    """Test that missing build_dir parameter raises ValueError."""
    with pytest.raises(ValueError, match=r"build_dir.*required"):
        resolve_scip_inputs(
            [],
            ScipResolverInput.build(
                repo="test-repo",
                commit="abc",
                paths=ScipPathConfig.from_strings(
                    repo_root=tmp_path,
                    document_output_dir=tmp_path / "docs",
                ),
            ),
        )


def test_resolve_scip_inputs_missing_document_output_dir_raises_value_error(tmp_path: Path) -> None:
    """Test that missing document_output_dir parameter raises ValueError."""
    with pytest.raises(ValueError, match=r"document_output_dir.*required"):
        resolve_scip_inputs(
            [],
            ScipResolverInput.build(
                repo="test-repo",
                commit="abc",
                paths=ScipPathConfig.from_strings(
                    repo_root=tmp_path,
                    build_dir=tmp_path / "build",
                ),
            ),
        )


def test_resolve_scip_inputs_uses_inputs_values(tmp_path: Path) -> None:
    """Test that ScipResolverInput values are used correctly."""
    repo_root, build_dir = _scip_paths(tmp_path)
    inputs = ScipResolverInput(
        repo="inputs-repo",
        commit="inputs-commit",
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=build_dir / "inputs-docs",
    )

    result = resolve_scip_inputs([], inputs)

    expect_equal(result.repo, "inputs-repo")
    expect_equal(result.commit, "inputs-commit")


def test_module_record_create(tmp_path: Path) -> None:
    """Test creating a ModuleRecord with all fields."""
    repo_root, _build_dir = _scip_paths(tmp_path)
    module = module_records_for_paths(["pkg/mod.py"], repo_root)[0]

    expect_equal(module.rel_path, "pkg/mod.py")
    expect_equal(module.module_name, "pkg.mod")
    expect_equal(module.file_path, repo_root / "pkg/mod.py")
    expect_equal(module.index, EXPECTED_START_LINE)
    expect_equal(module.total, EXPECTED_END_LINE)


def test_module_record_frozen(tmp_path: Path) -> None:
    """Test that ModuleRecord is immutable."""
    module = module_records_for_paths(["main.py"], tmp_path)[0]

    assert_cannot_setattr(module, "module_name", "new_name")
