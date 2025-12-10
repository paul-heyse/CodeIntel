"""Tests for SCIP resolver utilities.

This module tests the SCIP resolver helper functions that normalize
SCIP ingestion inputs into a resolved configuration.
"""

from __future__ import annotations

from pathlib import Path

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
    RepoVariantOptions,
    build_ingestion_context_bundle,
    module_records_for_paths,
)

# Test constants for magic values
EXPECTED_START_LINE = 1
EXPECTED_END_LINE = 1


# --- ResolvedScipConfig Tests ---


def test_resolved_scip_config_create_minimal(tmp_path: Path) -> None:
    """Test creating ResolvedScipConfig with minimal required fields."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    build_dir = tmp_path / "build"
    doc_dir = tmp_path / "docs"

    config = ResolvedScipConfig(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=doc_dir,
        scip_python_bin=None,
        scip_bin=None,
        modules=[],
    )

    expect_equal(config.repo, "test-org/test-repo")
    expect_equal(config.commit, "abc123")
    expect_equal(config.repo_root, repo_root)
    expect_equal(config.build_dir, build_dir)
    expect_equal(config.document_output_dir, doc_dir)
    expect_is_none(config.scip_python_bin)
    expect_is_none(config.scip_bin)
    expect_equal(config.modules, [])


def test_resolved_scip_config_create_with_modules(tmp_path: Path) -> None:
    """Test creating ResolvedScipConfig with module records."""
    context = build_ingestion_context_bundle(
        tmp_path,
        variants=RepoVariantOptions(
            repo_structure={"src/main.py": "def main() -> None:\n    pass\n"}
        ),
    )
    module = module_records_for_paths(["src/main.py"], context.repo_root)[0]

    config = ResolvedScipConfig(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=context.repo_root,
        build_dir=tmp_path / "build",
        document_output_dir=tmp_path / "docs",
        scip_python_bin="/usr/bin/scip-python",
        scip_bin="/usr/bin/scip",
        modules=[module],
    )

    expect_equal(len(config.modules), 1)
    expect_equal(config.modules[0].module_name, "src.main")
    expect_equal(config.scip_python_bin, "/usr/bin/scip-python")
    expect_equal(config.scip_bin, "/usr/bin/scip")


def test_resolved_scip_config_frozen_dataclass(tmp_path: Path) -> None:
    """Test that ResolvedScipConfig is immutable."""
    config = ResolvedScipConfig(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=tmp_path,
        build_dir=tmp_path / "build",
        document_output_dir=tmp_path / "docs",
        scip_python_bin=None,
        scip_bin=None,
        modules=[],
    )

    assert_cannot_setattr(config, "repo", "new-repo")


# --- ScipResolverInput Tests ---


def test_scip_resolver_input_create_empty() -> None:
    """Test creating ScipResolverInput with all defaults."""
    inputs = ScipResolverInput()

    expect_is_none(inputs.repo)
    expect_is_none(inputs.commit)
    expect_is_none(inputs.repo_root)
    expect_is_none(inputs.build_dir)
    expect_is_none(inputs.document_output_dir)
    expect_is_none(inputs.scip_python_bin)
    expect_is_none(inputs.scip_bin)
    expect_is_none(inputs.modules)


def test_scip_resolver_input_create_with_explicit_params(tmp_path: Path) -> None:
    """Test creating ScipResolverInput with explicit parameters."""
    repo_root = tmp_path / "repo"
    build_dir = tmp_path / "build"
    doc_dir = tmp_path / "docs"

    inputs = ScipResolverInput(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=doc_dir,
        scip_python_bin="/usr/bin/scip-python",
        scip_bin="/usr/bin/scip",
    )

    expect_equal(inputs.repo, "test-org/test-repo")
    expect_equal(inputs.commit, "abc123")
    expect_equal(inputs.repo_root, repo_root)
    expect_equal(inputs.build_dir, build_dir)
    expect_equal(inputs.document_output_dir, doc_dir)
    expect_equal(inputs.scip_python_bin, "/usr/bin/scip-python")
    expect_equal(inputs.scip_bin, "/usr/bin/scip")


def test_scip_resolver_input_create_with_modules(tmp_path: Path) -> None:
    """Test creating ScipResolverInput with pre-computed modules."""
    modules = module_records_for_paths(["main.py"], tmp_path)

    inputs = ScipResolverInput(
        repo="test-repo",
        commit="abc",
        repo_root=tmp_path,
        build_dir=tmp_path / "build",
        document_output_dir=tmp_path / "docs",
        modules=modules,
    )

    modules = inputs.modules
    if modules is None:
        pytest.fail("Expected modules to be populated")

    expect_equal(len(modules), 1)
    expect_equal(modules[0].module_name, "main")


def test_scip_resolver_input_frozen_dataclass() -> None:
    """Test that ScipResolverInput is immutable."""
    inputs = ScipResolverInput(repo="test-repo")

    assert_cannot_setattr(inputs, "repo", "new-repo")


# --- resolve_scip_inputs Tests ---


def test_resolve_scip_inputs_with_explicit_params(tmp_path: Path) -> None:
    """Test resolving with explicit ScipResolverInput.build()."""
    context = build_ingestion_context_bundle(tmp_path)
    repo_root = context.repo_root
    build_dir = tmp_path / "build"
    doc_dir = tmp_path / "docs"

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
    context = build_ingestion_context_bundle(tmp_path)
    repo_root = context.repo_root
    build_dir = tmp_path / "build"
    doc_dir = tmp_path / "docs"

    inputs = ScipResolverInput(
        repo="test-org/test-repo",
        commit="def456",
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=doc_dir,
        scip_python_bin="/usr/bin/scip-python",
        scip_bin="/usr/bin/scip",
    )

    result = resolve_scip_inputs([], inputs)

    expect_equal(result.repo, "test-org/test-repo")
    expect_equal(result.commit, "def456")
    expect_equal(result.scip_python_bin, "/usr/bin/scip-python")
    expect_equal(result.scip_bin, "/usr/bin/scip")


def test_resolve_scip_inputs_with_modules_sequence(tmp_path: Path) -> None:
    """Test resolving with modules passed as first argument."""
    context = build_ingestion_context_bundle(tmp_path)
    repo_root = context.repo_root
    modules = module_records_for_paths(context.module_paths, repo_root)

    result = resolve_scip_inputs(
        modules,
        ScipResolverInput.build(
            repo=context.ctx.repo,
            commit=context.ctx.commit,
            paths=ScipPathConfig.from_strings(
                repo_root=repo_root,
                build_dir=context.ctx.build_dir,
                document_output_dir=context.ctx.build_dir / "docs",
            ),
        ),
    )

    expect_equal(len(result.modules), len(modules))
    expect_equal(result.modules[0].module_name, modules[0].module_name)


def test_resolve_scip_inputs_with_modules_in_input(tmp_path: Path) -> None:
    """Test resolving with modules passed via ScipResolverInput."""
    context = build_ingestion_context_bundle(tmp_path)
    repo_root = context.repo_root
    modules = module_records_for_paths(context.module_paths, repo_root)

    result = resolve_scip_inputs(
        [],  # Empty, modules in input takes precedence
        ScipResolverInput.build(
            repo=context.ctx.repo,
            commit=context.ctx.commit,
            paths=ScipPathConfig.from_strings(
                repo_root=repo_root,
                build_dir=context.ctx.build_dir,
                document_output_dir=context.ctx.build_dir / "docs",
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
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    inputs = ScipResolverInput(
        repo="inputs-repo",
        commit="inputs-commit",
        repo_root=repo_root,
        build_dir=tmp_path / "inputs-build",
        document_output_dir=tmp_path / "inputs-docs",
    )

    result = resolve_scip_inputs([], inputs)

    expect_equal(result.repo, "inputs-repo")
    expect_equal(result.commit, "inputs-commit")


# --- ModuleRecord Tests ---


def test_module_record_create(tmp_path: Path) -> None:
    """Test creating a ModuleRecord with all fields."""
    repo_root = build_ingestion_context_bundle(tmp_path).repo_root
    module = module_records_for_paths(["src/module.py"], repo_root)[0]

    expect_equal(module.rel_path, "src/module.py")
    expect_equal(module.module_name, "src.module")
    expect_equal(module.file_path, repo_root / "src/module.py")
    expect_equal(module.index, EXPECTED_START_LINE)
    expect_equal(module.total, EXPECTED_END_LINE)


def test_module_record_frozen(tmp_path: Path) -> None:
    """Test that ModuleRecord is immutable."""
    module = module_records_for_paths(["main.py"], tmp_path)[0]

    assert_cannot_setattr(module, "module_name", "new_name")
