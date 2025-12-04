"""Tests for SCIP resolver utilities.

This module tests the SCIP resolver helper functions that normalize
SCIP ingestion inputs into a resolved configuration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.ingestion.utilities import (
    ResolvedScipConfig,
    ScipPathConfig,
    ScipResolverInput,
    resolve_scip_inputs,
)
from tests._helpers.assertions import assert_cannot_setattr

# Test constants for magic values
EXPECTED_START_LINE = 5
EXPECTED_END_LINE = 10


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

    assert config.repo == "test-org/test-repo"
    assert config.commit == "abc123"
    assert config.repo_root == repo_root
    assert config.build_dir == build_dir
    assert config.document_output_dir == doc_dir
    assert config.scip_python_bin is None
    assert config.scip_bin is None
    assert config.modules == []


def test_resolved_scip_config_create_with_modules(tmp_path: Path) -> None:
    """Test creating ResolvedScipConfig with module records."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    module = ModuleRecord(
        rel_path="src/main.py",
        module_name="src.main",
        file_path=repo_root / "src" / "main.py",
        index=1,
        total=1,
    )

    config = ResolvedScipConfig(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=repo_root,
        build_dir=tmp_path / "build",
        document_output_dir=tmp_path / "docs",
        scip_python_bin="/usr/bin/scip-python",
        scip_bin="/usr/bin/scip",
        modules=[module],
    )

    assert len(config.modules) == 1
    assert config.modules[0].module_name == "src.main"
    assert config.scip_python_bin == "/usr/bin/scip-python"
    assert config.scip_bin == "/usr/bin/scip"


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

    assert inputs.repo is None
    assert inputs.commit is None
    assert inputs.repo_root is None
    assert inputs.build_dir is None
    assert inputs.document_output_dir is None
    assert inputs.scip_python_bin is None
    assert inputs.scip_bin is None
    assert inputs.modules is None


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

    assert inputs.repo == "test-org/test-repo"
    assert inputs.commit == "abc123"
    assert inputs.repo_root == repo_root
    assert inputs.build_dir == build_dir
    assert inputs.document_output_dir == doc_dir
    assert inputs.scip_python_bin == "/usr/bin/scip-python"
    assert inputs.scip_bin == "/usr/bin/scip"


def test_scip_resolver_input_create_with_modules(tmp_path: Path) -> None:
    """Test creating ScipResolverInput with pre-computed modules."""
    module = ModuleRecord(
        rel_path="main.py",
        module_name="main",
        file_path=tmp_path / "main.py",
        index=1,
        total=1,
    )

    inputs = ScipResolverInput(
        repo="test-repo",
        commit="abc",
        repo_root=tmp_path,
        build_dir=tmp_path / "build",
        document_output_dir=tmp_path / "docs",
        modules=[module],
    )

    assert inputs.modules is not None
    assert len(inputs.modules) == 1
    assert inputs.modules[0].module_name == "main"


def test_scip_resolver_input_frozen_dataclass() -> None:
    """Test that ScipResolverInput is immutable."""
    inputs = ScipResolverInput(repo="test-repo")

    assert_cannot_setattr(inputs, "repo", "new-repo")


# --- resolve_scip_inputs Tests ---


def test_resolve_scip_inputs_with_explicit_params(tmp_path: Path) -> None:
    """Test resolving with explicit ScipResolverInput.build()."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
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

    assert isinstance(result, ResolvedScipConfig)
    assert result.repo == "test-org/test-repo"
    assert result.commit == "abc123"
    assert result.repo_root == repo_root
    assert result.build_dir == build_dir
    assert result.document_output_dir == doc_dir


def test_resolve_scip_inputs_with_scip_resolver_input(tmp_path: Path) -> None:
    """Test resolving with ScipResolverInput dataclass."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
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

    assert result.repo == "test-org/test-repo"
    assert result.commit == "def456"
    assert result.scip_python_bin == "/usr/bin/scip-python"
    assert result.scip_bin == "/usr/bin/scip"


def test_resolve_scip_inputs_with_modules_sequence(tmp_path: Path) -> None:
    """Test resolving with modules passed as first argument."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    module = ModuleRecord(
        rel_path="main.py",
        module_name="main",
        file_path=repo_root / "main.py",
        index=1,
        total=1,
    )

    result = resolve_scip_inputs(
        [module],
        ScipResolverInput.build(
            repo="test-repo",
            commit="abc",
            paths=ScipPathConfig.from_strings(
                repo_root=repo_root,
                build_dir=tmp_path / "build",
                document_output_dir=tmp_path / "docs",
            ),
        ),
    )

    assert len(result.modules) == 1
    assert result.modules[0].module_name == "main"


def test_resolve_scip_inputs_with_modules_in_input(tmp_path: Path) -> None:
    """Test resolving with modules passed via ScipResolverInput."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    module = ModuleRecord(
        rel_path="util.py",
        module_name="util",
        file_path=repo_root / "util.py",
        index=1,
        total=1,
    )

    result = resolve_scip_inputs(
        [],  # Empty, modules in input takes precedence
        ScipResolverInput.build(
            repo="test-repo",
            commit="abc",
            paths=ScipPathConfig.from_strings(
                repo_root=repo_root,
                build_dir=tmp_path / "build",
                document_output_dir=tmp_path / "docs",
            ),
            modules=[module],
        ),
    )

    assert len(result.modules) == 1
    assert result.modules[0].module_name == "util"


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

    assert result.repo == "inputs-repo"
    assert result.commit == "inputs-commit"


# --- ModuleRecord Tests ---


def test_module_record_create(tmp_path: Path) -> None:
    """Test creating a ModuleRecord with all fields."""
    file_path = tmp_path / "src" / "module.py"

    record = ModuleRecord(
        rel_path="src/module.py",
        module_name="src.module",
        file_path=file_path,
        index=EXPECTED_START_LINE,
        total=EXPECTED_END_LINE,
    )

    assert record.rel_path == "src/module.py"
    assert record.module_name == "src.module"
    assert record.file_path == file_path
    assert record.index == EXPECTED_START_LINE
    assert record.total == EXPECTED_END_LINE


def test_module_record_frozen(tmp_path: Path) -> None:
    """Test that ModuleRecord is immutable."""
    record = ModuleRecord(
        rel_path="main.py",
        module_name="main",
        file_path=tmp_path / "main.py",
        index=1,
        total=1,
    )

    assert_cannot_setattr(record, "module_name", "new_name")
