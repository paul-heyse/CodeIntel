"""Tests for SCIP resolver utilities.

This module tests the _scip_resolver.py helper functions that normalize
SCIP ingestion inputs into a resolved configuration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.infrastructure_utilities._scip_resolver import (
    ResolvedScipConfig,
    ScipResolverInput,
    resolve_scip_inputs,
)
from codeintel.ingestion.ports.discovery import ModuleRecord


class TestResolvedScipConfig:
    """Tests for ResolvedScipConfig dataclass."""

    def test_create_minimal(self, tmp_path: Path) -> None:
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
            cfg_source="explicit",
            cfg=None,
        )

        assert config.repo == "test-org/test-repo"
        assert config.commit == "abc123"
        assert config.repo_root == repo_root
        assert config.build_dir == build_dir
        assert config.document_output_dir == doc_dir
        assert config.scip_python_bin is None
        assert config.scip_bin is None
        assert config.modules == []
        assert config.cfg_source == "explicit"
        assert config.cfg is None

    def test_create_with_modules(self, tmp_path: Path) -> None:
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
            cfg_source="explicit",
            cfg=None,
        )

        assert len(config.modules) == 1
        assert config.modules[0].module_name == "src.main"
        assert config.scip_python_bin == "/usr/bin/scip-python"
        assert config.scip_bin == "/usr/bin/scip"

    def test_frozen_dataclass(self, tmp_path: Path) -> None:
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
            cfg_source="explicit",
            cfg=None,
        )

        with pytest.raises(AttributeError):
            config.repo = "new-repo"  # type: ignore[misc]


class TestScipResolverInput:
    """Tests for ScipResolverInput dataclass."""

    def test_create_empty(self) -> None:
        """Test creating ScipResolverInput with all defaults."""
        inputs = ScipResolverInput()

        assert inputs.cfg is None
        assert inputs.repo is None
        assert inputs.commit is None
        assert inputs.repo_root is None
        assert inputs.build_dir is None
        assert inputs.document_output_dir is None
        assert inputs.scip_python_bin is None
        assert inputs.scip_bin is None
        assert inputs.modules is None

    def test_create_with_explicit_params(self, tmp_path: Path) -> None:
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

    def test_create_with_modules(self, tmp_path: Path) -> None:
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

    def test_frozen_dataclass(self) -> None:
        """Test that ScipResolverInput is immutable."""
        inputs = ScipResolverInput(repo="test-repo")

        with pytest.raises(AttributeError):
            inputs.repo = "new-repo"  # type: ignore[misc]


class TestResolveScipInputs:
    """Tests for resolve_scip_inputs function."""

    def test_resolve_with_explicit_params(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
        """Test resolving with explicit keyword parameters."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        build_dir = tmp_path / "build"
        doc_dir = tmp_path / "docs"

        result = resolve_scip_inputs(
            fresh_gateway,  # type: ignore[arg-type]
            modules_or_cfg=[],
            repo="test-org/test-repo",
            commit="abc123",
            repo_root=repo_root,
            build_dir=build_dir,
            document_output_dir=doc_dir,
        )

        assert isinstance(result, ResolvedScipConfig)
        assert result.repo == "test-org/test-repo"
        assert result.commit == "abc123"
        assert result.repo_root == repo_root
        assert result.build_dir == build_dir
        assert result.document_output_dir == doc_dir
        assert result.cfg_source == "explicit"
        assert result.cfg is None

    def test_resolve_with_scip_resolver_input(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
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

        result = resolve_scip_inputs(
            fresh_gateway,  # type: ignore[arg-type]
            modules_or_cfg=[],
            inputs=inputs,
        )

        assert result.repo == "test-org/test-repo"
        assert result.commit == "def456"
        assert result.scip_python_bin == "/usr/bin/scip-python"
        assert result.scip_bin == "/usr/bin/scip"
        assert result.cfg_source == "explicit"

    def test_resolve_with_modules_sequence(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
        """Test resolving with modules passed as modules_or_cfg sequence."""
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
            fresh_gateway,  # type: ignore[arg-type]
            modules_or_cfg=[module],
            repo="test-repo",
            commit="abc",
            repo_root=repo_root,
            build_dir=tmp_path / "build",
            document_output_dir=tmp_path / "docs",
        )

        assert len(result.modules) == 1
        assert result.modules[0].module_name == "main"

    def test_resolve_with_modules_kwarg(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
        """Test resolving with modules passed as keyword argument."""
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
            fresh_gateway,  # type: ignore[arg-type]
            modules_or_cfg=[],  # Empty, modules kwarg takes precedence
            repo="test-repo",
            commit="abc",
            repo_root=repo_root,
            build_dir=tmp_path / "build",
            document_output_dir=tmp_path / "docs",
            modules=[module],
        )

        assert len(result.modules) == 1
        assert result.modules[0].module_name == "util"

    def test_resolve_missing_repo_raises_value_error(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
        """Test that missing repo parameter raises ValueError."""
        with pytest.raises(ValueError, match="repo.*required"):
            resolve_scip_inputs(
                fresh_gateway,  # type: ignore[arg-type]
                modules_or_cfg=[],
                commit="abc",
                repo_root=tmp_path,
                build_dir=tmp_path / "build",
                document_output_dir=tmp_path / "docs",
            )

    def test_resolve_missing_commit_raises_value_error(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
        """Test that missing commit parameter raises ValueError."""
        with pytest.raises(ValueError, match="commit.*required"):
            resolve_scip_inputs(
                fresh_gateway,  # type: ignore[arg-type]
                modules_or_cfg=[],
                repo="test-repo",
                repo_root=tmp_path,
                build_dir=tmp_path / "build",
                document_output_dir=tmp_path / "docs",
            )

    def test_resolve_missing_repo_root_raises_value_error(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
        """Test that missing repo_root parameter raises ValueError."""
        with pytest.raises(ValueError, match="repo_root.*required"):
            resolve_scip_inputs(
                fresh_gateway,  # type: ignore[arg-type]
                modules_or_cfg=[],
                repo="test-repo",
                commit="abc",
                build_dir=tmp_path / "build",
                document_output_dir=tmp_path / "docs",
            )

    def test_resolve_missing_build_dir_raises_value_error(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
        """Test that missing build_dir parameter raises ValueError."""
        with pytest.raises(ValueError, match="build_dir.*required"):
            resolve_scip_inputs(
                fresh_gateway,  # type: ignore[arg-type]
                modules_or_cfg=[],
                repo="test-repo",
                commit="abc",
                repo_root=tmp_path,
                document_output_dir=tmp_path / "docs",
            )

    def test_resolve_missing_document_output_dir_raises_value_error(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
        """Test that missing document_output_dir parameter raises ValueError."""
        with pytest.raises(ValueError, match="document_output_dir.*required"):
            resolve_scip_inputs(
                fresh_gateway,  # type: ignore[arg-type]
                modules_or_cfg=[],
                repo="test-repo",
                commit="abc",
                repo_root=tmp_path,
                build_dir=tmp_path / "build",
            )

    def test_resolve_inputs_override_kwargs(
        self, fresh_gateway: object, tmp_path: Path
    ) -> None:
        """Test that ScipResolverInput values override keyword args."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        inputs = ScipResolverInput(
            repo="inputs-repo",
            commit="inputs-commit",
            repo_root=repo_root,
            build_dir=tmp_path / "inputs-build",
            document_output_dir=tmp_path / "inputs-docs",
        )

        # Even if we pass different kwargs, inputs should take precedence
        result = resolve_scip_inputs(
            fresh_gateway,  # type: ignore[arg-type]
            modules_or_cfg=[],
            inputs=inputs,
            repo="kwarg-repo",  # Should be overridden
            commit="kwarg-commit",  # Should be overridden
        )

        assert result.repo == "inputs-repo"
        assert result.commit == "inputs-commit"


class TestModuleRecord:
    """Tests for ModuleRecord dataclass."""

    def test_create_module_record(self, tmp_path: Path) -> None:
        """Test creating a ModuleRecord with all fields."""
        file_path = tmp_path / "src" / "module.py"

        record = ModuleRecord(
            rel_path="src/module.py",
            module_name="src.module",
            file_path=file_path,
            index=5,
            total=10,
        )

        assert record.rel_path == "src/module.py"
        assert record.module_name == "src.module"
        assert record.file_path == file_path
        assert record.index == 5
        assert record.total == 10

    def test_module_record_frozen(self, tmp_path: Path) -> None:
        """Test that ModuleRecord is immutable."""
        record = ModuleRecord(
            rel_path="main.py",
            module_name="main",
            file_path=tmp_path / "main.py",
            index=1,
            total=1,
        )

        with pytest.raises(AttributeError):
            record.module_name = "new_name"  # type: ignore[misc]
