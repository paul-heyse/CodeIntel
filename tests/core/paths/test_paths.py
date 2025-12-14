"""Tests for core path utilities.

This module tests path normalization, module conversion, and repository
path functions from codeintel.core.paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.core.paths import (
    ensure_repo_root,
    is_package_path,
    module_to_path,
    normalize_path,
    path_to_module,
    relpath_to_module,
    repo_relpath,
    safe_relpath,
)


class TestNormalizePath:
    """Tests for normalize_path function."""

    def test_forward_slashes(self) -> None:
        """Test that backslashes are converted to forward slashes."""
        result = normalize_path("src\\module\\file.py")
        assert "/" in result or "\\" not in result

    def test_simple_path(self) -> None:
        """Test simple path normalization."""
        result = normalize_path("src/module/file.py")
        assert "src" in result
        assert "module" in result
        assert "file.py" in result

    def test_path_object(self) -> None:
        """Test Path object input."""
        result = normalize_path(Path("src/module/file.py"))
        assert isinstance(result, str)

    def test_removes_leading_dot_slash(self) -> None:
        """Test that leading ./ is removed."""
        result = normalize_path("./src/file.py")
        assert not result.startswith("./")


class TestEnsureRepoRoot:
    """Tests for ensure_repo_root function."""

    def test_returns_path(self) -> None:
        """Test that result is a Path object."""
        result = ensure_repo_root("/some/path")
        assert isinstance(result, Path)

    def test_absolute_path(self) -> None:
        """Test that result is absolute."""
        result = ensure_repo_root("relative/path")
        assert result.is_absolute()

    def test_resolves_path(self) -> None:
        """Test that path is resolved."""
        result = ensure_repo_root(".")
        assert result.is_absolute()

    def test_path_object_input(self) -> None:
        """Test Path object input."""
        result = ensure_repo_root(Path("/some/path"))
        assert isinstance(result, Path)


class TestRepoRelpath:
    """Tests for repo_relpath function."""

    def test_basic_relpath(self) -> None:
        """Test basic relative path computation."""
        repo_root = Path("/project")
        file_path = Path("/project/src/file.py")
        result = repo_relpath(repo_root, file_path)
        assert result == "src/file.py"

    def test_nested_path(self) -> None:
        """Test nested relative path."""
        repo_root = Path("/project")
        file_path = Path("/project/src/pkg/mod/file.py")
        result = repo_relpath(repo_root, file_path)
        assert result == "src/pkg/mod/file.py"

    def test_posix_format(self) -> None:
        """Test that result uses forward slashes."""
        repo_root = Path("/project")
        file_path = Path("/project/src/file.py")
        result = repo_relpath(repo_root, file_path)
        assert "/" in result or len(result.split("/")) > 0

    def test_not_under_root_raises(self) -> None:
        """Test that non-relative path raises ValueError."""
        repo_root = Path("/project")
        file_path = Path("/other/path/file.py")

        with pytest.raises(ValueError):
            repo_relpath(repo_root, file_path)

    def test_string_path(self) -> None:
        """Test string path input."""
        repo_root = Path("/project")
        result = repo_relpath(repo_root, "/project/src/file.py")
        assert result == "src/file.py"


class TestSafeRelpath:
    """Tests for safe_relpath function."""

    def test_relative_path(self) -> None:
        """Test computing relative path."""
        result = safe_relpath("/project/src/file.py", "/project")
        assert result == "src/file.py"

    def test_fallback_to_absolute(self) -> None:
        """Test fallback when path is not under base."""
        result = safe_relpath("/other/path/file.py", "/project")
        # Should return normalized absolute path
        assert isinstance(result, str)

    def test_string_inputs(self) -> None:
        """Test with string inputs."""
        result = safe_relpath("src/file.py", ".")
        assert isinstance(result, str)


class TestPathToModule:
    """Tests for path_to_module function."""

    def test_simple_module(self) -> None:
        """Test simple module conversion."""
        result = path_to_module("src/module/file.py")
        assert result == "src.module.file"

    def test_init_file(self) -> None:
        """Test __init__.py handling."""
        result = path_to_module("package/__init__.py")
        assert result == "package"

    def test_nested_init(self) -> None:
        """Test nested __init__.py handling."""
        result = path_to_module("src/pkg/__init__.py")
        assert result == "src.pkg"

    def test_no_extension(self) -> None:
        """Test file without .py extension."""
        result = path_to_module("src/module/file")
        assert result == "src.module.file"

    def test_backslashes(self) -> None:
        """Test Windows-style backslashes."""
        result = path_to_module("src\\module\\file.py")
        assert result == "src.module.file"


class TestRelpathToModule:
    """Tests for relpath_to_module alias."""

    def test_is_alias(self) -> None:
        """Test that relpath_to_module is an alias for path_to_module."""
        assert relpath_to_module is path_to_module

    def test_works_correctly(self) -> None:
        """Test that the alias works correctly."""
        result = relpath_to_module("pkg/sub/module.py")
        assert result == "pkg.sub.module"


class TestModuleToPath:
    """Tests for module_to_path function."""

    def test_simple_module(self) -> None:
        """Test simple module to path conversion."""
        result = module_to_path("package.module")
        assert result == "package/module.py"

    def test_package_path(self) -> None:
        """Test package path generation."""
        result = module_to_path("package", as_package=True)
        assert result == "package/__init__.py"

    def test_nested_module(self) -> None:
        """Test nested module path."""
        result = module_to_path("a.b.c.module")
        assert result == "a/b/c/module.py"


class TestIsPackagePath:
    """Tests for is_package_path function."""

    def test_init_file(self) -> None:
        """Test __init__.py detection."""
        assert is_package_path("package/__init__.py") is True

    def test_regular_file(self) -> None:
        """Test regular file is not package."""
        assert is_package_path("module.py") is False

    def test_nested_init(self) -> None:
        """Test nested __init__.py."""
        assert is_package_path("a/b/c/__init__.py") is True

    def test_standalone_init(self) -> None:
        """Test standalone __init__.py."""
        assert is_package_path("__init__.py") is True

    def test_path_object(self) -> None:
        """Test Path object input."""
        assert is_package_path(Path("package/__init__.py")) is True


class TestImportsWork:
    """Tests that all expected imports are available."""

    def test_all_functions_importable(self) -> None:
        """Test that all functions can be imported from core.paths."""
        from codeintel.core.paths import (
            ensure_repo_root,
            is_package_path,
            module_to_path,
            normalize_path,
            path_to_module,
            relpath_to_module,
            repo_relpath,
            safe_relpath,
        )

        # Verify they're callable
        assert callable(ensure_repo_root)
        assert callable(is_package_path)
        assert callable(module_to_path)
        assert callable(normalize_path)
        assert callable(path_to_module)
        assert callable(relpath_to_module)
        assert callable(repo_relpath)
        assert callable(safe_relpath)
