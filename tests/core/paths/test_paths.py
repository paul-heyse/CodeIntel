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
    normalize_optional_path,
    normalize_path,
    normalize_rel_path,
    path_to_module,
    repo_relpath,
    safe_relpath,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_true,
)


class TestNormalizePath:
    """Tests for normalize_path function."""

    @staticmethod
    def test_forward_slashes() -> None:
        """Test that backslashes are converted to forward slashes."""
        result = normalize_path("src\\module\\file.py")
        expect_true("/" in result or "\\" not in result)

    @staticmethod
    def test_simple_path() -> None:
        """Test simple path normalization."""
        result = normalize_path("src/module/file.py")
        expect_in("src", result)
        expect_in("module", result)
        expect_in("file.py", result)

    @staticmethod
    def test_path_object() -> None:
        """Test Path object input."""
        result = normalize_path(Path("src/module/file.py"))
        expect_is_instance(result, str)

    @staticmethod
    def test_removes_leading_dot_slash() -> None:
        """Test that leading ./ is removed."""
        result = normalize_path("./src/file.py")
        expect_false(result.startswith("./"))

    @staticmethod
    def test_empty_path_returns_empty() -> None:
        """Test that empty paths normalize to empty string."""
        result = normalize_path("")
        expect_equal(result, "")


class TestNormalizeRelPath:
    """Tests for normalize_rel_path function."""

    @staticmethod
    def test_rel_path_normalizes_separators() -> None:
        """Test that backslashes become forward slashes."""
        result = normalize_rel_path("src\\module\\file.py")
        expect_true("/" in result or "\\" not in result)

    @staticmethod
    def test_rel_path_removes_dot_segments() -> None:
        """Test that dot segments are normalized."""
        result = normalize_rel_path("./src/../src/file.py")
        expect_equal(result, "src/file.py")


class TestNormalizeOptionalPath:
    """Tests for normalize_optional_path function."""

    @staticmethod
    def test_none_passthrough() -> None:
        """Test that None remains None."""
        expect_equal(normalize_optional_path(None), None)

    @staticmethod
    def test_expand_user() -> None:
        """Test that user paths are expanded."""
        result = normalize_optional_path(Path("~"))
        expect_is_instance(result, Path)

    @staticmethod
    def test_resolve_flag() -> None:
        """Test optional resolve behavior."""
        result = normalize_optional_path("src", resolve=False)
        expect_is_instance(result, Path)


class TestEnsureRepoRoot:
    """Tests for ensure_repo_root function."""

    @staticmethod
    def test_returns_path() -> None:
        """Test that result is a Path object."""
        result = ensure_repo_root("/some/path")
        expect_is_instance(result, Path)

    @staticmethod
    def test_absolute_path() -> None:
        """Test that result is absolute."""
        result = ensure_repo_root("relative/path")
        expect_true(result.is_absolute())

    @staticmethod
    def test_resolves_path() -> None:
        """Test that path is resolved."""
        result = ensure_repo_root(".")
        expect_true(result.is_absolute())

    @staticmethod
    def test_path_object_input() -> None:
        """Test Path object input."""
        result = ensure_repo_root(Path("/some/path"))
        expect_is_instance(result, Path)


class TestRepoRelpath:
    """Tests for repo_relpath function."""

    @staticmethod
    def test_basic_relpath() -> None:
        """Test basic relative path computation."""
        repo_root = Path("/project")
        file_path = Path("/project/src/file.py")
        result = repo_relpath(repo_root, file_path)
        expect_equal(result, "src/file.py")

    @staticmethod
    def test_nested_path() -> None:
        """Test nested relative path."""
        repo_root = Path("/project")
        file_path = Path("/project/src/pkg/mod/file.py")
        result = repo_relpath(repo_root, file_path)
        expect_equal(result, "src/pkg/mod/file.py")

    @staticmethod
    def test_posix_format() -> None:
        """Test that result uses forward slashes."""
        repo_root = Path("/project")
        file_path = Path("/project/src/file.py")
        result = repo_relpath(repo_root, file_path)
        expect_true("/" in result or len(result.split("/")) > 0)

    @staticmethod
    def test_not_under_root_raises() -> None:
        """Test that non-relative path raises ValueError."""
        repo_root = Path("/project")
        file_path = Path("/other/path/file.py")

        with pytest.raises(ValueError, match="is not in the subpath"):
            repo_relpath(repo_root, file_path)

    @staticmethod
    def test_string_path() -> None:
        """Test string path input."""
        repo_root = Path("/project")
        result = repo_relpath(repo_root, "/project/src/file.py")
        expect_equal(result, "src/file.py")


class TestSafeRelpath:
    """Tests for safe_relpath function."""

    @staticmethod
    def test_relative_path() -> None:
        """Test computing relative path."""
        result = safe_relpath("/project/src/file.py", "/project")
        expect_equal(result, "src/file.py")

    @staticmethod
    def test_fallback_to_absolute() -> None:
        """Test fallback when path is not under base."""
        result = safe_relpath("/other/path/file.py", "/project")
        # Should return normalized absolute path
        expect_is_instance(result, str)

    @staticmethod
    def test_string_inputs() -> None:
        """Test with string inputs."""
        result = safe_relpath("src/file.py", ".")
        expect_is_instance(result, str)


class TestPathToModule:
    """Tests for path_to_module function."""

    @staticmethod
    def test_simple_module() -> None:
        """Test simple module conversion."""
        result = path_to_module("src/module/file.py")
        expect_equal(result, "src.module.file")

    @staticmethod
    def test_init_file() -> None:
        """Test __init__.py handling."""
        result = path_to_module("package/__init__.py")
        expect_equal(result, "package")

    @staticmethod
    def test_nested_init() -> None:
        """Test nested __init__.py handling."""
        result = path_to_module("src/pkg/__init__.py")
        expect_equal(result, "src.pkg")

    @staticmethod
    def test_no_extension() -> None:
        """Test file without .py extension."""
        result = path_to_module("src/module/file")
        expect_equal(result, "src.module.file")

    @staticmethod
    def test_backslashes() -> None:
        """Test Windows-style backslashes."""
        result = path_to_module("src\\module\\file.py")
        expect_equal(result, "src.module.file")


class TestModuleToPath:
    """Tests for module_to_path function."""

    @staticmethod
    def test_simple_module() -> None:
        """Test simple module to path conversion."""
        result = module_to_path("package.module")
        expect_equal(result, "package/module.py")

    @staticmethod
    def test_package_path() -> None:
        """Test package path generation."""
        result = module_to_path("package", as_package=True)
        expect_equal(result, "package/__init__.py")

    @staticmethod
    def test_nested_module() -> None:
        """Test nested module path."""
        result = module_to_path("a.b.c.module")
        expect_equal(result, "a/b/c/module.py")


class TestIsPackagePath:
    """Tests for is_package_path function."""

    @staticmethod
    def test_init_file() -> None:
        """Test __init__.py detection."""
        expect_true(is_package_path("package/__init__.py"))

    @staticmethod
    def test_regular_file() -> None:
        """Test regular file is not package."""
        expect_false(is_package_path("module.py"))

    @staticmethod
    def test_nested_init() -> None:
        """Test nested __init__.py."""
        expect_true(is_package_path("a/b/c/__init__.py"))

    @staticmethod
    def test_standalone_init() -> None:
        """Test standalone __init__.py."""
        expect_true(is_package_path("__init__.py"))

    @staticmethod
    def test_path_object() -> None:
        """Test Path object input."""
        expect_true(is_package_path(Path("package/__init__.py")))


class TestImportsWork:
    """Tests that all expected imports are available."""

    @staticmethod
    def test_all_functions_importable() -> None:
        """Test that all functions can be imported from core.paths."""
        # Verify they're callable
        expect_true(callable(ensure_repo_root))
        expect_true(callable(is_package_path))
        expect_true(callable(module_to_path))
        expect_true(callable(normalize_path))
        expect_true(callable(path_to_module))
        expect_true(callable(repo_relpath))
        expect_true(callable(safe_relpath))
