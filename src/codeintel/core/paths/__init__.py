r"""Unified path utilities.

This module provides path handling utilities for the codebase,
including path normalization, module conversion, and repository paths.

Examples
--------
>>> from codeintel.core.paths import normalize_path, path_to_module
>>>
>>> normalize_path("src\\module\\file.py")
'src/module/file.py'
>>> path_to_module("src/module/file.py")
'src.module.file'
"""

from codeintel.core.paths.module import (
    is_package_path,
    module_to_path,
    path_to_module,
)
from codeintel.core.paths.normalize import (
    ensure_repo_root,
    normalize_optional_path,
    normalize_path,
    normalize_rel_path,
    repo_relpath,
    safe_relpath,
)

__all__ = [
    "ensure_repo_root",
    "is_package_path",
    "module_to_path",
    "normalize_optional_path",
    "normalize_path",
    "normalize_rel_path",
    "path_to_module",
    "repo_relpath",
    "safe_relpath",
]
