"""Configuration dataclasses for coverage test environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config import ConfigBuilder
    from codeintel.storage.gateway import StorageGateway

# Default constants for coverage tests
REPO = "demo/repo"
COMMIT = "deadbeef"
MODULE_IMPORT = "pkg.mod"
FUNCTION_NAME = "func"
TEST_ID = "pkg/mod.py::test_func"


@dataclass
class CoverageEdgeEnv:
    """Environment for computing test coverage edges end-to-end."""

    repo_root: Path
    gateway: StorageGateway
    builder: ConfigBuilder
    module_import: str
    function_name: str
    test_id: str
    function_goid: int
    test_goid: int


@dataclass(frozen=True)
class CoverageSeedConfig:
    """Configuration for seeding coverage edge fixtures."""

    module_import: str = MODULE_IMPORT
    function_name: str = FUNCTION_NAME
    function_urn: str | None = None
    function_qualname: str | None = None
    test_id: str = TEST_ID
    test_urn: str | None = None
    test_qualname: str | None = None
    repo: str = REPO
    commit: str = COMMIT
    function_goid: int = 1
    test_goid: int = 99


__all__ = [
    "COMMIT",
    "FUNCTION_NAME",
    "MODULE_IMPORT",
    "REPO",
    "TEST_ID",
    "CoverageEdgeEnv",
    "CoverageSeedConfig",
]
