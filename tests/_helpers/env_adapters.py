"""Adapters to bridge old environment classes with new hexagonal architecture.

This module provides adapter functions and classes that allow existing tests
using SpanTestEnv, PipelineEnv, and CoverageEdgeEnv to gradually migrate
to the new TestContext-based architecture.

The adapters wrap TestContext to provide backward-compatible interfaces
while internally using the hexagonal patterns.

Migration Strategy
------------------
1. New tests should use TestContext and seed packs directly.
2. Existing tests can use these adapters as an intermediate step.
3. Eventually, migrate existing tests to use TestContext directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config import ConfigBuilder
from codeintel.config.primitives import BuildPaths
from tests._helpers.builders import (
    GoidRow,
    ModuleRow,
    TestCatalogRow,
    insert_goids,
    insert_modules,
    insert_test_catalog,
)
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.seeds import COVERAGE_PACK, GRAPH_PACK

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Span Environment Adapter
# =============================================================================


@dataclass
class SpanEnvAdapter:
    """Adapter providing SpanTestEnv-like interface using TestContext.

    This adapter wraps a TestContext to provide the same interface as the
    legacy SpanTestEnv dataclass, enabling gradual migration.

    Attributes
    ----------
    ctx : TestContext
        Underlying hexagonal test context.
    expected_goid : int
        Expected GOID for the caller function.
    """

    ctx: TestContext
    expected_goid: int

    @property
    def repo_root(self) -> Path:
        """Return the repository root path.

        Returns
        -------
        Path
            Repository root path.
        """
        return self.ctx.repo_root

    @property
    def builder(self) -> ConfigBuilder:
        """Return a ConfigBuilder for the context.

        Returns
        -------
        ConfigBuilder
            Config builder configured with context snapshot.
        """
        return ConfigBuilder.from_snapshot(
            repo=self.ctx.repo,
            commit=self.ctx.commit,
            repo_root=self.ctx.repo_root,
        )

    @property
    def gateway(self) -> StorageGateway:
        """Return the storage gateway.

        Returns
        -------
        StorageGateway
            Gateway for database access.
        """
        return self.ctx.gateway


def create_span_env_from_context(
    tmp_path: Path,
    *,
    repo: str = "demo/repo",
    commit: str = "deadbeef",
) -> SpanEnvAdapter:
    """Create a SpanEnvAdapter using the hexagonal architecture.

    This is a drop-in replacement for create_span_test_env that uses
    TestContext internally.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    SpanEnvAdapter
        Adapter providing SpanTestEnv-like interface.
    """
    ctx = create_test_context(tmp_path, repo=repo, commit=commit)

    # Write repo files
    pkg_dir = ctx.repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    (pkg_dir / "a.py").write_text("def callee():\n    return 1\n", encoding="utf8")
    (pkg_dir / "b.py").write_text(
        "from pkg.a import callee\n\ndef caller():\n    return callee()\n",
        encoding="utf8",
    )

    # Seed modules and GOIDs
    now = datetime.now(UTC)
    expected_goid = 200
    caller_start, caller_end = 3, 4

    insert_modules(
        ctx.gateway,
        [
            ModuleRow(module="pkg.a", path="pkg/a.py", repo=repo, commit=commit),
            ModuleRow(module="pkg.b", path="pkg/b.py", repo=repo, commit=commit),
        ],
    )

    insert_goids(
        ctx.gateway,
        [
            GoidRow(
                goid_h128=expected_goid,
                urn="urn:pkg.b.caller",
                repo=repo,
                commit=commit,
                rel_path="pkg/b.py",
                kind="function",
                qualname="pkg.b.caller",
                start_line=caller_start,
                end_line=caller_end,
                language="python",
                created_at=now,
            ),
        ],
    )

    # Seed test catalog
    insert_test_catalog(
        ctx.gateway,
        [
            TestCatalogRow(
                test_id="tests/test_sample.py::test_caller",
                repo=repo,
                commit=commit,
                rel_path="pkg/b.py",
                qualname="pkg.b.caller",
                status="passed",
                created_at=now,
            ),
        ],
    )

    return SpanEnvAdapter(ctx=ctx, expected_goid=expected_goid)


# =============================================================================
# Pipeline Environment Adapter
# =============================================================================


@dataclass
class PipelineEnvAdapter:
    """Adapter providing PipelineEnv-like interface using TestContext.

    This adapter wraps a TestContext to provide the same interface as the
    legacy PipelineEnv dataclass, enabling gradual migration.

    Attributes
    ----------
    ctx : TestContext
        Underlying hexagonal test context.
    caller_lines : tuple[int, int]
        Start and end lines for the caller function.
    """

    ctx: TestContext
    caller_lines: tuple[int, int]

    @property
    def repo_root(self) -> Path:
        """Return the repository root path.

        Returns
        -------
        Path
            Repository root path.
        """
        return self.ctx.repo_root

    @property
    def build_paths(self) -> BuildPaths:
        """Return build paths configuration.

        Returns
        -------
        BuildPaths
            Build paths for the test.
        """
        return self.ctx.build_paths

    @property
    def gateway(self) -> StorageGateway:
        """Return the storage gateway.

        Returns
        -------
        StorageGateway
            Gateway for database access.
        """
        return self.ctx.gateway


def create_pipeline_env_from_context(
    tmp_path: Path,
    *,
    repo: str = "demo/repo",
    commit: str = "deadbeef",
) -> PipelineEnvAdapter:
    """Create a PipelineEnvAdapter using the hexagonal architecture.

    This is a drop-in replacement for create_pipeline_env that uses
    TestContext internally.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    PipelineEnvAdapter
        Adapter providing PipelineEnv-like interface.
    """
    # Use file-backed DB for pipeline tests
    ctx = create_test_context(tmp_path, repo=repo, commit=commit, file_backed=True)

    # Write repo files
    pkg_dir = ctx.repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    (pkg_dir / "a.py").write_text("def callee():\n    return 1\n", encoding="utf8")
    (pkg_dir / "b.py").write_text(
        "from pkg.a import callee\n\ndef caller():\n    return callee()\n",
        encoding="utf8",
    )

    caller_lines = (3, 4)

    # Seed test catalog
    now = datetime.now(UTC)
    insert_test_catalog(
        ctx.gateway,
        [
            TestCatalogRow(
                test_id="tests/test_sample.py::test_caller",
                repo=repo,
                commit=commit,
                rel_path="pkg/b.py",
                qualname="pkg.b.caller",
                status="passed",
                created_at=now,
            ),
        ],
    )

    return PipelineEnvAdapter(ctx=ctx, caller_lines=caller_lines)


# =============================================================================
# Coverage Environment Adapter
# =============================================================================


@dataclass(frozen=True)
class CoverageEnvConfig:
    """Configuration for coverage environment adapter.

    Attributes
    ----------
    repo : str
        Repository identifier.
    commit : str
        Commit hash.
    module_import : str
        Module import path.
    function_name : str
        Function name.
    test_id : str
        Test identifier.
    function_goid : int
        GOID for the function.
    test_goid : int
        GOID for the test.
    """

    repo: str = "demo/repo"
    commit: str = "deadbeef"
    module_import: str = "pkg.mod"
    function_name: str = "func"
    test_id: str = "pkg/mod.py::test_func"
    function_goid: int = 1
    test_goid: int = 99


@dataclass
class CoverageEnvAdapter:
    """Adapter providing CoverageEdgeEnv-like interface using TestContext.

    This adapter wraps a TestContext to provide the same interface as the
    legacy CoverageEdgeEnv dataclass, enabling gradual migration.

    Attributes
    ----------
    ctx : TestContext
        Underlying hexagonal test context.
    config : CoverageEnvConfig
        Configuration for the coverage environment.
    """

    ctx: TestContext
    config: CoverageEnvConfig

    @property
    def module_import(self) -> str:
        """Return the module import path."""
        return self.config.module_import

    @property
    def function_name(self) -> str:
        """Return the function name."""
        return self.config.function_name

    @property
    def test_id(self) -> str:
        """Return the test identifier."""
        return self.config.test_id

    @property
    def function_goid(self) -> int:
        """Return the function GOID."""
        return self.config.function_goid

    @property
    def test_goid(self) -> int:
        """Return the test GOID."""
        return self.config.test_goid

    @property
    def repo_root(self) -> Path:
        """Return the repository root path.

        Returns
        -------
        Path
            Repository root path.
        """
        return self.ctx.repo_root

    @property
    def gateway(self) -> StorageGateway:
        """Return the storage gateway.

        Returns
        -------
        StorageGateway
            Gateway for database access.
        """
        return self.ctx.gateway

    @property
    def builder(self) -> ConfigBuilder:
        """Return a ConfigBuilder for the context.

        Returns
        -------
        ConfigBuilder
            Config builder configured with context snapshot.
        """
        return ConfigBuilder.from_snapshot(
            repo=self.ctx.repo,
            commit=self.ctx.commit,
            repo_root=self.ctx.repo_root,
        )


def create_coverage_env_from_context(
    tmp_path: Path,
    config: CoverageEnvConfig | None = None,
) -> CoverageEnvAdapter:
    """Create a CoverageEnvAdapter using the hexagonal architecture.

    This is a drop-in replacement for create_coverage_edge_env that uses
    TestContext internally.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    config
        Optional configuration; uses defaults if not provided.

    Returns
    -------
    CoverageEnvAdapter
        Adapter providing CoverageEdgeEnv-like interface.
    """
    cfg = config or CoverageEnvConfig()

    # Use file-backed DB for coverage tests
    ctx = create_test_context(tmp_path, repo=cfg.repo, commit=cfg.commit, file_backed=True)

    # Write repo file
    rel_path = Path(cfg.module_import.replace(".", "/")).with_suffix(".py")
    target_file = ctx.repo_root / rel_path
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(
        f"def {cfg.function_name}():\n    return 1\n",
        encoding="utf8",
    )

    # Seed module and GOIDs
    now = datetime.now(UTC)
    function_urn = f"goid:{cfg.repo}#python:function:{cfg.module_import}.{cfg.function_name}"
    function_qualname = f"{cfg.module_import}.{cfg.function_name}"
    test_urn = f"goid:{cfg.repo}#python:function:{cfg.module_import}.test_func"
    test_qualname = f"{cfg.module_import}.test_func"

    insert_modules(
        ctx.gateway,
        [
            ModuleRow(
                module=cfg.module_import,
                path=rel_path.as_posix(),
                repo=cfg.repo,
                commit=cfg.commit,
            ),
        ],
    )

    insert_goids(
        ctx.gateway,
        [
            GoidRow(
                goid_h128=cfg.function_goid,
                urn=function_urn,
                repo=cfg.repo,
                commit=cfg.commit,
                rel_path=rel_path.as_posix(),
                kind="function",
                qualname=function_qualname,
                start_line=1,
                end_line=2,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=cfg.test_goid,
                urn=test_urn,
                repo=cfg.repo,
                commit=cfg.commit,
                rel_path=rel_path.as_posix(),
                kind="test",
                qualname=test_qualname,
                start_line=1,
                end_line=2,
                language="python",
                created_at=now,
            ),
        ],
    )

    # Seed test catalog
    insert_test_catalog(
        ctx.gateway,
        [
            TestCatalogRow(
                test_id=cfg.test_id,
                repo=cfg.repo,
                commit=cfg.commit,
                rel_path=rel_path.as_posix(),
                qualname=test_qualname,
                status="passed",
                created_at=now,
            ),
        ],
    )

    return CoverageEnvAdapter(ctx=ctx, config=cfg)


# =============================================================================
# Convenience Functions
# =============================================================================


def seeded_graph_context(tmp_path: Path) -> TestContext:
    """Create a TestContext pre-seeded with graph data.

    This is equivalent to using graph_ctx fixture but as a function.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Returns
    -------
    TestContext
        Context with GRAPH_PACK applied.
    """
    ctx = create_test_context(tmp_path)
    ctx.require(GRAPH_PACK)
    return ctx


def seeded_coverage_context(tmp_path: Path) -> TestContext:
    """Create a TestContext pre-seeded with coverage data.

    This is equivalent to using coverage_ctx fixture but as a function.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Returns
    -------
    TestContext
        Context with COVERAGE_PACK applied.
    """
    ctx = create_test_context(tmp_path)
    ctx.require(COVERAGE_PACK)
    return ctx


__all__ = [
    "CoverageEnvAdapter",
    "CoverageEnvConfig",
    "PipelineEnvAdapter",
    "SpanEnvAdapter",
    "create_coverage_env_from_context",
    "create_pipeline_env_from_context",
    "create_span_env_from_context",
    "seeded_coverage_context",
    "seeded_graph_context",
]
