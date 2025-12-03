"""Scenario builder for declarative test setup.

This module provides a fluent TestScenario builder that allows declarative
test environment configuration. Instead of manual setup code, tests can
use a builder pattern to compose environments from seed packs.

Example
-------
```python
# Before: 30+ lines of manual setup
gateway = open_memory_gateway(...)
insert_modules(gateway, [...])
insert_goids(gateway, [...])
...

# After: declarative scenario builder
ctx = TestScenario.minimal().with_seeds(GRAPH_PACK, METRICS_PACK).build(tmp_path)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

from tests._helpers.context import (
    DEFAULT_COMMIT,
    DEFAULT_REPO,
    TestContext,
    create_test_context,
)
from tests._helpers.ports.repo import FileSystemRepo
from tests._helpers.seeds import CORE_PACK, COVERAGE_PACK, GRAPH_PACK, METRICS_PACK

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack


# =============================================================================
# Scenario Configuration
# =============================================================================


@dataclass
class ScenarioConfig:
    """Configuration for a test scenario.

    Attributes
    ----------
    repo : str
        Repository identifier.
    commit : str
        Commit hash.
    seed_packs : list[SeedPack]
        Seed packs to apply.
    file_backed : bool
        Whether to use file-backed DuckDB.
    write_files : bool
        Whether to write sample Python files.
    extra : dict[str, object]
        Additional scenario-specific configuration.
    """

    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    seed_packs: list[SeedPack] = field(default_factory=list)
    file_backed: bool = False
    write_files: bool = False
    extra: dict[str, object] = field(default_factory=dict)


# =============================================================================
# Sample File Content
# =============================================================================

# Minimal Python module content for realistic tests
SAMPLE_MOD_A = '''\
"""Module A for testing."""


def func_a(x: int, y: int) -> int:
    """Add two numbers and call func_b.

    Parameters
    ----------
    x : int
        First number.
    y : int
        Second number.

    Returns
    -------
    int
        Sum of x and y.
    """
    from pkg.mod_b import func_b

    result = func_b(x)
    return result + y
'''

SAMPLE_MOD_B = '''\
"""Module B for testing."""


def func_b(x: int) -> int:
    """Double a number and call func_c.

    Parameters
    ----------
    x : int
        Number to double.

    Returns
    -------
    int
        Doubled value.
    """
    from pkg.mod_c import func_c

    func_c()
    return x * 2
'''

SAMPLE_MOD_C = '''\
"""Module C for testing."""


def func_c():
    """Yield values for iteration."""
    yield 1
    yield 2
    yield 3
'''

SAMPLE_UTIL = '''\
"""Utility module for testing."""


def helper(value: int) -> int:
    """Return the value unchanged.

    Parameters
    ----------
    value : int
        Value to pass through.

    Returns
    -------
    int
        Same value.
    """
    return value
'''


# =============================================================================
# Scenario Builder
# =============================================================================


@dataclass
class TestScenario:
    """Fluent builder for test scenarios.

    Provides a declarative API for configuring test environments.
    Methods return self for chaining.

    Attributes
    ----------
    config : ScenarioConfig
        Current configuration state.
    """

    __test__ = False  # Prevent pytest collection

    config: ScenarioConfig = field(default_factory=ScenarioConfig)

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def minimal(cls) -> TestScenario:
        """Create scenario with minimal core data only.

        Returns
        -------
        TestScenario
            Scenario configured with CORE_PACK.
        """
        scenario = cls()
        scenario.config.seed_packs.append(CORE_PACK)
        return scenario

    @classmethod
    def with_graph(cls) -> TestScenario:
        """Create scenario with core and graph data.

        Returns
        -------
        TestScenario
            Scenario configured with CORE_PACK and GRAPH_PACK.
        """
        scenario = cls()
        scenario.config.seed_packs.extend([CORE_PACK, GRAPH_PACK])
        return scenario

    @classmethod
    def with_coverage(cls) -> TestScenario:
        """Create scenario with core and coverage data.

        Returns
        -------
        TestScenario
            Scenario configured with CORE_PACK and COVERAGE_PACK.
        """
        scenario = cls()
        scenario.config.seed_packs.extend([CORE_PACK, COVERAGE_PACK])
        return scenario

    @classmethod
    def with_metrics(cls) -> TestScenario:
        """Create scenario with core and metrics data.

        Returns
        -------
        TestScenario
            Scenario configured with CORE_PACK and METRICS_PACK.
        """
        scenario = cls()
        scenario.config.seed_packs.extend([CORE_PACK, METRICS_PACK])
        return scenario

    @classmethod
    def full_stack(cls) -> TestScenario:
        """Create scenario with all seed packs.

        Returns
        -------
        TestScenario
            Scenario with CORE, GRAPH, COVERAGE, and METRICS packs.
        """
        scenario = cls()
        scenario.config.seed_packs.extend([CORE_PACK, GRAPH_PACK, COVERAGE_PACK, METRICS_PACK])
        return scenario

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def with_repo(self, repo: str) -> Self:
        """Set repository identifier.

        Parameters
        ----------
        repo
            Repository slug.

        Returns
        -------
        Self
            Self for chaining.
        """
        self.config.repo = repo
        return self

    def with_commit(self, commit: str) -> Self:
        """Set commit hash.

        Parameters
        ----------
        commit
            Commit identifier.

        Returns
        -------
        Self
            Self for chaining.
        """
        self.config.commit = commit
        return self

    def with_seeds(self, *packs: SeedPack) -> Self:
        """Add seed packs to apply.

        Parameters
        ----------
        packs
            Seed packs to add.

        Returns
        -------
        Self
            Self for chaining.
        """
        self.config.seed_packs.extend(packs)
        return self

    def file_backed(self) -> Self:
        """Use file-backed DuckDB instead of in-memory.

        Returns
        -------
        Self
            Self for chaining.
        """
        self.config.file_backed = True
        return self

    def with_sample_files(self) -> Self:
        """Write sample Python files to repo directory.

        Returns
        -------
        Self
            Self for chaining.
        """
        self.config.write_files = True
        return self

    def with_extra(self, key: str, value: object) -> Self:
        """Add extra configuration value.

        Parameters
        ----------
        key
            Configuration key.
        value
            Configuration value.

        Returns
        -------
        Self
            Self for chaining.
        """
        self.config.extra[key] = value
        return self

    # -------------------------------------------------------------------------
    # Build Methods
    # -------------------------------------------------------------------------

    def build(self, tmp_path: Path) -> TestContext:
        """Build the test context from configuration.

        Creates the test environment, applies seeds, and optionally
        writes sample files.

        Parameters
        ----------
        tmp_path
            Temporary directory for test artifacts.

        Returns
        -------
        TestContext
            Configured and seeded test context.
        """
        # Create base context
        ctx = create_test_context(
            tmp_path,
            repo=self.config.repo,
            commit=self.config.commit,
            file_backed=self.config.file_backed,
        )

        # Copy extra config
        ctx.extra.update(self.config.extra)

        # Write sample files if requested
        if self.config.write_files:
            self._write_sample_files(ctx.repo_root)

        # Apply seed packs
        for pack in self.config.seed_packs:
            ctx.require(pack)

        return ctx

    @staticmethod
    def _write_sample_files(repo_root: Path) -> None:
        """Write sample Python files to repository.

        Parameters
        ----------
        repo_root
            Repository root path.
        """
        repo = FileSystemRepo(repo_root)

        # Create package structure
        repo.write_file("pkg/__init__.py", '"""Test package."""\n')
        repo.write_file("pkg/mod_a.py", SAMPLE_MOD_A)
        repo.write_file("pkg/mod_b.py", SAMPLE_MOD_B)
        repo.write_file("pkg/mod_c.py", SAMPLE_MOD_C)
        repo.write_file("pkg/util.py", SAMPLE_UTIL)


# =============================================================================
# Convenience Functions
# =============================================================================


def minimal_context(tmp_path: Path) -> TestContext:
    """Create minimal test context with core seeds.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Returns
    -------
    TestContext
        Context with CORE_PACK applied.
    """
    return TestScenario.minimal().build(tmp_path)


def graph_context(tmp_path: Path) -> TestContext:
    """Create test context with graph seeds.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Returns
    -------
    TestContext
        Context with CORE_PACK and GRAPH_PACK applied.
    """
    return TestScenario.with_graph().build(tmp_path)


def coverage_context(tmp_path: Path) -> TestContext:
    """Create test context with coverage seeds.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Returns
    -------
    TestContext
        Context with CORE_PACK and COVERAGE_PACK applied.
    """
    return TestScenario.with_coverage().build(tmp_path)


def full_context(tmp_path: Path) -> TestContext:
    """Create test context with all seed packs.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Returns
    -------
    TestContext
        Context with all packs applied.
    """
    return TestScenario.full_stack().build(tmp_path)


__all__ = [
    "SAMPLE_MOD_A",
    "SAMPLE_MOD_B",
    "SAMPLE_MOD_C",
    "SAMPLE_UTIL",
    "ScenarioConfig",
    "TestScenario",
    "coverage_context",
    "full_context",
    "graph_context",
    "minimal_context",
]
