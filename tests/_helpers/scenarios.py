"""Scenario builder for declarative test setup.

Uses canonical environment/repo builders to keep fixtures parallel-safe and
aligned with production wiring. Seed packs remain composable and idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from tests._helpers.env import create_test_env
from tests._helpers.env_options import EnvOptions
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT, SnapshotVariant
from tests._helpers.seeds import CORE_PACK, COVERAGE_PACK, GRAPH_PACK, METRICS_PACK
from tests._helpers.seeds.ast_metrics import AST_METRICS_PACK
from tests._helpers.seeds.config import CONFIG_PACK
from tests._helpers.seeds.function_types import FunctionTypesPack
from tests._helpers.seeds.profile import ProfilePack

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.context import SeedPack, TestContext


@dataclass
class ScenarioConfig:
    """Configuration for a test scenario."""

    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT
    seed_packs: list[SeedPack] = field(default_factory=list)
    file_backed: bool = False
    write_files: bool = False
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class TestScenario:
    """Fluent builder for test scenarios."""

    __test__ = False

    config: ScenarioConfig = field(default_factory=ScenarioConfig)

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
    def with_function_types(cls) -> TestScenario:
        """Create scenario with core and function types seeds.

        Returns
        -------
        TestScenario
            Scenario configured with CORE_PACK and FunctionTypesPack.
        """
        scenario = cls()
        scenario.config.seed_packs.extend([CORE_PACK, FunctionTypesPack()])
        return scenario

    @classmethod
    def with_ast_metrics(cls) -> TestScenario:
        """Create scenario with core data and AST metrics seeds.

        Returns
        -------
        TestScenario
            Scenario configured with CORE_PACK and AST metrics pack.
        """
        scenario = cls()
        scenario.config.seed_packs.extend([CORE_PACK, AST_METRICS_PACK])
        return scenario

    @classmethod
    def with_profiles(cls) -> TestScenario:
        """Create scenario with comprehensive profile seeds.

        Returns
        -------
        TestScenario
            Scenario configured with ProfilePack.
        """
        scenario = cls()
        scenario.config.seed_packs.extend([ProfilePack()])
        return scenario

    @classmethod
    def with_dependencies(cls) -> TestScenario:
        """Create scenario with configuration/dependency seeds.

        Returns
        -------
        TestScenario
            Scenario configured with config seeds and core data.
        """
        scenario = cls()
        scenario.config.seed_packs.extend([CORE_PACK, CONFIG_PACK])
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

    def with_snapshot_variant(self, variant: SnapshotVariant) -> Self:
        """Set snapshot variant for this scenario.

        Returns
        -------
        Self
            Self for chaining.
        """
        self.config.snapshot_variant = variant
        return self

    def with_seeds(self, *packs: SeedPack) -> Self:
        """Add seed packs to apply.

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
        """Write canonical sample Python files to repo directory.

        Returns
        -------
        Self
            Self for chaining.
        """
        self.config.write_files = True
        return self

    def with_extra(self, key: str, value: object) -> Self:
        """Add extra configuration value.

        Returns
        -------
        Self
            Self for chaining.
        """
        self.config.extra[key] = value
        return self

    def build(self, tmp_path: Path) -> TestContext:
        """Build the test context from configuration (parallel-safe).

        Returns
        -------
        TestContext
            Configured and seeded test context.
        """
        ctx = create_test_env(
            tmp_path,
            options=EnvOptions(
                file_backed=self.config.file_backed,
                snapshot_variant=self.config.snapshot_variant,
            ),
        )

        ctx.extra.update(self.config.extra)

        if self.config.write_files:
            ctx.ensure_canonical_repo()

        for pack in self.config.seed_packs:
            ctx.require(pack)

        return ctx


def minimal_context(tmp_path: Path) -> TestContext:
    """Create minimal test context with core seeds.

    Returns
    -------
    TestContext
        Context with CORE_PACK applied.
    """
    return TestScenario.minimal().build(tmp_path)


def graph_context(tmp_path: Path) -> TestContext:
    """Create test context with graph seeds.

    Returns
    -------
    TestContext
        Context with CORE_PACK and GRAPH_PACK applied.
    """
    return TestScenario.with_graph().build(tmp_path)


def coverage_context(tmp_path: Path) -> TestContext:
    """Create test context with coverage seeds.

    Returns
    -------
    TestContext
        Context with CORE_PACK and COVERAGE_PACK applied.
    """
    return TestScenario.with_coverage().build(tmp_path)


def full_context(tmp_path: Path) -> TestContext:
    """Create test context with all seed packs.

    Returns
    -------
    TestContext
        Context with all packs applied.
    """
    return TestScenario.full_stack().build(tmp_path)


__all__ = [
    "ScenarioConfig",
    "TestScenario",
    "coverage_context",
    "full_context",
    "graph_context",
    "minimal_context",
]
