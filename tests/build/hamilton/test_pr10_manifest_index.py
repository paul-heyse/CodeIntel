"""Tests for PR-10: Manifest index prefetch and hash cascade.

Validates that BuildEnv carries manifest index and hash computation
uses cascading semantics properly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hashing import compute_input_hash
from codeintel.build.manifest import OutputManifest
from codeintel.build.targets import OutputTarget
from tests._helpers.build import make_build_config, make_build_paths, make_snapshot
from tests._helpers.fakes.fake_providers import FakeProviders

if TYPE_CHECKING:
    from pathlib import Path

    from tests.build.hamilton.conftest import FakeGateway


class TestBuildEnvManifestIndex:
    """Tests for BuildEnv manifest_index field."""

    @staticmethod
    def test_build_env_accepts_manifest_index(tmp_path: Path) -> None:
        """Verify BuildEnv has manifest_index field."""
        manifest = OutputManifest(
            target="modules",
            repo="test/repo",
            commit="abc123",
            plugin="ingestion.modules",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="hash123",
            output_hash="out123",
        )
        manifest_index = {"modules": manifest}

        env = BuildEnv(
            gateway=None,  # type: ignore[arg-type]
            snapshot=make_snapshot(tmp_path),
            paths=make_build_paths(tmp_path),
            providers=FakeProviders.defaults(),  # type: ignore[arg-type]
            config=make_build_config(),
            manifest_index=manifest_index,
        )

        if env.manifest_index is None:
            pytest.fail("BuildEnv.manifest_index should be set")
        if "modules" not in env.manifest_index:
            pytest.fail("manifest_index should contain 'modules'")

    @staticmethod
    def test_manifest_index_default_is_none(tmp_path: Path) -> None:
        """Verify manifest_index defaults to None when not provided."""
        env = BuildEnv(
            gateway=None,  # type: ignore[arg-type]
            snapshot=make_snapshot(tmp_path),
            paths=make_build_paths(tmp_path),
            providers=FakeProviders.defaults(),  # type: ignore[arg-type]
            config=make_build_config(),
        )

        if env.manifest_index is not None:
            pytest.fail("manifest_index should default to None")


class TestHashComputation:
    """Tests for hash computation using manifest index."""

    @staticmethod
    def test_hash_computation_uses_manifest_index(
        fake_gateway: FakeGateway,
        tmp_path: Path,
    ) -> None:
        """Verify hash computation uses manifest_index when provided.

        Raises
        ------
        RuntimeError
            Propagated if the fake gateway attempts an unexpected load_manifest call.
        """
        # Create manifest for dependency
        dep_manifest = OutputManifest(
            target="dep",
            repo="test/repo",
            commit="abc123",
            plugin="ingestion.dep",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="dep_input_hash",
            output_hash="dep_output_hash",
        )

        # Set up gateway to track load_manifest calls
        fake_gateway.build.manifests["dep"] = dep_manifest
        fake_gateway.build.raise_on_load = True  # Fail if load_manifest is called

        # Create target with dependency
        target = OutputTarget(
            name="main",
            module="analytics",
            plugin="analytics.main",
            dependencies=("dep",),
        )

        snapshot = make_snapshot(tmp_path, repo="test/repo", commit="abc123")
        manifest_index = {"dep": dep_manifest}

        # compute_input_hash should use manifest_index and not call load_manifest
        try:
            hash_value = compute_input_hash(
                target,
                snapshot,
                fake_gateway,  # type: ignore[arg-type]
                options_hash="opts",
                manifests=manifest_index,
            )
        except RuntimeError as e:
            if "load_manifest called" in str(e):
                pytest.fail("compute_input_hash should use manifest_index")
            raise

        if not hash_value:
            pytest.fail("Hash computation should return a value")

    @staticmethod
    def test_hash_cascade_changes_downstream(
        fake_gateway: FakeGateway,
        tmp_path: Path,
    ) -> None:
        """Verify changing upstream hash changes downstream input_hash."""
        snapshot = make_snapshot(tmp_path, repo="test/repo", commit="abc123")

        target = OutputTarget(
            name="downstream",
            module="analytics",
            plugin="analytics.downstream",
            dependencies=("upstream",),
        )

        # First manifest state
        manifest_v1 = OutputManifest(
            target="upstream",
            repo="test/repo",
            commit="abc123",
            plugin="ingestion.upstream",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="hash_v1",
            output_hash="out_v1",
        )

        hash_1 = compute_input_hash(
            target,
            snapshot,
            fake_gateway,  # type: ignore[arg-type]
            options_hash="opts",
            manifests={"upstream": manifest_v1},
        )

        # Second manifest state - different input_hash
        manifest_v2 = OutputManifest(
            target="upstream",
            repo="test/repo",
            commit="abc123",
            plugin="ingestion.upstream",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="hash_v2",  # Changed!
            output_hash="out_v2",
        )

        hash_2 = compute_input_hash(
            target,
            snapshot,
            fake_gateway,  # type: ignore[arg-type]
            options_hash="opts",
            manifests={"upstream": manifest_v2},
        )

        if hash_1 == hash_2:
            pytest.fail(
                f"Downstream hash should change when upstream hash changes: "
                f"v1={hash_1}, v2={hash_2}"
            )


class TestSkipCheckManifestIndex:
    """Tests for skip check using manifest index."""

    @staticmethod
    def test_skip_check_uses_manifest_index(
        fake_gateway: FakeGateway,
    ) -> None:
        """Verify should_skip uses manifest_index when provided.

        Raises
        ------
        RuntimeError
            Propagated if the fake gateway attempts an unexpected load_manifest call.
        """
        from codeintel.build.hamilton.manifest_hook import should_skip  # noqa: PLC0415

        # Create matching manifest
        manifest = OutputManifest(
            target="modules",
            repo="test/repo",
            commit="abc123",
            plugin="ingestion.modules",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="exact_hash",
            output_hash="out",
        )

        # Set up gateway to fail if load_manifest is called
        fake_gateway.build.raise_on_load = True

        manifest_index = {"modules": manifest}

        # should_skip should use manifest_index
        try:
            result = should_skip(
                target="modules",
                repo="test/repo",
                commit="abc123",
                gateway=fake_gateway,  # type: ignore[arg-type]
                input_hash="exact_hash",
                manifest_index=manifest_index,
            )
        except RuntimeError as e:
            if "load_manifest called" in str(e):
                pytest.fail("should_skip should use manifest_index")
            raise

        # With matching hash, should return True (skip)
        if not result:
            pytest.fail("should_skip should return True when hashes match")


class TestManifestPrefetch:
    """Tests for manifest prefetch optimization."""

    @staticmethod
    def test_manifest_index_eliminates_per_target_loads(
        fake_gateway: FakeGateway,
        tmp_path: Path,
    ) -> None:
        """Verify planner uses manifest_index without per-target loads.

        When manifest_index is provided, compute_plan should not call
        load_manifest for individual targets.

        Raises
        ------
        RuntimeError
            Propagated if the fake gateway attempts unexpected manifest loads.
        """
        from codeintel.build.hamilton.env import BuildEnv  # noqa: PLC0415
        from codeintel.build.hamilton.planner import compute_plan  # noqa: PLC0415
        from codeintel.build.targets import TargetGraph  # noqa: PLC0415

        # Create manifests for all targets in graph
        manifest_a = OutputManifest(
            target="a",
            repo="test/repo",
            commit="abc123",
            plugin="ingestion.a",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="hash_a",
            output_hash="out_a",
        )
        manifest_b = OutputManifest(
            target="b",
            repo="test/repo",
            commit="abc123",
            plugin="graphs.b",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="hash_b",
            output_hash="out_b",
        )

        # Pre-load manifest index
        manifest_index = {"a": manifest_a, "b": manifest_b}

        # Configure gateway to fail on individual loads
        fake_gateway.build.raise_on_load = True

        # Build minimal graph
        graph = TargetGraph()
        graph.register(OutputTarget(name="a", module="ingestion", plugin="ingestion.a"))
        graph.register(
            OutputTarget(name="b", module="graphs", plugin="graphs.b", dependencies=("a",))
        )

        env = BuildEnv(
            gateway=fake_gateway,  # type: ignore[arg-type]
            snapshot=make_snapshot(tmp_path, repo="test/repo", commit="abc123"),
            paths=make_build_paths(tmp_path),
            providers=FakeProviders.defaults(),  # type: ignore[arg-type]
            config=make_build_config(),
            manifest_index=manifest_index,
        )

        # compute_plan should succeed using manifest_index
        try:
            plan = compute_plan(
                env=env,
                graph=graph,
                requested=("b",),
                mode="generated",
            )
        except RuntimeError as e:
            if "load_manifest called" in str(e):
                pytest.fail("compute_plan should use manifest_index, not load_manifest")
            raise

        # Verify plan was computed
        if len(plan.entries) == 0:
            pytest.fail("Plan should have entries")


class TestHashCascadeComplete:
    """Tests for complete hash cascade semantics."""

    @staticmethod
    def test_hash_cascade_through_multiple_levels(
        fake_gateway: FakeGateway,
        tmp_path: Path,
    ) -> None:
        """Verify hash changes cascade through multi-level dependency chain.

        Chain: a -> b -> c
        Changing a's hash should affect both b and c's input hashes.
        """
        snapshot = make_snapshot(tmp_path, repo="test/repo", commit="abc123")

        target_b = OutputTarget(
            name="b",
            module="graphs",
            plugin="graphs.b",
            dependencies=("a",),
        )
        target_c = OutputTarget(
            name="c",
            module="analytics",
            plugin="analytics.c",
            dependencies=("b",),
        )

        # Initial state
        manifest_a_v1 = OutputManifest(
            target="a",
            repo="test/repo",
            commit="abc123",
            plugin="ingestion.a",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="a_hash_v1",
            output_hash="a_out_v1",
        )
        manifest_b_v1 = OutputManifest(
            target="b",
            repo="test/repo",
            commit="abc123",
            plugin="graphs.b",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="b_hash_v1",
            output_hash="b_out_v1",
        )

        manifests_v1 = {"a": manifest_a_v1, "b": manifest_b_v1}

        hash_b_v1 = compute_input_hash(
            target_b,
            snapshot,
            fake_gateway,  # type: ignore[arg-type]
            options_hash="opts",
            manifests=manifests_v1,
        )
        hash_c_v1 = compute_input_hash(
            target_c,
            snapshot,
            fake_gateway,  # type: ignore[arg-type]
            options_hash="opts",
            manifests=manifests_v1,
        )

        # Change root (a) hash
        manifest_a_v2 = OutputManifest(
            target="a",
            repo="test/repo",
            commit="abc123",
            plugin="ingestion.a",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="a_hash_v2",  # Changed!
            output_hash="a_out_v2",
        )
        manifest_b_v2 = OutputManifest(
            target="b",
            repo="test/repo",
            commit="abc123",
            plugin="graphs.b",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="b_hash_v2",  # Would change due to a
            output_hash="b_out_v2",
        )

        manifests_v2 = {"a": manifest_a_v2, "b": manifest_b_v2}

        hash_b_v2 = compute_input_hash(
            target_b,
            snapshot,
            fake_gateway,  # type: ignore[arg-type]
            options_hash="opts",
            manifests=manifests_v2,
        )
        hash_c_v2 = compute_input_hash(
            target_c,
            snapshot,
            fake_gateway,  # type: ignore[arg-type]
            options_hash="opts",
            manifests=manifests_v2,
        )

        # b's hash should change (direct dependency on a)
        if hash_b_v1 == hash_b_v2:
            pytest.fail("b's hash should change when a changes")

        # c's hash should change (transitive dependency through b)
        if hash_c_v1 == hash_c_v2:
            pytest.fail("c's hash should change when a changes (cascade)")

    @staticmethod
    def test_options_hash_affects_input_hash(
        fake_gateway: FakeGateway,
        tmp_path: Path,
    ) -> None:
        """Verify options_hash contributes to input_hash."""
        snapshot = make_snapshot(tmp_path, repo="test/repo", commit="abc123")

        target = OutputTarget(
            name="target",
            module="analytics",
            plugin="analytics.target",
        )

        hash_1 = compute_input_hash(
            target,
            snapshot,
            fake_gateway,  # type: ignore[arg-type]
            options_hash="opts_v1",
            manifests={},
        )

        hash_2 = compute_input_hash(
            target,
            snapshot,
            fake_gateway,  # type: ignore[arg-type]
            options_hash="opts_v2",  # Different options
            manifests={},
        )

        if hash_1 == hash_2:
            pytest.fail("input_hash should change when options_hash changes")
