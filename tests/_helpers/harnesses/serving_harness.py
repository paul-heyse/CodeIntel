"""Serving target harness helpers for Hamilton execution tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

from codeintel.build.hamilton.native.export.serving_artifacts import (
    SERVING_ARTIFACT_BUILDSPEC,
    SERVING_ARTIFACT_SCHEMA_MANIFEST,
    SERVING_ARTIFACT_SEMANTIC_REGISTRY,
    SERVING_ARTIFACTS_TARGET_NAME,
)
from codeintel.build.serving.publisher import (
    PublishServingSnapshotRequest,
    publish_serving_snapshot,
)
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
    HarnessOpenOptions,
)
from tests._helpers.orchestration.repo_writers import write_sample_repo

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.hamilton.run_records import TargetRunRecord
    from codeintel.build.serving.manifest import ServingSnapshotManifest


@dataclass
class ServingTargetHarness:
    """Harness wrapper for serving artifact and snapshot publishing tests."""

    harness: HamiltonBuildHarness

    @classmethod
    def open(
        cls,
        tmp_path: Path,
        *,
        harness_config: HarnessConfig | None = None,
        options: HarnessOpenOptions | None = None,
    ) -> ServingTargetHarness:
        """Create a serving harness with a file-backed gateway.

        Returns
        -------
        ServingTargetHarness
            Harness wrapper bound to a HamiltonBuildHarness.
        """
        config = harness_config or HarnessConfig(repo="test/repo", commit="deadbeef")
        config = HarnessConfig(
            repo=config.repo,
            commit=config.commit,
            profile=config.profile,
            file_backed_db=True,
            strict_contracts=config.strict_contracts,
            validate_outputs=config.validate_outputs,
            parallel_backend=config.parallel_backend,
            max_workers=config.max_workers,
            enable_hamilton_cache=config.enable_hamilton_cache,
            cache_dir=config.cache_dir,
        )
        resolved = options or HarnessOpenOptions(
            repo_strategy="writer",
            repo_writer=write_sample_repo,
        )
        base = HamiltonBuildHarness.open(tmp_path, harness=config, options=resolved)
        return cls(base)

    def run_targets(self, targets: Iterable[str] | None = None) -> dict[str, TargetRunRecord]:
        """Run serving targets and return records by target name.

        Returns
        -------
        dict[str, TargetRunRecord]
            Mapping of target name to TargetRunRecord.
        """
        requested = tuple(targets or (SERVING_ARTIFACTS_TARGET_NAME,))
        result = self.harness.run_targets(requested)
        return {target: self.harness.record(target, result=result) for target in requested}

    def publish_snapshot(
        self,
        *,
        run_id: str = "test-run",
        keep_last: int = 2,
    ) -> ServingSnapshotManifest:
        """Publish a serving snapshot using compiled artifacts.

        Returns
        -------
        ServingSnapshotManifest
            Published serving snapshot manifest.
        """
        record = self.harness.record(SERVING_ARTIFACTS_TARGET_NAME)
        artifact_paths = {artifact.name: artifact.path for artifact in record.artifacts}
        semantic_registry = _require_path(artifact_paths, SERVING_ARTIFACT_SEMANTIC_REGISTRY)
        schema_manifest = _require_path(artifact_paths, SERVING_ARTIFACT_SCHEMA_MANIFEST)
        buildspec = _require_path(artifact_paths, SERVING_ARTIFACT_BUILDSPEC)

        serve_dir = self.harness.ctx.build_paths.build_dir / "serving"
        request = PublishServingSnapshotRequest(
            run_id=run_id,
            serve_dir=serve_dir,
            semantic_registry_path=Path(semantic_registry),
            schema_manifest_path=Path(schema_manifest),
            buildspec_path=Path(buildspec),
            keep_last=keep_last,
        )
        return publish_serving_snapshot(gateway=self.harness.ctx.gateway, request=request)

    def assert_artifacts_exist(self) -> None:
        """Assert serving artifacts exist on disk.

        Raises
        ------
        AssertionError
            If an artifact is missing a path or file on disk.
        """
        record = self.harness.record(SERVING_ARTIFACTS_TARGET_NAME)
        for artifact in record.artifacts:
            if artifact.path is None:
                message = f"Artifact {artifact.name} missing path"
                raise AssertionError(message)
            if not Path(artifact.path).is_file():
                message = f"Artifact {artifact.name} missing at {artifact.path}"
                raise AssertionError(message)

    def close(self) -> None:
        """Close the underlying HamiltonBuildHarness."""
        self.harness.close()

    def __enter__(self) -> Self:
        """Return the harness for context manager usage.

        Returns
        -------
        Self
            This harness instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.close()


def _require_path(paths: dict[str, str | None], name: str) -> str:
    path = paths.get(name)
    if path is None:
        message = f"Missing artifact path for {name}"
        raise AssertionError(message)
    return path


__all__ = [
    "ServingTargetHarness",
]
