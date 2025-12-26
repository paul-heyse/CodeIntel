"""Manifest priming helpers for Hamilton build tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.run_records import compute_target_input_hash, options_hash_for_target
from codeintel.build.hashing import InputHashOptions
from codeintel.core.build_manifest import BuildRunRecord, OutputManifest

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv


class _HarnessProtocol(Protocol):
    """Minimal protocol for manifest priming."""

    def build_env(self) -> BuildEnv: ...


@dataclass(frozen=True)
class ManifestPriming:
    """Insert manifests and minimal state for skip logic tests."""

    harness: _HarnessProtocol

    @dataclass(frozen=True)
    class ManifestSpec:
        """Specification for a manifest to be written."""

        target: str
        input_hash: str
        options_hash: str | None
        duration_ms: float = 0.0
        plugin: str | None = None
        row_count: int | None = None
        change_delta: dict[str, object] | None = None
        computed_at: datetime | None = None

    def prime_manifest(self, spec: ManifestSpec) -> OutputManifest:
        """Insert an OutputManifest row for a target.

        Parameters
        ----------
        spec
            Manifest specification to persist.

        Returns
        -------
        OutputManifest
            Saved manifest record.
        """
        env = self.harness.build_env()
        when = spec.computed_at or datetime.now(tz=UTC)
        manifest = OutputManifest(
            target=spec.target,
            repo=env.repo,
            commit=env.commit,
            impl_kind=spec.plugin or f"native:{spec.target}",
            computed_at=when,
            duration_ms=spec.duration_ms,
            input_hash=spec.input_hash,
            output_hash=None,
            row_count=spec.row_count,
            options_hash=spec.options_hash,
            dep_hashes=None,
            change_delta=spec.change_delta,
        )
        env.gateway.build.save_manifest(manifest)
        return manifest

    def prime_modules_manifest(
        self,
        *,
        file_state_hash: str,
        row_count: int | None = None,
        change_delta: dict[str, object] | None = None,
    ) -> OutputManifest:
        """Prime the modules manifest for deterministic skip behavior.

        Parameters
        ----------
        file_state_hash
            File state hash used to compute the modules input hash.
        row_count
            Optional row count to store on the manifest.
        change_delta
            Optional change delta payload to store.

        Returns
        -------
        OutputManifest
            Saved manifest record for modules.

        Raises
        ------
        RuntimeError
            Raised when the modules target is missing from the graph.
        """
        env = self.harness.build_env()
        runtime = build_driver(config={"profile": env.profile})
        target = runtime.graph.get("modules")
        if target is None:
            message = "Target 'modules' not found in target graph"
            raise RuntimeError(message)

        opts_hash = options_hash_for_target(env, "modules")
        input_hash = compute_target_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=InputHashOptions(
                options_hash=opts_hash,
                file_state_hash=file_state_hash,
                manifests=None,
            ),
        )
        spec = self.ManifestSpec(
            target="modules",
            input_hash=input_hash,
            options_hash=opts_hash,
            row_count=row_count,
            change_delta=change_delta,
        )
        return self.prime_manifest(spec)

    def prime_target_manifest(
        self,
        target: str,
        *,
        file_state_hash: str | None = None,
        row_count: int | None = None,
        change_delta: dict[str, object] | None = None,
    ) -> OutputManifest:
        """Prime a manifest for an arbitrary target.

        Parameters
        ----------
        target
            Target name to prime.
        file_state_hash
            Optional file state hash used in input hash computation.
        row_count
            Optional row count to store on the manifest.
        change_delta
            Optional change delta payload to store.

        Returns
        -------
        OutputManifest
            Saved manifest record for the target.

        Raises
        ------
        RuntimeError
            If the target is not found in the target graph.
        """
        env = self.harness.build_env()
        runtime = build_driver(config={"profile": env.profile})
        node = runtime.graph.get(target)
        if node is None:
            message = f"Target '{target}' not found in target graph"
            raise RuntimeError(message)

        opts_hash = options_hash_for_target(env, target)
        input_hash = compute_target_input_hash(
            target=node,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=InputHashOptions(
                options_hash=opts_hash,
                file_state_hash=file_state_hash,
                manifests=None,
            ),
        )
        spec = self.ManifestSpec(
            target=target,
            input_hash=input_hash,
            options_hash=opts_hash,
            row_count=row_count,
            change_delta=change_delta,
        )
        return self.prime_manifest(spec)

    def prime_target_as_upstream_success(
        self,
        target: str,
        *,
        file_state_hash: str | None = None,
        row_count: int | None = None,
        change_delta: dict[str, object] | None = None,
        run_id: str | None = None,
    ) -> OutputManifest:
        """Prime a target manifest and a successful run record.

        Parameters
        ----------
        target
            Target name to prime.
        file_state_hash
            Optional file state hash for input hash computation.
        row_count
            Optional row count to store on the manifest.
        change_delta
            Optional change delta payload to store.
        run_id
            Optional run ID; defaults to a generated UUID.

        Returns
        -------
        OutputManifest
            Saved manifest record for the target.
        """
        manifest = self.prime_target_manifest(
            target,
            file_state_hash=file_state_hash,
            row_count=row_count,
            change_delta=change_delta,
        )
        env = self.harness.build_env()
        started_at = manifest.computed_at
        resolved_run_id = run_id or f"primed-{target}-{uuid4().hex[:8]}"
        env.gateway.build.start_run(
            BuildRunRecord(
                run_id=resolved_run_id,
                repo=env.repo,
                commit=env.commit,
                requested_targets=(target,),
                computed_targets=(),
                skipped_targets=(),
                started_at=started_at,
                status="running",
            )
        )
        env.gateway.build.complete_run(
            run_id=resolved_run_id,
            status="succeeded",
            computed_targets=(target,),
            skipped_targets=(),
        )
        return manifest


__all__ = ["ManifestPriming"]
