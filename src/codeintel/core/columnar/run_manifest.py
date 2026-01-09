"""Run manifest helpers for columnar pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from codeintel.core.columnar.dedupe_ops import DedupeTier
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.ordering import OrderingLevel, OrderingSpec, SortKey
from codeintel.core.columnar.streaming import ScanTelemetry
from codeintel.core.manifests import ManifestStruct, write_manifest_json

MANIFEST_VERSION = 1


class OrderingManifest(ManifestStruct, frozen=True):
    """Serialized ordering metadata for a run manifest."""

    level: OrderingLevel
    keys: tuple[SortKey, ...] = ()
    pipeline_breaker: bool = False
    reason: str | None = None


class ScanTelemetryManifest(ManifestStruct, frozen=True):
    """Serialized scan telemetry for a run manifest."""

    fragment_count: int | None = None
    estimated_rows: int | None = None


class RunManifest(ManifestStruct, frozen=True):
    """Manifest describing runtime ordering and telemetry."""

    manifest_version: int
    generated_at: str
    arrow_version: str
    determinism: DedupeTier | None = None
    profile_name: str | None = None
    scan_profile: str | None = None
    ordering: OrderingManifest | None = None
    scan_telemetry: ScanTelemetryManifest | None = None
    plan_seconds: float | None = None
    post_seconds: float | None = None
    finalize_seconds: float | None = None
    extras: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class RunManifestOptions:
    """Options for writing a run manifest."""

    determinism: DedupeTier | None = None
    ordering: OrderingSpec | None = None
    scan_telemetry: ScanTelemetry | None = None
    profile_name: str | None = None
    scan_profile: str | None = None
    plan_seconds: float | None = None
    post_seconds: float | None = None
    finalize_seconds: float | None = None
    extras: Mapping[str, object] | None = None
    filename: str = "run_manifest.json"


def write_run_manifest(
    output_dir: Path,
    *,
    options: RunManifestOptions | None = None,
) -> Path:
    """Write a run manifest payload to disk.

    Parameters
    ----------
    output_dir
        Directory to write the manifest into.
    options
        Run manifest options payload.

    Returns
    -------
    pathlib.Path
        Path to the written manifest.
    """
    resolved = options or RunManifestOptions()
    ordering_payload = _ordering_manifest(resolved.ordering)
    telemetry_payload = _scan_telemetry_manifest(resolved.scan_telemetry)
    manifest = RunManifest(
        manifest_version=MANIFEST_VERSION,
        generated_at=_utc_now(),
        arrow_version=str(pa.__version__),
        determinism=resolved.determinism,
        profile_name=resolved.profile_name,
        scan_profile=resolved.scan_profile,
        ordering=ordering_payload,
        scan_telemetry=telemetry_payload,
        plan_seconds=resolved.plan_seconds,
        post_seconds=resolved.post_seconds,
        finalize_seconds=resolved.finalize_seconds,
        extras=resolved.extras,
    )
    path = output_dir / resolved.filename
    write_manifest_json(path, manifest)
    return path


def run_manifest_options_for_context(
    *,
    ctx: ExecutionContext | None,
    ordering: OrderingSpec | None,
    scan_telemetry: ScanTelemetry | None,
    options: RunManifestOptions | None = None,
) -> RunManifestOptions:
    """Return run manifest options with context-derived defaults applied.

    Parameters
    ----------
    ctx
        Optional execution context providing runtime profile defaults.
    ordering
        Ordering metadata from the executed plan.
    scan_telemetry
        Optional scan telemetry metadata.
    options
        Optional base options to overlay.

    Returns
    -------
    RunManifestOptions
        Run manifest options with resolved defaults applied.
    """
    resolved = options or RunManifestOptions()
    determinism = resolved.determinism
    profile_name = resolved.profile_name
    scan_profile = resolved.scan_profile
    if ctx is not None:
        if determinism is None:
            determinism = ctx.resolve_determinism()
        profile = ctx.runtime_profile
        if profile is not None:
            profile_name = profile_name or profile.name
            scan_profile = scan_profile or profile.scan_profile
    return RunManifestOptions(
        determinism=determinism,
        ordering=resolved.ordering or ordering,
        scan_telemetry=resolved.scan_telemetry or scan_telemetry,
        profile_name=profile_name,
        scan_profile=scan_profile,
        plan_seconds=resolved.plan_seconds,
        post_seconds=resolved.post_seconds,
        finalize_seconds=resolved.finalize_seconds,
        extras=resolved.extras,
        filename=resolved.filename,
    )


def _ordering_manifest(ordering: OrderingSpec | None) -> OrderingManifest | None:
    if ordering is None:
        return None
    return OrderingManifest(
        level=ordering.level,
        keys=ordering.keys,
        pipeline_breaker=ordering.pipeline_breaker,
        reason=ordering.reason,
    )


def _scan_telemetry_manifest(
    telemetry: ScanTelemetry | None,
) -> ScanTelemetryManifest | None:
    if telemetry is None:
        return None
    return ScanTelemetryManifest(
        fragment_count=telemetry.fragment_count,
        estimated_rows=telemetry.estimated_rows,
    )


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


__all__ = [
    "MANIFEST_VERSION",
    "OrderingManifest",
    "RunManifest",
    "RunManifestOptions",
    "ScanTelemetryManifest",
    "run_manifest_options_for_context",
    "write_run_manifest",
]
