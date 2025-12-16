"""Native Hamilton implementation for config target.

This module implements config ingestion as a native Hamilton pipeline with:
- t__config__scan: Discover config files using profile
- t__config__ingest: Execute ConfigIngestStep to flatten configs
- t__config: Materialize with validators and return TargetRunRecord

Phase 2: Ingestion domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import ConfigIngestStep
from codeintel.ingestion.infrastructure.scanning import default_config_profile

if TYPE_CHECKING:
    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class ConfigScanResult:
    """Result from config file discovery.

    Attributes
    ----------
    success
        Whether discovery completed successfully.
    config_files
        List of discovered config files.
    error
        Error message if discovery failed.
    """

    success: bool
    config_files: list[ModuleRecord] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ConfigIngestResult:
    """Result from config ingestion.

    Attributes
    ----------
    success
        Whether ingestion completed successfully.
    table_counts
        Row counts per produced table.
    errors
        List of parse errors (non-fatal).
    error
        Fatal error message if ingestion failed.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    error: str | None = None


@tag(domain="ingestion", target="config_ingest", node_type="compute")
def t__config_ingest__scan(env: BuildEnv) -> ConfigScanResult:
    """Discover config files in repository.

    This compute node scans the repository for configuration files
    (YAML, JSON, TOML, INI) using the default config profile.

    Parameters
    ----------
    env
        Build environment with snapshot.

    Returns
    -------
    ConfigScanResult
        Result containing discovered config files.
    """
    try:
        profile = default_config_profile(env.snapshot.repo_root)
        config_files = list(
            FilesystemDiscoveryAdapter.discover_modules(env.snapshot.repo_root, profile)
        )

        if not config_files:
            log.info("No config files found matching profile")

        return ConfigScanResult(
            success=True,
            config_files=config_files,
        )

    except Exception:
        log.exception("Config file discovery failed")
        return ConfigScanResult(
            success=False,
            error="Config file discovery failed with exception",
        )


@tag(domain="ingestion", target="config_ingest", node_type="compute")
def t__config_ingest__ingest(
    env: BuildEnv,
    t__config_ingest__scan: ConfigScanResult,
) -> ConfigIngestResult:
    """Execute config ingestion to flatten config files.

    This is the main compute node for the config target. It reads
    various configuration files and flattens their structure into
    key-value pairs.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__config_ingest__scan
        Upstream scan result with discovered config files.

    Returns
    -------
    ConfigIngestResult
        Result containing table row counts and any parse errors.

    Notes
    -----
    Produces:
    - core.config_values: Flattened config key-value pairs
    """
    if not t__config_ingest__scan.success:
        return ConfigIngestResult(
            success=False,
            error=f"Config scan failed: {t__config_ingest__scan.error}",
        )

    config_files = t__config_ingest__scan.config_files
    if not config_files:
        return ConfigIngestResult(
            success=True,
            table_counts={},
        )

    try:
        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)

        step = ConfigIngestStep(storage=storage, discovery=discovery)
        result = step.execute(
            config_files,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        # Check if there were any errors
        if result.errors and result.rows_written == 0:
            errors = "; ".join(result.errors)
            return ConfigIngestResult(
                success=False,
                error=f"Config ingest failed: {errors}",
            )

        return ConfigIngestResult(
            success=True,
            table_counts=result.table_counts or {},
            errors=list(result.errors) if result.errors else [],
        )

    except Exception:
        log.exception("Config ingestion failed")
        return ConfigIngestResult(
            success=False,
            error="Config ingestion failed with exception",
        )


@tag(domain="ingestion", target="config_ingest", node_type="materialize")
def t__config_ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__config_ingest__ingest: ConfigIngestResult,
) -> TargetRunRecord:
    """Materialize config target with validation.

    This is the entry point for the config target. It orchestrates
    config ingestion and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__config_ingest__ingest
        Ingestion result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "config_ingest")

    if executor.should_skip():
        return executor.skip()

    if not t__config_ingest__ingest.success:
        return executor.fail(
            RuntimeError(t__config_ingest__ingest.error or "Config ingestion failed")
        )

    for error in t__config_ingest__ingest.errors:
        log.warning("Config parse warning: %s", error)

    def compute() -> dict[str, int]:
        return dict(t__config_ingest__ingest.table_counts)

    return executor.execute(compute)


__all__ = [
    "ConfigIngestResult",
    "ConfigScanResult",
    "t__config_ingest",
    "t__config_ingest__ingest",
    "t__config_ingest__scan",
]
