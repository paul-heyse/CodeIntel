"""Runtime and scan profile definitions for columnar pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypedDict

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.schema import DEFAULT_SCHEMA_PROMOTE_OPTIONS, SchemaPromoteOptions
from codeintel.core.constants import (
    DEFAULT_ARROW_BATCH_READAHEAD,
    DEFAULT_ARROW_BATCH_SIZE,
    DEFAULT_ARROW_CACHE_METADATA,
    DEFAULT_ARROW_FRAGMENT_READAHEAD,
    DEFAULT_ARROW_PARQUET_BUFFER_SIZE,
    DEFAULT_ARROW_PARQUET_PRE_BUFFER,
    DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM,
    DEFAULT_ARROW_USE_THREADS,
)

if TYPE_CHECKING:
    from codeintel.core.columnar.dedupe_ops import DedupeTier
    from codeintel.core.config.settings import ColumnarRuntimeSettings


@dataclass(frozen=True, slots=True)
class DatasetScanOptions:
    """Options for Arrow dataset scanning."""

    batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    batch_readahead: int | None = DEFAULT_ARROW_BATCH_READAHEAD
    fragment_readahead: int | None = DEFAULT_ARROW_FRAGMENT_READAHEAD
    filter_expression: ds.Expression | None = None
    cache_metadata: bool | None = DEFAULT_ARROW_CACHE_METADATA
    use_threads: bool | None = DEFAULT_ARROW_USE_THREADS
    parquet_pre_buffer: bool | None = DEFAULT_ARROW_PARQUET_PRE_BUFFER
    parquet_use_buffered_stream: bool | None = DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM
    parquet_buffer_size: int | None = DEFAULT_ARROW_PARQUET_BUFFER_SIZE
    memory_pool: pa.MemoryPool | None = None
    schema: pa.Schema | None = None
    columns: Sequence[str] | Mapping[str, ds.Expression] | None = None
    provenance_columns: Sequence[str] = ()
    implicit_ordering: bool | None = None
    require_sequenced_output: bool | None = None
    unify_schemas: bool = False
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS
    metrics_enabled: bool = False

    def projection_columns(
        self,
    ) -> Sequence[str] | Mapping[str, ds.Expression] | None:
        """Return projection columns merged with provenance columns.

        Returns
        -------
        Sequence[str] | Mapping[str, ds.Expression] | None
            Projection columns including provenance columns.
        """
        return _merge_scan_columns(self.columns, self.provenance_columns)


@dataclass(frozen=True, slots=True)
class ScanProfile:
    """Named scan profile bundling DatasetScanOptions."""

    name: str
    options: DatasetScanOptions


class ScanProfileOverrides(TypedDict, total=False):
    """Typed overrides for scan profile defaults."""

    batch_size: int
    batch_readahead: int | None
    fragment_readahead: int | None
    use_threads: bool | None


_SCAN_PROFILE_OVERRIDES: dict[str, ScanProfileOverrides] = {
    "dev_fast": {},
    "ci_stable": {
        "use_threads": False,
        "batch_readahead": 4,
        "fragment_readahead": 2,
    },
    "prod_throughput": {},
}


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Defaults for scan/runtime behavior across columnar pipelines."""

    name: str | None = None
    scan_profile: str | None = None
    cpu_threads: int | None = None
    io_threads: int | None = None
    implicit_ordering: bool | None = None
    require_sequenced_output: bool | None = None
    determinism: DedupeTier | None = None
    use_threads: bool | None = None
    plan_use_threads: bool | None = None
    provenance: bool | None = None

    def resolve_plan_use_threads(self, *, default: bool) -> bool:
        """Return the resolved plan use_threads value.

        Returns
        -------
        bool
            Effective plan use_threads toggle.
        """
        if self.plan_use_threads is not None:
            return self.plan_use_threads
        return default if self.use_threads is None else self.use_threads

    def resolve_use_threads(self, *, default: bool) -> bool:
        """Return the resolved plan use_threads value.

        Returns
        -------
        bool
            Effective plan use_threads toggle.
        """
        return self.resolve_plan_use_threads(default=default)

    def resolve_scan_use_threads(self, *, default: bool | None) -> bool | None:
        """Return the resolved scan use_threads value.

        Returns
        -------
        bool | None
            Effective scan use_threads toggle.
        """
        return default if self.use_threads is None else self.use_threads

    def resolve_cpu_threads(self, *, default: int | None) -> int | None:
        """Return the resolved CPU thread count.

        Returns
        -------
        int | None
            Effective CPU thread count.
        """
        return default if self.cpu_threads is None else self.cpu_threads

    def resolve_io_threads(self, *, default: int | None) -> int | None:
        """Return the resolved IO thread count.

        Returns
        -------
        int | None
            Effective IO thread count.
        """
        return default if self.io_threads is None else self.io_threads

    def resolve_scan_profile(self, *, default: str | None) -> str | None:
        """Return the resolved scan profile name.

        Returns
        -------
        str | None
            Effective scan profile name.
        """
        return default if self.scan_profile is None else self.scan_profile

    def resolve_determinism(self, default: DedupeTier) -> DedupeTier:
        """Return the resolved determinism tier.

        Returns
        -------
        DedupeTier
            Effective determinism tier.
        """
        return default if self.determinism is None else self.determinism

    def resolve_provenance(self, *, default: bool) -> bool:
        """Return the resolved provenance toggle.

        Returns
        -------
        bool
            Effective provenance toggle.
        """
        return default if self.provenance is None else self.provenance

    def resolve_implicit_ordering(self, *, default: bool | None) -> bool | None:
        """Return the resolved implicit ordering default.

        Returns
        -------
        bool | None
            Effective implicit ordering toggle.
        """
        return default if self.implicit_ordering is None else self.implicit_ordering

    def resolve_require_sequenced_output(self, *, default: bool | None) -> bool | None:
        """Return the resolved sequenced output default.

        Returns
        -------
        bool | None
            Effective sequenced output toggle.
        """
        return default if self.require_sequenced_output is None else self.require_sequenced_output


DEV_FAST = RuntimeProfile(
    name="DEV_FAST",
    scan_profile="dev_fast",
    cpu_threads=None,
    io_threads=None,
    plan_use_threads=True,
    determinism="stable_set",
    provenance=True,
)
DEV_DETERMINISTIC = RuntimeProfile(
    name="DEV_DETERMINISTIC",
    scan_profile="ci_stable",
    cpu_threads=1,
    io_threads=1,
    plan_use_threads=False,
    determinism="canonical",
    provenance=True,
)
CI_STABLE = RuntimeProfile(
    name="CI_STABLE",
    scan_profile="ci_stable",
    cpu_threads=1,
    io_threads=1,
    plan_use_threads=False,
    determinism="canonical",
    provenance=True,
)
PROD_THROUGHPUT = RuntimeProfile(
    name="PROD_THROUGHPUT",
    scan_profile="prod_throughput",
    cpu_threads=None,
    io_threads=None,
    plan_use_threads=True,
    determinism="throughput",
    provenance=False,
)

_RUNTIME_PROFILE_REGISTRY: dict[str, RuntimeProfile] = {
    "dev_fast": DEV_FAST,
    "dev_deterministic": DEV_DETERMINISTIC,
    "ci_stable": CI_STABLE,
    "prod_throughput": PROD_THROUGHPUT,
}


def scan_profile_options(
    name: str,
    *,
    base: DatasetScanOptions | None = None,
) -> DatasetScanOptions:
    """Return scan options for a named profile.

    Returns
    -------
    DatasetScanOptions
        Scan options with profile overrides applied.

    Raises
    ------
    ValueError
        Raised when the profile name is empty or unknown.
    """
    normalized = name.strip().lower()
    if not normalized:
        msg = "Scan profile name must be non-empty."
        raise ValueError(msg)
    overrides = _SCAN_PROFILE_OVERRIDES.get(normalized)
    if overrides is None:
        msg = f"Unknown scan profile '{normalized}'."
        raise ValueError(msg)
    base_options = base or DatasetScanOptions()
    return _apply_scan_profile_overrides(base_options, overrides)


def runtime_profile_registry() -> dict[str, RuntimeProfile]:
    """Return the runtime profile registry.

    Returns
    -------
    dict[str, RuntimeProfile]
        Mapping of profile names to RuntimeProfile defaults.
    """
    return dict(_RUNTIME_PROFILE_REGISTRY)


def resolve_runtime_profile(name: str | None) -> RuntimeProfile | None:
    """Resolve a runtime profile by name.

    Parameters
    ----------
    name
        Runtime profile name (case-insensitive).

    Returns
    -------
    RuntimeProfile | None
        Runtime profile when resolved, otherwise None.

    Raises
    ------
    ValueError
        Raised when a non-empty name is unknown.
    """
    if name is None:
        return None
    normalized = name.strip().lower()
    if not normalized:
        return None
    profile = _RUNTIME_PROFILE_REGISTRY.get(normalized)
    if profile is None:
        msg = f"Unknown runtime profile '{name}'."
        raise ValueError(msg)
    return profile


def runtime_profile_from_settings(
    settings: ColumnarRuntimeSettings | None,
) -> RuntimeProfile | None:
    """Return the runtime profile derived from columnar runtime settings.

    Returns
    -------
    RuntimeProfile | None
        Resolved runtime profile when configured.
    """
    if settings is None:
        return None
    return resolve_runtime_profile(settings.profile)


def _apply_scan_profile_overrides(
    base_options: DatasetScanOptions,
    overrides: ScanProfileOverrides,
) -> DatasetScanOptions:
    options = base_options
    if "batch_size" in overrides:
        options = replace(options, batch_size=overrides["batch_size"])
    if "batch_readahead" in overrides:
        options = replace(options, batch_readahead=overrides["batch_readahead"])
    if "fragment_readahead" in overrides:
        options = replace(options, fragment_readahead=overrides["fragment_readahead"])
    if "use_threads" in overrides:
        options = replace(options, use_threads=overrides["use_threads"])
    return options


def _merge_scan_columns(
    columns: Sequence[str] | Mapping[str, ds.Expression] | None,
    provenance_columns: Sequence[str],
) -> Sequence[str] | Mapping[str, ds.Expression] | None:
    if not provenance_columns:
        return columns
    if columns is None:
        return tuple(provenance_columns)
    if isinstance(columns, Mapping):
        merged = dict(columns)
        for name in provenance_columns:
            merged.setdefault(name, ds.field(name))
        return merged
    merged = list(columns)
    for name in provenance_columns:
        if name not in columns:
            merged.append(name)
    return tuple(merged)


__all__ = [
    "CI_STABLE",
    "DEV_DETERMINISTIC",
    "DEV_FAST",
    "PROD_THROUGHPUT",
    "DatasetScanOptions",
    "RuntimeProfile",
    "ScanProfile",
    "resolve_runtime_profile",
    "runtime_profile_from_settings",
    "runtime_profile_registry",
    "scan_profile_options",
]
