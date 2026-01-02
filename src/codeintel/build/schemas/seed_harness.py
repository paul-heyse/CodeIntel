"""Seed harness for deterministic schema compilation.

The harness builds q__ inputs from observed Arrow schemas when available,
falling back to declared schemas as needed. Optional dataset scanning can
sample real data when configured.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

import pyarrow as pa

from codeintel.core.columnar.dataset_scanner import (
    empty_reader_from_schema,
    sample_reader,
    scan_dataset_reader,
)
from codeintel.core.columnar.ipc import schema_from_ipc_payload
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.paths import dataset_snapshot_dir
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from types import ModuleType

    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord


def qparam_to_table_key(qparam: str) -> str:
    """Convert a q__ parameter name into a table key.

    Parameters
    ----------
    qparam
        Parameter name in the form ``q__schema__table``.

    Returns
    -------
    str
        Table key in the form ``schema.table``.

    Raises
    ------
    ValueError
        If qparam does not start with ``q__`` or cannot be parsed.
    """
    if not qparam.startswith("q__"):
        msg = f"Expected q__ parameter, got: {qparam}"
        raise ValueError(msg)
    payload = qparam.removeprefix("q__")
    schema, rest = payload.split("__", 1)
    return f"{schema}.{rest}"


def extract_qparams_from_callable(fn: Callable[..., Any]) -> set[str]:
    """Return q__ parameter names declared by a callable.

    Parameters
    ----------
    fn
        Callable to inspect.

    Returns
    -------
    set[str]
        Set of parameter names that begin with ``q__``.
    """
    return {name for name in inspect.signature(fn).parameters if name.startswith("q__")}


def extract_qparams_for_target_module(target: str, module: ModuleType) -> set[str]:
    """Union q__ parameters across functions belonging to a target module.

    This avoids depending on Hamilton internals while remaining robust to a
    target being split into multiple Hamilton nodes.

    Parameters
    ----------
    target
        Target name (e.g., "function_types").
    module
        Python module containing target node functions.

    Returns
    -------
    set[str]
        Union of q__ parameter names across compute functions for the target.
    """
    prefix = f"t__{target}__"
    qparams: set[str] = set()
    for name, obj in vars(module).items():
        if not name.startswith(prefix) or not callable(obj):
            continue
        qparams |= extract_qparams_from_callable(obj)
    return qparams


class SchemaObservationProvider(Protocol):
    """Protocol for resolving schema observation records."""

    def load_latest_schema_observation(
        self,
        *,
        table_key: str,
    ) -> SchemaObservationRecord | None:
        """Return the latest schema observation for a table key."""
        ...


SeedScanMode = Literal["none", "sample"]


@dataclass(frozen=True, slots=True)
class SeedScanSettings:
    """Options for scanning datasets when seeding q__ inputs."""

    mode: SeedScanMode = "none"
    sample_rows: int = DEFAULT_ARROW_BATCH_SIZE
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    fragment_readahead: int | None = None


@dataclass(frozen=True, slots=True)
class SeedDatasetConfig:
    """Configuration for dataset-backed seed scans."""

    dataset_root_dir: Path | None = None
    snapshot_id: str | None = None
    scan_settings: SeedScanSettings = field(default_factory=SeedScanSettings)


@dataclass
class DatasetSeedHarness:
    """Seed upstream inputs using observed Arrow schemas and optional scans."""

    schema_provider: SchemaProvider
    observation_provider: SchemaObservationProvider | None = None
    dataset_root_dir: Path | None = None
    snapshot_id: str | None = None
    scan_settings: SeedScanSettings = field(default_factory=SeedScanSettings)
    _schema_cache: dict[str, pa.Schema] = field(default_factory=dict, repr=False)

    def seed_table(self, table_key: str) -> pa.RecordBatchReader:
        """Create a reader seed for the requested table key.

        Returns
        -------
        pa.RecordBatchReader
            Empty or sampled reader matching the observed/declared schema.
        """
        schema = self._schema_for_table(table_key)
        reader = self._scan_reader(table_key, schema)
        if reader is not None:
            return reader
        return empty_reader_from_schema(schema)

    def seeded_table_keys(self) -> tuple[str, ...]:
        """Return seeded table keys in deterministic order.

        Returns
        -------
        tuple[str, ...]
            Sorted table keys seeded by this harness.
        """
        return tuple(sorted(self._schema_cache))

    def seed_input(self, qparam: str) -> pa.RecordBatchReader:
        """Return a reader seed for a q__ parameter.

        Returns
        -------
        pa.RecordBatchReader
            Reader seeded for the referenced q__ input.
        """
        table_key = qparam_to_table_key(qparam)
        return self.seed_table(table_key)

    def build_inputs(self, qparams: set[str]) -> Mapping[str, pa.RecordBatchReader]:
        """Build a deterministic mapping of q__ inputs for compute execution.

        Returns
        -------
        Mapping[str, pa.RecordBatchReader]
            Mapping from q__ parameter name to seeded readers.
        """
        return {q: self.seed_input(q) for q in sorted(qparams)}

    def _schema_for_table(self, table_key: str) -> pa.Schema:
        cached = self._schema_cache.get(table_key)
        if cached is not None:
            return cached
        observed = self._observed_schema(table_key)
        if observed is not None:
            self._schema_cache[table_key] = observed
            return observed
        table_schema = self.schema_provider.require_table_schema(table_key)
        arrow_schema = arrow_schema_from_table_schema(table_schema=table_schema)
        self._schema_cache[table_key] = arrow_schema
        return arrow_schema

    def _observed_schema(self, table_key: str) -> pa.Schema | None:
        if self.observation_provider is None:
            return None
        try:
            observation = self.observation_provider.load_latest_schema_observation(
                table_key=table_key
            )
        except (RuntimeError, TypeError, ValueError):
            return None
        if observation is None:
            return None
        return schema_from_ipc_payload(observation.arrow_schema_ipc_b64)

    def _scan_reader(self, table_key: str, schema: pa.Schema) -> pa.RecordBatchReader | None:
        if self.scan_settings.mode == "none":
            return None
        if self.dataset_root_dir is None or self.snapshot_id is None:
            return None
        snapshot_dir = dataset_snapshot_dir(
            self.dataset_root_dir,
            table_key=table_key,
            snapshot_id=self.snapshot_id,
        )
        if not snapshot_dir.is_dir():
            return None
        reader = scan_dataset_reader(
            snapshot_dir,
            columns=schema.names,
            batch_size=self.scan_settings.batch_size,
            fragment_readahead=self.scan_settings.fragment_readahead,
        )
        if reader is None:
            return None
        return sample_reader(reader, max_rows=self.scan_settings.sample_rows)


__all__ = [
    "DatasetSeedHarness",
    "SeedDatasetConfig",
    "SeedScanMode",
    "SeedScanSettings",
    "extract_qparams_for_target_module",
    "extract_qparams_from_callable",
    "qparam_to_table_key",
]
