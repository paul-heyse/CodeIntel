"""Generate schema inference inventory diagnostics for Phase 0 readiness."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.schemas.inference_service import inferability_inventory
from codeintel.build.schemas.provider_unified import declared_schema_provider
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle
from codeintel.cli.resolution.runtime import resolve_from_params
from codeintel.storage.gateway import StorageConfig, open_gateway

if TYPE_CHECKING:
    from codeintel.runtime.runtime_bundle import RuntimeBundle

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class InventoryEntry:
    """Single row in the schema inference inventory."""

    table_key: str
    status: str
    inferability_status: str | None
    inferability_reason: str | None
    target_name: str | None
    target_module: str | None
    saver_node: str | None
    compute_node: str | None
    sink: str | None
    qparams: tuple[str, ...] | None
    requires_env: bool | None
    requires_catalog: bool | None


@dataclass(frozen=True)
class InventorySummary:
    """Aggregate counts for the schema inference inventory."""

    outputs_total: int
    inferable_total: int
    override_total: int
    source_only_total: int
    blocked_total: int
    inferability_rate: float | None


@dataclass(frozen=True)
class InventoryReport:
    """Serializable report capturing schema inference inventory state."""

    generated_at: str
    summary: InventorySummary
    entries: tuple[InventoryEntry, ...]


def _compose_runtime_bundle() -> RuntimeBundle:
    runtime = resolve_from_params({"project_root": Path.cwd()})
    config = StorageConfig.for_readonly(runtime.paths.db_path)
    gateway = open_gateway(config)
    try:
        return compose_cli_runtime_bundle(runtime=runtime, gateway=gateway)
    finally:
        gateway.close()


def _schema_status(*, table_key: str, inferable: frozenset[str]) -> str:
    return "inferable" if table_key in inferable else "override"


def _build_entries(report_runtime: RuntimeBundle) -> list[InventoryEntry]:
    schema_index = report_runtime.schema_index
    if schema_index is None:
        msg = "RuntimeBundle.schema_index is required to build inference inventory."
        raise ValueError(msg)
    inferable = schema_index.inferable_table_keys
    target_cache = report_runtime.catalog.targets
    entries: list[InventoryEntry] = []

    for record in inferability_inventory(
        driver=report_runtime.driver,
        catalog=report_runtime.catalog,
    ):
        target = target_cache.get(record.target_name)
        entries.append(
            InventoryEntry(
                table_key=record.table_key,
                status=_schema_status(table_key=record.table_key, inferable=inferable),
                inferability_status=record.status,
                inferability_reason=record.reason,
                target_name=record.target_name,
                target_module=target.module if target is not None else None,
                saver_node=record.saver_node,
                compute_node=record.compute_node,
                sink=record.sink,
                qparams=record.qparams,
                requires_env=record.requires_env,
                requires_catalog=record.requires_catalog,
            )
        )

    source_provider = declared_schema_provider(runtime=report_runtime)
    entries.extend(
        InventoryEntry(
            table_key=schema.table_key,
            status="source_only",
            inferability_status=None,
            inferability_reason=None,
            target_name=None,
            target_module=None,
            saver_node=None,
            compute_node=None,
            sink=None,
            qparams=None,
            requires_env=None,
            requires_catalog=None,
        )
        for schema in source_provider.iter_table_schemas()
    )

    return entries


def _summary(entries: tuple[InventoryEntry, ...], outputs_total: int) -> InventorySummary:
    inferable_total = sum(1 for entry in entries if entry.status == "inferable")
    override_total = sum(1 for entry in entries if entry.status == "override")
    source_only_total = sum(1 for entry in entries if entry.status == "source_only")
    blocked_total = sum(1 for entry in entries if entry.inferability_status == "non_inferable")
    inferability_rate = None
    if outputs_total > 0:
        inferability_rate = inferable_total / outputs_total
    return InventorySummary(
        outputs_total=outputs_total,
        inferable_total=inferable_total,
        override_total=override_total,
        source_only_total=source_only_total,
        blocked_total=blocked_total,
        inferability_rate=inferability_rate,
    )


def _write_report(report: InventoryReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Generate and write the schema inference inventory."""
    report_runtime = _compose_runtime_bundle()
    entries = _build_entries(report_runtime)
    entries_sorted = tuple(sorted(entries, key=lambda entry: entry.table_key))
    summary = _summary(entries_sorted, outputs_total=len(report_runtime.catalog.table_outputs))
    report = InventoryReport(
        generated_at=datetime.now(tz=UTC).isoformat(),
        summary=summary,
        entries=entries_sorted,
    )
    output_path = Path("build/diagnostics/schema_inference_inventory.json")
    _write_report(report, output_path)
    LOG.info("Wrote schema inference inventory to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
