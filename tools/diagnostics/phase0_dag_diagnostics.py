"""Generate Phase 0 DAG diagnostics for tag/schema hygiene."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle
from codeintel.cli.resolution.runtime import resolve_from_params
from codeintel.core.hamilton.tags import TAG_NODE_TYPE
from codeintel.runtime.runtime_bundle import RuntimeBundle
from codeintel.storage.gateway import StorageConfig, open_gateway

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DagDiagnostics:
    """Summary of DAG diagnostics for Phase 0 validation."""

    nodes_missing_tags: tuple[str, ...]
    invalid_identifiers: tuple[str, ...]
    targets_missing_schemas: tuple[str, ...]


def _node_has_type_tag(tags: object) -> bool:
    if not isinstance(tags, dict):
        return False
    return TAG_NODE_TYPE in tags


def _collect_nodes_missing_tags(nodes: Mapping[str, object]) -> tuple[str, ...]:
    missing = [
        name for name, node in nodes.items() if not _node_has_type_tag(getattr(node, "tags", None))
    ]
    return tuple(sorted(missing))


def _collect_invalid_identifiers(nodes: Mapping[str, object]) -> tuple[str, ...]:
    invalid = [name for name in nodes if not name.isidentifier()]
    return tuple(sorted(invalid))


def _compose_runtime_bundle() -> RuntimeBundle:
    runtime = resolve_from_params({"project_root": Path.cwd()})
    config = StorageConfig.for_readonly(runtime.paths.db_path)
    gateway = open_gateway(config)
    try:
        return compose_cli_runtime_bundle(runtime=runtime, gateway=gateway)
    finally:
        gateway.close()


def _collect_targets_missing_schemas(runtime: RuntimeBundle) -> tuple[str, ...]:
    missing: set[str] = set()
    for target in runtime.catalog.all_targets:
        for table_key in target.table_keys:
            if SCHEMA_REGISTRY.get(table_key) is None:
                missing.add(table_key)
    return tuple(sorted(missing))


def _write_json(output_path: Path, diagnostics: DagDiagnostics) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(diagnostics)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Generate diagnostics and write to the default output path."""
    runtime = _compose_runtime_bundle()
    nodes = runtime.dr.graph.nodes
    diagnostics = DagDiagnostics(
        nodes_missing_tags=_collect_nodes_missing_tags(nodes),
        invalid_identifiers=_collect_invalid_identifiers(nodes),
        targets_missing_schemas=_collect_targets_missing_schemas(runtime),
    )
    output_path = Path("build/diagnostics/phase0_dag_diagnostics.json")
    _write_json(output_path, diagnostics)
    LOG.info("Wrote Phase 0 DAG diagnostics to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
