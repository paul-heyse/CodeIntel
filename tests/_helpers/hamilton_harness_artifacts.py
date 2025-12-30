"""Artifact helpers for Hamilton build tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.config.primitives import BuildPaths
from codeintel.core.schemas.contracts import (
    table_schema_from_json_obj,
    table_schema_to_json_obj,
)
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.views.inventory import discover_derived_docs_views, view_builder_modules
from codeintel.storage.views.schema_inference import derive_view_schemas
from tests._helpers.scip_proto import ensure_proto_module, write_scip_index
from tests._helpers.tool_payloads import pytest_report_payload

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType


@dataclass(frozen=True)
class HarnessArtifacts:
    """Write realistic artifacts to paths used by native targets."""

    repo_root: Path
    paths: BuildPaths

    def write_pytest_report(
        self,
        *,
        tests: Iterable[Mapping[str, Any]] = (),
        summary: Mapping[str, int] | None = None,
        prefer: str = "build_paths",
    ) -> Path:
        """Write a minimal pytest-json-report payload.

        Parameters
        ----------
        tests
            Test entry payloads to include.
        summary
            Summary payload for the report.
        prefer
            Controls which candidate location is used.

        Returns
        -------
        Path
            Path to the generated report file.
        """
        payload = pytest_report_payload(
            tests=tests,
            summary=summary or {"passed": 0, "failed": 0, "skipped": 0},
        )
        payload["root"] = str(self.repo_root)
        payload["environment"] = {}

        if prefer == "repo_root_flat":
            out = self.repo_root / "pytest-report.json"
        elif prefer == "repo_root_test_results":
            out = self.repo_root / "test-results" / "pytest-report.json"
        else:
            out = self.paths.pytest_report

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def write_schema_manifest(
        self,
        *,
        tables: Iterable[Mapping[str, Any]] = (),
        views: Iterable[Mapping[str, Any]] = (),
        view_modules: tuple[ModuleType, ...] | None = None,
        path: Path | None = None,
        derive_views: bool = True,
    ) -> Path:
        """Write a minimal schema manifest for serving/export tests.

        Parameters
        ----------
        tables
            Table entries to include in the manifest.
        views
            Optional view entries to include in the manifest.
        view_modules
            Optional view builder modules to use for derivation.
        path
            Optional output path override.
        derive_views
            When True, derive docs view schemas from view builders and tables.

        Returns
        -------
        Path
            Path to the generated schema manifest file.
        """
        table_entries = list(tables)
        view_entries = list(views)
        if derive_views:
            table_schemas = {}
            for table in table_entries:
                if not isinstance(table, Mapping):
                    continue
                try:
                    schema = table_schema_from_json_obj(table)
                except TypeError:
                    continue
                table_schemas[schema.table_key] = schema
            if table_schemas:
                provider = MappingSchemaProvider(table_schemas)
                modules = view_modules if view_modules is not None else view_builder_modules()
                derived = derive_view_schemas(
                    provider=provider,
                    view_keys=discover_derived_docs_views(),
                    modules=modules,
                )
                existing_keys = {
                    _manifest_entry_key(entry) for entry in view_entries if isinstance(entry, Mapping)
                }
                for table_key, schema in derived.items():
                    if table_key in existing_keys:
                        continue
                    entry = table_schema_to_json_obj(schema)
                    entry["table_key"] = table_key
                    entry["schema_hash"] = schema_hash(schema)
                    view_entries.append(entry)
        payload = {
            "version": "v2",
            "tables": table_entries,
            "views": view_entries,
            "artifacts": [],
        }
        out = path or (self.paths.build_dir / "serving" / "artifacts" / "schema_manifest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return out

    def write_buildspec(
        self,
        *,
        datasets: Iterable[Mapping[str, Any]] = (),
        path: Path | None = None,
    ) -> Path:
        """Write a minimal buildspec artifact.

        Parameters
        ----------
        datasets
            Dataset entries to include in the buildspec.
        path
            Optional output path override.

        Returns
        -------
        Path
            Path to the generated buildspec file.
        """
        payload = {
            "spec_version": 1,
            "targets": [],
            "datasets": list(datasets),
        }
        out = path or (self.paths.build_dir / "serving" / "artifacts" / "buildspec.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return out

    def write_semantic_registry(
        self,
        *,
        views: Iterable[Mapping[str, Any]] = (),
        path: Path | None = None,
    ) -> Path:
        """Write a minimal semantic registry artifact.

        Parameters
        ----------
        views
            View entries to include in the registry.
        path
            Optional output path override.

        Returns
        -------
        Path
            Path to the generated semantic registry file.
        """
        payload = {
            "version": "v1",
            "views": list(views),
        }
        out = path or (self.paths.build_dir / "serving" / "artifacts" / "semantic_registry.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return out

    def write_dummy_scip_artifacts(
        self,
        *,
        documents: list[dict[str, Any]] | None = None,
        scip_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        """Write minimal SCIP artifacts for tests.

        Parameters
        ----------
        documents
            Optional SCIP document payloads.
        scip_dir
            Optional override for the SCIP artifact directory.

        Returns
        -------
        tuple[Path, Path]
            Paths to index.scip and scip_pb2.py.
        """
        out_dir = scip_dir or self.paths.scip_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        index_scip = out_dir / "index.scip"
        proto_dir = out_dir / "proto"
        proto_dir.mkdir(parents=True, exist_ok=True)
        proto_source = ensure_proto_module()
        proto_dest = proto_dir / "scip_pb2.py"
        proto_dest.write_text(proto_source.read_text(encoding="utf-8"), encoding="utf-8")

        docs = documents or [
            {
                "relative_path": "pkg/mod_a.py",
                "symbols": [{"symbol": "scip-python python pkg/mod_a foo()."}],
                "occurrences": [
                    {
                        "symbol": "scip-python python pkg/mod_a foo().",
                        "range": [1, 0, 1, 1],
                        "symbol_roles": 1,
                    }
                ],
            }
        ]
        write_scip_index(index_scip, proto_module_path=proto_dest, documents=docs)
        return index_scip, proto_dest

    def touch_coverage_file(self, *, prefer: str = "repo_root") -> Path:
        """Create a coverage artifact where ingestion searches.

        Parameters
        ----------
        prefer
            Location preference ("repo_root" or "build_dir").

        Returns
        -------
        Path
            Path to the created coverage artifact.
        """
        if prefer == "build_dir":
            out = self.paths.build_dir / "coverage.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}", encoding="utf-8")
            return out

        out = self.repo_root / ".coverage"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out


def _manifest_entry_key(entry: Mapping[str, object]) -> str:
    table_key = entry.get("table_key")
    if isinstance(table_key, str) and table_key.strip():
        return table_key
    schema = entry.get("schema")
    name = entry.get("name")
    if isinstance(schema, str) and schema.strip() and isinstance(name, str) and name.strip():
        return f"{schema}.{name}"
    return ""


__all__ = ["HarnessArtifacts"]
