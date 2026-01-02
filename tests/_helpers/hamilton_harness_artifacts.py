"""Artifact helpers for Hamilton build tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.config.primitives import BuildPaths
from tests._helpers.scip_proto import ensure_proto_module, write_scip_index
from tests._helpers.tool_payloads import pytest_report_payload

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


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
        path: Path | None = None,
    ) -> Path:
        """Write a minimal schema manifest for serving/export tests.

        Parameters
        ----------
        tables
            Table entries to include in the manifest.
        path
            Optional output path override.

        Returns
        -------
        Path
            Path to the generated schema manifest file.
        """
        payload = {
            "version": "v2",
            "tables": list(tables),
            "views": [],
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


__all__ = ["HarnessArtifacts"]
