"""Telemetry primitives for SCIP indexing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ScipRunIdentity:
    """Identity fields for a SCIP run."""

    repo: str
    commit: str
    run_id: str
    options_hash: str | None
    project_version: str | None
    project_namespace: str | None
    environment_source: str | None


@dataclass
class ScipRunTelemetry:
    """Telemetry payload for a SCIP indexing run.

    Attributes
    ----------
    repo
        Repository slug.
    commit
        Commit SHA.
    run_id
        Run identifier for correlation.
    mode
        Run mode: "full" or "incremental".
    options_hash
        Options hash used for the run.
    project_version
        Project version passed to scip-python.
    project_namespace
        Project namespace passed to scip-python.
    environment_source
        Environment discovery source (pip or json).
    tool_log_dir
        Directory holding scip-python stdout/stderr logs for this run.
    tool_version
        Resolved scip-python version string.
    total_modules
        Total modules considered for the run.
    changed_modules
        Count of changed modules.
    deleted_modules
        Count of deleted modules.
    changed_ratio
        Ratio of changed + deleted modules to total.
    batch_size
        Target batch size for shard indexing.
    batch_count
        Number of scip-python runs performed for shards.
    decision
        Decision reason (options mismatch, threshold, force, etc.).
    ratio_gate_applied
        Whether ratio gating was applied to rebuild decisions.
    ratio_gate_min_modules
        Minimum module count required for ratio-based rebuilds.
    ratio_gate_min_changed
        Minimum changed modules required for ratio-based rebuilds.
    hash_source
        Hash source used (file_state, computed, mixed).
    hash_source_breakdown
        Breakdown of hash sources used.
    hash_reused
        Count of hashes reused from file state.
    hash_computed
        Count of hashes computed from disk.
    plan_ms
        Duration of planning phase in milliseconds.
    hash_ms
        Duration of hashing work in milliseconds.
    tool_ms
        Duration of scip-python execution in milliseconds.
    parse_ms
        Duration of protobuf parsing in milliseconds.
    merge_ms
        Duration of merge phase in milliseconds.
    write_ms
        Duration of manifest/write phase in milliseconds.
    total_ms
        Total duration in milliseconds.
    status
        Result status: succeeded, failed, or skipped.
    error_summary
        Error summary if failed.
    output_scip
        Output path for the index.scip.
    recorded_at
        Timestamp when telemetry was captured.
    """

    repo: str
    commit: str
    run_id: str
    mode: str
    options_hash: str | None
    project_version: str | None
    project_namespace: str | None
    environment_source: str | None
    tool_log_dir: str | None
    tool_version: str | None
    total_modules: int
    changed_modules: int
    deleted_modules: int
    changed_ratio: float | None
    batch_size: int | None
    batch_count: int
    decision: str | None
    ratio_gate_applied: bool | None
    ratio_gate_min_modules: int | None
    ratio_gate_min_changed: int | None
    hash_source: str | None
    hash_source_breakdown: str | None
    hash_reused: int
    hash_computed: int
    plan_ms: float | None
    hash_ms: float | None
    tool_ms: float | None
    parse_ms: float | None
    merge_ms: float | None
    write_ms: float | None
    total_ms: float | None
    status: str
    error_summary: str | None
    output_scip: str | None
    recorded_at: datetime

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable payload.

        Returns
        -------
        dict[str, object]
            JSON-ready telemetry payload.
        """
        return {
            "repo": self.repo,
            "commit": self.commit,
            "run_id": self.run_id,
            "mode": self.mode,
            "options_hash": self.options_hash,
            "project_version": self.project_version,
            "project_namespace": self.project_namespace,
            "environment_source": self.environment_source,
            "tool_log_dir": self.tool_log_dir,
            "tool_version": self.tool_version,
            "total_modules": self.total_modules,
            "changed_modules": self.changed_modules,
            "deleted_modules": self.deleted_modules,
            "changed_ratio": self.changed_ratio,
            "batch_size": self.batch_size,
            "batch_count": self.batch_count,
            "decision": self.decision,
            "ratio_gate_applied": self.ratio_gate_applied,
            "ratio_gate_min_modules": self.ratio_gate_min_modules,
            "ratio_gate_min_changed": self.ratio_gate_min_changed,
            "hash_source": self.hash_source,
            "hash_source_breakdown": self.hash_source_breakdown,
            "hash_reused": self.hash_reused,
            "hash_computed": self.hash_computed,
            "plan_ms": self.plan_ms,
            "hash_ms": self.hash_ms,
            "tool_ms": self.tool_ms,
            "parse_ms": self.parse_ms,
            "merge_ms": self.merge_ms,
            "write_ms": self.write_ms,
            "total_ms": self.total_ms,
            "status": self.status,
            "error_summary": self.error_summary,
            "output_scip": self.output_scip,
            "recorded_at": self.recorded_at.isoformat(),
        }

    @classmethod
    def create(
        cls,
        *,
        identity: ScipRunIdentity,
    ) -> ScipRunTelemetry:
        """Create a telemetry object with default values.

        Returns
        -------
        ScipRunTelemetry
            Initialized telemetry payload.
        """
        return cls(
            repo=identity.repo,
            commit=identity.commit,
            run_id=identity.run_id,
            mode="incremental",
            options_hash=identity.options_hash,
            project_version=identity.project_version,
            project_namespace=identity.project_namespace,
            environment_source=identity.environment_source,
            tool_log_dir=None,
            tool_version=None,
            total_modules=0,
            changed_modules=0,
            deleted_modules=0,
            changed_ratio=None,
            batch_size=None,
            batch_count=0,
            decision=None,
            ratio_gate_applied=None,
            ratio_gate_min_modules=None,
            ratio_gate_min_changed=None,
            hash_source=None,
            hash_source_breakdown=None,
            hash_reused=0,
            hash_computed=0,
            plan_ms=None,
            hash_ms=None,
            tool_ms=None,
            parse_ms=None,
            merge_ms=None,
            write_ms=None,
            total_ms=None,
            status="unknown",
            error_summary=None,
            output_scip=None,
            recorded_at=datetime.now(tz=UTC),
        )


def _sanitize_label(label: str) -> str:
    normalized = label.strip().lower()
    if not normalized:
        return "scip"
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in normalized)


def write_tool_logs(
    *,
    scip_dir: Path,
    run_id: str,
    label: str,
    stdout: str,
    stderr: str,
) -> Path:
    """Persist scip-python stdout/stderr logs and return the run directory.

    Returns
    -------
    Path
        Directory containing the persisted log files.
    """
    run_dir = scip_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _sanitize_label(label)
    (run_dir / f"{safe_label}.stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / f"{safe_label}.stderr.log").write_text(stderr, encoding="utf-8")
    return run_dir


__all__ = ["ScipRunIdentity", "ScipRunTelemetry", "write_tool_logs"]
