"""Configuration dataclasses for pipeline test environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codeintel.config import BuildPaths
from codeintel.pipeline.execution.context import PipelineContext
from codeintel.storage.gateway import StorageGateway

# Default constants for pipeline tests
REPO = "demo/repo"
COMMIT = "deadbeef"


@dataclass
class PipelineEnv:
    """Reusable environment for pipeline graph/coverage assertions."""

    repo_root: Path
    build_paths: BuildPaths
    gateway: StorageGateway
    ctx: PipelineContext
    caller_lines: tuple[int, int]


__all__ = [
    "COMMIT",
    "REPO",
    "PipelineEnv",
]
