"""Options for building Hamilton drivers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hamilton.caching.adapter import HamiltonCacheAdapter
    from hamilton.io.materialization import ExtractorFactory, MaterializerFactory
    from hamilton.lifecycle.base import LifecycleAdapter

    from codeintel.build.hamilton.dag_catalog import DagCatalog


@dataclass(frozen=True, slots=True)
class BuildDriverOptions:
    """Optional settings for building Hamilton drivers."""

    adapters: Sequence[LifecycleAdapter] | None = None
    adapter_factory: Callable[[DagCatalog], Sequence[LifecycleAdapter]] | None = None
    materializers: Sequence[ExtractorFactory | MaterializerFactory] | None = None
    enable_cache: bool = False
    cache_dir: str | Path | None = None
    cache_adapter: HamiltonCacheAdapter | None = None


__all__ = ["BuildDriverOptions"]
