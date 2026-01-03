"""Graph validation context for CheckProtocol-based checks.

This module provides the context object passed to all graph validation checks,
enabling them to implement the CheckProtocol from core/validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

    from codeintel.build.graphs.engine import GraphEngine
    from codeintel.build.graphs.runtime import GraphRuntime
    from codeintel.storage.catalog import FunctionCatalog

log = logging.getLogger(__name__)


@dataclass
class GraphValidationContext:
    """Context passed to all graph validation checks.

    This context provides all the data needed by validation checks,
    enabling them to implement the CheckProtocol from core/validation.
    Validation expects Parquet-backed base tables after materialization.

    Attributes
    ----------
    dataset_root_dir
        Root directory for Parquet dataset snapshots.
    repo
        Repository identifier.
    commit
        Commit identifier.
    engine
        Optional graph engine for accessing graphs.
    catalog
        Optional function catalog.
    runtime
        Optional graph runtime.
    logger
        Logger for validation output.
    call_graph
        Optional pre-loaded call graph.
    import_graph
        Optional pre-loaded import graph.
    symbol_graph
        Optional pre-loaded symbol graph.
    """

    dataset_root_dir: Path | None
    repo: str
    commit: str
    engine: GraphEngine | None = None
    catalog: FunctionCatalog | None = None
    runtime: GraphRuntime | None = None
    logger: logging.Logger = field(default_factory=lambda: log)
    call_graph: nx.DiGraph | None = None
    import_graph: nx.DiGraph | None = None
    symbol_graph: nx.Graph | None = None


__all__ = ["GraphValidationContext"]
