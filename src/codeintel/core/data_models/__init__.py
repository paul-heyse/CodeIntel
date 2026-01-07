"""Canonical graph and core row data models.

This package provides the single source of truth for graph and core table row
types. All row types are defined as frozen dataclasses with a `to_tuple()` method
for DuckDB serialization.

Row Types
---------
CFGBlockRow
    Row data for graph.cfg_blocks table.
CFGEdgeRow
    Row data for graph.cfg_edges table.
DFGEdgeRow
    Row data for graph.dfg_edges table.
ImportModuleRow
    Row data for graph.import_modules table.
ImportEdgeRow
    Row data for graph.import_graph_edges table.
SymbolUseRow
    Row data for graph.symbol_use_edges table.
GoidRow
    Row data for core.goids table.
GoidCrosswalkRow
    Row data for core.goid_crosswalk table.

Example
-------
```python
from codeintel.core.data_models import CFGBlockRow

row = CFGBlockRow(
    repo="github.com/owner/repo",
    commit="abc123def456",
    function_goid_h128=123456789,
    block_idx=0,
    block_id="entry",
    label="entry",
    file_path="src/main.py",
    start_line=1,
    end_line=10,
    kind="entry",
    stmts_json=b"[]",
    in_degree=0,
    out_degree=1,
)


values = row.to_tuple()
```
"""

from __future__ import annotations

from codeintel.core.data_models.ids import as_int, normalize_decimal_id
from codeintel.core.data_models.rows import (
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    GoidCrosswalkRow,
    GoidRow,
    ImportEdgeRow,
    ImportModuleRow,
    SymbolUseRow,
)

__all__ = [
    "CFGBlockRow",
    "CFGEdgeRow",
    "DFGEdgeRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "ImportEdgeRow",
    "ImportModuleRow",
    "SymbolUseRow",
    "as_int",
    "normalize_decimal_id",
]
