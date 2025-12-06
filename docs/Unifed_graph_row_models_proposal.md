1. Unify graph row models between config and graphs
What I’m seeing

You’re defining the same graph table row shapes twice, once as TypedDicts (for dataset contracts) and once as @dataclasses (for computation):

In config (DuckDB table schemas):

config/config/datasets/rows/graph.py:

CFGBlockRow(TypedDict)

CFGEdgeRow(TypedDict)

DFGEdgeRow(TypedDict)

ImportEdgeRow(TypedDict)

ImportModuleRow(TypedDict)

SymbolUseRow(TypedDict)

config/config/datasets/rows/core.py:

GoidCrosswalkRow(TypedDict)

In graphs (graph computation):

graphs/graphs/compute/cfg.py: @dataclass(frozen=True) CFGBlockRow, @dataclass(frozen=True) CFGEdgeRow

graphs/graphs/compute/dfg.py: @dataclass(frozen=True) DFGEdgeRow

graphs/graphs/compute/imports.py: @dataclass(frozen=True) ImportModuleRow, ImportEdgeRow

graphs/graphs/compute/symbols.py: @dataclass(frozen=True) SymbolUseRow

graphs/graphs/compute/goid.py: @dataclass(frozen=True) GoidCrosswalkRow

The fields line up almost exactly (same names, same count, same semantics). The only real difference is representation (TypedDict vs dataclass) and where they live.

Refactor idea

Introduce a single canonical set of graph row models and reuse them everywhere:

Create a module like:

codeintel/graphs/data_models/rows.py


with @dataclass(frozen=True) definitions:

@dataclass(frozen=True)
class CFGBlockRow:
    function_goid_h128: int
    block_idx: int
    block_id: str
    label: str
    file_path: str
    start_line: int
    end_line: int
    kind: str
    stmts_json: str
    in_degree: int
    out_degree: int


and similarly for CFGEdgeRow, DFGEdgeRow, ImportModuleRow, ImportEdgeRow, SymbolUseRow, GoidCrosswalkRow.

In graphs:

Delete the local dataclass definitions in graphs/graphs/compute/*.py.

Import the canonical ones from codeintel.graphs.data_models.rows.

In config:

Replace the TypedDict definitions in config/config/datasets/rows/graph.py and core.py with thin wrappers or helpers that use the dataclasses, e.g.:

from codeintel.graphs.data_models.rows import CFGBlockRow as _CFGBlockRow

CFGBlockRow = TypedDict(
    "CFGBlockRow",
    {field.name: field.type for field in dataclasses.fields(_CFGBlockRow)},
)


or more simply: use the dataclasses directly for DuckDB inserts (dataclasses.asdict).

Why this helps

Removes a whole family of duplicated types across config and graphs.

Prevents subtle drift: if you ever add a field to CFG blocks, you now change it in one place.

Makes schema reasoning easier: “graph rows” live in one canonical module.

Caveats

You’ll need to touch any code that currently assumes TypedDict vs dataclass (mainly type-checking and how you build rows before writing to DuckDB).

Incremental path: start by having the dataclasses be canonical and adapt config to generate TypedDict types, then later drop the TypedDict layer if you want.