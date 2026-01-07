# Arrow Join Policy

## Purpose
This document defines the canonical Arrow join workflow for build-time tabular pipelines.
It standardizes join configuration, normalization, and residual filter usage to keep join
behavior predictable and performant.

## Canonical APIs
- Use `codeintel.build.tabular.arrow_ops.arrow_join_tables` for Arrow joins.
- Use `codeintel.build.tabular.arrow_ops.ArrowJoinSpec` to express join keys and behavior.
- Use `codeintel.build.tabular.arrow_ops.ArrowJoinOptions` to pass join tuning.
- Use `codeintel.build.tabular.arrow_ops.build_join_options` for thread tuning defaults.
- Use `codeintel.build.tabular.arrow_ops.join_filter_expr` and
  `codeintel.build.tabular.arrow_ops.combine_join_filters` to safely compose residual filters.

## Join Configuration
- Always provide join keys via `ArrowJoinSpec` (`on`, `left_on`, `right_on`).
- Prefer `how="left"` for enrichment joins, then filter as needed.
- Avoid ad-hoc `Table.join` calls outside `arrow_ops` to keep suffix and normalization
  behavior consistent.

## Residual Join Filters
Arrow supports residual filters during joins via `filter_expression`.
These filters are evaluated on matching row pairs (not post-join rows).

Guidelines:
- Use `join_filter_expr` to build filters only when the column exists in the join output.
- Use `resolve_join_filter_field` if you need manual control of suffix behavior.
- Only apply residual filters when they are equivalent to post-join filtering semantics.

Example:
```python
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    JoinFilterClause,
    build_join_options,
    join_filter_expr,
)
from codeintel.build.tabular.compute_masks import is_valid_expr

spec = ArrowJoinSpec(on=["function_goid_h128"], how="left")
filter_expr = join_filter_expr(
    left=left_table,
    right=right_table,
    spec=spec,
    clause=JoinFilterClause(
        field="cpg_node_id",
        predicate=is_valid_expr,
        side="right",
    ),
)
options = build_join_options(left_table, right_table, filter_expression=filter_expr)
joined = arrow_join_tables(left_table, right_table, spec=spec, options=options)
```

## Threading Heuristics
`build_join_options` applies a size-based heuristic to enable threading for larger joins.
This avoids thread overhead on small tables while enabling parallelism for large joins.

If you need custom behavior, pass `use_threads` explicitly in `build_join_options`.

## Normalization
- Default join behavior normalizes inputs (dictionary unification, chunk compaction)
  using `normalize_table_for_join` inside `arrow_join_tables`.
- Avoid manual normalization unless you have a known, measured reason.

## Call-Site Pattern
```python
spec = ArrowJoinSpec(on=["repo", "commit", "rel_path"], how="left")
options = build_join_options(left_table, right_table)
joined = arrow_join_tables(left_table, right_table, spec=spec, options=options)
```
