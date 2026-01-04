# Arrow Join Policy

## Purpose
Track join-like operations in build code with their join keys and expected
cardinality. This keeps Arrow-first joins deterministic and auditable.

## Conventions
- `m:1` means many left rows can match a single right row (right is unique on keys).
- `1:1` means both sides are unique on the keys.
- If a join is optional, left rows may have zero matches.

## Graph joins (Arrow-first)
- `cpg._cfg_blocks_to_cpg`: `cfg_blocks` + `goids` on `function_goid_h128` (`m:1`).
- `call_wiring.cpg_call_targets`: `call_targets` + `cfg_blocks.entry` on
  `callee_goid_h128 -> function_goid_h128` (`m:1`), and the same for exit blocks.
- `call_graph._call_graph_node_rows`: GOID function rows + parsed function defs on
  `(rel_path, qualname)` (`m:1`, zero matches allowed).
- `goids._joined_ast_nodes`: `ast_nodes` + `modules` on `path` (`m:1`).

## Ingestion joins (build-time)
- `syntax_augment._weld_coverage_frame`: `ts_counts` + `mapped` on
  `(repo, commit, rel_path, language)` (`m:1`).
- `syntax_enrich.syntax_enrich__occurrence_resolution`: `occurrence_syntax_xref` +
  `occurrence_span_xref` on `(repo, commit, rel_path, scip_symbol, occ_start_line,
  occ_start_col, occ_end_line, occ_end_col)` (`m:1`).
- `syntax_enrich._occurrence_byte_join_spec`: facts + occurrences on
  `(repo, commit, rel_path, producer, start_byte, end_byte) ->
  (repo, commit, rel_path, producer, occ_start_byte, occ_end_byte)` (`m:1`).
- `syntax_enrich._occurrence_line_join_spec`: facts + occurrences on
  `(repo, commit, rel_path, producer, start_line, start_col, end_line, end_col) ->
  (repo, commit, rel_path, producer, occ_start_line, occ_start_col, occ_end_line,
  occ_end_col)` (`m:1`).
- `scip_resolution._symbol_goid_xref_frame`: definitions + goids on
  `(rel_path, start_line, end_line)` (`m:1`).
- `scip_resolution._occurrence_span_xref_frame`: occurrences + symbol_info on
  `(repo, commit, scip_symbol)` (`m:1`), then + goid lookup on the same keys (`m:1`).

## Analytics joins (Arrow-first)
- `subsystems.cache.build_subsystem_profile_cache_frame`: `analytics.subsystems` +
  `analytics.subsystem_graph_metrics` on `(repo, commit, subsystem_id)` (`m:1`).

## Maintenance
When introducing new joins (Arrow or Polars), add an entry here with the join
keys and expected cardinality so join validation stays consistent.
