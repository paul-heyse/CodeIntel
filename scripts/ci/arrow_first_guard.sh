#!/usr/bin/env bash
set -euo pipefail

pattern="import polars|polars\\.LazyFrame|pl\\.LazyFrame"
search_paths=(
  src/codeintel/build/graphs/engine
  src/codeintel/build/graphs/validation
  src/codeintel/build/hamilton/native/analytics
  src/codeintel/build/hamilton/native/graphs
  src/codeintel/build/hamilton/native/ingestion
)
if command -v rg >/dev/null 2>&1; then
  search_cmd=(rg -n "$pattern" "${search_paths[@]}")
else
  search_cmd=(grep -R -n -E "$pattern" "${search_paths[@]}")
fi
if "${search_cmd[@]}"; then
  echo "Arrow-first guard failed: Polars usage detected in graph compute modules." >&2
  exit 1
fi
