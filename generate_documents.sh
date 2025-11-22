#!/usr/bin/env bash
#
# Trigger the export_docs Prefect deployment to generate Document Output artifacts.

set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'USAGE'
Usage: generate_documents.sh <repo_root> <repo_slug> <commit_sha> [db_path] [build_dir] [skip_scip]

Examples:
  ./generate_documents.sh /home/paul/CodeIntel myorg/repo deadbeef
  ./generate_documents.sh /repo org/repo abc123 /repo/build/db/codeintel.duckdb /repo/build true

Arguments:
  repo_root  : Path to the repository root (must exist).
  repo_slug  : Repository slug (e.g., org/repo).
  commit_sha : Commit SHA for this run.
  db_path    : Optional path to the DuckDB file (default: <repo_root>/build/db/codeintel.duckdb).
  build_dir  : Optional build directory (default: <repo_root>/build).
  skip_scip  : Optional boolean to bypass SCIP ingestion (default: false).
USAGE
  exit 1
fi

repo_root=$(realpath "$1")
repo_slug=$2
commit_sha=$3
db_path=${4:-"$repo_root/build/db/codeintel.duckdb"}
build_dir=${5:-"$repo_root/build"}
skip_scip=${6:-false}

if [[ ! -d "$repo_root" ]]; then
  echo "repo_root does not exist: $repo_root" >&2
  exit 1
fi

db_path=$(realpath -m "$db_path")
build_dir=$(realpath -m "$build_dir")

mkdir -p "$build_dir" "$(dirname "$db_path")"

params=$(cat <<EOF
{
  "repo_root": "$repo_root",
  "repo": "$repo_slug",
  "commit": "$commit_sha",
  "db_path": "$db_path",
  "build_dir": "$build_dir",
  "skip_scip": $skip_scip
}
EOF
)

echo "Registering deployment..."
uv run prefect deploy prefect_deployments/export_docs.yaml

echo "Triggering export_docs deployment..."
uv run prefect deployment run export_docs_flow/export_docs --params "$params"

echo "Done. Check ${repo_root}/Document Output for generated artifacts."
