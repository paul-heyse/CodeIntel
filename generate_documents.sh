#!/usr/bin/env bash
#
## Generate document_output artifacts by running the Prefect flow directly.
## Defaults assume this repository is checked out at $PWD.

set -euo pipefail

repo_root=${1:-"$(pwd)"}
repo_slug=${2:-"paul-heyse/CodeIntel"}
commit_sha=${3:-""}
db_path=${4:-"$repo_root/build/db/codeintel_prefect.duckdb"}
build_dir=${5:-"$repo_root/build"}
skip_scip=${6:-false}
document_output_dir=${7:-"$repo_root/document_output"}
serve_db_path=${8:-"$repo_root/build/db/codeintel.duckdb"}

if [[ -z "$commit_sha" ]]; then
  if git -C "$repo_root" rev-parse --verify HEAD >/dev/null 2>&1; then
    commit_sha=$(git -C "$repo_root" rev-parse HEAD)
  else
    commit_sha="unknown"
  fi
fi

db_path=$(realpath -m "$db_path")
build_dir=$(realpath -m "$build_dir")
repo_root=$(realpath -m "$repo_root")
document_output_dir=$(realpath -m "$document_output_dir")
serve_db_path=$(realpath -m "$serve_db_path")

mkdir -p "$build_dir" "$(dirname "$db_path")" "$document_output_dir"

export GEN_DOCS_REPO_ROOT="$repo_root"
export GEN_DOCS_REPO="$repo_slug"
export GEN_DOCS_COMMIT="$commit_sha"
export GEN_DOCS_DB_PATH="$db_path"
export GEN_DOCS_BUILD_DIR="$build_dir"
export GEN_DOCS_SKIP_SCIP="$skip_scip"
export CODEINTEL_OUTPUT_DIR="$document_output_dir"
export CODEINTEL_SKIP_SCIP="$skip_scip"
export GEN_DOCS_SERVE_DB="$serve_db_path"

echo "Running export_docs_flow directly (no external Prefect services needed)..."
uv run python - <<'PY'
from pathlib import Path
import os

from codeintel.orchestration.prefect_flow import ExportArgs, export_docs_flow

repo_root = Path(os.environ["GEN_DOCS_REPO_ROOT"])
repo = os.environ["GEN_DOCS_REPO"]
commit = os.environ["GEN_DOCS_COMMIT"]
db_path = Path(os.environ["GEN_DOCS_DB_PATH"])
build_dir = Path(os.environ["GEN_DOCS_BUILD_DIR"])
serve_db = Path(os.environ["GEN_DOCS_SERVE_DB"])
skip_scip = os.environ["GEN_DOCS_SKIP_SCIP"].lower() == "true"

export_docs_flow(
    args=ExportArgs(
        repo_root=repo_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        serve_db_path=serve_db,
    )
)
PY

echo "Done. Check ${repo_root}/document_output/ for generated artifacts."
