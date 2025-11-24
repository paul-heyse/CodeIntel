#!/usr/bin/env bash
# scripts/bootstrap_codex.sh
# Purpose: Make sure Codex CLI can *use* the existing uv-managed env
#          without trying to install uv or Python or write outside the repo.

set -Eeuo pipefail

# Must be run from repo root
if [ ! -f "pyproject.toml" ]; then
  echo "Run scripts/bootstrap_codex.sh from the repository root (pyproject.toml not found)." >&2
  exit 1
fi

project_root="$(pwd)"
venv_path="${project_root}/.venv"

# Optional: constrain uv to project-local storage if you ever *do* call uv here.
# This keeps all uv writes in the workspace, which is sandbox-safe.
UV_ROOT="${UV_ROOT:-"$project_root/.uv"}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-"$UV_ROOT/cache"}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-"$UV_ROOT/python"}"
export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR:-"$UV_ROOT/python/bin"}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-"$venv_path"}"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR"

# Sanity: env must have been created already by scripts/bootstrap.sh (outside Codex)
if [ ! -d "${venv_path}" ]; then
  cat >&2 <<EOF
Error: Expected project environment at ${venv_path}, but it does not exist.

For Codex CLI, do NOT try to create the env from inside the sandbox.

Instead, from a normal shell on this machine run:

  bash scripts/bootstrap.sh

That will install uv (if needed), Python ${PY_VER_DEFAULT:-3.13.9}, create .venv,
and install all dependencies. Then re-run this script from Codex.
EOF
  exit 1
fi

python_bin="${venv_path}/bin/python"

if [ ! -x "${python_bin}" ]; then
  echo "Error: ${python_bin} is missing or not executable. Re-run scripts/bootstrap.sh outside Codex." >&2
  exit 1
fi

# Optionally mimic "activation" for the *current process* (useful if this script is
# chained with other commands in a single Codex shell tool call).
export VIRTUAL_ENV="${venv_path}"
case ":${PATH}:" in
  *":${venv_path}/bin:"*) ;;
  *) export PATH="${venv_path}/bin:${PATH}" ;;
esac
hash -r 2>/dev/null || true

# Make sure src is importable in this process
export PYTHONPATH="${project_root}/src:${PYTHONPATH:-}"

db_path="${project_root}/build/db/codeintel.duckdb"
if [ -f "${db_path}" ]; then
  echo "Applying schemas to existing DB at ${db_path} ..."
  CODEINTEL_DB_PATH="${db_path}" uv run python - <<'PY'
import os
import duckdb
from codeintel.storage.schemas import apply_all_schemas

db_path = os.environ["CODEINTEL_DB_PATH"]
con = duckdb.connect(db_path)
apply_all_schemas(con)
con.close()
print(f"Schemas applied to {db_path}")
PY
else
  echo "No DB at ${db_path}; skipping schema apply."
fi

echo "Environment looks good:"
echo "  python: $(${python_bin} -V 2>&1)"
echo "  where : ${python_bin}"
