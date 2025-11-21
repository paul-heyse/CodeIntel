#!/usr/bin/env bash
# scripts/bootstrap.sh
# Purpose: Provision a **deterministic, agent-grade** Python dev environment using uv.
# Works on Linux and macOS (bash 3.2+). Safe for local machines and remote containers.
#
# What it does:
#   1) Installs uv if missing (user-local)
#   2) Ensures Python 3.13.9 is installed via uv and pinned for this project
#   3) Creates/uses a project-local .venv (no system Python)
#   4) Installs deps via `uv sync` (prefers --locked when uv.lock present)
#   5) Optionally generates PATH_MAP for editor deep links in remote containers
#   6) Prints a ready-to-run command summary
#
# Exit on first error, unset var, or failed pipe; propagate failures out of subshells.
set -Eeuo pipefail

# ------------- Config (change defaults here if needed) -------------
PY_VER_DEFAULT="${PY_VER_DEFAULT:-3.13.9}"
PIN_PYTHON="${PIN_PYTHON:-1}"             # 1=uv python pin <ver>
GENERATE_PATH_MAP="${GENERATE_PATH_MAP:-1}" # 1=create docs/_build/path_map.txt if in container
USE_LOCK="${USE_LOCK:-no}"                 # yes|no|auto -> --locked only when explicitly requested
EDITOR_URI_TEMPLATE_DEFAULT='vscode-remote://dev-container+{container_id}{path}:{line}'
UV_MIN_VERSION="${UV_MIN_VERSION:-0.9.6}"  # Require uv version >= 0.9.6
EXCLUDE_EXTRAS="${EXCLUDE_EXTRAS:-gpu}"   # comma/space-separated extras to skip during uv sync

# ------------- CLI flags -------------
# Support a few handy flags so CI or developers can tailor behavior.
#   --no-pin-python      : do not uv python pin
#   --no-path-map        : don't generate docs/_build/path_map.txt
#   --use-lock[=yes|no|auto] : optionally pass --locked to uv sync
#   --py 3.13.9          : override Python version
usage() {
  cat <<'USAGE'
Usage: scripts/bootstrap.sh [options]

Options:
  --py <x.y.z>         Pin/install this Python version via uv (default: 3.13.9)
  --no-pin-python      Skip `uv python pin`
  --no-path-map        Do not generate docs/_build/path_map.txt
  --use-lock=<auto|yes|no>
  -h, --help           Show this help
USAGE
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0;;
    --no-pin-python) PIN_PYTHON=0;;
    --no-path-map) GENERATE_PATH_MAP=0;;
    --use-lock=*) USE_LOCK="${arg#*=}";;
    --py) shift; PY_VER_DEFAULT="${1:?--py requires a version like 3.13.9}";;
    --py=*) PY_VER_DEFAULT="${arg#*=}";;
  esac
done

# ------------- Logging helpers -------------
# Basic color output (graceful fallback if tput missing/unsupported).
if command -v tput >/dev/null 2>&1; then
  if [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
    BOLD="$(tput bold || true)"; DIM="$(tput dim || true)"; RESET="$(tput sgr0 || true)"
    GREEN="$(tput setaf 2 || true)"; YELLOW="$(tput setaf 3 || true)"; RED="$(tput setaf 1 || true)"
  fi
fi
BOLD="${BOLD:-}"; DIM="${DIM:-}"; RESET="${RESET:-}"
GREEN="${GREEN:-}"; YELLOW="${YELLOW:-}"; RED="${RED:-}"

info() { echo "${DIM}>>${RESET} $*"; }
ok()   { echo "${GREEN}✔${RESET} $*"; }
warn() { echo "${YELLOW}⚠${RESET} $*" 1>&2; }
err()  { echo "${RED}✘${RESET} $*" 1>&2; }

have() { command -v "$1" >/dev/null 2>&1; }

version_gt() {
  local lhs="${1:-}"
  local rhs="${2:-}"
  if [ -z "${lhs}" ] || [ -z "${rhs}" ]; then
    return 1
  fi
  if [ "${lhs}" = "${rhs}" ]; then
    return 1
  fi

  local IFS='.'
  local -a lhs_parts=() rhs_parts=()
  read -r -a lhs_parts <<<"${lhs}"
  read -r -a rhs_parts <<<"${rhs}"

  local max_len="${#lhs_parts[@]}"
  if [ "${#rhs_parts[@]}" -gt "${max_len}" ]; then
    max_len="${#rhs_parts[@]}"
  fi

  local i=0
  while [ "${i}" -lt "${max_len}" ]; do
    local l_part="${lhs_parts[i]:-0}"
    local r_part="${rhs_parts[i]:-0}"
    if ((10#${l_part} > 10#${r_part})); then
      return 0
    fi
    if ((10#${l_part} < 10#${r_part})); then
      return 1
    fi
    i=$((i + 1))
  done
  return 1
}

version_ge() {
  local lhs="${1:-}"
  local rhs="${2:-}"
  if [ -z "${lhs}" ] || [ -z "${rhs}" ]; then
    return 1
  fi
  if [ "${lhs}" = "${rhs}" ]; then
    return 0
  fi
  version_gt "${lhs}" "${rhs}"
}

# ------------- Sanity: repo root & OS -------------
if [ ! -f "pyproject.toml" ]; then
  err "Run this script from the repository root (pyproject.toml not found)."
  exit 1
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"   # linux|darwin|...
REPO_NAME="$(basename "$(pwd)")"

# ------------- Check for git binary -------------
if ! command -v git &> /dev/null; then
  err "Error: git binary not found. Please install git."
  exit 1
fi

# ------------- Ensure uv is installed -------------
ensure_uv() {
  local current_version=""
  if have uv; then
    current_version="$(uv --version 2>/dev/null | head -n1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)"
    if version_ge "${current_version}" "${UV_MIN_VERSION}"; then
      ok "uv present: $(uv --version | head -n1)"
      return 0
    fi
    warn "uv version ${current_version:-unknown} detected; upgrading to latest (need >= ${UV_MIN_VERSION})."
  else
    warn "uv not found; installing user-local uv (no sudo)."
  fi
  # Official installer from Astral: https://astral.sh/uv/docs/install
  # -sSf: silent + show errors + fail on errors
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Ensure uv is on PATH for this shell (installer prints its target dir).
  if ! have uv; then
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$HOME/.local/share/uv/bin:$PATH"
  fi
  have uv || { err "uv still not on PATH after install. Add it to PATH, then re-run."; exit 1; }
  current_version="$(uv --version 2>/dev/null | head -n1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)"
  if ! version_ge "${current_version}" "${UV_MIN_VERSION}"; then
    err "Installed uv version ${current_version:-unknown} does not satisfy requirement (>= ${UV_MIN_VERSION})."
    exit 1
  fi
  ok "Installed uv: $(uv --version | head -n1)"
}

# ------------- Ensure Python toolchain via uv -------------
ensure_python() {
  local ver="${1:?python version required}"
  if ! uv python list | grep -q "${ver}"; then
    info "Installing Python ${ver} via uv…"
    uv python install "${ver}"
  fi
  if [ "${PIN_PYTHON}" = "1" ]; then
    uv python pin "${ver}"
    ok "Pinned Python ${ver} for this project"
  else
    warn "Skipping uv python pin (requested)"
  fi
}

# ------------- Sync dependencies -------------
sync_env() {
  # Prefer locked resolution when uv.lock exists (or forced by flag).
  local lock_flag=""
  case "${USE_LOCK}" in
    auto) [ -f uv.lock ] && lock_flag="--locked" ;;
    yes)  lock_flag="--locked" ;;
    no)   lock_flag="" ;;
    *)    warn "--use-lock must be auto|yes|no (got: ${USE_LOCK}); defaulting to auto"; [ -f uv.lock ] && lock_flag="--locked" ;;
  esac

  local -a sync_cmd=("uv" "sync" "--frozen" "--no-default-groups")
  if [ -n "${lock_flag}" ]; then
    sync_cmd+=("${lock_flag}")
  fi

  if [ -n "${EXCLUDE_EXTRAS:-}" ]; then
    local normalized="${EXCLUDE_EXTRAS//,/ }"
    local extra=""
    for extra in ${normalized}; do
      extra="${extra// /}"
      if [ -n "${extra}" ]; then
        sync_cmd+=("--no-extra" "${extra}")
      fi
    done
  fi

  info "Syncing dependencies (${sync_cmd[*]})…"
  "${sync_cmd[@]}"
  ok "Environment synced"
}

# ------------- Generate PATH_MAP for remote editors (optional) -------------
generate_path_map() {
  if [ "${GENERATE_PATH_MAP}" != "1" ]; then
    return 0
  fi
  mkdir -p docs/_build
  local pm="docs/_build/path_map.txt"
  if [ -f "${pm}" ]; then
    ok "PATH_MAP already exists at ${pm}"
    return 0
  fi

  # Heuristic: Dev Containers often mount the repo at /workspaces/<name>;
  # Codespaces/VS Code Remote use similar layouts. Provide sensible defaults.
  local container_prefix=""
  local editor_prefix=""
  if [ -d "/workspace" ]; then
    container_prefix="/workspace"
    editor_prefix="/workspaces/${REPO_NAME}"
  elif [ -d "/workspaces/${REPO_NAME}" ]; then
    container_prefix="/workspaces/${REPO_NAME}"
    editor_prefix="/workspaces/${REPO_NAME}"
  else
    # Fallback: map repo root to itself (useful for local shells)
    container_prefix="$(pwd)"
    editor_prefix="$(pwd)"
  fi

  {
    echo "${container_prefix} => ${editor_prefix}"
  } > "${pm}"

  ok "Generated ${pm} mapping: ${container_prefix} => ${editor_prefix}"
  if [ -z "${EDITOR_URI_TEMPLATE:-}" ]; then
    warn "EDITOR_URI_TEMPLATE not set; using default when generating links."
    export EDITOR_URI_TEMPLATE="${EDITOR_URI_TEMPLATE_DEFAULT}"
  fi
}

# ------------- Ensure the project venv is active in this shell -------------
activate_project_env() {
  # uv manages a project-local .venv/ directory by default when sync_env runs.
  # Mirror the important pieces of "source .venv/bin/activate" without relying
  # on shell-specific activate scripts so the bootstrap process leaves the
  # current shell primed to use the freshly synced environment.
  local project_root="$(pwd)"
  local venv_path="${project_root}/.venv"

  if [ ! -d "${venv_path}" ]; then
    warn "Expected uv-managed environment at ${venv_path} not found; skipping activation."
    return 0
  fi

  # Export VIRTUAL_ENV so downstream tooling (pre-commit, editors, etc.)
  # recognizes that the project environment is active.
  export VIRTUAL_ENV="${venv_path}"

  # Prepend the venv bin dir to PATH when it's not already present so `python`
  # and friends resolve to the project environment by default.
  case ":${PATH}:" in
    *":${venv_path}/bin:"*) ;;
    *) export PATH="${venv_path}/bin:${PATH}" ;;
  esac

  # Ensure the current shell sees the updated PATH before we call python below.
  hash -r 2>/dev/null || true

  ok "Activated project environment (${venv_path})"
}

ensure_src_pth() {
  local site_packages
  site_packages="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
  local pth_file="${site_packages}/kgfoundry_src.pth"
  if [ ! -f "${pth_file}" ] || ! grep -qx "$(pwd)/src" "${pth_file}"; then
    printf "%s\n" "$(pwd)/src" > "${pth_file}"
    ok "Ensured ${pth_file} points to src"
  else
    ok "${pth_file} already points to src"
  fi
}

# ------------- Diagnostics -------------
print_summary() {
  cat <<EOF

${BOLD}Environment ready.${RESET}
  Python     : $(python -V 2>&1 || true)
  uv         : $(uv --version 2>/dev/null | head -n1 || echo "n/a")
  Location   : $(pwd)
  OS         : ${OS}
  Venv       : .venv (created by uv)

Next steps:
  - Lint/format : uv run ruff format && uv run ruff check --fix
  - Type-check  : uv run pyright --warnings --pythonversion=3.13 && uv run pyrefly check
  - Tests       : SKIP_GPU_WARMUP=1 uv run pytest -q    # drop env var on GPU-capable hosts
EOF
}

# ------------- Main flow -------------
ensure_uv
ensure_python "${PY_VER_DEFAULT}"
sync_env
activate_project_env
if [ -d .wheelhouse ]; then
  ok ".wheelhouse directory already present"
else
  # uv pip install honors [tool.uv.pip]::find-links=.wheelhouse; ensure it exists so
  # the editable install doesn't fail on fresh clones where the directory is absent.
  mkdir -p .wheelhouse
  ok "Created .wheelhouse directory for uv find-links"
fi
info "Installing project package in editable mode…"
uv pip install -e .
ok "Editable install complete"
ensure_src_pth
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
ok "PYTHONPATH includes src (current: ${PYTHONPATH})"
generate_path_map
print_summary
