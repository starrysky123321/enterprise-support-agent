#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
dashscope_env_file="${DASHSCOPE_ENV_FILE:-${project_dir}/.env}"

if [[ -f "${dashscope_env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090 -- path may be supplied by the deployment environment.
  source "${dashscope_env_file}"
  set +a
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required. Set it in .env or the environment." >&2
  exit 1
fi

cd "${project_dir}"
exec uv run uvicorn main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
