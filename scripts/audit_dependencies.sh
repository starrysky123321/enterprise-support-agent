#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
requirements_file="$(mktemp)"
trap 'rm -f "${requirements_file}"' EXIT

cd "${project_dir}"
uv export --quiet --frozen --no-dev --format requirements-txt --no-hashes \
  --output-file "${requirements_file}"

# Chroma has no fixed release for these server-mode advisories. The application
# uses only embedded PersistentClient; see SECURITY.md for the deployment rule.
uvx pip-audit -r "${requirements_file}" \
  --ignore-vuln PYSEC-2026-311 \
  --ignore-vuln PYSEC-2026-3813 \
  --ignore-vuln PYSEC-2026-3814 \
  --ignore-vuln PYSEC-2026-3815
