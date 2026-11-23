#!/usr/bin/env bash
set -eu
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir/pcb_defect_inspector"
exec "${PYTHON:-python3}" main.py "$@"
