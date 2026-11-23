#!/usr/bin/env bash
set -e
set -eu
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$app_dir/../deploy/start_pcb_inspector.sh" "$@"
