#!/usr/bin/env bash
set -e
APP_DIR="/home/elf/face_detect_app_v2/new/face_detect_app_v2"
LOG_DIR="$APP_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$APP_DIR"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
exec /usr/bin/python3 main.py >> "$LOG_DIR/desktop_launcher.log" 2>&1
