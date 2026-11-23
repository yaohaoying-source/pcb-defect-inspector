#!/usr/bin/env bash
set -eu
backlight=/sys/class/backlight/backlight-dsi0
if [[ ! -r "$backlight/max_brightness" || ! -w "$backlight/brightness" ]]; then
    echo "Backlight path is unavailable or not writable: $backlight" >&2
    exit 1
fi
cat "$backlight/max_brightness" > "$backlight/brightness"
