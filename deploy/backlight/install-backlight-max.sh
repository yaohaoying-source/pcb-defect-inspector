#!/usr/bin/env bash
set -eu
if [[ "$(id -u)" != 0 ]]; then
    echo "Run with sudo after checking /sys/class/backlight on this device." >&2
    exit 1
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install -m 0755 "$here/set-backlight-max.sh" /usr/local/sbin/set-backlight-max.sh
install -m 0644 "$here/backlight-max.service" /etc/systemd/system/backlight-max.service
systemctl daemon-reload
systemctl enable --now backlight-max.service
systemctl --no-pager status backlight-max.service
