#!/usr/bin/env bash
set -u

for d in /dev/video* /dev/v4l-subdev*; do
  [ -e "$d" ] || continue
  echo "===== $d ====="
  v4l2-ctl -d "$d" --all 2>&1 | sed -n '1,45p'
  echo "--- controls"
  v4l2-ctl -d "$d" --list-ctrls 2>&1 | grep -Ei 'exposure|gain|flicker|band|power|frequency|blank|frame|rate|pixel|analogue|digital' || true
done
