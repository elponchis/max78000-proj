#!/bin/bash
# 보드 시리얼을 N초 읽는다 — **WSL root** (usbipd attach 뒤).
#   wsl.exe -d max78000 -u root -- bash scripts/fw_serial.sh 30
# 깨진 글자(`*` 만 반복 등)가 나오면 보율이 아니라 **POR** 을 먼저 의심할 것
# (docs/board-bringup.md 2′절 — SWD 리셋은 클럭 설정을 되돌리지 않는다).
set -u
SEC=${1:-20}
DEV=${2:-/dev/ttyACM0}
[ -e "$DEV" ] || modprobe cdc_acm 2>/dev/null
[ -e "$DEV" ] || { echo "[에러] $DEV 가 없다 — usbipd attach / modprobe cdc_acm 확인"; exit 2; }
stty -F "$DEV" 115200 raw -echo -hupcl
timeout "$SEC" cat "$DEV"
echo
echo "--- (${SEC}초 읽음)"
