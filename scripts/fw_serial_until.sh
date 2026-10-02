#!/bin/bash
# 시리얼을 `=== END ===` 가 한 번 나올 때까지(최대 N초) 읽는다 — **WSL root**.
#   wsl.exe -d max78000 -u root -- bash scripts/fw_serial_until.sh 2700
# 오래 걸리는 측정(M4 대조군 1000회 등)을 기다릴 때 쓴다.
set -u
SEC=${1:-1800}
DEV=${2:-/dev/ttyACM0}
[ -e "$DEV" ] || modprobe cdc_acm 2>/dev/null
[ -e "$DEV" ] || { echo "[에러] $DEV 가 없다"; exit 2; }
stty -F "$DEV" 115200 raw -echo -hupcl
timeout "$SEC" sed '/=== END ===/q' "$DEV"
echo "--- ($(date +%H:%M:%S) 종료)"
