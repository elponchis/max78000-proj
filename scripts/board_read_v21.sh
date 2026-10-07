#!/bin/bash
# POR 뒤 시리얼을 `=== END ===` 까지 읽어 로그로 남긴다 — **WSL root**.
#   bash board_read.sh <이름> [초]
set -u
NAME=${1:?이름}; SEC=${2:-600}
cd /home/max78000/max78000-proj || exit 1
modprobe usbhid 2>/dev/null; modprobe cdc_acm 2>/dev/null
for i in 1 2 3 4 5 6 7 8 9 10; do [ -e /dev/ttyACM0 ] && break; sleep 1; done
[ -e /dev/ttyACM0 ] || { echo "[에러] /dev/ttyACM0 없음 (attach 됐나?)"; lsusb | grep 0d28; exit 2; }
LOG=data/logs-local/board_v21_$NAME.log
bash scripts/fw_serial_until.sh "$SEC" | tee "$LOG"
echo "--- 저장: $LOG"
