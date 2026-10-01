#!/bin/bash
# MAX78000FTHR 플래싱 — **WSL2 + OpenOCD (CMSIS-DAP).**
#
# ⚠️ **sudo 가 필요하다.** usbipd 로 넘어온 USB 장치의 `/dev/bus/usb/*` 가
#    `crw------- root root` 라 일반 사용자가 열 수 없다. OpenOCD 가
#    "unable to find a matching CMSIS-DAP device" 를 내면 대개 이 권한
#    문제다 (장치가 없는 것이 아니다 — `lsusb` 로 먼저 확인할 것).
#
# 왜 MSD 드래그가 아니라 이것인가: DAPLink MSD 플래싱이 이 보드에서
# **두 번 모두 `transfer timed out`** 으로 실패했다 (탐색기 드래그 +
# 명령줄 단일 쓰기 + WriteThrough 전부). 리마운트는 되니 보드 문제가
# 아니고 MSD 전송 경로 문제다. CMSIS-DAP(SWD)로 우회한다.
#
# 선행 조건 (Windows, 관리자 PowerShell에서 1회):
#   usbipd bind   --busid 1-3
#   usbipd attach --wsl --busid 1-3
#
# 사용 (WSL2):
#   sudo bash scripts/fw_flash.sh Hello_World
#   sudo bash scripts/fw_flash.sh kws20_demo
#   sudo bash scripts/fw_flash.sh /path/to/any.elf
set -u

SCRIPTS=${OOCD_SCRIPTS:-/usr/local/share/openocd/scripts}
DROP=${DROP:-/mnt/c/Users/1124j/Downloads/max78000-fw}

WHAT=${1:-Hello_World}
if [ -f "$WHAT" ]; then
  ELF="$WHAT"
else
  ELF="$DROP/$WHAT.elf"
fi

if [ ! -f "$ELF" ]; then
  echo "[에러] elf 가 없다: $ELF"
  echo "       먼저 빌드할 것: bash scripts/fw_build.sh $WHAT"
  exit 1
fi

echo "=== 대상"
echo "  elf     = $ELF ($(stat -c %s "$ELF") B)"
echo "  scripts = $SCRIPTS"
echo

echo "=== USB 장치 확인"
if ! lsusb 2>/dev/null | grep -qi '0d28:0204'; then
  echo "  ✘ CMSIS-DAP(0d28:0204) 가 안 보인다."
  echo "    Windows 관리자 PowerShell 에서:"
  echo "      usbipd attach --wsl --busid 1-3"
  exit 2
fi
lsusb | grep -i '0d28:0204' | sed 's/^/  /'
if [ "$(id -u)" -ne 0 ]; then
  echo
  echo "  ⚠️ root 가 아니다. /dev/bus/usb 가 root 전용이라 실패할 것이다."
  echo "     sudo bash scripts/fw_flash.sh $WHAT"
fi
echo

echo "=== 플래싱"
openocd -s "$SCRIPTS" \
  -f interface/cmsis-dap.cfg \
  -f target/max78000.cfg \
  -c "adapter speed 2000" \
  -c "program $ELF verify reset exit"
rc=$?
echo
if [ "$rc" -eq 0 ]; then
  echo "✔ 플래싱 성공 — 보드가 리셋됐다. 시리얼을 읽으면 출력이 보인다."
else
  echo "✘ 플래싱 실패 (exit $rc)"
  echo "  · 'unable to find a matching CMSIS-DAP device' → 권한(sudo) 또는 attach 확인"
  echo "  · 'Error: Could not initialize the debug port' → 보드 RST 를 누른 뒤 재시도"
fi
exit "$rc"
