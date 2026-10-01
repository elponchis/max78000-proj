#!/bin/bash
# MAX78000FTHR 펌웨어 빌드 — **WSL2 에서 돌린다.**
#
# 왜 WSL 인가: MSDK·arm-none-eabi·openocd 가 이미 WSL 에 있다
# (`~/msdk`, `/usr/bin/arm-none-eabi-gcc`, `/usr/local/bin/openocd`).
# Windows 에 MSDK 를 또 깔 이유가 없다. **플래싱만 Windows 쪽에서**
# DAPLink 드라이브에 `.bin` 을 끌어다 놓는다 (usbipd 불필요).
#
# ⚠️ CLAUDE.md 6장: 펌웨어 명령을 WSL 기준으로 안내하지 말 것 —
#    **빌드는 예외**다. 빌드 산출물만 Windows 로 넘긴다. 플래싱·시리얼·
#    디버깅은 Windows 쪽 절차이고 이 스크립트가 다루지 않는다.
#
# 사용 (WSL2):
#   bash scripts/fw_build.sh Hello_World
#   bash scripts/fw_build.sh CNN/kws20_demo
#   bash scripts/fw_build.sh Hello_World clean
#
# 산출물은 **Windows 에서 보이는 폴더**로 복사한다:
#   C:\Users\1124j\Downloads\max78000-fw\<이름>.bin
set -u

MSDK=${MSDK:-$HOME/msdk}
BOARD=${BOARD:-FTHR_RevA}
TARGET=${TARGET:-MAX78000}
DROP=${DROP:-/mnt/c/Users/1124j/Downloads/max78000-fw}

EX=${1:-}
ACT=${2:-build}
if [ -z "$EX" ]; then
  echo "사용: bash scripts/fw_build.sh <예제경로> [clean]"
  echo "  예: Hello_World / CNN/kws20_demo"
  exit 2
fi

DIR="$MSDK/Examples/$TARGET/$EX"
if [ ! -d "$DIR" ]; then
  echo "[에러] 예제가 없다: $DIR"
  echo "       있는 것:"
  ls "$MSDK/Examples/$TARGET" | sed 's/^/         /' | head -30
  exit 1
fi

export MAXIM_PATH="$MSDK"
NAME=$(basename "$EX")

echo "=== 환경"
echo "  MAXIM_PATH = $MAXIM_PATH"
echo "  BOARD      = $BOARD"
echo "  예제       = $DIR"
echo "  gcc        = $(arm-none-eabi-gcc -dumpversion 2>/dev/null || echo 없음)"
echo

cd "$DIR" || exit 1

if [ "$ACT" = "clean" ]; then
  echo "=== clean"
  make clean BOARD="$BOARD" MAXIM_PATH="$MAXIM_PATH" > /dev/null 2>&1
  echo "  완료"
  exit 0
fi

echo "=== 빌드 (로그: /tmp/fw_build_$NAME.log)"
make -j"$(nproc)" BOARD="$BOARD" MAXIM_PATH="$MAXIM_PATH" \
  > "/tmp/fw_build_$NAME.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "  ✘ 실패 (exit $rc) — 마지막 30줄:"
  tail -30 "/tmp/fw_build_$NAME.log" | sed 's/^/     /'
  exit 1
fi
echo "  ✔ 성공"
echo

ELF="$DIR/build/$TARGET.elf"
BIN="$DIR/build/$TARGET.bin"
[ -f "$ELF" ] || ELF=$(find "$DIR/build" -maxdepth 1 -name '*.elf' | head -1)
[ -f "$BIN" ] || BIN=$(find "$DIR/build" -maxdepth 1 -name '*.bin' | head -1)

if [ ! -f "$BIN" ] && [ -f "$ELF" ]; then
  echo "=== .bin 이 없어 .elf 에서 만든다"
  BIN="$DIR/build/$TARGET.bin"
  arm-none-eabi-objcopy -O binary "$ELF" "$BIN" || exit 1
fi

echo "=== 크기"
if [ -f "$ELF" ]; then
  arm-none-eabi-size "$ELF" | sed 's/^/  /'
  # Flash 512KB / SRAM 128KB 대비 (CLAUDE.md 3장)
  arm-none-eabi-size "$ELF" | awk 'NR==2 {
    printf "  → Flash(text+data) %d B / 524288 (%.1f%%)\n", $1+$2, 100*($1+$2)/524288;
    printf "  → SRAM(data+bss)   %d B / 131072 (%.1f%%)\n", $2+$3, 100*($2+$3)/131072 }'
fi
echo "  bin: $(stat -c %s "$BIN" 2>/dev/null || echo ?) B"
echo

echo "=== Windows 로 복사 (드래그앤드롭용)"
mkdir -p "$DROP" 2>/dev/null
if cp "$BIN" "$DROP/$NAME.bin" 2>/dev/null; then
  echo "  ✔ $DROP/$NAME.bin"
  echo "    Windows 경로: C:\\Users\\1124j\\Downloads\\max78000-fw\\$NAME.bin"
else
  echo "  ⚠️ 복사 실패 — WSL 경로에서 직접 가져갈 것:"
  echo "    \\\\wsl.localhost\\max78000$BIN"
fi
[ -f "$ELF" ] && cp "$ELF" "$DROP/$NAME.elf" 2>/dev/null && \
  echo "  ✔ $DROP/$NAME.elf  (OpenOCD·GDB 쓸 때 필요)"
echo
echo "다음: Windows 탐색기에서 위 .bin 을 DAPLINK 드라이브에 끌어다 놓는다."
