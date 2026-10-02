#!/bin/bash
# CPU 멜 전처리 측정 프로젝트를 만든다 — **WSL2**.
#   bash firmware/cpu/make_melfeat.sh && bash scripts/fw_build.sh firmware/cpu/melfeat
# 공용 소스(firmware/common)와 KAT 벡터(tools/kat_vectors)를 복사해 온다.
# 프로젝트 안의 파일은 전부 생성물이다 — 고칠 때는 원본을 고치고 다시 돌린다.
set -eu
P=$(cd "$(dirname "$0")/../.." && pwd)
DST=$P/firmware/cpu/melfeat
mkdir -p "$DST"
cp "$P/firmware/npu/safesound_wave/Makefile" "$DST/Makefile"     # MSDK 공용 Makefile
cp "$P/firmware/common/melfeat.c" "$P/firmware/common/melfeat.h" \
   "$P/firmware/common/mel_tables.h" "$DST/"
cp "$P/tools/kat_vectors/melkat_vectors.h" "$DST/"
cp "$P/firmware/common/melfeat_main.c" "$DST/main.c"
cat > "$DST/project.mk" <<'EOF'
# @generated — firmware/cpu/make_melfeat.sh
BOARD = FTHR_RevA
LIB_CMSIS_DSP = 1
PROJ_CFLAGS += -DMEASURE_BUILD
EOF
echo "✔ $DST"; ls "$DST"
