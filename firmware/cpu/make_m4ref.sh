#!/bin/bash
# M4 대조군(④ 가중치의 소프트웨어 추론, 단순 C 참조 구현) 측정 프로젝트 — **WSL2**.
#   ~/ai8x-training/venv/bin/python tools/export_m4_weights.py   # 가중치 헤더
#   bash firmware/cpu/make_m4ref.sh && bash scripts/fw_build.sh firmware/cpu/m4ref
# 프로젝트 안의 파일은 전부 생성물이다 — 고칠 때는 원본을 고치고 다시 돌린다.
set -eu
P=$(cd "$(dirname "$0")/../.." && pwd)
DST=$P/firmware/cpu/m4ref
mkdir -p "$DST"
cp "$P/firmware/npu/safesound_wave/Makefile" "$DST/Makefile"     # MSDK 공용 Makefile
cp "$P/firmware/common/m4ref.c" "$P/firmware/common/m4ref.h" \
   "$P/firmware/common/m4_weights.h" "$DST/"
cp "$P/firmware/common/m4ref_main.c" "$DST/main.c"
cat > "$DST/project.mk" <<'EOF'
# @generated — firmware/cpu/make_m4ref.sh
BOARD = FTHR_RevA
PROJ_CFLAGS += -DMEASURE_BUILD
EOF
echo "✔ $DST"; ls "$DST"
