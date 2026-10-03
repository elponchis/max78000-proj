#!/bin/bash
# CPU 멜 전처리 측정 프로젝트를 만든다 — **WSL2**.
#   bash firmware/cpu/make_melfeat.sh               # ①  전량 재계산     → firmware/cpu/melfeat
#   bash firmware/cpu/make_melfeat.sh inc           # ①′ 증분 (기준)     → firmware/cpu/melinc
#   bash firmware/cpu/make_melfeat.sh opt           # ①′ 최적화 판 비교  → firmware/cpu/melopt
#   bash firmware/cpu/make_melfeat.sh opt h500      # 곡선 변형 (h500 / h500m32 / m32 / h400)
#                                                   #                    → firmware/cpu/melopt_h500
#   bash scripts/fw_build.sh firmware/cpu/<이름>
# 공용 소스(firmware/common)와 KAT 벡터(tools/kat_vectors)를 복사해 온다.
# 프로젝트 안의 파일은 전부 생성물이다 — 고칠 때는 원본을 고치고 다시 돌린다.
# 곡선 변형은 멜 표·KAT 기대값을 그 프레임 정의로 **프로젝트 안에 새로 만든다**.
set -eu
P=$(cd "$(dirname "$0")/../.." && pwd)
FR=${2:-inc}
case "${1:-}" in
  inc) NAME=melinc; MAIN=melinc_main.c ;;
  opt) NAME=melopt; MAIN=melopt_main.c; [ "$FR" != inc ] && NAME=melopt_$FR ;;
  *)   NAME=melfeat; MAIN=melfeat_main.c ;;
esac
DST=$P/firmware/cpu/$NAME
mkdir -p "$DST"
cp "$P/firmware/npu/safesound_wave/Makefile" "$DST/Makefile"     # MSDK 공용 Makefile
cp "$P/firmware/common/melfeat.c" "$P/firmware/common/melfeat.h" "$DST/"
cp "$P/tools/kat_vectors/melkat_vectors.h" "$DST/"
if [ "$FR" = inc ]; then
  cp "$P/firmware/common/mel_tables.h" "$DST/"
  [ "$NAME" != melfeat ] && cp "$P/firmware/common/melinc_kat.h" "$DST/"
else
  OPENBLAS_NUM_THREADS=1 python3 "$P/tools/gen_mel_tables.py" --framing "$FR" --out "$DST/mel_tables.h"
  OPENBLAS_NUM_THREADS=1 python3 "$P/tools/gen_melinc_kat.py" --scheme "log_$FR" --out "$DST/melinc_kat.h"
fi
cp "$P/firmware/common/$MAIN" "$DST/main.c"
cat > "$DST/project.mk" <<'EOF'
# @generated — firmware/cpu/make_melfeat.sh
BOARD = FTHR_RevA
LIB_CMSIS_DSP = 1
PROJ_CFLAGS += -DMEASURE_BUILD
EOF
echo "✔ $DST"; ls "$DST"
