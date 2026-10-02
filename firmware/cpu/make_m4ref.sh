#!/bin/bash
# M4 대조군(④ 가중치의 소프트웨어 추론) 측정 프로젝트 — **WSL2**.
#   ~/ai8x-training/venv/bin/python tools/export_m4_weights.py   # 가중치 헤더
#   bash firmware/cpu/make_m4ref.sh          # 단순 C 참조 구현 → firmware/cpu/m4ref
#   bash firmware/cpu/make_m4ref.sh cmsis    # CMSIS-NN 판      → firmware/cpu/m4cmsis
#   bash scripts/fw_build.sh firmware/cpu/m4ref   (또는 m4cmsis)
# 프로젝트 안의 파일은 전부 생성물이다 — 고칠 때는 원본을 고치고 다시 돌린다.
# CMSIS-NN 은 레포에 넣지 않고 ~/CMSIS-NN (또는 $CMSIS_NN) 을 참조한다.
set -eu
P=$(cd "$(dirname "$0")/../.." && pwd)
if [ "${1:-}" = "cmsis" ]; then NAME=m4cmsis; SRC=m4cmsis.c; else NAME=m4ref; SRC=m4ref.c; fi
DST=$P/firmware/cpu/$NAME
mkdir -p "$DST"
cp "$P/firmware/npu/safesound_wave/Makefile" "$DST/Makefile"     # MSDK 공용 Makefile
cp "$P/firmware/common/$SRC" "$P/firmware/common/m4ref.h" \
   "$P/firmware/common/m4_weights.h" "$DST/"
cp "$P/firmware/common/m4ref_main.c" "$DST/main.c"
{
  echo "# @generated — firmware/cpu/make_m4ref.sh"
  echo "BOARD = FTHR_RevA"
  echo "PROJ_CFLAGS += -DMEASURE_BUILD"
  if [ "$NAME" = m4cmsis ]; then
    C=${CMSIS_NN:-$HOME/CMSIS-NN}
    echo "# CMSIS-NN ($C, 커밋 $(git -C "$C" rev-parse --short HEAD))"
    echo "# ⚠️ SINGLE_ROUNDING 이 NPU 와의 비트 일치 조건이다 (m4cmsis.c 머리말)"
    echo "PROJ_CFLAGS += -DM4_CMSIS -DCMSIS_NN_USE_SINGLE_ROUNDING"
    # MSDK Makefile 은 VPATH 의 *.c 를 전부 끌어온다 — CMSIS-NN 폴더에는 M4 에서
    # 컴파일되지 않는 f16 소스가 있으므로 자동 탐색을 끄고 필요한 것만 적는다.
    echo "AUTOSEARCH = 0"
    echo "SRCS += main.c m4cmsis.c"
    echo "IPATH += $C/Include"
    echo "VPATH += $C/Source/ConvolutionFunctions"
    echo "VPATH += $C/Source/NNSupportFunctions"
    for f in $(CMSIS_NN=$C bash "$P/firmware/cpu/cmsis_nn_srcs.sh"); do
      echo "SRCS += $(basename "$f")"
    done
  fi
} > "$DST/project.mk"
echo "✔ $DST"; ls "$DST"
