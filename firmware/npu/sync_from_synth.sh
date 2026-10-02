#!/bin/bash
# 합성 산출물 → 펌웨어 프로젝트. **WSL2 에서 돌린다.**
#
# `ai8xize.py` 가 만든 프로젝트에서 **벤더 생성 파일은 그대로** 가져오고,
# main.c 는 우리 것(`common/measure_main.c`)으로 바꾼다. 생성된 main.c 에서는
# 샘플 입력 적재(`load_input`)와 기준 출력 대조(`check_output`)만 떼어
# `kat_io.c` 로 둔다 — 손으로 옮기면 주소·길이를 틀린다.
#
# 사용:
#   bash firmware/npu/sync_from_synth.sh safesound_wave
#   bash firmware/npu/sync_from_synth.sh safesound_mel
set -eu
NAME=${1:?합성 prefix (예: safesound_wave)}
P=$(cd "$(dirname "$0")/../.." && pwd)
SRC=$P/data/synth/out/$NAME
DST=$P/firmware/npu/$NAME
[ -d "$SRC" ] || { echo "[에러] 합성 산출물이 없다: $SRC"; exit 1; }
mkdir -p "$DST"

cp "$SRC"/{cnn.c,cnn.h,weights.h,softmax.c,sampledata.h,sampleoutput.h,Makefile} "$DST/"

# 생성 main.c 에서 입력 배열 ~ check_output 까지 (Classification layer 앞까지)
{
  echo "// @generated — firmware/npu/sync_from_synth.sh 가 $NAME/main.c 에서 떼어 냈다."
  echo "// 손으로 고치지 말 것. 합성을 다시 하면 이 스크립트를 다시 돌린다."
  echo '#include <stdint.h>'
  echo '#include <stdio.h>'
  echo '#include "mxc.h"'
  echo '#include "cnn.h"'
  echo '#include "sampledata.h"'
  echo '#include "sampleoutput.h"'
  echo
  awk '/^\/\/ [0-9]+-channel .* data input/ {on=1} /^\/\/ Classification layer:/ {on=0} on' \
    "$SRC/main.c"
} > "$DST/kat_io.c"
grep -q 'void load_input' "$DST/kat_io.c" && grep -q 'int check_output' "$DST/kat_io.c" \
  || { echo "[에러] kat_io.c 추출 실패 — 생성 main.c 형식이 바뀌었다"; exit 1; }

cp "$P/firmware/common/measure_main.c" "$DST/main.c"
cat > "$DST/project.mk" <<EOF
# @generated — firmware/npu/sync_from_synth.sh
# 측정 빌드: LED 마킹·소프트웨어 타이머를 뺀다 (CLAUDE.md 6장 MEASURE_BUILD).
BOARD = FTHR_RevA
PROJ_CFLAGS += -DMEASURE_BUILD -DNET_NAME=\\"$NAME\\"
EOF

# cnn.h 의 LED 마킹·타이머를 측정 빌드에서 끈다 (생성 파일 뒤에 덧붙인다)
cat >> "$DST/cnn.h" <<'EOF'

// ── firmware/npu/sync_from_synth.sh 가 덧붙임 ─────────────────────────
// 측정 빌드에서는 추론 구간에 LED 토글·소프트웨어 타이머를 넣지 않는다.
// (UI 전력·오버헤드가 측정에 섞인다 — CLAUDE.md 6장)
#ifdef MEASURE_BUILD
#undef CNN_START
#undef CNN_COMPLETE
#undef SYS_START
#undef SYS_COMPLETE
#undef CNN_INFERENCE_TIMER
#define CNN_START
#define CNN_COMPLETE
#define SYS_START
#define SYS_COMPLETE
#endif
EOF
echo "✔ $DST"
ls "$DST"
