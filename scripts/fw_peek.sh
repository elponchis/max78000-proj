#!/bin/bash
# 측정 펌웨어의 결과 배열을 **SWD 로 RAM 에서 직접** 읽는다 — WSL root.
#   wsl.exe -d max78000 -u root -- bash scripts/fw_peek.sh firmware/npu/safesound_wave
# 시리얼이 깨졌거나(POR 전) 안 나올 때 쓴다. 펌웨어가 재출력 루프에 들어가
# 있어야 한다 (`report()` 가 배열을 정렬해 둔 뒤 — [0]=최소, [N/2]=중앙, [N-1]=최악).
set -u
DIR=${1:?프로젝트 디렉터리}
ELF=$DIR/build/max78000.elf
N=${N_ITER:-1000}
SCRIPTS=${OOCD_SCRIPTS:-/usr/local/share/openocd/scripts}
sym() { arm-none-eabi-nm "$ELF" | awk -v s="$1" '$3==s {print "0x"$1}'; }
CMDS=(-c init -c halt -c "reg pc")
for s in t_load t_infer t_unload; do
  a=$(sym $s); [ -n "$a" ] || { echo "[에러] 심볼 없음: $s"; exit 1; }
  CMDS+=(-c "echo SYM_$s" -c "mdw $a 1" -c "mdw $((a + 4 * (N / 2))) 1" -c "mdw $((a + 4 * (N - 1))) 1")
done
CMDS+=(-c "echo SYM_use_systick" -c "mdw $(sym use_systick) 1" -c resume -c exit)
openocd -s "$SCRIPTS" -f interface/cmsis-dap.cfg -f target/max78000.cfg \
  -c "adapter speed 2000" "${CMDS[@]}" 2>&1 | grep -E '^pc|^SYM_|^0x'
