#!/bin/bash
# chain28 재개 — 호스트 재부팅(2026-10-10 23:06)으로 끊긴 v4+D½ s2·s3 만 다시 돈다. 설정은 run_chain28.sh 와 같다.
set -u
T=~/ai8x-training; P=~/max78000-proj; OUT=$P/data/logs-local
export OPENBLAS_NUM_THREADS=1
export SAFESOUND_KD_T=2.0 SAFESOUND_KD_ALPHA=0.5
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "=== $(date +%F_%H:%M:%S)  chain28 재개 (v4+D½ s2·s3)  MemAvailable ${AVAIL} MB"
[ "$AVAIL" -lt 4000 ] && echo "!!! 메모리 부족" && exit 3
cd "$T" || exit 1
run () {
  echo "=== $(date +%F_%H:%M:%S)  시작 $1"
  "$T/venv/bin/python" "$T/train.py" --epochs 150 --optimizer Adam --wd 0 --deterministic --seed "$4" --lr 0.001 \
    --compress "$P/colab/schedule_safesound.yaml" --qat-policy "$P/colab/qat_policy_safesound.yaml" \
    --model "$2" --dataset "$3" --device MAX78000 --confusion --batch-size 128 --workers 2 --cpu \
    --name "$1" --out-dir "$OUT" > "$OUT/$1.out" 2>&1
  local rc=$?; echo "=== $(date +%F_%H:%M:%S)  종료 $1 (exit $rc)"
  [ "$rc" -ne 0 ] && echo "!!! $1 실패" && exit 1
  return 0
}
run safesound-v4dh-melh400-s2 ai85safesoundmelnet SafeSoundMelH400V4Dh 2
run safesound-v4dh-melh400-s3 ai85safesoundmelnet SafeSoundMelH400V4Dh 3
echo "=== $(date +%F_%H:%M:%S)  chain28 끝 (재개분 v4+D½ s2·s3)"
