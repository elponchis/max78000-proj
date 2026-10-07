#!/bin/bash
# chain26 — 데이터셋 v3 1단계: ④ 와 간격 400, 각 3시드. WSL2 CPU.
#   사전 등록: docs/results/dataset-v3-design.md 5·6·11절. 학습 설정·시드는 v1·v2.1(chain25)과 같다 —
#   데이터셋 이름만 V3 (루트 SafeSoundV3 = v1 샤드 링크 + AudioSet 샤드, 증강 v2.1 그대로).
#   2단계(가·다·①′)는 1단계 판정(조건 A~E) 통과 후에만 별도로 돌린다 — 이 스크립트에 넣지 않는다.
#     setsid nohup bash scripts/run_chain26.sh > data/logs-local/chain26.out 2>&1 &
set -u
T=~/ai8x-training; P=~/max78000-proj; OUT=$P/data/logs-local
export OPENBLAS_NUM_THREADS=1
while pgrep -f 'train\.py' > /dev/null; do sleep 60; done
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "=== $(date +%H:%M:%S)  MemAvailable ${AVAIL} MB (필요 ${MIN_MB:=4000} MB)"
if [ "$AVAIL" -lt "$MIN_MB" ]; then
  echo "!!! 메모리 부족 — chain26 를 시작하지 않는다"; exit 3
fi
if pgrep -f 'download_clips\.py' > /dev/null; then
  echo "!!! 다운로드가 돌고 있다 — chain26 를 시작하지 않는다 (설계안 5절)"; exit 4
fi
echo "=== $(date +%H:%M:%S)  chain26 시작  (v3 MANIFEST: $(python3 -c "import json;m=json.load(open('$P/data/processed/safesound_v3/MANIFEST.json'));print(m['tag'],m['created'],{c:v['total'] for c,v in m['counts'].items()})"))"
cd "$T" || exit 1

run () {   # $1 이름  $2 모델  $3 데이터셋  $4 시드
  echo "=== $(date +%H:%M:%S)  시작 $1"
  "$T/venv/bin/python" "$T/train.py" \
    --epochs 150 --optimizer Adam --wd 0 --deterministic --seed "$4" \
    --lr 0.001 \
    --compress "$P/colab/schedule_safesound.yaml" \
    --qat-policy "$P/colab/qat_policy_safesound.yaml" \
    --model "$2" --dataset "$3" --device MAX78000 \
    --confusion --batch-size 128 --workers 2 --cpu \
    --name "$1" --out-dir "$OUT" > "$OUT/$1.out" 2>&1
  local rc=$?
  echo "=== $(date +%H:%M:%S)  종료 $1 (exit $rc)"
  [ "$rc" -ne 0 ] && echo "!!! $1 실패 — 남은 학습을 중단한다" && exit 1
  return 0
}

seeds () {   # $1 이름  $2 모델  $3 데이터셋  $4 시드 수
  run "$1" "$2" "$3" 1
  for s in $(seq 2 "$4"); do run "$1-s$s" "$2" "$3" "$s"; done
}

seeds safesound-v3-wave     ai85safesoundnet      SafeSoundV3         3
seeds safesound-v3-melh400  ai85safesoundmelnet   SafeSoundMelH400V3  3
echo "=== $(date +%H:%M:%S)  chain26 끝 (1단계) — tools/audit_runs.py 로 완주 확인 후 tools/eval_v3.py --v3"
