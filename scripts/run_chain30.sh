#!/bin/bash
# chain30 — 구성 (a) NPU 로그 근사(구간선형, `models/ai85net-safesound-logstage.py`)를 v4+D 설정으로 3시드. chain29 끝난 뒤 자동 시작.
#   사용자 지시 2026-10-11: "설정은 다른 v4+D 구성과 같게, 구조는 v1 때 (a)와 같게".
#   · 학습 설정 = run_chain28.sh 의 v4+D 와 같다 (epochs 150, Adam, wd 0, lr 0.001, schedule_safesound.yaml, QAT yaml,
#     KD T=2.0 α=0.5, 데이터셋 SafeSoundV4D = ④ v4+D 와 같은 창·교사). ⚠️ v1 의 (a)(chain13)는 lr 0.0003 + schedule_safesound_ft.yaml
#     이었다 — 이번엔 지시대로 v4+D 설정을 쓴다. 이 차이는 문서에 적는다.
#   · 구조 = v1 (a) 그대로: --model ai85safesoundlogstage --use-bias, 초기 가중치 data/logstage-init-fb.pth.tar
#     (D-1 멜 필터뱅크 conv1 + 로그 구간선형 K=4 마디 초기화 — 이 초기화가 "(a)" 의 정의다).
#     setsid nohup bash scripts/run_chain30.sh > data/logs-local/chain30.out 2>&1 &
set -u
T=~/ai8x-training; P=~/max78000-proj; OUT=$P/data/logs-local
export OPENBLAS_NUM_THREADS=1
export SAFESOUND_KD_T=2.0 SAFESOUND_KD_ALPHA=0.5
# chain29 가 끝났다는 표식을 기다린다 (실행 사이의 빈틈에 끼지 않도록 pgrep 대신 로그 표식)
until grep -q "chain29 끝" "$OUT/chain29.out" 2>/dev/null; do sleep 120; done
while pgrep -f 'train\.py|teacher_cache\.py' > /dev/null; do sleep 60; done
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "=== $(date +%F_%H:%M:%S)  MemAvailable ${AVAIL} MB (필요 ${MIN_MB:=4000} MB)"
if [ "$AVAIL" -lt "$MIN_MB" ]; then
  echo "!!! 메모리 부족 — chain30 를 시작하지 않는다"; exit 3
fi
[ -f "$P/data/logstage-init-fb.pth.tar" ] || { echo "!!! 초기 가중치 없음"; exit 5; }
echo "=== $(date +%F_%H:%M:%S)  chain30 시작  ((a) logstage, v4+D 설정, KD T=$SAFESOUND_KD_T α=$SAFESOUND_KD_ALPHA)"
cd "$T" || exit 1

run () {   # $1 이름  $2 시드
  echo "=== $(date +%F_%H:%M:%S)  시작 $1"
  "$T/venv/bin/python" "$T/train.py" \
    --epochs 150 --optimizer Adam --wd 0 --deterministic --seed "$2" \
    --lr 0.001 \
    --compress "$P/colab/schedule_safesound.yaml" \
    --qat-policy "$P/colab/qat_policy_safesound.yaml" \
    --model ai85safesoundlogstage --dataset SafeSoundV4D --device MAX78000 \
    --confusion --batch-size 128 --workers 2 --cpu --use-bias \
    --exp-load-weights-from "$P/data/logstage-init-fb.pth.tar" \
    --name "$1" --out-dir "$OUT" > "$OUT/$1.out" 2>&1
  local rc=$?
  echo "=== $(date +%F_%H:%M:%S)  종료 $1 (exit $rc)"
  [ "$rc" -ne 0 ] && echo "!!! $1 실패 — 남은 학습을 중단한다" && exit 1
  return 0
}
run safesound-v4d-logstage    1
run safesound-v4d-logstage-s2 2
run safesound-v4d-logstage-s3 3
echo "=== $(date +%F_%H:%M:%S)  chain30 끝 ((a) v4+D 3회) — tools/audit_runs.py 로 완주 확인 뒤 채점."
