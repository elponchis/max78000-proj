#!/bin/bash
# chain29 — 후보 가(u1000)·다(u800)·①′(멜 증분) 를 v4+D 설정(정리 데이터 + PANNs 프로브 증류, 라벨 없는 창 전체 U)으로 3시드씩.
#   사용자 결정 2026-10-11: 본 표를 v4+D 로 바꾸므로 다섯 구성을 같은 설정으로 채운다. 설정은 run_chain28.sh 의
#   safesound-v4d-melh400 과 **완전히 같다** (epochs 150, Adam, wd 0, lr 0.001, 같은 스케줄·QAT yaml, KD T=2.0 α=0.5, 시드 1~3).
#   다른 것은 표현(scheme)과 그 표현의 모델(가·다 는 풀링 4단 ai85safesoundmelnet_p4)뿐 — v2.1·v3 체인과 같은 짝이다.
#   채점은 끝난 뒤 tools/audit_runs.py 확인 → tools/eval_v4.py (testL / real31L).
#     setsid nohup bash scripts/run_chain29.sh > data/logs-local/chain29.out 2>&1 &
set -u
T=~/ai8x-training; P=~/max78000-proj; OUT=$P/data/logs-local
export OPENBLAS_NUM_THREADS=1
export SAFESOUND_KD_T=2.0 SAFESOUND_KD_ALPHA=0.5
while pgrep -f 'train\.py|teacher_cache\.py' > /dev/null; do sleep 60; done
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "=== $(date +%F_%H:%M:%S)  MemAvailable ${AVAIL} MB (필요 ${MIN_MB:=4000} MB)"
if [ "$AVAIL" -lt "$MIN_MB" ]; then
  echo "!!! 메모리 부족 — chain29 를 시작하지 않는다"; exit 3
fi
if pgrep -f 'download_clips\.py' > /dev/null; then
  echo "!!! 다운로드가 돌고 있다 — chain29 를 시작하지 않는다"; exit 4
fi
echo "=== $(date +%F_%H:%M:%S)  chain29 시작  (KD T=$SAFESOUND_KD_T α=$SAFESOUND_KD_ALPHA; 데이터 루트 SafeSoundV4 + SafeSoundV4U, 교사 safesound_v4/teacher)"
cd "$T" || exit 1

run () {   # $1 이름  $2 모델  $3 데이터셋  $4 시드   — run_chain28.sh 의 run 과 글자 단위로 같은 인자
  echo "=== $(date +%F_%H:%M:%S)  시작 $1"
  "$T/venv/bin/python" "$T/train.py" \
    --epochs 150 --optimizer Adam --wd 0 --deterministic --seed "$4" \
    --lr 0.001 \
    --compress "$P/colab/schedule_safesound.yaml" \
    --qat-policy "$P/colab/qat_policy_safesound.yaml" \
    --model "$2" --dataset "$3" --device MAX78000 \
    --confusion --batch-size 128 --workers 2 --cpu \
    --name "$1" --out-dir "$OUT" > "$OUT/$1.out" 2>&1
  local rc=$?
  echo "=== $(date +%F_%H:%M:%S)  종료 $1 (exit $rc)"
  [ "$rc" -ne 0 ] && echo "!!! $1 실패 — 남은 학습을 중단한다" && exit 1
  return 0
}

seeds () {   # $1 이름  $2 모델  $3 데이터셋  $4 시드 수
  run "$1" "$2" "$3" 1
  for s in $(seq 2 "$4"); do run "$1-s$s" "$2" "$3" "$s"; done
}

seeds safesound-v4d-melu1000 ai85safesoundmelnet_p4 SafeSoundMelU1000V4D 3
seeds safesound-v4d-melu800  ai85safesoundmelnet_p4 SafeSoundMelU800V4D  3
seeds safesound-v4d-melinc   ai85safesoundmelnet    SafeSoundMelIncV4D   3
echo "=== $(date +%F_%H:%M:%S)  chain29 끝 (가·다·①′ v4+D 9회) — tools/audit_runs.py 로 완주 확인 뒤 채점."
