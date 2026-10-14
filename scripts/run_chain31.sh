#!/bin/bash
# chain31 — v5+D: 청취 X·? 130창 제외 + 고함 정의 분리 (scream-plan.md 9절, 사용자 승인 2026-10-14) 로 다섯 구성 3시드씩.
#   설정은 run_chain29.sh / run_chain28.sh 의 run 과 **글자 단위로 같다** (epochs 150, Adam, wd 0, lr 0.001, 같은 스케줄·QAT yaml,
#   KD T=2.0 α=0.5, 시드 1~3, 라벨 없는 창 전체 U). 다른 것은 데이터셋(*V5D: 루트 SafeSoundV5 + SafeSoundV5U, 교사 safesound_v5/teacher)뿐.
#   순서는 짧은 것부터 (④ → 다 → 가 → 간격 400 → ①′) — 이른 신호를 보고 ①′ 전에 멈출 수 있게. (a) 는 돌리지 않는다.
#   ⚠️ 시작 조건: testL-S3(시험셋 비명 청취 정답) 이 고정·커밋된 뒤에만 시작한다 (scream-plan.md 9.3·9.7). 채점은 완주 뒤 한 번에.
#     setsid nohup bash scripts/run_chain31.sh > data/logs-local/chain31.out 2>&1 &
set -u
T=~/ai8x-training; P=~/max78000-proj; OUT=$P/data/logs-local
export OPENBLAS_NUM_THREADS=1
export SAFESOUND_KD_T=2.0 SAFESOUND_KD_ALPHA=0.5
while pgrep -f 'train\.py|teacher_cache\.py' > /dev/null; do sleep 60; done
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "=== $(date +%F_%H:%M:%S)  MemAvailable ${AVAIL} MB (필요 ${MIN_MB:=4000} MB)"
if [ "$AVAIL" -lt "$MIN_MB" ]; then
  echo "!!! 메모리 부족 — chain31 을 시작하지 않는다"; exit 3
fi
if pgrep -f 'download_clips\.py' > /dev/null; then
  echo "!!! 다운로드가 돌고 있다 — chain31 을 시작하지 않는다"; exit 4
fi
if [ ! -s "$P/data_listened_test_scream.csv" ]; then
  echo "!!! testL-S3 정답(data_listened_test_scream.csv)이 없다 — chain31 을 시작하지 않는다 (scream-plan.md 9.3)"; exit 5
fi
MAN=$(python3 -c "import json;m=json.load(open('$P/data/processed/safesound_v5/MANIFEST.json'));print(m['tag'],m['created'],{c:v['total'] for c,v in m['counts'].items()})")
TJ=$(python3 -c "import json;j=json.load(open('$P/data/processed/safesound_v5/teacher/teacher.json'));p=j['probe'];print('probe wd',p['wd'],'ep',p['epoch'],'valF1',round(p['val_macro_f1_argmax'],4),'effT2',round(j['logits']['labeled']['eff_classes_T2'],2),'/',round(j['logits']['unlabeled']['eff_classes_T2'],2))")
echo "=== $(date +%F_%H:%M:%S)  chain31 시작  (v5 MANIFEST: $MAN; 교사: $TJ; KD T=$SAFESOUND_KD_T α=$SAFESOUND_KD_ALPHA)"
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

seeds safesound-v5d-wave     ai85safesoundnet       SafeSoundV5D         3
seeds safesound-v5d-melu800  ai85safesoundmelnet_p4 SafeSoundMelU800V5D  3
seeds safesound-v5d-melu1000 ai85safesoundmelnet_p4 SafeSoundMelU1000V5D 3
seeds safesound-v5d-melh400  ai85safesoundmelnet    SafeSoundMelH400V5D  3
seeds safesound-v5d-melinc   ai85safesoundmelnet    SafeSoundMelIncV5D   3
echo "=== $(date +%F_%H:%M:%S)  chain31 끝 (v5+D 5구성 15회) — tools/audit_runs.py 로 완주 확인 뒤 채점 (testL-S3 / real31L)."
