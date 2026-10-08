#!/bin/bash
# chain28 — v4 (정리) → v4+D (PANNs 프로브 증류 + 라벨 없는 창 U) → v4+D½ (라벨 없는 창 영상 절반). WSL2 CPU.
#   사전 등록: docs/results/v4-distill-design.md 4·5절 (2026-10-09). 학습 설정·시드는 chain25~27 과 같다.
#   ⚠️ 채점은 하지 않는다 — 사용자 청취(audit_test A·B·C) 가 끝난 뒤 같은 정답으로 v2.1·v3.1·v4·v4+D·v4+D½ 를 한꺼번에.
#   증류 온도·비중: SAFESOUND_KD_T=2.0, SAFESOUND_KD_ALPHA=0.5 (train.py 의 캐시 교사 분기).
#     setsid nohup bash scripts/run_chain28.sh > data/logs-local/chain28.out 2>&1 &
set -u
T=~/ai8x-training; P=~/max78000-proj; OUT=$P/data/logs-local
export OPENBLAS_NUM_THREADS=1
export SAFESOUND_KD_T=2.0 SAFESOUND_KD_ALPHA=0.5
while pgrep -f 'train\.py|teacher_cache\.py' > /dev/null; do sleep 60; done
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "=== $(date +%H:%M:%S)  MemAvailable ${AVAIL} MB (필요 ${MIN_MB:=4000} MB)"
if [ "$AVAIL" -lt "$MIN_MB" ]; then
  echo "!!! 메모리 부족 — chain28 를 시작하지 않는다"; exit 3
fi
if pgrep -f 'download_clips\.py' > /dev/null; then
  echo "!!! 다운로드가 돌고 있다 — chain28 를 시작하지 않는다"; exit 4
fi
echo "=== $(date +%H:%M:%S)  chain28 시작  (v4 MANIFEST: $(python3 -c "import json;m=json.load(open('$P/data/processed/safesound_v4/MANIFEST.json'));print(m['tag'],m['created'],{c:v['total'] for c,v in m['counts'].items()})"); 교사: $(python3 -c "import json;m=json.load(open('$P/data/processed/safesound_v4/teacher/teacher.json'));print('probe wd',m['probe']['wd'],'ep',m['probe']['epoch'],'valF1',round(m['probe']['val_macro_f1_argmax'],4),'effT2',round(m['logits']['labeled']['eff_classes_T2'],2),'/',round(m['logits']['unlabeled']['eff_classes_T2'],2))"); KD T=$SAFESOUND_KD_T α=$SAFESOUND_KD_ALPHA)"
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

seeds safesound-v4-wave      ai85safesoundnet      SafeSoundV4           3
seeds safesound-v4-melh400   ai85safesoundmelnet   SafeSoundMelH400V4    3
echo "=== $(date +%H:%M:%S)  v4 끝 (정리만, 6회)"
seeds safesound-v4d-wave     ai85safesoundnet      SafeSoundV4D          3
seeds safesound-v4d-melh400  ai85safesoundmelnet   SafeSoundMelH400V4D   3
echo "=== $(date +%H:%M:%S)  v4+D 끝 (U, 6회) — 중간 보고 시점"
seeds safesound-v4dh-melh400 ai85safesoundmelnet   SafeSoundMelH400V4Dh  3
echo "=== $(date +%H:%M:%S)  chain28 끝 (v4+D½ 3회) — tools/audit_runs.py 로 완주 확인. 채점은 청취 정답 확정 뒤."
