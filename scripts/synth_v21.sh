#!/bin/bash
# v2.1 시드 1 체크포인트 합성 → NPU 측정 펌웨어 빌드 (5개 구성). **WSL2.**
#   bash data/logs-local/synth_v21.sh wave|h400|u1000|u800|melinc   (인자 없으면 다섯 전부)
# 산출물 prefix 는 `<v1 prefix>_v21` — v1 합성·펌웨어(git 에 있음)를 덮지 않는다.
# 절차는 synth_variant.sh / synth_melinc.sh / synth_w050.sh 와 같다 (양자화 → -8 평가로
# 샘플 입력 → ai8xize --softmax → sync_from_synth → fw_build). 샘플 입력은 v1 과 같은
# 시험셋 로더로 만든다 (v2.1 은 학습 증강만 다르고 시험셋은 같다).
set -u
P=~/max78000-proj; T=~/ai8x-training; S=~/ai8x-synthesis
export OPENBLAS_NUM_THREADS=1

one() {
  local V=$1 RUN=$2 MODEL=$3 DS=$4 YAML=$5 PREFIX=$6
  local D CK Q LOG LOW
  D=$(ls -dt $P/data/logs-local/${RUN}___* | head -1)          # 시드 1 = 접미사 없는 실행
  CK=$D/${RUN}_qat_best.pth.tar
  Q=$P/data/synth/${RUN}-q8.pth.tar
  LOG=$P/data/logs-local/synth_v21_$V
  LOW=$(echo "$DS" | tr '[:upper:]' '[:lower:]')
  echo "===== $V  ($CK)"
  [ -f "$CK" ] || { echo "[에러] 체크포인트 없음"; return 1; }

  echo "## 1) 양자화"
  cd "$S" || return 1
  venv/bin/python quantize.py "$CK" "$Q" --device MAX78000 > "$LOG-quantize.out" 2>&1
  echo "exit $?"

  echo "## 2) 샘플 입력 (-8 평가, 양자화 체크포인트로, 시험셋 로더 $DS)"
  cd "$T" || return 1
  venv/bin/python train.py --model "$MODEL" --dataset "$DS" \
    --device MAX78000 --cpu --evaluate -8 --save-sample 10 --confusion \
    --batch-size 128 --workers 2 --exp-load-weights-from "$Q" \
    --out-dir /tmp/synth_v21_${V}_eval --name synthv21$V > "$LOG-eval8.out" 2>&1
  echo "exit $?"; grep -E '==> Top1' "$LOG-eval8.out" | tail -1
  ls -la "sample_$LOW.npy" && cp "sample_$LOW.npy" "$S/tests/" \
    && cp "sample_$LOW.npy" "$P/data/synth/sample_${RUN}.npy"

  echo "## 3) 합성"
  cd "$S" || return 1
  venv/bin/python ai8xize.py --verbose --test-dir "$P/data/synth/out" \
    --prefix "$PREFIX" --checkpoint-file "$Q" --config-file "$P/synthesis/$YAML" \
    --softmax --device MAX78000 --compact-data --mexpress --timer 0 \
    --display-checkpoint --overwrite > "$LOG-ai8xize.out" 2>&1
  echo "exit $?"; grep -E 'ERROR' "$LOG-ai8xize.out" | head -3
  grep -nE 'Total +[0-9,]+ cycles|Weight memory|Bias memory' \
    "$P/data/synth/out/$PREFIX/log.txt" | tail -3
  grep -A3 'SAMPLE_OUTPUT' "$P/data/synth/out/$PREFIX/sampleoutput.h" | head -4

  echo "## 4) 펌웨어 프로젝트·빌드"
  cd "$P" && bash firmware/npu/sync_from_synth.sh "$PREFIX" | head -1 \
    && bash scripts/fw_build.sh "firmware/npu/$PREFIX" 2>&1 | grep -E '성공|실패|Flash|SRAM'
}

#      V       실행 이름                모델                     샘플용 데이터셋      yaml                        prefix
run_wave()   { one wave   safesound-v21-wave     ai85safesoundnet        SafeSound          safesound-wave-hwc.yaml     safesound_wave_v21; }
run_h400()   { one h400   safesound-v21-melh400  ai85safesoundmelnet     SafeSoundMelH400   safesound-melh400-hwc.yaml  safesound_melh400_v21; }
run_u1000()  { one u1000  safesound-v21-melu1000 ai85safesoundmelnet_p4  SafeSoundMelU1000  safesound-melu1000-hwc.yaml safesound_melu1000_v21; }
run_u800()   { one u800   safesound-v21-melu800  ai85safesoundmelnet_p4  SafeSoundMelU800   safesound-melu800-hwc.yaml  safesound_melu800_v21; }
run_melinc() { one melinc safesound-v21-melinc   ai85safesoundmelnet     SafeSoundMelInc    safesound-melinc-hwc.yaml   safesound_melinc_v21; }

if [ $# -eq 0 ]; then set -- wave h400 u1000 u800 melinc; fi
for v in "$@"; do run_$v; done
echo "=== synth_v21 끝 $(date +%H:%M) ==="
