#!/usr/bin/env python3
"""Colab 에 올릴 **우리 코드만** tar 로 묶는다.

레포 전체를 clone 하지 않는 이유는 둘이다. 원본 오디오·샤드가 들어 있는
`data/` 를 빼야 하고, GitHub 에 push 하지 않은 상태에서도 Colab 이 돌아가야 한다.
`colab/baseline.ipynb` 가 이 tar 를 Drive 에서 받아 푼다.

코드가 바뀌면 **다시 묶어 Drive 에 올려야 한다** — 노트북은 tar 안의 코드를 쓴다.

사용법 (WSL2):
    python3 tools/pack_code.py
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Colab 학습·평가에 실제로 필요한 것만. 늘릴 때는 "노트북이 이걸 import 하나?"를
# 먼저 물을 것 — 안 쓰는 파일이 들어가면 버전이 어긋나도 모른다.
INCLUDE = [
    "tools/eval_threshold.py",            # 배경음 로짓 오프셋 스윕
    "tools/parse_trainlog.py",            # 학습 곡선·QAT 전환 요약
    "tools/export_errors.py",             # 오분류 창 wav 추출 (청취 판정)
    "datasets/safesound.py",              # 데이터로더 (ai8x datasets/ 로 링크)
    "models/ai85net-safesound.py",        # 모델 (ai8x models/ 로 링크)
    "scripts/prepare_safesound.py",       # stream_eval 이 전처리 상수를 가져다 쓴다
    "tools/eval_confusion.py",            # 창/원본 혼동행렬 + CI + 오경보
    "tools/stream_eval.py",               # 빈틈없는 스트리밍 오탐 (원본 오디오 필요)
    "tools/kat_safesound.py",             # 학습 전 필수 검증
    "tools/kat_vectors",                  # KAT 기준 벡터 (디렉터리)
    # ── G8 구성 ① (로그 멜 + 2D CNN). 구성 ④와 같은 샤드를 읽는다
    "datasets/melfeat.py",                # 로그 멜 프론트엔드 (numpy 전용)
    "datasets/safesound_mel.py",          # SafeSound 상속 로더
    "datasets/safesound_wave2d.py",       # wave2D 대조 (표현/구조 분리)
    "models/ai85net-safesound-mel.py",    # 2D CNN
    "tools/kat_safesound_mel.py",         # 멜 경로 KAT
    "tools/kat_models.py",                # 모델 forward 회귀 (배치 크기·레이아웃)
    "tools/init_from_kws20.py",           # B: KWS20 v3 사전학습 초기화
    "tools/init_filterbank.py",           # D-1: 멜 필터뱅크 초기화 (구성 ③)
    "colab/schedule_safesound_ft.yaml",   # 파인튜닝 LR 스케줄
    "tools/mel_range_sweep.py",           # dB 구간 근거 재현용
    "tools/input_stats.py",               # 입력 int8 통계 (붕괴 원인 판별)
    "colab/qat_policy_safesound.yaml",
    "colab/schedule_safesound.yaml",
    "CLAUDE.md",                          # 규칙 참조용 (셀에서 인용한다)
]


def git(*args):
    try:
        return subprocess.run(("git",) + args, cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/safesound-code-v1.tar.gz")
    a = ap.parse_args()

    missing = [p for p in INCLUDE if not os.path.exists(os.path.join(REPO, p))]
    if missing:
        sys.exit(f"[에러] 없는 경로: {missing}")

    manifest = {
        "kind": "safesound-code",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "files": INCLUDE,
    }
    if manifest["dirty"]:
        print("⚠️ 커밋되지 않은 변경이 있다 — 재현하려면 먼저 커밋할 것.")

    out = os.path.join(REPO, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mpath = os.path.join(REPO, "CODE_MANIFEST.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    try:
        with tarfile.open(out, "w:gz") as tf:
            for p in INCLUDE:
                tf.add(os.path.join(REPO, p), arcname=p)
            tf.add(mpath, arcname="CODE_MANIFEST.json")
    finally:
        os.remove(mpath)

    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    with open(out + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{h.hexdigest()}  {os.path.basename(out)}\n")

    print(f"tar    {a.out}  ({os.path.getsize(out) / 1024:.0f}KB, {len(INCLUDE)}개 경로)")
    print(f"sha256 {h.hexdigest()}")
    print(f"commit {manifest['commit'][:9]}  dirty={manifest['dirty']}")


if __name__ == "__main__":
    main()
