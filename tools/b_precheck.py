#!/usr/bin/env python3
"""B 사전 확인 — STFT 를 NPU 의 고정 Conv1d 로 옮기면 특징이 얼마나 달라지나.

구성 B: NPU 고정 Conv1d(DFT 기저 × Hann, 정수 가중치) → int32 → CPU 가
파워·멜·로그 → 기존 ①′ 모델. 프레임 정의는 ①′ (hop 250, 패딩 없음).

이 스크립트는 **재학습 없이** 답할 수 있는 것만 본다.
  1. 하드웨어 수용: Conv1d 형태, 가중치 바이트, 442KB 에 ① 모델과 함께 들어가나
  2. 특징 차이: 정수 기저(8bit / 4bit) 대 float FFT — int8 로그 멜의 불일치·최대 차이
정확도(①′ 체크포인트를 그 특징으로 평가)는 `cmp_all.sh` 가 낸다.

사용 (WSL2):  python3 tools/b_precheck.py [--n 100]
"""

import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
import melfeat as MF                                         # noqa: E402

WIN, MARGIN = 16384, 1600
CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
MODEL_BYTES = 157472          # ① / ①′ 2D CNN (합성 실측)
LIMIT = 442368


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--root", default=os.path.join(REPO, "data/processed/safesound"))
    a = ap.parse_args()

    print("1. 하드웨어 수용 (계산)")
    hop, nfft = MF.HOP_INC, MF.N_FFT
    k = -(-nfft // hop)                       # 한 프레임이 걸치는 hop 수 (올림)
    bins = nfft // 2 - 1                      # 빈 1..255 (0·256 은 멜에 안 걸린다)
    co = 2 * bins
    print(f"  입력 배치: 채널 {hop} (hop 한 칸의 샘플) x 시간. 프레임 = 연속 {k}칸")
    print(f"  Conv1d(in={hop}, out={co} (re/im x {bins}빈), k={k}), 활성화 없음, 32bit 출력")
    print(f"  ⚠️ {k}번째 탭은 {nfft - (k - 1) * hop}샘플만 쓰고 나머지 가중치는 0 이다")
    print(f"  ⚠️ 32bit 출력은 **마지막 층에서만** 된다 → STFT 층이 독립 네트워크여야 한다")
    print(f"     ((c) 처럼 두 네트워크를 함께 적재 — synthesis-check 와 STATUS 5.5절)")
    n_w = hop * co * k
    print(f"  {'가중치':<8}{'STFT 층':>12}{'+ ① 모델':>12}{'442KB 대비':>12}  판정")
    for bits in (8, 4, 2):
        b = n_w * bits // 8
        tot = b + MODEL_BYTES
        print(f"  {bits}bit   {b:>11,}B{tot:>11,}B{100 * tot / LIMIT:>11.1f}%  "
              f"{'들어간다' if tot <= LIMIT else '**초과**'}")
    macc = n_w * MF.N_FRAMES
    print(f"  MAC: 판단마다 64프레임 전부면 {macc:,} (① 모델 18.9M 의 {macc / 18.94e6:.2f}배)")
    print(f"       증분(16프레임)이면 {macc // 4:,}")
    print(f"  출력 데이터: {co}ch x 64프레임 x 4B = {co * 64 * 4:,} B (512KB 중 "
          f"{100 * co * 64 * 4 / 524288:.1f}%)")

    print("\n2. 특징 차이 — 정수 기저 대 float FFT (int8 로그 멜, 시험셋)")
    print(f"  {'기저':<8}{'창':>6}{'불일치':>10}{'>1 LSB':>10}{'>2 LSB':>10}{'최대':>6}{'평균|차|':>10}")
    rng = np.random.default_rng(0)
    rows = []
    for c in CLASSES:
        r = []
        for f in sorted(glob.glob(os.path.join(a.root, "test", c, "shard_*.npy"))):
            arr = np.load(f, mmap_mode="r")
            r += [(arr, i) for i in range(len(arr))]
        rows += [r[j] for j in rng.permutation(len(r))[:a.n]]
    for scheme in ("loginc_q8", "loginc_q4"):
        n = bad = g1 = g2 = 0
        mx, sm = 0, 0.0
        for arr, i in rows:
            w = np.asarray(arr[i][MARGIN:MARGIN + WIN])
            d = np.abs(MF.mel_int8(w, scheme).astype(np.int16)
                       - MF.mel_int8(w, "loginc").astype(np.int16))
            n += d.size
            bad += int((d > 0).sum())
            g1 += int((d > 1).sum())
            g2 += int((d > 2).sum())
            mx = max(mx, int(d.max()))
            sm += float(d.sum())
        print(f"  {scheme[-2:]:<8}{len(rows):>6}{100 * bad / n:>9.2f}%{100 * g1 / n:>9.2f}%"
              f"{100 * g2 / n:>9.2f}%{mx:>6}{sm / n:>10.3f}")


if __name__ == "__main__":
    main()
