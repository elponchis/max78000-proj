#!/usr/bin/env python3
"""C 멜 전처리(`firmware/common/melfeat.c`) 대 `datasets/melfeat.py` 대조 — PC.

펌웨어와 **같은 소스**를 `-DMELFEAT_HOST` 로 PC 에서 컴파일해 (FFT 만 단순
radix-2 로 바뀐다) 시험셋 창에 돌리고, 기준 구현의 int8 출력과 비교한다.

  · 불일치 원소 비율, 최대 절대 차이, 2 LSB 초과 비율 (허용 오차 ±2 LSB —
    `tools/kat_safesound_mel.py` 의 TOL_LSB)
  · `--dump` 를 주면 C 특징을 npy 로 저장한다 (① 체크포인트 재평가용)

⚠️ 보드의 FFT 는 CMSIS-DSP 다. 이 대조는 **FFT 를 뺀 나머지**(reflect·창·멜·
로그·양자화)의 이식을 본다. FFT 까지 포함한 대조는 보드 KAT 가 한다.

사용 (WSL2):  python3 tools/kat_melfeat_c.py [--n 500]
"""

import argparse
import ctypes
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
import melfeat as MF                                         # noqa: E402

WIN, MARGIN = 16384, 1600
CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]


def build():
    so = os.path.join(tempfile.gettempdir(), "libmelfeat_host.so")
    src = os.path.join(REPO, "firmware", "common", "melfeat.c")
    subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-DMELFEAT_HOST",
                    "-o", so, src, "-lm"], check=True)
    lib = ctypes.CDLL(so)
    lib.melfeat_init()
    return lib


def c_mel(lib, w):
    out = np.zeros(MF.N_MELS * MF.N_FRAMES, dtype=np.int8)
    w = np.ascontiguousarray(w, dtype=np.int8)
    lib.melfeat_int8(w.ctypes.data_as(ctypes.c_void_p),
                     out.ctypes.data_as(ctypes.c_void_p))
    return out.reshape(MF.N_MELS, MF.N_FRAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="클래스당 최대 창 수")
    ap.add_argument("--root", default=os.path.join(REPO, "data/processed/safesound"))
    a = ap.parse_args()
    lib = build()

    # 1) 고정 KAT 벡터
    d = os.path.join(REPO, "tools", "kat_vectors")
    win = np.load(os.path.join(d, "melkat_window_int8.npy"))
    ref = np.load(os.path.join(d, "melkat_logmel_int8.npy"))
    got = c_mel(lib, win)
    df = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    print(f"KAT 벡터: 불일치 {int((df > 0).sum())}/{df.size}  최대 차이 {int(df.max())}")

    # 2) 시험셋 창
    print(f"\n시험셋 (클래스당 최대 {a.n}창)")
    print(f"  {'class':<12}{'창':>6}{'불일치':>10}{'>1 LSB':>10}{'>2 LSB':>10}{'최대':>6}")
    rng = np.random.default_rng(0)
    worst, tot_bad, tot = 0, 0, 0
    for c in CLASSES:
        rows = []
        for f in sorted(glob.glob(os.path.join(a.root, "test", c, "shard_*.npy"))):
            arr = np.load(f, mmap_mode="r")
            rows += [(arr, i) for i in range(len(arr))]
        sel = rng.permutation(len(rows))[:a.n]
        n = bad = g1 = g2 = 0
        mx = 0
        for j in sel:
            arr, i = rows[j]
            w = np.asarray(arr[i][MARGIN:MARGIN + WIN])
            df = np.abs(c_mel(lib, w).astype(np.int16)
                        - MF.log_mel_int8(w).astype(np.int16))
            n += df.size
            bad += int((df > 0).sum())
            g1 += int((df > 1).sum())
            g2 += int((df > 2).sum())
            mx = max(mx, int(df.max()))
        worst = max(worst, mx)
        tot_bad += bad
        tot += n
        print(f"  {c:<12}{len(sel):>6}{100 * bad / n:>9.3f}%{100 * g1 / n:>9.4f}%"
              f"{100 * g2 / n:>9.4f}%{mx:>6}")
    print(f"\n전체 불일치 {100 * tot_bad / tot:.3f}%  최대 차이 {worst} LSB "
          f"(허용 ±2 LSB) → {'통과' if worst <= 2 else '**실패**'}")
    sys.exit(0 if worst <= 2 else 1)


if __name__ == "__main__":
    main()
