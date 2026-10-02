#!/usr/bin/env python3
"""int16 샤드가 기존 int8 샤드와 **같은 창**인지 대조하고 게인 분포를 낸다.

라운드 6 (b) 2단계의 전제 검사다. int16 샤드로 학습한 결과를 ④(int8 샤드)와
비교하려면 두 샤드가 **저장 형식만** 달라야 한다.

  1. `index.csv` 가 클래스·split 마다 **바이트 단위로 같다** (같은 클립·같은
     시작점·같은 행 순서)
  2. int16 을 기준 변환(`(x+32)>>6`)으로 내린 값이 기존 int8 과 **1 LSB 안**
     (float→int8 과 float→int16→int8 은 이중 반올림이라 정확히 같지는 않다)
  3. 정규화 게인 분포 — 상한(GMAX)에 붙는 비율, 감쇠되는 비율 (클래스별)

사용 (WSL2):
    ~/ai8x-training/venv/bin/python tools/verify_shards16.py
"""

import filecmp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets")]
import safesound as S                                       # noqa: E402

R8 = os.path.join(REPO, "data", "processed", "safesound")
R16 = os.path.join(REPO, "data", "processed", "safesound16")


def main():
    bad = 0
    print("1·2. 인덱스 일치 / int8 대조")
    print(f"  {'split/class':<20}{'창':>7}{'인덱스':>8}{'최대차':>7}{'다른 샘플':>11}")
    for sp in ("train", "test"):
        for c in S.CLASSES:
            a, b = (os.path.join(r, sp, c) for r in (R8, R16))
            same = filecmp.cmp(os.path.join(a, "index.csv"),
                               os.path.join(b, "index.csv"), shallow=False)
            mx, ndiff, n = 0, 0.0, 0
            for f in sorted(x for x in os.listdir(a) if x.endswith(".npy")):
                x8 = np.load(os.path.join(a, f), mmap_mode="r")
                x16 = np.load(os.path.join(b, f), mmap_mode="r")
                if x16.dtype != np.int16 or x8.shape != x16.shape:
                    mx, same = 999, False
                    continue
                d = np.abs(S.int16_to_int8(np.asarray(x16)).astype(np.int16)
                           - np.asarray(x8).astype(np.int16))
                mx = max(mx, int(d.max()))
                ndiff += float((d > 0).sum())
                n += d.size
            ok = same and mx <= 1
            bad += not ok
            print(f"  {sp + '/' + c:<20}{n // S.STORE:>7}"
                  f"{'같음' if same else '**다름**':>8}{mx:>7}"
                  f"{100 * ndiff / max(n, 1):>10.3f}%")

    print("\n3. 정규화 게인 분포 (시험셋, 무증강 — 가운데 1초)")
    for gmax in (4, 64):
        print(f"  GMAX x{gmax}")
        print(f"  {'class':<12}{'감쇠':>8}{'증폭<상한':>11}{'상한':>8}"
              f"{'게인 중앙(dB)':>15}")
        for c in S.CLASSES:
            d = os.path.join(R16, "test", c)
            gs = []
            for f in sorted(x for x in os.listdir(d) if x.endswith(".npy")):
                arr = np.load(os.path.join(d, f), mmap_mode="r")
                for row in arr:
                    g, _ = S.norm16(row[S.MARGIN:S.MARGIN + S.WIN], gmax)
                    gs.append(g)
            gs = np.array(gs, dtype=np.float64)
            cap = gmax * S.N16_UNITY
            print(f"  {c:<12}{100 * (gs < S.N16_UNITY).mean():>7.1f}%"
                  f"{100 * ((gs >= S.N16_UNITY) & (gs < cap)).mean():>10.1f}%"
                  f"{100 * (gs >= cap).mean():>7.1f}%"
                  f"{20 * np.log10(np.median(gs) / S.N16_UNITY):>15.1f}")

    print()
    if bad:
        print(f"실패 {bad}건 — int16 샤드가 기존 샤드와 같은 창이 아니다. 학습 금지.")
        sys.exit(1)
    print("통과 — 두 샤드는 저장 형식만 다르다")


if __name__ == "__main__":
    main()
