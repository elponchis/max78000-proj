#!/usr/bin/env python3
"""음량 정규화(라운드 6 (b) 1단계)의 게인 분포와 상한 포화를 학습 전에 본다.

`docs/results/g8-round6-design.md` b.6 의 판정 항목 중 두 개를 미리 잰다.

  · **게인 분포** — `g` 가 대부분 GMAX 에 붙어 있으면 TARGET 이 너무 높다
  · **`g = GMAX` 로 잘린 창 비율** — 상한이 실제로 작동하는지

그리고 셋째를 더 본다.

  · **조용한 배경음 쿼터가 얼마나 증폭되는가** — b.6 의 핵심 위험이다.
    정규화가 조용한 창의 잡음을 최대 4배 키우므로, 그 창들이 이벤트처럼
    보이면 G7(무인 8시간)이 깨진다. 학습 전에 **증폭 후 int8 분포**를 보고
    위험을 미리 가늠한다 (최종 판정은 학습 후 오탐률이다).

사용 (WSL2):
    python3 tools/norm_gain_stats.py --root data/processed/safesound
"""

import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# safesound.py 가 ai8x 를 import 하므로 경로를 함께 넣는다 (다른 도구와 같다)
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
WIN = 16384
MARGIN = 1600


def load(root, split, cls, n, rng):
    d = os.path.join(root, split, cls)
    rows = []
    with open(os.path.join(d, "index.csv"), encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        for r in rd:
            rows.append((int(r["shard"]), int(r["row"]), r.get("note", "")))
    take = (rows if len(rows) <= n
            else [rows[i] for i in rng.choice(len(rows), n, replace=False)])
    maps, out = {}, []
    for sh, row, note in take:
        m = maps.get(sh)
        if m is None:
            m = np.load(os.path.join(d, f"shard_{sh:04d}.npy"), mmap_mode="r")
            maps[sh] = m
        out.append((np.asarray(m[row][MARGIN:MARGIN + WIN]), note))
    return out, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/safesound")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=400, help="클래스당 창 수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import safesound as S

    class _N(S.SafeSound):                    # 정규화 메서드만 쓴다
        def __init__(self):
            self.norm_gmax_sh = S.NORM_GMAX_SH
            self.norm_target = S.NORM_TARGET

    nz = _N()
    rng = np.random.default_rng(a.seed)
    gmax = S.NORM_GMAX_SH

    print(f"TARGET={S.NORM_TARGET}  GMAX=x{2**gmax} (+{6*gmax}dB)  "
          f"split={a.split}  클래스당 {a.n}창")
    print()
    print(f"{'클래스':<12}{'n':>6}" + "".join(f"{'x'+str(2**k):>8}" for k in
                                             range(gmax + 1))
          + f"{'상한포화%':>10}{'peak 중앙':>10}{'증폭후 0비율':>13}")
    print("-" * 74)

    tot = {k: 0 for k in range(gmax + 1)}
    for cls in CLASSES:
        ws, _ = load(a.root, a.split, cls, a.n, rng)
        hist = [0] * (gmax + 1)
        peaks, zr = [], []
        for w, _note in ws:
            sh, y = nz._norm_pow2(w)
            hist[sh] += 1
            tot[sh] += 1
            peaks.append(int(np.abs(w.astype(np.int16)).max()))
            zr.append(float((y == 0).mean()))
        n = len(ws)
        print(f"{cls:<12}{n:>6}"
              + "".join(f"{100*h/n:>7.1f}%" for h in hist)
              + f"{100*hist[gmax]/n:>9.1f}%"
              + f"{np.median(peaks):>10.0f}"
              + f"{100*np.mean(zr):>12.1f}%")

    N = sum(tot.values())
    print("-" * 74)
    print(f"{'전체':<12}{N:>6}"
          + "".join(f"{100*tot[k]/N:>7.1f}%" for k in range(gmax + 1)))
    print()
    print("읽는 법")
    print(f"  · 상한(x{2**gmax})에 붙은 비율이 높으면 TARGET 이 너무 높거나")
    print("    창들이 전반적으로 조용하다는 뜻이다. 그 창들은 정규화가")
    print("    **목표에 도달하지 못한 채** 잡음만 키운 것이다")
    print("  · x1 비율이 높으면 정규화가 거의 아무 일도 안 한 것이다 —")
    print("    그러면 이 실험 자체가 무의미해진다")
    print("  · `증폭후 0비율` 은 시프트라 **원본과 같아야 한다** (시프트는 0을")
    print("    0으로 보낸다). 다르면 구현이 잘못됐다")
    print()
    print("⚠️ 이 표는 **위험을 미리 가늠**하는 것이고 판정이 아니다.")
    print("   최종 판정은 학습 후 `eval_threshold.py` 의 조용한 배경음 쿼터")
    print("   자체 오탐률이다 (기준선 1.5배를 넘으면 기각).")


if __name__ == "__main__":
    main()
