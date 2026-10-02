#!/usr/bin/env python3
"""(c) 로그단 출력의 **창 채움률** — int8 로그값이 70dB 창을 어떻게 쓰는가.

`docs/STATUS.md` 5.6 의 "학습 후 확인할 것". 로그단(`_IntLog`) 출력 int8 의
  · 바닥(−128)·천장(+127) 포화 비율
  · p1~p99 가 255 단계 중 차지하는 비율 (채움률)
  · 평균·표준편차
를 체크포인트별로 낸다. 판정은 "높을수록 좋다" 가 아니라 **① 과의 비교**다
(① 로그 멜: 바닥 0.00% / 천장 0.01% / 채움 37% / 평균 26.9 / sd 24.8).

사용 (WSL2, ai8x venv):
    ~/ai8x-training/venv/bin/python tools/cstage_fill.py \\
        --run "(c) s1:<ckpt>" --run "(c)' s1:<ckpt>" [--split train --step 4]
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets"), os.path.join(REPO, "tools")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]


def measure(ck, data, split, step, batch=128):
    import torch
    import ai8x
    import eval_confusion as EC

    with contextlib.redirect_stdout(io.StringIO()):
        model = EC.load_model(ck, len(CLASSES), AI8X, False, False,
                              config="wave_cstage")
        ds = EC.make_dataset("wave_cstage", data, split,
                             ai8x.normalize(args=argparse.Namespace(
                                 act_mode_8bit=False)))
    # forward 가 모듈 전역 `_IntLog` 를 부른다 — 그 이름을 감싸 출력을 가로챈다
    g = model.forward.__globals__
    orig = g["_IntLog"]
    hist = np.zeros(256, dtype=np.int64)

    class Tap:
        @staticmethod
        def apply(y):
            out = orig.apply(y)
            q = out.detach().round().clamp(-128, 127).to(torch.int64) + 128
            hist[:] += np.bincount(q.flatten().numpy(), minlength=256)
            return out

    g["_IntLog"] = Tap
    try:
        idx = list(range(0, len(ds), step))
        with torch.no_grad():
            for b in range(0, len(idx), batch):
                model(torch.stack([ds[j][0] for j in idx[b:b + batch]]))
    finally:
        g["_IntLog"] = orig

    return _stats(hist, len(idx))


def measure_mel(data, split, step):
    """기준 — ① 로그 멜 **입력** int8 의 같은 통계 (같은 창 표본)."""
    import ai8x
    import eval_confusion as EC
    with contextlib.redirect_stdout(io.StringIO()):
        ds = EC.make_dataset("mel", data, split, ai8x.normalize(
            args=argparse.Namespace(act_mode_8bit=True)))
    hist = np.zeros(256, dtype=np.int64)
    idx = list(range(0, len(ds), step))
    for j in idx:
        q = ds[j][0].round().clamp(-128, 127).numpy().astype(np.int64) + 128
        hist += np.bincount(q.flatten(), minlength=256)
    return _stats(hist, len(idx))


def _stats(hist, n_win):
    n = hist.sum()
    v = np.arange(-128, 128)
    cdf = np.cumsum(hist) / n
    p1 = int(v[np.searchsorted(cdf, 0.01)])
    p99 = int(v[np.searchsorted(cdf, 0.99)])
    mean = float((hist * v).sum() / n)
    sd = float(np.sqrt((hist * (v - mean) ** 2).sum() / n))
    return {"floor": hist[0] / n, "ceil": hist[255] / n,
            "fill": (p99 - p1) / 255.0, "p1": p1, "p99": p99,
            "mean": mean, "sd": sd, "n_win": n_win}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="이름:체크포인트")
    ap.add_argument("--data", default=os.path.join(AI8X, "data"))
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--step", type=int, default=4, help="N창마다 하나 (무증강)")
    a = ap.parse_args()

    print(f"로그단 출력 int8 분포 — {a.split} 셋, {a.step}창마다 1개, 무증강")
    print(f"{'구성':<14}{'창':>6}{'바닥 -128':>11}{'천장 +127':>11}"
          f"{'p1':>6}{'p99':>6}{'채움':>7}{'평균':>8}{'sd':>7}")
    print("-" * 76)
    for spec in ["① 로그 멜:"] + a.run:
        name, ck = spec.split(":", 1)
        if not ck:
            r = measure_mel(a.data, a.split, a.step)
        else:
            r = measure(os.path.join(REPO, ck) if not os.path.isabs(ck) else ck,
                        a.data, a.split, a.step)
        print(f"{name:<14}{r['n_win']:>6}{100*r['floor']:>10.2f}%"
              f"{100*r['ceil']:>10.2f}%{r['p1']:>6}{r['p99']:>6}"
              f"{100*r['fill']:>6.0f}%{r['mean']:>8.1f}{r['sd']:>7.1f}")
    print()
    print("첫 행(① 로그 멜 입력)이 기준이다 — 같은 창 표본에서 잰다.")
    print("⚠️ 채움률이 높은 것이 좋은 것이 아니다 (① 이 가장 낮고 가장 좋다).")


if __name__ == "__main__":
    main()
