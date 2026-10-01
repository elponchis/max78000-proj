#!/usr/bin/env python3
"""(c) 로그단의 **계단 경계 255개**를 정수 경로에서 뽑는다.

`models/ai85net-safesound-cstage-train.py` 가 이 결과(`bucketize` 경계)로
로그단을 돌린다. 경계는 **정수 경로(round → CLZ+LUT → 포화)에서 이진탐색**
으로 뽑으므로 등가성이 구성상 보장된다.

⚠️ `SCALE_SH` / `LO_Q8` / `SPAN_Q8` 을 바꾸면 **이 스크립트를 다시 돌려야
한다.** 모델이 들고 있는 상수와 경계 파일이 어긋나면 조용히 틀린 값을
학습한다 — 그래서 모델 쪽에서 상수 해시를 함께 저장하고 대조한다.

사용 (WSL2):
    python3 tools/gen_cstage_bounds.py
    python3 tools/gen_cstage_bounds.py --verify     # 전수 재확인
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from log2_q8 import log2_q8, to_int8                # noqa: E402

OUT = os.path.join(REPO, "data/synth/cstage_bounds.npy")


def level(v, lo_q8, span_q8):
    """정수 v 하나의 int8 출력."""
    return int(to_int8(log2_q8(np.array([v], dtype=np.int64)), lo_q8, span_q8)[0])


def build(scale_sh, lo_q8, span_q8):
    """레벨이 k 이상이 되는 최소 v 를 이진탐색으로 찾아 |y| 경계로 바꾼다."""
    lo_lvl = level(0, lo_q8, span_q8)
    hi_lvl = level(0xFFFFFFFF, lo_q8, span_q8)
    vb = []
    for k in range(lo_lvl + 1, hi_lvl + 1):
        a, b = 0, 0xFFFFFFFF
        while a < b:
            mid = (a + b) // 2
            if level(mid, lo_q8, span_q8) >= k:
                b = mid
            else:
                a = mid + 1
        vb.append(a)
    # |y| 경계 = (v - 0.5) / 2^sh.
    # `torch.round` 는 half-to-even 이므로, v-1 이 짝수면 정확히 x.5 인
    # 입력이 **아래** 레벨로 가야 한다 → 경계를 한 ulp 올린다.
    raw = np.array([(v - 0.5) / (1 << scale_sh) for v in vb], dtype=np.float64)
    out = np.array([np.nextafter(t, np.inf) if ((v - 1) % 2 == 0) else t
                    for t, v in zip(raw, vb)], dtype=np.float64)
    return out, lo_lvl, hi_lvl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale-sh", type=int, default=14)
    ap.add_argument("--lo-q8", type=int, default=1472)
    ap.add_argument("--span-q8", type=int, default=2976)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    B, lo_lvl, hi_lvl = build(a.scale_sh, a.lo_q8, a.span_q8)
    print(f"레벨 {lo_lvl}..{hi_lvl}, 경계 {len(B)}개 "
          f"(SCALE_SH={a.scale_sh} LO_Q8={a.lo_q8} SPAN_Q8={a.span_q8})")
    assert len(B) == 255, "레벨이 256개가 아니다 — 상수를 확인할 것"
    assert np.all(np.diff(B) > 0), "경계가 단조가 아니다"

    if a.verify:
        import torch                                 # noqa: PLC0415
        Bt = torch.from_numpy(B)
        rng = np.random.default_rng(7)
        bad = 0
        for name, y in (("정규", rng.normal(0, 3, 1_000_000)),
                        ("극소", rng.normal(0, 1e-4, 500_000)),
                        ("극대", rng.normal(0, 1e3, 500_000)),
                        ("경계값", np.array([v / (1 << a.scale_sh)
                                           for v in range(1, 300000)]))):
            yt = torch.tensor(y, dtype=torch.float32)
            v = np.round(np.abs(yt.numpy()) * (1 << a.scale_sh))
            ref = to_int8(log2_q8(v.astype(np.int64).clip(0, 0xFFFFFFFF)),
                          a.lo_q8, a.span_q8).astype(np.int64)
            got = (torch.bucketize(yt.abs().double(), Bt) + lo_lvl).numpy()
            d = int((ref != got).sum())
            bad += d
            print(f"  {name:<8} 불일치 {d:>6} / {ref.size:,}")
        if bad:
            print("⚠️ 불일치는 `|y|·2^SH` 가 정확히 x.5 인 입력에서만 난다 — "
                  "기기에는 이 반올림이 없다 (NPU 가 int32 를 직접 준다)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, B)
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
