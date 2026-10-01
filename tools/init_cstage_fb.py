#!/usr/bin/env python3
"""(c) 의 멜 필터뱅크 초기화 체크포인트를 만들고, **로그단 창을 정한다.**

왜 필요한가: **chain16 의 (c) 5시드는 초기화를 받지 못했다** (2026-10-01).
D-1 은 `data/fb-init.pth.tar`, (a) 는 `data/logstage-init-fb.pth.tar` 를
`--exp-load-weights-from` 으로 받았는데 `run_chain16.sh` 에는 그 인자가
없었다. 실측 결과 그 차이가 결정적이다 — conv1 과 멜 필터뱅크의 코사인
유사도가

| 구성 | 초기화 | 같은 자리 cos |
|---|---|---:|
| D-1 | fb-init | **0.977** |
| (a) | logstage-fb | **0.977** |
| **(c)** | **없음** | **0.07** (무작위 기준선 0.063) |

→ **D-1·(a) 의 conv1 은 초기값에서 거의 움직이지 않는다.** 즉 그 두
구성에서 conv1 은 사실상 **손으로 설계한 필터뱅크**이고, (c) 는 그것을
스스로 배워야 했는데 **배우지 못했다.**

⚠️ 그래서 창 폭 오설정(1.58배)도 **독립적인 결함이 아니라 이것의 결과**다.
창을 70 dB 로 잡은 근거는 *필터뱅크로 초기화된* conv1 의 출력 분포
(p1~p99 = 71.8 dB) 였고, (c) 의 conv1 이 무작위라 분포가 44 dB 로
좁았던 것이다.

사용 (WSL2):
    ~/ai8x-training/venv/bin/python tools/init_cstage_fb.py
    ~/ai8x-training/venv/bin/python tools/init_cstage_fb.py --window-only
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "datasets"))
from init_filterbank import OUT_CH, TAPS, filterbank_weights, to_conv_weight  # noqa: E402

OUT = os.path.join(REPO, "data", "cstage-init-fb.pth.tar")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--data", default=os.path.expanduser(
        "~/ai8x-training/data/SafeSound"))
    ap.add_argument("--n", type=int, default=400,
                    help="창 결정에 쓸 **학습셋** 표본 수")
    ap.add_argument("--window-only", action="store_true",
                    help="체크포인트를 쓰지 않고 창만 계산한다")
    a = ap.parse_args()

    sys.path.insert(0, a.ai8x)
    import torch                                    # noqa: PLC0415

    import ai8x                                     # noqa: PLC0415
    ai8x.set_device(85, False, False)

    m = load(os.path.join(REPO, "models/ai85net-safesound-cstage-train.py"), "ct")
    net = m.ai85safesoundcstage(num_channels=128)

    # ── conv1 을 멜 필터뱅크로 ────────────────────────────────────
    W, desc, centers = filterbank_weights(OUT_CH // 2, 125.0, 7000.0)
    Wc = to_conv_weight(W, 1)                        # (100, 128, 1)
    sd = net.state_dict()
    key = "voice_conv1.op.weight"
    assert tuple(sd[key].shape) == Wc.shape, (sd[key].shape, Wc.shape)
    sd[key] = torch.from_numpy(Wc.copy())
    net.load_state_dict(sd, strict=True)
    print(f"conv1 ← 멜 필터뱅크 {Wc.shape}, 밴드 {len(centers)}개 "
          f"{centers[0]:.0f}~{centers[-1]:.0f} Hz")
    print(f"  (D-1·(a) 와 **같은** 초기값이다 — 그래야 로그단만 비교가 된다)")

    # ── 창 결정: **학습셋**에서 conv1 출력 분포를 본다 ─────────────
    import safesound as S                            # noqa: PLC0415
    ds = S.SafeSound(a.data, d_type='train')
    idx = np.random.default_rng(0).choice(len(ds), min(a.n, len(ds)),
                                          replace=False)
    xs = torch.stack([ds[int(i)][0] for i in idx])
    net.eval()
    with torch.no_grad():
        y = net.voice_conv1(xs).numpy()
    v = np.abs(y).ravel()
    nz = v[v > 0]
    sh = m.SCALE_SH
    lg = np.log2(np.maximum(np.round(nz * (1 << sh)), 1.0))
    p1, p50, p99 = (np.percentile(lg, q) for q in (1, 50, 99))
    print(f"\n학습셋 {len(idx)}창에서 conv1 출력 (필터뱅크 초기값):")
    print(f"  |y| 0 비율 {100*(v == 0).mean():.2f}%")
    print(f"  log2 p1 {p1:.2f} / p50 {p50:.2f} / p99 {p99:.2f}")
    span = p99 - p1
    print(f"  p1~p99 = {span:.2f} log2 = {span*6.0206:.1f} dB")

    # 창은 p1~p99 를 **꽉 채우도록** 잡는다 (255 레벨이 그 폭을 덮는다)
    span_q8 = int(round(span * 256))
    lo_q8 = int(round(p1 * 256))
    print(f"\n권고 창:  SPAN_Q8 = {span_q8}  ({span:.2f} log2 = "
          f"{span*6.0206:.1f} dB)")
    print(f"          LO_Q8   = {lo_q8}")
    print(f"  현재 모델 상수: SPAN_Q8 = {m.SPAN_Q8} "
          f"({m.SPAN_Q8/256*6.0206:.1f} dB), LO_Q8 = {m.LO_Q8}")
    ratio = (m.SPAN_Q8 / 256) / span
    print(f"  현재 창은 이 분포의 {ratio:.2f}배")
    if abs(ratio - 1.0) < 0.15:
        print("  → 현재 값을 **그대로 둔다** (15% 안)")
    else:
        print("  → 바꿀 것. 모델 상수와 tools/gen_cstage_bounds.py 인자를 함께.")

    if a.window_only:
        return 0

    torch.save({"state_dict": net.state_dict(),
                "arch": "ai85safesoundcstage", "epoch": 0,
                "extras": {"best_top1": 0.0}}, a.out)
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
