#!/usr/bin/env python3
"""모든 모델 진입점을 **train.py 가 부르는 방식 그대로** 생성해 본다.

`train.py:792-800` 이 model_args 를 이렇게 만든다:
    pretrained / num_classes / num_channels=dimensions[0] /
    dimensions=(dimensions[1], dimensions[2]) / bias=args.use_bias /
    weight_bits / bias_bits / quantize_activation

진입점이 이 중 하나를 **하드코딩**하면 `got multiple values for keyword argument`
로 죽는다. 실제로 `bias`(무력화)와 `dimensions`(크래시) 두 번 당했으므로,
데이터셋의 `input` 과 짝지어 전수 확인한다.
"""
import importlib.util
import os
import sys

AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
for _i, _a in enumerate(sys.argv):
    if _a == "--ai8x" and _i + 1 < len(sys.argv):
        AI8X = sys.argv[_i + 1]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets")]

import ai8x  # noqa: E402

ai8x.set_device(85, False, False)


def load(fname):
    spec = importlib.util.spec_from_file_location(
        "m_" + fname.replace("-", "_").replace(".py", ""),
        os.path.join(REPO, "models", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# (모델 파일, 진입점, 그 모델을 쓰는 데이터셋의 input)
CASES = [
    ("ai85net-safesound.py", "ai85safesoundnet", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_bias", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_fb", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_fb_relu", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_fb_k2", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_fb_k4", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_w025", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_w050", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_w150", (128, 128)),
    ("ai85net-safesound-mel.py", "ai85safesoundmelnet", (1, 64, 64)),
    ("ai85net-safesound-mel.py", "ai85safesoundmelnet_w050", (1, 64, 64)),
    ("ai85net-safesound-mel.py", "ai85safesoundmelnet_w150", (1, 64, 64)),
    ("ai85net-safesound-mel.py", "ai85safesoundwave2dnet", (1, 128, 128)),
    ("ai85net-safesound-mel.py", "ai85safesoundwave2dfoldnet", (4, 64, 64)),
    ("ai85net-safesound-mel.py", "ai85safesoundmelteacher", (128, 128)),
]

print(f"{'진입점':<30}{'입력':<16}{'params':>10}  결과")
print("-" * 72)
bad = []
for fname, ep, dims in CASES:
    m = load(fname)
    registered = {d["name"] for d in getattr(m, "models", [])}
    if ep not in registered:
        bad.append(f"{ep}: models 리스트 미등록 (train.py --model 선택지에 없다)")
        print(f"{ep:<30}{str(dims):<16}{'—':>10}  ✘ models 미등록")
        continue
    fn = getattr(m, ep, None)
    if fn is None:
        bad.append(f"{ep}: 진입점 없음")
        print(f"{ep:<30}{str(dims):<16}{'—':>10}  ✘ 진입점 없음")
        continue
    # train.py 와 **똑같이** 만든다
    if len(dims) == 2:                      # 1D: input 이 (채널, 길이)
        num_channels, dimensions = dims[0], (dims[1], 1)
    else:                                   # 2D: (채널, 높이, 너비)
        num_channels, dimensions = dims[0], (dims[1], dims[2])
    args = dict(pretrained=False, num_classes=5, num_channels=num_channels,
                dimensions=dimensions, bias=False,
                weight_bits=None, bias_bits=None, quantize_activation=False)
    try:
        net = fn(**args)
        n = sum(p.numel() for p in net.parameters())
        print(f"{ep:<30}{str(dims):<16}{n:>10,}  ✔")
    except Exception as e:                  # noqa: BLE001
        bad.append(f"{ep}: {type(e).__name__}: {e}")
        print(f"{ep:<30}{str(dims):<16}{'—':>10}  ✘ {type(e).__name__}")
        print(f"    {str(e)[:110]}")

print()
if bad:
    print(f"실패 {len(bad)}건:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("전부 통과 — train.py 인자 규약과 충돌하는 진입점 없음")
