#!/usr/bin/env python3
""""곡선" 변형의 **합성 가능 여부와 NPU 사이클**을 학습 전에 확인한다.

NPU 사이클 수와 메모리는 **가중치 값과 무관하고 층 모양에만** 달려 있다. 그래서
무작위 int8 가중치로 만든 가짜 체크포인트를 `ai8xize.py` 에 넣어도 사이클·메모리
수치는 학습한 모델과 같다 (KAT 출력값만 의미가 없다).

입력 모양별로 체크포인트·샘플·yaml 을 만든다. 합성은
`data/logs-local/synth_curve_check.sh` 가 한다.

사용 (WSL2, ai8x-training venv):
    ~/ai8x-training/venv/bin/python tools/curve_synth_check.py
"""

import os

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.expanduser("~/ai8x-synthesis")
OUT = os.path.join(REPO, "data", "synth")
SHAPES = {"h500": (64, 32), "h500m32": (32, 32), "m32": (32, 64), "h400": (64, 40)}
# (이름, out_ch, in_ch, 커널) — models/ai85net-safesound-mel.py 1× 폭
CONVS = [("conv1", 32, 1, 3), ("conv2", 32, 32, 3), ("conv3", 64, 32, 3),
         ("conv4", 64, 64, 3), ("conv5", 128, 64, 3), ("conv6", 128, 128, 1)]


def flat_len(h, w):
    for _ in range(5):                       # conv2..conv6 의 2×2 풀링
        h, w = h // 2, w // 2
    return 128 * h * w


def main():
    rng = np.random.default_rng(0)
    base = open(os.path.join(REPO, "synthesis", "safesound-mel-hwc.yaml"),
                encoding="utf-8").read()
    for name, (nm, nf) in SHAPES.items():
        sd = {}
        for lname, co, ci, k in CONVS:
            sd[f"{lname}.output_shift"] = torch.tensor([0.0])
            sd[f"{lname}.weight_bits"] = torch.tensor([8])
            sd[f"{lname}.bias_bits"] = torch.tensor([8])
            sd[f"{lname}.quantize_activation"] = torch.tensor([1.0])
            sd[f"{lname}.adjust_output_shift"] = torch.tensor([0.0])
            sd[f"{lname}.shift_quantile"] = torch.tensor([1.0])
            sd[f"{lname}.op.weight"] = torch.from_numpy(
                rng.integers(-20, 21, (co, ci, k, k)).astype(np.float32))
        fl = flat_len(nm, nf)
        for key, val in (("output_shift", torch.tensor([0.0])),
                         ("weight_bits", torch.tensor([8])),
                         ("bias_bits", torch.tensor([8])),
                         ("quantize_activation", torch.tensor([1.0])),
                         ("adjust_output_shift", torch.tensor([0.0])),
                         ("shift_quantile", torch.tensor([1.0]))):
            sd[f"fc.{key}"] = val
        sd["fc.op.weight"] = torch.from_numpy(
            rng.integers(-20, 21, (5, fl)).astype(np.float32))
        torch.save({"epoch": 0, "state_dict": sd, "arch": "ai85safesoundmelnet"},
                   os.path.join(OUT, f"curve-{name}-fake.pth.tar"))
        ds = f"CurveCheck{name}"
        np.save(os.path.join(SYN, "tests", f"sample_{ds.lower()}.npy"),
                rng.integers(-128, 128, (1, nm, nf)).astype(np.int64))
        with open(os.path.join(OUT, f"curve-{name}.yaml"), "w", encoding="utf-8") as f:
            f.write(base.replace("dataset: SafeSoundMel", f"dataset: {ds}"))
        n = sum(v.numel() for k, v in sd.items() if k.endswith("op.weight"))
        print(f"{name}: 입력 (1,{nm},{nf})  FC 입력 {fl}  가중치 {n:,}")


if __name__ == "__main__":
    main()
