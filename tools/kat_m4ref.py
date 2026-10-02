#!/usr/bin/env python3
"""M4 참조 구현(`firmware/common/m4ref.c`) 대조 — PC.

펌웨어와 **같은 소스**를 PC 에서 컴파일해
  1. 합성 KAT 입력 → 보드 NPU 가 낸 로짓(`m4_kat_output`)과 **비트 일치**
  2. 시험셋 창 N개 → `tools/export_m4_weights.py` 의 numpy 기준 계산과 비트 일치
를 본다. 2 는 KAT 한 벡터가 우연히 맞는 경우를 걸러낸다.

사용 (WSL2, ai8x-training venv):
    ~/ai8x-training/venv/bin/python tools/kat_m4ref.py [--n 200]
"""

import argparse
import ctypes
import glob
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import export_m4_weights as E                                # noqa: E402

WIN, MARGIN = 16384, 1600


def build():
    so = os.path.join(tempfile.gettempdir(), "libm4ref_host.so")
    subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-o", so,
                    os.path.join(REPO, "firmware/common/m4ref.c")], check=True)
    return ctypes.CDLL(so)


def c_infer(lib, w):
    w = np.ascontiguousarray(w, dtype=np.int8)
    out = np.zeros(5, dtype=np.int32)
    lib.m4ref_infer(w.ctypes.data_as(ctypes.c_void_p),
                    out.ctypes.data_as(ctypes.c_void_p))
    return out.tolist()


def np_infer(ws, shifts, fc_shift, w):
    # 모델 입력 x[채널 r][시간 c] = sample[c*128 + r]
    x = np.asarray(w, dtype=np.int64).reshape(128, 128).T
    for (_, pk, pad, _), wt, s in zip(E.LAYERS, ws, shifts):
        if pk:
            x = E.pool(x, pk)
        x = np.clip(E.scale(E.conv1d(x, wt, pad), s), 0, 127)
    return ((ws[-1].reshape(5, -1) @ x.reshape(-1)) << fc_shift).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    lib = build()

    hdr = open(os.path.join(REPO, "firmware/common/m4_weights.h"), encoding="utf-8").read()
    fc_shift = int(re.search(r"#define M4_FC_SHIFT (\d+)", hdr).group(1))
    kat_in = np.array(re.search(r"m4_kat_input\[\d+\] = \{(.*?)\};", hdr, re.S)
                      .group(1).replace("\n", "").split(","), dtype=np.int64)
    got = c_infer(lib, kat_in)
    ok1 = got == E.KAT_RAW
    print(f"1. KAT: C {got}\n        보드 NPU {E.KAT_RAW}  → {'일치' if ok1 else '**불일치**'}")

    sd = torch.load(E.CK, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.get("state_dict", sd).items()}
    names = [l[0] for l in E.LAYERS] + ["fc"]
    ws = [sd[f"{n}.op.weight"].numpy().astype(np.int64) for n in names]
    shifts = [int(sd[f"{n}.output_shift"].item()) for n in names]

    rows = []
    for f in sorted(glob.glob(os.path.join(
            REPO, "data/processed/safesound/test/*/shard_*.npy"))):
        arr = np.load(f, mmap_mode="r")
        rows += [(arr, i) for i in range(len(arr))]
    sel = np.random.default_rng(0).permutation(len(rows))[:a.n]
    bad = 0
    for j in sel:
        arr, i = rows[j]
        w = np.asarray(arr[i][MARGIN:MARGIN + WIN])
        bad += c_infer(lib, w) != np_infer(ws, shifts, fc_shift, w)
    print(f"2. 시험셋 {len(sel)}창: C != numpy 기준 {bad}건  → "
          f"{'전부 일치' if bad == 0 else '**불일치**'}")
    sys.exit(0 if ok1 and bad == 0 else 1)


if __name__ == "__main__":
    main()
