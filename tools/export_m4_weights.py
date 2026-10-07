#!/usr/bin/env python3
"""④의 양자화 가중치를 M4 소프트웨어 추론용 C 헤더로 낸다 + numpy 기준 계산.

M4 대조군은 **NPU 와 같은 가중치·같은 정수 규약**으로 추론한다 (새로 학습하지
않는다 — CLAUDE.md 6장). 규약은 `ai8x-synthesis/izer/simulate.py` 그대로다:

    conv:  out = clip( floor(0.5 + acc * 2^shift / 128), -128, 127 )   (ReLU 면 0..127)
    max pool 2 / avg pool 2 (0 쪽으로 버림)
    마지막 FC (output_width 32): 스케일·포화 없이 누산값 그대로

이 스크립트는
  1. `data/synth/safesound-wave-q8.pth.tar` 에서 int8 가중치와 층별 shift 를 읽어
     `firmware/common/m4_weights.h` 로 쓴다
  2. 같은 규약의 numpy 기준 계산으로 합성 KAT 입력을 추론해 **보드 NPU 가 낸
     값과 같은지** 본다 (KAT_RAW — `docs/results/synthesis-check.md` 8절)
  3. 그 입력·기대 출력을 헤더에 함께 넣는다 (보드 KAT 용)

사용 (WSL2, ai8x-training venv — torch 가 필요하다):
    ~/ai8x-training/venv/bin/python tools/export_m4_weights.py            # ④ v1 (기본값)
    ~/ai8x-training/venv/bin/python tools/export_m4_weights.py \
        --ck data/synth/safesound-v21-wave-q8.pth.tar \
        --sample data/synth/sample_safesound-v21-wave.npy \
        --kat -1 -2 3 -4 5          # 보드 NPU 가 같은 샘플에 낸 raw 로짓 5개 (시리얼 "class i: raw")
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(REPO, "data/synth/safesound-wave-q8.pth.tar")
SAMPLE = os.path.join(REPO, "data/synth/sample_safesound-wave.npy")
OUT = os.path.join(REPO, "firmware/common/m4_weights.h")

# (이름, 풀링, 패딩, 활성화) — models/ai85net-safesound.py 와 합성 yaml 순서
LAYERS = [
    ("voice_conv1", None, 0, True),
    ("voice_conv2", None, 0, True),
    ("voice_conv3", "max", 1, True),
    ("voice_conv4", None, 0, True),
    ("kws_conv1", "max", 1, True),
    ("kws_conv2", None, 0, True),
    ("kws_conv3", "avg", 1, True),
    ("kws_conv4", "max", 1, True),
]
# 보드 NPU 가 합성 KAT 입력에 낸 값 (POR 직후 시리얼, 2026-10-02)
KAT_RAW = [-259664, -248192, 89872, -34416, 31632]


def scale(acc, shift):
    d = 1 << (7 - shift)                       # 128 / 2^shift, shift <= 0
    return np.clip((acc + d // 2) >> (7 - shift), -128, 127)


def pool(x, kind):
    n = x.shape[1] // 2
    a, b = x[:, 0:2 * n:2], x[:, 1:2 * n:2]
    if kind == "max":
        return np.maximum(a, b)
    s = a + b
    return np.where(s < 0, -((-s) // 2), s // 2)   # 0 쪽으로 버림


def conv1d(x, w, pad):
    co, ci, k = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad)))
    n = x.shape[1] - k + 1
    acc = np.zeros((co, n), dtype=np.int64)
    for j in range(k):
        acc += w[:, :, j].astype(np.int64) @ x[:, j:j + n].astype(np.int64)
    return acc


def main():
    global CK, SAMPLE, OUT, KAT_RAW
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default=CK, help="④ 양자화 체크포인트 (quantize.py 출력)")
    ap.add_argument("--sample", default=SAMPLE, help="합성 샘플 입력 .npy")
    ap.add_argument("--kat", type=int, nargs=5, default=KAT_RAW,
                    help="보드 NPU 가 그 샘플에 낸 raw 로짓 5개 (기본값은 ④ v1, 2026-10-02)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    CK, SAMPLE, KAT_RAW, OUT = a.ck, a.sample, a.kat, a.out
    print("체크포인트:", CK, "\n샘플:", SAMPLE)
    sd = torch.load(CK, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    ws, shifts = [], []
    for name, _, _, _ in LAYERS + [("fc", None, 0, False)]:
        w = sd[f"{name}.op.weight"].numpy()
        assert np.array_equal(w, np.round(w)) and np.abs(w).max() <= 128, name
        ws.append(w.astype(np.int64))
        shifts.append(int(sd[f"{name}.output_shift"].item()))
    print("shift:", shifts)

    x = np.load(SAMPLE).astype(np.int64)
    x = x.reshape(128, 128)                          # (채널, 길이)
    x_in = x.copy()
    for (name, pk, pad, _), w, s in zip(LAYERS, ws, shifts):
        if pk:
            x = pool(x, pk)
        x = np.clip(scale(conv1d(x, w, pad), s), 0, 127)
        print(f"  {name:<12} -> {x.shape}")
    acc = ws[-1].reshape(5, -1) @ x.reshape(-1)       # flatten: 채널 우선 (C x L)
    print("FC 누산값 :", acc.tolist())
    print("보드 NPU  :", KAT_RAW)
    ratio = [k / a for k, a in zip(KAT_RAW, acc.tolist()) if a]
    print("비        :", [round(r, 4) for r in ratio])
    fc_shift = None
    for sh in range(0, 12):
        if (acc << sh).tolist() == KAT_RAW:
            fc_shift = sh
    if fc_shift is None:
        sys.exit("[실패] numpy 기준 계산이 보드 NPU 출력과 맞지 않는다 — 헤더를 쓰지 않는다")
    print(f"→ 일치: 보드 값 = 누산값 << {fc_shift}")

    def carr(name, a, ctype="int8_t", per=24):
        a = np.asarray(a).reshape(-1)
        body = ",\n".join("    " + ", ".join(str(int(v)) for v in a[i:i + per])
                          for i in range(0, len(a), per))
        return f"static const {ctype} {name}[{len(a)}] = {{\n{body}\n}};\n\n"

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("/* @generated — tools/export_m4_weights.py. 손으로 고치지 말 것.\n"
                f" * ④ ({os.path.basename(CK)}) 의 int8 가중치 — NPU 에 올린 것과 같은 값.\n"
                " * 가중치 배치: w[co][k][ci] (입력 채널이 가장 안쪽 — 내적이 연속 메모리) */\n"
                "#ifndef M4_WEIGHTS_H\n#define M4_WEIGHTS_H\n#include <stdint.h>\n\n"
                f"#define M4_FC_SHIFT {fc_shift}\n\n")
        for (name, _, _, _), w in zip(LAYERS, ws[:-1]):
            f.write(carr(f"w_{name}", np.transpose(w, (0, 2, 1))))
        f.write(carr("w_fc", ws[-1].reshape(5, -1)))
        f.write(carr("m4_shift", shifts, "int8_t"))
        # C 쪽 활성값 배치는 [시간][채널] 이다 (내적이 연속 메모리). ④의 입력은
        # x[채널 r][시간 c] = sample[c*128 + r] 이므로 그 배치가 곧 **원본 파형
        # 순서**다 — 입력 적재가 memcpy 하나로 끝난다.
        f.write("/* 합성 KAT 입력 — [시간][채널] 순서 (= 1초 int8 파형 그대로) 와\n"
                " * 보드 NPU 가 그 입력에 낸 출력 */\n")
        f.write(carr("m4_kat_input", x_in.T))
        f.write(carr("m4_kat_output", KAT_RAW, "int32_t", 5))
        f.write("#endif\n")
    print(f"저장: {OUT} ({sum(w.size for w in ws):,} 가중치)")


if __name__ == "__main__":
    main()
