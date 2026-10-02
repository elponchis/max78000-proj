#!/usr/bin/env python3
""""정확도 대 에너지 곡선" 변형의 KAT — 특징 정의와 모델 모양.

변형 넷 (melfeat.INC_SPECS: h500 / h500m32 / m32 / h400)과 기준 ①′(inc)에 대해
  1. 특징 모양 = (멜, 프레임), 끝 정렬 (마지막 프레임이 창의 마지막 샘플까지)
  2. **증분 성질** — 4,000 샘플 뒤의 창이 (프레임 − 판단당 새 프레임)개를 비트
     단위로 공유한다
  3. 모델(`ai85safesoundmelnet`)이 그 입력으로 만들어지고 forward 가 된다.
     FC 입력 길이·파라미터 수를 찍는다 (442KB 대비)
  4. 기존 구성이 그대로다: ① 64×64 → 157,472 파라미터, 고정 KAT 벡터 일치

사용 (WSL2, ai8x-training venv):
    ~/ai8x-training/venv/bin/python tools/kat_curve.py
"""

import importlib.util
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets")]
import ai8x                                                 # noqa: E402
import melfeat as MF                                         # noqa: E402

SCHEMES = {"loginc": "inc", "log_h500": "h500", "log_h500m32": "h500m32",
           "log_m32": "m32", "log_h400": "h400"}
FAIL = []


def check(name, ok, info=""):
    print(f"  [{'OK' if ok else '실패'}] {name} {info}")
    if not ok:
        FAIL.append(name)


def main():
    ai8x.set_device(85, False, False)
    spec = importlib.util.spec_from_file_location(
        "melnet", os.path.join(REPO, "models", "ai85net-safesound-mel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    t = np.arange(16384 + 4000)
    s = np.clip(np.round(40 * np.sin(2 * np.pi * 900 * t / 16000)
                         + rng.normal(0, 12, len(t))), -128, 127).astype(np.int8)

    print(f"  {'법칙':<13}{'hop':>5}{'프레임':>6}{'멜':>4}{'새 프레임':>9}{'미사용 샘플':>11}"
          f"{'겹침':>6}{'FC 입력':>8}{'파라미터':>10}{'442KB':>7}")
    for scheme, fr in SCHEMES.items():
        hop, nf, nm = MF.INC_SPECS[fr]
        off = MF.inc_offset(fr)
        new = 4000 // hop
        a = MF.mel_int8(s[:16384], scheme)
        b = MF.mel_int8(s[4000:4000 + 16384], scheme)
        m = mod.ai85safesoundmelnet(num_channels=1, dimensions=(nm, nf))
        n_par = sum(p.numel() for n_, p in m.named_parameters() if n_.endswith("op.weight"))
        with torch.no_grad():
            y = m(torch.zeros(2, 1, nm, nf))
        print(f"  {scheme:<13}{hop:>5}{nf:>6}{nm:>4}{new:>9}{off:>11}"
              f"{100 * (MF.N_FFT - hop) / MF.N_FFT:>5.0f}%{m.fc.op.in_features:>8}"
              f"{n_par:>10,}{100 * n_par / 442368:>6.1f}%")
        check(f"{scheme}: 모양 ({nm},{nf})", a.shape == (nm, nf))
        check(f"{scheme}: 판단 주기가 hop 의 배수", 4000 % hop == 0)
        check(f"{scheme}: 끝 정렬", off >= 0 and off + (nf - 1) * hop + MF.N_FFT == 16384)
        check(f"{scheme}: 증분 — {nf - new}프레임 공유 (비트 일치)",
              np.array_equal(a[:, new:], b[:, :nf - new]))
        check(f"{scheme}: forward 출력 (2,5)", tuple(y.shape) == (2, 5))

    m0 = mod.ai85safesoundmelnet(num_channels=1, dimensions=(64, 64))
    n0 = sum(p.numel() for n_, p in m0.named_parameters() if n_.endswith("op.weight"))
    check("기존 ① 모델 파라미터 157,472", n0 == 157472, f"({n0:,})")
    d = os.path.join(REPO, "tools", "kat_vectors")
    kw = np.load(os.path.join(d, "melkat_window_int8.npy"))
    kr = np.load(os.path.join(d, "melkat_logmel_int8.npy"))
    check("기존 ① 특징 불변 (고정 KAT 벡터)", np.array_equal(MF.mel_int8(kw, "log"), kr))

    print()
    if FAIL:
        print(f"실패 {len(FAIL)}건: {FAIL}")
        sys.exit(1)
    print("전부 통과")


if __name__ == "__main__":
    main()
