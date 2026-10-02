#!/usr/bin/env python3
"""①′ 프레임 정의(`melfeat` framing "inc") KAT — numpy 만.

  1. 상수: hop 250, 끝 정렬 오프셋 122, 쓰는 샘플 16,262
  2. **증분 성질**: 4,000 샘플(판단 주기) 뒤의 창에서, 앞 48프레임이 이전 창의
     뒤 48프레임과 **같은 값**이다 (판단마다 새 프레임 16개만 계산하면 된다)
  3. 프레임 하나를 손으로 계산한 값과 일치
  4. 기존 ① 경로("log")가 바뀌지 않았다 — 고정 KAT 벡터와 대조

사용 (WSL2):  python3 tools/kat_melinc.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
import melfeat as MF                                         # noqa: E402

FAIL = []


def check(name, ok, info=""):
    print(f"  [{'OK' if ok else '실패'}] {name} {info}")
    if not ok:
        FAIL.append(name)


def main():
    check("오프셋 122, hop 250", MF.OFF_INC == 122 and MF.HOP_INC == 250)
    used = (MF.N_FRAMES - 1) * MF.HOP_INC + MF.N_FFT
    check("쓰는 샘플 16,262 (1초 창의 99.3%)", used == 16262, f"({used})")
    check("판단 주기 4,000 = 16프레임", 4000 % MF.HOP_INC == 0 and 4000 // MF.HOP_INC == 16)

    rng = np.random.default_rng(0)
    t = np.arange(16384 + 3 * 4000)
    s = (40 * np.sin(2 * np.pi * 700 * t / 16000) + rng.normal(0, 12, len(t)))
    s = np.clip(np.round(s), -128, 127).astype(np.int8)
    ok, mx = True, 0
    for k in range(3):
        a = MF.mel_int8(s[4000 * k:4000 * k + 16384], "loginc")
        b = MF.mel_int8(s[4000 * (k + 1):4000 * (k + 1) + 16384], "loginc")
        d = np.abs(a[:, 16:].astype(int) - b[:, :48].astype(int))
        ok &= bool((d == 0).all())
        mx = max(mx, int(d.max()))
    check("증분 성질: 이웃 창이 48프레임을 공유한다 (비트 일치)", ok, f"(최대 차이 {mx})")

    w = s[:16384]
    f = 37
    seg = w[MF.OFF_INC + f * MF.HOP_INC:MF.OFF_INC + f * MF.HOP_INC + MF.N_FFT] / 128.0
    sp = np.fft.rfft(seg * MF.hann())
    mel = MF.mel_filterbank() @ ((sp.real ** 2 + sp.imag ** 2) / MF._REF)  # noqa: SLF001
    ref = MF.db_to_int8(10.0 * np.log10(mel + MF.EPS))
    got = MF.mel_int8(w, "loginc")[:, f]
    check("프레임 37 손계산과 일치", np.array_equal(ref, got),
          f"(최대 차이 {int(np.abs(ref.astype(int) - got.astype(int)).max())})")
    check("마지막 프레임이 창의 마지막 샘플까지 쓴다",
          MF.OFF_INC + 63 * MF.HOP_INC + MF.N_FFT == 16384)

    d = os.path.join(REPO, "tools", "kat_vectors")
    kw = np.load(os.path.join(d, "melkat_window_int8.npy"))
    kr = np.load(os.path.join(d, "melkat_logmel_int8.npy"))
    check("기존 ① 경로 불변 (고정 KAT 벡터)", np.array_equal(MF.mel_int8(kw, "log"), kr))
    check("①′ 는 ① 과 다른 특징이다", not np.array_equal(MF.mel_int8(kw, "loginc"), kr))

    print()
    if FAIL:
        print(f"실패 {len(FAIL)}건")
        sys.exit(1)
    print("전부 통과")


if __name__ == "__main__":
    main()
