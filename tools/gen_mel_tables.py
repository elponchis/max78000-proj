#!/usr/bin/env python3
"""`datasets/melfeat.py` 의 상수를 C 헤더로 낸다 — `firmware/common/mel_tables.h`.

펌웨어의 멜 전처리(`firmware/common/melfeat.c`)가 쓰는 표다. **손으로 고치지
말 것** — melfeat.py 를 바꾸면 이 스크립트를 다시 돌린다.

  · Hann 창 (512, float32). int8 → ±1.0 변환(÷128)을 **미리 곱해 둔다**
  · 멜 필터뱅크 — 삼각형이라 대부분 0 이다. 필터마다 (시작 빈, 길이, 가중치)
    만 싣는다 (희소). 257×64 = 16,448 개 중 실제로 쓰는 것만
  · 1/_REF, dB→int8 아핀 계수

사용 (WSL2):  python3 tools/gen_mel_tables.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
import melfeat as MF                                         # noqa: E402

OUT = os.path.join(REPO, "firmware", "common", "mel_tables.h")


def cf(v):
    """C float 리터럴. `%.9g` 가 `0`·`127` 처럼 소수점 없이 나오면 `f` 접미사가
    정수 상수에 붙어 컴파일이 깨진다 — 반드시 소수점을 넣는다."""
    s = f"{float(v):.9g}"
    if "." not in s and "e" not in s and "n" not in s:
        s += ".0"
    return s + "f"


def farr(name, a, per=6):
    a = np.asarray(a, dtype=np.float32)
    body = ",\n".join(
        "    " + ", ".join(cf(v) for v in a[i:i + per])
        for i in range(0, len(a), per))
    return f"static const float {name}[{len(a)}] = {{\n{body}\n}};\n\n"


def iarr(name, a, ctype="uint16_t", per=16):
    body = ",\n".join(
        "    " + ", ".join(str(int(v)) for v in a[i:i + per])
        for i in range(0, len(a), per))
    return f"static const {ctype} {name}[{len(a)}] = {{\n{body}\n}};\n\n"


def main():
    fb = MF._FB                                              # noqa: SLF001
    win = MF._WIN / 128.0                                    # noqa: SLF001
    start, length, weights = [], [], []
    # 꼭대기 필터의 위쪽 끝이 8000 Hz 인데 mel↔Hz 왕복의 반올림으로 나이퀴스트
    # 빈에 1e-13 급 가중치가 남는다. float32 로는 0 이므로 버린다 (버린 최대값을 찍는다).
    TINY = 1e-9
    dropped = float(fb[(fb > 0.0) & (fb <= TINY)].max(initial=0.0))
    print(f"버린 가중치 최대값: {dropped:.3g} (문턱 {TINY:g})")
    for m in range(MF.N_MELS):
        nz = np.nonzero(fb[m] > TINY)[0]
        assert len(nz) and (np.diff(nz) == 1).all(), f"필터 {m} 가 연속 구간이 아니다"
        start.append(nz[0])
        length.append(len(nz))
        weights.extend(fb[m, nz[0]:nz[-1] + 1])
    lo, hi = min(start), max(s + n - 1 for s, n in zip(start, length))
    # DC(0)·나이퀴스트(256) 빈은 어느 필터에도 걸리지 않는다 — C 쪽이 그 두 빈을
    # 계산하지 않아도 되는 근거다 (CMSIS rfft 의 출력 패킹 차이를 피한다).
    assert lo >= 1 and hi <= MF.N_FFT // 2 - 1, (lo, hi)

    k = (10.0 / np.log(10.0)) * (255.0 / MF.SPAN_DB)         # ln → int8 단계
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(
            "/* @generated — tools/gen_mel_tables.py (datasets/melfeat.py 에서).\n"
            " * 손으로 고치지 말 것. melfeat.py 를 바꾸면 다시 생성한다.\n"
            f" * n_fft {MF.N_FFT}, hop {MF.HOP}, pad {MF.PAD}(reflect), "
            f"n_mels {MF.N_MELS}, frames {MF.N_FRAMES},\n"
            f" * {MF.FMIN:g}-{MF.FMAX:g} Hz HTK, TOP_DB {MF.TOP_DB}, "
            f"SPAN_DB {MF.SPAN_DB}. 쓰는 빈 {lo}..{hi}, 가중치 {len(weights)}개 */\n"
            "#ifndef MEL_TABLES_H\n#define MEL_TABLES_H\n#include <stdint.h>\n\n"
            f"#define MEL_WIN {16384}\n#define MEL_N_FFT {MF.N_FFT}\n"
            f"#define MEL_HOP {MF.HOP}\n#define MEL_PAD {MF.PAD}\n"
            f"#define MEL_N_MELS {MF.N_MELS}\n#define MEL_N_FRAMES {MF.N_FRAMES}\n"
            "/* (1)' 증분 프레임 정의 - hop 250, 패딩 없음, 끝 정렬 */\n"
            f"#define MEL_HOP_INC {MF.HOP_INC}\n#define MEL_OFF_INC {MF.OFF_INC}\n"
            f"#define MEL_INC_STEP 4000   /* 판단 주기 (샘플) */\n"
            f"#define MEL_INC_FRAMES {4000 // MF.HOP_INC}  /* 판단당 새 프레임 */\n"
            f"#define MEL_INC_TAIL {MF.N_FFT - MF.HOP_INC}   /* 이어 붙일 직전 샘플 */\n"
            f"#define MEL_INV_REF {cf(1.0 / MF._REF)}\n"          # noqa: SLF001
            f"#define MEL_EPS {cf(MF.EPS)}\n"
            f"/* q = round((10*log10(p) - TOP_DB) * 255/SPAN_DB) + 127\n"
            f" *   = round(MEL_K_LN * ln(p) + MEL_Q_OFF) */\n"
            f"#define MEL_K_LN {cf(k)}\n"
            f"#define MEL_Q_OFF {cf(127.0 - MF.TOP_DB * 255.0 / MF.SPAN_DB)}\n\n")
        f.write("/* 주기형 Hann x (1/128) — int8 을 곱하면 바로 +-1.0 눈금 */\n")
        f.write(farr("mel_win", win))
        f.write(iarr("mel_fb_start", start))
        f.write(iarr("mel_fb_len", length, "uint8_t"))
        f.write(farr("mel_fb_w", weights))
        f.write("#endif\n")
    print(f"저장: {OUT}  (빈 {lo}..{hi}, 가중치 {len(weights)}개)")


if __name__ == "__main__":
    main()
