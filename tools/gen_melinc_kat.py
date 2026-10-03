#!/usr/bin/env python3
"""증분 로그 멜의 보드 KAT 벡터 — `melinc_kat.h`.

입력 창은 ① 의 KAT 와 **같은** `melkat_window` 를 쓰고, 기대값만 해당 프레임
정의(`melfeat.mel_int8(w, scheme)`)로 낸다. 손으로 고치지 말 것.

사용 (WSL2):
    python3 tools/gen_melinc_kat.py                       # ①′ (loginc)
    python3 tools/gen_melinc_kat.py --scheme log_h500 --out <프로젝트>/melinc_kat.h
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
import melfeat as MF                                         # noqa: E402

OUT = os.path.join(REPO, "firmware", "common", "melinc_kat.h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="loginc")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    w = np.load(os.path.join(REPO, "tools", "kat_vectors", "melkat_window_int8.npy"))
    feat = MF.mel_int8(w, a.scheme)
    ref = feat.reshape(-1)
    body = ",\n".join("    " + ", ".join(f"{int(v):4d}" for v in ref[i:i + 16])
                      for i in range(0, len(ref), 16))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("/* @generated — tools/gen_melinc_kat.py. 손으로 고치지 말 것.\n"
                " * melkat_window (melkat_vectors.h) 에 대한 기대값:\n"
                f" * melfeat.py scheme \"{a.scheme}\" — 모양 {feat.shape} (멜, 프레임),\n"
                " * mel-major (out[m*프레임수 + f]). 허용 오차 +-2 LSB */\n"
                "#ifndef MELINC_KAT_H\n#define MELINC_KAT_H\n#include <stdint.h>\n\n"
                f"static const int8_t melinc_logmel[{len(ref)}] = {{\n{body}\n}};\n\n"
                "#endif\n")
    print(f"저장: {a.out}  ({a.scheme}, {feat.shape})")


if __name__ == "__main__":
    main()
