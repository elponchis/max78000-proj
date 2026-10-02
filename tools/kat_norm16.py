#!/usr/bin/env python3
"""`norm16` KAT — int8 변환 전 음량 정규화의 정의 대조와 로더 배선 확인.

  1. numpy 구현 == **C 의미론을 그대로 옮긴 순수 파이썬 참조** (int32 산술,
     산술 시프트, 포화). 펌웨어는 이 참조와 같은 식으로 짠다
  2. 경계: 피크 0 / 1 / TARGET 근처 / int8 포화점(8,128) / int16 끝(±32,768)
  3. `x * G` 가 int32 를 넘지 않는다
  4. 로더: train/test 한 샘플씩 꺼내 shape·범위 확인, 시험셋 출력이 `norm16` 과 일치

사용 (WSL2):
    ~/ai8x-training/venv/bin/python tools/kat_norm16.py
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets")]
import ai8x                                                 # noqa: E402
import safesound as S                                       # noqa: E402

FAIL = []


def check(name, ok, info=""):
    print(f"  [{'OK' if ok else '실패'}] {name} {info}")
    if not ok:
        FAIL.append(name)


def ref_norm16(x, gmax):
    """C 참조. 펌웨어 구현의 기준이다.

        int32_t peak = 0;  for (...) { int32_t a = x[n] < 0 ? -x[n] : x[n]; if (a > peak) peak = a; }
        int32_t g = (96 << 16) / (peak > 0 ? peak : 1);
        if (g > gmax << 10) g = gmax << 10;   if (g < 1) g = 1;
        y[n] = __SSAT((x[n] * g + 32768) >> 16, 8);
    """
    peak = max(abs(int(v)) for v in x)
    g = (96 << 16) // max(peak, 1)
    g = max(1, min(g, gmax << 10))
    out, mx = [], 0
    for v in x:
        p = int(v) * g
        mx = max(mx, abs(p) + 32768)
        y = (p + 32768) >> 16            # 파이썬 >> 는 산술 시프트 (C 의 int32 와 같다)
        out.append(max(-128, min(127, y)))
    return g, out, mx


def main():
    rng = np.random.default_rng(0)
    print("1~3. numpy 대 C 참조")
    cases = {
        "전부 0": np.zeros(64, np.int16),
        "피크 1": np.array([0, 1, -1, 0] * 16, np.int16),
        "피크 95/96/97": np.array([95, -96, 97, 3] * 16, np.int16),
        "int8 포화점 8128": np.array([8128, -8128, 4064, 64] * 16, np.int16),
        "int16 끝": np.array([32767, -32768, 12345, -1] * 16, np.int16),
    }
    for k in range(40):
        amp = int(10 ** rng.uniform(0, 4.5))
        cases[f"무작위{k} (진폭 {amp})"] = rng.integers(
            -min(amp, 32768), min(amp, 32767) + 1, 512).astype(np.int16)
    ok_all, mx_all = True, 0
    for name, x in cases.items():
        for gmax in (4, 64):
            g, y = S.norm16(x, gmax)
            rg, ry, mx = ref_norm16(x.tolist(), gmax)
            mx_all = max(mx_all, mx)
            if g != rg or y.tolist() != ry:
                ok_all = False
                print(f"    불일치: {name} gmax={gmax}")
    check("numpy == C 참조 (45종 x 2)", ok_all)
    check("x*G + 반올림이 int32 안", mx_all < (1 << 31), f"(최대 {mx_all:,})")
    g, y = S.norm16(cases["int8 포화점 8128"], 4)
    check("풀스케일 창은 피크 96 으로 줄어든다", int(np.abs(y).max()) == 96,
          f"(G={g}, 기준 {S.N16_UNITY})")
    g, y = S.norm16(cases["피크 1"], 4)
    check("조용한 창은 상한에서 멈춘다", g == 4 * S.N16_UNITY and int(np.abs(y).max()) == 0)

    print("4. 로더 배선")
    root = os.path.join(AI8X, "data", "SafeSound16")
    # act_mode_8bit=True 면 ai8x.normalize 가 [-128,127] 정수 눈금으로 돌려준다
    # (False 는 [-1,1) — 같은 값의 1/128 배라 대조에는 정수 쪽이 편하다)
    tf = ai8x.normalize(args=argparse.Namespace(act_mode_8bit=True))
    for gmax in (4, 64):
        te = S.SafeSound16(root, "test", transform=tf, gmax=gmax)
        tr = S.SafeSound16(root, "train", transform=tf, gmax=gmax)
        x, t = te[0]
        check(f"gmax {gmax} test shape/범위", tuple(x.shape) == (128, 128)
              and float(x.min()) >= -128 and float(x.max()) <= 127,
              f"{tuple(x.shape)} [{float(x.min()):.0f}, {float(x.max()):.0f}]")
        _, shard, row, _, _ = te.index[0]
        w = np.asarray(te._shard(shard)[row])[S.MARGIN:S.MARGIN + S.WIN]  # noqa: SLF001
        _, y8 = S.norm16(w, gmax)
        back = x.transpose(1, 0).reshape(-1).numpy().astype(np.int16)
        check(f"gmax {gmax} test 출력 == norm16", np.array_equal(back, y8.astype(np.int16)))
        xs = [tr[i][0] for i in range(0, len(tr), max(1, len(tr) // 64))]
        pk = [float(v.abs().max()) for v in xs]
        check(f"gmax {gmax} train 증강 출력 범위", max(pk) <= 128,
              f"(피크 중앙 {np.median(pk):.0f}, 최소 {min(pk):.0f})")

    print()
    if FAIL:
        print(f"실패 {len(FAIL)}건")
        sys.exit(1)
    print("전부 통과")


if __name__ == "__main__":
    main()
