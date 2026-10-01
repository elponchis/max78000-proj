#!/usr/bin/env python3
"""(c) 2단계 KAT — 파이썬 `log2_q8` 과 C `log2_q8` 의 **비트 단위 일치**.

**이게 통과하지 않으면 (c) 를 하지 않는다** (`docs/results/g8-c-design.md` 3.2).
학습은 파이썬 구현으로 돌고 실기기는 C 구현으로 돈다. 한 비트라도 다르면
train/serve 불일치가 되고, G4(온디바이스 실측 vs PC 시뮬레이션 괴리)의
원인을 스스로 만든다.

## 무엇을 비교하나

| | |
|---|---|
| 전수 | `v = 0 … 65,535` (하위 16bit 전부) |
| 경계 | 모든 `2^k`, `2^k ± 1` (k = 0..31) |
| 로그 격자 | 2^k 구간마다 고르게 뽑은 값 — 상위 비트까지 훑는다 |
| 무작위 | uint32 전역에서 1e7 개, 시드 고정 |
| `to_int8` | 위 값들에 대해 포화·바닥 나눗셈까지 함께 |

전 범위(2^32)를 다 돌면 수 분이 걸리고 얻는 것이 없다 — LUT 는 가수 상위
8bit 만 보므로 **같은 (e, m) 쌍이면 결과가 같다.** 위 네 집합이 모든
(e, m) 조합을 덮는다 (전수 하위 16bit 가 e ≤ 15 전부, 로그 격자가 e ≥ 16).

사용 (WSL2):
    python3 tools/kat_logstage.py
    python3 tools/kat_logstage.py --random 100000000   # 더 세게
"""
import argparse
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from log2_q8 import LUT, log2_q8, to_int8          # noqa: E402

C_SRC = os.path.join(REPO, "firmware/common/log2_q8.c")
LUT_INC = os.path.join(REPO, "firmware/common/log2_q8_lut.inc")


def write_lut():
    """LUT 를 **파이썬 쪽에서** 생성해 C 가 포함하게 한다.

    ⚠️ 이게 핵심이다. 두 구현이 각자 LUT 를 들고 있으면 언젠가 갈린다.
    **하나의 소스**에서 나와야 비트 일치가 유지된다.
    """
    lines = []
    for i in range(0, 256, 8):
        lines.append("    " + " ".join(f"{int(v):6d}," for v in LUT[i:i + 8]))
    body = ("/* tools/kat_logstage.py 가 생성한다 — 손으로 고치지 말 것.\n"
            " * round(log2(1 + i/256) * 256), i = 0..255 */\n"
            + "\n".join(lines) + "\n")
    old = None
    if os.path.isfile(LUT_INC):
        with open(LUT_INC, encoding="utf-8") as f:
            old = f.read()
    if old != body:
        os.makedirs(os.path.dirname(LUT_INC), exist_ok=True)
        with open(LUT_INC, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        print(f"  LUT 생성: {LUT_INC}")
    else:
        print("  LUT 변화 없음")


def build():
    """C 구현을 x86 공유 라이브러리로 빌드해 ctypes 로 부른다."""
    so = os.path.join(tempfile.gettempdir(), "log2_q8_kat.so")
    cmd = ["gcc", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
           "-I", os.path.dirname(C_SRC), C_SRC, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print("빌드 실패:\n" + r.stderr)
        sys.exit(2)
    lib = ctypes.CDLL(so)
    lib.log2_q8.argtypes = [ctypes.c_uint32]
    lib.log2_q8.restype = ctypes.c_int32
    lib.log2_q8_to_int8.argtypes = [ctypes.c_int32, ctypes.c_int32,
                                    ctypes.c_int32]
    lib.log2_q8_to_int8.restype = ctypes.c_int8
    return lib


def c_log2(lib, vs):
    return np.fromiter((lib.log2_q8(int(v)) for v in vs),
                       dtype=np.int32, count=len(vs))


def check(name, lib, vs):
    vs = np.unique(np.asarray(vs, dtype=np.int64))
    vs = vs[(vs >= 0) & (vs <= 0xFFFFFFFF)]
    py = log2_q8(vs)
    c = c_log2(lib, vs)
    bad = np.nonzero(py != c)[0]
    ok = len(bad) == 0
    print(f"  {name:<28} {len(vs):>11,}개  "
          + ("✔ 전부 일치" if ok else f"✘ 불일치 {len(bad):,}개"))
    if not ok:
        for i in bad[:5]:
            print(f"      v={int(vs[i]):>12}  py={int(py[i]):>8}  c={int(c[i]):>8}")
    return ok, vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=20261001)
    a = ap.parse_args()

    print("=== (c) 2단계 KAT — log2_q8 파이썬 vs C")
    write_lut()
    lib = build()
    print()

    rng = np.random.default_rng(a.seed)
    ks = np.arange(32, dtype=np.int64)
    pow2 = 1 << ks
    sets = [
        ("전수 하위 16bit", np.arange(0, 1 << 16, dtype=np.int64)),
        ("2^k 경계 ±1", np.concatenate([pow2 - 1, pow2, pow2 + 1])),
        ("로그 격자", np.concatenate(
            [np.unique(np.linspace(int(p), int(min(2 * p - 1, 0xFFFFFFFF)),
                                   512).astype(np.int64)) for p in pow2])),
        (f"무작위 {a.random:,}", rng.integers(0, 1 << 32, size=a.random,
                                            dtype=np.int64)),
    ]
    all_ok, pool = True, []
    for n, vs in sets:
        ok, u = check(n, lib, vs)
        all_ok &= ok
        pool.append(u[:200000])

    # to_int8 — 포화와 바닥 나눗셈까지
    print()
    vs = np.unique(np.concatenate(pool))
    lg = log2_q8(vs)
    for lo, span in ((0, 8 * 256), (4 * 256, 10 * 256), (-2 * 256, 3 * 256)):
        py = to_int8(lg, lo, span)
        c = np.fromiter((lib.log2_q8_to_int8(int(x), lo, span) for x in lg),
                        dtype=np.int8, count=len(lg))
        bad = np.nonzero(py != c)[0]
        ok = len(bad) == 0
        all_ok &= ok
        print(f"  to_int8 lo={lo:>6} span={span:>5}  {len(lg):>9,}개  "
              + ("✔ 전부 일치" if ok else f"✘ 불일치 {len(bad):,}개"))
        if not ok:
            for i in bad[:5]:
                print(f"      lg={int(lg[i])}  py={int(py[i])}  c={int(c[i])}")

    print()
    if all_ok:
        print("✔ KAT 통과 — 파이썬과 C 가 비트 단위로 같다. (c) 3단계로 갈 수 있다")
        return 0
    print("✘ KAT 실패 — **(c) 를 진행하지 않는다**")
    return 1


if __name__ == "__main__":
    sys.exit(main())
