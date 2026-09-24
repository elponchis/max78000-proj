#!/usr/bin/env python3
"""`datasets/safesound.py` known-answer 테스트.

데이터로더의 실수는 **학습이 끝난 뒤에야** 드러난다 — 축이 뒤바뀌어도 손실은
내려가고, 합성해서 보드에 올린 뒤에야 어긋난 게 보인다. 그래서 값을 손으로
아는 입력을 넣고 결과를 자리별로 확인한다.

확인하는 것
  1. (128,128) 변환이 `kws20.py:558-560` 의 `torch.transpose(reshape(-1,128),1,0)`
     과 **자리 단위로** 같은가
  2. int8 → `/256` → `ai8x.normalize` 왕복이 원래 int8 값을 되돌리는가
     (kws20.py:592 + ai8x.py:29 와 같은 관례)
  3. shift 가 재절단으로 동작하는가 — 여유(`left/right_margin`) 밖으로 나가지
     않고, 증강을 끄면 언제나 가운데 1초인가
  4. 랜덤 게인이 규칙을 지키는가 — 클리핑을 만들지 않고, int8 에서 빈 창을
     만들지 않고, 범위 안인가
  5. lazy 인가 — 샤드를 mmap 으로만 열고 전체를 메모리에 올리지 않는가

입력 벡터는 `tools/kat_vectors/` 에 저장된다. **보드 도착 후 펌웨어 쪽에서 같은
입력을 넣어 같은 값이 나오는지 확인하는 데 그대로 쓴다** (TASKS.md 관문 C).
PC 전처리와 펌웨어 전처리가 어긋나면 정확도 괴리(G4)의 원인이 되는데, 같은
벡터를 양쪽에 넣어 보면 어긋난 지점이 바로 드러난다.

사용법 (WSL2):
    python3 tools/kat_safesound.py                # 검증 (벡터 없으면 생성)
    python3 tools/kat_safesound.py --save         # 벡터 다시 쓰기
"""

import csv
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

# ai8x 가 없어도 돌도록 최소 스텁을 넣는다 (정규화 규약만 재현).
try:
    import ai8x                                            # noqa: F401
except ImportError:
    import types as _t

    import torch as _torch

    class _Norm:
        def __init__(self, args=None):
            self.args = args

        def __call__(self, img):
            # ai8x.py:29 `normalize` 의 act_mode_8bit 경로와 동일
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)

    sys.modules["ai8x"] = _t.SimpleNamespace(normalize=_Norm)
    del _torch

import torch                                               # noqa: E402

import safesound as S                                      # noqa: E402

OK, FAIL = "  [통과]", "  [실패]"
fails = []


def check(name, cond, detail=""):
    print(f"{OK if cond else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def make_fixture(root, rows):
    """샤드 하나짜리 가짜 데이터셋. rows = [(cls, STORE 배열, left, right)]"""
    by = {}
    for cls, w, left, right in rows:
        by.setdefault(cls, []).append((w, left, right))
    for cls, items in by.items():
        d = os.path.join(root, "train", cls)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "shard_0000.npy"),
                np.stack([w for w, _l, _r in items]).astype(np.int8))
        with open(os.path.join(d, "index.csv"), "w", newline="",
                  encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["shard", "row", "clip_id", "fsid", "start_sample",
                         "left_margin", "right_margin", "note"])
            for i, (_w, left, right) in enumerate(items):
                wr.writerow([0, i, f"kat{i}", f"kat{i}", 0, left, right, ""])


VEC_DIR = os.path.join(REPO, "tools", "kat_vectors")


def c_array(name, arr, per_line=16):
    """int8 배열 → C 헤더 문자열. 펌웨어에 그대로 붙여 넣는다."""
    flat = arr.reshape(-1).astype(np.int8)
    body = ""
    for i in range(0, len(flat), per_line):
        body += "    " + ", ".join(f"{int(v):4d}" for v in flat[i:i + per_line]) + ",\n"
    return f"static const int8_t {name}[{len(flat)}] = {{\n{body}}};\n"


def save_vectors(win_int8, tensor_int8, d=VEC_DIR):
    """KAT 입력·기대 출력을 파일로 남긴다 (npy + C 헤더)."""
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "kat_window_int8.npy"), win_int8)
    np.save(os.path.join(d, "kat_input2d_int8.npy"), tensor_int8)
    with open(os.path.join(d, "kat_vectors.h"), "w", encoding="utf-8") as f:
        f.write("/* SafeSound known-answer 벡터 — tools/kat_safesound.py 생성.\n"
                " * kat_window: 전처리가 내놓는 1초 창 (int8, 16384샘플).\n"
                " * kat_input2d: 그것을 (128,128) 로 편 NPU 입력. 행 r, 열 c 는\n"
                " *   원본 인덱스 c*128+r 이다 (kws20.py:558-560 과 같은 순서).\n"
                " * 펌웨어에서 kat_window 를 입력으로 넣고 CNN 입력 버퍼가\n"
                " * kat_input2d 와 일치하는지 확인할 것. 손으로 고치지 말 것. */\n"
                "#ifndef KAT_VECTORS_H\n#define KAT_VECTORS_H\n#include <stdint.h>\n\n")
        f.write(c_array("kat_window", win_int8))
        f.write("\n")
        f.write(c_array("kat_input2d", tensor_int8))
        f.write("\n#endif /* KAT_VECTORS_H */\n")
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# KAT 벡터 (SafeSound)\n\n"
            "`tools/kat_safesound.py` 가 생성한다. **손으로 고치지 말 것.**\n\n"
            "| 파일 | 내용 |\n|---|---|\n"
            "| `kat_window_int8.npy` | 전처리 출력 1초 창 (int8, 16384) |\n"
            "| `kat_input2d_int8.npy` | (128,128) NPU 입력 |\n"
            "| `kat_vectors.h` | 위 둘의 C 배열 — 펌웨어용 |\n\n"
            "입력은 `(i % 255) - 127` 램프다. 자리마다 값이 달라 축이 뒤바뀌면\n"
            "바로 드러난다.\n\n"
            "## 쓰는 곳\n\n"
            "1. PC: `python3 tools/kat_safesound.py` — 데이터로더 회귀 검사\n"
            "2. 보드(관문 C): `kat_window` 를 펌웨어 전처리에 넣고 CNN 입력\n"
            "   버퍼가 `kat_input2d` 와 일치하는지 확인. 어긋나면 전처리 축이나\n"
            "   스케일이 PC 와 다른 것이다 (CLAUDE.md 7장).\n")
    return d


def main():
    save = "--save" in sys.argv
    tmp = tempfile.mkdtemp(prefix="kat_safesound_")
    root = os.path.join(tmp, "SafeSound")

    # 자리 확인용: 가운데 1초가 0..16383 을 127 로 접은 값이 되도록 만든다.
    ramp = (np.arange(S.WIN) % 255 - 127).astype(np.int8)
    center = np.zeros(S.STORE, dtype=np.int8)
    center[S.MARGIN:S.MARGIN + S.WIN] = ramp
    # 여유 구간은 상수로 채워 재절단이 실제로 다른 값을 가져오는지 본다
    center[:S.MARGIN] = 11
    center[S.MARGIN + S.WIN:] = -11

    quiet = np.zeros(S.STORE, dtype=np.int8)               # 완전 0 (배경음용)
    loud = np.full(S.STORE, 100, dtype=np.int8)
    make_fixture(root, [("siren", center, S.MARGIN, S.MARGIN),
                        ("glass", center, 0, 0),           # 여유 없음
                        ("background", quiet, S.MARGIN, S.MARGIN),
                        ("background", loud, S.MARGIN, S.MARGIN)])

    ds = S.SafeSound(root, "train", transform=None, augment=False)
    print(f"\n샘플 {len(ds)}개, 클래스 {S.CLASSES}")

    # ── 1. reshape 이 kws20 과 자리 단위로 같은가
    x, target = ds[0]
    ref = torch.transpose(
        ((torch.from_numpy(ramp.astype(np.int16)) + 128).float() / 256.0)
        .reshape((-1, 128)), 1, 0)
    check("(128,128) 변환이 kws20.py:558-560 공식과 일치",
          x.shape == (128, 128) and torch.equal(x, ref))
    # 자리 하나를 손으로 검산: 전치 후 x[r][c] 는 원본 인덱스 c*128+r
    r_, c_ = 3, 5
    want = (int(ramp[c_ * 128 + r_]) + 128) / 256.0
    check("전치 방향 검산 x[3][5] == 원본[5*128+3]",
          abs(float(x[r_][c_]) - want) < 1e-6,
          f"{float(x[r_][c_]):.6f} vs {want:.6f}")

    # ── 2. int8 왕복
    norm = sys.modules["ai8x"].normalize(args=None)
    ds_n = S.SafeSound(root, "train", transform=norm, augment=False)
    xn, _ = ds_n[0]
    back = torch.transpose(xn, 1, 0).reshape(-1).numpy().astype(np.int16)
    check("int8 → /256 → normalize 왕복이 원본 int8 을 되돌린다",
          np.array_equal(back, ramp.astype(np.int16)),
          f"최대 오차 {int(np.abs(back - ramp.astype(np.int16)).max())}")

    # ── 3. shift = 재절단
    check("증강 off 면 언제나 가운데 1초",
          all(torch.equal(ds[0][0], ds[0][0]) for _ in range(3)))
    ds_a = S.SafeSound(root, "train", transform=None, augment=True)
    # 재절단만 보려면 게인을 꺼야 한다 — 켜두면 값 변화가 시프트인지 게인인지
    # 구별되지 않는다 (이 테스트를 처음 썼을 때 실제로 헷갈렸다).
    ds_s = S.SafeSound(root, "train", transform=None, augment=True, gain_db=0.0)
    seen = {float(ds_s[0][0][0][0]) for _ in range(40)}
    check("증강 on 이면 자르는 위치가 흔들린다", len(seen) > 1,
          f"관측된 첫 샘플 값 {len(seen)}종")
    firsts = {float(ds_s[1][0][0][0]) for _ in range(40)}
    center_first = (int(ramp[0]) + 128) / 256.0
    check("여유가 0인 행은 흔들리지 않는다 (인덱스 margin 준수)",
          firsts == {center_first}, f"{sorted(firsts)[:3]}")
    # 잘라낸 1초가 저장 행의 **연속 구간**이고, 그 시작점이 여유 안인지 본다.
    offs = set()
    for _ in range(50):
        w = ds_s._crop(center, S.MARGIN, S.MARGIN,            # noqa: SLF001
                       np.random.default_rng())
        hit = [o for o in range(S.STORE - S.WIN + 1)
               if np.array_equal(center[o:o + S.WIN], w)]
        offs.update(hit)
    check("잘라낸 창이 저장 행의 연속 구간이다", bool(offs))
    check("시프트가 ±100ms 여유를 벗어나지 않는다",
          offs and min(offs) >= S.MARGIN - ds_s.shift
          and max(offs) <= S.MARGIN + ds_s.shift,
          f"오프셋 {min(offs)}~{max(offs)} (허용 "
          f"{S.MARGIN - ds_s.shift}~{S.MARGIN + ds_s.shift})")

    # ── 4. 게인 규칙
    rng = np.random.default_rng(0)
    w = (ramp // 2).astype(np.int8)
    outs = [ds_a._rand_gain(w, rng) for _ in range(200)]    # noqa: SLF001
    ratios = [float(np.abs(o).max()) / max(float(np.abs(w).max()), 1) for o in outs]
    check("게인이 int8 범위를 넘지 않는다",
          all(np.abs(o).max() <= 127 for o in outs))
    check("게인 후에도 int8 에서 비지 않는다 (floor_zero 준수)",
          all(float((o == 0).mean()) <= ds_a.floor_zero for o in outs))
    check("게인 폭이 ±12dB 안", max(ratios) <= 10 ** (S.GAIN_DB / 20) + 0.01,
          f"최대 배율 {max(ratios):.2f} (허용 {10 ** (S.GAIN_DB / 20):.2f})")

    # ── 5. lazy
    m = ds._shard(ds.index[0][1])                           # noqa: SLF001
    check("샤드를 mmap 으로 연다 (전체 로딩 아님)",
          isinstance(m, np.memmap), type(m).__name__)
    check("인덱스만 들고 있고 샘플 배열은 들고 있지 않다",
          not any(isinstance(v, np.ndarray) and v.size > S.STORE
                  for v in ds.__dict__.values()))

    # ── 6. 벡터 파일 — 보드 도착 후 펌웨어 검증에 그대로 쓴다
    tensor_int8 = xn.numpy().astype(np.int8)     # normalize 후 (128,128) int8
    if save or not os.path.isfile(os.path.join(VEC_DIR, "kat_input2d_int8.npy")):
        d = save_vectors(ramp, tensor_int8)
        print(f"\n  벡터 저장 → {os.path.relpath(d, REPO)}")
    else:
        old_w = np.load(os.path.join(VEC_DIR, "kat_window_int8.npy"))
        old_t = np.load(os.path.join(VEC_DIR, "kat_input2d_int8.npy"))
        check("저장된 KAT 벡터와 현재 파이프라인이 일치 (회귀 검사)",
              np.array_equal(old_w, ramp) and np.array_equal(old_t, tensor_int8),
              "다르면 전처리·로더가 바뀐 것이다 — 의도한 변경이면 --save")

    print()
    if fails:
        print(f"실패 {len(fails)}건: " + ", ".join(fails))
        sys.exit(1)
    print("전부 통과")


if __name__ == "__main__":
    main()
