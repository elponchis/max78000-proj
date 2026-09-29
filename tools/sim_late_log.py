#!/usr/bin/env python3
"""(a) 와 (c) 중 어느 쪽을 할지 **학습 없이** 가른다.

`docs/results/g8-round6-design.md` a.4 / c.5 의 사전 확인.

**질문**: 로그를 **언제** 거는가가 중요한가.
  ① 은 float 대역 파워에 로그를 걸고 **그다음** int8 로 양자화한다.
  (a) 는 NPU 층 출력이 이미 int8 이라 **선형 양자화 뒤에** 로그를 건다 —
      1 LSB 아래로 내려간 대역은 이미 0 이고 로그를 걸어도 돌아오지 않는다.
  (c) 는 1층을 wide(32bit)로 빼서 **양자화 전에** CPU 가 로그를 건다.

**방법**: ① 의 멜 파워를 받아 네 경로로 int8 을 만들고, **학습된 ① 모델로
평가만** 한다 (재학습 없음).

| 경로 | 흉내 내는 것 | 기대 |
|---|---|---|
| S0 | ① 그대로 (float 로그 → int8) | 상한 |
| S1 | 대역크기 **int8** → 로그 → int8 | **(a)** |
| S2 | 대역크기 **int32** → 로그 → int8 | **(c)** |
| S3 | 대역크기 int8, 로그 없음 | 하한 (①-lin 근사) |

⚠️ **S1/S2 는 ① 모델로 평가하는 것이라 상한이다.** 그 입력으로 재학습하면
달라진다. **순서를 가리는 용도**이지 최종 수치가 아니다.

⚠️ 여기서 "대역크기 int8" 은 **NPU 활성화의 8bit 양자화**를 흉내 낸 것이다.
실제 NPU 는 `output_shift` 로 스케일을 맞추므로, 여기서는 **이벤트 클래스
대역 크기의 p99.9 를 풀스케일로** 두는 최선의 스케일을 가정한다 —
(a) 에 유리한 쪽이다. 그래도 S1 이 나쁘면 결론이 강해진다.

사용 (WSL2, ai8x venv):
    ~/ai8x-training/venv/bin/python tools/sim_late_log.py \\
        --checkpoint data/safesound-mel-v1_qat_best.pth.tar
"""

import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets"), os.path.join(REPO, "tools"),
                os.path.join(REPO, "scripts")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
WIN = 16384
MARGIN = 1600


def windows(data_root, split="test"):
    """①과 같은 창을 같은 순서로 (ceiling_check.iter_windows 와 같은 규약)."""
    import ceiling_check as CC
    for rec in CC.iter_windows(data_root, split):
        yield rec


def band_mag(w_int8):
    """int8 파형 → 멜 대역 **크기** (mels, frames). 로그 이전 단계."""
    import melfeat as MF
    x = np.asarray(w_int8, dtype=np.float64) / 128.0
    p = (MF.stft_power(x) @ MF._FB.T).T
    return np.sqrt(np.maximum(p, 0.0))          # 크기 = sqrt(파워)


def q_uniform(a, full, bits=8):
    """[0, full] 을 (2^bits − 1) 단계로 균등 양자화한 뒤 되돌린다.

    NPU 활성화 양자화를 흉내 낸다. 되돌리는 이유는 그 뒤에 로그를 걸기
    위해서다 — 로그는 값을 보지 계단 번호를 보지 않는다.
    """
    lv = (1 << bits) - 1
    q = np.clip(np.round(a / full * lv), 0, lv)
    return q * full / lv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="data/safesound-mel-v1_qat_best.pth.tar")
    ap.add_argument("--data", default=os.path.expanduser("~/ai8x-training/data"))
    ap.add_argument("--ai8x", default=AI8X)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    import ai8x
    import melfeat as MF
    import eval_confusion as EC

    ai8x.set_device(85, False, False)
    model = EC.load_model(a.checkpoint, len(CLASSES), a.ai8x, False, False,
                          config="mel")
    norm = ai8x.normalize(args=argparse.Namespace(act_mode_8bit=False))

    # 1) 대역 크기를 모아 스케일(p99.9)을 정한다 — (a) 에 최선인 스케일이다
    print("창을 읽고 대역 크기를 만든다...")
    ys, fs, mags = [], [], []
    for i, (_i, t, _c, fsid, _s, w) in enumerate(windows(a.data, a.split)):
        if a.limit and i >= a.limit:
            break
        ys.append(t)
        fs.append(fsid)
        mags.append(band_mag(w).astype(np.float32))
    ys, fs = np.array(ys), np.array(fs)
    M = np.stack(mags)                                   # (N, mels, frames)
    full = float(np.percentile(M, 99.9))
    print(f"  창 {len(ys):,}개, 대역 크기 p99.9 = {full:.5f} "
          f"(= {20*np.log10(max(full,1e-12)):.1f} dB) 를 int8 풀스케일로 둔다")

    # 2) 네 경로의 int8 입력
    def to_db_int8(mag):
        db = 20.0 * np.log10(np.maximum(mag, 0.0) + MF.EPS)
        return MF.db_to_int8(db)                         # TOP_DB/SPAN_DB 그대로

    paths = {}
    paths["S0 ① 그대로"] = np.stack([to_db_int8(m) for m in M])
    q8 = q_uniform(M, full, 8)
    paths["S1 int8 → 로그  ((a))"] = np.stack([to_db_int8(m) for m in q8])
    q32 = q_uniform(M, full, 24)      # 32bit 누산 중 유효 비트를 넉넉히
    paths["S2 int32 → 로그 ((c))"] = np.stack([to_db_int8(m) for m in q32])
    # S1b: (a) 가 **실제로** 쓰는 것 — 정확한 로그가 아니라 K마디
    #      구간선형 오목 근사다. f(v) = a·v − Σ c_k·ReLU(v − t_k).
    #      마디는 로그 곡선 위에 로그 간격으로 놓고, 기울기는 그 구간의
    #      할선(secant)으로 맞춘다 (오목성이 보장된다).
    def pwl_log(mag, K=4):
        v = mag / full                       # [0,1] 정규화 (int8 활성화 축)
        lo = 1.0 / 255.0                     # 1 LSB
        knots = np.concatenate([[0.0], np.exp(
            np.linspace(np.log(lo), np.log(1.0), K + 1))[:-1]])
        fk = np.log10(np.maximum(knots, lo))
        out = np.full_like(v, fk[0])
        for j in range(len(knots) - 1):
            x0, x1 = knots[j], knots[j + 1]
            y0, y1 = fk[j], np.log10(max(x1, lo))
            sl = (y1 - y0) / max(x1 - x0, 1e-12)
            out = np.where(v > x0, y0 + sl * np.minimum(v - x0, x1 - x0), out)
        x_last = knots[-1]
        sl = (0.0 - np.log10(max(x_last, lo))) / max(1.0 - x_last, 1e-12)
        out = np.where(v > x_last,
                       np.log10(max(x_last, lo)) + sl * (v - x_last), out)
        return 20.0 * out + 20.0 * np.log10(full)     # dB 축으로 되돌린다

    paths["S1b int8 → 구간선형 K=4 ((a) 실제)"] = np.stack(
        [MF.db_to_int8(pwl_log(m)) for m in q8])

    # S3: 로그 없이 선형 크기를 int8 로 (①-lin 근사)
    paths["S3 로그 없음 (하한)"] = np.clip(
        np.round(M / full * 255.0) - 128.0, -128, 127).astype(np.int8)

    # 3) 평가
    print()
    print(f"{'경로':<36}{'@300/h':>9}{'@1000/h':>10}{'argmax F1':>11}"
          f"{'0 비율':>9}")
    print("-" * 76)
    out = {}
    for name, X in paths.items():
        lg = []
        with torch.no_grad():
            for b in range(0, len(X), a.batch_size):
                t = (torch.from_numpy(X[b:b + a.batch_size].astype(np.int16))
                     + 128).float() / 256.0
                lg.append(model(norm(t.unsqueeze(1))).numpy())
        lg = np.concatenate(lg)
        fx = EC.fixed_fa_points(lg, ys, CLASSES, hop_ms=250)
        tg = fx.get("targets", {})
        f3 = (tg.get("300/h") or {}).get("macro_f1")
        f1k = (tg.get("1000/h") or {}).get("macro_f1")
        am = EC.macro_f1(ys, lg.argmax(1), len(CLASSES))
        zero = float((X <= -128).mean())
        out[name] = {"f300": f3, "f1000": f1k, "argmax": am, "floor": zero}
        print(f"{name:<36}{f3:>9.4f}{f1k:>10.4f}{am:>11.4f}{100*zero:>8.1f}%")

    # 4) 판정
    s0 = out["S0 ① 그대로"]["f300"]
    s1 = out["S1 int8 → 로그  ((a))"]["f300"]
    s2 = out["S2 int32 → 로그 ((c))"]["f300"]
    s3 = out["S3 로그 없음 (하한)"]["f300"]
    print()
    print("=== 판정 (결과 보기 전에 정한 규칙, g8-round6-design.md c.5) ===")
    print(f"  S0 상한 {s0:.4f}   S3 하한 {s3:.4f}   폭 {s0-s3:.4f}")
    rec = lambda v: (v - s3) / (s0 - s3) if s0 > s3 else float("nan")
    s1b = out["S1b int8 → 구간선형 K=4 ((a) 실제)"]["f300"]
    print(f"  S1  ((a) 정확한 로그) 회복률 {100*rec(s1):.1f}%")
    print(f"  S1b ((a) 구간선형 K=4)  회복률 {100*rec(s1b):.1f}%  "
          f"← **이쪽이 (a) 가 실제로 할 수 있는 것**")
    print(f"  S2  ((c) wide 출력)     회복률 {100*rec(s2):.1f}%")
    print()
    if rec(s2) - rec(s1b) > 0.20:
        print("  → **(c) 가 (a) 보다 유망하다.** wide 출력이 '늦은 로그' 문제를")
        print("     실제로 피한다. (c) 의 합성 확인(c.2)으로 넘어갈 것.")
    elif rec(s1b) > 0.60:
        print("  → **(a) 로도 대부분 회복된다.** 더 싼 (a) 를 먼저 할 것.")
    else:
        print("  → **둘 다 회복이 시원치 않다.** 로그를 늦게 거는 것 자체가")
        print("     문제이거나, 대역 크기까지의 표현이 이미 부족한 것이다.")
        print("     둘 다 착수하지 말고 다른 후보를 찾을 것.")
    print()
    print("  ⚠️ S1/S2 는 ① 모델로 평가한 값이라 **상한**이다. 그 입력으로")
    print("     재학습하면 달라진다 — 순서를 가리는 용도다.")

    if a.json:
        import json
        json.dump(out, open(a.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"  저장: {a.json}")


if __name__ == "__main__":
    main()
