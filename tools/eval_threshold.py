#!/usr/bin/env python3
"""배경음 로짓 오프셋 스윕 — 오경보와 재현율의 맞교환 곡선.

기준선에서 배경음 오탐이 34%(시간당 4,896회)로 나왔다. 재학습 전에 **이미 있는
체크포인트로** 어디까지 내려갈 수 있는지부터 본다. 배경음 로짓에 상수 `b` 를 더하면
결정 경계가 배경음 쪽으로 밀리고, `b` 하나로 오경보-재현율 곡선 전체가 그려진다.

여기서 나오는 답은 두 갈래다.
  · **적당한 `b` 로 오경보를 쓸 만한 수준까지 낮출 수 있다** → 재학습이 아니라
    운용 임계값 문제다. 그 `b` 를 펌웨어 후처리에 넣는다
  · **어떤 `b` 에서도 재현율이 무너진다** → 모델·데이터 문제다. 재학습으로 간다

⚠️ 창 단위 수치다. `tools/eval_confusion.py` 의 N프레임 표와 같은 한계를 가지며
(창이 시간적으로 떨어져 있다), 논문에 실을 오경보는 `tools/stream_eval.py` 와
보드 구동에서 얻는다. 여기서는 **상대 비교**(어느 `b` 가 얼마나 낫나)만 쓴다.

사용법 (Colab 또는 WSL2):
    python3 tools/eval_threshold.py --checkpoint <qat_best.pth.tar> \\
        --data /content/ai8x-training/data --ai8x /content/ai8x-training \\
        --png /content/drive/MyDrive/max78000/threshold-sweep.png
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sweep(logits, y_true, k, bg, offsets, per_hour):
    """오프셋마다 (시간당 오경보, 클래스별 재현율)."""
    from eval_confusion import recalls

    rows = []
    sel = y_true == bg
    for b in offsets:
        adj = logits.copy()
        adj[:, bg] += b
        pred = adj.argmax(1)
        fa = float((pred[sel] != bg).mean()) if sel.sum() else float("nan")
        rows.append({"b": float(b), "fa_rate": fa, "per_hour": fa * per_hour,
                     "recall": recalls(y_true, pred, k)})
    return rows


def at_targets(rows, names, targets=(100.0, 10.0, 1.0)):
    """목표 오경보(시간당) 이하가 되는 **첫 지점**을 고른다 (b 가 커질수록 감소)."""
    out = {}
    for t in targets:
        hit = next((r for r in rows if r["per_hour"] <= t), None)
        out[t] = hit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--ai8x", default="/content/ai8x-training")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--hop-ms", type=int, default=250)
    ap.add_argument("--lo", type=float, default=-2.0)
    ap.add_argument("--hi", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--png", help="곡선 PNG 저장 경로")
    ap.add_argument("--json", help="스윕 결과 JSON 저장 경로")
    a = ap.parse_args()

    # safesound 는 import 시점에 ai8x 를 부른다 → 경로를 먼저 넣는다
    if os.path.isdir(a.ai8x):
        sys.path.insert(0, a.ai8x)
    import safesound as S
    from eval_confusion import collect_logits

    names = S.CLASSES
    k, bg = len(names), names.index("background")
    per_hour = 3600.0 / (a.hop_ms / 1000.0)

    print(f"체크포인트: {a.checkpoint}")
    _f, _c, _s, y_true, logits = collect_logits(a, names)
    n_bg = int((y_true == bg).sum())
    print(f"테스트 창 {len(y_true):,} (배경음 {n_bg:,})  hop {a.hop_ms}ms "
          f"→ 시간당 {per_hour:,.0f}회 추론\n")

    offsets = np.arange(a.lo, a.hi + 1e-9, a.step)
    rows = sweep(logits, y_true, k, bg, offsets, per_hour)

    print(f"{'b':>6}{'오경보/h':>11}{'오탐률':>9}   " +
          "".join(f"{n[:8]:>9}" for n in names))
    print("-" * (26 + 9 * k))
    for r in rows:
        if abs(r["b"] * 4 - round(r["b"] * 4)) < 1e-6 and round(r["b"] * 2) == r["b"] * 2:
            pass                                   # 0.5 간격만 표로 (나머지는 JSON)
        else:
            continue
        rec = "".join(f"{100*v:>8.1f}%" for v in r["recall"])
        print(f"{r['b']:>6.2f}{r['per_hour']:>11,.0f}{r['fa_rate']:>8.1%}   {rec}")

    print(f"\n=== 오경보 목표 지점 (창 단위 상대 비교) ===")
    hits = at_targets(rows, names)
    print(f"{'목표/h':>8}{'b':>7}{'실제/h':>10}   " +
          "".join(f"{n[:8]:>9}" for n in names))
    print("-" * (25 + 9 * k))
    for t, r in hits.items():
        if r is None:
            print(f"{t:>8.0f}{'—':>7}{'도달 불가':>10}   "
                  f"(b={a.hi} 까지 올려도 {rows[-1]['per_hour']:,.0f}회)")
            continue
        rec = "".join(f"{100*v:>8.1f}%" for v in r["recall"])
        print(f"{t:>8.0f}{r['b']:>7.2f}{r['per_hour']:>10,.0f}   {rec}")

    base = rows[int(round((0.0 - a.lo) / a.step))] if a.lo <= 0 <= a.hi else None
    if base:
        print(f"\n  b=0 (원래 모델): 시간당 {base['per_hour']:,.0f}회, "
              + ", ".join(f"{n} {100*v:.1f}%" for n, v in zip(names, base['recall'])))
    print("\n  ⚠️ 창 단위 상대 비교용이다. 논문에 실을 오경보는 stream_eval.py 와")
    print("     보드 구동에서 얻는다. 여기서 고른 b 는 펌웨어 후처리에 그대로 넣는다.")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ph = [r["per_hour"] for r in rows]
        for c, n in enumerate(names):
            if c == bg:
                continue
            ax[0].plot(ph, [100 * r["recall"][c] for r in rows], marker=".", label=n)
        ax[0].set_xscale("symlog", linthresh=1)
        ax[0].invert_xaxis()
        ax[0].set_xlabel("시간당 오경보 (창 단위, 로그)")
        ax[0].set_ylabel("재현율 (%)")
        ax[0].set_title("오경보 vs 클래스별 재현율")
        ax[0].grid(alpha=.3)
        ax[0].legend()
        ax[1].plot([r["b"] for r in rows], ph, marker=".")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("배경음 로짓 오프셋 b")
        ax[1].set_ylabel("시간당 오경보")
        ax[1].set_title("오프셋 vs 오경보")
        ax[1].grid(alpha=.3)
        for t in (100, 10, 1):
            ax[1].axhline(t, ls="--", lw=.8, color="gray")
        fig.suptitle(f"배경음 로짓 오프셋 스윕 — {os.path.basename(a.checkpoint)}")
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.png) or ".", exist_ok=True)
        fig.savefig(a.png, dpi=120)
        print(f"\n  그림: {a.png}")

    if a.json:
        out = {"checkpoint": a.checkpoint, "hop_ms": a.hop_ms,
               "classes": names, "n_windows": int(len(y_true)), "n_background": n_bg,
               "sweep": [{"b": r["b"], "per_hour": round(r["per_hour"], 1),
                          "fa_rate": round(r["fa_rate"], 5),
                          "recall": {n: round(float(v), 4)
                                     for n, v in zip(names, r["recall"])}}
                         for r in rows]}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  JSON: {a.json}")


if __name__ == "__main__":
    main()
