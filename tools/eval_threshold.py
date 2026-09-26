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


def half_split_by_origin(fsids, y_true, seed=0):
    """테스트셋을 **원본(fsid) 단위**로 반분한다. 클래스별로 번갈아 배정한다.

    창 단위로 나누면 같은 녹음의 창이 양쪽에 흩어져 두 반이 서로 독립이 아니다 —
    한쪽에서 고른 임계값이 다른 쪽에 이미 맞춰져 있는 셈이 된다. 원본 단위로
    갈라야 "본 적 없는 녹음에서 그 임계값이 통하나"를 볼 수 있다.

    클래스별로 시드 고정 순열을 만든 뒤 번갈아 배정해, 두 반의 클래스 구성이
    비슷하게 유지된다 (원본 하나는 한 클래스에만 속한다 — 규칙 1).
    """
    cls_of = {}
    for f, t in zip(fsids, y_true):
        cls_of[f] = int(t)
    rng = np.random.default_rng(seed)
    a, b = set(), set()
    for c in sorted(set(cls_of.values())):
        fs = sorted(f for f, t in cls_of.items() if t == c)
        for i, j in enumerate(rng.permutation(len(fs))):
            (a if i % 2 == 0 else b).add(fs[j])
    return a, b


def print_halves(fsids, y_true, a, b, names):
    """두 반의 클래스별 원본·창 수 — 구성이 치우쳤는지 눈으로 확인한다."""
    fs = np.asarray(fsids)
    ma = np.isin(fs, list(a))
    mb = np.isin(fs, list(b))
    print(f"  {'클래스':<12}{'A 원본':>8}{'B 원본':>8}{'A 창':>8}{'B 창':>8}")
    print("  " + "-" * 44)
    for c, n in enumerate(names):
        oa = len({f for f, t in zip(fsids, y_true) if t == c and f in a})
        ob = len({f for f, t in zip(fsids, y_true) if t == c and f in b})
        print(f"  {n:<12}{oa:>8}{ob:>8}{int((ma & (y_true == c)).sum()):>8}"
              f"{int((mb & (y_true == c)).sum()):>8}")
    print(f"  {'합계':<12}{len(a):>8}{len(b):>8}{int(ma.sum()):>8}{int(mb.sum()):>8}")
    return ma, mb


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
    ap.add_argument("--pick-on", choices=["test-half", "val", "test"],
                    default="test-half",
                    help="운용 지점(b)을 고를 데이터. test-half(기본)=테스트셋을 "
                         "**원본 단위**로 반분해 한쪽에서 고르고 다른 쪽에 적용. "
                         "val=ai8x 검증셋(창 단위 분리라 낙관적). "
                         "test=같은 데이터에서 고름(진단용, 낙관치)")
    ap.add_argument("--half-seed", type=int, default=0,
                    help="원본 단위 반분 시드 (실험 간 비교를 위해 고정한다)")
    ap.add_argument("--validation-split", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="ai8x 가 검증셋을 떼는 시드. distiller 가 get_data_loaders "
                         "안에서 set_deterministic() 을 인자 없이 불러 0 이 된다")
    ap.add_argument("--png", help="곡선 PNG 저장 경로")
    ap.add_argument("--json", help="스윕 결과 JSON 저장 경로")
    a = ap.parse_args()

    # safesound 는 import 시점에 ai8x 를 부른다 → 경로를 먼저 넣는다
    if os.path.isdir(a.ai8x):
        sys.path.insert(0, a.ai8x)
    import safesound as S
    from eval_confusion import ai8x_valid_indices, collect_logits

    names = S.CLASSES
    k, bg = len(names), names.index("background")
    per_hour = 3600.0 / (a.hop_ms / 1000.0)
    print(f"체크포인트: {a.checkpoint}")

    # ── 운용 지점을 고를 데이터
    if a.pick_on == "val":
        # ai8x 가 train 에서 떼는 검증셋을 그대로 재현한다
        tr = S.SafeSound(os.path.join(a.data, "SafeSound"), "train",
                         transform=None, augment=False)
        vidx = ai8x_valid_indices(len(tr), a.validation_split, a.split_seed)
        print(f"검증셋 재현: train {len(tr):,} 중 {len(vidx):,}창 "
              f"(split {a.validation_split}, seed {a.split_seed})")
        del tr
        _f, _c, _s, y_pick, lg_pick = collect_logits(a, names, "train", vidx)
    else:
        y_pick = lg_pick = None

    # ── 테스트셋 (항상 본다)
    fsids, _c, _s, y_true, logits = collect_logits(a, names, "test")
    n_bg = int((y_true == bg).sum())
    print(f"테스트 창 {len(y_true):,} (배경음 {n_bg:,})  hop {a.hop_ms}ms "
          f"→ 시간당 {per_hour:,.0f}회 추론\n")

    offsets = np.arange(a.lo, a.hi + 1e-9, a.step)
    rows = sweep(logits, y_true, k, bg, offsets, per_hour)   # 곡선은 테스트 전체

    # ── 원본 단위 반분 (기본 모드)
    halves = None
    if a.pick_on == "test-half":
        A, B = half_split_by_origin(fsids, y_true, a.half_seed)
        print(f"=== 테스트셋 원본 단위 반분 (seed {a.half_seed}) ===")
        ma, mb = print_halves(fsids, y_true, A, B, names)
        halves = [("A→B", ma, mb), ("B→A", mb, ma)]
        y_pick = lg_pick = None
    elif y_pick is None:
        y_pick, lg_pick = y_true, logits

    if y_pick is not None:
        pick_rows = sweep(lg_pick, y_pick, k, bg, offsets, per_hour)
    else:
        pick_rows = rows                                     # 표 출력용 (아래서 재계산)

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

    # ── 운용 지점: 고르는 데이터와 보고하는 데이터를 분리한다
    applied = {}

    def one_direction(label, pick_mask, apply_mask):
        """pick_mask 에서 b 를 고르고 apply_mask 에 적용한 결과를 찍는다."""
        pr = sweep(lg_all[pick_mask], y_true[pick_mask], k, bg, offsets, per_hour)
        ar = {round(r["b"], 4): r
              for r in sweep(lg_all[apply_mask], y_true[apply_mask], k, bg,
                             offsets, per_hour)}
        print(f"\n  [{label}] b 는 앞쪽 반에서, 수치는 뒤쪽 반에서")
        print(f"  {'목표/h':>8}{'b':>7}{'고른 반/h':>11}{'적용 반/h':>11}   " +
              "".join(f"{n[:8]:>9}" for n in names))
        print("  " + "-" * (37 + 9 * k))
        for t, r in at_targets(pr, names).items():
            if r is None:
                print(f"  {t:>8.0f}{'—':>7}{'도달 불가':>11}{'—':>11}")
                continue
            ap_ = ar.get(round(r["b"], 4))
            rec = "".join(f"{100*v:>8.1f}%" for v in ap_["recall"])
            print(f"  {t:>8.0f}{r['b']:>7.2f}{r['per_hour']:>11,.0f}"
                  f"{ap_['per_hour']:>11,.0f}   {rec}")
            applied.setdefault(label, {})[t] = {
                "b": r["b"], "pick_per_hour": r["per_hour"],
                "apply_per_hour": ap_["per_hour"],
                "apply_recall": {n: round(float(v), 4)
                                 for n, v in zip(names, ap_["recall"])}}

    lg_all = logits
    if halves:
        print(f"\n=== 운용 지점 — **원본 단위로 분리**해 고르고 적용한다 ===")
        for label, pm, am in halves:
            one_direction(label, pm, am)
        print("\n  두 방향(A→B, B→A)이 크게 다르면 b 가 표본에 민감하다는 뜻이다 —")
        print("  그때는 어느 한 값을 운용 지점으로 확정하지 말 것.")
        print("  원본 단위로 갈랐으므로 같은 녹음이 양쪽에 걸치지 않는다. 다만 창")
        print("  단위 수치라 절대값이 아니라 상대 비교용이고, 논문 오경보는")
        print("  stream_eval.py 와 보드 8시간 구동에서 얻는다.")
    else:
        where = "ai8x 검증셋" if a.pick_on == "val" else "테스트셋(진단용)"
        hits = at_targets(pick_rows, names)
        by_b = {round(r["b"], 4): r for r in rows}
        print(f"\n=== 운용 지점 — b 는 **{where}**에서 고르고, 아래 수치는 "
              f"{'테스트셋' if a.pick_on == 'val' else '같은 데이터'} ===")
        print(f"{'목표/h':>8}{'b':>7}{'고른 곳/h':>11}{'테스트/h':>11}   " +
              "".join(f"{n[:8]:>9}" for n in names))
        print("-" * (37 + 9 * k))
        for t, r in hits.items():
            if r is None:
                print(f"{t:>8.0f}{'—':>7}{'도달 불가':>11}{'—':>11}   "
                      f"(b={a.hi} 까지 올려도 {pick_rows[-1]['per_hour']:,.0f}회)")
                continue
            te = by_b.get(round(r["b"], 4))
            rec = "".join(f"{100*v:>8.1f}%" for v in te["recall"])
            print(f"{t:>8.0f}{r['b']:>7.2f}{r['per_hour']:>11,.0f}"
                  f"{te['per_hour']:>11,.0f}   {rec}")
            applied[t] = {"b": r["b"], "pick_per_hour": r["per_hour"],
                          "test_per_hour": te["per_hour"],
                          "test_recall": {n: round(float(v), 4)
                                          for n, v in zip(names, te["recall"])}}

        if a.pick_on == "val":
            print("\n  ⚠️ **창 단위 분리라 낙관적이다.** ai8x 는 train 을 창 단위로")
            print("     쪼개므로, 검증 창의 대부분은 **같은 원본의 다른 창이 학습에")
            print("     들어가 있다.** 모델이 그 녹음을 이미 봤으니 여기서 고른 b 는")
            print("     처음 듣는 녹음에서보다 좋게 보인다.")
            print("     보고용으로는 `--pick-on test-half`(원본 단위 분리)를 쓸 것.")
        else:
            print("\n  ⚠️ **진단용, 낙관치다.** 같은 테스트셋에서 b 를 고르고 그 위에서")
            print("     보고했으므로 실제 운용 성능보다 좋게 나온다.")
            print("     보고하려면 `--pick-on test-half` 로 다시 돌릴 것.")

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
               "pick_on": a.pick_on, "operating_points": applied,
               "half_seed": a.half_seed if a.pick_on == "test-half" else None,
               # val 은 창 단위 분리라 낙관적이다 (같은 원본의 다른 창이 학습에 있다)
               "diagnostic_only": a.pick_on in ("test", "val"),
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
