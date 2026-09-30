#!/usr/bin/env python3
"""`siren` 학습 곡선 — 원본 수 대 성능, **원본 단위 부트스트랩 95% CI 병기**.

`docs/results/siren-data-sufficiency.md` 1절의 실행 도구.

**왜 CI 가 필수인가**: 테스트 `siren` 원본이 **65개**뿐이다. 1회 값만으로는
25%→100% 의 상승이 실재인지 표본 변동인지 가를 수 없다. 신뢰구간은
**창이 아니라 원본 수**로 계산한다 (CLAUDE.md 5장 규칙 1 — 같은 녹음의
창들을 독립 표본으로 세면 구간이 실제보다 좁아진다).

**판정 규칙 (학습 전에 박는다)**

> **75% → 100% 구간의 상승이 CI 폭보다 크면 "데이터 추가 가치 있음".**

즉 100% 지점에서도 곡선의 기울기가 잡음을 넘으면 원본을 더 넣을 값어치가
있다는 뜻이다. 평평하면(상승 ≤ CI 폭) 원본 수가 병목이 아니고, 2·3·4번
(원본 확보·증강·재집계)이 무의미해진다.

⚠️ 25%→50%→75% 구간이 올라가는 것만으로는 부족하다 — 어떤 학습 곡선이든
초반에는 오른다. **끝 구간의 기울기**가 판정 대상이다.

사용 (WSL2):
    # 학습 (chain11 이 돌린다)
    # 평가
    python3 tools/siren_curve.py --json data/logs-local/siren-curve.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets"), os.path.join(REPO, "tools")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
SIREN, BG = 0, 4
POINTS = ((25, "safesound-mel-siren25-v1"), (50, "safesound-mel-siren50-v1"),
          (75, "safesound-mel-siren75-v1"), (100, "safesound-mel-v1"))


def boot_ci(fsids, y, pred, cls, n_boot=2000, seed=0):
    """**원본 단위** 부트스트랩으로 (recall, F1) 의 95% CI.

    원본을 복원추출하고, 뽑힌 원본에 속한 **창 전부**를 함께 가져온다.
    창을 직접 추출하면 같은 녹음의 창들을 독립으로 세어 구간이 좁아진다.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(fsids)
    idx_by = {f: np.flatnonzero(fsids == f) for f in uniq}
    rec, f1 = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        sel = np.concatenate([idx_by[uniq[i]] for i in pick])
        yy, pp = y[sel], pred[sel]
        tp = int(((yy == cls) & (pp == cls)).sum())
        fp = int(((yy != cls) & (pp == cls)).sum())
        fn = int(((yy == cls) & (pp != cls)).sum())
        rec.append(tp / (tp + fn) if (tp + fn) else np.nan)
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
    q = lambda v: (float(np.nanpercentile(v, 2.5)),
                   float(np.nanpercentile(v, 97.5)))
    return q(rec), q(f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/logs-local")
    ap.add_argument("--data", default=os.path.join(AI8X, "data"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import eval_confusion as EC
    from eval_threshold import margin_of, margin_predict

    rows = []
    for frac, base in POINTS:
        d = sorted(glob.glob(os.path.join(a.logs, base + "___*")),
                   key=os.path.getmtime)
        ck = None
        if d:
            ck = os.path.join(d[-1], os.path.basename(d[-1]).split("___")[0]
                              + "_qat_best.pth.tar")
        if frac == 100 and (not ck or not os.path.isfile(ck)):
            ck = "data/safesound-mel-v1_qat_best.pth.tar"   # Colab 에서 받은 것
        if not ck or not os.path.isfile(ck):
            print(f"  [대기] {frac}% — 체크포인트 없음")
            continue
        args = argparse.Namespace(
            checkpoint=ck, config="mel", data=a.data, ai8x=AI8X,
            simulate=False, bias=False, batch_size=128, split="test")
        import contextlib
        import io as _io
        with contextlib.redirect_stdout(_io.StringIO()):
            fs, _cl, _st, y, lg = EC.collect_logits(args, CLASSES, "test")
        fx = EC.fixed_fa_points(lg, y, CLASSES, hop_ms=250)
        t = fx["targets"]["300/h"]
        # @300/h 동작점에서의 예측으로 siren F1 을 낸다 (argmax 가 아니다)
        marg_ev, ev_best = margin_of(lg, BG)
        pred = margin_predict(marg_ev, ev_best, lg[:, BG], BG, t["thr"])
        (rlo, rhi), (flo, fhi) = boot_ci(fs, y, pred, SIREN, a.n_boot)
        tp = int(((y == SIREN) & (pred == SIREN)).sum())
        fp = int(((y != SIREN) & (pred == SIREN)).sum())
        fn = int(((y == SIREN) & (pred != SIREN)).sum())
        rows.append({
            "frac": frac, "ckpt": ck,
            "siren_recall": tp / (tp + fn), "recall_ci": [rlo, rhi],
            "siren_f1": 2 * tp / (2 * tp + fp + fn), "f1_ci": [flo, fhi],
            "macro_f1": t["macro_f1"],
            "n_origin": int(len(np.unique(fs[y == SIREN]))),
        })

    if not rows:
        sys.exit("[에러] 평가할 체크포인트가 없다")

    print(f"{'siren train %':<14}{'원본':>6}{'recall @300/h':>26}"
          f"{'F1 @300/h':>26}{'macro-F1':>10}")
    print("-" * 84)
    for r in rows:
        print(f"{r['frac']:>12}% {r['n_origin']:>6}"
              f"{100*r['siren_recall']:>13.1f}% "
              f"[{100*r['recall_ci'][0]:>5.1f},{100*r['recall_ci'][1]:>5.1f}]"
              f"{100*r['siren_f1']:>13.1f}% "
              f"[{100*r['f1_ci'][0]:>5.1f},{100*r['f1_ci'][1]:>5.1f}]"
              f"{r['macro_f1']:>10.4f}")

    print()
    print("=== 판정 (학습 전에 정한 규칙) ===")
    by = {r["frac"]: r for r in rows}
    if 75 in by and 100 in by:
        d_rec = by[100]["siren_recall"] - by[75]["siren_recall"]
        d_f1 = by[100]["siren_f1"] - by[75]["siren_f1"]
        w_rec = by[100]["recall_ci"][1] - by[100]["recall_ci"][0]
        w_f1 = by[100]["f1_ci"][1] - by[100]["f1_ci"][0]
        print(f"  75% → 100%  recall {100*d_rec:+.1f}pp   "
              f"CI 폭 {100*w_rec:.1f}pp   "
              f"{'**데이터 추가 가치 있음**' if d_rec > w_rec else '평평 (가치 미확인)'}")
        print(f"              F1     {100*d_f1:+.1f}pp   "
              f"CI 폭 {100*w_f1:.1f}pp   "
              f"{'**데이터 추가 가치 있음**' if d_f1 > w_f1 else '평평 (가치 미확인)'}")
        print()
        print("  ⚠️ 25→50→75 구간이 오르는 것만으로는 부족하다 — 어떤 학습")
        print("     곡선이든 초반에는 오른다. **끝 구간의 기울기**가 판정이다.")
        print("  ⚠️ CI 는 **원본 단위** 부트스트랩이다 (테스트 siren 원본 65개).")
        print("     창 단위로 내면 같은 녹음의 창을 독립으로 세어 좁아진다.")
    else:
        print("  75%/100% 두 점이 모두 있어야 판정할 수 있다")

    if a.json:
        json.dump(rows, open(a.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n저장: {a.json}")


if __name__ == "__main__":
    main()
