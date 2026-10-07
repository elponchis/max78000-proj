#!/usr/bin/env python3
"""데이터셋 v2.1 채점 — 사전 등록 `dataset-v2.1-design.md` 2·5절.

구성마다 v1 / v2 / v2.1 을 같은 시드끼리 비교한다 (v2.1 − v1 이 판정 기준,
v2.1 − v2 는 참고). 조건 넷:
  1 외부 F1 (시험셋 @300/h 문턱값 고정)  v2.1 − v1 CI 하한 > 0
  2 시험셋 F1 @300/h                      v2.1 − v1 CI 하한 ≥ −0.02
  3 외부 배경 기준 @300/h 외부 F1          v2.1 − v1 CI 하한 ≥ −0.02 그리고 점 추정 ≥ 0
  4 배경+배경 혼합 창 발화율 (시험셋 문턱값) 시드 평균 ≤ 2.6 %
채택: 돌린 구성의 과반이 넷 다 충족. 아직 없는 구성은 건너뛴다.
5시드 구성의 시드 4·5 는 v2.1 안의 구성 간 비교(5-d)에만 쓴다.

사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):  python tools/eval_v21.py
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]
import eval_external as E                                      # noqa: E402
import eval_v2 as V                                            # noqa: E402

# (이름, config, v1 접두, v2 접두, v2.1 접두, v1·v2 시드, v2.1 시드)
PAIRS = [
    ("④", "wave", "safesound-v1cpu", "safesound-v2-wave", "safesound-v21-wave", 5, 5),
    ("간격 400", "mel_h400", "safesound-melh400-v1", "safesound-v2-melh400",
     "safesound-v21-melh400", 3, 5),
    ("가 u1000", "mel_u1000", "safesound-melu1000-v1", "safesound-v2-melu1000",
     "safesound-v21-melu1000", 3, 3),
    ("다 u800", "mel_u800", "safesound-melu800-v1", "safesound-v2-melu800",
     "safesound-v21-melu800", 3, 3),
    ("①′", "mel_inc", "safesound-melinc-v1", "safesound-v2-melinc",
     "safesound-v21-melinc", 3, 5),
]
BGBG_MAX = 0.026


def ext_matched(config, ck, ext_root, isbg, yv):
    """외부 배경 기준 @300/h 문턱값에서의 외부 macro-F1."""
    _c, _y, le = E.logits_for(config, ck, ext_root, "ext")
    thr = E.thr_at(le, isbg)
    return E.metrics(yv[yv >= 0], E.predict(le, thr)[yv >= 0])[0]


def main():
    X, M = E.load_external()
    ext_root = E.build_shards(X, M)
    seg = np.array([r["segment_id"] for r in M])
    E.YV = E.labels(M, "v1")
    isbg = np.array([r["group"] == "background" for r in M])
    bg_root = V.bgbg_root()
    out = {"pairs": []}
    for name, config, p1, p2, p21, n_old, n_new in PAIRS:
        try:
            c21 = [E.checkpoint(p21, s) for s in range(1, n_new + 1)]
        except SystemExit as e:
            print(f"  {name}: 건너뜀 — {e}")
            continue
        c1 = [E.checkpoint(p1, s) for s in range(1, n_old + 1)]
        c2 = [E.checkpoint(p2, s) for s in range(1, n_old + 1)]
        R = {}
        for tag, cks in (("v1", c1), ("v2", c2), ("v21", c21)):
            rs = [V.per_ckpt(config, ck, ext_root, bg_root, seg) for ck in cks]
            for r, ck in zip(rs, cks):
                r["ext_matched"] = ext_matched(config, ck, ext_root, isbg, E.YV)
            R[tag] = rs
        k = n_old                         # v1·v2 비교는 같은 시드(1..n_old)끼리
        row = {"name": name, "config": config, "n_old": n_old, "n_new": n_new,
               "seeds": {}, "mean": {}, "delta_v1": {}, "delta_v2": {}, "class": {}}
        for key in ("test_f1", "ext_f1", "ext_matched", "ext_fa", "bgbg_fire"):
            for tag in R:
                row["seeds"].setdefault(tag, {})[key] = [r[key] for r in R[tag]]
                row["mean"].setdefault(tag, {})[key] = float(np.mean([r[key] for r in R[tag]]))
            a21 = [r[key] for r in R["v21"][:k]]
            row["delta_v1"][key] = V.welch([r[key] for r in R["v1"]], a21)
            row["delta_v2"][key] = V.welch([r[key] for r in R["v2"]], a21)
        for key in ("test_recall", "test_f1_class", "ext_recall", "mix_mix-gen-10"):
            for tag in R:
                row["class"].setdefault(tag, {})[key] = np.nanmean(
                    [r[key] for r in R[tag]], axis=0).tolist()
            row["class"].setdefault("delta_v1", {})[key] = [
                V.welch([r[key][c] for r in R["v1"]], [r[key][c] for r in R["v21"][:k]])
                for c in range(4)]
        d1, lo1, _ = row["delta_v1"]["ext_f1"]
        d2, lo2, _ = row["delta_v1"]["test_f1"]
        d3, lo3, _ = row["delta_v1"]["ext_matched"]
        bgbg = float(np.mean([r["bgbg_fire"] for r in R["v21"][:k]]))
        row["cond"] = {"1_ext": bool(lo1 > 0), "2_test": bool(lo2 >= -0.02),
                       "3_matched": bool(lo3 >= -0.02 and d3 >= 0), "4_bgbg": bool(bgbg <= BGBG_MAX)}
        row["pass_all"] = all(row["cond"].values())
        row["warn"] = [E.CLASSES[c] for c in range(4)
                       if 100 * row["class"]["delta_v1"]["test_recall"][c][1] < -5]
        out["pairs"].append(row)
        m = row["mean"]
        print(f"  {name}: 시험셋 v1 {m['v1']['test_f1']:.4f} / v2 {m['v2']['test_f1']:.4f} / "
              f"v2.1 {m['v21']['test_f1']:.4f} (v2.1−v1 {d2:+.4f} [{lo2:+.4f}, {row['delta_v1']['test_f1'][2]:+.4f}])"
              f"\n      외부 {m['v1']['ext_f1']:.4f} / {m['v2']['ext_f1']:.4f} / {m['v21']['ext_f1']:.4f} "
              f"({d1:+.4f} [{lo1:+.4f}, {row['delta_v1']['ext_f1'][2]:+.4f}])  외부 맞춤 "
              f"{m['v1']['ext_matched']:.4f} / {m['v2']['ext_matched']:.4f} / {m['v21']['ext_matched']:.4f} "
              f"({d3:+.4f} [{lo3:+.4f}, {row['delta_v1']['ext_matched'][2]:+.4f}])"
              f"\n      외부 오경보/h {m['v1']['ext_fa']:.0f} / {m['v2']['ext_fa']:.0f} / {m['v21']['ext_fa']:.0f}  "
              f"배경+배경 발화 {100*m['v1']['bgbg_fire']:.2f} / {100*m['v2']['bgbg_fire']:.2f} / "
              f"{100*bgbg:.2f}%  조건 {row['cond']}  경고 {row['warn']}", flush=True)
    ok = sum(p["pass_all"] for p in out["pairs"])
    out["n_pass"], out["n_total"] = ok, len(out["pairs"])
    out["adopt"] = bool(ok * 2 > len(out["pairs"]))
    with open(os.path.join(E.WORK, "v21_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"네 조건 충족 {ok}/{len(out['pairs'])} → {'채택' if out['adopt'] else '기각'} "
          f"(돌린 구성의 과반. 미완 구성 제외)")


if __name__ == "__main__":
    main()
