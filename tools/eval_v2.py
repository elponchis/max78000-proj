#!/usr/bin/env python3
"""데이터셋 v2 (배경 혼합 증강) 채점 — 사전 등록 `dataset-v2-design.md` 8절·3절.

구성마다 v1 대 v2 (같은 시드 수):
  · 기존 시험셋 macro-F1 @300/h, 클래스별 recall·F1          (채택 조건 2)
  · 외부 세트 macro-F1 (v1 정의, 시험셋 @300/h 문턱값 고정)  (채택 조건 1)
  · v2 − v1 의 Welch 95% CI (시드 간)
  · 혼합 시험셋 (진단의 일반 배경 +10 / +5 dB) recall — 참고
  · 지름길 점검 3-(다): 배경 + 배경 혼합 창의 발화율
  · 순위 유지 3-(라)
채택: 5개 구성 중 3개 이상이 두 조건을 모두 충족. 클래스 경고: 기존 시험셋
recall 의 v2 − v1 CI 하한 < −5pp.

사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):  python tools/eval_v2.py
"""

import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]
import eval_external as E                                      # noqa: E402

# (이름, config, v1 접두, v2 접두, 시드 수)
PAIRS = [
    ("④", "wave", "safesound-v1cpu", "safesound-v2-wave", 5),
    ("가 u1000", "mel_u1000", "safesound-melu1000-v1", "safesound-v2-melu1000", 3),
    ("다 u800", "mel_u800", "safesound-melu800-v1", "safesound-v2-melu800", 3),
    ("간격 400", "mel_h400", "safesound-melh400-v1", "safesound-v2-melh400", 3),
    ("①′", "mel_inc", "safesound-melinc-v1", "safesound-v2-melinc", 3),
]
TEST = os.path.join(E.AI8X, "data")
DIAG = os.path.join(E.WORK, "diag")
MIXSETS = {"mix-gen-10": "일반 배경 +10 dB", "mix-gen-5": "일반 배경 +5 dB"}
WARN_PP = -5.0


def welch(a, b):
    """b − a 의 평균과 Welch 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return d, d, d
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    try:
        from scipy.stats import t
        q = float(t.ppf(0.975, df))
    except Exception:                     # scipy 없음 — t 분위수 근사표
        q = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306}.get(int(round(df)), 2.0 if df > 8 else 12.71)
    return d, d - q * se, d + q * se


def bgbg_root(seed=78000):
    """지름길 점검용: 시험셋 배경 창끼리 섞은 창 (SNR 0~20 균등). 한 번 만든다."""
    import diag_external_drop as D
    import safesound as S
    root = os.path.join(DIAG, "bgbg")
    if os.path.isfile(os.path.join(root, "SafeSound", "test", "background", "index.csv")):
        return root
    T = D.load_test()
    rows, idx = T["background"]
    usable = np.flatnonzero(D.zero_frac(D.crop(rows)) <= 0.5)
    rng = np.random.default_rng(seed)
    out = rows.copy()
    for k, i in enumerate(usable):
        j = int(rng.choice(usable))
        snr = float(rng.uniform(0, 20))
        out[i, S.MARGIN:S.MARGIN + S.WIN] = S.mix_int8(
            rows[i, S.MARGIN:S.MARGIN + S.WIN], rows[j, S.MARGIN:S.MARGIN + S.WIN], snr)
    D.write_cond("bgbg", {"background": (out[usable], [idx[i] for i in usable])})
    return root


def per_ckpt(config, ck, ext_root, bg_root, seg):
    """체크포인트 하나의 지표 묶음."""
    _c, yt, lt = E.logits_for(config, ck, TEST, "test")
    thr = E.thr_at(lt, yt == E.BG)
    p = E.predict(lt, thr)
    f1, rec, _ = E.metrics(yt, p)
    f1c = []
    for c in range(5):
        tp = float(((yt == c) & (p == c)).sum()); fp = float(((yt != c) & (p == c)).sum())
        fn = float(((yt == c) & (p != c)).sum())
        f1c.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
    r = {"thr": thr, "test_f1": f1, "test_recall": rec.tolist(), "test_f1_class": f1c}
    clips, _y, le = E.logits_for(config, ck, ext_root, "ext")
    assert np.array_equal(clips, seg)
    pe = E.predict(le, thr)
    r["ext_f1"], rece, r["ext_fa"] = E.metrics(E.YV[E.YV >= 0], pe[E.YV >= 0])
    r["ext_recall"] = rece.tolist()
    for m in MIXSETS:
        _c2, ym, lm = E.logits_for(config, ck, os.path.join(DIAG, m), "diag-" + m)
        r["mix_" + m] = E.metrics(ym, E.predict(lm, thr))[1][:4].tolist()
    _c3, yb, lb = E.logits_for(config, ck, bg_root, "diag-bgbg")
    r["bgbg_fire"] = float((E.predict(lb, thr) != E.BG).mean())
    r["bg_fire"] = float((p[yt == E.BG] != E.BG).mean())
    return r


def main():
    X, M = E.load_external()
    ext_root = E.build_shards(X, M)
    seg = np.array([r["segment_id"] for r in M])
    E.YV = E.labels(M, "v1")
    bg_root = bgbg_root()
    out = {"pairs": []}
    for name, config, p1, p2, n in PAIRS:
        try:
            cks = [(E.checkpoint(p1, s), E.checkpoint(p2, s)) for s in range(1, n + 1)]
        except SystemExit as e:
            print(f"  {name}: 건너뜀 — {e}")
            continue
        R1 = [per_ckpt(config, a, ext_root, bg_root, seg) for a, _b in cks]
        R2 = [per_ckpt(config, b, ext_root, bg_root, seg) for _a, b in cks]
        row = {"name": name, "config": config, "n_seeds": n, "v1": {}, "v2": {}, "delta": {}}
        for key in ("test_f1", "ext_f1", "ext_fa", "bgbg_fire", "bg_fire"):
            a = [r[key] for r in R1]; b = [r[key] for r in R2]
            row["v1"][key], row["v2"][key] = float(np.mean(a)), float(np.mean(b))
            row["delta"][key] = welch(a, b)
        for key in ("test_recall", "test_f1_class", "ext_recall", "mix_mix-gen-10", "mix_mix-gen-5"):
            k = len(R1[0][key])
            row["v1"][key] = np.nanmean([r[key] for r in R1], axis=0).tolist()
            row["v2"][key] = np.nanmean([r[key] for r in R2], axis=0).tolist()
            row["delta"][key] = [welch([r[key][c] for r in R1], [r[key][c] for r in R2])
                                 for c in range(k)]
        row["v1"]["test_f1_seeds"] = [r["test_f1"] for r in R1]
        row["v2"]["test_f1_seeds"] = [r["test_f1"] for r in R2]
        row["v1"]["ext_f1_seeds"] = [r["ext_f1"] for r in R1]
        row["v2"]["ext_f1_seeds"] = [r["ext_f1"] for r in R2]
        d_ext, lo_ext, _ = row["delta"]["ext_f1"]
        d_t, lo_t, _ = row["delta"]["test_f1"]
        row["cond_ext"] = bool(lo_ext > 0)
        row["cond_test"] = bool(lo_t >= -0.02)
        row["warn"] = [E.CLASSES[c] for c in range(4)
                       if 100 * row["delta"]["test_recall"][c][1] < WARN_PP]
        out["pairs"].append(row)
        print(f"  {name}: 시험셋 {row['v1']['test_f1']:.4f}→{row['v2']['test_f1']:.4f} "
              f"({d_t:+.4f} [{lo_t:+.4f}, {row['delta']['test_f1'][2]:+.4f}])  외부 "
              f"{row['v1']['ext_f1']:.4f}→{row['v2']['ext_f1']:.4f} ({d_ext:+.4f} "
              f"[{lo_ext:+.4f}, {row['delta']['ext_f1'][2]:+.4f}])  조건 "
              f"{'외부○' if row['cond_ext'] else '외부×'}/{'시험○' if row['cond_test'] else '시험×'}"
              f"  경고 {row['warn']}", flush=True)
    ok = sum(p["cond_ext"] and p["cond_test"] for p in out["pairs"])
    out["n_pass"], out["n_total"] = ok, len(out["pairs"])
    out["adopt"] = bool(ok >= 3)
    # 순위 (v1 시험셋 / v2 시험셋 / v2 외부)
    names = [p["name"] for p in out["pairs"]]
    out["rank"] = {"names": names,
                   "v1_test": E.rank_of([p["v1"]["test_f1"] for p in out["pairs"]]).tolist(),
                   "v2_test": E.rank_of([p["v2"]["test_f1"] for p in out["pairs"]]).tolist(),
                   "v2_ext": E.rank_of([p["v2"]["ext_f1"] for p in out["pairs"]]).tolist()}
    with open(os.path.join(E.WORK, "v2_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"두 조건 충족 {ok}/{len(out['pairs'])} → {'채택' if out['adopt'] else '기각'} "
          f"(5개 중 3개 이상이면 채택. ①′ 미완이면 보류)")


if __name__ == "__main__":
    main()
