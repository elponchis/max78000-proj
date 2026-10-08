#!/usr/bin/env python3
"""데이터셋 v3 채점 — 사전 등록 `docs/results/dataset-v3-design.md` 6절 (조건 A~E, "해결" 기준).

외부 라벨은 **v1eq** (siren = `Siren` + 하위 4종, 나머지 v1 과 같음; 사전 등록 1).
비교는 v3 − v2.1 **같은 시드(1~3)끼리** Welch 95% CI. v1 은 참고열.

  --baseline   v1·v2.1 체크포인트(캐시 로짓)를 v1eq 라벨로 다시 채점해 비교 기준을 만든다.
               v1 라벨과 나란히 출력 → data/external_eval/v3_baseline.json
  --v3         v3 체크포인트(safesound-v3-*)를 채점하고 조건 A~E·해결 기준을 판정한다.
               → data/external_eval/v3_eval.json. 아직 없는 구성은 건너뛴다.
  --v31        v3.1 (safesound-v31-*, dataset-v3.1-design.md): 외부를 **두 정의**로 채점 — 주 기준 실사용(real31:
               양성 real_primary, 음성 = 배경 + Yell·Shout 만 + def_gap), 보조 v1eq. 조건 A~E 는 주 기준으로,
               **F** = 시험셋 scream recall 점 추정 ≥ −5 pp (v2.1 대비) 추가. → v31_eval.json
  --train-by-source   v3 체크포인트로 **학습 창**(무증강)의 재현율을 출처별(v1 계열 / AudioSet)로 낸다
               (사전 등록 3). v3 학습 루트의 index.csv note 로 출처를 가른다.

지표 (외부, v1eq 라벨; 양성 siren 121·glass 74·scream 81·dog_bark 87, 배경 264):
  ext_f1       시험셋 @300/h 문턱값 고정 macro-F1           ext_recall  클래스별 recall (같은 문턱값)
  ext_fire_n   외부 배경 발화 창 수 (같은 문턱값; 1창 = 54.5회/h)
  ext_m5 / ext_m26   외부 배경 5창(@300/h) / 26창(10%) 에 맞춘 macro-F1 (보고만)
  ext_map      문턱값 무관 클래스별 AP 평균 (점수 = l_c − l_bg, 음성 = 외부 배경)
시험셋·배경+배경 발화율은 eval_v2.per_ckpt 그대로.

사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):  python tools/eval_v3.py --baseline
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]
import eval_external as E                                      # noqa: E402
import eval_v2 as V                                            # noqa: E402

# (이름, config, v1 접두, v1 시드, v2.1 접두, v2.1 시드, v3 접두, v3 시드)
PAIRS = [
    ("④", "wave", "safesound-v1cpu", 5, "safesound-v21-wave", 5, "safesound-v3-wave", 3),
    ("간격 400", "mel_h400", "safesound-melh400-v1", 3, "safesound-v21-melh400", 5, "safesound-v3-melh400", 3),
    ("가 u1000", "mel_u1000", "safesound-melu1000-v1", 3, "safesound-v21-melu1000", 3, "safesound-v3-melu1000", 3),
    ("다 u800", "mel_u800", "safesound-melu800-v1", 3, "safesound-v21-melu800", 3, "safesound-v3-melu800", 3),
    ("①′", "mel_inc", "safesound-melinc-v1", 3, "safesound-v21-melinc", 5, "safesound-v3-melinc", 3),
]
K = 3                                   # v3 − v2.1 비교 시드 수 (1~3)
COND_F_PP = -0.05                       # F (v3.1): 시험셋 scream recall 점 추정 ≥ −5 pp
COND_C_MIN = -0.02                      # C: 시험셋 F1 점 추정 ≥ −0.02
COND_D_WIN = 2.0                        # D: 외부 배경 발화 창 수 시드 평균 ≤ v2.1 + 2창
COND_E_BGBG = 0.026                     # E: 배경+배경 혼합 창 발화율 ≤ 2.6%
SOLVE_GAP = 0.5                         # 해결: 시험셋 F1 과의 격차를 절반 이상 닫음
SOLVE_PP = 0.10                         # 해결: siren·scream 외부 recall +10 pp
SOLVE_KEEP = -0.03                      # 해결: glass·dog_bark 외부 recall ≥ −3 pp
SOLVE_MAP = 0.04                        # 해결: 외부 mAP +0.04
OUT_BASE = os.path.join(E.WORK, "v3_baseline.json")
OUT_V3 = os.path.join(E.WORK, "v3_eval.json")
OUT_V31 = os.path.join(E.WORK, "v31_eval.json")
V3_ROOT = os.path.join(REPO, "data", "processed", "safesound_v3")


def ap_of(score, pos):
    o = np.argsort(-score)
    p = pos[o]
    tp = np.cumsum(p)
    prec = tp / np.arange(1, len(p) + 1)
    return float((prec * p).sum() / max(1.0, p.sum()))


def ext_extra(config, ck, ext_root, isbg, y):
    """외부 세트의 추가 지표: 맞춤 지점 둘, mAP, 발화 창 수는 per_ckpt 의 문턱값으로."""
    _c, _y, le = E.logits_for(config, ck, ext_root, "ext")
    sel = y >= 0
    r = {}
    for tag, target in (("ext_m5", E.TARGET), ("ext_m26", 0.10 * E.PER_HOUR)):
        thr = E.thr_at(le, isbg, target=target)
        r[tag] = E.metrics(y[sel], E.predict(le, thr)[sel])[0]
    aps = []
    for c in range(4):
        s = (y == c) | isbg
        aps.append(ap_of((le[:, c] - le[:, E.BG])[s], (y[s] == c).astype(float)))
    r["ext_map"] = float(np.mean(aps))
    r["ext_ap"] = aps
    return r


def score(config, ck, ext_root, bg_root, seg, isbg, y):
    r = V.per_ckpt(config, ck, ext_root, bg_root, seg)        # E.YV 는 호출 전에 v1eq 로 둔다
    _c, _y, le = E.logits_for(config, ck, ext_root, "ext")
    r["ext_fire_n"] = int((E.predict(le, r["thr"])[isbg] != E.BG).sum())
    r.update(ext_extra(config, ck, ext_root, isbg, y))
    return r


def seeds_of(prefix, n):
    return [E.checkpoint(prefix, s) for s in range(1, n + 1)]


def setup():
    X, M = E.load_external()
    ext_root = E.build_shards(X, M)
    seg = np.array([r["segment_id"] for r in M])
    isbg = np.array([r["group"] == "background" for r in M])
    return M, ext_root, seg, isbg, V.bgbg_root()


def fmt_ci(t):
    d, lo, hi = t
    return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def baseline():
    M, ext_root, seg, isbg, bg_root = setup()
    y_v1, y_eq = E.labels(M, "v1"), E.labels(M, "v1eq")
    print(f"외부 양성 v1 {[(c, int((y_v1 == i).sum())) for i, c in enumerate(E.EVENTS)]}  "
          f"v1eq {[(c, int((y_eq == i).sum())) for i, c in enumerate(E.EVENTS)]}  배경 {int(isbg.sum())}")
    out = {"label": "v1eq", "n_pos": {c: int((y_eq == i).sum()) for i, c in enumerate(E.EVENTS)}, "pairs": []}
    print("\n| 구성 (판, n) | 외부 F1 v1 라벨 → v1eq | siren recall v1 → v1eq (pp) | 외부 발화 창 | 맞춤 5창 | 맞춤 26창 | mAP |")
    print("|---|---|---|---:|---:|---:|---:|")
    for name, config, p1, n1, p21, n21, _p3, _n3 in PAIRS:
        row = {"name": name, "config": config, "seeds": {}}
        for tag, pre, n in (("v1", p1, n1), ("v21", p21, n21)):
            cks = seeds_of(pre, n)
            R_old, R_new = [], []
            for ck in cks:
                E.YV = y_v1
                R_old.append(V.per_ckpt(config, ck, ext_root, bg_root, seg))
                E.YV = y_eq
                R_new.append(score(config, ck, ext_root, bg_root, seg, isbg, y_eq))
            row["seeds"][tag] = {k: [r[k] for r in R_new] for k in
                                 ("test_f1", "ext_f1", "ext_recall", "ext_fire_n", "ext_m5", "ext_m26",
                                  "ext_map", "ext_ap", "bgbg_fire", "test_recall", "thr")}
            row["seeds"][tag]["ext_f1_v1label"] = [r["ext_f1"] for r in R_old]
            row["seeds"][tag]["ext_recall_v1label"] = [r["ext_recall"] for r in R_old]
            m = lambda k: float(np.mean([r[k] for r in R_new]))      # noqa: E731
            sr_old = 100 * np.mean([r["ext_recall"][0] for r in R_old])
            sr_new = 100 * np.mean([r["ext_recall"][0] for r in R_new])
            print(f"| {name} ({'v1' if tag == 'v1' else 'v2.1'}, {n}) | "
                  f"{np.mean([r['ext_f1'] for r in R_old]):.4f} → {m('ext_f1'):.4f} | "
                  f"{sr_old:.1f} → {sr_new:.1f} | {m('ext_fire_n'):.1f} | {m('ext_m5'):.4f} | "
                  f"{m('ext_m26'):.4f} | {m('ext_map'):.4f} |", flush=True)
        out["pairs"].append(row)
    with open(OUT_BASE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("저장:", OUT_BASE)


def judge(name, R21, R3, n21, cond_f=False):
    """조건 A~E 와 해결 기준. R21 은 v2.1 시드 1~K, R3 는 v3 시드 1~K. cond_f: v3.1 의 F 추가."""
    g = lambda R, k: [r[k] for r in R]                               # noqa: E731
    gc = lambda R, k, c: [r[k][c] for r in R]                       # noqa: E731
    d = {"A_ext_f1": V.welch(g(R21, "ext_f1"), g(R3, "ext_f1")),
         "B_ext_map": V.welch(g(R21, "ext_map"), g(R3, "ext_map")),
         "C_test_f1": V.welch(g(R21, "test_f1"), g(R3, "test_f1")),
         "D_fire_n": (float(np.mean(g(R3, "ext_fire_n"))), float(np.mean(g(R21, "ext_fire_n")))),
         "E_bgbg": float(np.mean(g(R3, "bgbg_fire"))),
         "ext_m5": V.welch(g(R21, "ext_m5"), g(R3, "ext_m5")),
         "ext_m26": V.welch(g(R21, "ext_m26"), g(R3, "ext_m26")),
         "ext_recall": [V.welch(gc(R21, "ext_recall", c), gc(R3, "ext_recall", c)) for c in range(4)],
         "test_recall": [V.welch(gc(R21, "test_recall", c), gc(R3, "test_recall", c)) for c in range(4)]}
    cond = {"A": bool(d["A_ext_f1"][1] > 0), "B": bool(d["B_ext_map"][1] > 0),
            "C": bool(d["C_test_f1"][0] >= COND_C_MIN),
            "D": bool(d["D_fire_n"][0] <= d["D_fire_n"][1] + COND_D_WIN),
            "E": bool(d["E_bgbg"] <= COND_E_BGBG)}
    if cond_f:
        cond["F"] = bool(d["test_recall"][2][0] >= COND_F_PP)
    test3, ext3 = float(np.mean(g(R3, "test_f1"))), float(np.mean(g(R3, "ext_f1")))
    test21, ext21 = float(np.mean(g(R21, "test_f1"))), float(np.mean(g(R21, "ext_f1")))
    gap21 = test21 - ext21
    solve = {"gap_half": bool(ext3 >= test21 - SOLVE_GAP * gap21),
             "scream_pp": bool(d["ext_recall"][2][0] >= SOLVE_PP),
             "siren_pp": bool(d["ext_recall"][0][0] >= SOLVE_PP),
             "glass_keep": bool(d["ext_recall"][1][0] >= SOLVE_KEEP),
             "dog_keep": bool(d["ext_recall"][3][0] >= SOLVE_KEEP),
             "map_up": bool(d["B_ext_map"][0] >= SOLVE_MAP)}
    return d, cond, solve, dict(test21=test21, ext21=ext21, test3=test3, ext3=ext3, gap21=gap21,
                                ext_target=test21 - SOLVE_GAP * gap21)


def v3(edition="v3"):
    M, ext_root, seg, isbg, bg_root = setup()
    main_def = "real31" if edition == "v31" else "v1eq"
    y_main = E.labels(M, main_def)
    y_eq = E.labels(M, "v1eq")
    out = {"label": main_def, "edition": edition, "n_pos": {c: int((y_main == i).sum()) for i, c in enumerate(E.EVENTS)},
           "n_neg": int((y_main == E.BG).sum()), "pairs": []}
    print(f"[{edition}] 주 기준 {main_def}: 양성 {out['n_pos']} 음성 {out['n_neg']} (v1eq 음성 {int((y_eq == E.BG).sum())})")
    for name, config, p1, n1, p21, n21, p3, n3 in PAIRS:
        if edition == "v31":
            p3 = p3.replace("safesound-v3-", "safesound-v31-")
        try:
            c3 = seeds_of(p3, n3)
        except SystemExit as e:
            print(f"  {name}: 건너뜀 — {e}")
            continue
        E.YV = y_main
        R3 = [score(config, ck, ext_root, bg_root, seg, isbg, y_main) for ck in c3]
        R21 = [score(config, ck, ext_root, bg_root, seg, isbg, y_main) for ck in seeds_of(p21, K)]
        R1 = [score(config, ck, ext_root, bg_root, seg, isbg, y_main) for ck in seeds_of(p1, min(n1, K))]
        d, cond, solve, s = judge(name, R21, R3, n21, cond_f=(edition == "v31"))
        if edition == "v31":                      # 보조: v1eq 정의로도 같은 표
            E.YV = y_eq
            R3e = [score(config, ck, ext_root, bg_root, seg, isbg, y_eq) for ck in c3]
            R21e = [score(config, ck, ext_root, bg_root, seg, isbg, y_eq) for ck in seeds_of(p21, K)]
            de, conde, _se, _ss = judge(name, R21e, R3e, n21)
            m2 = lambda R, k: float(np.mean([r[k] for r in R]))     # noqa: E731
            print(f"\n== {name} [보조 v1eq]  외부 F1 v2.1 {m2(R21e, 'ext_f1'):.4f} → v3.1 {m2(R3e, 'ext_f1'):.4f} ({fmt_ci(de['A_ext_f1'])}) "
                  f"mAP {m2(R21e, 'ext_map'):.4f} → {m2(R3e, 'ext_map'):.4f} ({fmt_ci(de['B_ext_map'])})  A {'○' if conde['A'] else '×'} B {'○' if conde['B'] else '×'}"
                  f"\n   recall v2.1 → v3.1 (pp): " + "  ".join(
                      f"{E.EVENTS[c]} {100 * np.mean([r['ext_recall'][c] for r in R21e]):.1f}→{100 * np.mean([r['ext_recall'][c] for r in R3e]):.1f} "
                      f"({100 * de['ext_recall'][c][0]:+.1f} [{100 * de['ext_recall'][c][1]:+.1f}, {100 * de['ext_recall'][c][2]:+.1f}])" for c in range(4)), flush=True)
        row = {"name": name, "config": config, "n3": n3,
               "seeds": {tag: {k: [r[k] for r in R] for k in
                               ("test_f1", "ext_f1", "ext_recall", "ext_fire_n", "ext_m5", "ext_m26",
                                "ext_map", "ext_ap", "bgbg_fire", "test_recall", "test_f1_class", "thr",
                                "mix_mix-gen-10")}
                         for tag, R in (("v1", R1), ("v21", R21), ("v3", R3))},
               "delta": {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()
                         if k not in ("ext_recall", "test_recall")},
               "delta_recall": {"ext": [list(t) for t in d["ext_recall"]], "test": [list(t) for t in d["test_recall"]]},
               "cond": cond, "pass_all": all(cond.values()), "solve": solve, "solve_all": all(solve.values()),
               "summary": s}
        if edition == "v31":
            row["v1eq"] = {"seeds": {tag: {k: [r[k] for r in R] for k in ("ext_f1", "ext_recall", "ext_map", "ext_m5", "ext_m26", "ext_fire_n")}
                                     for tag, R in (("v21", R21e), ("v3", R3e))},
                           "delta": {k: (list(v) if isinstance(v, tuple) else v) for k, v in de.items() if k not in ("ext_recall", "test_recall")},
                           "delta_recall_ext": [list(t) for t in de["ext_recall"]], "cond": conde}
        out["pairs"].append(row)
        m = lambda R, k: float(np.mean([r[k] for r in R]))          # noqa: E731
        print(f"\n== {name} (v3 {n3}시드, v2.1 시드 1~{K})"
              f"\n  시험셋 F1  v1 {m(R1, 'test_f1'):.4f} / v2.1 {m(R21, 'test_f1'):.4f} / v3 {m(R3, 'test_f1'):.4f}"
              f"   C {fmt_ci(d['C_test_f1'])} {'○' if cond['C'] else '×'}"
              f"\n  외부 F1    v1 {m(R1, 'ext_f1'):.4f} / v2.1 {m(R21, 'ext_f1'):.4f} / v3 {m(R3, 'ext_f1'):.4f}"
              f"   A {fmt_ci(d['A_ext_f1'])} {'○' if cond['A'] else '×'}"
              f"\n  외부 mAP   v1 {m(R1, 'ext_map'):.4f} / v2.1 {m(R21, 'ext_map'):.4f} / v3 {m(R3, 'ext_map'):.4f}"
              f"   B {fmt_ci(d['B_ext_map'])} {'○' if cond['B'] else '×'}"
              f"\n  외부 발화 창  v2.1 {d['D_fire_n'][1]:.1f} / v3 {d['D_fire_n'][0]:.1f}  D {'○' if cond['D'] else '×'}"
              f"   배경+배경 v3 {100 * d['E_bgbg']:.2f}%  E {'○' if cond['E'] else '×'}"
              f"\n  맞춤 5창 {m(R21, 'ext_m5'):.4f} → {m(R3, 'ext_m5'):.4f} ({fmt_ci(d['ext_m5'])})"
              f"   맞춤 26창 {m(R21, 'ext_m26'):.4f} → {m(R3, 'ext_m26'):.4f} ({fmt_ci(d['ext_m26'])})"
              f"\n  외부 recall v2.1 → v3 (pp): " + "  ".join(
                  f"{E.EVENTS[c]} {100 * np.mean([r['ext_recall'][c] for r in R21]):.1f}"
                  f"→{100 * np.mean([r['ext_recall'][c] for r in R3]):.1f} ({100 * d['ext_recall'][c][0]:+.1f} "
                  f"[{100 * d['ext_recall'][c][1]:+.1f}, {100 * d['ext_recall'][c][2]:+.1f}])" for c in range(4))
              + (f"\n  시험셋 recall v2.1 → v3 (pp): " + "  ".join(
                  f"{E.EVENTS[c]} {100 * np.mean([r['test_recall'][c] for r in R21]):.1f}→{100 * np.mean([r['test_recall'][c] for r in R3]):.1f} "
                  f"({100 * d['test_recall'][c][0]:+.1f} [{100 * d['test_recall'][c][1]:+.1f}, {100 * d['test_recall'][c][2]:+.1f}])" for c in range(4)))
              + f"\n  조건 {cond} → {'통과' if row['pass_all'] else '미통과'}"
              f"\n  해결 {solve} (외부 F1 목표 ≥ {s['ext_target']:.3f}) → {'해결' if row['solve_all'] else '미해결'}",
              flush=True)
    path = OUT_V31 if edition == "v31" else OUT_V3
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("저장:", path)


def train_by_source(edition="v3"):
    """학습 창(무증강)의 재현율을 출처별로. 루트: diag/<edition>train/SafeSound/test → safesound_<edition>/train 링크."""
    v3_root = V3_ROOT if edition == "v3" else os.path.join(REPO, "data", "processed", "safesound_" + edition)
    root = os.path.join(E.WORK, "diag", edition + "train")
    d = os.path.join(root, "SafeSound", "test")
    if not os.path.isdir(d):
        os.makedirs(os.path.join(root, "SafeSound"), exist_ok=True)
        os.symlink(os.path.join(v3_root, "train"), d)
    src = {}
    for cls in E.CLASSES:
        with open(os.path.join(v3_root, "train", cls, "index.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                src[r["clip_id"]] = "audioset" if r["note"].startswith("audioset") else "v1"
    print("| 구성 | 시드 | 출처 | " + " | ".join(E.CLASSES) + " |")
    print("|---|---:|---|" + "---:|" * 5)
    for name, config, _p1, _n1, _p21, _n21, p3, n3 in PAIRS:
        if edition == "v31":
            p3 = p3.replace("safesound-v3-", "safesound-v31-")
        try:
            cks = seeds_of(p3, n3)
        except SystemExit:
            continue
        acc = {s: [] for s in ("v1", "audioset")}
        for ck in cks:
            clips, y, lg = E.logits_for(config, ck, root, "diag-" + edition + "train")
            _c, yt, lt = E.logits_for(config, ck, V.TEST, "test")
            thr = E.thr_at(lt, yt == E.BG)
            p = E.predict(lg, thr)
            s_arr = np.array([src.get(c, "v1") for c in clips])
            for s in acc:
                sel = s_arr == s
                acc[s].append(E.metrics(y[sel], p[sel])[1])
        for s in acc:
            rec = 100 * np.nanmean(acc[s], axis=0)
            print(f"| {name} | {len(cks)} | {s} | " + " | ".join(f"{v:.1f}" for v in rec) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--v3", action="store_true")
    ap.add_argument("--v31", action="store_true")
    ap.add_argument("--train-by-source", action="store_true")
    a = ap.parse_args()
    if a.baseline:
        baseline()
    if a.v3:
        v3()
    if a.v31:
        v3("v31")
    if a.train_by_source:
        train_by_source("v31" if a.v31 else "v3")
    if not (a.baseline or a.v3 or a.v31 or a.train_by_source):
        ap.print_help()
