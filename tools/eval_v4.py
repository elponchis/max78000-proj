#!/usr/bin/env python3
"""청취 확정 정답으로 전 판을 한꺼번에 채점 — `v4-distill-design.md` 10.3·10.4, `eval-baseline-change.md` 4절.

정답: 시험셋 **testL** (glass 창 중 청취 X·? 제외), 외부 **real31L** (주) / **v1eqL** (보조) — 정의별 양성 위에
청취 판정을 겹친 것 (`listened_labels.ext_overlay`). 옛 정답(test / real31 / v1eq)도 나란히 낸다.
외부 유리는 B 첫 판정으로도 채점한다 (`*_Bfirst`, 민감도 — 조건·순위가 바뀌는 곳을 ★ 로 표시).
문턱값은 **기존 시험셋 배경 @300/h** 그대로 (배경은 바뀌지 않으므로 불변). 창 제외는 채점 단계에서만.

  --score        v1·v2.1·v3.1·v4·v4+D·v4+D½ 채점 + 비교(v3.1−v2.1, v4−v3.1, v4+D−v4, v4+D−v4+D½) 조건 A~G
                 → data/external_eval/v4_eval.json
  --panns        PANNs CNN14 zero-shot 기준선을 같은 정답으로 (외부 @300/h 두 지점, 시험셋 @300/h)
  --paper-glass  논문 표(v1 9구성, v2.1 5구성)의 유리 관련 수치가 새 정답에서 얼마나 바뀌는지

사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):  python tools/eval_v4.py --score --panns --paper-glass
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]
import eval_external as E                                      # noqa: E402
import eval_v2 as V                                            # noqa: E402
import eval_v3 as V3                                           # noqa: E402
import listened_labels as LL                                   # noqa: E402

K = 3
COND_G_PP = -0.05                       # G: 시험셋 glass recall 점 추정 ≥ −5 pp
HALF_GAIN = 0.02                        # v4+D − v4+D½: 점 추정 ≥ +0.02 이고 CI 하한 > −0.02
HALF_LO = -0.02
OUT = os.path.join(E.WORK, "v4_eval.json")
# (표시 이름, eval_confusion 구성 키). 가·다·①′ 는 v4+D 에만 있다 (chain29, 2026-10-11 결정).
CONFIGS = [("④", "wave"), ("간격 400", "mel_h400"), ("가 u1000", "mel_u1000"), ("다 u800", "mel_u800"), ("①′", "mel_inc"),
           ("(a) NPU 로그 근사", "wave_logstage")]          # (a) 는 다섯 구성 표의 **별도 행** (chain30) — 순위에는 넣지 않는다
EXTRA = {"(a) NPU 로그 근사"}
BIAS_CONFIGS = {"wave_logstage"}                           # --use-bias 로 학습한 구성
# 판 → (④, 간격 400, 가, 다, ①′, (a) 접두). 시드 1~3. None = 그 판에 없는 구성.
EDITIONS = [("v1", "safesound-v1cpu", "safesound-melh400-v1", None, None, None, "safesound-logstage-v1"),
            ("v2.1", "safesound-v21-wave", "safesound-v21-melh400", None, None, None, None),
            ("v3.1", "safesound-v31-wave", "safesound-v31-melh400", None, None, None, None),
            ("v4", "safesound-v4-wave", "safesound-v4-melh400", None, None, None, None),
            ("v4+D", "safesound-v4d-wave", "safesound-v4d-melh400", "safesound-v4d-melu1000", "safesound-v4d-melu800", "safesound-v4d-melinc", "safesound-v4d-logstage"),
            ("v4+D½", None, "safesound-v4dh-melh400", None, None, None, None)]
EQUIV = 0.02                            # 재판정 "동등": (변형 − ①′) CI 가 [−0.02, +0.02] 안에 완전히 (curve-design.md)
COMPARE = [("v3.1 − v2.1", "v2.1", "v3.1"), ("v4 − v3.1", "v3.1", "v4"), ("v4+D − v4", "v4", "v4+D")]
# *L = 청취 주 정답(B 재청취), *L_Bfirst = 외부 유리만 B 첫 판정으로 바꾼 민감도 정답 (label-audit.md 7.3)
LABEL_SETS = ("real31", "real31L", "real31L_Bfirst", "v1eq", "v1eqL", "v1eqL_Bfirst")
# 비교 판정에 쓰는 (태그, 외부 정의, 시험셋 태그). 첫 줄이 사전 등록 주 기준.
JUDGE_TAGS = (("L", "real31L", "L"), ("L_Bfirst", "real31L_Bfirst", "L"), ("old", "real31", "old"), ("v1eqL", "v1eqL", "L"))


def class_f1(y, p):
    out = []
    for c in range(5):
        tp = float(((y == c) & (p == c)).sum()); fp = float(((y != c) & (p == c)).sum())
        fn = float(((y == c) & (p != c)).sum())
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
    return out


def setup():
    X, M = E.load_external()
    ext_root = E.build_shards(X, M)
    seg = np.array([r["segment_id"] for r in M])
    isbg = np.array([r["group"] == "background" for r in M])
    rows = LL.load()
    rows_first = LL.load(b_first=True)
    Y, st = {}, {}
    for d in ("real31", "v1eq"):
        Y[d] = E.labels(M, d)
        Y[d + "L"], st[d + "L"] = LL.ext_overlay(M, Y[d], rows)
        Y[d + "L_Bfirst"], st[d + "L_Bfirst"] = LL.ext_overlay(M, Y[d], rows_first)
        # 비명 정답은 두 정답에서 같아야 한다 (B 는 유리만 바꾼다)
        assert np.array_equal((Y[d + "L"] == 2), (Y[d + "L_Bfirst"] == 2))
    # 시험셋 순서 = 로짓 캐시 순서 (= ceiling 캐시 순서). 창 시작 샘플은 ceiling 캐시에서.
    zt = np.load(os.path.join(REPO, "data", "ceiling-panns-zs.npz"), allow_pickle=True)
    keep_t, nt = LL.test_keep(rows, zt["clips"], zt["starts"], zt["y"])
    return dict(M=M, ext_root=ext_root, seg=seg, isbg=isbg, rows=rows, Y=Y, overlay=st, keep_t=keep_t, test_glass=nt,
                test_clips=zt["clips"], test_y=zt["y"], bg_root=V.bgbg_root())


def score_ckpt(config, ck, S):
    """체크포인트 하나 — 옛 정답과 새 정답 모두."""
    bias = config in BIAS_CONFIGS
    _c, yt, lt = E.logits_for(config, ck, V.TEST, "test", bias=bias)
    assert np.array_equal(_c.astype(str), S["test_clips"].astype(str)) and np.array_equal(yt, S["test_y"])
    thr = E.thr_at(lt, yt == E.BG)
    p = E.predict(lt, thr)
    r = {"thr": thr}
    for tag, keep in (("old", np.ones(len(yt), bool)), ("L", S["keep_t"])):
        f1, rec, _ = E.metrics(yt[keep], p[keep])
        r[f"test_f1_{tag}"] = f1; r[f"test_recall_{tag}"] = rec.tolist(); r[f"test_f1_class_{tag}"] = class_f1(yt[keep], p[keep])
    clips, _y, le = E.logits_for(config, ck, S["ext_root"], "ext", bias=bias)
    assert np.array_equal(clips, S["seg"])
    pe = E.predict(le, thr)
    isbg = S["isbg"]
    r["ext_fire_n"] = int((pe[isbg] != E.BG).sum())
    for d, y in S["Y"].items():
        sel = y >= 0
        f1, rec, _ = E.metrics(y[sel], pe[sel])
        r[f"ext_f1_{d}"] = f1; r[f"ext_recall_{d}"] = rec.tolist()
        for tag, target in (("m5", E.TARGET), ("m26", 0.10 * E.PER_HOUR)):
            t2 = E.thr_at(le, isbg, target=target)
            r[f"ext_{tag}_{d}"] = E.metrics(y[sel], E.predict(le, t2)[sel])[0]
        aps = []
        for c in range(4):
            s = (y == c) | isbg
            aps.append(V3.ap_of((le[:, c] - le[:, E.BG])[s], (y[s] == c).astype(float)))
        r[f"ext_map_{d}"] = float(np.mean(aps)); r[f"ext_ap_{d}"] = aps
    _c3, yb, lb = E.logits_for(config, ck, S["bg_root"], "diag-bgbg", bias=bias)
    r["bgbg_fire"] = float((E.predict(lb, thr) != E.BG).mean())
    return r


def view(r, ext_def, test_tag):
    """judge() 가 기대하는 이름으로 (ext_f1, ext_map, ext_m5, ext_m26, ext_recall, ext_fire_n, test_f1, test_recall, bgbg_fire)."""
    return {"ext_f1": r[f"ext_f1_{ext_def}"], "ext_map": r[f"ext_map_{ext_def}"], "ext_m5": r[f"ext_m5_{ext_def}"],
            "ext_m26": r[f"ext_m26_{ext_def}"], "ext_recall": r[f"ext_recall_{ext_def}"], "ext_fire_n": r["ext_fire_n"],
            "test_f1": r[f"test_f1_{test_tag}"], "test_recall": r[f"test_recall_{test_tag}"], "bgbg_fire": r["bgbg_fire"]}


def judge_g(name, Rb, Rn, ext_def, test_tag):
    d, cond, solve, s = V3.judge(name, [view(r, ext_def, test_tag) for r in Rb], [view(r, ext_def, test_tag) for r in Rn],
                                 len(Rb), cond_f=True)
    cond["G"] = bool(d["test_recall"][1][0] >= COND_G_PP)
    return d, cond, solve, s


def fmt(t):
    return V3.fmt_ci(t)


def seeds_of(prefix, n=K):
    cks = []
    for s in range(1, n + 1):
        try:
            cks.append(E.checkpoint(prefix, s))
        except SystemExit:
            break
    return cks


def m(R, k, c=None):
    return float(np.mean([r[k] if c is None else r[k][c] for r in R]))


def score_all(S):
    out = {"labels": {"ext_n_pos": {d: {c: int((y == i).sum()) for i, c in enumerate(E.EVENTS)} for d, y in S["Y"].items()},
                      "ext_n_neg": {d: int((y == E.BG).sum()) for d, y in S["Y"].items()},
                      "overlay": S["overlay"], "test_glass": S["test_glass"],
                      "test_n": {"old": int(len(S["keep_t"])), "L": int(S["keep_t"].sum())}},
           "editions": {}, "compare": []}
    print("\n외부 양성:", out["labels"]["ext_n_pos"], " 음성:", out["labels"]["ext_n_neg"])
    print("겹침 통계:", S["overlay"], " 시험셋 glass:", S["test_glass"], " 시험셋 창:", out["labels"]["test_n"])
    R = {}
    for ed, *pres in EDITIONS:
        out["editions"][ed] = {}
        for (cname, config), pre in zip(CONFIGS, pres):
            if pre is None:
                continue
            cks = seeds_of(pre)
            if not cks:
                print(f"  {ed} {cname}: 체크포인트 없음 — 건너뜀"); continue
            rs = [score_ckpt(config, ck, S) for ck in cks]
            R[(ed, cname)] = rs
            out["editions"][ed][cname] = {"prefix": pre, "n": len(rs), "seeds": rs}
            print(f"  {ed} {cname} ({len(rs)}시드): 시험셋 F1 {m(rs, 'test_f1_old'):.4f} → {m(rs, 'test_f1_L'):.4f}"
                  f" | glass recall {100 * m(rs, 'test_recall_old', 1):.1f} → {100 * m(rs, 'test_recall_L', 1):.1f}"
                  f" | 외부 F1 real31 {m(rs, 'ext_f1_real31'):.4f} → real31L {m(rs, 'ext_f1_real31L'):.4f}"
                  f" | v1eq {m(rs, 'ext_f1_v1eq'):.4f} → v1eqL {m(rs, 'ext_f1_v1eqL'):.4f}"
                  f" | mAP real31L {m(rs, 'ext_map_real31L'):.4f} | 발화 {m(rs, 'ext_fire_n'):.1f}", flush=True)

    # ── 표 1: 판·구성별 옛 정답 → 새 정답
    print("\n| 판 | 구성 | n | 시험셋 F1 old → testL | 시험셋 glass recall old → L | 시험셋 glass F1 old → L | 외부 F1 real31 → real31L | 외부 F1 v1eq → v1eqL | 외부 glass recall real31 → L | 외부 scream recall real31 → L | 외부 mAP real31 → L | 외부 발화 창 |")
    print("|---|---|---:|---|---|---|---|---|---|---|---|---:|")
    for ed, *_p in EDITIONS:
        for cname, _cfg in CONFIGS:
            rs = R.get((ed, cname))
            if not rs:
                continue
            print(f"| {ed} | {cname} | {len(rs)} | {m(rs, 'test_f1_old'):.4f} → {m(rs, 'test_f1_L'):.4f} | "
                  f"{100 * m(rs, 'test_recall_old', 1):.1f} → {100 * m(rs, 'test_recall_L', 1):.1f} | "
                  f"{m(rs, 'test_f1_class_old', 1):.3f} → {m(rs, 'test_f1_class_L', 1):.3f} | "
                  f"{m(rs, 'ext_f1_real31'):.4f} → {m(rs, 'ext_f1_real31L'):.4f} | {m(rs, 'ext_f1_v1eq'):.4f} → {m(rs, 'ext_f1_v1eqL'):.4f} | "
                  f"{100 * m(rs, 'ext_recall_real31', 1):.1f} → {100 * m(rs, 'ext_recall_real31L', 1):.1f} | "
                  f"{100 * m(rs, 'ext_recall_real31', 2):.1f} → {100 * m(rs, 'ext_recall_real31L', 2):.1f} | "
                  f"{m(rs, 'ext_map_real31'):.4f} → {m(rs, 'ext_map_real31L'):.4f} | {m(rs, 'ext_fire_n'):.1f} |")

    # ── 표 1b: 외부 유리가 들어가는 지표 — B 재청취(주) 와 B 첫 판정(민감도) 나란히
    print("\n| 판 | 구성 | n | 외부 F1 real31L (주) / B첫 | 외부 mAP real31L (주) / B첫 | 외부 glass recall L (주) / B첫 | 외부 glass AP L (주) / B첫 | 맞춤 5창 F1 L (주) / B첫 | 외부 F1 v1eqL (주) / B첫 |")
    print("|---|---|---:|---|---|---|---|---|---|")
    for ed, *_p in EDITIONS:
        for cname, _cfg in CONFIGS:
            rs = R.get((ed, cname))
            if not rs:
                continue
            print(f"| {ed} | {cname} | {len(rs)} | {m(rs, 'ext_f1_real31L'):.4f} / {m(rs, 'ext_f1_real31L_Bfirst'):.4f} | "
                  f"{m(rs, 'ext_map_real31L'):.4f} / {m(rs, 'ext_map_real31L_Bfirst'):.4f} | "
                  f"{100 * m(rs, 'ext_recall_real31L', 1):.1f} / {100 * m(rs, 'ext_recall_real31L_Bfirst', 1):.1f} | "
                  f"{m(rs, 'ext_ap_real31L', 1):.3f} / {m(rs, 'ext_ap_real31L_Bfirst', 1):.3f} | "
                  f"{m(rs, 'ext_m5_real31L'):.4f} / {m(rs, 'ext_m5_real31L_Bfirst'):.4f} | "
                  f"{m(rs, 'ext_f1_v1eqL'):.4f} / {m(rs, 'ext_f1_v1eqL_Bfirst'):.4f} |")

    # ── 구성 순위: 외부 F1·mAP(real31L) 로 판을 정렬 — 두 정답에서 순위가 다른 곳 표시
    out["ranking"] = {}
    for cname, _cfg in CONFIGS:
        eds = [ed for ed, *_p in EDITIONS if (ed, cname) in R]
        if len(eds) < 2:
            continue
        for key in ("ext_f1", "ext_map"):
            rk = {tag: sorted(eds, key=lambda e: -m(R[(e, cname)], f"{key}_real31{tag}")) for tag in ("L", "L_Bfirst")}
            same = rk["L"] == rk["L_Bfirst"]
            out["ranking"][f"{cname}/{key}"] = {"L": rk["L"], "L_Bfirst": rk["L_Bfirst"], "same": same}
            print(f"  순위 {cname} {key}: 주 {' > '.join(rk['L'])}  |  B첫 {' > '.join(rk['L_Bfirst'])}  {'(같음)' if same else '(★ 순위 다름)'}")

    # ── 비교: 주 기준 real31L + testL (사전 등록), B 첫 판정 민감도, 참고로 옛 정답·v1eqL
    for label, base, new in COMPARE:
        for cname, _cfg in CONFIGS:
            Rb, Rn = R.get((base, cname)), R.get((new, cname))
            if not Rb or not Rn:
                continue
            res = {"name": label, "config": cname, "n_base": len(Rb), "n_new": len(Rn)}
            for tag, ext_def, test_tag in JUDGE_TAGS:
                d, cond, solve, s = judge_g(label, Rb, Rn, ext_def, test_tag)
                res[tag] = {"delta": {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items() if k not in ("ext_recall", "test_recall")},
                            "delta_recall": {"ext": [list(t) for t in d["ext_recall"]], "test": [list(t) for t in d["test_recall"]]},
                            "cond": cond, "pass_all": all(cond.values()), "solve": solve, "solve_all": all(solve.values()), "summary": s}
                if tag == "L_Bfirst":
                    diff = [k for k in cond if cond[k] != res["L"]["cond"][k]]
                    res[tag]["cond_differs_from_L"] = diff
                    print(f"  [B첫 판정 민감도] A 외부 F1 {m(Rb, 'ext_f1_real31L_Bfirst'):.4f} → {m(Rn, 'ext_f1_real31L_Bfirst'):.4f} {fmt(d['A_ext_f1'])}"
                          f"  B mAP {fmt(d['B_ext_map'])}  glass recall {100 * m(Rb, 'ext_recall_real31L_Bfirst', 1):.1f}→{100 * m(Rn, 'ext_recall_real31L_Bfirst', 1):.1f}"
                          f" ({100 * d['ext_recall'][1][0]:+.1f} [{100 * d['ext_recall'][1][1]:+.1f}, {100 * d['ext_recall'][1][2]:+.1f}])"
                          f"  맞춤 5창 {fmt(d['ext_m5'])}  조건 {cond} → {'통과' if all(cond.values()) else '미통과'}"
                          + (f"  ★ 주 기준과 다른 조건: {diff}" if diff else "  (주 기준과 판정 같음)"))
                elif tag == "L":
                    print(f"\n== {label} / {cname}  [주 기준 real31L + testL]  (기준 {len(Rb)}시드, 비교 {len(Rn)}시드)"
                          f"\n  A 외부 F1  {m(Rb, 'ext_f1_real31L'):.4f} → {m(Rn, 'ext_f1_real31L'):.4f}  {fmt(d['A_ext_f1'])} {'○' if cond['A'] else '×'}"
                          f"\n  B 외부 mAP {m(Rb, 'ext_map_real31L'):.4f} → {m(Rn, 'ext_map_real31L'):.4f}  {fmt(d['B_ext_map'])} {'○' if cond['B'] else '×'}"
                          f"\n  C 시험셋 F1 {m(Rb, 'test_f1_L'):.4f} → {m(Rn, 'test_f1_L'):.4f}  {fmt(d['C_test_f1'])} {'○' if cond['C'] else '×'}"
                          f"\n  D 외부 발화 창 {d['D_fire_n'][1]:.1f} → {d['D_fire_n'][0]:.1f} {'○' if cond['D'] else '×'}"
                          f"   E 배경+배경 {100 * d['E_bgbg']:.2f}% {'○' if cond['E'] else '×'}"
                          f"\n  F 시험셋 scream recall {100 * m(Rb, 'test_recall_L', 2):.1f} → {100 * m(Rn, 'test_recall_L', 2):.1f} ({100 * d['test_recall'][2][0]:+.1f} pp) {'○' if cond['F'] else '×'}"
                          f"   G 시험셋 glass recall {100 * m(Rb, 'test_recall_L', 1):.1f} → {100 * m(Rn, 'test_recall_L', 1):.1f} ({100 * d['test_recall'][1][0]:+.1f} pp) {'○' if cond['G'] else '×'}"
                          f"\n  맞춤 5창 {m(Rb, 'ext_m5_real31L'):.4f} → {m(Rn, 'ext_m5_real31L'):.4f} ({fmt(d['ext_m5'])})  맞춤 26창 {fmt(d['ext_m26'])}"
                          f"\n  외부 recall (pp): " + "  ".join(
                              f"{E.EVENTS[c]} {100 * m(Rb, 'ext_recall_real31L', c):.1f}→{100 * m(Rn, 'ext_recall_real31L', c):.1f} "
                              f"({100 * d['ext_recall'][c][0]:+.1f} [{100 * d['ext_recall'][c][1]:+.1f}, {100 * d['ext_recall'][c][2]:+.1f}])" for c in range(4))
                          + f"\n  시험셋 recall (pp): " + "  ".join(
                              f"{E.EVENTS[c]} {100 * m(Rb, 'test_recall_L', c):.1f}→{100 * m(Rn, 'test_recall_L', c):.1f} ({100 * d['test_recall'][c][0]:+.1f})" for c in range(4))
                          + f"\n  조건 {cond} → {'통과' if all(cond.values()) else '미통과'}"
                          f"\n  해결 {solve} (외부 F1 목표 ≥ {s['ext_target']:.3f}) → {'해결' if all(solve.values()) else '미해결'}", flush=True)
                else:
                    print(f"  [{tag}] A {fmt(d['A_ext_f1'])} B {fmt(d['B_ext_map'])} C {fmt(d['C_test_f1'])} D {d['D_fire_n'][1]:.1f}→{d['D_fire_n'][0]:.1f}"
                          f" F {100 * d['test_recall'][2][0]:+.1f}pp G {100 * d['test_recall'][1][0]:+.1f}pp → {cond}")
            out["compare"].append(res)

    # ── v4+D − v4+D½ (간격 400): 데이터 양 효과
    Rh, Rf = R.get(("v4+D½", "간격 400")), R.get(("v4+D", "간격 400"))
    if Rh and Rf:
        res = {"name": "v4+D − v4+D½", "config": "간격 400", "n_half": len(Rh), "n_full": len(Rf)}
        for d in ("real31L", "real31L_Bfirst", "real31", "v1eqL"):
            w = V.welch([r[f"ext_f1_{d}"] for r in Rh], [r[f"ext_f1_{d}"] for r in Rf])
            wm = V.welch([r[f"ext_map_{d}"] for r in Rh], [r[f"ext_map_{d}"] for r in Rf])
            res[d] = {"ext_f1": list(w), "ext_map": list(wm), "worth": bool(w[0] >= HALF_GAIN and w[1] > HALF_LO)}
            flag = ""
            if d == "real31L_Bfirst" and res[d]["worth"] != res["real31L"]["worth"]:
                flag = "  ★ 주 기준과 판정 다름"
            print(f"\n== v4+D − v4+D½ / 간격 400 [{d}] (절반 {len(Rh)}시드, 전체 {len(Rf)}시드): 외부 F1 {m(Rh, f'ext_f1_{d}'):.4f} → {m(Rf, f'ext_f1_{d}'):.4f} {fmt(w)}"
                  f"  mAP {fmt(wm)} → {'AudioSet 을 더 받을 가치 있음' if res[d]['worth'] else '이 분해능으로 안 보임'}{flag}")
        wt = V.welch([r["test_f1_L"] for r in Rh], [r["test_f1_L"] for r in Rf])
        res["test_f1_L"] = list(wt)
        print(f"   시험셋 F1 (testL) {m(Rh, 'test_f1_L'):.4f} → {m(Rf, 'test_f1_L'):.4f} {fmt(wt)}")
        out["compare"].append(res)

    # ── v4+D 다섯 구성 — 구성 표·순위·재판정 (curve-design.md 의 사전 등록 항목을 testL / real31L 로 다시)
    V4D = {c: R.get(("v4+D", c)) for c, _cfg in CONFIGS}
    have = [c for c, _cfg in CONFIGS if V4D[c] and c not in EXTRA]          # 다섯 구성 (순위 대상)
    extra = [c for c, _cfg in CONFIGS if V4D[c] and c in EXTRA]            # (a) — 별도 행
    if len(have) >= 2:
        out["v4d_configs"] = {}
        print("\n## v4+D 다섯 구성 (시드 1~3 평균 ± sd; testL / real31L) — 아래 별도 행: (a) NPU 로그 근사")
        print("| 구성 | n | 시험셋 F1 testL | 외부 F1 real31L | 외부 mAP | 시험셋 recall siren/glass/scream/dog | 외부 recall siren/glass/scream/dog | 외부 배경 발화 | 배경+배경 |")
        print("|---|---:|---|---|---|---|---|---:|---:|")
        for c in have + extra:
            rs = V4D[c]
            sd = lambda k: float(np.std([r[k] for r in rs], ddof=1)) if len(rs) > 1 else 0.0
            out["v4d_configs"][c] = {"n": len(rs), "test_f1_L": [m(rs, "test_f1_L"), sd("test_f1_L")],
                                     "ext_f1_real31L": [m(rs, "ext_f1_real31L"), sd("ext_f1_real31L")],
                                     "ext_map_real31L": [m(rs, "ext_map_real31L"), sd("ext_map_real31L")],
                                     "test_recall_L": [m(rs, "test_recall_L", i) for i in range(4)],
                                     "ext_recall_real31L": [m(rs, "ext_recall_real31L", i) for i in range(4)],
                                     "ext_fire_n": m(rs, "ext_fire_n"), "bgbg_fire": m(rs, "bgbg_fire")}
            print(f"| {c} | {len(rs)} | {m(rs, 'test_f1_L'):.4f} ± {sd('test_f1_L'):.4f} | {m(rs, 'ext_f1_real31L'):.4f} ± {sd('ext_f1_real31L'):.4f} | "
                  f"{m(rs, 'ext_map_real31L'):.4f} ± {sd('ext_map_real31L'):.4f} | "
                  + "/".join(f"{100 * m(rs, 'test_recall_L', i):.1f}" for i in range(4)) + " | "
                  + "/".join(f"{100 * m(rs, 'ext_recall_real31L', i):.1f}" for i in range(4))
                  + f" | {m(rs, 'ext_fire_n'):.1f} | {100 * m(rs, 'bgbg_fire'):.2f}% |")
        for key, lab in (("test_f1_L", "시험셋 F1 testL"), ("ext_f1_real31L", "외부 F1 real31L"), ("ext_map_real31L", "외부 mAP")):
            rk = sorted(have, key=lambda c: -m(V4D[c], key))
            out["v4d_configs"][f"rank_{key}"] = rk
            print(f"  순위 ({lab}): " + " > ".join(f"{c} {m(V4D[c], key):.3f}" for c in rk))
        # 재판정: (변형 − ①′) 동등 ±0.02 (시험셋 macro-F1, 같은 시드), 가·다 대 ④·①′ (Welch 95% CI)
        pairs = [("간격 400", "①′"), ("가 u1000", "①′"), ("다 u800", "①′"), ("가 u1000", "④"), ("다 u800", "④"), ("간격 400", "④"),
                 ("(a) NPU 로그 근사", "④")]
        out["v4d_rejudge"] = []
        print("\n| 비교 (앞 − 뒤) | 시험셋 F1 testL (Welch 95% CI) | 동등(±0.02 안에 완전히) | 외부 F1 real31L | 외부 mAP |")
        print("|---|---|---|---|---|")
        for a, b in pairs:
            if not (V4D.get(a) and V4D.get(b)):
                continue
            wt = V.welch([r["test_f1_L"] for r in V4D[b]], [r["test_f1_L"] for r in V4D[a]])
            we = V.welch([r["ext_f1_real31L"] for r in V4D[b]], [r["ext_f1_real31L"] for r in V4D[a]])
            wm = V.welch([r["ext_map_real31L"] for r in V4D[b]], [r["ext_map_real31L"] for r in V4D[a]])
            eq = bool(wt[1] >= -EQUIV and wt[2] <= EQUIV)
            verdict = "동등" if eq else ("못 가른다 (CI 가 0 포함, 폭 초과)" if wt[1] < 0 < wt[2] else ("유의하게 높음" if wt[1] > 0 else "유의하게 낮음"))
            out["v4d_rejudge"].append({"pair": f"{a} − {b}", "test_f1_L": list(wt), "ext_f1_real31L": list(we), "ext_map_real31L": list(wm), "equiv": eq, "verdict": verdict})
            print(f"| {a} − {b} | {fmt(wt)} | {verdict} | {fmt(we)} | {fmt(wm)} |")
        # (a) − ④: v1 에서의 관계와 나란히 (같은 체크포인트, 시드 1~3; 원 사전 등록 값은 5시드 옛 정답 +0.003 [−0.028, +0.034], STATUS chain13)
        A1, W1 = R.get(("v1", "(a) NPU 로그 근사")), R.get(("v1", "④"))
        if A1 and W1 and V4D.get("(a) NPU 로그 근사"):
            print("\n| 판 | (a) − ④ 시험셋 F1 옛 정답 test | (a) − ④ 시험셋 F1 testL | (a) − ④ 외부 F1 real31L | (a) − ④ 외부 mAP |")
            print("|---|---|---|---|---|")
            rows_a = {}
            for ed, A, W in (("v1 (시드 1~3)", A1, W1), ("v4+D (시드 1~3)", V4D["(a) NPU 로그 근사"], V4D["④"])):
                wo = V.welch([r["test_f1_old"] for r in W], [r["test_f1_old"] for r in A])
                wl = V.welch([r["test_f1_L"] for r in W], [r["test_f1_L"] for r in A])
                we = V.welch([r["ext_f1_real31L"] for r in W], [r["ext_f1_real31L"] for r in A])
                wm = V.welch([r["ext_map_real31L"] for r in W], [r["ext_map_real31L"] for r in A])
                rows_a[ed] = {"test_old": list(wo), "test_L": list(wl), "ext_f1": list(we), "ext_map": list(wm)}
                print(f"| {ed} | {fmt(wo)} | {fmt(wl)} | {fmt(we)} | {fmt(wm)} |")
            out["a_vs_wave"] = rows_a
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)
    return out


def panns(S):
    import label_audit as LA
    names = LA.label_names()
    ext = LA.load("ext"); S_p = LA.zeroshot(ext["P"], names)
    assert np.array_equal(ext["clips"].astype(str), S["seg"])
    zt = np.load(LA.TEST_CACHE, allow_pickle=True); S_t = LA.zeroshot(zt["P"], names)
    thr_t = E.thr_at(S_t, zt["y"] == E.BG)
    p_t = E.predict(S_t, thr_t)
    rows = []
    print("\n## PANNs CNN14 zero-shot — 같은 정답")
    for tag, keep in (("test (옛)", np.ones(len(zt["y"]), bool)), ("testL", S["keep_t"])):
        f1, rec, _ = E.metrics(zt["y"][keep], p_t[keep])
        print(f"  시험셋 {tag} @300/h: macro-F1 {f1:.4f}  recall {np.round(100 * rec[:4], 1).tolist()}")
        rows.append(dict(where="test", definition=tag, f1=f1, recall=rec.tolist()))
    isbg = S["isbg"]
    thr_e = E.thr_at(S_p, isbg)
    print("\n| 정의 | 지점 | 외부 F1 | siren | glass | scream | dog_bark | 오경보/h | mAP |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in LABEL_SETS:
        y = S["Y"][d]
        for point, thr in (("ext300", thr_e), ("test300", thr_t)):
            f1, rec, fa, mAP, aps = LA.ext_eval(S_p, y, isbg, thr)
            rows.append(dict(where="ext", definition=d, point=point, f1=f1, recall=rec.tolist(), fa=fa, map=mAP, ap=aps))
            print(f"| {d} | {point} | {f1:.4f} | {100 * rec[0]:.1f} | {100 * rec[1]:.1f} | {100 * rec[2]:.1f} | {100 * rec[3]:.1f} | {fa:.0f} | {mAP:.4f} |")
    path = os.path.join(E.WORK, "v4_panns.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("저장:", path)


def paper_glass(S):
    """논문 표 v1 (9구성, eval_external.RUNS 시드 수) 과 v2.1 (5구성, eval_v3.PAIRS 시드 수) — 유리 관련 수치 변화."""
    print("\n## 논문 표 수치가 새 정답(testL / real31L)에서 얼마나 바뀌는가  (외부 열의 괄호 = B 첫 판정 민감도)")
    print("| 판 | 구성 | n | macro-F1 old (평균 ± sd) | macro-F1 testL | Δ | glass recall old → L (pp) | glass F1 old → L | 외부 F1 real31 → real31L (B첫) | 외부 glass recall real31 → L (B첫) |")
    print("|---|---|---:|---|---|---:|---|---|---|---|")
    rows = []
    specs = [("v1", name, cfg, pre, n) for name, cfg, pre, n in E.RUNS] + \
            [("v2.1", name, cfg, p21, n21) for name, cfg, _p1, _n1, p21, n21, _p3, _n3 in V3.PAIRS]
    for ed, name, cfg, pre, n in specs:
        cks = seeds_of(pre, n)
        rs = [score_ckpt(cfg, ck, S) for ck in cks]
        if not rs:
            continue
        fo, fl = [r["test_f1_old"] for r in rs], [r["test_f1_L"] for r in rs]
        row = dict(edition=ed, name=name, config=cfg, n=len(rs), f1_old=fo, f1_L=fl,
                   glass_recall_old=m(rs, "test_recall_old", 1), glass_recall_L=m(rs, "test_recall_L", 1),
                   glass_f1_old=m(rs, "test_f1_class_old", 1), glass_f1_L=m(rs, "test_f1_class_L", 1),
                   ext_f1_real31=m(rs, "ext_f1_real31"), ext_f1_real31L=m(rs, "ext_f1_real31L"), ext_f1_real31L_Bfirst=m(rs, "ext_f1_real31L_Bfirst"),
                   ext_glass_real31=m(rs, "ext_recall_real31", 1), ext_glass_real31L=m(rs, "ext_recall_real31L", 1),
                   ext_glass_real31L_Bfirst=m(rs, "ext_recall_real31L_Bfirst", 1))
        rows.append(row)
        print(f"| {ed} | {name} | {len(rs)} | {np.mean(fo):.3f} ± {np.std(fo, ddof=1) if len(fo) > 1 else 0:.3f} | "
              f"{np.mean(fl):.3f} ± {np.std(fl, ddof=1) if len(fl) > 1 else 0:.3f} | {np.mean(fl) - np.mean(fo):+.3f} | "
              f"{100 * row['glass_recall_old']:.1f} → {100 * row['glass_recall_L']:.1f} | {row['glass_f1_old']:.3f} → {row['glass_f1_L']:.3f} | "
              f"{row['ext_f1_real31']:.3f} → {row['ext_f1_real31L']:.3f} ({row['ext_f1_real31L_Bfirst']:.3f}) | "
              f"{100 * row['ext_glass_real31']:.1f} → {100 * row['ext_glass_real31L']:.1f} ({100 * row['ext_glass_real31L_Bfirst']:.1f}) |", flush=True)
    path = os.path.join(E.WORK, "v4_paper_glass.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("저장:", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--panns", action="store_true")
    ap.add_argument("--paper-glass", action="store_true")
    a = ap.parse_args()
    if not (a.score or a.panns or a.paper_glass):
        ap.print_help(); sys.exit(0)
    S = setup()
    if a.score:
        score_all(S)
    if a.panns:
        panns(S)
    if a.paper_glass:
        paper_glass(S)
