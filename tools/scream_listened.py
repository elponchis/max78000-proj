#!/usr/bin/env python3
"""학습 비명 청취(2단계, `scream-plan.md` 7절) 판정 → 정리·추정·testL-S2 재채점 (사용자 지시 2026-10-14).

입력  ~/safesound-external/audit_train_scream/key.csv                 세트·파일 → 출처·창 (저장소 밖)
      ~/safesound-external/audit_train_scream/verdict_S_train_scream.csv  사용자 판정 (file, verdict, memo)
출력  data_listened_scream.csv (레포 루트, **커밋** — 창 ID 와 판정만, 오디오 없음)
      data/external_eval/scream_listened.json, 콘솔 표

  --build   판정 파일을 key.csv 와 대조해 data_listened_scream.csv 를 쓴다. 대조 창 O 비율이 CONTRAST_MIN 미만이면 멈춘다.
  --score   testL-S2 재채점: 시험셋 비명 중 `Yell`·`Shout` 만 보유한 66창을 **일괄 제외(testL-S)** 하는 대신
            청취 판정으로 가른다 — O 는 비명 양성으로 남기고 X·? 만 평가에서 뺀다 (배경으로 옮기지 않는다).
            문턱값은 기존대로 시험셋 배경 @300/h. 외부(real31L)는 그대로. 로짓은 scream_stage1 의 캐시를 쓴다.

규칙: 청취 판정은 라벨 규칙보다 우선한다 (data_overrides.csv 와 같은 원칙). ? 는 평가 제외. PANNs 점수는 정답에 쓰지 않는다.
사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):  python tools/scream_listened.py --build --score
"""
import argparse
import collections
import csv
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]

AUDIT = os.path.expanduser("~/safesound-external/audit_train_scream")
SET = "S_train_scream"
LISTENED = os.path.join(REPO, "data_listened_scream.csv")
OUT = os.path.join(REPO, "data", "external_eval", "scream_listened.json")
VERDICTS = ("O", "X", "?")
CONTRAST_MIN = 0.80                       # 이전 세트 A 87% / C 93% — 이보다 크게 낮으면 채점 전에 멈춘다
GROUPS = [("yellshout_train", "학습 비명 — Yell·Shout 만 (전수)", 90), ("audioset_train", "학습 비명 — AudioSet (전수)", 124),
          ("screaming_train", "학습 비명 — Screaming 만 (315 중 무작위)", 100), ("yellshout_test", "시험셋 비명 — Yell·Shout 만 (전수)", 66)]
# 학습 비명 559창의 구성 (scream-plan.md 2절). Screaming 만 315 중 100 청취, Screaming+Yell·Shout 30 은 미청취.
TRAIN_TOTAL = {"yellshout_train": 90, "audioset_train": 124, "screaming_only": 315, "screaming_plus": 30}
N_YELLSHOUT_TEST = 66
SCREAM = 2
COLUMNS = ["set", "file", "role", "source", "cls", "definition", "clip", "start_sample", "labels", "verdict", "memo"]
HEADER = ["# 학습·시험셋 비명 창 청취 판정 (2026-10-14, scream-plan.md 8절). 규칙: O 비명 / X 비명 아님 / ? 판단 불가. PANNs 점수는 쓰지 않았다.",
          "# 출처 v4_train = v4+D 학습 비명 창, v4_test = 시험셋 비명 창 (Yell·Shout 만 보유 66), audioset = 학습 AudioSet 비명 창.",
          "# role contrast = 대조 창(PANNs ≥ 0.50 학습 창, 일관성 점검용 — 정답·제외에 쓰지 않는다). labels = FSD50K 비명 라벨 조합 / AudioSet 메모.",
          "# 쓰임: (1) testL-S2 — 시험셋 Yell·Shout 만 창 중 O 는 비명 양성 유지, X·? 평가 제외 (tools/scream_listened.py --score);",
          "#       (2) 3단계 v5 학습 제외 목록 (X·? 창) — scream-plan.md 9절, 실행은 사용자 승인 뒤. 오디오는 저장소에 없다."]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def read_verdicts(path):
    import listened_labels as LL
    return LL.read_verdicts(path)


def build(audit_dir=AUDIT, out=LISTENED):
    with open(os.path.join(audit_dir, "key.csv"), encoding="utf-8") as f:
        key = [r for r in csv.DictReader(f) if r["set"] == SET]
    vd = read_verdicts(os.path.join(audit_dir, f"verdict_{SET}.csv"))
    missing = [r["file"] for r in key if r["file"] not in vd]
    extra = sorted(set(vd) - {r["file"] for r in key})
    bad = [(fn, v) for fn, (v, _m) in vd.items() if v not in VERDICTS]
    if missing or extra or bad:
        sys.exit(f"[에러] 빠진 파일 {missing[:5]} ({len(missing)}), 모르는 파일 {extra[:5]}, 잘못된 판정 {bad[:5]}")
    rows = []
    for r in key:
        v, m = vd[r["file"]]
        rows.append({c: r.get(c, "") for c in COLUMNS[:-2]} | {"verdict": v, "memo": m})
    c = collections.Counter(r["verdict"] for r in rows if r["role"] == "contrast")
    n = sum(c.values()); ratio = c["O"] / max(1, n)
    print(f"## 대조 창 {n}: O {c['O']} / X {c['X']} / ? {c['?']} → O 비율 {100 * ratio:.0f}% (이전 세트 A 87% / C 93%)")
    if ratio < CONTRAST_MIN:
        sys.exit(f"!!! 대조 창 O 비율 {100 * ratio:.0f}% < {100 * CONTRAST_MIN:.0f}% — 기준이 흔들렸을 수 있다. 채점 전에 멈춘다.")
    with open(out, "w", encoding="utf-8", newline="") as f:
        for h in HEADER:
            f.write(h + "\n")
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    print(f"→ {out}  ({len(rows)}행)")
    return rows


def load(path=LISTENED):
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def summary(rows):
    """그룹별 분포 + 학습 비명 559창 추정. dict 로 돌려준다."""
    rep = {"groups": {}, "by_labels": {}, "train_estimate": {}}
    print("\n## 그룹별 판정 분포 (O 비율의 Wilson 95% 구간)")
    print("| 그룹 | 창 | O | X | ? | O / 전체 [95%] | X / 전체 | O / (O+X) [95%] |")
    print("|---|---:|---:|---:|---:|---|---:|---|")
    for g, name, n_exp in GROUPS:
        v = [r["verdict"] for r in rows if r["definition"] == g and r["role"] == "target"]
        assert len(v) == n_exp, (g, len(v), n_exp)
        c = collections.Counter(v); n = len(v); o, x, q = c["O"], c["X"], c["?"]
        lo, hi = wilson(o, n); lo2, hi2 = wilson(o, o + x); xlo, xhi = wilson(x, n)
        rep["groups"][g] = dict(n=n, O=o, X=x, q=q, o_ratio=o / n, o_ci=[lo, hi], x_ratio=x / n, x_ci=[xlo, xhi],
                                o_of_ox=o / (o + x), o_of_ox_ci=[lo2, hi2])
        print(f"| {name} | {n} | {o} | {x} | {q} | {100 * o / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] | {100 * x / n:.1f}% | "
              f"{100 * o / (o + x):.1f}% [{100 * lo2:.1f}, {100 * hi2:.1f}] |")
    print("\n## Yell·Shout 만 창 — 라벨 조합별")
    print("| 그룹 | 라벨 | 창 | O | X | ? |")
    print("|---|---|---:|---:|---:|---:|")
    for g in ("yellshout_train", "yellshout_test"):
        for lab in ("Yell", "Shout", "Yell+Shout"):
            v = collections.Counter(r["verdict"] for r in rows if r["definition"] == g and r["labels"] == lab)
            n = sum(v.values())
            if n:
                rep["by_labels"][f"{g}/{lab}"] = dict(n=n, O=v["O"], X=v["X"], q=v["?"])
                print(f"| {g} | {lab} | {n} | {v['O']} | {v['X']} | {v['?']} |")
    # ── 학습 비명 559창 추정: 청취한 그룹은 그대로, Screaming 만 나머지 215 + Screaming+Yell·Shout 30 (= 245) 은 Screaming 만 표본의 비율로.
    G = rep["groups"]
    n_unl = TRAIN_TOTAL["screaming_only"] - G["screaming_train"]["n"] + TRAIN_TOTAL["screaming_plus"]
    est = {}
    for k in ("O", "X", "q"):
        exact = G["yellshout_train"][k] + G["audioset_train"][k] + G["screaming_train"][k]
        rate = G["screaming_train"][k] / G["screaming_train"]["n"]
        est[k] = {"exact": exact, "extrapolated": n_unl * rate, "total": exact + n_unl * rate}
    xlo, xhi = wilson(G["screaming_train"]["X"], G["screaming_train"]["n"])
    qlo, qhi = wilson(G["screaming_train"]["q"], G["screaming_train"]["n"])
    total = sum(TRAIN_TOTAL.values())
    x_tot = est["X"]["total"]; q_tot = est["q"]["total"]
    x_range = [est["X"]["exact"] + n_unl * xlo, est["X"]["exact"] + n_unl * xhi]
    q_range = [est["q"]["exact"] + n_unl * qlo, est["q"]["exact"] + n_unl * qhi]
    rep["train_estimate"] = dict(total=total, n_unlistened=n_unl, est=est, x_total=x_tot, x_total_range=x_range,
                                 q_total=q_tot, q_total_range=q_range, x_ratio=x_tot / total, x_ratio_range=[v / total for v in x_range],
                                 xq_ratio=(x_tot + q_tot) / total)
    print(f"\n## 학습 비명 {total}창 추정 (청취 그룹은 실측, 미청취 {n_unl}창은 Screaming 만 표본 비율로 외삽)")
    print("| 판정 | 청취 실측 (314창) | 미청취 외삽 (245창) | 합계 | 비율 |")
    print("|---|---:|---:|---:|---:|")
    for k, lab in (("O", "O 비명"), ("X", "X 비명 아님"), ("q", "? 판단 불가")):
        e = est[k]
        print(f"| {lab} | {e['exact']} | {e['extrapolated']:.1f} | {e['total']:.1f} | {100 * e['total'] / total:.1f}% |")
    print(f"X 합계 구간 (Screaming 만 X 비율 Wilson 95% [{100 * xlo:.1f}, {100 * xhi:.1f}]% 를 외삽): {x_range[0]:.0f}~{x_range[1]:.0f}창 "
          f"({100 * x_range[0] / total:.1f}~{100 * x_range[1] / total:.1f}%); X+? 합계 {x_tot + q_tot:.0f}창 ({100 * (x_tot + q_tot) / total:.1f}%)")
    share = {g: G[g]["X"] / x_tot for g in ("yellshout_train", "audioset_train")}
    share["screaming(외삽 포함)"] = (G["screaming_train"]["X"] + est["X"]["extrapolated"]) / x_tot
    rep["train_estimate"]["x_share"] = share
    print("X 의 출처 비중: " + ", ".join(f"{g} {100 * v:.0f}%" for g, v in share.items()))
    return rep


def score(rows, configs=None):
    """testL / testL-S / testL-S2 를 같은 체크포인트·같은 문턱값으로. scream_stage1 과 같은 경로."""
    import eval_external as E
    import eval_v2 as V
    import eval_v4 as Q
    import scream_stage1 as S1
    gt = S1.fsd_labels()
    drop_all = S1.yellshout_only_test(gt)
    verdict = {(r["clip"], int(r["start_sample"])): r["verdict"] for r in rows if r["definition"] == "yellshout_test"}
    assert set(verdict) == drop_all, (len(verdict), len(drop_all))
    S = Q.setup()
    zt = np.load(os.path.join(REPO, "data", "ceiling-panns-zs.npz"), allow_pickle=True)
    keepL = S["keep_t"]
    keepS, _n = S1.test_keep_S(S, zt, drop_all)
    keepS2 = keepL.copy(); cnt = collections.Counter()
    for i, (c, s, y) in enumerate(zip(zt["clips"], zt["starts"], zt["y"])):
        k = (str(c), int(s))
        if y == SCREAM and k in verdict:
            cnt[verdict[k]] += 1
            if verdict[k] != "O":
                keepS2[i] = False
    assert sum(cnt.values()) == N_YELLSHOUT_TEST, cnt
    y_ext = S["Y"]["real31L"]
    n_scr = lambda keep: int(((zt["y"] == SCREAM) & keep).sum())
    print(f"\n## testL-S2 — 시험셋 Yell·Shout 만 66창을 청취로 가른다: O {cnt['O']} 유지, X {cnt['X']} + ? {cnt['?']} 제외")
    print(f"채점 창 testL {int(keepL.sum())} / testL-S {int(keepS.sum())} / testL-S2 {int(keepS2.sum())}; "
          f"비명 {n_scr(keepL)} / {n_scr(keepS)} / {n_scr(keepS2)}")
    out = {"n": {"testL": int(keepL.sum()), "testL_S": int(keepS.sum()), "testL_S2": int(keepS2.sum()),
                 "scream": [n_scr(keepL), n_scr(keepS), n_scr(keepS2)], "yellshout_test_verdicts": dict(cnt)}, "configs": {}}
    R = {}
    for name, config, pre in S1.CONFIGS:
        if configs and name not in configs:
            continue
        bias = config in Q.BIAS_CONFIGS
        res = []
        for ck in Q.seeds_of(pre):
            _c, yt, lt = E.logits_for(config, ck, Q.V.TEST, "test", bias=bias)
            assert np.array_equal(yt, zt["y"])
            _c2, _y2, le = E.logits_for(config, ck, S["ext_root"], "ext", bias=bias)
            thr = E.thr_at(lt, yt == E.BG)
            pt, pe = E.predict(lt, thr), E.predict(le, thr)
            one = {"ckpt": os.path.basename(ck), "thr": thr}
            for tag, keep in (("testL", keepL), ("testL_S", keepS), ("testL_S2", keepS2)):
                f1, rec, _ = E.metrics(yt[keep], pt[keep])
                one[tag] = {"f1": f1, "scream_recall": float(rec[SCREAM])}
            sel = y_ext >= 0
            f1, rec, _ = E.metrics(y_ext[sel], pe[sel])
            one["ext"] = {"f1": f1, "scream_recall": float(rec[SCREAM])}
            res.append(one)
        R[name] = res
        out["configs"][name] = {"n": len(res), "seeds": res}

    def ms(rs, tag, k):
        v = [r[tag][k] for r in rs]
        return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    def cell(rs, tag, k, pct=False):
        m, s = ms(rs, tag, k)
        return f"{100 * m:.1f} ± {100 * s:.1f}" if pct else f"{m:.3f} ± {s:.3f}"

    print("\n| 구성 | n | macro-F1 testL | testL-S | **testL-S2** | S2 − L | 비명 recall testL | testL-S | **testL-S2** | 외부 비명 recall (참고) |")
    print("|---|---:|---|---|---|---:|---|---|---|---|")
    out["s2_minus_l"] = {}
    for name, rs in R.items():
        dL, dS2 = ms(rs, "testL", "f1")[0], ms(rs, "testL_S2", "f1")[0]
        print(f"| {name} | {len(rs)} | {cell(rs, 'testL', 'f1')} | {cell(rs, 'testL_S', 'f1')} | {cell(rs, 'testL_S2', 'f1')} | {dS2 - dL:+.3f} | "
              f"{cell(rs, 'testL', 'scream_recall', True)} | {cell(rs, 'testL_S', 'scream_recall', True)} | {cell(rs, 'testL_S2', 'scream_recall', True)} | "
              f"{cell(rs, 'ext', 'scream_recall', True)} |")
        if len(rs) > 1:
            out["s2_minus_l"][name] = {"f1": list(V.welch([r["testL"]["f1"] for r in rs], [r["testL_S2"]["f1"] for r in rs])),
                                       "scream": list(V.welch([r["testL"]["scream_recall"] for r in rs], [r["testL_S2"]["scream_recall"] for r in rs]))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--audit-dir", default=AUDIT)
    a = ap.parse_args()
    rows = build(a.audit_dir) if a.build else load()
    rep = summary(rows)
    if a.score:
        rep["score"] = score(rows, a.configs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)
