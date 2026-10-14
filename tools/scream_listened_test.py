#!/usr/bin/env python3
"""시험셋 비명 정답 확정 청취(`scream-plan.md` 9.3·9.7) 판정 → testL-S3 고정 + v4+D 재채점 (사용자 지시 2026-10-14).

입력  ~/safesound-external/audit_test_scream/key.csv                    세트·파일 → 출처·창 (저장소 밖)
      ~/safesound-external/audit_test_scream/verdict_T_test_scream.csv   사용자 판정 (file, verdict, memo)
      data_listened_scream.csv                                           2단계 판정 — 시험셋 Yell·Shout 만 66창 (testL-S2)
출력  data_listened_test_scream.csv (레포 루트, **커밋** — 창 ID 와 판정만, 오디오 없음) ← chain31 시작 조건
      data/external_eval/scream_listened_test.json, 콘솔 표

  --build   판정을 key.csv 와 대조해 data_listened_test_scream.csv 를 쓴다. 대조 창 O 비율이 CONTRAST_MIN 미만이면 멈춘다.
  --score   testL-S3 로 v4+D 를 다시 채점한다 (v5 비교 기준). testL / testL-S2 도 나란히.

testL-S3 = 시험셋 비명 225창 전부 청취 판정으로 가른다: `Screaming` 보유 159 (이 세트) + `Yell`·`Shout` 만 66 (2단계 세트).
  O 는 비명 양성으로 남기고 X·? 는 평가에서 뺀다 (배경으로 옮기지 않는다). 다른 클래스·배경은 testL 그대로.
  문턱값은 기존대로 시험셋 배경 @300/h (제외 창은 배경이 아니라 문턱값에 영향 없음). 외부(real31L)는 그대로.
학습 동반 30창 (`Screaming`+`Yell`/`Shout`) 의 X·? 는 v5 제외 목록에 더한다 (`tools/make_exclude_v5.py`).

규칙: 청취 판정은 라벨 규칙보다 우선한다. ? 는 평가 제외. PANNs 점수는 정답에 쓰지 않는다. 판정 뒤에는 바꾸지 않는다.
사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):  python tools/scream_listened_test.py --build --score
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]
import scream_listened as SL                     # noqa: E402  (wilson, read_verdicts, COLUMNS, load)

AUDIT = os.path.expanduser("~/safesound-external/audit_test_scream")
SET = "T_test_scream"
LISTENED = os.path.join(REPO, "data_listened_test_scream.csv")
OUT = os.path.join(REPO, "data", "external_eval", "scream_listened_test.json")
CONTRAST_MIN = SL.CONTRAST_MIN                    # 이전 세트 A 87% / C 93% / S 100%
GROUPS = [("screaming_test", "시험셋 비명 — Screaming 보유 (전수)", 159),
          ("screaming_plus_train", "학습 비명 — Screaming+Yell·Shout 동반 (전수)", 30)]
N_TEST_SCREAM = 225                                # 시험셋 비명 창 전체 = 159 + 66
SCREAM = SL.SCREAM
HEADER = ["# 시험셋 비명 정답 확정 청취 판정 (2026-10-14, scream-plan.md 9.3·10절). 규칙: O 비명 / X 비명 아님 / ? 판단 불가. PANNs 점수는 쓰지 않았다.",
          "# 출처 v4_test = 시험셋 비명 창 중 FSD50K Screaming 보유 159 (Yell·Shout 만 66 은 data_listened_scream.csv), v4_train = 학습 비명 중 Screaming+Yell/Shout 동반 30.",
          "# role contrast = 대조 창(PANNs ≥ 0.50 학습 창, 일관성 점검용 — 정답·제외에 쓰지 않는다). labels = FSD50K 비명 라벨 조합.",
          "# 쓰임: (1) testL-S3 — 시험셋 비명 225창 = 이 파일 159 + data_listened_scream.csv 의 yellshout_test 66; O 양성, X·? 평가 제외 (배경 아님).",
          "#       v5 (chain31) 의 주 정답으로 학습 시작 전에 고정 — 이후 바꾸지 않는다 (tools/scream_listened_test.py --score; run_chain31.sh 시작 조건).",
          "#       (2) 학습 동반 30창의 X·? → v5 제외 목록 (tools/make_exclude_v5.py). 오디오는 저장소에 없다."]


def build(audit_dir=AUDIT, out=LISTENED):
    with open(os.path.join(audit_dir, "key.csv"), encoding="utf-8") as f:
        key = [r for r in csv.DictReader(f) if r["set"] == SET]
    vd = SL.read_verdicts(os.path.join(audit_dir, f"verdict_{SET}.csv"))
    missing = [r["file"] for r in key if r["file"] not in vd]
    extra = sorted(set(vd) - {r["file"] for r in key})
    bad = [(fn, v) for fn, (v, _m) in vd.items() if v not in SL.VERDICTS]
    if missing or extra or bad:
        sys.exit(f"[에러] 빠진 파일 {missing[:5]} ({len(missing)}), 모르는 파일 {extra[:5]}, 잘못된 판정 {bad[:5]}")
    rows = []
    for r in key:
        v, m = vd[r["file"]]
        rows.append({c: r.get(c, "") for c in SL.COLUMNS[:-2]} | {"verdict": v, "memo": m})
    c = collections.Counter(r["verdict"] for r in rows if r["role"] == "contrast")
    n = sum(c.values()); ratio = c["O"] / max(1, n)
    print(f"## 대조 창 {n}: O {c['O']} / X {c['X']} / ? {c['?']} → O 비율 {100 * ratio:.0f}% (이전 세트 A 87% / C 93% / S 100%)")
    if ratio < CONTRAST_MIN:
        sys.exit(f"!!! 대조 창 O 비율 {100 * ratio:.0f}% < {100 * CONTRAST_MIN:.0f}% — 기준이 흔들렸을 수 있다. 채점 전에 멈춘다.")
    with open(out, "w", encoding="utf-8", newline="") as f:
        for h in HEADER:
            f.write(h + "\n")
        w = csv.DictWriter(f, fieldnames=SL.COLUMNS); w.writeheader(); w.writerows(rows)
    print(f"→ {out}  ({len(rows)}행)")
    return rows


def load(path=LISTENED):
    return SL.load(path)


def test_verdicts(rows_t, rows_s):
    """시험셋 비명 225창 전부의 (clip, start) → 판정. 159 는 이 세트, 66 은 2단계 세트."""
    v = {(r["clip"], int(r["start_sample"])): r["verdict"] for r in rows_t if r["definition"] == "screaming_test"}
    v2 = {(r["clip"], int(r["start_sample"])): r["verdict"] for r in rows_s if r["definition"] == "yellshout_test"}
    assert len(v) == 159 and len(v2) == 66 and not (set(v) & set(v2)), (len(v), len(v2))
    return v | v2


def summary(rows_t, rows_s):
    rep = {"groups": {}, "by_labels": {}, "test_scream": {}}
    print("\n## 그룹별 판정 분포 (O 비율의 Wilson 95% 구간)")
    print("| 그룹 | 창 | O | X | ? | O / 전체 [95%] | X / 전체 [95%] | O / (O+X) [95%] |")
    print("|---|---:|---:|---:|---:|---|---|---|")
    for g, name, n_exp in GROUPS:
        v = [r["verdict"] for r in rows_t if r["definition"] == g and r["role"] == "target"]
        assert len(v) == n_exp, (g, len(v), n_exp)
        c = collections.Counter(v); n = len(v); o, x, q = c["O"], c["X"], c["?"]
        lo, hi = SL.wilson(o, n); lo2, hi2 = SL.wilson(o, o + x); xlo, xhi = SL.wilson(x, n)
        rep["groups"][g] = dict(n=n, O=o, X=x, q=q, o_ratio=o / n, o_ci=[lo, hi], x_ratio=x / n, x_ci=[xlo, xhi],
                                o_of_ox=o / (o + x), o_of_ox_ci=[lo2, hi2])
        print(f"| {name} | {n} | {o} | {x} | {q} | {100 * o / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] | "
              f"{100 * x / n:.1f}% [{100 * xlo:.1f}, {100 * xhi:.1f}] | {100 * o / (o + x):.1f}% [{100 * lo2:.1f}, {100 * hi2:.1f}] |")
    print("\n## 라벨 조합별")
    print("| 그룹 | 라벨 | 창 | O | X | ? |")
    print("|---|---|---:|---:|---:|---:|")
    for g, _name, _n in GROUPS:
        labs = sorted({r["labels"] for r in rows_t if r["definition"] == g})
        for lab in labs:
            v = collections.Counter(r["verdict"] for r in rows_t if r["definition"] == g and r["labels"] == lab)
            n = sum(v.values())
            rep["by_labels"][f"{g}/{lab}"] = dict(n=n, O=v["O"], X=v["X"], q=v["?"])
            print(f"| {g} | {lab} | {n} | {v['O']} | {v['X']} | {v['?']} |")
    # ── testL-S3: 시험셋 비명 225창 전체
    tv = test_verdicts(rows_t, rows_s)
    c_all = collections.Counter(tv.values())
    c_ys = collections.Counter(r["verdict"] for r in rows_s if r["definition"] == "yellshout_test")
    c_sc = collections.Counter(r["verdict"] for r in rows_t if r["definition"] == "screaming_test")
    rep["test_scream"] = {"n": len(tv), "all": dict(c_all), "screaming": dict(c_sc), "yellshout": dict(c_ys),
                          "positives_S3": c_all["O"], "excluded_S3": len(tv) - c_all["O"]}
    print(f"\n## testL-S3 — 시험셋 비명 {len(tv)}창 청취 확정: O {c_all['O']} (Screaming 보유 {c_sc['O']} + Yell·Shout 만 {c_ys['O']}) 양성, "
          f"X {c_all['X']} + ? {c_all['?']} = {len(tv) - c_all['O']} 평가 제외 (배경 아님)")
    lo, hi = SL.wilson(c_all["O"], len(tv))
    print(f"시험셋 비명 라벨 225창 중 실제 비명 O 비율 {100 * c_all['O'] / len(tv):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
    return rep


def score(rows_t, rows_s, configs=None):
    """v4+D 를 testL / testL-S2 / testL-S3 로. scream_listened.score 와 같은 경로·같은 문턱값."""
    import eval_external as E
    import eval_v2 as V
    import eval_v4 as Q
    import scream_stage1 as S1
    tv = test_verdicts(rows_t, rows_s)
    S = Q.setup()
    zt = np.load(os.path.join(REPO, "data", "ceiling-panns-zs.npz"), allow_pickle=True)
    keepL = S["keep_t"]
    keepS2, keepS3 = keepL.copy(), keepL.copy()
    cnt = collections.Counter(); n_scr_test = 0
    for i, (c, s, y) in enumerate(zip(zt["clips"], zt["starts"], zt["y"])):
        if y != SCREAM:
            continue
        n_scr_test += 1
        k = (str(c), int(s))
        if k not in tv:
            sys.exit(f"[에러] 시험셋 비명 창에 판정이 없다: {k}")
        cnt[tv[k]] += 1
        if tv[k] != "O":
            keepS3[i] = False
            if k in {(r["clip"], int(r["start_sample"])) for r in rows_s if r["definition"] == "yellshout_test"}:
                keepS2[i] = False
    assert n_scr_test == N_TEST_SCREAM == sum(cnt.values()), (n_scr_test, cnt)
    y_ext = S["Y"]["real31L"]
    n_scr = lambda keep: int(((zt["y"] == SCREAM) & keep).sum())
    print(f"\n## v4+D 재채점 — testL-S3 (v5 비교 기준): 시험셋 비명 {N_TEST_SCREAM}창 → O {cnt['O']} 양성, X {cnt['X']} + ? {cnt['?']} 제외")
    print(f"채점 창 testL {int(keepL.sum())} / testL-S2 {int(keepS2.sum())} / testL-S3 {int(keepS3.sum())}; "
          f"비명 {n_scr(keepL)} / {n_scr(keepS2)} / {n_scr(keepS3)}; 외부 비명 양성 {int((y_ext == SCREAM).sum())}")
    out = {"n": {"testL": int(keepL.sum()), "testL_S2": int(keepS2.sum()), "testL_S3": int(keepS3.sum()),
                 "scream": [n_scr(keepL), n_scr(keepS2), n_scr(keepS3)], "test_scream_verdicts": dict(cnt)}, "configs": {}}
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
            for tag, keep in (("testL", keepL), ("testL_S2", keepS2), ("testL_S3", keepS3)):
                f1, rec, _ = E.metrics(yt[keep], pt[keep])
                one[tag] = {"f1": f1, "scream_recall": float(rec[SCREAM]), "recall": [float(v) for v in rec]}
            sel = y_ext >= 0
            f1, rec, _ = E.metrics(y_ext[sel], pe[sel])
            # 외부 배경 발화: eval_v4 와 같은 정의 — 외부 배경 구간(group == background, 264창) 발화 수, 배경+배경 혼합 창 발화율
            isbg_e = S["isbg"]
            _c3, _yb, lb = E.logits_for(config, ck, S["bg_root"], "diag-bgbg", bias=bias)
            one["ext"] = {"f1": f1, "scream_recall": float(rec[SCREAM]), "recall": [float(v) for v in rec],
                          "ext_fire_n": int((pe[isbg_e] != E.BG).sum()), "ext_bg_n": int(isbg_e.sum()),
                          "bgbg_fire": float((E.predict(lb, thr) != E.BG).mean())}
            res.append(one)
        R[name] = res
        out["configs"][name] = {"n": len(res), "seeds": res}

    def ms(rs, tag, k):
        v = [r[tag][k] for r in rs]
        return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    def cell(rs, tag, k, pct=False):
        m, s = ms(rs, tag, k)
        return f"{100 * m:.1f} ± {100 * s:.1f}" if pct else f"{m:.3f} ± {s:.3f}"

    print("\n| 구성 | n | macro-F1 testL | testL-S2 | **testL-S3** | S3 − L | 비명 recall testL | testL-S2 | **testL-S3** | 외부 F1 | 외부 비명 recall | 외부 배경 발화 (264창) | 배경+배경 발화율 |")
    print("|---|---:|---|---|---|---:|---|---|---|---|---|---:|---:|")
    out["s3_minus_l"] = {}
    for name, rs in R.items():
        dL, dS3 = ms(rs, "testL", "f1")[0], ms(rs, "testL_S3", "f1")[0]
        print(f"| {name} | {len(rs)} | {cell(rs, 'testL', 'f1')} | {cell(rs, 'testL_S2', 'f1')} | {cell(rs, 'testL_S3', 'f1')} | {dS3 - dL:+.3f} | "
              f"{cell(rs, 'testL', 'scream_recall', True)} | {cell(rs, 'testL_S2', 'scream_recall', True)} | {cell(rs, 'testL_S3', 'scream_recall', True)} | "
              f"{cell(rs, 'ext', 'f1')} | {cell(rs, 'ext', 'scream_recall', True)} | {ms(rs, 'ext', 'ext_fire_n')[0]:.1f}창 | {100 * ms(rs, 'ext', 'bgbg_fire')[0]:.2f}% |")
        if len(rs) > 1:
            out["s3_minus_l"][name] = {"f1": list(V.welch([r["testL"]["f1"] for r in rs], [r["testL_S3"]["f1"] for r in rs])),
                                       "scream": list(V.welch([r["testL"]["scream_recall"] for r in rs], [r["testL_S3"]["scream_recall"] for r in rs]))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--audit-dir", default=AUDIT)
    a = ap.parse_args()
    rows_t = build(a.audit_dir) if a.build else load()
    rows_s = SL.load()
    rep = summary(rows_t, rows_s)
    if a.score:
        rep["score"] = score(rows_t, rows_s, a.configs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)
