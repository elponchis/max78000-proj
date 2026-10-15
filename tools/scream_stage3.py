#!/usr/bin/env python3
"""비명 3단계 판정 — v5+D (chain31) 대 v4+D, 같은 정답 testL-S3 / real31L, `scream-plan.md` 9.5 (사전 등록 2026-10-14).

정답: 시험셋 **testL-S3** (비명 134창 = 청취 O 만, X·? 평가 제외, `tools/scream_listened_test.py`), 외부 **real31L** 그대로.
문턱값: 기존대로 시험셋 배경 @300/h (단일 문턱값). 로짓은 `eval_external.logits_for` 캐시.
비교: 구성마다 v5+D − v4+D (시드 1~3, Welch 95% CI, `eval_v2.welch`).

| 조건 | 지표 | 기준 | 대상 |
| A | 외부 비명 recall | 점 ≥ +5 pp 이고 CI 하한 > 0 → "개선"; 점 ≥ 0 이나 CI 가 0 포함 → "못 가른다"; 그 외 "악화" | 간격 400, ①′ |
| B | 외부 F1 | CI 하한 > −0.02 | 다섯 구성 |
| C | 시험셋 macro-F1 (testL-S3) | CI 하한 > −0.02 | 다섯 구성 |
| D | 외부 배경 발화 (배경+배경 혼합 창 발화율) | 점 ≤ v4+D + 0.5 pp | 다섯 구성 |
| E | 시험셋 비명 recall (testL-S3) | 점 ≥ 0 (보고만) | 간격 400, ①′ |
판정: A(둘 다 개선) ∧ B ∧ C ∧ D → "3단계 통과"; B·C·D 통과이고 A 못 가른다 → "해 없음, 효과 미확인"; B·C·D 중 하나라도 미통과 → "미통과".
통과·미통과와 무관하게 **보고하고 멈춘다** — 본 표 교체는 사용자 결정 (9.7).

사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):  python tools/scream_stage3.py   → data/external_eval/scream_stage3.json
  완주하지 않은 시드는 건너뛴다 (eval_v4.seeds_of) — 중간 점검에도 쓸 수 있으나 판정은 15/15 완주 뒤의 것만 유효하다.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]
import eval_external as E                       # noqa: E402
import eval_v2 as V                             # noqa: E402
import eval_v4 as Q                             # noqa: E402
import scream_listened as SL                    # noqa: E402
import scream_listened_test as ST               # noqa: E402

OUT = os.path.join(REPO, "data", "external_eval", "scream_stage3.json")
SCREAM = 2
CLS = ["siren", "glass", "scream", "dog_bark"]
# (표시 이름, 구성 키, v4+D 접두, v5+D 접두)
CONFIGS = [("④", "wave", "safesound-v4d-wave", "safesound-v5d-wave"),
           ("다 u800", "mel_u800", "safesound-v4d-melu800", "safesound-v5d-melu800"),
           ("가 u1000", "mel_u1000", "safesound-v4d-melu1000", "safesound-v5d-melu1000"),
           ("간격 400", "mel_h400", "safesound-v4d-melh400", "safesound-v5d-melh400"),
           ("①′", "mel_inc", "safesound-v4d-melinc", "safesound-v5d-melinc")]
A_TARGETS = ("간격 400", "①′")
A_PP = 0.05                 # A: 외부 비명 recall 점 추정 ≥ +5 pp
HARM = -0.02                # B·C: CI 하한 > −0.02
D_PP = 0.005                # D: 배경+배경 발화율 점 추정 ≤ v4+D + 0.5 pp


def keep_s3(S, zt):
    """testL-S3 채점 마스크 — scream_listened_test.score 와 같은 규칙."""
    tv = ST.test_verdicts(ST.load(), SL.load())
    keep = S["keep_t"].copy(); cnt = collections.Counter()
    for i, (c, s, y) in enumerate(zip(zt["clips"], zt["starts"], zt["y"])):
        if y != SCREAM:
            continue
        k = (str(c), int(s))
        if k not in tv:
            sys.exit(f"[에러] 시험셋 비명 창에 판정이 없다: {k}")
        cnt[tv[k]] += 1
        if tv[k] != "O":
            keep[i] = False
    assert sum(cnt.values()) == ST.N_TEST_SCREAM, cnt
    return keep, cnt


def score_ckpt(config, ck, S, zt, keep):
    bias = config in Q.BIAS_CONFIGS
    _c, yt, lt = E.logits_for(config, ck, Q.V.TEST, "test", bias=bias)
    assert np.array_equal(yt, zt["y"])
    _c2, _y2, le = E.logits_for(config, ck, S["ext_root"], "ext", bias=bias)
    _c3, _yb, lb = E.logits_for(config, ck, S["bg_root"], "diag-bgbg", bias=bias)
    thr = E.thr_at(lt, yt == E.BG)
    pt, pe = E.predict(lt, thr), E.predict(le, thr)
    y_ext = S["Y"]["real31L"]; sel = y_ext >= 0
    f1t, rect, _ = E.metrics(yt[keep], pt[keep])
    f1e, rece, _ = E.metrics(y_ext[sel], pe[sel])
    isbg_e = S["isbg"]
    return {"ckpt": os.path.basename(ck), "thr": float(thr),
            "test_f1": f1t, "test_recall": [float(v) for v in rect[:4]], "test_scream": float(rect[SCREAM]),
            "test_fa_h": float((pt[yt == E.BG] != E.BG).mean() * E.PER_HOUR),
            "ext_f1": f1e, "ext_recall": [float(v) for v in rece[:4]], "ext_scream": float(rece[SCREAM]),
            "ext_fire_n": int((pe[isbg_e] != E.BG).sum()), "ext_fa_h": float((pe[isbg_e] != E.BG).mean() * E.PER_HOUR),
            "bgbg_fire": float((E.predict(lb, thr) != E.BG).mean())}


def ms(rs, k):
    v = [r[k] for r in rs]
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def cell(rs, k, pct=False):
    mu, sd = ms(rs, k)
    return f"{100 * mu:.1f} ± {100 * sd:.1f}" if pct else f"{mu:.3f} ± {sd:.3f}"


def ci(rb, rn, k):
    d, lo, hi = V.welch([r[k] for r in rb], [r[k] for r in rn])
    return float(d), float(lo), float(hi)


def fci(t, pct=False):
    d, lo, hi = t
    return (f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] pp" if pct else f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None, help="표시 이름으로 일부만")
    a = ap.parse_args()
    S = Q.setup()
    zt = np.load(os.path.join(REPO, "data", "ceiling-panns-zs.npz"), allow_pickle=True)
    keep, cnt = keep_s3(S, zt)
    y_ext = S["Y"]["real31L"]
    print(f"## 정답: testL-S3 채점 창 {int(keep.sum())} (비명 {int(((zt['y'] == SCREAM) & keep).sum())}; 225 중 O {cnt['O']} / X {cnt['X']} / ? {cnt['?']}), "
          f"외부 real31L 비명 양성 {int((y_ext == SCREAM).sum())}, 외부 배경 구간 {int(S['isbg'].sum())}창. 문턱값 시험셋 배경 @300/h.")
    out = {"truth": {"testL_S3_n": int(keep.sum()), "test_scream_n": int(((zt["y"] == SCREAM) & keep).sum()), "ext_scream_n": int((y_ext == SCREAM).sum())},
           "editions": {"v4+D": {}, "v5+D": {}}, "delta": {}, "cond": {}, "verdict": None}
    R = {"v4+D": {}, "v5+D": {}}
    for name, config, p4, p5 in CONFIGS:
        if a.configs and name not in a.configs:
            continue
        for ed, pre in (("v4+D", p4), ("v5+D", p5)):
            cks = Q.seeds_of(pre)
            R[ed][name] = [score_ckpt(config, ck, S, zt, keep) for ck in cks]
            out["editions"][ed][name] = {"n": len(cks), "seeds": R[ed][name]}

    # ── 표 1: 두 판 나란히
    print("\n## 표 1 — v4+D / v5+D, 같은 정답 (testL-S3 · real31L), 시드 평균 ± 표준편차")
    print("| 구성 | 판 | n | 시험셋 macro-F1 | 시험셋 비명 recall | 시험셋 recall siren/glass/scream/dog | 외부 F1 | 외부 비명 recall | 외부 recall siren/glass/scream/dog | 외부 배경 발화 (264창) | 배경+배경 발화율 |")
    print("|---|---|---:|---|---|---|---|---|---|---:|---:|")
    for name, *_ in CONFIGS:
        for ed in ("v4+D", "v5+D"):
            rs = R[ed].get(name) or []
            if not rs:
                print(f"| {name} | {ed} | 0 | — | — | — | — | — | — | — | — |"); continue
            tr = " / ".join(f"{100 * np.mean([r['test_recall'][i] for r in rs]):.0f}" for i in range(4))
            er = " / ".join(f"{100 * np.mean([r['ext_recall'][i] for r in rs]):.0f}" for i in range(4))
            print(f"| {name} | {ed} | {len(rs)} | {cell(rs, 'test_f1')} | {cell(rs, 'test_scream', True)} | {tr} | {cell(rs, 'ext_f1')} | "
                  f"{cell(rs, 'ext_scream', True)} | {er} | {ms(rs, 'ext_fire_n')[0]:.1f}창 | {100 * ms(rs, 'bgbg_fire')[0]:.2f}% |")

    # ── 표 2: 차이와 조건
    print("\n## 표 2 — v5+D − v4+D (Welch 95% CI) 와 9.5 조건")
    print("| 구성 | n | A 외부 비명 recall | B 외부 F1 | C 시험셋 macro-F1 | D 배경+배경 (v4+D → v5+D) | E 시험셋 비명 recall | 판정 |")
    print("|---|---:|---|---|---|---|---|---|")
    cond = {}
    for name, *_ in CONFIGS:
        rb, rn = R["v4+D"].get(name) or [], R["v5+D"].get(name) or []
        if len(rb) < 2 or len(rn) < 2:
            print(f"| {name} | {len(rb)}/{len(rn)} | — | — | — | — | — | 시드 부족 |"); continue
        dA, dB, dC, dE = ci(rb, rn, "ext_scream"), ci(rb, rn, "ext_f1"), ci(rb, rn, "test_f1"), ci(rb, rn, "test_scream")
        b4, b5 = ms(rb, "bgbg_fire")[0], ms(rn, "bgbg_fire")[0]
        c = {"A": ("개선" if (dA[0] >= A_PP and dA[1] > 0) else "못 가른다" if dA[0] >= 0 else "악화") if name in A_TARGETS else None,
             "B": bool(dB[1] > HARM), "C": bool(dC[1] > HARM), "D": bool(b5 <= b4 + D_PP),
             "E": (bool(dE[0] >= 0) if name in A_TARGETS else None)}
        cond[name] = c
        out["delta"][name] = {"A_ext_scream": dA, "B_ext_f1": dB, "C_test_f1": dC, "D_bgbg": [b4, b5], "E_test_scream": dE,
                              "ext_fire_n": [ms(rb, "ext_fire_n")[0], ms(rn, "ext_fire_n")[0]]}
        mark = lambda ok: "○" if ok else "×"
        a_txt = f"{fci(dA, True)} → {c['A']}" if c["A"] else f"{fci(dA, True)} (참고)"
        e_txt = f"{fci(dE, True)} {mark(c['E'])}" if c["E"] is not None else f"{fci(dE, True)} (참고)"
        tags = [k for k in ("B", "C", "D") if not c[k]]
        print(f"| {name} | {len(rb)}/{len(rn)} | {a_txt} | {fci(dB)} {mark(c['B'])} | {fci(dC)} {mark(c['C'])} | "
              f"{100 * b4:.2f} → {100 * b5:.2f}% {mark(c['D'])} | {e_txt} | {'B·C·D 통과' if not tags else '미통과 ' + '·'.join(tags)} |")
    out["cond"] = cond
    # ── 종합 판정 (다섯 구성 전부 3시드일 때만 유효)
    full = all(len(R["v4+D"].get(n) or []) == Q.K and len(R["v5+D"].get(n) or []) == Q.K for n, *_ in CONFIGS)
    if cond and all(n in cond for n, *_ in CONFIGS):
        bcd = all(cond[n]["B"] and cond[n]["C"] and cond[n]["D"] for n, *_ in CONFIGS)
        a_all = [cond[n]["A"] for n in A_TARGETS]
        if not bcd:
            verdict = "미통과 (B·C·D 중 미충족)"
        elif all(v == "개선" for v in a_all):
            verdict = "3단계 통과"
        elif all(v in ("개선", "못 가른다") for v in a_all):
            verdict = "해 없음, 효과 미확인"
        else:
            verdict = "미통과 (A 악화)"
        out["verdict"] = verdict + ("" if full else " — ⚠️ 시드 미완 (중간 점검, 판정 무효)")
        print(f"\n## 판정: **{out['verdict']}**  (A 간격 400 {cond['간격 400']['A']} / ①′ {cond['①′']['A']}; B·C·D {'전부 통과' if bcd else '미충족 있음'})")
        print("보고하고 멈춘다 — 본 표 교체·주 정답 변경은 사용자 결정 (scream-plan.md 9.7).")
    else:
        print("\n## 판정 보류 — 비교할 시드가 부족한 구성이 있다.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
